"""
Control-Image Joint Training (CJT) for Qwen-Image-Edit.

This training script supports the public Inter-Edit JSON schema and keeps the
same preprocessing path used at inference time by
``pipeline_qwenimage_edit_plus.py``.
"""

import argparse
import copy
import logging
import os
import shutil
import glob
import math

import torch
from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb
import gc

# Import custom pipeline
from dataset_utils import (
    get_config_value,
    get_edited_image_path,
    get_instruction,
    get_mask_image_path,
    get_original_image_path,
    get_target_image_name,
    load_json_samples,
)
from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

logger = get_logger(__name__, log_level="INFO")

# Constants from pipeline
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def convert_lora_state_dict_to_diffusers_format(state_dict):
    """
    Convert LoRA state dict to diffusers-compatible format.

    Converts:
        .lora.down.weight -> .lora_A.weight
        .lora.up.weight -> .lora_B.weight

    This ensures all LoRA weights use consistent naming format.
    """
    converted_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Convert lora.down -> lora_A
        if '.lora.down.' in key:
            new_key = key.replace('.lora.down.', '.lora_A.')
        # Convert lora.up -> lora_B
        elif '.lora.up.' in key:
            new_key = key.replace('.lora.up.', '.lora_B.')
        converted_dict[new_key] = value
    return converted_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline-aligned CJT training for Qwen-Image-Edit LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config file",
    )
    args = parser.parse_args()
    return args.config


def calculate_dimensions(target_area, ratio):
    """Calculate dimensions exactly as in pipeline (line 158-165)"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


def precompute_text_embeddings(args, pipeline, accelerator):
    """
    Precompute text embeddings using pipeline's encode_prompt method.
    This follows the exact same process as pipeline.__call__ (line 700-708)
    """
    logger.info("Precomputing text embeddings using pipeline.encode_prompt...")
    cached_text_embeddings = {}

    use_json_dataset = hasattr(args.data_config, 'json_file') and args.data_config.json_file is not None

    with torch.no_grad():
        if use_json_dataset:
            logger.info(f"Loading JSON dataset: {args.data_config.json_file}")
            data = load_json_samples(
                args.data_config.json_file,
                only_better_data=get_config_value(args.data_config, "only_better_data", False),
                max_samples=get_config_value(args.data_config, "max_samples", None),
            )
            data_root = get_config_value(args.data_config, "data_root", None)

            logger.info(f"Total samples: {len(data)}")

            # Precompute embeddings
            for sample in tqdm(data, desc="Encoding prompts"):
                # Load control images
                control_image_paths = [
                    get_original_image_path(sample, data_root=data_root),
                    get_mask_image_path(sample, data_root=data_root),
                ]
                control_images = [Image.open(p).convert('RGB') for p in control_image_paths]

                # Prepare condition images exactly as in pipeline (line 676-683)
                condition_images = []
                for img in control_images:
                    image_width, image_height = img.size
                    condition_width, condition_height = calculate_dimensions(
                        CONDITION_IMAGE_SIZE, image_width / image_height
                    )
                    # Use pipeline's image_processor.resize (line 683)
                    condition_img = pipeline.image_processor.resize(img, condition_height, condition_width)
                    condition_images.append(condition_img)

                instruction = get_instruction(sample)

                # Use pipeline's encode_prompt method (line 700-708)
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    image=condition_images,
                    prompt=[instruction],
                    device=pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

                cache_key = os.path.splitext(get_target_image_name(sample))[0] + '.txt'
                cached_text_embeddings[cache_key] = {
                    'prompt_embeds': prompt_embeds[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                }

                # Empty embedding for CFG
                prompt_embeds_empty, prompt_embeds_mask_empty = pipeline.encode_prompt(
                    image=condition_images,
                    prompt=[' '],
                    device=pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                cached_text_embeddings[cache_key + '_empty_embedding'] = {
                    'prompt_embeds': prompt_embeds_empty[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu')
                }

        else:
            # Directory-based dataset
            logger.info("Using directory-based dataset")
            img_files = [f for f in os.listdir(args.data_config.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in tqdm(img_files, desc="Encoding prompts"):
                base_name = os.path.splitext(img_name)[0]
                txt_path = os.path.join(args.data_config.img_dir, base_name + '.txt')

                if not os.path.exists(txt_path):
                    continue

                # Find all control images for this sample
                control_pattern = os.path.join(args.data_config.control_dir, f"{base_name}_*")
                control_paths = glob.glob(control_pattern + ".jpg") + glob.glob(control_pattern + ".png")

                if not control_paths:
                    continue

                control_paths.sort()

                # Load control images
                control_images_pil = [Image.open(p).convert('RGB') for p in control_paths]

                # Prepare condition images exactly as in pipeline
                condition_images = []
                for img in control_images_pil:
                    image_width, image_height = img.size
                    condition_width, condition_height = calculate_dimensions(
                        CONDITION_IMAGE_SIZE, image_width / image_height
                    )
                    condition_img = pipeline.image_processor.resize(img, condition_height, condition_width)
                    condition_images.append(condition_img)

                prompt = open(txt_path, encoding='utf-8').read().strip()

                # Use pipeline's encode_prompt
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    image=condition_images,
                    prompt=[prompt],
                    device=pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

                cached_text_embeddings[base_name + '.txt'] = {
                    'prompt_embeds': prompt_embeds[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                }

                # Empty embedding
                prompt_embeds_empty, prompt_embeds_mask_empty = pipeline.encode_prompt(
                    image=condition_images,
                    prompt=[' '],
                    device=pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                cached_text_embeddings[base_name + '.txt' + '_empty_embedding'] = {
                    'prompt_embeds': prompt_embeds_empty[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu')
                }

    logger.info(f"Precomputed {len(cached_text_embeddings)} text embeddings")
    return cached_text_embeddings


def precompute_image_embeddings(args, pipeline, accelerator, weight_dtype):
    """
    Precompute image embeddings using pipeline's _encode_vae_image method.
    This follows pipeline.prepare_latents (line 454-480)
    """
    logger.info("Precomputing image embeddings using pipeline VAE...")
    cached_image_embeddings = {}
    cached_control_embeddings = {}

    use_json_dataset = hasattr(args.data_config, 'json_file') and args.data_config.json_file is not None

    # Move VAE to device
    pipeline.vae.to(accelerator.device, dtype=weight_dtype)

    with torch.no_grad():
        if use_json_dataset:
            data = load_json_samples(
                args.data_config.json_file,
                only_better_data=get_config_value(args.data_config, "only_better_data", False),
                max_samples=get_config_value(args.data_config, "max_samples", None),
            )
            data_root = get_config_value(args.data_config, "data_root", None)

            logger.info("Encoding target images...")
            for sample in tqdm(data, desc="Target images"):
                img = Image.open(get_edited_image_path(sample, data_root=data_root)).convert('RGB')
                image_width, image_height = img.size
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                # Use pipeline's image_processor.preprocess (line 684)
                vae_image = pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)
                vae_image = vae_image.to(dtype=weight_dtype, device=accelerator.device)

                # Use pipeline's _encode_vae_image method (line 411-432)
                latent = pipeline.vae.encode(vae_image).latent_dist.sample()

                # Ensure temporal dimension is 1
                if latent.shape[2] > 1:
                    latent = latent[:, :, 0:1, :, :]

                cache_key = get_target_image_name(sample)
                cached_image_embeddings[cache_key] = latent[0].to('cpu')

            logger.info("Encoding control images...")
            for sample in tqdm(data, desc="Control images"):
                control_paths = [
                    get_original_image_path(sample, data_root=data_root),
                    get_mask_image_path(sample, data_root=data_root),
                ]

                for idx, ctrl_path in enumerate(control_paths):
                    img = Image.open(ctrl_path).convert('RGB')
                    image_width, image_height = img.size
                    vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                    vae_image = pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)
                    vae_image = vae_image.to(dtype=weight_dtype, device=accelerator.device)

                    latent = pipeline.vae.encode(vae_image).latent_dist.sample()

                    if latent.shape[2] > 1:
                        latent = latent[:, :, 0:1, :, :]

                    cache_key = f"{get_target_image_name(sample)}_{idx}"
                    cached_control_embeddings[cache_key] = latent[0].to('cpu')

        else:
            # Directory-based dataset
            logger.info("Encoding target images...")
            img_files = [f for f in os.listdir(args.data_config.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in tqdm(img_files, desc="Target images"):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                image_width, image_height = img.size
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                vae_image = pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)
                vae_image = vae_image.to(dtype=weight_dtype, device=accelerator.device)

                latent = pipeline.vae.encode(vae_image).latent_dist.sample()

                if latent.shape[2] > 1:
                    latent = latent[:, :, 0:1, :, :]

                cached_image_embeddings[img_name] = latent[0].to('cpu')

            logger.info("Encoding control images...")
            control_files = glob.glob(os.path.join(args.data_config.control_dir, "*_*.[jp][pn]g"))

            for ctrl_path in tqdm(control_files, desc="Control images"):
                img = Image.open(ctrl_path).convert('RGB')
                image_width, image_height = img.size
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                vae_image = pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)
                vae_image = vae_image.to(dtype=weight_dtype, device=accelerator.device)

                latent = pipeline.vae.encode(vae_image).latent_dist.sample()

                if latent.shape[2] > 1:
                    latent = latent[:, :, 0:1, :, :]

                cache_key = os.path.basename(ctrl_path)
                cached_control_embeddings[cache_key] = latent[0].to('cpu')

    logger.info(f"Precomputed {len(cached_image_embeddings)} target embeddings and {len(cached_control_embeddings)} control embeddings")

    # Release VAE memory
    logger.info("Moving VAE to CPU and freeing memory...")
    pipeline.vae.to('cpu')
    torch.cuda.empty_cache()

    return cached_image_embeddings, cached_control_embeddings


def create_dataloader(args, cached_text_embeddings, cached_image_embeddings, cached_control_embeddings):
    """
    Create a simple dataloader that returns properly formatted data.
    This replaces the complex dataset loaders.
    """
    from torch.utils.data import Dataset, DataLoader

    use_json_dataset = hasattr(args.data_config, 'json_file') and args.data_config.json_file is not None

    # Check if precompute is enabled
    use_precompute = (cached_text_embeddings is not None and
                     cached_image_embeddings is not None and
                     cached_control_embeddings is not None)

    if not use_precompute:
        logger.warning("=" * 80)
        logger.warning("Running in ONLINE ENCODING mode (no precompute)")
        logger.warning("=" * 80)
        logger.warning("This will:")
        logger.warning("  - Encode images and text on-the-fly during training")
        logger.warning("  - Use more GPU memory (~10GB extra)")
        logger.warning("  - Train SLOWER (~3x slower)")
        logger.warning("")
        logger.warning("Recommended: Enable precompute for faster training")
        logger.warning("  precompute_text_embeddings: true")
        logger.warning("  precompute_image_embeddings: true")
        logger.warning("=" * 80)

    class SimpleMultiControlDataset(Dataset):
        def __init__(self, args, cached_text, cached_img, cached_ctrl):
            self.args = args
            self.cached_text = cached_text
            self.cached_img = cached_img
            self.cached_ctrl = cached_ctrl
            self.samples = []
            self.use_precompute = use_precompute

            if use_json_dataset:
                # JSON-based dataset
                logger.info(f"Loading JSON file: {args.data_config.json_file}")
                data = load_json_samples(
                    args.data_config.json_file,
                    only_better_data=get_config_value(args.data_config, "only_better_data", False),
                    max_samples=get_config_value(args.data_config, "max_samples", None),
                )
                data_root = get_config_value(args.data_config, "data_root", None)

                logger.info(f"Total samples in JSON: {len(data)}")

                # Limit samples if using precomputed cache
                if use_precompute and self.cached_img is not None:
                    if hasattr(self.cached_img, '__len__'):
                        cache_size = len(self.cached_img)
                        logger.info(f"Cache contains {cache_size} samples")

                        if len(data) > cache_size:
                            logger.warning("=" * 80)
                            logger.warning(f"WARNING: Dataset size mismatch!")
                            logger.warning(f"  JSON has {len(data)} samples")
                            logger.warning(f"  Cache has {cache_size} samples (based on image_embeddings)")
                            logger.warning(f"  Using only first {cache_size} samples to match cache")
                            logger.warning("=" * 80)
                            data = data[:cache_size]

                        logger.info(f"Will use {len(data)} samples for training")

                logger.info(f"Processing {len(data)} samples...")

                for sample in tqdm(data, desc="Processing samples"):
                    if use_precompute:
                        target_img_name = get_target_image_name(sample)
                        text_key = os.path.splitext(target_img_name)[0] + '.txt'
                        control_keys = [f"{target_img_name}_0", f"{target_img_name}_1"]

                        # Verify keys exist in cache
                        if target_img_name not in self.cached_img:
                            continue
                        if text_key not in self.cached_text:
                            continue
                        if control_keys[0] not in self.cached_ctrl or control_keys[1] not in self.cached_ctrl:
                            continue

                        self.samples.append({
                            'text_key': text_key,
                            'target_key': target_img_name,
                            'control_keys': control_keys
                        })
                    else:
                        # Online mode: save raw paths
                        self.samples.append({
                            'original_image_path': get_original_image_path(sample, data_root=data_root),
                            'edited_image_path': get_edited_image_path(sample, data_root=data_root),
                            'mask_image_path': get_mask_image_path(sample, data_root=data_root),
                            'instruction': get_instruction(sample),
                        })
            else:
                # Directory-based dataset
                logger.info(f"Scanning directory: {args.data_config.img_dir}")
                img_files = [f for f in os.listdir(args.data_config.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

                logger.info(f"Found {len(img_files)} image files, processing...")
                for img_name in tqdm(img_files, desc="Processing samples"):
                    base_name = os.path.splitext(img_name)[0]
                    text_key = base_name + '.txt'
                    txt_path = os.path.join(args.data_config.img_dir, text_key)

                    # Find control images
                    control_pattern = os.path.join(args.data_config.control_dir, f"{base_name}_*")
                    control_paths = glob.glob(control_pattern + ".jpg") + glob.glob(control_pattern + ".png")
                    control_paths.sort()

                    if use_precompute:
                        control_keys = [os.path.basename(p) for p in control_paths]
                        self.samples.append({
                            'text_key': text_key,
                            'target_key': img_name,
                            'control_keys': control_keys
                        })
                    else:
                        self.samples.append({
                            'target_image_path': os.path.join(args.data_config.img_dir, img_name),
                            'control_image_paths': control_paths,
                            'text_path': txt_path
                        })

            logger.info(f"Created dataset with {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            if self.use_precompute:
                target_latent = self.cached_img[sample['target_key']]
                control_latents = [self.cached_ctrl[k] for k in sample['control_keys']]
                prompt_embeds = self.cached_text[sample['text_key']]['prompt_embeds']
                prompt_embeds_mask = self.cached_text[sample['text_key']]['prompt_embeds_mask']

                return target_latent, control_latents, prompt_embeds, prompt_embeds_mask
            else:
                return sample

    dataset = SimpleMultiControlDataset(args, cached_text_embeddings, cached_image_embeddings, cached_control_embeddings)

    def collate_fn(batch):
        """Collate function to handle variable-length control images"""
        if use_precompute:
            target_latents = []
            all_control_latents = []
            prompt_embeds_list = []
            prompt_masks_list = []

            for target, controls, prompt_emb, prompt_mask in batch:
                target_latents.append(target)
                all_control_latents.append(controls)
                prompt_embeds_list.append(prompt_emb)
                prompt_masks_list.append(prompt_mask)

            target_latents = torch.stack(target_latents)

            # Pad and stack prompt embeddings
            max_seq_len = max([p.shape[0] for p in prompt_embeds_list])
            padded_prompts = []
            padded_masks = []

            for prompt_emb, prompt_mask in zip(prompt_embeds_list, prompt_masks_list):
                pad_len = max_seq_len - prompt_emb.shape[0]
                if pad_len > 0:
                    prompt_emb = torch.cat([prompt_emb, torch.zeros(pad_len, prompt_emb.shape[1])])
                    prompt_mask = torch.cat([prompt_mask, torch.zeros(pad_len, dtype=prompt_mask.dtype)])
                padded_prompts.append(prompt_emb)
                padded_masks.append(prompt_mask)

            prompt_embeds = torch.stack(padded_prompts)
            prompt_masks = torch.stack(padded_masks)

            return target_latents, all_control_latents, prompt_embeds, prompt_masks
        else:
            return batch

    # Use num_workers=0 for sharded cache
    if use_precompute and hasattr(cached_image_embeddings, '_key_to_shard'):
        num_workers = 0
        logger.info("Using num_workers=0 for sharded cache (required for multiprocess compatibility)")
    else:
        num_workers = args.data_config.get('num_workers', 4)

    dataloader = DataLoader(
        dataset,
        batch_size=args.data_config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


def main():
    args = OmegaConf.load(parse_args())

    # Set default values
    args.precompute_text_embeddings = getattr(args, 'precompute_text_embeddings', True)
    args.precompute_image_embeddings = getattr(args, 'precompute_image_embeddings', True)
    args.quantize = getattr(args, 'quantize', False)
    args.adam8bit = getattr(args, 'adam8bit', False)
    args.max_control_images = getattr(args, 'max_control_images', 2)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f"Weight dtype: {weight_dtype}")
    logger.info(f"Loading pipeline from: {args.pretrained_model_name_or_path}")

    # Load pipeline WITHOUT transformer and vae to save memory
    logger.info("Loading text encoding pipeline (without transformer/vae)...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype
    )
    pipeline.to(accelerator.device)

    # Load VAE separately for precomputation
    logger.info("Loading VAE...")
    pipeline.vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )

    logger.info("Pipeline components loaded successfully")

    # Save VAE configuration values
    logger.info("Saving VAE configuration values...")
    vae_scale_factor = 2 ** len(pipeline.vae.temperal_downsample)
    latent_channels = pipeline.vae.config.z_dim
    latents_mean = pipeline.vae.config.latents_mean
    latents_std = pipeline.vae.config.latents_std
    logger.info(f"  VAE scale factor: {vae_scale_factor}")
    logger.info(f"  Latent channels: {latent_channels}")

    # Check if we should load cached embeddings from disk
    cached_embeddings_dir = getattr(args, 'cached_embeddings_dir', None)

    if cached_embeddings_dir and os.path.exists(cached_embeddings_dir):
        # Check if using sharded format
        shards_dir = os.path.join(cached_embeddings_dir, "shards")
        shard_info_path = os.path.join(cached_embeddings_dir, "shard_info.pt")

        if os.path.exists(shards_dir) and os.path.exists(shard_info_path):
            # Use sharded cache files
            logger.info("=" * 80)
            logger.info("Using sharded cached embeddings...")
            logger.info("=" * 80)

            shard_info = torch.load(shard_info_path, map_location='cpu')
            logger.info(f"Found sharded embeddings:")
            for filename, num_shards in shard_info['num_shards'].items():
                logger.info(f"  - {filename}: {num_shards} shards")
            logger.info("=" * 80)

            # Create lazy loading wrapper for sharded files
            class ShardedEmbeddingDict:
                def __init__(self, base_filename, shards_directory, num_shards, shard_size):
                    self.base_filename = base_filename
                    self.shards_dir = shards_directory
                    self.num_shards = num_shards
                    self.shard_size = shard_size
                    self._loaded_shards = {}
                    self._key_to_shard = {}
                    self._index_built = False

                def _build_index(self):
                    """Build an index of which shard contains which key"""
                    if self._index_built:
                        return

                    logger.info(f"Building index for {self.base_filename}...")
                    logger.info("  (This will take a few minutes, but only needs to be done once)")
                    base_name = os.path.splitext(self.base_filename)[0]

                    for shard_idx in tqdm(range(self.num_shards), desc=f"Indexing {base_name}"):
                        shard_file = os.path.join(self.shards_dir, f"{base_name}_shard_{shard_idx:04d}.pt")
                        if os.path.exists(shard_file):
                            shard_data = torch.load(shard_file, map_location='cpu')
                            for key in shard_data.keys():
                                self._key_to_shard[key] = shard_idx
                            if shard_idx == 0:
                                self._loaded_shards[shard_idx] = shard_data
                            del shard_data

                    self._index_built = True
                    logger.info(f"Indexed {len(self._key_to_shard)} keys across {self.num_shards} shards")

                def _load_shard(self, shard_idx):
                    """Load a specific shard into memory"""
                    if shard_idx not in self._loaded_shards:
                        base_name = os.path.splitext(self.base_filename)[0]
                        shard_file = os.path.join(self.shards_dir, f"{base_name}_shard_{shard_idx:04d}.pt")
                        self._loaded_shards[shard_idx] = torch.load(shard_file, map_location='cpu')

                        # Memory management: keep only last 3 shards in memory
                        if len(self._loaded_shards) > 3:
                            oldest_shard = min(self._loaded_shards.keys())
                            if oldest_shard != shard_idx:
                                del self._loaded_shards[oldest_shard]

                    return self._loaded_shards[shard_idx]

                def __getitem__(self, key):
                    self._build_index()
                    if key not in self._key_to_shard:
                        raise KeyError(f"Key not found: {key}")

                    shard_idx = self._key_to_shard[key]
                    shard_data = self._load_shard(shard_idx)
                    return shard_data[key]

                def __contains__(self, key):
                    self._build_index()
                    return key in self._key_to_shard

                def keys(self):
                    self._build_index()
                    return self._key_to_shard.keys()

                def __len__(self):
                    self._build_index()
                    return len(self._key_to_shard)

            # Create sharded embedding dicts
            text_num_shards = shard_info['num_shards'].get('text_embeddings.pt', 0)
            img_num_shards = shard_info['num_shards'].get('image_embeddings.pt', 0)
            ctrl_num_shards = shard_info['num_shards'].get('control_embeddings.pt', 0)
            shard_size = shard_info['shard_size']

            logger.info("")
            logger.info("Creating sharded embedding dictionaries...")
            logger.info("Will build full index on first access (takes ~5-10 minutes)")
            logger.info("")

            cached_text_embeddings = ShardedEmbeddingDict('text_embeddings.pt', shards_dir, text_num_shards, shard_size)
            cached_image_embeddings = ShardedEmbeddingDict('image_embeddings.pt', shards_dir, img_num_shards, shard_size)
            cached_control_embeddings = ShardedEmbeddingDict('control_embeddings.pt', shards_dir, ctrl_num_shards, shard_size)

            # Build indexes now
            logger.info("Building indexes now (will take a few minutes)...")
            logger.info("")
            len(cached_text_embeddings)
            len(cached_image_embeddings)
            len(cached_control_embeddings)
            logger.info("")
            logger.info("All indexes built successfully!")
            logger.info("")

            args.precompute_text_embeddings = True
            args.precompute_image_embeddings = True

        else:
            # Try to load single files (legacy format)
            text_emb_path = os.path.join(cached_embeddings_dir, "text_embeddings.pt")
            img_emb_path = os.path.join(cached_embeddings_dir, "image_embeddings.pt")
            ctrl_emb_path = os.path.join(cached_embeddings_dir, "control_embeddings.pt")

            if os.path.exists(text_emb_path) and os.path.exists(img_emb_path) and os.path.exists(ctrl_emb_path):
                logger.warning("=" * 80)
                logger.warning("WARNING: Using large single-file cache format")
                logger.warning("=" * 80)
                logger.warning("  This may cause OOM errors for large datasets!")
                logger.warning("  Recommendation: Split files using: python split_cache_files.py")
                logger.warning("=" * 80)

                logger.info(f"Loading text embeddings from {text_emb_path}...")
                cached_text_embeddings = torch.load(text_emb_path, map_location='cpu')

                logger.info(f"Loading image embeddings from {img_emb_path}...")
                cached_image_embeddings = torch.load(img_emb_path, map_location='cpu')

                logger.info(f"Loading control embeddings from {ctrl_emb_path}...")
                cached_control_embeddings = torch.load(ctrl_emb_path, map_location='cpu')

                logger.info(f"Loaded {len(cached_text_embeddings)} text embeddings")
                logger.info(f"Loaded {len(cached_image_embeddings)} image embeddings")
                logger.info(f"Loaded {len(cached_control_embeddings)} control embeddings")

                args.precompute_text_embeddings = True
                args.precompute_image_embeddings = True
            else:
                logger.warning(f"Cached embeddings directory exists but files are missing!")
                logger.warning(f"  Falling back to online encoding...")
                cached_text_embeddings = None
                cached_image_embeddings = None
                cached_control_embeddings = None
    else:
        # Precompute embeddings (original logic)
        if args.precompute_text_embeddings:
            cached_text_embeddings = precompute_text_embeddings(args, pipeline, accelerator)
        else:
            cached_text_embeddings = None

        if args.precompute_image_embeddings:
            cached_image_embeddings, cached_control_embeddings = precompute_image_embeddings(
                args, pipeline, accelerator, weight_dtype
            )
        else:
            cached_image_embeddings = None
            cached_control_embeddings = None

    # Move pipeline components to CPU to save memory
    logger.info("Freeing memory after precomputation...")

    if args.precompute_text_embeddings:
        logger.info("  Deleting text encoder (precomputed)...")
        pipeline.text_encoder.to('cpu')
        del pipeline.text_encoder
        pipeline.tokenizer = None
        pipeline.processor = None
    else:
        logger.info("  Moving text encoder to CPU (will use for runtime encoding)...")
        pipeline.text_encoder.to('cpu')

    if args.precompute_image_embeddings:
        logger.info("  Deleting VAE (precomputed)...")
        pipeline.vae.to('cpu')
        del pipeline.vae
    else:
        logger.info("  Moving VAE to CPU (will use for runtime encoding)...")
        pipeline.vae.to('cpu')

    torch.cuda.empty_cache()
    gc.collect()

    logger.info("  Memory cleanup completed")

    if args.precompute_text_embeddings and args.precompute_image_embeddings:
        logger.info("  Deleting pipeline object (all embeddings precomputed)...")
        del pipeline
        pipeline = None
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info("  Keeping pipeline for runtime encoding...")

    # Load transformer separately for training
    logger.info("Loading transformer for training...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )

    # Apply quantization if enabled
    if args.quantize:
        logger.info("Quantizing transformer...")
        device = accelerator.device
        for block in tqdm(list(transformer.transformer_blocks), desc="Quantizing"):
            block.to(device, dtype=weight_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')

        transformer.to(device, dtype=weight_dtype)
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

    # Setup LoRA
    logger.info(f"Adding LoRA adapter with rank={args.rank}")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "img_mod.1", "txt_mlp.net.0.proj", "txt_mlp.net.2", "txt_mod.1"],
    )
    if args.quantize:
        transformer.to(accelerator.device)
    else:
        transformer.to(accelerator.device, dtype=weight_dtype)

    transformer.add_adapter(lora_config)

    # Freeze base model
    transformer.requires_grad_(False)
    for n, param in transformer.named_parameters():
        if 'lora' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1_000_000
    logger.info(f"Trainable parameters: {trainable_params:.2f}M")

    transformer.train()
    transformer.enable_gradient_checkpointing()

    # Setup optimizer
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    if args.adam8bit:
        logger.info("Using 8-bit Adam optimizer")
        optimizer = bnb.optim.Adam8bit(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Create dataloader
    train_dataloader = create_dataloader(args, cached_text_embeddings, cached_image_embeddings, cached_control_embeddings)

    # Setup scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Setup noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"config": dict(args)})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running Training *****")
    logger.info(f"  Model: {args.pretrained_model_name_or_path}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    # Training loop
    for epoch in range(1):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Check if we need to do online encoding
                use_precompute_mode = not isinstance(batch, list) or not isinstance(batch[0], dict)

                if not use_precompute_mode:
                    # Online encoding mode
                    if step % 10 == 0:
                        logger.info(f"Online encoding batch (step {step})...")

                    with torch.no_grad():
                        pipeline.vae.to(accelerator.device, dtype=weight_dtype)
                        pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)

                        target_latents_list = []
                        all_control_latents_list = []
                        prompt_embeds_list = []
                        prompt_embeds_mask_list = []

                        for sample in batch:
                            if 'edited_image_path' in sample:
                                target_img_path = sample['edited_image_path']
                                control_img_paths = [sample['original_image_path'], sample['mask_image_path']]
                                instruction = sample['instruction']
                            else:
                                target_img_path = sample['target_image_path']
                                control_img_paths = sample['control_image_paths']
                                with open(sample['text_path'], 'r', encoding='utf-8') as f:
                                    instruction = f.read().strip()

                            target_img = Image.open(target_img_path).convert('RGB')
                            control_imgs = [Image.open(p).convert('RGB') for p in control_img_paths]

                            condition_images = []
                            for img in control_imgs:
                                image_width, image_height = img.size
                                condition_width, condition_height = calculate_dimensions(
                                    CONDITION_IMAGE_SIZE, image_width / image_height
                                )
                                condition_img = pipeline.image_processor.resize(img, condition_height, condition_width)
                                condition_images.append(condition_img)

                            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                                image=condition_images,
                                prompt=[instruction],
                                device=accelerator.device,
                                num_images_per_prompt=1,
                                max_sequence_length=1024,
                            )
                            prompt_embeds_list.append(prompt_embeds[0])
                            prompt_embeds_mask_list.append(prompt_embeds_mask[0])

                            image_width, image_height = target_img.size
                            vae_width, vae_height = calculate_dimensions(
                                VAE_IMAGE_SIZE, image_width / image_height
                            )
                            vae_image = pipeline.image_processor.preprocess(target_img, vae_height, vae_width).unsqueeze(2)
                            vae_image = vae_image.to(dtype=weight_dtype, device=accelerator.device)
                            target_latent = pipeline.vae.encode(vae_image).latent_dist.sample()

                            if target_latent.shape[2] > 1:
                                target_latent = target_latent[:, :, 0:1, :, :]

                            target_latents_list.append(target_latent[0])

                            control_latents = []
                            for ctrl_img in control_imgs:
                                image_width, image_height = ctrl_img.size
                                vae_width, vae_height = calculate_dimensions(
                                    VAE_IMAGE_SIZE, image_width / image_height
                                )
                                vae_image = pipeline.image_processor.preprocess(ctrl_img, vae_height, vae_width).unsqueeze(2)
                                vae_image = vae_image.to(dtype=weight_dtype, device=accelerator.device)
                                ctrl_latent = pipeline.vae.encode(vae_image).latent_dist.sample()

                                if ctrl_latent.shape[2] > 1:
                                    ctrl_latent = ctrl_latent[:, :, 0:1, :, :]

                                control_latents.append(ctrl_latent[0])

                            all_control_latents_list.append(control_latents)

                        pipeline.vae.to('cpu')
                        pipeline.text_encoder.to('cpu')
                        torch.cuda.empty_cache()

                        target_latents = torch.stack(target_latents_list)

                        max_seq_len = max([p.shape[0] for p in prompt_embeds_list])
                        padded_prompts = []
                        padded_masks = []

                        for prompt_emb, prompt_mask in zip(prompt_embeds_list, prompt_embeds_mask_list):
                            pad_len = max_seq_len - prompt_emb.shape[0]
                            if pad_len > 0:
                                prompt_emb = torch.cat([prompt_emb, torch.zeros(pad_len, prompt_emb.shape[1], device=accelerator.device)])
                                prompt_mask = torch.cat([prompt_mask, torch.zeros(pad_len, dtype=prompt_mask.dtype, device=accelerator.device)])
                            padded_prompts.append(prompt_emb)
                            padded_masks.append(prompt_mask)

                        prompt_embeds = torch.stack(padded_prompts)
                        prompt_embeds_mask = torch.stack(padded_masks)
                else:
                    target_latents, all_control_latents_list, prompt_embeds, prompt_embeds_mask = batch

                # Move to device
                target_latents = target_latents.to(dtype=weight_dtype, device=accelerator.device)
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)
                prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32, device=accelerator.device)

                with torch.no_grad():
                    latents_mean_tensor = (
                        torch.tensor(latents_mean)
                        .view(1, latent_channels, 1, 1, 1)
                        .to(target_latents.device, target_latents.dtype)
                    )
                    latents_std_tensor = (
                        torch.tensor(latents_std)
                        .view(1, latent_channels, 1, 1, 1)
                        .to(target_latents.device, target_latents.dtype)
                    )

                    pixel_latents = (target_latents - latents_mean_tensor) / latents_std_tensor

                    bsz = pixel_latents.shape[0]

                    all_control_latents_normalized = []
                    for batch_idx in range(bsz):
                        batch_controls = []
                        for ctrl_latent in all_control_latents_list[batch_idx]:
                            ctrl_latent = ctrl_latent.to(dtype=weight_dtype, device=accelerator.device)
                            ctrl_latent = (ctrl_latent - latents_mean_tensor) / latents_std_tensor
                            batch_controls.append(ctrl_latent)
                        all_control_latents_normalized.append(batch_controls)

                    noise = torch.randn_like(pixel_latents)

                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                    sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                    noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                    pixel_latents_squeezed = pixel_latents.squeeze(2)
                    noisy_input_squeezed = noisy_model_input.squeeze(2)

                    packed_noisy_input = QwenImageEditPlusPipeline._pack_latents(
                        noisy_input_squeezed,
                        bsz,
                        pixel_latents_squeezed.shape[1],
                        pixel_latents_squeezed.shape[2],
                        pixel_latents_squeezed.shape[3],
                    )

                    all_packed_controls = []
                    img_shapes_list = []

                    for batch_idx in range(bsz):
                        sample_img_shapes = [(1, pixel_latents_squeezed.shape[2] // 2, pixel_latents_squeezed.shape[3] // 2)]

                        for ctrl_latent in all_control_latents_normalized[batch_idx]:
                            ctrl_squeezed = ctrl_latent.squeeze(2)

                            packed_ctrl = QwenImageEditPlusPipeline._pack_latents(
                                ctrl_squeezed,
                                1,
                                ctrl_squeezed.shape[1],
                                ctrl_squeezed.shape[2],
                                ctrl_squeezed.shape[3],
                            )
                            all_packed_controls.append(packed_ctrl)

                            sample_img_shapes.append((1, ctrl_squeezed.shape[2] // 2, ctrl_squeezed.shape[3] // 2))

                        img_shapes_list.append(sample_img_shapes)

                    if all_packed_controls:
                        image_latents = torch.cat(all_packed_controls, dim=0)
                        if len(all_packed_controls) % bsz == 0:
                            num_controls_per_sample = len(all_packed_controls) // bsz
                            image_latents = image_latents.view(bsz, num_controls_per_sample, -1, image_latents.shape[-1])
                            image_latents = image_latents.reshape(bsz, -1, image_latents.shape[-1])

                        latent_model_input = torch.cat([packed_noisy_input, image_latents], dim=1)
                    else:
                        latent_model_input = packed_noisy_input

                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                timestep = timesteps / 1000
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes_list,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                model_pred = model_pred[:, : packed_noisy_input.size(1)]

                model_pred = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred,
                    pixel_latents_squeezed.shape[2] * vae_scale_factor,
                    pixel_latents_squeezed.shape[3] * vae_scale_factor,
                    vae_scale_factor,
                )

                if model_pred.ndim == 4:
                    model_pred = model_pred.unsqueeze(1)
                elif model_pred.ndim == 5:
                    if model_pred.shape[2] == 1:
                        model_pred = model_pred.permute(0, 2, 1, 3, 4)
                else:
                    raise ValueError(f"Unexpected model_pred dimensions: {model_pred.shape}")

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"Removing {len(removing_checkpoints)} old checkpoints")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        unwrapped_transformer = unwrap_model(transformer)
                        transformer_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_transformer)
                        )

                        # Convert to diffusers-compatible format (lora.down/up -> lora_A/B)
                        transformer_lora_state_dict = convert_lora_state_dict_to_diffusers_format(
                            transformer_lora_state_dict
                        )

                        QwenImageEditPlusPipeline.save_lora_weights(
                            save_path,
                            transformer_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved checkpoint to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
