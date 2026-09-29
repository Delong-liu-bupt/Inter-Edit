"""Run inference for Control-Image Joint Training (CJT) checkpoints."""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import torch
from PIL import Image

from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Inter-Edit CJT checkpoints")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Base Qwen-Image-Edit model name or local path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Checkpoint directory or LoRA weight directory.",
    )
    parser.add_argument(
        "--control_images",
        type=str,
        nargs="+",
        default=None,
        help="Ordered control images: original image followed by mask image.",
    )
    parser.add_argument(
        "--control_dir",
        type=str,
        default=None,
        help="Directory containing control images. Files are loaded in sorted order.",
    )
    parser.add_argument(
        "--control_pattern",
        type=str,
        default=None,
        help="Glob pattern for control images, e.g. 'examples/sample_*'.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Editing instruction.")
    parser.add_argument("--output", type=str, default="edited_image.png", help="Output image path.")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Negative prompt.")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument(
        "--cfg_scale",
        "--true_cfg_scale",
        dest="cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale.",
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device.")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="bf16",
        help="Torch dtype used to load the base model.",
    )
    return parser.parse_args()


def resolve_control_image_paths(args: argparse.Namespace) -> List[str]:
    if args.control_pattern:
        paths = sorted(glob.glob(args.control_pattern))
    elif args.control_dir:
        paths = []
        for extension in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            paths.extend(glob.glob(os.path.join(args.control_dir, extension)))
        paths = sorted(paths)
    else:
        paths = args.control_images or []

    if len(paths) < 2:
        raise ValueError("Expected at least two control images: the source image and the mask image.")

    return paths


def load_pipeline(base_model: str, lora_path: str | None, device: str, dtype: torch.dtype) -> QwenImageEditPlusPipeline:
    print(f"Loading base model: {base_model}")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipeline.to(device)

    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        pipeline.load_lora_weights(lora_path)

    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def run_edit(
    pipeline: QwenImageEditPlusPipeline,
    control_image_paths: List[str],
    prompt: str,
    output_path: str,
    negative_prompt: str,
    num_inference_steps: int,
    cfg_scale: float,
    guidance_scale: float,
    seed: int,
) -> str:
    print(f"Using {len(control_image_paths)} control images")
    control_images = [Image.open(path).convert("RGB") for path in control_image_paths]
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    with torch.inference_mode():
        result = pipeline(
            image=control_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.images[0].save(output_path)
    return os.path.abspath(output_path)


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
        if args.dtype != "fp32":
            print("Switching dtype to fp32 for CPU execution.")
            args.dtype = "fp32"

    control_image_paths = resolve_control_image_paths(args)
    pipeline = load_pipeline(
        base_model=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
    )

    output_path = run_edit(
        pipeline=pipeline,
        control_image_paths=control_image_paths,
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    print(f"Saved edited image to: {output_path}")


if __name__ == "__main__":
    main()
