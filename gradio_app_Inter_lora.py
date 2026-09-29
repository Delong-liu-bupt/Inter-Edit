"""Gradio demo for Inter-Edit CJT checkpoints."""

from __future__ import annotations

import argparse
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline


pipeline = None


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for Inter-Edit CJT checkpoints")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint directory.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen-Image-Edit-2511",
        help="Base Qwen-Image-Edit model name or local path.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Execution device.")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="bf16",
        help="Torch dtype used to load the base model.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port used by the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument(
        "--root_path",
        type=str,
        default=None,
        help="Optional reverse-proxy root path, e.g. '/demo'.",
    )
    return parser.parse_args()


def convert_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    converted_dict = {}
    converted_count = 0
    for key, value in state_dict.items():
        new_key = key
        if ".lora.down." in key:
            new_key = key.replace(".lora.down.", ".lora_A.")
            converted_count += 1
        elif ".lora.up." in key:
            new_key = key.replace(".lora.up.", ".lora_B.")
            converted_count += 1
        converted_dict[new_key] = value
    return converted_dict, converted_count


def check_and_convert_lora_weights(lora_path: str) -> None:
    weights_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

    state_dict = load_file(weights_path)
    if not any(".lora.down." in key or ".lora.up." in key for key in state_dict):
        return

    converted_dict, converted_count = convert_lora_state_dict(state_dict)
    backup_path = weights_path.replace(".safetensors", "_backup.safetensors")
    if not os.path.exists(backup_path):
        os.rename(weights_path, backup_path)
    save_file(converted_dict, weights_path)
    print(f"Converted {converted_count} legacy LoRA keys to diffusers format.")


def load_pipeline(base_model: str, lora_path: str, device: str, dtype: torch.dtype) -> None:
    global pipeline

    print("=" * 80)
    print("Loading Inter-Edit pipeline")
    print("=" * 80)
    print(f"Base model: {base_model}")
    print(f"LoRA path: {lora_path}")

    check_and_convert_lora_weights(lora_path)
    pipeline = QwenImageEditPlusPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipeline.load_lora_weights(lora_path)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)


def process_mask(mask_payload) -> Image.Image | None:
    if mask_payload is None or "mask" not in mask_payload:
        return None

    mask_array = mask_payload["mask"]
    if mask_array is None:
        return None

    mask = Image.fromarray(mask_array).convert("L")
    mask_np = np.array(mask)
    binary_mask = np.where(mask_np > 128, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_mask).convert("RGB")


def edit_image(image_with_mask, prompt, num_inference_steps, cfg_scale, guidance_scale, seed):
    global pipeline

    if pipeline is None:
        return None, "Pipeline is not loaded."
    if image_with_mask is None:
        return None, "Please upload an image and paint a mask."
    if not prompt or not prompt.strip():
        return None, "Please provide an editing instruction."

    try:
        original_image = Image.fromarray(image_with_mask["image"]).convert("RGB")
        mask_image = process_mask(image_with_mask)
        if mask_image is None:
            return None, "Please paint a mask before running inference."

        mask_np = np.array(mask_image)
        if mask_np.max() == 0:
            return None, "The mask is empty. Paint the region you want to edit."

        generator = torch.Generator(device=pipeline.device).manual_seed(int(seed))
        with torch.inference_mode():
            result = pipeline(
                image=[original_image, mask_image],
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=int(num_inference_steps),
                true_cfg_scale=float(cfg_scale),
                guidance_scale=float(guidance_scale),
                generator=generator,
                num_images_per_prompt=1,
            )

        return result.images[0], "Inference completed successfully."
    except Exception as exc:  # pragma: no cover - UI error path
        return None, f"Inference failed: {exc}"


def create_demo(lora_path: str, base_model: str, device: str, dtype: torch.dtype) -> gr.Blocks:
    load_pipeline(base_model=base_model, lora_path=lora_path, device=device, dtype=dtype)

    with gr.Blocks(title="Inter-Edit CJT Demo") as demo:
        gr.Markdown(
            """
# Inter-Edit CJT Demo

1. Upload a source image.
2. Paint the editable region in white.
3. Enter a natural-language instruction.
4. Run the model to generate the edited result.
            """
        )

        with gr.Row():
            image_input = gr.Image(
                label="Source image and editable region",
                tool="sketch",
                type="numpy",
                brush_radius=20,
                brush_color="#FFFFFF",
            )

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Instruction",
                    placeholder="Example: Replace the mug with a transparent glass vase.",
                    lines=2,
                )
                with gr.Accordion("Advanced settings", open=False):
                    num_steps = gr.Slider(20, 100, value=40, step=1, label="Inference steps")
                    cfg_scale = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="True CFG scale")
                    guidance_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="Guidance scale")
                    seed = gr.Slider(0, 999999, value=42, step=1, label="Seed")
                generate_button = gr.Button("Generate", variant="primary", size="lg")
                status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            output_image = gr.Image(label="Edited image", type="pil")

        gr.Examples(
            examples=[
                ["Turn the wall into white marble."],
                ["Remove the person in the background."],
                ["Replace the table with clear glass."],
                ["Change the sky to sunset lighting."],
                ["Add a window on the left side."],
            ],
            inputs=prompt_input,
            label="Prompt examples",
        )

        generate_button.click(
            fn=edit_image,
            inputs=[image_input, prompt_input, num_steps, cfg_scale, guidance_scale, seed],
            outputs=[output_image, status_box],
        )

    return demo


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
        if args.dtype != "fp32":
            args.dtype = "fp32"

    demo = create_demo(
        lora_path=args.lora_path,
        base_model=args.base_model,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
    )

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": args.share,
        "debug": True,
    }
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
