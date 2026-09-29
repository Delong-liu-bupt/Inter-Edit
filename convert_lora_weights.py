"""Convert legacy LoRA checkpoints to the diffusers naming convention."""

from __future__ import annotations

import argparse
import os

from safetensors.torch import load_file, save_file


def convert_lora_weights(input_path: str, output_path: str | None = None) -> str:
    """Convert `.lora.down/.lora.up` keys into `.lora_A/.lora_B` keys."""
    if output_path is None:
        output_path = input_path.replace(".safetensors", "_converted.safetensors")

    print(f"Loading: {input_path}")
    state_dict = load_file(input_path)

    converted_dict = {}
    converted_count = 0
    kept_count = 0

    for key, value in state_dict.items():
        new_key = key
        if ".lora.down." in key:
            new_key = key.replace(".lora.down.", ".lora_A.")
            converted_count += 1
        elif ".lora.up." in key:
            new_key = key.replace(".lora.up.", ".lora_B.")
            converted_count += 1
        else:
            kept_count += 1
        converted_dict[new_key] = value

    print(f"Converted {converted_count} keys")
    print(f"Kept {kept_count} keys unchanged")
    print(f"Saving to: {output_path}")
    save_file(converted_dict, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LoRA weights to diffusers format")
    parser.add_argument(
        "input",
        type=str,
        help="Input safetensors file or checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. If omitted, '<input>_converted.safetensors' is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original safetensors file.",
    )
    args = parser.parse_args()

    input_path = args.input
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "pytorch_lora_weights.safetensors")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = input_path if args.overwrite else args.output
    final_path = convert_lora_weights(input_path, output_path)
    print(f"Done. Saved converted weights to: {final_path}")


if __name__ == "__main__":
    main()
