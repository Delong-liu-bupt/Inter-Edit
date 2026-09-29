"""Sample a masked subset from an Inter-Edit benchmark JSON."""

from __future__ import annotations

import argparse
import json
import os
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a reproducible benchmark subset")
    parser.add_argument("--input-json", required=True, help="Full benchmark JSON path.")
    parser.add_argument("--data-root", required=True, help="Root directory for relative image paths.")
    parser.add_argument("--output-json", required=True, help="Output subset JSON path.")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of items to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--index-key", default="index", help="Index key in the benchmark JSON.")
    parser.add_argument("--mask-key", default="mask", help="Mask key in the benchmark JSON.")
    return parser.parse_args()


def resolve_path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def main() -> None:
    args = parse_args()
    with open(args.input_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    masked_items = []
    for item in data:
        if args.mask_key not in item:
            continue
        mask_path = resolve_path(args.data_root, item[args.mask_key])
        if os.path.exists(mask_path):
            masked_items.append(item)

    if not masked_items:
        raise SystemExit("No masked items were found.")

    sample_size = min(args.sample_size, len(masked_items))
    random.seed(args.seed)
    sampled = random.sample(masked_items, sample_size)
    sampled.sort(key=lambda row: row.get(args.index_key, 0))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(sampled, file, indent=2, ensure_ascii=False)

    index_file = args.output_json.replace(".json", "_indices.json")
    with open(index_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": args.seed,
                "sample_size": sample_size,
                "indices": [item[args.index_key] for item in sampled],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Loaded {len(data)} benchmark items")
    print(f"Found {len(masked_items)} items with valid masks")
    print(f"Saved {sample_size} sampled items to: {os.path.abspath(args.output_json)}")
    print(f"Saved sampled indices to: {os.path.abspath(index_file)}")


if __name__ == "__main__":
    main()
