"""Compute the paper's objective editing metrics for Inter-Edit benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    import alpha_clip
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "`alpha_clip` is required for objective evaluation. Install AlphaCLIP first."
    ) from exc

try:
    import lpips
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("`lpips` is required for objective evaluation.") from exc


DEFAULT_ALPHA_CLIP_MODEL = "ViT-L/14"


@dataclass
class BenchmarkKeys:
    index: str
    source: str
    target: str
    mask: str
    reference: str
    result: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inter-Edit objective metrics")
    parser.add_argument("--benchmark-json", required=True, help="Benchmark JSON path.")
    parser.add_argument("--data-root", required=True, help="Root directory for benchmark image paths.")
    parser.add_argument("--results-dir", required=True, help="Directory containing generated images.")
    parser.add_argument("--method-suffix", required=True, help="Generated filename suffix, e.g. ours_result.")
    parser.add_argument("--alpha-clip-ckpt", required=True, help="AlphaCLIP checkpoint path.")
    parser.add_argument("--output-file", default="evaluation/objective_metrics.json", help="Output JSON file.")
    parser.add_argument(
        "--alpha-clip-model",
        default=DEFAULT_ALPHA_CLIP_MODEL,
        help="AlphaCLIP visual backbone identifier.",
    )
    parser.add_argument("--clip-cache-dir", default=os.environ.get("ALPHACLIP_CACHE_DIR"), help="Optional CLIP cache dir.")
    parser.add_argument("--torch-home", default=os.environ.get("TORCH_HOME"), help="Optional torch cache dir.")
    parser.add_argument("--resolution", type=int, default=None, help="Optional square resize resolution.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples.")
    parser.add_argument("--index-key", default="index", help="Index key in the benchmark JSON.")
    parser.add_argument("--source-key", default="background_image", help="Source image key.")
    parser.add_argument("--target-key", default="target_image", help="Target image key.")
    parser.add_argument("--mask-key", default="mask", help="Mask image key.")
    parser.add_argument("--reference-key", default="reference_image", help="Reference image key.")
    parser.add_argument(
        "--result-key",
        default=None,
        help="Optional key containing a direct path to the generated image.",
    )
    parser.add_argument(
        "--filename-template",
        default="{index}_{method_suffix}.png",
        help="Template used when --result-key is not provided.",
    )
    return parser.parse_args()


def resolve_path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def load_image(path: str, size: tuple[int, int] | None = None, resample: int = Image.Resampling.LANCZOS) -> Image.Image | None:
    try:
        image = Image.open(path).convert("RGB")
        if size is not None:
            image = image.resize(size, resample)
        return image
    except Exception as exc:
        print(f"[warning] failed to load {path}: {exc}")
        return None


def to_binary_mask(mask: Image.Image) -> np.ndarray:
    return (np.array(mask.convert("L")) > 128).astype(np.uint8)


def get_lpips_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


def calculate_boundary_score(image: Image.Image, mask: np.ndarray, kernel_size: int = 11) -> float:
    try:
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif np.max(mask) == 1:
            mask = mask.astype(np.uint8) * 255

        image_gray = np.array(image.convert("L"))
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        inner_ring = mask - eroded_mask
        outer_ring = dilated_mask - mask

        inner_idx = inner_ring > 0
        outer_idx = outer_ring > 0
        if inner_idx.sum() == 0 or outer_idx.sum() == 0:
            return 0.0

        avg_inner = float(grad_magnitude[inner_idx].mean())
        avg_outer = float(grad_magnitude[outer_idx].mean())
        return abs(avg_inner - avg_outer)
    except Exception as exc:
        print(f"[warning] boundary score failed: {exc}")
        return -1.0


def load_models(args: argparse.Namespace, device: torch.device):
    if args.torch_home:
        os.environ["TORCH_HOME"] = args.torch_home

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    alpha_model, alpha_preprocess = alpha_clip.load(
        args.alpha_clip_model,
        alpha_vision_ckpt_pth=args.alpha_clip_ckpt,
        device=str(device),
        download_root=args.clip_cache_dir,
    )
    alpha_model.eval()

    resize_dim = 224 if "336" not in args.alpha_clip_model else 336
    mask_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_dim, resize_dim)),
            transforms.Normalize(0.5, 0.26),
        ]
    )
    return lpips_model, alpha_model, alpha_preprocess, mask_transform


@torch.no_grad()
def score_item(
    item: dict[str, Any],
    keys: BenchmarkKeys,
    args: argparse.Namespace,
    device: torch.device,
    models,
    transforms_pack,
) -> dict[str, float] | None:
    index_value = item[keys.index]
    result_path = (
        resolve_path(args.data_root, item[keys.result])
        if keys.result and item.get(keys.result)
        else os.path.join(
            args.results_dir,
            args.filename_template.format(index=index_value, method_suffix=args.method_suffix),
        )
    )

    paths = {
        "source": resolve_path(args.data_root, item[keys.source]),
        "target": resolve_path(args.data_root, item[keys.target]),
        "mask": resolve_path(args.data_root, item[keys.mask]),
        "reference": resolve_path(args.data_root, item[keys.reference]),
        "result": result_path,
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        print(f"[skip] {index_value}: missing {', '.join(missing)}")
        return None

    source_image = load_image(paths["source"])
    if source_image is None:
        return None

    base_size = (args.resolution, args.resolution) if args.resolution else source_image.size
    if args.resolution:
        source_image = source_image.resize(base_size, Image.Resampling.LANCZOS)

    target_image = load_image(paths["target"], base_size)
    result_image = load_image(paths["result"], base_size)
    reference_image = load_image(paths["reference"], base_size)
    mask_image = load_image(paths["mask"], base_size, Image.Resampling.NEAREST)
    if any(image is None for image in (target_image, result_image, reference_image, mask_image)):
        return None

    lpips_model, alpha_model = models
    alpha_preprocess, mask_transform = transforms_pack
    clip_dtype = torch.float16 if device.type == "cuda" else torch.float32

    result_lpips = get_lpips_tensor(result_image, device)
    target_lpips = get_lpips_tensor(target_image, device)

    result_clip = alpha_preprocess(result_image).unsqueeze(0).to(device=device, dtype=clip_dtype)
    target_clip = alpha_preprocess(target_image).unsqueeze(0).to(device=device, dtype=clip_dtype)
    source_clip = alpha_preprocess(source_image).unsqueeze(0).to(device=device, dtype=clip_dtype)
    reference_clip = alpha_preprocess(reference_image).unsqueeze(0).to(device=device, dtype=clip_dtype)

    mask_np = to_binary_mask(mask_image)
    inverse_mask_np = 1 - mask_np
    full_mask_np = np.ones_like(mask_np, dtype=np.uint8)

    alpha_mask = mask_transform(mask_np * 255).unsqueeze(0).to(device=device, dtype=clip_dtype)
    alpha_inverse_mask = mask_transform(inverse_mask_np * 255).unsqueeze(0).to(device=device, dtype=clip_dtype)
    alpha_full_mask = mask_transform(full_mask_np * 255).unsqueeze(0).to(device=device, dtype=clip_dtype)

    global_lpips = lpips_model(result_lpips, target_lpips).item()

    result_mask_feat = alpha_model.visual(result_clip, alpha_mask)
    target_mask_feat = alpha_model.visual(target_clip, alpha_mask)
    source_inverse_feat = alpha_model.visual(source_clip, alpha_inverse_mask)
    result_inverse_feat = alpha_model.visual(result_clip, alpha_inverse_mask)
    reference_full_feat = alpha_model.visual(reference_clip, alpha_full_mask)

    return {
        "S_global_LPIPS": float(global_lpips),
        "S_in_AlphaCLIP": float(torch.nn.functional.cosine_similarity(result_mask_feat, target_mask_feat, dim=1).item()),
        "S_out_AlphaCLIP": float(torch.nn.functional.cosine_similarity(result_inverse_feat, source_inverse_feat, dim=1).item()),
        "BSS_Laplacian": float(calculate_boundary_score(result_image, mask_np)),
        "S_ref_AlphaCLIP": float(torch.nn.functional.cosine_similarity(reference_full_feat, result_mask_feat, dim=1).item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.benchmark_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    if args.limit is not None:
        data = data[: args.limit]

    keys = BenchmarkKeys(
        index=args.index_key,
        source=args.source_key,
        target=args.target_key,
        mask=args.mask_key,
        reference=args.reference_key,
        result=args.result_key,
    )

    lpips_model, alpha_model, alpha_preprocess, mask_transform = load_models(args, device)

    detailed_results = []
    skipped = 0
    for item in tqdm(data, desc=f"Objective metrics ({args.method_suffix})"):
        try:
            metrics = score_item(
                item=item,
                keys=keys,
                args=args,
                device=device,
                models=(lpips_model, alpha_model),
                transforms_pack=(alpha_preprocess, mask_transform),
            )
        except KeyError as exc:
            print(f"[skip] malformed item: missing {exc}")
            skipped += 1
            continue

        if metrics is None:
            skipped += 1
            continue

        metrics[args.index_key] = item[args.index_key]
        metrics["method"] = args.method_suffix
        detailed_results.append(metrics)

    if not detailed_results:
        raise SystemExit("No samples were successfully evaluated.")

    average_scores = {
        "method": args.method_suffix,
        "count": len(detailed_results),
        "resolution": args.resolution if args.resolution is not None else "original",
        "avg_S_global_LPIPS (Lower is Better)": float(np.mean([row["S_global_LPIPS"] for row in detailed_results])),
        "avg_S_in_AlphaCLIP (Higher is Better)": float(np.mean([row["S_in_AlphaCLIP"] for row in detailed_results])),
        "avg_S_out_AlphaCLIP (Higher is Better)": float(np.mean([row["S_out_AlphaCLIP"] for row in detailed_results])),
        "avg_BSS_Laplacian (Lower is Better)": float(np.mean([row["BSS_Laplacian"] for row in detailed_results])),
        "avg_S_ref_AlphaCLIP (Higher is Better)": float(np.mean([row["S_ref_AlphaCLIP"] for row in detailed_results])),
    }

    output = {
        "metadata": {
            "benchmark_json": args.benchmark_json,
            "data_root": args.data_root,
            "results_dir": args.results_dir,
            "method_suffix": args.method_suffix,
            "alpha_clip_model": args.alpha_clip_model,
            "alpha_clip_ckpt": args.alpha_clip_ckpt,
            "skipped": skipped,
        },
        "average_scores": average_scores,
        "detailed_results": detailed_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(json.dumps(average_scores, indent=2))
    print(f"Saved objective metrics to: {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()
