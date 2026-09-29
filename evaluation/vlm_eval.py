"""Run the paper's VLM-based subjective evaluation for Inter-Edit."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import requests
from tqdm import tqdm


SYSTEM_PROMPT = """
You are a strict expert judge for reference-based image editing.
You will receive four inputs:
1. Reference object image
2. Background/source image
3. Editing instruction
4. Edited result image

Score the result from 1 to 10 on four criteria:
- Edit Success
- Alignment
- Naturalness
- Reference Consistency

Use the full 1-10 range. A mediocre result should be around 5-6, not 8.

Return this exact structure:
Rationale
Edit Success: <short explanation>
Alignment: <short explanation>
Naturalness: <short explanation>
Reference Consistency: <short explanation>

Final Scores
Edit Success: <1-10>
Alignment: <1-10>
Naturalness: <1-10>
Reference Consistency: <1-10>
""".strip()


METHOD_SUFFIX_MAP = {
    "brushnet": "brushnet_result",
    "flux2": "flux2_result",
    "flux2_inst": "flux2_inst_result",
    "fluxedit": "flux_result",
    "fluxfill": "fluxfill_result",
    "icedit": "icedit_result",
    "ip2p": "ip2p_result",
    "kontext": "kontext_result",
    "ominicontrol": "ominicontrol_result",
    "ours": "ours_result",
    "powerpaint": "powerpaint_result",
    "qwenedit": "qwenedit_result",
    "qwenedit_inst": "qwenedit_inst_result",
    "sd": "sd_result",
    "sdxl": "sdxl_result",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inter-Edit VLM evaluation")
    parser.add_argument("--benchmark-json", required=True, help="Benchmark JSON path.")
    parser.add_argument("--method-name", required=True, help="Method name, e.g. ours or fluxedit.")
    parser.add_argument("--results-base-dir", default=".", help="Base directory containing results_<method> folders.")
    parser.add_argument("--results-dir", default=None, help="Optional direct path to generated results.")
    parser.add_argument("--data-root", default=".", help="Root directory for benchmark image paths.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI-compatible API key.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        help="OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument("--model", required=True, help="Vision-language model name.")
    parser.add_argument("--output-json", default="evaluation/vlm_eval.json", help="Output JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items.")
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample N masked items.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--sample-index-file",
        default=None,
        help="Optional JSON file used to save or reload sampled indices for fair comparisons.",
    )
    parser.add_argument("--method-suffix", default=None, help="Override the generated image filename suffix.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent request workers.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed samples for up to five extra rounds.")
    parser.add_argument("--index-key", default="index", help="Index key in the benchmark JSON.")
    parser.add_argument("--reference-key", default="reference_image", help="Reference image key.")
    parser.add_argument("--source-key", default="background_image", help="Source image key.")
    parser.add_argument("--prompt-key", default="concise_edit_prompt", help="Instruction key.")
    parser.add_argument("--mask-key", default="mask", help="Mask key, used for sampling.")
    parser.add_argument("--result-key", default=None, help="Optional direct path key for generated images.")
    parser.add_argument(
        "--filename-template",
        default="{index}_{method_suffix}.png",
        help="Template used when --result-key is not provided.",
    )
    return parser.parse_args()


def resolve_path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def encode_image_as_data_uri(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"
    with open(path, "rb") as file:
        payload = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def build_messages(prompt_text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": prompt_text}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": encode_image_as_data_uri(path)}})
    return [{"role": "user", "content": content}]


def call_vlm_api(full_prompt: str, image_paths: list[str], api_key: str, model: str, base_url: str, max_retries: int) -> dict[str, Any] | None:
    payload = {
        "model": model,
        "messages": build_messages(full_prompt, image_paths),
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(base_url, json=payload, headers=headers, timeout=240)
            response.raise_for_status()
            data = response.json()
            if not data.get("choices"):
                raise ValueError("Response does not contain choices.")
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            if not content or len(content.strip()) < 20:
                raise ValueError("Response content is empty or too short.")
            return data
        except Exception as exc:
            tqdm.write(f"[retry {attempt}/{max_retries}] request failed: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def parse_scores(response_text: str) -> dict[str, Any]:
    patterns = {
        "edit_success": r"Edit Success:\s*(\d+)",
        "alignment": r"Alignment:\s*(\d+)",
        "naturalness": r"Naturalness:\s*(\d+)",
        "reference_consistency": r"Reference Consistency:\s*(\d+)",
    }
    scores = {key: None for key in patterns}
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))

    if not all(isinstance(value, int) for value in scores.values()):
        scores["error"] = "Failed to parse one or more scores."
    else:
        scores["error"] = None
    return scores


def calculate_statistics(results_list: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = ["edit_success", "alignment", "naturalness", "reference_consistency"]
    valid_results = []
    for result in results_list:
        scores = result.get("scores", {})
        if scores.get("error"):
            continue
        if all(isinstance(scores.get(key), int) for key in criteria):
            valid_results.append(scores)

    if not valid_results:
        return {"error": "No valid parsed evaluations were found."}

    by_criterion = {}
    flat_scores = []
    for criterion in criteria:
        values = [scores[criterion] for scores in valid_results]
        flat_scores.extend(values)
        distribution = {str(score): 0 for score in range(1, 11)}
        for value in values:
            distribution[str(value)] += 1
        by_criterion[criterion] = {
            "count": len(values),
            "average": float(np.mean(values)),
            "distribution_1_to_10": distribution,
        }

    return {
        "overall_average": float(np.mean(flat_scores)),
        "valid_evaluation_count": len(valid_results),
        "by_criterion": by_criterion,
    }


def resolve_result_path(item: dict[str, Any], args: argparse.Namespace, results_dir: str, method_suffix: str) -> str:
    if args.result_key and item.get(args.result_key):
        return resolve_path(args.data_root, item[args.result_key])
    return os.path.join(
        results_dir,
        args.filename_template.format(index=item[args.index_key], method_suffix=method_suffix),
    )


def evaluate_item(item: dict[str, Any], args: argparse.Namespace, results_dir: str, method_suffix: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    item_id = item.get(args.index_key)
    try:
        reference_image_path = resolve_path(args.data_root, item[args.reference_key])
        source_image_path = resolve_path(args.data_root, item[args.source_key])
        prompt = item[args.prompt_key]
        edited_image_path = resolve_result_path(item, args, results_dir, method_suffix)
    except KeyError as exc:
        tqdm.write(f"[skip] {item_id}: missing key {exc}")
        return None, item

    required_paths = [reference_image_path, source_image_path, edited_image_path]
    if any(not os.path.exists(path) for path in required_paths):
        tqdm.write(f"[skip] {item_id}: missing image file")
        return None, item

    user_prompt = "\n".join(
        [
            "Reference Object Image: [IMAGE_1]",
            "Background Image: [IMAGE_2]",
            f"Editing Instruction: {prompt}",
            "Edited Result Image: [IMAGE_3]",
        ]
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\nTask:\n{user_prompt}\n\nYour evaluation:"

    raw_response_text = None
    scores = None
    for attempt in range(1, args.max_retries + 1):
        response = call_vlm_api(
            full_prompt=full_prompt,
            image_paths=[reference_image_path, source_image_path, edited_image_path],
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            max_retries=1,
        )
        if response is None:
            if attempt < args.max_retries:
                time.sleep(2)
                continue
            return None, item

        raw_response_text = response["choices"][0]["message"]["content"]
        scores = parse_scores(raw_response_text)
        if not scores.get("error"):
            break
        tqdm.write(f"[retry {attempt}/{args.max_retries}] parse failed for {item_id}: {scores['error']}")
        if attempt < args.max_retries:
            time.sleep(2)

    if raw_response_text is None or scores is None:
        return None, item

    return {
        args.index_key: item_id,
        "method": args.method_name,
        "scores": scores,
        "instruction": prompt,
        "reference_image": item.get(args.reference_key),
        "background_image": item.get(args.source_key),
        "raw_response": raw_response_text,
    }, item


def sample_items(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_size is None:
        return data[: args.limit] if args.limit is not None else data

    masked_items = []
    for item in data:
        if args.mask_key not in item:
            continue
        mask_path = resolve_path(args.data_root, item[args.mask_key])
        if os.path.exists(mask_path):
            masked_items.append(item)

    if not masked_items:
        raise SystemExit("No masked items were found for sampling.")

    if args.sample_index_file and os.path.exists(args.sample_index_file):
        with open(args.sample_index_file, "r", encoding="utf-8") as file:
            sample_info = json.load(file)
        selected = set(sample_info["indices"])
        sampled = [item for item in masked_items if item.get(args.index_key) in selected]
        return sampled

    sample_size = min(args.sample_size, len(masked_items))
    random.seed(args.sample_seed)
    sampled = random.sample(masked_items, sample_size)

    if args.sample_index_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.sample_index_file)), exist_ok=True)
        with open(args.sample_index_file, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "seed": args.sample_seed,
                    "sample_size": sample_size,
                    "indices": [item[args.index_key] for item in sampled],
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
    return sampled


def run_round(items: list[dict[str, Any]], args: argparse.Namespace, results_dir: str, method_suffix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if args.workers <= 1:
        for item in tqdm(items, desc="VLM evaluation"):
            result, failed_item = evaluate_item(item, args, results_dir, method_suffix)
            if result is None:
                failures.append(failed_item)
            else:
                successes.append(result)
        return successes, failures

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(evaluate_item, item, args, results_dir, method_suffix): item for item in items
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="VLM evaluation"):
            result, failed_item = future.result()
            if result is None:
                failures.append(failed_item)
            else:
                successes.append(result)
    return successes, failures


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("An API key is required. Pass --api-key or set OPENAI_API_KEY.")

    method_suffix = args.method_suffix or METHOD_SUFFIX_MAP.get(args.method_name, f"{args.method_name}_result")
    results_dir = args.results_dir or os.path.join(args.results_base_dir, f"results_{args.method_name}")
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    with open(args.benchmark_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    data_to_process = sample_items(data, args)

    results, failed = run_round(data_to_process, args, results_dir, method_suffix)

    if args.retry_failed and failed:
        retry_rounds = 5
        pending = failed
        for round_idx in range(1, retry_rounds + 1):
            tqdm.write(f"Retry round {round_idx}: {len(pending)} pending items")
            recovered, pending = run_round(pending, args, results_dir, method_suffix)
            results.extend(recovered)
            if not pending:
                break
        failed = pending

    statistics = calculate_statistics(results)
    output = {
        "metadata": {
            "task": "Inter-Edit",
            "method": args.method_name,
            "method_suffix": method_suffix,
            "model": args.model,
            "base_url": args.base_url,
            "benchmark_json": args.benchmark_json,
            "results_dir": results_dir,
            "items_processed": len(data_to_process),
            "api_responses_received": len(results),
            "failed_items": len(failed),
        },
        "statistics": statistics,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(json.dumps(output["metadata"], indent=2))
    if "overall_average" in statistics:
        print(f"Overall average: {statistics['overall_average']:.3f}")
    print(f"Saved VLM evaluation to: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
