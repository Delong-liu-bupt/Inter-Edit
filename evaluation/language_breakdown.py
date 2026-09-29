"""Compute per-language score breakdowns for objective or VLM evaluation outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute language-wise score breakdowns")
    parser.add_argument("--results-json", required=True, help="Evaluation JSON produced by this repo.")
    parser.add_argument("--metadata-json", required=True, help="Benchmark metadata JSON containing language labels.")
    parser.add_argument("--output-json", default=None, help="Output path. Defaults to overwriting --results-json.")
    parser.add_argument("--language-key", default="language", help="Language key in the metadata JSON.")
    parser.add_argument("--id-key", default=None, help="Optional ID key override. Auto-detected when omitted.")
    parser.add_argument(
        "--keep-languages",
        nargs="*",
        default=None,
        help="Optional allowlist of languages to keep, e.g. English Chinese.",
    )
    return parser.parse_args()


def detect_result_entries(results_data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(results_data.get("detailed_results"), list):
        return "detailed_results", results_data["detailed_results"]
    if isinstance(results_data.get("results"), list):
        return "results", results_data["results"]
    raise SystemExit("Could not find `detailed_results` or `results` in the evaluation JSON.")


def detect_id_key(entries: list[dict[str, Any]], explicit_id_key: str | None) -> str:
    if explicit_id_key:
        return explicit_id_key
    for key in ("index", "id"):
        if entries and key in entries[0]:
            return key
    raise SystemExit("Could not infer the result ID key. Pass --id-key explicitly.")


def extract_metrics(entry: dict[str, Any]) -> dict[str, float]:
    if isinstance(entry.get("scores"), dict):
        return {
            key: float(value)
            for key, value in entry["scores"].items()
            if isinstance(value, (int, float))
        }

    skip_keys = {"index", "id", "method"}
    return {
        key: float(value)
        for key, value in entry.items()
        if key not in skip_keys and isinstance(value, (int, float))
    }


def main() -> None:
    args = parse_args()

    with open(args.results_json, "r", encoding="utf-8") as file:
        results_data = json.load(file)
    with open(args.metadata_json, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    result_key, entries = detect_result_entries(results_data)
    id_key = detect_id_key(entries, args.id_key)

    metadata_lookup = {}
    for item in metadata:
        item_id = item.get(id_key) or item.get("index") or item.get("id")
        if item_id is None:
            continue
        language = item.get(args.language_key)
        if language is None:
            continue
        metadata_lookup[item_id] = language

    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    item_counts: dict[str, int] = defaultdict(int)

    for entry in entries:
        entry_id = entry.get(id_key)
        if entry_id is None:
            continue
        language = metadata_lookup.get(entry_id)
        if language is None:
            continue
        if args.keep_languages and language not in args.keep_languages:
            continue

        metrics = extract_metrics(entry)
        if not metrics:
            continue

        item_counts[language] += 1
        for key, value in metrics.items():
            aggregates[language][key].append(value)

    language_breakdown = {}
    for language, metric_map in aggregates.items():
        language_breakdown[language] = {
            "count": item_counts[language],
            "metrics": {
                metric_name: {
                    "count": len(values),
                    "average": float(np.mean(values)),
                }
                for metric_name, values in metric_map.items()
                if values
            },
        }

    results_data["language_breakdown"] = language_breakdown
    if result_key == "results" and isinstance(results_data.get("statistics"), dict):
        results_data["statistics"]["language_breakdown"] = language_breakdown

    output_path = args.output_json or args.results_json
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results_data, file, indent=2, ensure_ascii=False)

    print(json.dumps(language_breakdown, indent=2, ensure_ascii=False))
    print(f"Saved language breakdown to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
