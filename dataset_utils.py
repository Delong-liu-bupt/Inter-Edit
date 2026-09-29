"""Helpers for loading public Inter-Edit annotations."""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Iterable


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a config value from a dict-like object or OmegaConf node."""
    if config is None:
        return default

    if isinstance(config, dict):
        return config.get(key, default)

    if hasattr(config, key):
        value = getattr(config, key)
        return default if value is None else value

    getter = getattr(config, "get", None)
    if callable(getter):
        value = getter(key, default)
        return default if value is None else value

    return default


def _first_present(sample: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = sample.get(key)
        if value not in (None, ""):
            return value
    raise KeyError(f"Missing required keys {list(keys)} in sample: {list(sample.keys())}")


def resolve_path(path: str, data_root: str | None = None) -> str:
    if os.path.isabs(path) or not data_root:
        return path
    return os.path.join(data_root, path)


def get_original_image_path(sample: dict[str, Any], data_root: str | None = None) -> str:
    return resolve_path(
        _first_present(
            sample,
            (
                "original_image_path",
                "original_image_url",
                "background_image",
                "image_path",
                "source_file",
                "source_path",
            ),
        ),
        data_root,
    )


def get_edited_image_path(sample: dict[str, Any], data_root: str | None = None) -> str:
    return resolve_path(
        _first_present(
            sample,
            (
                "edited_image_path",
                "edited_image_url",
                "target_image",
                "target_file",
                "gt_path",
            ),
        ),
        data_root,
    )


def get_mask_image_path(sample: dict[str, Any], data_root: str | None = None) -> str:
    return resolve_path(
        _first_present(sample, ("mask_image_path", "mask", "mask_file", "mask_path")),
        data_root,
    )


def get_instruction(sample: dict[str, Any]) -> str:
    return _first_present(sample, ("instruction", "edit_instruction", "concise_edit_prompt", "edit_prompt"))


def get_target_image_name(sample: dict[str, Any]) -> str:
    return os.path.basename(get_edited_image_path(sample))


def load_json_samples(
    json_file: str,
    *,
    only_better_data: bool = False,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if json_file.endswith(".jsonl"):
        with open(json_file, "r", encoding="utf-8") as file:
            samples = [json.loads(line) for line in file if line.strip()]
    elif json_file.endswith(".jsonl.gz"):
        with gzip.open(json_file, "rt", encoding="utf-8") as file:
            samples = [json.loads(line) for line in file if line.strip()]
    else:
        with open(json_file, "r", encoding="utf-8") as file:
            samples = json.load(file)

    if only_better_data:
        samples = [sample for sample in samples if sample.get("better_data", False)]

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples
