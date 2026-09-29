# Inter-Edit: Control-Image Joint Training (CJT)

[![GitHub](https://img.shields.io/badge/GitHub-Inter--Edit-black?logo=github)](https://github.com/Delong-liu-bupt/Inter-Edit)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Inter--Edit--Train-yellow?logo=huggingface)](https://huggingface.co/datasets/a1557811266/Inter-Edit-Train)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Inter--Edit--Test-yellow?logo=huggingface)](https://huggingface.co/datasets/a1557811266/Inter-Edit-Test)

Official code release for the **CVPR 2026** paper **Inter-Edit: First Benchmark for Interactive Instruction-Based Image Editing**.

This repository releases the **Control-Image Joint Training (CJT)** method from the paper. CJT is our strongest and most practical training recipe: it fine-tunes `Qwen/Qwen-Image-Edit-2511` with two aligned control inputs, the **source image** and a **binary edit mask**, while keeping the training-time and inference-time preprocessing pipelines consistent.

## Overview

Interactive image editing requires more than prompt following: the model must preserve the original scene, edit only the intended region, and stay faithful to the user instruction. Inter-Edit introduces a benchmark and data pipeline for this setting, and this repository provides the public implementation of the CJT training recipe used in the paper.

This release includes:

- CJT training with `accelerate`
- command-line inference for checkpoint evaluation
- an interactive Gradio demo
- objective metrics and VLM-based subjective evaluation
- benchmark subset sampling and language-wise analysis tools

## Highlights

- **Paper-aligned release**: this repository is focused on the CJT method reported in the CVPR 2026 paper.
- **Public-data ready**: the loader supports the released Inter-Edit JSON schema and several backward-compatible key variants.
- **Simple reproduction path**: training, inference, demo, and evaluation are all included in a compact codebase.
- **Benchmark utilities**: objective scoring, VLM judging, reproducible subset sampling, and language breakdown analysis are integrated.

## Released Assets

- Code: `https://github.com/Delong-liu-bupt/Inter-Edit`
- Training set: `https://huggingface.co/datasets/a1557811266/Inter-Edit-Train`
- Test benchmark: `https://huggingface.co/datasets/a1557811266/Inter-Edit-Test`

If Hugging Face access is slow, especially from mainland China, use the mirror before downloading models or datasets:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

A recommended local layout is:

```text
data/
  json/
    Inter-Edit-train.json
    Inter-Edit-Test.json
  images/
  benchmark/
outputs/
```

## Installation

```bash
conda create -n inter-edit python=3.10 -y
conda activate inter-edit
pip install -r requirements.txt
```

Notes:

- `acc_config.yaml` is the default single-machine, single-GPU `accelerate` config shipped with this repository.
- `evaluation/objective_metrics.py` additionally depends on **AlphaCLIP**. Please install AlphaCLIP separately before running objective evaluation.

## Training Data Format

The training loader accepts `.json`, `.jsonl`, and `.jsonl.gz`. The public Inter-Edit format is a JSON list such as:

```json
{
  "edit_type": "Add",
  "instruction": "Add a glowing pair of chopsticks",
  "bounding_box": [357, 694, 902, 926],
  "original_image_path": "images/source/sample_0001.png",
  "edited_image_path": "images/target/sample_0001.png",
  "mask_image_path": "images/mask/sample_0001.png",
  "bbox_reference_dimensions": {"width": 960, "height": 960},
  "better_data": true
}
```

Compatibility notes:

- Older keys such as `original_image_url`, `edited_image_url`, and prompt aliases such as `edit_prompt` are still supported.
- If image paths in the JSON are relative, set `data_config.data_root` in the YAML config.
- `only_better_data: true` restricts training to samples marked with `better_data = true`.

## Training

Two config files are provided:

- `train_configs/train_config.yaml`: short debug configuration for launch checks
- `train_configs/train_config_2511.yaml`: full CJT training recipe for `Qwen-Image-Edit-2511`

Quick smoke test:

```bash
bash train.sh ./train_configs/train_config.yaml
```

Full training:

```bash
bash train.sh ./train_configs/train_config_2511.yaml
```

The public config defaults to `./data/json/Inter-Edit-train.json`. If your JSON stores relative paths, configure:

```yaml
data_config:
  json_file: ./data/json/Inter-Edit-train.json
  data_root: ./data
```

Useful options:

- `precompute_text_embeddings`: caches prompt features for faster training
- `precompute_image_embeddings`: caches VAE features and reduces runtime overhead
- `cached_embeddings_dir`: reuses precomputed embeddings
- `max_samples`: useful for debugging on a small subset
- `output_dir`: checkpoint directory, for example `./outputs/cjt_qwen_2511`

## Inference

The control image order is always:

1. source image
2. binary mask image

Example:

```bash
python inference.py \
  --lora_path ./outputs/cjt_qwen_2511/checkpoint-20000 \
  --control_images /path/to/source.png /path/to/mask.png \
  --prompt "Replace the mug with a transparent glass vase." \
  --output ./outputs/example.png
```

The script also supports `--control_dir` and `--control_pattern` for batched or structured local data.

## Gradio Demo

```bash
bash gradio_app.sh ./outputs/cjt_qwen_2511/checkpoint-20000
```

The demo supports the standard interactive workflow: upload the source image, paint the editable region, enter an instruction, and generate the edited result.

For quick testing, we also provide an example LoRA checkpoint for the Gradio demo:

- Demo LoRA checkpoint: `https://drive.google.com/file/d/1iPecHlpF79FJWtLTtSCNPUhGY4vUuSA8/view?usp=sharing`

## Converting Legacy LoRA Checkpoints

Some older checkpoints may still use `.lora.down/.lora.up` naming. They can be converted with:

```bash
python convert_lora_weights.py /path/to/pytorch_lora_weights.safetensors
```

The Gradio demo automatically converts these legacy LoRA keys when necessary.

## Evaluation

### 1. Objective metrics

This script computes the objective metrics used in the paper: `S_global_LPIPS`, `S_in_AlphaCLIP`, `S_out_AlphaCLIP`, `BSS_Laplacian`, and `S_ref_AlphaCLIP`.

```bash
python evaluation/objective_metrics.py \
  --benchmark-json /path/to/Inter-Edit-Test.json \
  --data-root /path/to/benchmark_root \
  --results-dir ./results_ours \
  --method-suffix ours_result \
  --alpha-clip-ckpt /path/to/clip_l14_grit1m_fultune_8xe.pth \
  --output-file ./evaluation/objective_ours.json
```

By default, the benchmark loader expects keys such as `background_image`, `target_image`, `mask`, and `reference_image`. Override them with `--source-key`, `--target-key`, `--mask-key`, and `--reference-key` if your benchmark JSON uses different field names.

### 2. VLM-based subjective evaluation

`evaluation/vlm_eval.py` supports OpenAI-compatible endpoints.

```bash
export OPENAI_API_KEY=YOUR_API_KEY

python evaluation/vlm_eval.py \
  --benchmark-json /path/to/Inter-Edit-Test.json \
  --data-root /path/to/benchmark_root \
  --method-name ours \
  --results-dir ./results_ours \
  --model gpt-4.1-mini \
  --output-json ./evaluation/vlm_ours.json
```

### 3. Reproducible benchmark sampling

Sampling is provided **only for day-to-day experimentation**, such as ablations, quick sanity checks, and internal iteration. **All final numbers reported in the CVPR 2026 paper must be computed on the full test benchmark, not on a sampled subset.**

```bash
python evaluation/sample_benchmark.py \
  --input-json /path/to/Inter-Edit-Test.json \
  --data-root /path/to/benchmark_root \
  --output-json ./evaluation/sample500.json \
  --sample-size 500 \
  --seed 42
```

### 4. Language-wise breakdown

```bash
python evaluation/language_breakdown.py \
  --results-json ./evaluation/objective_ours.json \
  --metadata-json /path/to/Inter-Edit-Test.json
```

## Repository Structure

- `train_qwen_edit_lora.py`: main CJT training entrypoint
- `dataset_utils.py`: public schema parsing and path resolution helpers
- `pipeline_qwenimage_edit_plus.py`: shared pipeline used by training and inference
- `inference.py`: command-line editing script
- `gradio_app_Inter_lora.py`: interactive demo interface
- `evaluation/`: objective metrics, VLM judging, sampling, and language analysis
- `convert_lora_weights.py`: utility for legacy checkpoint conversion
- `train_configs/`: debug and full training presets

## Citation

If you find Inter-Edit or this codebase useful in your research, please cite:

```bibtex
@inproceedings{liu2026interedit,
  title={Inter-Edit: First Benchmark for Interactive Instruction-Based Image Editing},
  author={Liu, Delong and Hou, Haotian and Hou, Zhaohui and Huang, Zhiyuan and Han, Shihao and Zhan, Mingjie and Zhao, Zhicheng and Su, Fei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Acknowledgement

This release builds on the open-source ecosystems around **Qwen-Image-Edit**, **Diffusers**, **PEFT**, and **Accelerate**.
