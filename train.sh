#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-./train_configs/train_config_2511.yaml}"
ACC_CONFIG_PATH="${ACC_CONFIG_PATH:-./acc_config.yaml}"

printf 'Launching Inter-Edit CJT training\n'
printf '  config: %s\n' "$CONFIG_PATH"
printf '  accelerate config: %s\n' "$ACC_CONFIG_PATH"

accelerate launch --config_file "$ACC_CONFIG_PATH" \
  train_qwen_edit_lora.py \
  --config "$CONFIG_PATH"
