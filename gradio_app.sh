#!/usr/bin/env bash
set -euo pipefail

LORA_PATH="${1:-}"
PORT="${PORT:-7860}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen-Image-Edit-2511}"

if [[ -z "$LORA_PATH" ]]; then
  echo "Usage: bash gradio_app.sh /path/to/checkpoint [extra gradio args]"
  exit 1
fi

shift || true

python gradio_app_Inter_lora.py \
  --lora_path "$LORA_PATH" \
  --base_model "$BASE_MODEL" \
  --port "$PORT" \
  "$@"
