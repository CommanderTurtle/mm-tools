#!/usr/bin/env bash
# Single-node Nomas-PE endpoint (8 GPUs, DP=1, TP=8).
set -euo pipefail
: "${PE_MODEL_PATH:?Set PE_MODEL_PATH to the local Nomas-PE checkpoint}"
args=(
  --model-path "${PE_MODEL_PATH}"
  --served-model-name "${PE_MODEL:-Nomas-PE}"
  --trust-remote-code
  --tp-size "${PE_TP_SIZE:-8}"
  --dp-size "${PE_DP_SIZE:-1}"
  --nnodes 1
  --mem-fraction-static "${PE_MEM_FRACTION_STATIC:-0.85}"
  --reasoning-parser qwen3
  --host "${PE_HOST:-0.0.0.0}"
  --port "${PE_PORT:-30000}"
)
if [[ -n "${PE_API_KEY:-}" ]]; then args+=(--api-key "${PE_API_KEY}"); fi
if [[ -n "${PE_CONTEXT_LENGTH:-}" ]]; then args+=(--context-length "${PE_CONTEXT_LENGTH}"); fi
exec python3 -m sglang.launch_server "${args[@]}" "$@"
