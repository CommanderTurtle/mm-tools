#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f .env ]] || { printf 'Missing .env. Run ./setupwithuv first.\n' >&2; exit 1; }
[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./setupwithuv first.\n' >&2; exit 1; }

set -a
# shellcheck disable=SC1091
source .env
set +a

HOST="${ACESTEP_HOST:-0.0.0.0}"
PORT="${ACESTEP_PORT:-8250}"
MODEL_ROOT="${ACESTEP_CHECKPOINTS_DIR:-../models/Ace-Step--Ace-Step1.5}"
if [[ "$MODEL_ROOT" != /* ]]; then
  MODEL_ROOT="$ROOT/$MODEL_ROOT"
fi
MODEL_ROOT="$(realpath -m -- "$MODEL_ROOT")"
export ACESTEP_CHECKPOINTS_DIR="$MODEL_ROOT"

for required in \
  "$MODEL_ROOT/acestep-v15-xl-sft/model.safetensors.index.json" \
  "$MODEL_ROOT/acestep-v15-xl-sft/model-00001-of-00004.safetensors" \
  "$MODEL_ROOT/acestep-v15-xl-sft/model-00002-of-00004.safetensors" \
  "$MODEL_ROOT/acestep-v15-xl-sft/model-00003-of-00004.safetensors" \
  "$MODEL_ROOT/acestep-v15-xl-sft/model-00004-of-00004.safetensors" \
  "$MODEL_ROOT/acestep-5Hz-lm-4B/model.safetensors.index.json" \
  "$MODEL_ROOT/acestep-5Hz-lm-4B/model-00001-of-00002.safetensors" \
  "$MODEL_ROOT/acestep-5Hz-lm-4B/model-00002-of-00002.safetensors" \
  "$MODEL_ROOT/Qwen3-Embedding-0.6B/model.safetensors" \
  "$MODEL_ROOT/vae/diffusion_pytorch_model.safetensors"; do
  [[ -f "$required" ]] || {
    printf 'ACE-Step model artifact is missing: %s\nRun models/download_models.py first.\n' "$required" >&2
    exit 1
  }
done

LOCAL_URL="http://127.0.0.1:${PORT}"
if curl --fail --silent --max-time 2 "$LOCAL_URL" >/dev/null 2>&1; then
  printf 'ACE-Step is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  printf 'ACE-Step port %s is already occupied.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
printf 'ACE-Step private workbench: %s\n' "$LOCAL_URL"
printf 'The ACE-Step DiT, language model, and embedding model load in this process. Ctrl+C releases them.\n'
exec uv run --active --no-sync python -m acestep.acestep_v15_pipeline \
  --server-name "$HOST" \
  --port "$PORT" \
  --init_service true \
  --init_llm "${ACESTEP_INIT_LM:-true}" \
  --config_path "${ACESTEP_CONFIG_PATH:-acestep-v15-xl-sft}" \
  --lm_model_path "${ACESTEP_LM_MODEL:-acestep-5Hz-lm-4B}" \
  --backend "${ACESTEP_LM_BACKEND:-pt}" \
  --device "${ACESTEP_DEVICE:-auto}" \
  --offload_to_cpu "${ACESTEP_OFFLOAD_TO_CPU:-false}" \
  --offload_dit_to_cpu "${ACESTEP_OFFLOAD_DIT_TO_CPU:-false}"
