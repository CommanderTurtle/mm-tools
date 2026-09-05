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

HOST="${MINIMAX_HOST:-0.0.0.0}"
PORT="${MINIMAX_PORT:-8254}"
MODEL_ROOT="${MINIMAX_MODEL_DIR:-../models/Comfy-Org--Minimax-Music-3}"
if [[ "$MODEL_ROOT" != /* ]]; then
  MODEL_ROOT="$ROOT/$MODEL_ROOT"
fi
MODEL_ROOT="$(realpath -m -- "$MODEL_ROOT")"
export MINIMAX_MODEL_DIR="$MODEL_ROOT"

GUIDE_ROOT="${MINIMAX_GUIDE_MODEL_ROOT:-../models/qwen}"
if [[ "$GUIDE_ROOT" != /* ]]; then
  GUIDE_ROOT="$ROOT/$GUIDE_ROOT"
fi
GUIDE_ROOT="$(realpath -m -- "$GUIDE_ROOT")"
export MINIMAX_GUIDE_MODEL_ROOT="$GUIDE_ROOT"
GUIDE_DEFAULT="${MINIMAX_GUIDE_DEFAULT_MODEL:-text-encoder-vl-nvfp4/qwen3_vl_4b_nvfp4_full.safetensors}"

for artifact in \
  diffusion_models/minimax_music3_dit_fp16.safetensors \
  text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors \
  vae/minimax_music3_dav.safetensors; do
  [[ -f "$MODEL_ROOT/$artifact" ]] || {
    printf 'MiniMax Music 3 artifact is missing: %s\nRun models/download_models.py first.\n' "$MODEL_ROOT/$artifact" >&2
    exit 1
  }
done

if [[ ! -f "$GUIDE_ROOT/$GUIDE_DEFAULT" ]]; then
  printf 'Optional Prompt Guide checkpoint is not present: %s\nThe studio will still start; use Browse in the Prompt Guide tab or run models/download_models.py.\n' "$GUIDE_ROOT/$GUIDE_DEFAULT" >&2
fi

LOCAL_URL="http://127.0.0.1:${PORT}"
if curl --fail --silent --max-time 2 "$LOCAL_URL/api/health" >/dev/null 2>&1; then
  printf 'MiniMax Music Studio is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  printf 'MiniMax Music Studio port %s is already occupied.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
printf 'MiniMax Music Studio: %s\n' "$LOCAL_URL"
printf 'The song engine is private to this process. Prompt Guide owns a second loopback engine only while its switch is on. Ctrl+C unloads and stops both.\n'
exec python -m uvicorn local_app.server:app \
  --host "$HOST" \
  --port "$PORT"
