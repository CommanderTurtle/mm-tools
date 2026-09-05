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

HOST="${STABLE_AUDIO_HOST:-0.0.0.0}"
PORT="${STABLE_AUDIO_PORT:-8251}"
MODEL_ROOT="${STABLE_AUDIO_MODEL_DIR:-../models/RoyalCities--Foundation-1}"
T5_ROOT="${STABLE_AUDIO_T5_MODEL_PATH:-../models/google-t5--t5-base}"
[[ "$MODEL_ROOT" == /* ]] || MODEL_ROOT="$ROOT/$MODEL_ROOT"
[[ "$T5_ROOT" == /* ]] || T5_ROOT="$ROOT/$T5_ROOT"
MODEL_ROOT="$(realpath -m -- "$MODEL_ROOT")"
T5_ROOT="$(realpath -m -- "$T5_ROOT")"
export STABLE_AUDIO_MODEL_DIR="$MODEL_ROOT"
export STABLE_AUDIO_T5_MODEL_PATH="$T5_ROOT"

for required in \
  "$MODEL_ROOT/model_config.json" \
  "$MODEL_ROOT/Foundation_1.safetensors" \
  "$T5_ROOT/config.json"; do
  [[ -f "$required" ]] || {
    printf 'Stable Audio model artifact is missing: %s\nRun models/download_models.py first.\n' "$required" >&2
    exit 1
  }
done

LOCAL_URL="http://127.0.0.1:${PORT}"
if curl --fail --silent --max-time 2 "$LOCAL_URL" >/dev/null 2>&1; then
  printf 'Stable Audio is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  printf 'Stable Audio port %s is already occupied.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
printf 'Stable Audio Foundation private workbench: %s\n' "$LOCAL_URL"
printf 'Foundation-1 and its local T5 conditioner load in this process. Ctrl+C releases them.\n'
exec uv run --active --no-sync python run_gradio.py \
  --model-config "$MODEL_ROOT/model_config.json" \
  --ckpt-path "$MODEL_ROOT/Foundation_1.safetensors" \
  --host "$HOST" \
  --port "$PORT" \
  --title 'Foundation-1 Local Studio'
