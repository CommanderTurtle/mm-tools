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

HOST="${SYMPHONY_HOST:-0.0.0.0}"
PORT="${SYMPHONY_PORT:-8252}"
MODEL_ROOT="${SYMPHONY_MODEL_DIR:-../models/SymphonyGen--SymphonyGen}"
OUTPUT_ROOT="${SYMPHONY_OUTPUT_DIR:-outputs}"
[[ "$MODEL_ROOT" == /* ]] || MODEL_ROOT="$ROOT/$MODEL_ROOT"
[[ "$OUTPUT_ROOT" == /* ]] || OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
export SYMPHONY_MODEL_DIR="$(realpath -m -- "$MODEL_ROOT")"
export SYMPHONY_OUTPUT_DIR="$(realpath -m -- "$OUTPUT_ROOT")"

for name in stage_one_pretrained.pt stage_two_pretrained.pt grpo_clamp_epoch_10.pt 'grpo_clamp+track_epoch_6.pt'; do
  [[ -f "$SYMPHONY_MODEL_DIR/$name" ]] || {
    printf 'SymphonyGen checkpoint is missing: %s\nRun models/download_models.py first.\n' "$SYMPHONY_MODEL_DIR/$name" >&2
    exit 1
  }
done

LOCAL_URL="http://127.0.0.1:${PORT}"
if curl --fail --silent --max-time 2 "$LOCAL_URL/api/status" >/dev/null 2>&1; then
  printf 'SymphonyGen Studio is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  printf 'SymphonyGen port %s is already occupied.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
printf 'SymphonyGen private studio: %s\n' "$LOCAL_URL"
printf 'The server is model-free while idle; each generation worker releases its model on completion.\n'
exec uv run --active --no-sync python -m uvicorn local_app.server:app \
  --host "$HOST" \
  --port "$PORT"
