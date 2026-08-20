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

HOST="${VOCALRENDER_HOST:-0.0.0.0}"
PORT="${VOCALRENDER_PORT:-8253}"
MODEL_ROOT="${VOCALRENDER_MODEL_DIR:-../models/pymaster--VocalRender}"
OUTPUT_ROOT="${VOCALRENDER_OUTPUT_DIR:-outputs}"
[[ "$MODEL_ROOT" == /* ]] || MODEL_ROOT="$ROOT/$MODEL_ROOT"
[[ "$OUTPUT_ROOT" == /* ]] || OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
export VOCALRENDER_MODEL_DIR="$(realpath -m -- "$MODEL_ROOT")"
export VOCALRENDER_OUTPUT_DIR="$(realpath -m -- "$OUTPUT_ROOT")"

for variant in VocalRender VocalRender-Pro; do
  for name in config.json model.safetensors audiovae.pth tokenizer.json; do
    [[ -f "$VOCALRENDER_MODEL_DIR/$variant/$name" ]] || {
      printf 'VocalRender artifact is missing: %s\nRun models/download_models.py first.\n' "$VOCALRENDER_MODEL_DIR/$variant/$name" >&2
      exit 1
    }
  done
done

LOCAL_URL="http://127.0.0.1:${PORT}"
if curl --fail --silent --max-time 2 "$LOCAL_URL/api/status" >/dev/null 2>&1; then
  printf 'VocalRender Studio is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
  printf 'VocalRender port %s is already occupied.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
printf 'VocalRender private studio: %s\n' "$LOCAL_URL"
printf 'Use Load/Unload in the UI; Ctrl+C always releases the active model.\n'
exec uv run --active --no-sync python -m uvicorn local_app.server:app \
  --host "$HOST" \
  --port "$PORT"
