#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo "Run bash ./setupwithuv.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
set -a
[[ ! -f .env.local ]] || source .env.local
set +a
export PYTHONPATH="$ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python "$ROOT/local_app/server.py" \
  --manifest "$ROOT/local_app/studio.json" \
  --adapter "$ROOT/local_app/adapter.py" \
  --project-root "$ROOT" \
  --host "${MMTOOLS_HOST:-127.0.0.1}" \
  --port "${MMTOOLS_PORT:-8261}"
