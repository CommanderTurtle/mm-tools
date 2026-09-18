#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv/bin/python" ]] || { printf 'Run bash ./setupwithuv.sh first.\n' >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROOT/.venv/bin/activate"
if [[ -f "$ROOT/.env.local" ]]; then set -a; source "$ROOT/.env.local"; set +a; fi
export PYTHONPATH="$ROOT/../..:$ROOT:$ROOT/trellis2_x2${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FF_FIRE3D_MODEL_DTYPE="${FF_FIRE3D_MODEL_DTYPE:-bfloat16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MM_STUDIO_HOST="${FIRE3D_STUDIO_HOST:-${MM_STUDIO_HOST:-127.0.0.1}}"
export MM_STUDIO_PORT="${FIRE3D_STUDIO_PORT:-${MM_STUDIO_PORT:-8268}}"
printf 'Fire3D Scene Foundry: http://%s:%s\n' "$MM_STUDIO_HOST" "$MM_STUDIO_PORT"
exec python "$ROOT/local_app/server.py" \
  --project-root "$ROOT" \
  --manifest "$ROOT/local_app/studio.json" \
  --adapter "$ROOT/local_app/adapter.py" \
  --host "$MM_STUDIO_HOST" \
  --port "$MM_STUDIO_PORT"
