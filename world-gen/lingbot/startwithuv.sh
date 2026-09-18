#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv/bin/python" ]] || { printf 'Run bash ./setupwithuv.sh first.\n' >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROOT/.venv/bin/activate"
if [[ -f "$ROOT/.env.local" ]]; then set -a; source "$ROOT/.env.local"; set +a; fi
export PYTHONPATH="$ROOT/../..:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export MM_STUDIO_HOST="${LINGBOT_STUDIO_HOST:-${MM_STUDIO_HOST:-127.0.0.1}}"
export MM_STUDIO_PORT="${LINGBOT_STUDIO_PORT:-${MM_STUDIO_PORT:-8267}}"
printf 'LingBot World Studio: http://%s:%s\n' "$MM_STUDIO_HOST" "$MM_STUDIO_PORT"
exec python "$ROOT/local_app/server.py" --project-root "$ROOT" --manifest "$ROOT/local_app/studio.json" --adapter "$ROOT/local_app/adapter.py" --host "$MM_STUDIO_HOST" --port "$MM_STUDIO_PORT"
