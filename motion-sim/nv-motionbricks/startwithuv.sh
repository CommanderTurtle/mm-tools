#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -x "$ROOT/.venv/bin/python" && -f "$ROOT/node_modules/three/build/three.module.js" ]] || { printf 'Run setupwithuv.sh first.\n' >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8221}" --ws-max-size 16777216 --timeout-graceful-shutdown 8
