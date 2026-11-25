#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
[[ -x "$ROOT/.venv/bin/python" && -f "$ROOT/node_modules/three/build/three.module.js" ]] || { printf 'Run setupwithuv.sh first.\n' >&2; exit 1; }
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled WANDB_DISABLED=true
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/server.py"
