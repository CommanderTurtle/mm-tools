#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -x .venv/bin/python && -x .vl-venv/bin/python && -f node_modules/three/package.json ]] || {
  printf 'Run ./setupwithuv.sh first.\n' >&2; exit 1;
}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 WANDB_DISABLED=true OTEL_SDK_DISABLED=true PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=egl
exec .venv/bin/python server.py
