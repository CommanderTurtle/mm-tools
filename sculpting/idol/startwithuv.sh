#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv/bin/python" && -f "$ROOT/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js" ]] || {
  printf 'Run ./setupwithuv.sh first.\n' >&2; exit 1;
}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 OTEL_SDK_DISABLED=true WANDB_DISABLED=true
exec "$ROOT/.venv/bin/python" "$ROOT/server.py"
