#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv/bin/python" && -s "$ROOT/node_modules/three/build/three.module.js" && -s "$ROOT/.native-python.json" ]] || {
  printf 'Run ./setupwithuv.sh first.\n' >&2; exit 1;
}
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 OTEL_SDK_DISABLED=true
exec "$ROOT/.venv/bin/python" "$ROOT/server.py" "$@"
