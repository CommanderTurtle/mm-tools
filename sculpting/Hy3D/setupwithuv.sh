#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for tool in uv git bun nvcc; do
  command -v "$tool" >/dev/null || { printf '%s is required.\n' "$tool" >&2; exit 1; }
done
STACK="${1:-cu124}"
case "$STACK" in
  cu124) TORCH=2.5.0; VISION=0.20.0; SPCONV='spconv-cu120==2.3.6'; FLASH=2.7.4.post1 ;;
  cu128) TORCH=2.7.1; VISION=0.22.1; SPCONV='spconv-cu126==2.3.8'; FLASH=2.8.3.post2 ;;
  *) printf 'Usage: ./setupwithuv.sh [cu124|cu128]\n' >&2; exit 1 ;;
esac
if [[ -f "$ROOT/.venv/.hy3d-stack" && "$(< "$ROOT/.venv/.hy3d-stack")" != "$STACK" ]]; then
  printf 'Move the existing .venv before selecting another CUDA profile.\n' >&2; exit 1
fi
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 OTEL_SDK_DISABLED=true
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv "$ROOT/.venv" --python 3.10 --seed
PY="$ROOT/.venv/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Hy3D requires a Python 3.10 environment"'
printf '%s\n' "$STACK" > "$ROOT/.venv/.hy3d-stack"
"$PY" "$ROOT/setup.py"
uv pip install --python "$PY" --index-url "https://download.pytorch.org/whl/$STACK" \
  "torch==$TORCH+$STACK" "torchvision==$VISION+$STACK"
uv pip install --python "$PY" -c "$ROOT/requirements-$STACK.txt" -r "$ROOT/requirements.txt" "$SPCONV"
MAX_JOBS="${MAX_JOBS:-4}" uv pip install --python "$PY" -c "$ROOT/requirements-$STACK.txt" --no-build-isolation "flash-attn==$FLASH"
uv pip install --python "$PY" --no-deps "$ROOT/hi3d"
(cd "$ROOT" && bun install --frozen-lockfile --ignore-scripts)
printf '\nHy3D is ready (%s). Run ./startwithuv.sh\n' "$STACK"
