#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for cmd in uv bun cmake g++ ffmpeg ffprobe; do command -v "$cmd" >/dev/null || { printf 'Install %s first.\n' "$cmd" >&2; exit 1; }; done
PROFILE="${FLOWHMR_TORCH_PROFILE:-blackwell}"
case "$PROFILE" in
  blackwell) TORCH=2.7.1; VISION=0.22.1; CUDA=cu128 ;;
  upstream) TORCH=2.5.1; VISION=0.20.1; CUDA=cu118 ;;
  *) printf 'FLOWHMR_TORCH_PROFILE must be blackwell or upstream.\n' >&2; exit 1 ;;
esac
uv run --no-project --python 3.10 "$ROOT/setup.py"
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv --python 3.10 "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" "torch==$TORCH" "torchvision==$VISION" --index-url "https://download.pytorch.org/whl/$CUDA"
uv pip install --python "$PY" -r "$ROOT/requirements.txt" "torch==$TORCH" "torchvision==$VISION"
uv pip install --python "$PY" --no-build-isolation --no-deps -e "$ROOT/flowhmr/local_app/YOLOX"
uv pip install --python "$PY" --no-build-isolation --no-deps "$ROOT/flowhmr/local_app/KimodoMotionCorrection" "$ROOT/flowhmr/local_app/RGBMocapCorrection"
uv pip install --python "$PY" fbxsdkpy==2020.1.post2 --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
bun install --cwd "$ROOT" --frozen-lockfile --ignore-scripts
printf 'Ready (%s). Run %s/startwithuv.sh\n' "$PROFILE" "$ROOT"
