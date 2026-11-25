#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
cd "$ROOT"
for bin in uv git bun nvcc; do
  command -v "$bin" >/dev/null || { printf '%s is required on PATH.\n' "$bin" >&2; exit 1; }
done
PROFILE="${SCENEBLOOM_CUDA:-cu128}"
case "$PROFILE" in
  cu128) TORCH=2.7.1; VISION=0.22.1; ARCH='12.0' ;;
  cu121) TORCH=2.5.1; VISION=0.20.1; ARCH='8.0;8.6;8.9;9.0+PTX' ;;
  *) printf 'Choose SCENEBLOOM_CUDA=cu128 (5090) or cu121 (native profile).\n' >&2; exit 1 ;;
esac
export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}" FORCE_CUDA=1
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
if [[ -f .venv/scene-stack && "$(<.venv/scene-stack)" != "$PROFILE" ]]; then
  printf 'Move the existing .venv before changing its CUDA profile.\n' >&2; exit 1
fi
[[ -x .venv/bin/python ]] || uv venv .venv --python 3.10 --seed --managed-python
printf '%s' "$PROFILE" > .venv/scene-stack
PY="$ROOT/.venv/bin/python"
"$PY" setup.py
uv pip install --python "$PY" --index-url "https://download.pytorch.org/whl/$PROFILE" \
  "torch==$TORCH+$PROFILE" "torchvision==$VISION+$PROFILE"
CONSTRAINTS="$(mktemp)"
trap 'rm -f -- "$CONSTRAINTS"' EXIT
printf 'torch==%s+%s\ntorchvision==%s+%s\nnumpy==1.26.4\n' "$TORCH" "$PROFILE" "$VISION" "$PROFILE" > "$CONSTRAINTS"
uv pip install --python "$PY" -c "$CONSTRAINTS" -r requirements.txt
# Native Genesis deliberately uses this dependency set instead of its conflicting
# package pins. Extensions build only after Torch is fixed to the selected CUDA.
uv pip install --python "$PY" --no-deps genesis-world==0.2.1
for module in pytorch3d detectron2 groundingdino depthpro sam; do
  uv pip install --python "$PY" -c "$CONSTRAINTS" --no-build-isolation --no-deps "$ROOT/$module"
done
[[ -x .vl-venv/bin/python ]] || uv venv .vl-venv --python 3.12 --seed --managed-python
uv pip install --python .vl-venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130
uv pip install --python .vl-venv/bin/python -r "$ROOT/../../workflows/ComfyUI/requirements.txt"
bun install --frozen-lockfile --ignore-scripts
printf '\nSceneBloom ready. Start with ./startwithuv.sh\n'
