#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

uv venv --python 3.12.10 --seed --managed-python .venv

# PyTorch's CUDA wheels are selectable for future hosts. The workstation's
# established Blackwell/NVFP4 environments use the cached CUDA 13.0 build.
torch_index="${TRANSLATE_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
uv pip install --python .venv/bin/python \
  --index-url "$torch_index" torch torchvision torchaudio
uv pip install --python .venv/bin/python -r requirements-core.txt

mkdir -p .runtime
if [[ ! -d .runtime/ComfyUI/.git ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Comfy-Org/ComfyUI.git .runtime/ComfyUI
  git -C .runtime/ComfyUI sparse-checkout set comfy
else
  git -C .runtime/ComfyUI pull --ff-only
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

printf '%s\n' \
  'Local translation runtime prepared.' \
  'Review .env, then run ./starthttp.sh.'
