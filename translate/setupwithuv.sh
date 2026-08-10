#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12.10 --seed --managed-python .venv
else
  printf 'Reusing existing isolated environment: %s\n' "$here/.venv"
fi

accelerator="${TRANSLATE_ACCELERATOR:-gpu}"
if [[ "$accelerator" == "gpu" ]]; then
  torch_index="${TRANSLATE_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
else
  torch_index="${TRANSLATE_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
fi
uv pip install --python .venv/bin/python \
  --index-url "$torch_index" torch

if [[ "$accelerator" == "gpu" ]]; then
  cudacxx="${TRANSLATE_CUDACXX:-$(command -v nvcc || true)}"
  if [[ -z "$cudacxx" && -x /usr/local/cuda/bin/nvcc ]]; then
    cudacxx=/usr/local/cuda/bin/nvcc
  fi
  [[ -n "$cudacxx" ]] || {
    printf 'CUDA setup requested, but nvcc is not on PATH. Set TRANSLATE_CUDACXX or use TRANSLATE_ACCELERATOR=cpu.\n' >&2
    exit 1
  }
  CUDACXX="$cudacxx" CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$cudacxx" FORCE_CMAKE=1 \
    uv pip install --python .venv/bin/python -r requirements-core.txt
else
  uv pip install --python .venv/bin/python -r requirements-core.txt
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

printf '%s\n' \
  'Local EraX translation runtime prepared.' \
  'Review .env, then run ./starthttp.sh.'
