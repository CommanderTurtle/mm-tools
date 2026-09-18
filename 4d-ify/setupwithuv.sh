#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export UV_LINK_MODE=hardlink
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
command -v uv >/dev/null || { echo "uv is required." >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "An NVIDIA CUDA workstation is required." >&2; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_root in /usr/local/cuda /usr/local/cuda-*; do
    if [[ -x "$cuda_root/bin/nvcc" ]]; then
      export CUDA_HOME="$cuda_root"
      export PATH="$CUDA_HOME/bin:$PATH"
      break
    fi
  done
fi
command -v nvcc >/dev/null || { echo "The CUDA toolkit is required for DPVO." >&2; exit 1; }
command -v g++ >/dev/null || { echo "A C++ compiler is required for DPVO." >&2; exit 1; }
cuda_root="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_HOME="${CUDA_HOME:-$cuda_root}"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ -d /usr/lib/wsl/lib ]]; then
  export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
uv venv --python 3.11 --seed .venv
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" - <<'PY'
from pathlib import Path

import torch

header = Path(torch.__file__).resolve().parent / "include/ATen/core/List_inl.h"
source = header.read_text()
source = source.replace(
    "static_cast<typename decltype(impl_->list)::difference_type>(pos)",
    "static_cast<std::ptrdiff_t>(pos)",
)
header.write_text(source)
PY
nvcc_major="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -n1)"
torch_cuda_major="$("$PY" -c 'import torch; print(torch.version.cuda.split(".")[0])')"
if [[ -n "$nvcc_major" && "$nvcc_major" -gt "$torch_cuda_major" ]]; then
  export MMTOOLS_FORWARD_CUDA=1
fi
uv pip install --python "$PY" 'setuptools<81' wheel ninja -r requirements.txt -r local_app/requirements.txt
uv pip install --python "$PY" 'torch-scatter==2.1.2+pt28cu128' -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip uninstall --python "$PY" smplx >/dev/null 2>&1 || true
uv pip install --python "$PY" --no-build-isolation --no-deps "$ROOT"
PYTHONPATH="$ROOT/.." "$PY" - <<'PY'
from pathlib import Path

from fdanyone.assets import BIREFNET_FILES, MODEL_FILES
from fdanyone.download import create_classic_gvhmr_links, create_dpvo_gvhmr_link, ensure_smplx
import torch
import cuda_ba
import cuda_corr
import dpvo
import lietorch_backends
import pycolmap
import smplx
import torch_scatter

root = Path.cwd()
expected_smplx = (root / "smplx" / "__init__.py").resolve()
if Path(smplx.__file__).resolve() != expected_smplx:
    raise SystemExit(f"The integrated SMPL-X runtime is not active: {smplx.__file__}")
missing = [str(root / "models" / item) for item in MODEL_FILES if not (root / "models" / item).is_file()]
missing.extend(
    str(root / "models" / "birefnet" / item)
    for item in BIREFNET_FILES
    if not (root / "models" / "birefnet" / item).is_file()
)
if missing:
    raise SystemExit("Missing model artifacts. Run ../models/download_models.py for 4d first:\n" + "\n".join(missing))

ensure_smplx("models", ".")
create_classic_gvhmr_links("models", ".", require_smplx=True)
create_dpvo_gvhmr_link("models", ".", required=False)
assert torch.cuda.is_available(), "CUDA is required"
major, _ = torch.cuda.get_device_capability()
assert major >= 12, "The local RTX 5090 CUDA build is not active"
print(f"4DAnyone environment ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY
echo "4DAnyone Capture Studio is ready."
