#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCULPTING_VENV:-$ROOT/.venv}"
NATIVE="$ROOT/native"
STUDIO="$ROOT/local_app"

command -v uv >/dev/null 2>&1 || { printf 'uv is required. Install uv, then rerun this script.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'Sculpting Studio requires an NVIDIA CUDA workstation.\n' >&2; exit 1; }
# WSL installations commonly expose the toolkit at /usr/local/cuda/bin while
# non-login tmux/systemd shells omit it from PATH. Resolve the canonical
# toolkit before validating it so setup behaves the same interactively and in
# Sandwich-supervised/background runs.
if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_root in /usr/local/cuda /usr/local/cuda-*; do
    if [[ -x "$cuda_root/bin/nvcc" ]]; then
      export CUDA_HOME="$cuda_root"
      export PATH="$CUDA_HOME/bin:$PATH"
      break
    fi
  done
fi
command -v nvcc >/dev/null 2>&1 || { printf 'The CUDA toolkit (nvcc) is required to compile the pinned mesh kernels.\n' >&2; exit 1; }
command -v g++ >/dev/null 2>&1 || { printf 'A C++ compiler is required to compile the pinned mesh kernels.\n' >&2; exit 1; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [[ -z "$GPU_NAME" || "${GPU_MIB:-0}" -lt 30000 ]]; then
  printf 'The canonical sculpting profile requires a CUDA GPU with at least 30,000 MiB VRAM; found %s (%s MiB).\n' "${GPU_NAME:-none}" "${GPU_MIB:-0}" >&2
  exit 1
fi

required=(
  "$ROOT/pretrained/TRELLIS.2-4B/pipeline.json"
  "$ROOT/pretrained/TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
  "$ROOT/world/pretrained/Pixal3D/pipeline.json"
  "$ROOT/pretrained/deps/camenduru--dinov3-vitl16-pretrain-lvd1689m/model.safetensors"
  "$ROOT/pretrained/deps/ZhengPeng7--BiRefNet/model.safetensors"
  "$ROOT/pretrained/deps/Ruicheng--moge-2-vitl/model.pt"
  "$ROOT/pretrained/deps/valeoai--NAF/naf_release.pth"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { printf 'Missing model artifact: %s\nRun ../models/download_models.py sculpting first.\n' "$path" >&2; exit 1; }
done

mkdir -p "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"

export UV_LINK_MODE=hardlink
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${MAX_JOBS:-2}"
cuda_root="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_HOME="${CUDA_HOME:-$cuda_root}"
if [[ -d /usr/lib/wsl/lib ]]; then
  export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12 --seed --managed-python
fi

uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
uv pip install --python "$VENV/bin/python" \
  'setuptools<81' wheel packaging ninja \
  -r "$STUDIO/requirements.txt" \
  -r "$ROOT/requirements-local.txt"
uv pip install --python "$VENV/bin/python" \
  'natten==0.21.6+torch2110cu130' -f https://whl.natten.org

# CUDA 13.3's nvcc rejects this dependent decltype cast in the PyTorch 2.11
# header even though host C++ accepts it. Rewrite only the exact affected
# expression to its equivalent concrete ListImpl type before compiling the
# pinned extensions. Newer PyTorch headers are left untouched.
torch_list_header="$($VENV/bin/python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include/ATen/core/List_inl.h")')"
torch_list_cast='static_cast<typename decltype(impl_->list)::difference_type>(pos)'
if grep -Fq "$torch_list_cast" "$torch_list_header"; then
  perl -0pi -e 's/static_cast<typename decltype\(impl_->list\)::difference_type>\(pos\)/static_cast<c10::detail::ListImpl::list_type::difference_type>(pos)/g' "$torch_list_header"
fi

# Build the project-owned native components against the installed Blackwell stack.
for native_project in \
  "$NATIVE/nvdiffrast" "$NATIVE/nvdiffrec" "$NATIVE/CuMesh" \
  "$NATIVE/FlexGEMM" "$ROOT/o-voxel"; do
  uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps "$native_project"
done
uv pip install --python "$VENV/bin/python" --no-deps \
  "$NATIVE/utils3d" "$NATIVE/utils3d-moge" "$NATIVE/MoGe"

export PYTHONPATH="$ROOT/..:$ROOT:$ROOT/world:$NATIVE/NAF:$NATIVE/MoGe${PYTHONPATH:+:$PYTHONPATH}"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export SPARSE_CONV_BACKEND=flex_gemm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$VENV/bin/python" - <<'PY'
import torch
import torchvision
import natten
import cumesh
import flex_gemm
import nvdiffrast.torch
import nvdiffrec_render
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from moge.model.v2 import MoGeModel
assert torch.cuda.is_available(), "CUDA is not visible inside the sculpting environment"
properties = torch.cuda.get_device_properties(0)
assert properties.total_memory >= 30_000 * 1024**2
assert properties.major >= 12, f"Expected a Blackwell-class CUDA device; found capability {properties.major}.{properties.minor}"
print(f"Sculpting environment ready: torch={torch.__version__} torchvision={torchvision.__version__} natten={natten.__version__} GPU={torch.cuda.get_device_name(0)}")
PY

printf '\nSculpting Studio is ready on %s (%s MiB).\nRun ./startwithuv.sh to open the private studio.\n' "$GPU_NAME" "$GPU_MIB"
