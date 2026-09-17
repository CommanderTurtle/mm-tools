#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
VENDOR="$ROOT/.runtime/vendor"
STUDIO="$ROOT/../studio"

command -v uv >/dev/null 2>&1 || { printf 'uv is required. Install uv, then rerun this script.\n' >&2; exit 1; }
command -v git >/dev/null 2>&1 || { printf 'git is required to materialize pinned CUDA sources.\n' >&2; exit 1; }
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

mkdir -p "$VENDOR" "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"

checkout_source() {
  local name="$1" url="$2" revision="$3" recursive="${4:-no}"
  local destination="$VENDOR/$name" stage="$VENDOR/.$name.stage"
  if [[ -f "$destination/.source-revision" ]] && [[ "$(<"$destination/.source-revision")" == "$revision" ]]; then
    printf 'Pinned source ready: %s @ %s\n' "$name" "${revision:0:12}"
    return
  fi
  rm -rf -- "$stage"
  git init -q "$stage"
  git -C "$stage" remote add origin "$url"
  git -C "$stage" fetch -q --depth 1 origin "$revision"
  git -C "$stage" checkout -q --detach FETCH_HEAD
  if [[ "$recursive" == yes ]]; then
    git -C "$stage" submodule update --init --recursive --depth 1
  fi
  rm -rf -- "$stage/.git"
  printf '%s\n' "$revision" > "$stage/.source-revision"
  rm -rf -- "$destination"
  mv -- "$stage" "$destination"
  printf 'Pinned source installed: %s @ %s\n' "$name" "${revision:0:12}"
}

checkout_source CuMesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f yes
checkout_source FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e yes
checkout_source nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae no
checkout_source nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git b296927cc7fd01c2ac1087c8065c4d7248f72da4 no
checkout_source NAF https://github.com/valeoai/NAF.git 37f2dfc180f2de53d98bd601109c0da0dd6b0f43 no
checkout_source MoGe https://github.com/microsoft/MoGe.git 74fbce054ebed49800de42d0ad0e83495065719a no
checkout_source utils3d https://github.com/EasternJournalist/utils3d.git 9a4eb15e4021b67b12c460c7057d642626897ec8 no
# MoGe publishes its compatibility fork separately from utils3d. Pin the real
# upstream commit (the previous value transposed the commit after `62f09d`, so
# a clean setup could never fetch it).
checkout_source utils3d-moge https://github.com/EasternJournalist/utils3d-moge.git 62f09d58509485564e24d5d9f6aac9ee9ebc0c37 no

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

# Build every native component against the already-installed Blackwell PyTorch
# stack. --no-build-isolation prevents an extension build from silently pulling
# a second torch/CUDA combination into a temporary environment.
# All runtime requirements are resolved above. Installing the native projects
# without dependency resolution is intentional: o-voxel's upstream metadata
# names floating Git URLs for CuMesh/FlexGEMM, while this monorepo has already
# materialized and pinned those exact native sources locally.
for native_project in \
  "$VENDOR/nvdiffrast" "$VENDOR/nvdiffrec" "$VENDOR/CuMesh" \
  "$VENDOR/FlexGEMM" "$ROOT/o-voxel"; do
  uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps "$native_project"
done
uv pip install --python "$VENV/bin/python" --no-deps \
  "$VENDOR/utils3d" "$VENDOR/utils3d-moge" "$VENDOR/MoGe"

export PYTHONPATH="$ROOT/..:$ROOT:$ROOT/world:$VENDOR/NAF:$VENDOR/MoGe${PYTHONPATH:+:$PYTHONPATH}"
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
from studio.server import build_application

assert torch.cuda.is_available(), "CUDA is not visible inside the sculpting environment"
properties = torch.cuda.get_device_properties(0)
assert properties.total_memory >= 30_000 * 1024**2
assert properties.major >= 12, f"Expected a Blackwell-class CUDA device; found capability {properties.major}.{properties.minor}"
print(f"Sculpting environment ready: torch={torch.__version__} torchvision={torchvision.__version__} natten={natten.__version__} GPU={torch.cuda.get_device_name(0)}")
PY

printf '\nSculpting Studio is ready on %s (%s MiB).\nRun ./startwithuv.sh to open the private studio.\n' "$GPU_NAME" "$GPU_MIB"
