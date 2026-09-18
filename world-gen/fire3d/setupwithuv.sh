#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
command -v uv >/dev/null 2>&1 || { printf 'uv is required.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'Fire3D requires an NVIDIA CUDA workstation.\n' >&2; exit 1; }
command -v nvcc >/dev/null 2>&1 || { printf 'The CUDA toolkit (nvcc) is required for Fire3D native extensions.\n' >&2; exit 1; }
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
[[ "${GPU_MIB:-0}" -ge 30000 ]] || { printf 'The local staged profile requires at least 30,000 MiB VRAM.\n' >&2; exit 1; }

required=(
  "$ROOT/checkpoints/Fire3D/perception/model.pt"
  "$ROOT/checkpoints/Fire3D/reconstruction/flows/ss/model.pt"
  "$ROOT/checkpoints/Fire3D/reconstruction/flows/shape/model.pt"
  "$ROOT/checkpoints/Fire3D/reconstruction/flows/pbr/model.pt"
  "$ROOT/checkpoints/Fire3D/external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  "$ROOT/checkpoints/Fire3D/external/anyup_multi_backbone.pth"
  "$ROOT/checkpoints/Fire3D/external/trellis2/shape_dec_next_dc_f16c32_fp16.safetensors"
  "$ROOT/checkpoints/Fire3D/external/trellis2/tex_dec_next_dc_f16c32_fp16.safetensors"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { printf 'Missing allowlisted Fire3D artifact: %s\n' "$path" >&2; exit 1; }
done

export UV_LINK_MODE=hardlink HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
[[ -x "$VENV/bin/python" ]] || uv venv "$VENV" --python 3.10 --seed --managed-python
uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
uv pip install --python "$VENV/bin/python" \
  'setuptools<81' wheel packaging ninja cmake -e "$ROOT" -r "$ROOT/requirements-local.txt"
uv pip install --python "$VENV/bin/python" spconv-cu118==2.3.8 flash-attn==2.7.3 --no-build-isolation
uv pip install --python "$VENV/bin/python" --no-build-isolation \
  "$ROOT/native/nvdiffrast" \
  "$ROOT/native/pytorch3d" \
  "$ROOT/native/utils3d" \
  "$ROOT/native/FlexGEMM"

uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps "$ROOT/trellis2_x2/CuMesh"
uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps "$ROOT/trellis2_x2/o-voxel"
if [[ "${FIRE3D_WITH_BLENDER:-0}" == "1" ]]; then "$ROOT/scripts/install_blender.sh"; fi

mkdir -p "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"
PYTHONPATH="$ROOT/../..:$ROOT:$ROOT/trellis2_x2" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$VENV/bin/python" - <<'PY'
import torch
from fire3d.cli import examples
from local_app.server import build_application
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 30_000 * 1024**2
assert "single_image" in examples()
print(f"Fire3D studio runtime ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY
