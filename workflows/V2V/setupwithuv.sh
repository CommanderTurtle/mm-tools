#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${V2V_VENV:-$ROOT/.venv}"
COMFY="$ROOT/../ComfyUI"
ANIMATE_NODES="$ROOT/../animate/ComfyUI-WanAnimatePreprocess"
NODE_LINK="$COMFY/custom_nodes/mmtools_wan_animate_preprocess"

command -v uv >/dev/null 2>&1 || { printf 'uv is required. Install uv, then rerun this script.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'The video studios require an NVIDIA CUDA workstation.\n' >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { printf 'ffmpeg is required for video preparation and packaging.\n' >&2; exit 1; }
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [[ -z "$GPU_NAME" || "${GPU_MIB:-0}" -lt 30000 ]]; then
  printf 'The canonical video profile requires at least 30,000 MiB VRAM; found %s (%s MiB).\n' "${GPU_NAME:-none}" "${GPU_MIB:-0}" >&2
  exit 1
fi

required=(
  "$COMFY/models/diffusion_models/wan_animate_2_distill_int8_convrot.safetensors"
  "$COMFY/models/diffusion_models/wan_2.1_idv2v_int8_convrot.safetensors"
  "$COMFY/models/diffusion_models/wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors"
  "$COMFY/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "$COMFY/models/clip_vision/clip_vision_h.safetensors"
  "$COMFY/models/vae/Wan2_1_VAE_bf16.safetensors"
  "$COMFY/models/detection/vitpose_h_wholebody_model.onnx"
  "$COMFY/models/detection/vitpose_h_wholebody_data.bin"
  "$COMFY/models/detection/yolov10m.onnx"
  "$ROOT/../../sculpting/pretrained/deps/ZhengPeng7--BiRefNet/model.safetensors"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { printf 'Missing allowlisted artifact: %s\nRun ../../models/download_models.py v2v and ../download_models.py animate first.\n' "$path" >&2; exit 1; }
done

export UV_LINK_MODE=hardlink
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12 --seed --managed-python
fi

# The wheel contains its own CUDA runtime; the host driver provides forward
# compatibility. No compiler is required; Comfy manages weight placement
# between the GPU and host RAM natively.
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
uv pip install --python "$VENV/bin/python" \
  'setuptools<81' wheel packaging \
  -r "$COMFY/requirements.txt" \
  -r "$ROOT/requirements-local.txt"

mkdir -p "$COMFY/custom_nodes" "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"
if [[ -e "$NODE_LINK" && ! -L "$NODE_LINK" ]]; then
  printf 'Refusing to replace unexpected path: %s\n' "$NODE_LINK" >&2
  exit 1
fi
ln -sfn -- "$ANIMATE_NODES" "$NODE_LINK"

export PYTHONPATH="$ROOT/../..:$ROOT:$COMFY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
"$VENV/bin/python" - <<'PY'
import onnxruntime as ort
import torch
from local_app.server import build_application

assert torch.cuda.is_available(), "CUDA is unavailable inside the video environment"
props = torch.cuda.get_device_properties(0)
assert props.total_memory >= 30_000 * 1024**2
assert props.major >= 12, f"Expected Blackwell capability; found {props.major}.{props.minor}"
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
print(f"Video runtime ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)} ONNX=CUDA")
PY

printf '\nVideo runtime is ready in %s on %s (%s MiB).\n' "$VENV" "$GPU_NAME" "$GPU_MIB"
