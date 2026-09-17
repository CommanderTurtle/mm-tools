#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
command -v uv >/dev/null 2>&1 || { printf 'uv is required.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'LingBot requires an NVIDIA CUDA workstation.\n' >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { printf 'FFmpeg is required for video packaging.\n' >&2; exit 1; }
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
[[ "${GPU_MIB:-0}" -ge 30000 ]] || { printf 'The single-5090 profile requires at least 30,000 MiB VRAM.\n' >&2; exit 1; }
DIT="$ROOT/lingbot-world-v2-1.3b-causal-fast"
ASSETS="$ROOT/lingbot-world-v2-14b-causal-fast"
required=("$DIT/model.safetensors.index.json" "$ASSETS/models_t5_umt5-xxl-enc-bf16.pth" "$ASSETS/Wan2.1_VAE.pth" "$ASSETS/google/umt5-xxl/tokenizer.json")
for path in "${required[@]}"; do [[ -s "$path" ]] || { printf 'Missing allowlisted artifact: %s\nRun ../../models/download_models.py lingbot lingbot-5090.\n' "$path" >&2; exit 1; }; done
export UV_LINK_MODE=hardlink HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
[[ -x "$VENV/bin/python" ]] || uv venv "$VENV" --python 3.12 --seed --managed-python
uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu130 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
uv pip install --python "$VENV/bin/python" 'setuptools<81' wheel packaging ninja -r "$ROOT/requirements.txt" -r "$ROOT/requirements-local.txt"
uv pip install --python "$VENV/bin/python" flash-attn --no-build-isolation
mkdir -p "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"
PYTHONPATH="$ROOT/../..:$ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" - <<'PY'
import torch
from studio.server import build_application
from wan.configs import WAN_CONFIGS
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 30_000 * 1024**2
assert "i2v-1.3B" in WAN_CONFIGS
print(f"LingBot single-GPU runtime ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY
