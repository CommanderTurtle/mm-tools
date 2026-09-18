#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
command -v uv >/dev/null 2>&1 || { printf 'uv is required.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'An NVIDIA CUDA workstation is required.\n' >&2; exit 1; }
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
[[ "${GPU_MIB:-0}" -ge 30000 ]] || { printf 'The canonical profile needs at least 30,000 MiB VRAM.\n' >&2; exit 1; }
required=(
  "$ROOT/text-encoders/voxta/Llama-3-8B-LLM2Vec-ARDY-INT8/model.safetensors"
  "$ROOT/SOMA-X/assets/SOMA_neutral.npz" "$ROOT/SOMA-X/assets/SOMA_template_rig.usda"
  "$ROOT/SOMA-X/assets/SOMAHand.npz" "$ROOT/SOMA-X/assets/correctives_model.pt"
)
for model in ARDY-Core-RP-20FPS-Horizon40 ARDY-Core-RP-20FPS-Horizon8 ARDY-G1-RP-25FPS-Horizon52 ARDY-G1-RP-25FPS-Horizon8; do required+=("$ROOT/checkpoints/$model/denoiser.safetensors" "$ROOT/checkpoints/$model/tokenizer.safetensors" "$ROOT/checkpoints/$model/config.yaml"); done
for path in "${required[@]}"; do [[ -s "$path" ]] || { printf 'Missing allowlisted artifact: %s\n' "$path" >&2; exit 1; }; done
export UV_LINK_MODE=hardlink HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
[[ -x "$VENV/bin/python" ]] || uv venv "$VENV" --python 3.12 --seed --managed-python
uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu130 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
uv pip install --python "$VENV/bin/python" 'setuptools<81' wheel packaging -e "$ROOT" -e "$ROOT/SOMA-X" -r "$ROOT/requirements-local.txt" -r "$ROOT/local_app/requirements.txt"
mkdir -p "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"
export PYTHONPATH="$ROOT/..:$ROOT:$ROOT/SOMA-X${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
"$VENV/bin/python" - <<'PY'
import torch
from local_app.server import build_application
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
assert p.total_memory >= 30_000 * 1024**2
assert p.major >= 12
print(f"NVIDIA studio runtime ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY
