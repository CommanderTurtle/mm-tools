#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
STUDIO="$ROOT/local_app"

command -v uv >/dev/null 2>&1 || { printf 'uv is required. Install uv, then rerun this script.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'AuK requires an NVIDIA CUDA workstation.\n' >&2; exit 1; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [[ -z "$GPU_NAME" || "${GPU_MIB:-0}" -lt 30000 ]]; then
  printf 'The canonical AuK profile requires a CUDA GPU with at least 30,000 MiB VRAM; found %s (%s MiB).\n' "${GPU_NAME:-none}" "${GPU_MIB:-0}" >&2
  exit 1
fi

required=(
  "$ROOT/ckpts/AuK/auk_base.safetensors"
  "$ROOT/ckpts/AuK/vae.safetensors"
  "$ROOT/ckpts/AuK/config.yaml"
  "$ROOT/ckpts/Qwen2.5-Omni-3B/config.json"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { printf 'Missing model artifact: %s\nRun ../models/download_models.py aukspeech first.\n' "$path" >&2; exit 1; }
done

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12 --seed --managed-python
fi

uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1' 'torchvision==0.22.1' 'torchaudio==2.7.1'

uv pip install --python "$VENV/bin/python" \
  -e "$ROOT" \
  -r "$STUDIO/requirements.txt" \
  'PyYAML>=6,<7' 'openai>=1,<3' 'tencentcloud-sdk-python-asr>=3,<4' \
  'silero-vad>=6,<7' 'WeTextProcessing>=1.2,<2' 'pyloudnorm>=0.2,<1'

mkdir -p "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"

PYTHONPATH="$ROOT/..:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$VENV/bin/python" - <<'PY'
import torch
import torchaudio
from auk.infer.infer_auk import AukInfer
from local_app.server import build_application

assert torch.cuda.is_available(), "CUDA is not visible inside the AuK environment"
assert torch.cuda.get_device_properties(0).total_memory >= 30_000 * 1024**2
print(f"AuK environment ready: {torch.__version__=} {torchaudio.__version__=} GPU={torch.cuda.get_device_name(0)}")
PY

printf '\nAuK Speech Studio is ready on %s (%s MiB).\nRun ./startwithuv.sh for the WebUI or ./starthttp.sh for API-only mode.\n' "$GPU_NAME" "$GPU_MIB"
