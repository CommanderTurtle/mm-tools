#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
SHEET_VENV="$ROOT/.venv-sheetsage2"
STUDIO="$ROOT/local_app"

command -v uv >/dev/null 2>&1 || { printf 'uv is required. Install uv, then rerun this script.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'YuE2 requires an NVIDIA CUDA workstation.\n' >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { printf 'SheetSage2 requires FFmpeg and its shared libraries.\n' >&2; exit 1; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
if [[ -z "$GPU_NAME" || "${GPU_MIB:-0}" -lt 24000 ]]; then
  printf 'The canonical YuE2 profile requires a BF16 CUDA GPU with at least 24,000 MiB VRAM; found %s (%s MiB).\n' "${GPU_NAME:-none}" "${GPU_MIB:-0}" >&2
  exit 1
fi

required=(
  "$ROOT/models/YuE2-3B/model.safetensors"
  "$ROOT/models/YuE2-3B/qwen.tiktoken"
  "$ROOT/models/YuE2-Vae/model.safetensors"
  "$ROOT/models/SheetSage2/model.safetensors"
  "$ROOT/models/MERT-v2-FullSong/model.safetensors"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { printf 'Missing model artifact: %s\nRun ../../models/download_models.py yue2 first.\n' "$path" >&2; exit 1; }
done

# uv's shared cache and hardlinks deduplicate wheels between isolated project
# environments without weakening dependency boundaries.
export UV_LINK_MODE=hardlink
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12 --seed --managed-python
fi
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0'
uv pip install --python "$VENV/bin/python" \
  'transformers==4.57.6' 'huggingface-hub==0.36.2' 'safetensors==0.7.0' \
  'tiktoken==0.12.0' 'numpy==2.2.6' 'soundfile==0.13.1' 'accelerate==1.13.0' \
  -r "$STUDIO/requirements.txt"
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT"

if [[ ! -x "$SHEET_VENV/bin/python" ]]; then
  uv venv "$SHEET_VENV" --python 3.11 --seed --managed-python
fi
uv pip install --python "$SHEET_VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0' 'torchaudio==2.8.0'
uv pip install --python "$SHEET_VENV/bin/python" -r "$ROOT/models/SheetSage2/requirements.txt"

mkdir -p "$ROOT/.runtime/studio/assets" "$ROOT/.runtime/studio/outputs"

PYTHONPATH="$ROOT/../..:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$VENV/bin/python" - <<'PY'
import torch
from yue2 import YuE2Pipeline, SymbolicPlan
from local_app.server import build_application

assert torch.cuda.is_available(), "CUDA is not visible inside the YuE2 environment"
assert torch.cuda.is_bf16_supported(), "YuE2 requires BF16 support"
assert torch.cuda.get_device_properties(0).total_memory >= 24_000 * 1024**2
print(f"YuE2 environment ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY

"$SHEET_VENV/bin/python" "$ROOT/local_app/transcribe.py" --help >/dev/null
printf '\nYuE2 Composition Studio is ready on %s (%s MiB).\nRun ./startwithuv.sh to open the private studio.\n' "$GPU_NAME" "$GPU_MIB"
