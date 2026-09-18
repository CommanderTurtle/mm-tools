#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCULPT_ROOT="$(cd -- "$ROOT/.." && pwd)"
VENV="$ROOT/.venv"

command -v uv >/dev/null 2>&1 || { printf 'uv is required. Install uv, then rerun this script.\n' >&2; exit 1; }
SCULPTING_VENV="$VENV" "$SCULPT_ROOT/setupwithuv.sh"

required=(
  "$ROOT/pretrained/Pixal3D/pipeline.json"
  "$ROOT/pretrained/ss_ft64_mv_lora_ibr_texverse/config.json"
  "$ROOT/pretrained/ss_ft64_mv_lora_ibr_texverse/ckpts/denoiser_step0015000.pt"
  "$ROOT/pretrained/ss_ft64_mv_lora_ibr_texverse/ckpts/mv_aggregator_step0015000.pt"
  "$ROOT/pretrained/shape_ft1024_mv_lora_ibr_texverse_fixedmem05/config.json"
  "$ROOT/pretrained/shape_ft1024_mv_lora_ibr_texverse_fixedmem05/ckpts/denoiser_step0015000.pt"
  "$ROOT/pretrained/shape_ft1024_mv_lora_ibr_texverse_fixedmem05/ckpts/mv_aggregator_step0015000.pt"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { printf 'Missing WorldSculpt artifact: %s\nRun ../../models/download_models.py worldsculpt first.\n' "$path" >&2; exit 1; }
done

export UV_LINK_MODE=hardlink
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements-local.txt"

export PYTHONPATH="$SCULPT_ROOT/..:$SCULPT_ROOT:$ROOT:$SCULPT_ROOT/native/NAF:$SCULPT_ROOT/native/MoGe${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export SPARSE_CONV_BACKEND=flex_gemm
export PIXAL_NAF_SOURCE="$SCULPT_ROOT/native/NAF"
export PIXAL_NAF_CHECKPOINT="$SCULPT_ROOT/pretrained/deps/valeoai--NAF/naf_release.pth"

"$VENV/bin/python" - <<'PY'
import torch
import diffusers
import peft
import skimage
import sklearn
import fpsample
import iopath
import pycocotools
import ftfy
from pixal3d_multiview.mv_aggregator import IBRMVAggregator
assert torch.cuda.is_available(), "CUDA is unavailable"
properties = torch.cuda.get_device_properties(0)
assert properties.total_memory >= 30_000 * 1024**2
assert properties.major >= 12, f"Expected Blackwell capability 12.x, found {properties.major}.{properties.minor}"
print(f"WorldSculpt runtime ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY

printf '\nWorldSculpt Studio is ready. Run ./startwithuv.sh.\n'
