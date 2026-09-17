#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export UV_LINK_MODE=hardlink
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
command -v uv >/dev/null || { echo "uv is required." >&2; exit 1; }
uv venv --python 3.11 --seed .venv
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -r requirements.txt -r ../studio/requirements.txt
PYTHONPATH="$ROOT/.." "$PY" - <<'PY'
from pathlib import Path

from fdanyone.assets import BIREFNET_FILES, MODEL_FILES
from fdanyone.download import create_classic_gvhmr_links
import torch

root = Path.cwd()
missing = [str(root / "models" / item) for item in MODEL_FILES if not (root / "models" / item).is_file()]
missing.extend(
    str(root / "models" / "birefnet" / item)
    for item in BIREFNET_FILES
    if not (root / "models" / "birefnet" / item).is_file()
)
if missing:
    raise SystemExit("Missing model artifacts. Run ../models/download_models.py for 4d first:\n" + "\n".join(missing))

create_classic_gvhmr_links("models", "third_party/GVHMR", require_smplx=False)
assert torch.cuda.is_available(), "CUDA is required"
major, _ = torch.cuda.get_device_capability()
assert major >= 12, "The local RTX 5090 CUDA build is not active"
print(f"4DAnyone environment ready: torch={torch.__version__} GPU={torch.cuda.get_device_name(0)}")
PY
echo "4DAnyone Capture Studio is ready. Install SMPL-X from its setup mode before the first capture."
