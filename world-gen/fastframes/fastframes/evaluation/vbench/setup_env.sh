#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Create the VBench scoring environment (Python 3.10, PyTorch 2.9.1 + CUDA 12.8).
#
#   bash setup_env.sh [VENV_DIR]        # default: ./.venv-vbench
#
# Scoring needs its own environment: VBench pins transformers==4.33.2 and
# needs detectron2 (built from source at the commit used for the paper) for
# the object_class, multiple_objects, color, and spatial_relationship metrics.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$HERE/.venv-vbench}"
PYTHON="${PYTHON:-python3.10}"
DETECTRON2_COMMIT=b4a4a3bd136852dae5fb1de37978dee412653e31

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# PyTorch (and its own dependencies) from the PyTorch index; everything in requirements.txt from PyPI.
"$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128
"$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"
"$VENV/bin/python" -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@$DETECTRON2_COMMIT"

"$VENV/bin/python" - <<'PY'
import importlib, torch
assert torch.__version__ == "2.9.1+cu128", torch.__version__
for name in ("clip", "decord", "pyiqa", "cv2", "transformers", "timm", "fairscale", "fvcore", "detectron2"):
    importlib.import_module(name)
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
PY
# Build the scorer package from the vbench 0.1.5 wheel (see docs/inference.md#scorer-provenance).
if [ -d "$HERE/VBench/vbench" ]; then
  "$VENV/bin/python" "$HERE/setup_scorer.py" --verify
else
  "$VENV/bin/python" "$HERE/setup_scorer.py"
fi
echo "export VBENCH_PYTHON=$VENV/bin/python"
