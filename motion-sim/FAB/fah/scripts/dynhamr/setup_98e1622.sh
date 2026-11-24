#!/bin/bash
# Dyn-HaMR at its paper-time release (98e1622, 2025-08-01, before the 2025-11 update) in its own tree
# $THIRDPARTY/Dyn-HaMR-98e1622, with the HaMeR fork it pins (051dbaa: detectron2 ViTDet-H person boxes + ViTPose+-H
# whole-body 2D hand keypoints -> HaMeR) and that fork's ViTPose (d521645). Fetched, not vendored. Weights, BMC
# statistics, MANO and the pytorch3d stub are linked from the setup.sh tree (run it first). The optimisation runs in
# setup.sh's environment; the HaMeR / ViTPose / detectron2 step in its own (python 3.10, torch 2.7.0 cu128, detectron2
# built from source for TORCH_CUDA_ARCH_LIST, mmcv 1.3.9 without ops; pass its python as DYNHAMR_HAMER_PY to
# scripts/dynhamr/run.sh). CAMFIX=1 also builds the camera-offset fix tree (camfix.sh). Idempotent.
#   THIRDPARTY=<dir> [DYNHAMR_VIT_ENV=<dir>] [TORCH_CUDA_ARCH_LIST="8.6;8.9"] [CAMFIX=1] bash scripts/dynhamr/setup_98e1622.sh
set -ex
TP=${THIRDPARTY:-thirdparty}
TP=$(cd "$TP" && pwd)
D0=$TP/Dyn-HaMR
D=$TP/Dyn-HaMR-98e1622
ENV=${DYNHAMR_VIT_ENV:-$TP/envs/dynhamr_vit}
PY=$ENV/bin/python
[ -d "$D0/_DATA/BMC" ] || { echo "run scripts/dynhamr/setup.sh first"; exit 1; }
if [ ! -d "$D/.git" ]; then
  git clone https://github.com/ZhengdiYu/Dyn-HaMR.git "$D"
  git -C "$D" checkout 98e1622031
fi
git -C "$D" log -1 --format="Dyn-HaMR %h %cd %s"
[ -f "$D/third-party/hamer/run.py" ] || git -C "$D" submodule update --init third-party/hamer
[ -f "$D/third-party/hamer/third-party/ViTPose/setup.py" ] || \
  git -C "$D/third-party/hamer" submodule update --init third-party/ViTPose
git -C "$D/third-party/hamer" log -1 --format="hamer fork %h %cd %s"
mkdir -p "$D/_DATA"
for x in hamer_ckpts vitpose_ckpts data hmp_model BMC; do ln -sfn "$D0/_DATA/$x" "$D/_DATA/$x"; done
ln -sfn "$D0/shim" "$D/shim"
if [ ! -f "$ENV/.installed" ]; then
  uv venv --allow-existing --python 3.10 --python-preference only-managed "$ENV"
  uv pip install --no-config --python $PY torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install --no-config --python $PY "numpy<2" "setuptools>=64,<70" wheel ninja cython pip
  uv pip install --no-config --python $PY --no-build-isolation "git+https://github.com/mattloper/chumpy"
  MMCV_WITH_OPS=0 uv pip install --no-config --python $PY --no-build-isolation mmcv==1.3.9
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9}" MAX_JOBS=${MAX_JOBS:-8} uv pip install --no-config --python $PY \
    --no-build-isolation "git+https://github.com/facebookresearch/detectron2"
  uv pip install --no-config --python $PY gdown opencv-python pyrender pytorch-lightning scikit-image smplx==0.1.28 yacs \
    timm einops xtcocotools pandas hydra-core rich webdataset json_tricks munkres "mediapipe==0.10.14" matplotlib scipy tqdm \
    joblib "numpy<2" torch==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
  uv pip install --no-config --python $PY --no-build-isolation --no-deps -e "$D/third-party/hamer" \
    -e "$D/third-party/hamer/third-party/ViTPose"
  touch "$ENV/.installed"
fi
(cd "$D/third-party/hamer" && $PY -c "
import torch, detectron2, mmcv, mmpose, hamer, mediapipe
from vitpose_model import ViTPoseModel
print('HaMeR fork env ok', torch.__version__, detectron2.__version__, mmcv.__version__, mmpose.__version__)")
(cd "$D/dyn-hamr" && PYTHONPATH="$D/shim" "${DYNHAMR_ENV:-$TP/envs/dynhamr}/bin/python" -c "import run_opt; print('98e1622 run_opt imports ok')")
[ -z "$CAMFIX" ] || bash "$(dirname "$0")/camfix.sh" "$D"
