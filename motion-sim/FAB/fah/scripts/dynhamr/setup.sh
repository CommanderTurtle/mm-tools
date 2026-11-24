#!/bin/bash
# Dyn-HaMR (CVPR 2025, https://github.com/ZhengdiYu/Dyn-HaMR) for the supplement's world-space check: fetches its code at
# a pinned commit into $THIRDPARTY/Dyn-HaMR (not vendored; MIT License, its README limits code and model to non-commercial
# research use) and builds its own environment (python 3.10, torch 1.13 cu117 as its scripts/install_pip.sh), with
#   - HaMeR's demo data (checkpoint, ViTPose weights, MANO mean parameters) from a HaMeR checkout set up as in its README
#     ($THIRDPARTY/hamer, fetch_demo_data.sh) and WiLoR's hand detector ($THIRDPARTY/WiLoR/pretrained_models/detector.pt),
#     which its HaMeR fork uses for detection;
#   - MANO_RIGHT.pkl / MANO_LEFT.pkl from $MANO_DIR (MANO_NP1=1: re-pickle them for numpy 1, needed if they were saved
#     under numpy 2);
#   - its HMP motion prior (Google Drive link of its README) and the biomechanical-constraint statistics _DATA/BMC,
#     which it does not ship: computed with Hand-BMC-pytorch (MIT License) from that project's released 3D joints, as
#     its README says; MANO, HaMeR and the HMP weights stay under their own terms;
#   - an import stub for pytorch3d (only its renderer imports it; rendering stays off).
# Cameras come from bench/world_dynhamr.py (cameras.npz), so its DROID-SLAM / ViPE are not built. Needs uv and CUDA.
# Idempotent. Then scripts/dynhamr/camfix.sh $THIRDPARTY/Dyn-HaMR for the camera-offset fix, scripts/dynhamr/run.sh.
#   THIRDPARTY=<dir> MANO_DIR=<dir> [DYNHAMR_COMMIT=fa9cd74] [DYNHAMR_ENV=<dir>] bash scripts/dynhamr/setup.sh
set -ex
TP=${THIRDPARTY:-thirdparty}
TP=$(cd "$TP" && pwd)
D=${DYNHAMR_ROOT:-$TP/Dyn-HaMR}
HM=$TP/hamer
ENV=${DYNHAMR_ENV:-$TP/envs/dynhamr}
PY=$ENV/bin/python
COMMIT=${DYNHAMR_COMMIT:-fa9cd74}  # the version we evaluated (2025-11 update); paper-time release: setup_98e1622.sh
[ -f "$HM/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" ] || { echo "need a HaMeR checkout with its demo data in $HM"; exit 1; }
[ -f "$TP/WiLoR/pretrained_models/detector.pt" ] || { echo "need WiLoR's detector.pt in $TP/WiLoR/pretrained_models"; exit 1; }
[ -f "${MANO_DIR:?}/MANO_RIGHT.pkl" ] && [ -f "$MANO_DIR/MANO_LEFT.pkl" ]
if [ ! -d "$D/.git" ]; then
  git clone https://github.com/ZhengdiYu/Dyn-HaMR.git "$D"
  git -C "$D" checkout "$COMMIT"
fi
git -C "$D" log -1 --format="Dyn-HaMR %h %cd %s"
[ -f "$D/third-party/hamer/run.py" ] || git -C "$D" submodule update --init third-party/hamer

mkdir -p "$D/_DATA/data/mano" "$D/third-party/hamer/pretrained_models"
for x in hamer_ckpts vitpose_ckpts; do [ -e "$D/_DATA/$x" ] || ln -s "$HM/_DATA/$x" "$D/_DATA/$x"; done
cp "$HM/_DATA/data/mano_mean_params.npz" "$D/_DATA/data/"
cp "$TP/WiLoR/pretrained_models/detector.pt" "$D/third-party/hamer/pretrained_models/detector.pt"

if [ ! -f "$ENV/.installed" ]; then
  uv venv --allow-existing --python 3.10 --python-preference only-managed "$ENV"
  uv pip install --no-config --python $PY torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117
  uv pip install --no-config --python $PY "setuptools==59.5.0" pip wheel cython "numpy<1.24"
  uv pip install --no-config --python $PY --no-build-isolation "git+https://github.com/mattloper/chumpy" \
      "git+https://github.com/nghorbani/configer"
  # human_body_prior's current code needs python >= 3.11: pin the 2022 code Dyn-HaMR was written against
  uv pip install --no-config --python $PY --no-build-isolation --no-deps \
      "git+https://github.com/nghorbani/human_body_prior@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc" \
      "git+https://github.com/otaheri/MANO"  # optim/base_scene.py imports `mano` (unused)
  uv pip install --no-config --python $PY torchgeometry==0.1.2 tensorboard smplx==0.1.28 pyrender open3d imageio-ffmpeg matplotlib \
      opencv-python scipy scikit-image joblib tqdm hydra-core pyyaml gdown dill motmetrics einops timm==0.4.9 xtcocotools \
      pandas==1.4.0 ultralytics==8.1.34 pytorch-lightning==1.9.5 yacs rich webdataset trimesh loguru tensorboardX plyfile braceexpand "numpy<1.24" \
      "git+https://github.com/facebookresearch/hand_tracking_toolkit" torch==1.13.0 torchvision==0.14.0 "torchmetrics<0.12" \
      --extra-index-url https://download.pytorch.org/whl/cu117 --index-strategy unsafe-best-match  # keeps torch 1.13
  uv pip install --no-config --python $PY "setuptools>=64,<70"  # editable installs + pkg_resources (tensorboard, lightning 1.9)
  uv pip install --no-config --python $PY --no-build-isolation --no-deps -e "$D" -e "$D/third-party/hamer"
  touch "$ENV/.installed"
fi
if [ -n "$MANO_NP1" ]; then
  $PY -c "
import pickle, sys, numpy, scipy.sparse  # scipy first: its extensions must not see the alias
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
for s in ('RIGHT', 'LEFT'):
    d = pickle.load(open('$MANO_DIR/MANO_%s.pkl' % s, 'rb'), encoding='latin1')
    pickle.dump(d, open('$D/_DATA/data/mano/MANO_%s.pkl' % s, 'wb'), protocol=2)"
else
  cp "$MANO_DIR/MANO_RIGHT.pkl" "$MANO_DIR/MANO_LEFT.pkl" "$D/_DATA/data/mano/"
fi
mkdir -p "$D/shim/pytorch3d"
for m in renderer structures structures/meshes transforms; do mkdir -p "$D/shim/pytorch3d/$m"
  printf 'class _Stub:\n    def __init__(self, *a, **k):\n        raise RuntimeError("pytorch3d stub (scripts/dynhamr/setup.sh): rendering is disabled")\n\n\ndef __getattr__(name):\n    if name.startswith("__"):\n        raise AttributeError(name)\n    return _Stub\n' > "$D/shim/pytorch3d/$m/__init__.py"; done
touch "$D/shim/pytorch3d/__init__.py"
if [ ! -d "$D/_DATA/hmp_model" ]; then
  T=$(mktemp -d) && (cd $T && $ENV/bin/gdown "https://drive.google.com/uc?id=1_rp_SMxZA6Gjm6Z49M6hXam9kjMD6-GQ" -O hmp_model.zip \
    && unzip -q hmp_model.zip && mv hmp_model "$D/_DATA/")
fi
if [ ! -f "$D/_DATA/BMC/CONVEX_HULLS.npy" ]; then
  BM=$TP/Hand-BMC-pytorch
  [ -d "$BM/.git" ] || git clone --depth 1 https://github.com/MengHao666/Hand-BMC-pytorch "$BM"
  uv pip install --no-config --python $PY rdp
  [ -d "$BM/joints" ] || (cd "$BM" && $ENV/bin/gdown 1_wV8QjmsVCMBEBhm56gFA2XTyU8VEHzk -O joints.zip && unzip -q joints.zip)
  (cd "$BM" && MPLBACKEND=Agg $PY calculate_bmc.py && MPLBACKEND=Agg $PY -c "
import argparse, calculate_convex_hull as c  # its --visualize flag cannot be switched off (store_true, default True)
c.main(argparse.Namespace(path='BMC', visualize=False, epsilon=5e-4, ratio=0.9995, delta=0.0005))")
  mkdir -p "$D/_DATA/BMC" && cp "$BM"/BMC/*.npy "$D/_DATA/BMC/"
fi
cd "$D/dyn-hamr" && PYTHONPATH="$D/shim" $PY -c "
import torch, hamer, smplx, human_body_prior
from body_model import MANO
import run_opt
print('Dyn-HaMR imports ok', torch.__version__, torch.cuda.is_available())"
