#!/bin/bash
# Environments, checkouts and weights of the segmentation baselines (bench/seg/baselines.py, scripts/seg_compare.py), at
# the versions used for the paper. Nothing is vendored: each project is fetched from its own source and used under its
# own licence (SAM 2.1: Apache-2.0; SAM 3: SAM License, gated on Hugging Face -- accept it on huggingface.co/facebook/sam3
# and `hf auth login` first; detectron2: Apache-2.0; VISOR-HOS and Hands23: see their repositories).
#   THIRDPARTY=thirdparty ENVS=envs bash scripts/setup_seg_baselines.sh      (needs uv, git, wget, a CUDA 12.8 toolkit + driver)
set -e
T=${THIRDPARTY:-thirdparty}; V=${ENVS:-envs}; mkdir -p $T $V
PIP="uv pip install --no-config -q"
# 1) SAM 2.1 + SAM 3 (transformers): one env, Python 3.12
if [ ! -x $V/seg/bin/python ]; then
  uv venv $V/seg --python 3.12 --python-preference only-managed -q
  $PIP --python $V/seg/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  $PIP --python $V/seg/bin/python "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" \
    transformers==5.21.0 accelerate==1.15.0 huggingface-hub opencv-python-headless numpy
fi
$V/seg/bin/python -c "import sam2, transformers, torch; print('seg env', transformers.__version__, torch.__version__)"
# 2) VISOR-HOS (PointRend R-50) and Hands23 (Mask R-CNN X-101): detectron2 built from source, Python 3.10
fetch() {  # repo url, commit, dir
  [ -d "$3/.git" ] || git clone -q "$1" "$3"
  git -C "$3" checkout -q "$2"
}
fetch https://github.com/epic-kitchens/VISOR-HOS.git 8566507382add7dd037a83e7233950e0ad1ea78e $T/VISOR-HOS
fetch https://github.com/EvaCheng-cty/hands23_detector.git 1bc0f919ffa7a9f375e7e8042e1d26f3743d5819 $T/hands23_detector
fetch https://github.com/facebookresearch/detectron2.git 247691a1c292654867deb35cabb27f18d75a1363 $T/detectron2
get() {  # url, file, sha256
  [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; wget -q -O "$2" "$1" || wget -q --no-check-certificate -O "$2" "$1"; }
  echo "$3  $2" | sha256sum -c --quiet
}
get "https://www.dropbox.com/s/bfu94fpft2wi5sn/model_final_hos.pth?dl=1" $T/VISOR-HOS/checkpoints/model_final_hos.pth \
  174a4fd55629e3bc08c64eecc67ecef0214d3b66aec2560582d18e4d55200935
get https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth \
  $T/hands23_detector/model_weights/model_hands23.pth b3d2ff966d8a19b991a3ce170e2d8f86040f614ef2edf4a55423288b278f723c
if ! $V/d2/bin/python -c "from detectron2.projects import point_rend" 2>/dev/null; then
  [ -x $V/d2/bin/python ] || uv venv $V/d2 --python 3.10 --python-preference only-managed -q  # managed CPython ships Python.h
  $PIP --python $V/d2/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  $PIP --python $V/d2/bin/python opencv-python-headless "pillow==9.5.0" numpy tqdm parso ninja setuptools wheel fvcore iopath pycocotools
  D2PY=$(cd $V/d2/bin && pwd)/python
  (cd $T/detectron2 && MAX_JOBS=${MAX_JOBS:-16} $PIP --python $D2PY --no-build-isolation -e .)  # needs nvcc (CUDA 12.8 toolkit)
fi
$V/d2/bin/python -c "import detectron2, torch; from detectron2.projects import point_rend; print('d2 env', detectron2.__version__, torch.__version__)"
# 3) model files on first use: SAM 2.1 hiera-B+ / -L and SAM 3 from the Hugging Face hub (cached in $HF_HOME)
echo "run: $V/seg/bin/python scripts/seg_compare.py --arms sam2b_ours,sam2l_ours,sam3_ours,sam3_text ...;" \
  "$V/d2/bin/python scripts/seg_compare.py --arms hos,hands23 --thirdparty $T ..."
