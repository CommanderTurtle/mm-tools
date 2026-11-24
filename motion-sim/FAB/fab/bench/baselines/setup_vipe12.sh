#!/bin/bash
# ViPE 1.2 (https://github.com/nv-tlabs/vipe; MoGe-2 / DA3 depth priors: keyframe_depth=moge2-l,
# depth_align_model=adaptive_moge2-l_svda), the version that runs the camera + depth step of world mode, next to the
# fdc7935 env of bench/baselines/setup_vipe.sh: checkout $FAB_ENVS/src/vipe12 (its commit in .pinned_sha), env
# $FAB_ENVS/vipe12, its CUDA extension built in a scratch copy of the checkout (Eigen 3.4). Idempotent.
# Use with bench/world_vipe.py --python $FAB_ENVS/vipe12/bin/python --vipe $FAB_ENVS/src/vipe12.
set -ex
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}; FAB_ENVS=${FAB_ENVS:-$FAB_HOME/envs}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FAB_ENVS/uv-cache} PATH=$HOME/.local/bin:$PATH
mkdir -p $FAB_ENVS/src
V=${VIPE12_DIR:-$FAB_ENVS/src/vipe12}
E=$FAB_ENVS/src/eigen-3.4.0
PY=${FAB_VIPE12_PY:-$FAB_ENVS/vipe12/bin/python}
VENV=$(dirname $(dirname $PY))
# the ViPE 1.2 commit the paper's world rows used (upstream nv-tlabs/vipe, "Fix OOMs in long sequences and optimize
# performance (#104)"); VIPE12_SHA overrides it
VIPE12_SHA=${VIPE12_SHA:-8c9f36144e08d8f8c8cf60d20ad70c100d9cff07}
if [ ! -d $V/.git ]; then
  git clone https://github.com/nv-tlabs/vipe.git $V
  [ -n "$VIPE12_SHA" ] && git -C $V checkout $VIPE12_SHA
  git -C $V rev-parse HEAD > $V/.pinned_sha
fi
cat $V/.pinned_sha
[ -f $E/Eigen/Core ] || curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz -C $FAB_ENVS/src
mkdir -p $FAB_ENVS/src/eigen_inc && ln -sfn $E $FAB_ENVS/src/eigen_inc/eigen3
if [ ! -f $VENV/.installed ]; then
  PV=$(cat $V/.python-version 2>/dev/null || echo 3.12)  # pyproject: python >= 3.11
  uv venv --allow-existing --python $PV --python-preference only-managed $VENV
  uv pip install --no-config --python $PY torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install --no-config --python $PY "setuptools>=77" wheel ninja imageio imageio-ffmpeg
  # its exact gtsam-develop pin may be gone from PyPI (nightlies rotate): override to a kept nightly
  OV=$(mktemp); echo "gtsam-develop==${VIPE12_GTSAM:-4.3.2.dev202610301701}" > $OV
  # the extension is built in a scratch copy (VIPE_BUILD_LOCAL=1, default; 0 = in place in $V), so a checkout shared
  # by several machines is only read
  BD=$V; [ "${VIPE_BUILD_LOCAL:-1}" = 1 ] && { BD=$(mktemp -d); (cd $V && tar --exclude=./build --exclude='*.so' -cf - .) | tar -C $BD -xf -; }
  (cd $BD && CPATH=$FAB_ENVS/src/eigen_inc:$E TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6;8.9"} MAX_JOBS=${MAX_JOBS:-16} \
    uv pip install --no-config --python $PY --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match --prerelease allow --override $OV)
  touch $VENV/.installed
fi
$PY -c "import torch, vipe, vipe_ext; print('vipe12 ok', torch.__version__)"
