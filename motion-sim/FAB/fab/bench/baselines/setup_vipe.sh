#!/bin/bash
# NVIDIA ViPE (https://github.com/nv-tlabs/vipe @ fdc7935; video pose engine: dynamic-object masking, metric depth
# priors, bundle adjustment): the env that drives the ViPE runs and GeoCalib (bench/world_vipe.py, bench/world_geocalib.py).
# Checkout $FAB_ENVS/src/vipe, env $FAB_ENVS/vipe (python 3.10 + its pinned requirements), its CUDA extension built in
# a scratch copy of the checkout (Eigen 3.4 downloaded to $FAB_ENVS/src/eigen-3.4.0). ViPE downloads its own model weights
# on first use (their licences: docs/world.md). Idempotent.
set -ex
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}; FAB_ENVS=${FAB_ENVS:-$FAB_HOME/envs}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FAB_ENVS/uv-cache} PATH=$HOME/.local/bin:$PATH
mkdir -p $FAB_ENVS/src
V=${VIPE_DIR:-$FAB_ENVS/src/vipe}
E=$FAB_ENVS/src/eigen-3.4.0
PY=${FAB_VIPE_PY:-$FAB_ENVS/vipe/bin/python}
VENV=$(dirname $(dirname $PY))
if [ ! -d $V/.git ]; then
  git clone https://github.com/nv-tlabs/vipe.git $V
  git -C $V checkout fdc79359ec6d063c86b9d203ca3d7054a9636823
fi
[ -f $E/Eigen/Core ] || curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz -C $FAB_ENVS/src
mkdir -p $FAB_ENVS/src/eigen_inc && ln -sfn $E $FAB_ENVS/src/eigen_inc/eigen3
if [ ! -f $VENV/.installed ]; then
  uv venv --allow-existing --python 3.10 --python-preference only-managed $VENV
  uv pip install --no-config --python $PY -r $V/envs/requirements.txt --index-strategy unsafe-best-match
  uv pip install --no-config --python $PY setuptools wheel ninja imageio imageio-ffmpeg
  # the extension is built in a scratch copy (VIPE_BUILD_LOCAL=1, default; 0 = in place in $V), so a checkout shared
  # by several machines is only read
  BD=$V; [ "${VIPE_BUILD_LOCAL:-1}" = 1 ] && { BD=$(mktemp -d); (cd $V && tar --exclude=./build --exclude='*.so' -cf - .) | tar -C $BD -xf -; }
  (cd $BD && CPATH=$FAB_ENVS/src/eigen_inc:$E TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6;8.9"} MAX_JOBS=${MAX_JOBS:-16} \
    uv pip install --no-config --python $PY --no-build-isolation -e .)
  touch $VENV/.installed
fi
$PY -c "import torch, vipe, vipe_ext; print('vipe ok', torch.__version__)"
