#!/bin/bash
# Envs for the world-mode reconstruction steps (idempotent): ViPE (bench/baselines/setup_vipe.sh), and open3d / h5py /
# mujoco in the FAB env ($FAB_PY, default: the python on PATH) for TSDF scene fusion / raycasting.
set -e
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}; FAB_ENVS=${FAB_ENVS:-$FAB_HOME/envs}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FAB_ENVS/uv-cache} PATH=$HOME/.local/bin:$PATH
FAB_PY=${FAB_PY:-$(command -v python)}; FAB_VIPE_PY=${FAB_VIPE_PY:-$FAB_ENVS/vipe/bin/python}
mkdir -p $FAB_ENVS; LOG=$FAB_ENVS/setup_vipe.log
bash bench/baselines/setup_vipe.sh > $LOG 2>&1 || { tail -n 40 $LOG; exit 1; }
$FAB_PY -c "import open3d, h5py, mujoco" 2>/dev/null || \
  uv pip install --no-config --python $FAB_PY -q open3d h5py mujoco
$FAB_PY -c "import open3d; print('open3d', open3d.__version__)"
$FAB_VIPE_PY -c "import OpenEXR; print('exr ok')"
