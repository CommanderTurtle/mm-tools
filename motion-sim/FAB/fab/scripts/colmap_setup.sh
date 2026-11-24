#!/bin/bash
# COLMAP 3.13 with CUDA (conda-forge build) into $FAB_ENVS/colmap, for bench/scene_recon/mvs_depth.py (PatchMatch stereo).
# Uses micromamba under $FAB_ENVS/mm. Idempotent. Set FAB_COLMAP to use a COLMAP you installed yourself instead.
set -e
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}; FAB_ENVS=${FAB_ENVS:-$FAB_HOME/envs}
MM=$FAB_ENVS/mm; mkdir -p $FAB_ENVS
if [ ! -x $MM/bin/micromamba ]; then
  mkdir -p $MM/bin && curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $MM bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$MM
LOG=$FAB_ENVS/colmap_install.log
[ -x $FAB_ENVS/colmap/bin/colmap ] || $MM/bin/micromamba create -y -q -p $FAB_ENVS/colmap -c conda-forge "colmap=*=*cuda*" > $LOG 2>&1 || tail -20 $LOG
$FAB_ENVS/colmap/bin/colmap -h 2>&1 | head -3
$FAB_ENVS/colmap/bin/colmap help 2>&1 | grep -i "cuda\|with" | head -3
