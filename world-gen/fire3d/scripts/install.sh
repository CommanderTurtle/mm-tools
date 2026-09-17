#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${FIRE3D_ENV_NAME:-fire3d}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Miniforge first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
fi
# CUDA's Conda activation hook appends to these variables without guarding
# against `set -u` on all supported package revisions.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$ROOT[dev]"
python -m pip install spconv-cu118==2.3.8 flash-attn==2.7.3 --no-build-isolation
python -m pip install --no-build-isolation \
  'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae' \
  'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' \
  'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8' \
  'git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e'

DINO_DIR="$ROOT/third_party/dinov3"
if [[ ! -d "$DINO_DIR/.git" ]]; then
  git clone https://github.com/facebookresearch/dinov3.git "$DINO_DIR"
fi
git -C "$DINO_DIR" fetch origin 31703e4cbf1ccb7c4a72daa1350405f86754b6d1
git -C "$DINO_DIR" checkout --detach 31703e4cbf1ccb7c4a72daa1350405f86754b6d1

for eigen_dir in \
  "$ROOT/trellis2_x2/CuMesh/third_party/cubvh/third_party/eigen" \
  "$ROOT/trellis2_x2/o-voxel/third_party/eigen"; do
  if [[ ! -d "$eigen_dir/Eigen" ]]; then
    rm -rf "$eigen_dir"
    git clone --depth 1 --branch 3.4.0 \
      https://gitlab.com/libeigen/eigen.git "$eigen_dir"
  fi
done

python -m pip install --no-build-isolation --no-deps "$ROOT/trellis2_x2/CuMesh"
python -m pip install --no-build-isolation --no-deps "$ROOT/trellis2_x2/o-voxel"

echo "Fire3D environment ready. Run: conda activate $ENV_NAME"
