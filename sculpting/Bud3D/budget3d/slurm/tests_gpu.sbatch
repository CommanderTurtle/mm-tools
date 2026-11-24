#!/usr/bin/env bash
# The full test suite (tests/run_gpu.sh) on one GPU, about 20 minutes.
#   CKPT=weights/budget3d_idol.safetensors DOME_ROOT=data/dome_objaverse sbatch slurm/tests_gpu.sbatch
#SBATCH --job-name=budget3d-tests
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/budget3d-tests-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
bash tests/run_gpu.sh
