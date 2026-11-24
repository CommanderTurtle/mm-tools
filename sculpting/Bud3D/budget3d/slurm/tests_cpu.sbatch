#!/usr/bin/env bash
# The CPU tests (tests/run_cpu.sh), including loading and sampling released weights when CKPT / TURBO_CKPT are set.
#   CKPT=weights/budget3d_idol.safetensors TURBO_CKPT=weights/budget3d_idol_turbo.safetensors sbatch slurm/tests_cpu.sbatch
#SBATCH --job-name=budget3d-tests-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/budget3d-tests-cpu-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
bash tests/run_cpu.sh
