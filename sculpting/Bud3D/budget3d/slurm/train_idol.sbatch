#!/usr/bin/env bash
# Budget3D-Idol: 20,000 steps of online teacher-rollout distillation from the released Idol (configs/idol.yaml).
# The released checkpoint: 4x H100 80 GB, global batch 64 = 4 per GPU x 4 accumulation x 4 GPUs, 48.8 h.
# Resumable: resubmit and it continues from <OUT>/checkpoint-last.pth.
#
#   DATA=data/dome_objaverse OUT=runs/budget3d_idol sbatch slurm/train_idol.sbatch
#SBATCH --job-name=budget3d-idol
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/budget3d-idol-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
NGPU=${NGPU:-4}                      # 1, 2 or 4: the accumulation keeps the global batch at 64
srun --kill-on-bad-exit=1 python -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((20000 + RANDOM % 20000)) \
  scripts/train.py --config configs/idol.yaml --data "${DATA:-data/dome_objaverse}" --out "${OUT:-runs/budget3d_idol}" \
  --grad_accum $((16 / NGPU)) --resume
