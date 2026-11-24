#!/usr/bin/env bash
# Budget3D-Idol-Turbo: online teacher-rollout distillation from speridlabs/idol-turbo (Idol + phase-LoRA experts,
# 5 guidance-free steps), as the released checkpoint was trained on 4x RTX PRO 6000 96 GB:
#   PHASE=1  0 -> 10,000 steps (configs/idol_turbo.yaml: 4 per pass, every pass checkpointed), 33.0 h
#   PHASE=2  10,000 -> 20,000 steps from that checkpoint (configs/idol_turbo_phase2.yaml: learning-rate re-warm, new
#            seed, 2 per pass x 8, checkpointing only where memory needs it, one-pass LoRA), 24.3 h
# Global batch 64 throughout.  Resumable: resubmit and it continues from <OUT>/checkpoint-last.pth.
#
#   PHASE=1 DATA=data/dome_objaverse OUT=runs/budget3d_idol_turbo_10k sbatch slurm/train_idol_turbo.sbatch
#   PHASE=2 DATA=data/dome_objaverse OUT=runs/budget3d_idol_turbo_20k INIT=runs/budget3d_idol_turbo_10k/checkpoint-last.pth sbatch slurm/train_idol_turbo.sbatch
#SBATCH --job-name=budget3d-turbo
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/budget3d-turbo-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
NGPU=${NGPU:-4}
OUT=${OUT:-runs/budget3d_idol_turbo}
if [ "${PHASE:-1}" = 1 ]; then
  CONFIG=configs/idol_turbo.yaml; ACCUM=$((16 / NGPU))
else                                   # continue a phase-1 checkpoint to 20,000 steps
  mkdir -p "$OUT"
  [ -f "$OUT/checkpoint-last.pth" ] || cp "${INIT:?set INIT to the phase-1 checkpoint-last.pth}" "$OUT/checkpoint-last.pth"
  CONFIG=configs/idol_turbo_phase2.yaml; ACCUM=$((32 / NGPU))
fi
srun --kill-on-bad-exit=1 python -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=$((20000 + RANDOM % 20000)) \
  scripts/train_turbo.py --config "$CONFIG" --data "${DATA:-data/dome_objaverse}" --out "$OUT" --grad_accum "$ACCUM" --resume
