#!/usr/bin/env bash
# A budget sweep on 4 GPUs: every image at 100 % ... 10 % of the tokens with optimised inference, then the speed /
# memory report and the MP4s.  Resumable (finished objects are skipped).  With TURBO=1 the checkpoint is a
# Budget3D-Idol-Turbo one (5 guidance-free steps); CKPT=none gives the stock base model as a reference.
#
#   CKPT=weights/budget3d_idol_turbo.safetensors TURBO=1 OUT=gallery_turbo IMAGES="third_party/idol-sdk/demo/examples/*.webp" sbatch slurm/gallery.sbatch
#SBATCH --job-name=budget3d-gallery
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/budget3d-gallery-%j.out
# add your cluster's --partition / --account / --qos on the sbatch command line
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=8
NGPU=${NGPU:-4}
OUT=${OUT:-gallery}
IMAGES=${IMAGES:-third_party/idol-sdk/demo/examples/*.webp}
mkdir -p logs
SMI=logs/budget3d-gallery-${SLURM_JOB_ID:-local}-smi.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 5 >> "$SMI" &
SMI_PID=$!
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/gallery.py ${TURBO:+--turbo} --fast --ckpt "${CKPT:?set CKPT}" --out "$OUT" \
      --batch 8 --shard "$i" --num_shards "$NGPU" --images $IMAGES > "logs/budget3d-gallery-${SLURM_JOB_ID:-local}-gpu$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
kill $SMI_PID 2>/dev/null || true
python scripts/perf_report.py --out "$OUT" --smi "$SMI"
TO_MP4_WORKERS=12 python scripts/to_mp4.py --out "$OUT"
exit $fail
