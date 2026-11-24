#!/bin/bash
# Dyn-HaMR world runs (bench/world_dynhamr.py), sharded over GPUs; resumable (finished clips are skipped). Needs
# scripts/dynhamr/setup.sh. Run from the repository root. One work dir per clip is shared by all configurations (frames,
# HaMeR tracks), so run the first configuration of a clip alone before others reuse its tracks. The supplement's runs
# (HOT3D pilot clips CLIPS=clip-001881,clip-002025,clip-002490,clip-002742,clip-003034, every 20th benchmark clip; stages
# I-II on the camera of the WiLoR + SLAM row, world scale fixed):
#   current version:     DYNHAMR_ROOT=$THIRDPARTY/Dyn-HaMR[-camfix] EXP=slam_fixscale[_camfix] OVERRIDES=model.opt_scale=False
#   paper-time release:  DYNHAMR_ROOT=$THIRDPARTY/Dyn-HaMR-98e1622[-camfix] DYNHAMR_HAMER_PY=<setup_98e1622.sh env>/bin/python
#                        DATA_CFG=video EXP=98e_slam_fixscale[_camfix] OVERRIDES=model.opt_scale=False
#                        then the same with RESULT_STAGE=init OUT=$RESULTS/hot3d/dynhamr-98e_slam_fixscale-init: its HaMeR
#                        initialiser, read from that run's logs (for bench.world_dynhamr_pa)
#   [CAMS=<npz dir>] [CLIPS=a,b,...] GPUS=0,1 [PER_GPU=1] [DATASET=hot3d | egostandard] [TRACKS_ONLY=1] [WORK=<local dir>] \
#     bash scripts/dynhamr/run.sh
# then: python -m bench.world $RESULTS/<dataset>/dynhamr-$EXP --cams $RESULTS/<dataset>/dynhamr-$EXP/cams [--chunk 0]
# and python -m bench.world_dynhamr_pa --init $RESULTS/hot3d/dynhamr-98e_slam_fixscale-init --final $RESULTS/hot3d/dynhamr-98e_slam_fixscale.
set -e
TP=${THIRDPARTY:-thirdparty}
DATASET=${DATASET:-hot3d}
RESULTS=${RESULTS:-output}
case $DATASET in
  hot3d) ROOT=${DATA_ROOT:-data}/hot3d; SPLIT=bench/splits/hot3d_bench.json ;;
  egostandard) ROOT=${DATA_ROOT:-data}/egostandard; SPLIT=bench/splits/egostandard_bench.json ;;
esac
EXP=${EXP:-slam_fixscale}
OUT=${OUT:-$RESULTS/$DATASET/dynhamr-$EXP}
WORK=${WORK:-/tmp/dynhamr_work/$DATASET}  # a local disk: it rewrites a 12 MB optimiser state every step
SRC="--cams ${CAMS:-$RESULTS/world/$DATASET/slam_hawor_r1}"  # the camera of the WiLoR + SLAM row
PY=${DYNHAMR_ENV:-$TP/envs/dynhamr}/bin/python
MANO=${DYNHAMR_ROOT:-$TP/Dyn-HaMR}/_DATA/data/mano  # setup.sh's copies (loadable under numpy 1)
DEVS=(${GPUS//,/ })
N=$((${#DEVS[@]} * ${PER_GPU:-1}))
export OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4}
mkdir -p $OUT/logs $WORK
pids=()
for s in $(seq 0 $((N - 1))); do
  $PY -m bench.world_dynhamr --dataset $DATASET --split $SPLIT --data_root $ROOT --mano_dir $MANO $SRC --work $WORK \
    --out $OUT --exp_name $EXP --overrides "${OVERRIDES:-}" --gpu ${DEVS[$((s % ${#DEVS[@]}))]} ${CLIPS:+--clips $CLIPS} \
    ${TRACKS_ONLY:+--tracks_only} ${DATA_CFG:+--data_cfg $DATA_CFG} ${RESULT_STAGE:+--result_stage $RESULT_STAGE} \
    --shard $s --num_shards $N > $OUT/logs/shard_${s}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
grep -h "^\[dynhamr\]" $OUT/logs/shard_*.log | tail -n 5
ls $OUT/*.pkl 2>/dev/null | wc -l
[ $fail -eq 0 ] || { echo "some shards failed"; exit 1; }
