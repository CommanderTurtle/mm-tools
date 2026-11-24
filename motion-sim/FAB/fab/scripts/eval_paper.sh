#!/bin/bash
# The paper's camera-space tables for the released models, one section at a time (docs/reproduce.md):
#   predictions with bench/eval_hmr.py (one shard per local GPU; the released FABPredictor call on the dataset's boxes),
#   scored by bench/eval_standard_protocol.py (or the section's own scorer). Resumable: finished (model, dataset) pairs
#   are skipped.
#
#   SEC=std bash scripts/eval_paper.sh        # camera space (3DPW, RICH test), EMDB-1, more benchmarks (EgoBody, Harmony4D)
#   SEC=hoff bash scripts/eval_paper.sh       # camera space / EMDB-1: FAB-H through MHR's official SMPL fit (slow, ~1.4 s / person)
#   SEC=degrade bash scripts/eval_paper.sh    # blur / low resolution (motion-blurred and low-resolution copies)
#   SEC=auto bash scripts/eval_paper.sh       # automatic detection (the released person detector instead of the dataset boxes)
#   SEC=ssp3d bash scripts/eval_paper.sh      # body shape (SSP-3D)
#   SEC=coco bash scripts/eval_paper.sh       # each output on its own target: 2D rows (COCO val2017 keypoint AP)
#   SEC=parts bash scripts/eval_paper.sh      # each output on its own target: body-part rows (3DPW, EMDB-1, EMDB dev)
#
# Data: $FAB_DATA (default $FAB_HOME/data) laid out as in datasets/README.md. Outputs: $FAB_OUT (default
# $FAB_HOME/bench): preds/<model>/, scores/<section>/. MODELS overrides the models of a section (default per section).
set -uo pipefail
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}
D=${FAB_DATA:-$FAB_HOME/data}
O=${FAB_OUT:-$FAB_HOME/bench}
P=$O/preds
: "${SEC:?SEC=std|hoff|degrade|auto|ssp3d|coco|parts}"
NG=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NG" -ge 1 ] || { echo "no GPU found (set NUM_GPUS)"; exit 1; }
mkdir -p $O/logs $O/scores
log() { echo "== $(date '+%H:%M:%S') $*"; }

flags() {  # the released call per model: FAB-L / FAB-S with the released profile + fitter (PyTorch body); FAB-H both profiles
  case $1 in
    fab-l|fab-s) echo "--fab_ckpt $1 --compile --upright 0 --predictor --save_dense --predictor_kw FINAL";;
    fab-h) echo "--fab_ckpt fab-h --compile --upright 0 --predictor";;  # its default profile: the SMPL fitted to its MHR mesh
    fab-h-points) echo "--fab_ckpt fab-h --compile --upright 0 --predictor --save_dense --predictor_kw {\"profile\":\"points\"}";;  # fitted to its dense points
    fab-h-mesh) echo "--fab_ckpt fab-h --compile --upright 0 --predictor --predictor_kw DEPLOYED_NOFIT";;  # MHR mesh, no fit
    *) echo UNKNOWN;;
  esac
}
root_of() {
  local base=${1%_dark}; base=${base%_blur}; base=${base%_lowres}
  case $base in
    3dpw) echo $D/3dpw;;
    emdb1|emdb_dev) echo $D/emdb;;
    *) echo $D/$base;;
  esac
}
preds() {  # preds <out root> "<models>" "<dataset:stride ...>" [extra eval_hmr args]
  local out=$1 models=$2 sets=$3; shift 3
  for m in $models; do
    local F; F=$(flags $m)
    [ "$F" = UNKNOWN ] && { echo "unknown model $m"; exit 1; }
    for item in $sets; do
      local ds=${item%%:*} st=${item#*:}
      if [ "$(ls $out/$m/${ds}_pred_shard*.npz 2>/dev/null | wc -l)" = "$NG" ]; then echo "skip $m $ds (done)"; continue; fi
      log "predict $m $ds (stride $st, $NG GPUs)"
      local pids=()
      for g in $(seq 0 $((NG - 1))); do
        CUDA_VISIBLE_DEVICES=$g python -m bench.eval_hmr --dataset $ds --root $(root_of $ds) --out $out/$m \
          --shard $g --num_shards $NG --stride $st --save_pred $F "$@" > $O/logs/pred_${m}_${ds}_$g.log 2>&1 &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait $pid || { echo "FAILED $m $ds"; tail -n 15 $O/logs/pred_${m}_${ds}_*.log; exit 1; }; done
    done
  done
}
score() {  # score <section> <pred root> "<models>" <kind> <datasets...>
  local sec=$1 pr=$2 models=$3 kind=$4; shift 4
  local args=(); for m in $models; do args+=(--method "$m:$kind:$pr/$m"); done
  python -m bench.eval_standard_protocol --out_dir $O/scores/$sec --per_sample "${args[@]}" --datasets "$@" \
    > $O/logs/score_$sec.log 2>&1 || { tail -n 20 $O/logs/score_$sec.log; exit 1; }
  cat $O/scores/$sec/standard.md
}
DET="--det_model ${FAB_WEIGHTS:-$FAB_HOME/weights}/shared/detector.pt --det_conf 0.25 --det_min_iou 0.3"

case $SEC in
  std)
    preds $P "${MODELS:-fab-l fab-s fab-h fab-h-points}" "3dpw:4 rich_test:2 emdb1:2"
    score std $P "${MODELS:-fab-l fab-s fab-h fab-h-points}" smpl 3dpw rich_test emdb1
    preds $P "${MODELS:-fab-l fab-h fab-h-points}" "egobody_test:1 harmony4d_test:1"
    score std2 $P "${MODELS:-fab-l fab-h fab-h-points}" smpl egobody_test harmony4d_test ;;
  hoff)
    preds $P "fab-h-mesh" "3dpw:4 rich_test:2 emdb1:2"
    score hoff $P "fab-h-mesh" mhr 3dpw rich_test emdb1 ;;
  degrade)
    S="3dpw_blur:4 3dpw_lowres:4 emdb1_blur:2 emdb1_lowres:2 rich_test_blur:2 rich_test_lowres:2"
    preds $P "${MODELS:-fab-l fab-h fab-h-points}" "$S"
    score degrade $P "${MODELS:-fab-l fab-h fab-h-points}" smpl 3dpw_blur 3dpw_lowres emdb1_blur emdb1_lowres rich_test_blur rich_test_lowres ;;
  auto)
    preds $O/preds_auto "${MODELS:-fab-l}" "3dpw:10 emdb1:10 3dpw_dark:10 emdb1_dark:10" $DET
    score auto $O/preds_auto "${MODELS:-fab-l}" smpl 3dpw emdb1 3dpw_dark emdb1_dark
    python - $O/preds_auto 3dpw emdb1 3dpw_dark emdb1_dark <<'PY'
import glob, json, sys
root = sys.argv[1]
for ds in sys.argv[2:]:
    for d in sorted(glob.glob(f"{root}/*")):
        rq = dt = 0
        for f in glob.glob(f"{d}/{ds}_detections_shard*.json"):
            r = json.load(open(f)); rq += r["requested_samples"]; dt += r["detected_samples"]
        if rq:
            print("recall", ds, d.rsplit("/", 1)[1], dt, rq, round(100 * dt / rq, 1))
PY
    ;;
  ssp3d)
    preds $O/preds_ssp "${MODELS:-fab-l fab-h fab-h-points}" "ssp3d:1"
    args=(); for m in ${MODELS:-fab-l fab-h fab-h-points}; do args+=(--method "$m:$O/preds_ssp/$m"); done
    python -m bench.eval_ssp3d score --index $D/ssp3d/index_joints.pkl --out $O/scores/ssp3d.json "${args[@]}" \
      > $O/logs/ssp3d.log 2>&1 || { tail -n 20 $O/logs/ssp3d.log; exit 1; }
    cat $O/scores/ssp3d.json ;;
  coco)
    for m in ${MODELS:-fab-l fab-s}; do
      out=$O/scores/coco_$m
      if [ "$(ls $out/shard*.json 2>/dev/null | wc -l)" != "$NG" ]; then
        log "COCO $m"
        pids=()
        for g in $(seq 0 $((NG - 1))); do
          CUDA_VISIBLE_DEVICES=$g python -m bench.eval_coco2d --root $D/coco_val --fab_ckpt $m --predictor --body_engine torch \
            --out $out --shard $g --num_shards $NG > $O/logs/coco_${m}_$g.log 2>&1 &
          pids+=($!)
        done
        for pid in "${pids[@]}"; do wait $pid || { echo "FAILED coco $m"; tail -n 15 $O/logs/coco_${m}_*.log; exit 1; }; done
      fi
      python -m bench.eval_coco2d --root $D/coco_val --out $out --merge | tail -n 20
    done ;;
  parts)  # body-part rows: part mIoU / silhouette IoU against parts rendered from the GT mesh (every 5th frame)
    for ds in 3dpw emdb1 emdb_dev; do
      if [ "$(ls $O/scores/parts/${ds}_shard*.npz 2>/dev/null | wc -l)" != "$NG" ]; then
        log "parts $ds"
        pids=()
        for g in $(seq 0 $((NG - 1))); do
          CUDA_VISIBLE_DEVICES=$g python -m bench.eval_hmr --dataset $ds --root $(root_of $ds) --fab_ckpt fab-l --compile \
            --upright 0 --predictor --predictor_kw DEPLOYED:parts --parts_res 256 --stride 5 --out $O/scores/parts \
            --shard $g --num_shards $NG > $O/logs/parts_${ds}_$g.log 2>&1 &
          pids+=($!)
        done
        for pid in "${pids[@]}"; do wait $pid || { echo "FAILED parts $ds"; tail -n 15 $O/logs/parts_${ds}_*.log; exit 1; }; done
      fi
    done
    python - $O/scores/parts <<'PY'
import glob, sys
import numpy as np
for ds in ("3dpw", "emdb1", "emdb_dev"):
    z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{sys.argv[1]}/{ds}_shard*.npz"))]
    if not z:
        print(ds, "missing"); continue
    m = np.concatenate([x["miou_parts"] for x in z]); s = np.concatenate([x["sil_iou_parts"] for x in z])
    print(ds, "n", len(m), "part mIoU", round(float(np.nanmean(m)), 3), "silhouette IoU", round(float(np.nanmean(s)), 3))
PY
    ;;
  *) echo "SEC=std|hoff|degrade|auto|ssp3d|coco|parts"; exit 1 ;;
esac
log "$SEC done"
