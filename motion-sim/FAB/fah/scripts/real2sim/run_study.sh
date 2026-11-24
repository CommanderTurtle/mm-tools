#!/usr/bin/env bash
# The real-to-sim study of the supplementary material (docs/real2sim.md), stage by stage, on the clips of
# bench/splits/real2sim.json. Run from the repository root:
#
#   R2S=work/real2sim DATA=/data GROUP=eval_old bash scripts/real2sim/run_study.sh prep
#   ... hands | seg | recon | track | sim | diag | score     (sim / diag: CTRL=joint|cartesian REP=1|2|3)
#
# R2S: work directory (one sub-directory per clip); DATA: dataset root with dexycb/clips/<subject>/<seq>_<cam>.tar
# (bench.datasets.build_dexycb) and hot3d/train_aria/clip-*.tar (HOT3D-Clips); GROUP: dev | eval_old | eval_new
# (CL="a b" overrides the clip list); GPUS: GPU ids used round-robin (default 0).
# Interpreters: PY (FAH + scripts/real2sim requirements, default python), MOGE_PY (MoGe-2, default $PY),
# SAM3_PY (SAM 3), SAM3DO_PY (SAM 3D Objects); TRT: fah-build-trt engine directory (optional).
set -e
stage=$1
: "${R2S:?set R2S}"
SPLIT=${SPLIT:-bench/splits/real2sim.json}
PY=${PY:-python}; MOGE_PY=${MOGE_PY:-$PY}
G=(${GPUS:-0}); NG=${#G[@]}
export MUJOCO_GL=${MUJOCO_GL:-egl} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CL=${CL:-$($PY -c "import json; print(' '.join(json.load(open('$SPLIT'))['${GROUP:?set GROUP or CL}']))")}
info() { $PY -c "import json; d=json.load(open('$SPLIT')); c={**d['dev'], **d['eval_old'], **d['eval_new']}['$1']; print(c['source'], c['clip'], c.get('object', '').replace('_', ' '))"; }
two() { [ "$(info $1 | cut -d' ' -f1)" = hot3d ] && echo --two_hands 1; true; }
P="--contact 1 --hold_until_lift 1 --controller 1 --ctrl_pen 0.004 --gravity headset"  # the frozen replay settings
CTRL=${CTRL:-joint}; REP=${REP:-1}
TAG=$([ $CTRL = joint ] && echo joint || echo cart)$([ $REP = 1 ] || echo _r$REP)
ARGS="$P --coacd_seed $((REP - 1))$([ $CTRL = joint ] && echo ' --ctrl_flex 1')"
mkdir -p $R2S
k=0
for n in $CL; do
  g=${G[$((k % NG))]}; k=$((k + 1)); d=$R2S/$n
  case $stage in
    prep)
      read src clip obj <<< "$(info $n)"
      if [ ! -e $d/clip.json ]; then
        if [ $src = dexycb ]; then $PY scripts/real2sim/prep_clip.py --dexycb $DATA/dexycb/clips/$clip.tar --out $d
        else $PY scripts/real2sim/prep_clip.py --hot3d $DATA/hot3d/train_aria/$clip.tar --object "$obj" --out $d; fi
      fi
      [ -e $d/depth.npz ] || CUDA_VISIBLE_DEVICES=$g $MOGE_PY scripts/real2sim/depth.py --clip $d ;;
    hands)
      [ -e $d/hands.pkl ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/real2sim/hands.py --clip $d --model fah --mode video ${TRT:+--trt_dir $TRT} ;;
    seg)
      [ -e $d/masks.npz ] || CUDA_VISIBLE_DEVICES=$g ${SAM3_PY:?set SAM3_PY} scripts/real2sim/segment.py --clip $d --prompt text \
          --text_rule dataset --search 5 --relink 0.2 || echo "$n: segmentation failed" ;;
    recon)
      [ -e $d/masks.npz ] && [ ! -e $d/recon_s3do/layout.json ] && { LIDRA_SKIP_INIT=true CUDA_VISIBLE_DEVICES=$g ${SAM3DO_PY:?set SAM3DO_PY} \
          scripts/real2sim/reconstruct_s3do.py --clip $d --frame -1 --texture || echo "$n: no reconstruction"; } ;;
    track)
      [ -e $d/recon_s3do/layout.json ] && { CUDA_VISIBLE_DEVICES=$g $PY scripts/real2sim/track.py --clip $d --recon recon_s3do --init layout \
          --grasp_hold 1 $(two $n); [ -e $d/gt.npz ] && $PY scripts/real2sim/eval_gt.py --clip $d > /dev/null; } ;;
    sim)
      [ -e $d/track.npz ] && CUDA_VISIBLE_DEVICES=$g $PY scripts/real2sim/sim.py --clip $d --hand fah --object recon $ARGS $(two $n) \
          --render ${RENDER:-0} --tag $TAG ;;
    diag)  # DexYCB: FAH hand + GT object, GT hand + GT object (no contact optimisation for the GT hand), same replay
      [ -e $d/gt.npz ] && [ -e $d/track.npz ] && { CUDA_VISIBLE_DEVICES=$g $PY scripts/real2sim/sim.py --clip $d --hand fah --object gt $ARGS \
          --render 0 --tag diag_gtobj_$TAG; CUDA_VISIBLE_DEVICES=$g $PY scripts/real2sim/sim.py --clip $d --hand gt --object gt \
          ${ARGS/--contact 1/} --render 0 --tag diag_gtgt_$TAG; } ;;
    score) break ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
if [ $stage = score ]; then
  for c in joint cart; do
    $PY scripts/real2sim/repeats.py --root $R2S --split $SPLIT --groups ${SCORE_GROUPS:-eval_old,eval_new} --runs sim_$c,sim_${c}_r2,sim_${c}_r3 \
        --diags _$c,_${c}_r2,_${c}_r3 --json $R2S/repeats_$([ $c = joint ] && echo joint || echo cartesian).json
  done
fi
