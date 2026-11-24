#!/bin/bash
# The G1 references of the paper's retargeter table (FAB-Retarget points mode), rebuilt from each clip's fab-world body and
# FAB's per-frame dense points by the flat-floor reference lift the paper's G1 tables use (docs/reproduce.md, "G1 references"):
#   1. scripts/g1_refs_world_raw.py   the fab-world body in bench/sonic_motion.py's raw format (camera frame via inv(T_world_cam))
#   2. bench/sonic_motion.py          the world lift with fab-world's cameras and gravity, no extra smoothing, resampled to 50 fps
#                                     onto a flat floor at z = 0, mesh-rule contacts, FAB's dense points registered on the body
#                                     (--export_human / --export_pts; also the SMPL reference the trackers read,
#                                     <clip>__ikmot_motion.npz)
#   3. bench/videomimic/scene_retarget.py   the points-mode solve (fab/retarget/release_config.py solve_cfg_dp, SIGMA_IN)
#   4. bench/stance_pin.py            the stance pin (release_config.PIN), ALL clips in one call: the clips pinned together in
#                                     sorted order, a batch closed once it holds 1,500 frames at 50 fps (one to three of the 25
#                                     clips); the pin depends on which clips share a batch, so give it the same clip set
#   5. scripts/g1_refs_convert.py     one reference file per clip: the pinned G1 + the lift's SMPL fields
# Inputs (<inputs dir>), per clip, from your own runs (docs/reproduce.md, "G1 references"; scripts/g1_refs_inputs.py):
#   <clip>.npz       the fab-world output of the clip's frames (frames, T_world_cam, gravity_world, K, fps, pose, betas, trans,
#                    valid, ...)
#   <clip>_raw.npz   FAB's per-frame raw of the same run, with its dense points (pts, surface_idx): <fab-world out>/work/ours/fab/
# Default clips: the 25 of bench/sonic_clips.py FLAT_CLIPS. fab-retarget's own chain (scripts/fab_retarget.py: the reconstructed
# scene, level snap, terrain clean, one clip per stance pin) is not this lift and gives other references for the same clips.
#   bash scripts/reproduce_g1_refs.sh <inputs dir> <out dir> [<clip> ...]
#   -> <out>/refs/<clip>__ikmot_motion.npz (+ <out>/lift, <out>/human, <out>/g1/{g1v2,p} on the way)
set -euo pipefail
IN=$(cd "$1" && pwd); mkdir -p "$2"; OUT=$(cd "$2" && pwd); shift 2
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=${MUJOCO_GL:-egl}
if [ $# -gt 0 ]; then CLIPS="$*"; else CLIPS=$(python -c "from bench.sonic_clips import FLAT_CLIPS; print(' '.join(sorted(FLAT_CLIPS)))"); fi
miss=""
for c in $CLIPS; do for f in "$c.npz" "${c}_raw.npz"; do [ -f "$IN/$f" ] || miss="$miss $f"; done; done
if [ -n "$miss" ]; then
  echo "[g1_refs] missing inputs in $IN:$miss" >&2
  echo "[g1_refs] per clip: <clip>.npz (the fab-world output of the clip's frames) and <clip>_raw.npz (its work/ours/fab/<clip>_raw.npz);" >&2
  echo "[g1_refs] make them with fab-world (docs/reproduce.md, \"G1 references\"; scripts/g1_refs_inputs.py)" >&2
  exit 2
fi
# the reference lift's motion settings (bench/sonic_motion.py --motion_cfg: fab.retarget.motion overrides)
MCFG='{"h_contact": 0.04, "v_contact": 0.15, "w_dir": 1.0, "dir_mode": "links", "w_rrot": 1.0, "w_pose": 0.1, "w_foot_h": 300.0, "lr_sched": "cos", "lr_t": 0.002, "w_swing": 100.0, "swing_clear": 0.03, "w_acc": 64.0}'
SOLVE=$(python -c "import json; from fab.retarget import release_config as C; print(json.dumps(C.solve_cfg_dp()))")
SIG=$(python -c "from fab.retarget import release_config as C; print(C.SIGMA_IN)")
PIN=$(python -c "import json; from fab.retarget import release_config as C; print(json.dumps(C.PIN))")
mkdir -p "$OUT/lift" "$OUT/logs" "$OUT/g1/_no_head_files"
# points mode's dense-point -> G1 correspondence, built on first use for the installed G1 files (as fab-retarget does)
python -c "from fab.retarget import dense_corr as D; p = D.default_path(); D.make(out=p) if not p.exists() or D.stale(p) else None" \
  > "$OUT/logs/dense_corr.log" 2>&1 || { echo "[g1_refs] dense correspondence failed ($OUT/logs/dense_corr.log)"; exit 1; }
echo "[g1_refs] 1. fab-world bodies -> raw format"
python scripts/g1_refs_world_raw.py "$IN" "$IN" "$OUT/lift_raw" $CLIPS > "$OUT/logs/world_raw.log" 2>&1 \
  || { echo "[g1_refs] raw format failed ($OUT/logs/world_raw.log)"; exit 1; }
echo "[g1_refs] 2. reference lift"
for c in $CLIPS; do
  [ -f "$OUT/human/$c.npz" ] && [ -f "$OUT/lift/${c}__ikmot_motion.npz" ] && continue
  cp "$OUT/lift_raw/${c}_raw.npz" "$OUT/lift/${c}_raw.npz"
  FPS=$(python -c "import numpy as np; print(float(np.load('$IN/$c.npz')['fps']))")
  python -m bench.sonic_motion --seq "$c" --name "$c" --fps "$FPS" --out "$OUT/lift" --world "$IN/$c.npz" --world_gravity down \
    --world_only --reuse --no_input_video --fab_ckpt none --fit_iters 0 --sigma_rot 0 --sigma_pos 0 \
    --g1_ik --ik_motion --ik_suffix __ikmot --motion_cfg "$MCFG" --ik_flex_init --export_human "$OUT/human" --export_pts \
    > "$OUT/logs/lift_$c.log" 2>&1 || { echo "[g1_refs] $c: reference lift failed ($OUT/logs/lift_$c.log)"; exit 1; }
done
echo "[g1_refs] 3. solve"
ONLY=$(echo $CLIPS | tr ' ' ',')
python -m bench.videomimic.scene_retarget --recon "$OUT/human" --only "$ONLY" --out "$OUT/g1/g1v2" --sigma_in "$SIG" --cfg "$SOLVE" \
  > "$OUT/logs/solve.log" 2>&1 || { echo "[g1_refs] solve failed ($OUT/logs/solve.log)"; exit 1; }
echo "[g1_refs] 4. stance pin (one call; a batch closes at 1500 frames)"
python -m bench.stance_pin --run "$OUT/g1" --tag g1v2 --out_tag p --recon "$OUT/human" --only "$ONLY" --contacts_dir "$OUT/g1/_no_head_files" \
  --cfg "$PIN" --batch_frames 1500 --gate > "$OUT/logs/pin.log" 2>&1 || { echo "[g1_refs] stance pin failed ($OUT/logs/pin.log)"; exit 1; }
echo "[g1_refs] 5. reference files"
python scripts/g1_refs_convert.py "$OUT/g1" "$OUT/lift" "$OUT/refs" \
  "fab-retarget points mode: bench.videomimic.scene_retarget $SOLVE (sigma_in $SIG) + bench.stance_pin $PIN on the flat-floor reference lift" \
  $CLIPS > "$OUT/logs/convert.log" 2>&1 || { echo "[g1_refs] reference files failed ($OUT/logs/convert.log)"; exit 1; }
echo "[g1_refs] done: $(ls "$OUT/refs" | grep -c __ikmot_motion.npz) references in $OUT/refs"
