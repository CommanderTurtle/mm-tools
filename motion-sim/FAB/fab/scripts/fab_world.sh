#!/bin/bash
# FAB world motion + scene from one video in one command (world-mode entry point; the paper's world pipeline on one
# clip, with estimated intrinsics):
#   frames -> ViPE 1.2 + UniDepth (--depth) -> dense depth -> GeoCalib -> the focal rule (bench/scene_recon/focal_rule.py:
#   a static camera, p90 camera-centre spread < 0.5 % of the median static depth: median(ViPE, GeoCalib, MoGe-2) focal,
#   ViPE re-run with K fixed) -> person segmentation -> FAB (FAB-L, or --fab S: FAB-S; the released profile and SMPL
#   fitter, the released detector) -> contacts (the model's image contact head -> the causal video contact model, world
#   frame) -> world reconstruction (ViPE depth) -> MVS (static cameras: its no-parallax rule keeps the ViPE depth) -> world
#   reconstruction (MVS depth) -> joint human-scene step
# -> <out>/<name>.npz: the world body (pose / betas / trans / joints, z-up metric world, gravity -z), cameras (T_world_cam,
#    K), contacts, scale_fix, notes; <out>/<name>_scene.ply + <name>_hfield.npz (the scene); intermediate steps in <out>/work.
# Needs the world-mode tools (docs/world.md): the ViPE envs, COLMAP and MoGe-2 are installed here when missing.
#   bash scripts/fab_world.sh <video | frame dir> [--out DIR] [--name NAME] [--fps 30] [--fab L|S] [--subject_boxes box.npz] [--focal F]
#        [--bundle model.pt] [--fitter model.pt] [--contact_head model.pt] [--contact_video model.pt] [--mvs_reuse DIR]
#   (--subject_boxes: npz box (T, 4) xyxy of the target person per frame, NaN unknown: identity only; default the most
#    prominent tracked person. --focal F: opt-in for a calibrated camera, the known focal in px: one fixed-K ViPE run, no
#    focal rule; no paper number uses it. GPU: the first of CUDA_VISIBLE_DEVICES, so several runs can share a machine.
#    --bundle / --fitter / --contact_head / --contact_video: another body model, SMPL fitter or contact pair (e.g. one
#    you trained); defaults = the released ones under $FAB_WEIGHTS: fab-l (S: fab-s), shared/smpl_fitter.pt, the body
#    model itself as the image contact head, shared/contact_video.pt; all recorded in the export's notes. --mvs_reuse
#    DIR: an existing COLMAP PatchMatch workspace root (DIR/<name>[/<name>]/dense/stereo/depth_maps) of the same cameras:
#    linked (never written) and only mvs_depth's filtering re-run with this run's body (mvs_depth --reuse 1); a clip whose
#    workspace is incomplete gets a fresh PatchMatch run in a scratch dir)
# Tools (override with env vars): FAB_HOME (~/.cache/fab), FAB_ENVS ($FAB_HOME/envs), FAB_VIPE_PY ($FAB_ENVS/vipe/bin/python),
#   FAB_VIPE12_PY ($FAB_ENVS/vipe12/bin/python), VIPE12_DIR ($FAB_ENVS/src/vipe12), FAB_COLMAP ($FAB_ENVS/colmap/bin/colmap).
set -uo pipefail
IN=""; OUT=output/fab_world; NAME=""; FPS=30; FOCAL=""; SB=""; FAB=L; BUNDLE=""; FITTER=""; CHEAD=""; CVID=""; MVSR=""
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2;; --name) NAME=$2; shift 2;; --fps) FPS=$2; shift 2;; --focal) FOCAL=$2; shift 2;;
    --subject_boxes) SB=$2; shift 2;; --fab) FAB=$2; shift 2;; --bundle) BUNDLE=$2; shift 2;; --fitter) FITTER=$2; shift 2;;
    --contact_head) CHEAD=$2; shift 2;; --contact_video) CVID=$2; shift 2;; --mvs_reuse) MVSR=$2; shift 2;;
    -h|--help) sed -n 2,26p $0; exit 0;; *) IN=$1; shift;;
  esac
done
[ -n "$IN" ] || { sed -n 2,26p $0; exit 1; }
IN=$(realpath "$IN"); [ -n "$SB" ] && SB=$(realpath "$SB"); mkdir -p $OUT; OUT=$(cd $OUT && pwd)
for v in BUNDLE FITTER CHEAD CVID MVSR; do [ -n "${!v}" ] && { [ -e "${!v}" ] || { echo "$v ${!v}: not found"; exit 1; }; printf -v $v %s "$(realpath "${!v}")"; }; done
R=$(cd $(dirname $0)/.. && pwd); cd $R
FAB_HOME=${FAB_HOME:-$HOME/.cache/fab}; FAB_ENVS=${FAB_ENVS:-$FAB_HOME/envs}; FAB_WEIGHTS=${FAB_WEIGHTS:-$FAB_HOME/weights}
export FAB_HOME FAB_ENVS FAB_WEIGHTS
case $FAB in L) CK=$FAB_WEIGHTS/fab-l/model.pt;; S) CK=$FAB_WEIGHTS/fab-s/model.pt;; *) echo "--fab L|S"; exit 1;; esac  # FAB-L / FAB-S
CK=${BUNDLE:-$CK}; FITTER=${FITTER:-$FAB_WEIGHTS/shared/smpl_fitter.pt}
for f in $CK $FITTER; do [ -f $f ] || { echo "$f: not found (download the FAB weights: README.md)"; exit 1; }; done
GPU=${CUDA_VISIBLE_DEVICES:-0}; GPU=${GPU%%,*}  # ViPE's --gpu: this run's GPU (world_vipe sets CUDA_VISIBLE_DEVICES from it)
V12=${VIPE12_DIR:-$FAB_ENVS/src/vipe12}; PY12=${FAB_VIPE12_PY:-$FAB_ENVS/vipe12/bin/python}; PV=${FAB_VIPE_PY:-$FAB_ENVS/vipe/bin/python}
export FAB_VIPE12_PY=$PY12 FAB_VIPE_PY=$PV VIPE12_DIR=$V12
COLMAP=${FAB_COLMAP:-$FAB_ENVS/colmap/bin/colmap}; export FAB_COLMAP=$COLMAP
HEAD=${CHEAD:-$CK}; VID=${CVID:-$FAB_WEIGHTS/shared/contact_video.pt}  # the image contact head is part of the body model
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
c=$(basename "$IN"); [ -d "$IN" ] || c=${c%.*}; c=${NAME:-$c}; G=$OUT/work; L=$G/clips.json; Lc=$L
mkdir -p $G/logs $G/focal $G/ours $G/ours_v $G/ours_m $G/scratch
ln -sfn $G/ours/fab $G/ours_v/fab; ln -sfn $G/ours/fab $G/ours_m/fab
log() { echo "== $1 $(date +%H:%M)"; }
bash bench/baselines/setup_vipe12.sh > $G/logs/setup_vipe12.log 2>&1 || { tail -n 20 $G/logs/setup_vipe12.log; exit 1; }
bash bench/baselines/setup_vipe.sh > $G/logs/setup_vipe.log 2>&1 || tail -n 5 $G/logs/setup_vipe.log
$PV -c "import imageio.v2" 2>/dev/null || uv pip install --python $PV imageio imageio-ffmpeg > /dev/null 2>&1
[ -x "$(command -v $COLMAP)" ] || bash scripts/colmap_setup.sh > $G/logs/colmap_setup.log 2>&1
bash bench/videomimic/setup_recon.sh > $G/logs/setup_recon.log 2>&1 || tail -n 5 $G/logs/setup_recon.log
if [ -z "$FOCAL" ] && ! python -c "import moge" 2>/dev/null; then
  python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > $G/torch_pin.txt
  # MoGe at the commit the paper's runs used (MoGe-2 weights from Hugging Face at run time)
  python -m pip install -q -c $G/torch_pin.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" > /dev/null 2>&1
fi
# the clip: a frame directory as it is (sorted images), a video decoded to JPEG frames
python - "$IN" $c $FPS $L $G/frames <<'PY' || exit 1
import json, os, sys, cv2
src, c, fps, L, fd = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
if os.path.isdir(src):
    imgs = [os.path.abspath(os.path.join(src, f)) for f in sorted(os.listdir(src)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    os.makedirs(fd, exist_ok=True); cap, imgs = cv2.VideoCapture(src), []
    while True:
        ok, f = cap.read()
        if not ok: break
        imgs.append(f"{fd}/{len(imgs):05d}.jpg"); cv2.imwrite(imgs[-1], f, [cv2.IMWRITE_JPEG_QUALITY, 95])
json.dump({c: {"images": imgs, "fps": fps, "source": os.path.abspath(src)}}, open(L, "w"), indent=0)
print(c, len(imgs), "frames")
PY
[ -n "$SB" ] && { mkdir -p $G/subject_boxes; cp $SB $G/subject_boxes/$c.npz; }
log "setup ($c)"

# A: cameras + depth, GeoCalib, the focal rule (bench/scene_recon/focal_rule.py)
FR="python -m bench.scene_recon.focal_rule --work $G --clip $c"
if [ -n "$FOCAL" ]; then  # opt-in calibrated camera: one fixed-K ViPE run (the pipeline's pinhole keeps the principal point centred)
  $FR given --focal $FOCAL
else
  [ -f $G/vipe/cams/$c.npz ] || $PV -m bench.world_vipe --list $Lc --out $G/vipe --python $PY12 --vipe $V12 --depth --gpu $GPU --work $G/scratch/vipe_$c > $G/logs/vipe.log 2>&1
  [ -f $G/vipe/dense/$c.npz ] || $PY12 -m bench.videomimic.vipe_depth --vipe_out $G/vipe --vipe $V12 --name $c > $G/logs/dense.log 2>&1
  [ -f $G/geocalib/$c.npz ] || $PV -m bench.world_geocalib --list $Lc --cams $G/vipe/cams --out $G/geocalib > $G/logs/geo.log 2>&1
  [ -f $G/vipe/cams/$c.npz ] && [ -f $G/vipe/dense/$c.npz ] || { echo "ViPE failed"; tail -n 5 $G/logs/vipe.log $G/logs/dense.log; exit 1; }
  $FR static; rc=$?
  [ $rc = 0 ] || [ $rc = 10 ] || { echo "focal rule failed"; exit 1; }
  if [ $rc = 0 ]; then  # static camera: median(ViPE, GeoCalib, MoGe-2) focal, then ViPE with K fixed
    $PV -m bench.scene_recon.focal_compare geocalib --list $Lc --out $G/focal/geo_$c.json > $G/logs/focal_geo.log 2>&1
    python -m bench.scene_recon.focal_compare moge --list $Lc --out $G/focal/moge_$c.json > $G/logs/focal_moge.log 2>&1
    $FR focal
  fi
fi
if [ -f $G/focal/fixed_$c.json ]; then
  [ -f $G/vipe_fk/cams/$c.npz ] || $PV -m bench.world_vipe --list $Lc --out $G/vipe_fk --python $PY12 --vipe $V12 --depth \
    --fixed_focal_json $G/focal/fixed_$c.json --gpu $GPU --work $G/scratch/vipe_fk_$c > $G/logs/vipe_fk.log 2>&1
  [ -f $G/vipe_fk/dense/$c.npz ] || $PY12 -m bench.videomimic.vipe_depth --vipe_out $G/vipe_fk --vipe $V12 --name $c > $G/logs/dense_fk.log 2>&1
  [ -f $G/geocalib_fk/$c.npz ] || $PV -m bench.world_geocalib --list $Lc --cams $G/vipe_fk/cams --out $G/geocalib_fk > $G/logs/geo_fk.log 2>&1
fi
$FR select
log "cameras + depth"

# B: seg, FAB, contacts, the world chain
KW=$(python -c "import json; from fab.predictor import SIMPLE_PROFILE; k=dict(SIMPLE_PROFILE); k['smpl_fitter']='$FITTER'; print(json.dumps(k))")
[ -f $G/seg/${c}_seg.npz ] || python -m bench.scene_recon.clean_render seg --recon $G/vipe_sel/cams --dense $G/vipe_sel/dense --list $L --only $c --out $G/seg > $G/logs/seg.log 2>&1
[ -f $G/ours/fab/${c}_raw.npz ] || python -m bench.videomimic.recon fab --list $L --only $c --out $G/ours --vipe $G/vipe_sel --fab_ckpt $CK --detector deployed \
  --predictor_kw "$KW" ${SB:+--subject_boxes $G/subject_boxes} > $G/logs/fab.log 2>&1
[ -f $G/ours/fab/${c}_raw.npz ] || { echo "FAB stage failed"; tail -n 5 $G/logs/fab.log; exit 1; }
log "seg + FAB"
W=$G/contact_work
if [ ! -f $G/contacts/${c}__0.contact.npz ]; then
  mkdir -p $W; export FAB_CONTACT_CLIPS=$L FAB_CONTACT_RAW="$G/ours/fab/{c}_raw.npz" FAB_CONTACT_CAMS="$G/vipe_sel/cams/{c}.npz" CLIPS_ONLY=$c
  python scripts/contact_clips.py prep --set clip --work $W > $W/prep.log 2>&1
  python -m train.extract_contact_tracks --source clips:$W/clips_spec.json --name tracks --fab_ckpt $HEAD --out $W > $W/extract.log 2>&1
  python -m train.train_contact_video --eval_ckpt $VID --test $W/tracks --pred_out $W/pred --test_cams $W/cams > $W/video.log 2>&1
  python scripts/contact_clips.py export --set clip --work $W --out $G/contacts > $W/export.log 2>&1
fi
[ -f $G/contacts/${c}__0.contact.npz ] || { echo "contacts failed"; tail -n 3 $W/*.log; exit 1; }
log contacts
WA="--refine 0 --f_own 1.0 --vis_carve 1 --seg_dir $G/seg --geocalib $G/geo_sel"
[ -f $G/ours_v/recon/$c.npz ] || python -m bench.videomimic.recon world --list $L --only $c --out $G/ours_v --vipe $G/vipe_sel $WA > $G/logs/world_v.log 2>&1
MW=$G/scratch/mvs_$c; RU=""
if [ -n "$MVSR" ] && [ ! -f $G/vipe_mvs/dense/$c.npz ]; then  # link the existing workspace (symlinked files: mvs_depth never writes into it)
  src=$MVSR/$c; [ -d $src/$c/dense ] && src=$src/$c  # DIR/<name>/dense, or DIR/<name>/<name>/dense (a --work DIR/<name> run)
  rm -rf $MW; mkdir -p $MW; [ -d $src/dense ] && cp -rs $src $MW/$c; RU="--reuse 1"
fi
[ -f $G/vipe_mvs/dense/$c.npz ] || python -m bench.scene_recon.mvs_depth --vipe $G/vipe_sel --recon $G/ours_v/recon --list $L --only $c --seg_dir $G/seg \
  --out $G/vipe_mvs --work $MW $RU > $G/logs/mvs.log 2>&1
grep -h "\[mvs\]" $G/logs/mvs.log | cut -c1-160
[ -f $G/ours_m/recon/$c.npz ] || python -m bench.videomimic.recon world --list $L --only $c --out $G/ours_m --vipe $G/vipe_mvs $WA > $G/logs/world_m.log 2>&1
[ -f $G/joint/recon/$c.npz ] || python -m bench.scene_recon.joint --recon $G/ours_m/recon --vipe $G/vipe_mvs --list $L --only $c --out $G/joint \
  --contacts $G/contacts --snap 0 > $G/logs/joint.log 2>&1
[ -f $G/joint/recon/$c.npz ] || { echo "world chain failed"; tail -n 3 $G/logs/world_v.log $G/logs/mvs.log $G/logs/world_m.log $G/logs/joint.log; exit 1; }
grep -h "\[joint\]" $G/logs/joint.log | cut -c1-150
log "world chain"

# export: cameras + body in the joint-rescaled world, contacts, the scene
python - $G $c $OUT "$CK" "$FITTER" "$HEAD" "$VID" "$MVSR" <<'PY'
import hashlib, json, os, shutil, sys, numpy as np
G, c, X, ck, fitter, head, vid, mvsr = sys.argv[1:9]
md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
models = {k: {"path": os.path.realpath(f), "md5": md5(os.path.realpath(f))} for k, f in
          (("body", ck), ("smpl_fitter", fitter), ("contact_head", head), ("contact_video", vid))}
r = np.load(f"{G}/joint/recon/{c}.npz", allow_pickle=True)
fr = json.load(open(f"{G}/focal/{c}.json"))
np.savez(f"{X}/{c}.npz", frames=np.asarray(r["frames"]), T_world_cam=r["T_world_cam"].astype(np.float64), gravity_world=np.array([0.0, 0.0, -1.0]),
         K=r["K"], fps=r["fps"], pose=r["pose"], betas=r["betas"], trans=r["trans"], joints=r["joints"], valid=r["valid"],
         scale_fix=r["scale_fix"], joint=r["joint"], contacts=r["contacts"],
         notes=json.dumps({"pipeline": "FAB world pipeline (scripts/fab_world.sh): ViPE 1.2 + UniDepth, focal rule, GeoCalib z-up world, "
                           "FAB (the released profile + the SMPL fitter below), body smoothing (pelvis 3 / rotations 2 frames), MVS scene, "
                           "joint human-scene step (the contact image head -> causal video model below, --snap 0)", "models": models,
                           "world": "z-up, gravity -z; T_world_cam camera -> world, metric after the joint step's scale", "camera": fr["vipe"],
                           "static_camera": fr["static"], "focal": fr.get("focal", fr.get("fx_vipe")),
                           "mvs_reuse": mvsr or None}))
for s in ("_scene.ply", "_hfield.npz"):
    if os.path.exists(f"{G}/joint/recon/{c}{s}"):
        shutil.copy(f"{G}/joint/recon/{c}{s}", f"{X}/{c}{s}")
print("[fab_world]", c, len(r["frames"]), "frames, scale_fix", float(r["scale_fix"]), "->", f"{X}/{c}.npz")
hf = f"{G}/joint/recon/{c}_hfield.npz"
hv = float(np.load(hf)["valid"].mean()) if os.path.exists(hf) else 0.0
if json.loads(str(r["joint"])).get("skipped") == "empty scene mesh" or hv == 0.0:  # docs/world.md, Outputs
    print(f"[fab_world] WARNING {c}: no scene was reconstructed (empty scene mesh, heightfield valid {hv:.2f}): no "
          f"{c}_scene.ply, the contact labels are measured against z = 0 and fab-retarget's reference can float. Usual "
          "causes: the background masked out (a plain backdrop taken for sky), beyond the 8 m fusion range (a distant "
          "subject, a long lens) or unusable depth (docs/troubleshooting.md)", flush=True)
lab = json.loads(str(r["contacts"]))
if not any(np.any(v) for v in lab.values()):
    print(f"[fab_world] WARNING {c}: no contact label on any frame ({', '.join(lab)})", flush=True)
PY
log "fab_world done"
