#!/bin/bash
# TensorRT engines of a FAB model for THIS GPU (fab-build-trt): the body at batch 1 and at batch 2..4 (FABPredictor's
# body_batch "auto" runs frames with more than one person on them), BF16 "fullbb_all" (fab/trt_build.py), and the person
# detector (FP16, kept only if its boxes agree with the PyTorch detector on your images). Everything goes to
# $FAB_HOME/engines/<gpu>/ (fab.paths.engines_dir), where FABPredictor(fast=True) picks it up. Idempotent: finished steps
# are skipped; the log is <engines dir>/build.log.
#   fab-build-trt --images photos/ [--model fab-l|fab-s] [--batches "2 3 4"] [--mix 1] [--no_detector]
set -uo pipefail
MODEL=fab-l; IMAGES=${FAB_ENGINE_IMAGES:-}; BATCHES="2 3 4"; MIX=1; DETECTOR=1
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2;; --images) IMAGES=$2; shift 2;; --batches) BATCHES=$2; shift 2;; --mix) MIX=$2; shift 2;;
    --no_detector) DETECTOR=0; shift;; -h|--help) sed -n 2,7p "$0"; exit 0;; *) echo "unknown argument $1"; exit 1;;
  esac
done
[ -n "$IMAGES" ] && [ -d "$IMAGES" ] || { echo "pass --images DIR: a folder of photos with people (the agreement checks run on them)"; exit 1; }
IMAGES=$(cd "$IMAGES" && pwd)
cd "$(dirname "$0")/.."
export PYTHONPATH=.:${PYTHONPATH:-} YOLO_AUTOINSTALL=False  # the Ultralytics export must not install packages on its own
python -c 'import tensorrt, onnx, onnxslim, onnxruntime, onnxscript' 2>/dev/null || {
  echo "the engine extra is missing: pip install -e \".[engine]\""; exit 1; }
python - <<'PY' || exit 1
import importlib.metadata as m, re, sys
txt = open("pyproject.toml").read()
block = re.search(r"^engine = \[(.*?)\]", txt, re.S | re.M).group(1)
want = dict(re.findall(r'"([A-Za-z0-9_.\-]+)==([^"]+)"', block))
def ver(k):
    try:
        return m.version(k)
    except m.PackageNotFoundError:
        return None
bad = {k: (ver(k), v) for k, v in want.items() if ver(k) != v}
if bad:
    print("the [engine] extra's pins are not matched (installed, wanted):", bad); sys.exit(1)
PY
ERR=$(mktemp)
INFO=$(python -m fab.trt_build info --model $MODEL --mix $MIX 2>"$ERR" | tail -n 1)
[ -n "$INFO" ] || { echo "could not load $MODEL (README.md, Install; docs/troubleshooting.md):"; tail -n 5 "$ERR"; rm -f "$ERR"; exit 1; }
rm -f "$ERR"
get() { python -c "import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$INFO" "$1"; }
E=$(get engines_dir); SEG=$(get seg); M=$(get mix)
mkdir -p "$E"; LOG=$(dirname "$E")/build.log
st() { echo "$(date +%FT%T%z) $*" | tee -a "$LOG"; }
FAILED=0
st "$MODEL ($(get weights)) -> $E, 2.5D mix $M, export flags '$SEG', batches 1 $BATCHES"
for N in 1 $BATCHES; do
  suf=$([ $N -gt 1 ] && echo -b$N); W=$E/work/m$M$suf; ENG=$E/body-m$M$suf.engine; mkdir -p "$W"
  if [ -f "$ENG" ] && [ -f "$ENG.json" ]; then st "batch $N: $ENG (exists)"; continue; fi
  [ -f "$W/agreement.json" ] || python -m fab.trt_build export --model $MODEL --mix $MIX --images "$IMAGES" $SEG --batch $N \
    --out "$W/body.onnx" >> "$LOG" 2>&1
  python -m fab.trt_build build --onnx "$W/body.onnx" --out "$ENG" --prec bf16 --scope fullbb_all >> "$LOG" 2>&1
  if [ -f "$ENG" ]; then st "batch $N: $ENG"; else FAILED=1; st "batch $N: build FAILED, see $LOG (FABPredictor runs these people on the next smaller engine / PyTorch)"; fi
done
if [ "$DETECTOR" = 1 ]; then
  D=$(dirname "$E")/detector.engine
  [ -f "$D" ] || python -m fab.trt_build detector --images "$IMAGES" >> "$LOG" 2>&1
  st "detector: $([ -f "$D" ] && echo "$D" || echo 'not built (the PyTorch detector stays)')"
fi
exit $FAILED
