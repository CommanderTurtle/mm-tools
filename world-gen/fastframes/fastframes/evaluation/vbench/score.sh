#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Verify, filter, and score one generated corpus with all 16 VBench dimensions.
#
#   bash score.sh RUN_DIR [MODEL_NAME]
#
# RUN_DIR/raw_videos must hold the 6,220 protocol MP4s (see protocol.py).
# Outputs: RUN_DIR/{raw,filtered}_corpus_verification.json,
# RUN_DIR/temporal_filter_contract.json, RUN_DIR/vbench_results/, and
# RUN_DIR/score_summary.json.
# Environment: VBENCH_PYTHON (default: python), VBENCH_CACHE_DIR (metric
# weights, see download_models.sh), GPUS (default: "0 1 2 3 4 5 6 7"; the
# paper scores used 8 ranks), TORCH_HOME (default: $VBENCH_CACHE_DIR/torch).
set -euo pipefail

RUN_DIR="$(cd "${1:?usage: $0 RUN_DIR [MODEL_NAME]}" && pwd)"
MODEL="${2:-$(basename "$RUN_DIR")}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VBENCH_ROOT="$HERE/VBench"
PY="${VBENCH_PYTHON:-python}"
"$PY" "$HERE/setup_scorer.py" --verify  # the scorer must be the paper's (build it with setup_env.sh)
: "${VBENCH_CACHE_DIR:?set VBENCH_CACHE_DIR to the metric weight directory}"
export VBENCH_CACHE_DIR="$(cd "$VBENCH_CACHE_DIR" && pwd)"
export TORCH_HOME="${TORCH_HOME:-$VBENCH_CACHE_DIR/torch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NUM_GPUS="${#GPU_LIST[@]}"
VISIBLE="$(IFS=,; echo "${GPU_LIST[*]}")"
RESULTS="$RUN_DIR/vbench_results"
NON_TEMPORAL_DIMS=(subject_consistency background_consistency motion_smoothness dynamic_degree
  aesthetic_quality imaging_quality object_class multiple_objects color spatial_relationship
  scene temporal_style overall_consistency human_action appearance_style)

[ -s "$TORCH_HOME/hub/checkpoints/dino_vitbase16_pretrain.pth" ] \
  || { echo "missing DINO weights under $TORCH_HOME (run download_models.sh)" >&2; exit 2; }
[ -s "$VBENCH_CACHE_DIR/bert-base-uncased/config.json" ] \
  || { echo "missing $VBENCH_CACHE_DIR/bert-base-uncased (run download_models.sh)" >&2; exit 2; }
mkdir -p "$RESULTS/non_temporal" "$RESULTS/temporal_flickering" "$RUN_DIR/logs"

"$PY" "$HERE/protocol.py" verify --mode raw \
  --videos-dir "$RUN_DIR/raw_videos" --output "$RUN_DIR/raw_corpus_verification.json"
[ -s "$RUN_DIR/temporal_filter_contract.json" ] || bash "$HERE/filter_temporal.sh" "$RUN_DIR"
"$PY" "$HERE/protocol.py" verify --mode filtered_temporal \
  --videos-dir "$RUN_DIR/filtered_temporal" --filter-contract "$RUN_DIR/temporal_filter_contract.json" \
  --output "$RUN_DIR/filtered_corpus_verification.json"

# Run one VBench stage; reuse a single existing result, refuse ambiguity.
run_stage() {
  local videos="$1" out="$2" stage="$3"; shift 3
  local count
  count="$(find "$out" -maxdepth 1 -name '*_eval_results.json' | wc -l)"
  if [ "$count" -eq 1 ]; then echo "[score] reusing existing $stage result"; return; fi
  [ "$count" -eq 0 ] || { echo "$stage: $count result files in $out" >&2; exit 1; }
  # Scoring runs from the weight directory so that GRiT/Tag2Text resolve
  # the local bert-base-uncased snapshot.
  (cd "$VBENCH_CACHE_DIR" && CUDA_VISIBLE_DEVICES="$VISIBLE" "$PY" -m torch.distributed.run \
    --standalone --nproc_per_node="$NUM_GPUS" "$VBENCH_ROOT/evaluate.py" \
    --videos_path "$videos" --dimension "$@" --mode vbench_standard \
    --load_ckpt_from_local True --output_path "$out") 2>&1 | tee "$RUN_DIR/logs/score_${stage}.log"
  count="$(find "$out" -maxdepth 1 -name '*_eval_results.json' | wc -l)"
  [ "$count" -eq 1 ] || { echo "$stage produced $count result files" >&2; exit 1; }
}

run_stage "$RUN_DIR/raw_videos" "$RESULTS/non_temporal" non_temporal "${NON_TEMPORAL_DIMS[@]}"
run_stage "$RUN_DIR/filtered_temporal" "$RESULTS/temporal_flickering" temporal_flickering temporal_flickering

"$PY" "$HERE/summarize.py" --model "$MODEL" \
  --non-temporal "$(find "$RESULTS/non_temporal" -maxdepth 1 -name '*_eval_results.json')" \
  --temporal "$(find "$RESULTS/temporal_flickering" -maxdepth 1 -name '*_eval_results.json')" \
  --filter-contract "$RUN_DIR/temporal_filter_contract.json" \
  --output "$RUN_DIR/score_summary.json"
