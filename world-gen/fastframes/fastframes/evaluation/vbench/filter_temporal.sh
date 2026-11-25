#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# RAFT static filter for the 75 temporal-flickering prompts, one shard per GPU.
#
#   bash filter_temporal.sh RUN_DIR
#
# Reads RUN_DIR/raw_videos and writes RUN_DIR/filtered_temporal,
# RUN_DIR/filter_shards/, and RUN_DIR/temporal_filter_contract.json.
# Environment: VBENCH_PYTHON (default: python), VBENCH_CACHE_DIR (metric
# weights; models/download_models.py fastframes:3 populates
# models/fastframes/vbench), GPUS (default: "0 1 2 3 4 5 6 7").
set -euo pipefail

RUN_DIR="$(cd "${1:?usage: $0 RUN_DIR}" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${VBENCH_PYTHON:-python}"
"$PY" "$HERE/setup_scorer.py" --verify  # the scorer must be the paper's (build it with setup_env.sh)
: "${VBENCH_CACHE_DIR:?set VBENCH_CACHE_DIR to the metric weight directory}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS="${#GPU_LIST[@]}"
RAFT_MODEL="$VBENCH_CACHE_DIR/raft_model/models/raft-things.pth"
[ -s "$RAFT_MODEL" ] || { echo "missing RAFT model: $RAFT_MODEL" >&2; exit 2; }
mkdir -p "$RUN_DIR/filtered_temporal" "$RUN_DIR/filter_shards" "$RUN_DIR/logs"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$shard]}" "$PY" "$HERE/filter_temporal.py" \
    --raw-dir "$RUN_DIR/raw_videos" --filtered-dir "$RUN_DIR/filtered_temporal" \
    --model "$RAFT_MODEL" --shard "$shard" --num-shards "$NUM_SHARDS" \
    --shard-output "$RUN_DIR/filter_shards/shard_${shard}.json" \
    > "$RUN_DIR/logs/filter_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] || { echo "static filtering failed; see $RUN_DIR/logs/filter_shard_*.log" >&2; exit 1; }

"$PY" "$HERE/filter_temporal.py" \
  --raw-dir "$RUN_DIR/raw_videos" --filtered-dir "$RUN_DIR/filtered_temporal" \
  --num-shards "$NUM_SHARDS" --merge-only --shard-dir "$RUN_DIR/filter_shards" \
  --output "$RUN_DIR/temporal_filter_contract.json"
