#!/usr/bin/env bash
# Common settings and helper functions used by all eval_*.sh scripts.
#
#   MODEL_PATH   Path to the model checkpoint (required).
#   EVAL_ROOT    Root directory of the evaluation data, which contains
#                _annotations/{box_eval,point_eval,refseg,matting}/<DATASET>.jsonl.
#                Image and alpha paths in the annotations are relative to it.
#   OUTPUT_ROOT  Output directory. Results are saved to
#                $OUTPUT_ROOT/<DATASET>/{answer.jsonl,eval_results.json}.
#   NUM_GPUS     Number of GPUs to use (0 means all visible GPUs).
#   PY           Python interpreter used for inference (default: python3).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
: "${MODEL_PATH:?set MODEL_PATH to the checkpoint directory}"
EVAL_ROOT="${EVAL_ROOT:-data/eval_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation/output/$(basename "${MODEL_PATH%/}")}"
PY="${PY:-python3}"

visible_gpus() {
  local ids
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ids="${CUDA_VISIBLE_DEVICES//,/ }"
  else
    ids="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  fi
  [[ -n "${ids// /}" ]] || ids="0"
  local -a gpus=($ids)
  if [[ "${NUM_GPUS:-0}" -gt 0 && "${NUM_GPUS}" -lt "${#gpus[@]}" ]]; then
    gpus=("${gpus[@]:0:${NUM_GPUS}}")
  fi
  echo "${gpus[@]}"
}

# run_sharded INFER_SCRIPT ANNOTATION ANSWER_JSONL [inference args...]
# Splits ANNOTATION into one contiguous part per GPU, runs INFER_SCRIPT on each
# part, and merges the results into ANSWER_JSONL in the original order.
run_sharded() {
  local script="$1" annotation="$2" answer="$3"
  shift 3
  [[ -f "$annotation" ]] || { echo "Annotation not found: $annotation" >&2; return 2; }
  local total
  total=$(grep -c . "$annotation")
  local -a gpus=($(visible_gpus))
  local workers=${#gpus[@]}
  (( workers > total )) && workers=$total
  (( workers < 1 )) && workers=1
  local chunk=$(( (total + workers - 1) / workers ))
  local shard_dir="${answer%.jsonl}_shards"
  rm -rf "$shard_dir" && mkdir -p "$shard_dir"

  echo "Inference: $total rows on $workers GPU(s) -> $answer"
  local -a pids=() shards=()
  local i start end shard
  for (( i = 0; i < workers; i++ )); do
    start=$(( i * chunk ))
    end=$(( start + chunk < total ? start + chunk : total ))
    (( start >= end )) && break
    shard="${shard_dir}/$(printf '%03d' "$i").jsonl"
    shards+=("$shard")
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$PY" "$script" \
      --test_jsonl_path "$annotation" --save_path "$shard" \
      --start_idx "$start" --end_idx "$end" "$@" \
      > "${shard%.jsonl}.log" 2>&1 &
    pids+=($!)
  done

  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Worker $i failed; see ${shards[$i]%.jsonl}.log" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || return 1
  cat "${shards[@]}" > "$answer"
  local merged
  merged=$(grep -c . "$answer")
  [[ "$merged" -eq "$total" ]] || {
    echo "Merged $merged rows, expected $total: $answer" >&2
    return 1
  }
  rm -rf "$shard_dir"
}
