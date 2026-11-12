#!/usr/bin/env bash
# Referring segmentation evaluation: cIoU (RefCOCO/+/g, Ref-ZOM) and gIoU (ReasonSeg).
#
#   MODEL_PATH=... EVAL_ROOT=... bash evaluation/scripts/eval_refseg.sh \
#       RefCOCO_seg RefCOCO+_seg RefCOCOg_seg ReasonSeg_seg_val ReasonSeg_seg_test Ref-ZOM_seg_test
#
# Annotations are read from $EVAL_ROOT/_annotations/refseg/<DATASET>.jsonl
# and are created by evaluation/prepare/refseg/prepare_refseg.py.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
(( $# > 0 )) || { sed -n '2,8p' "$0"; exit 2; }

for dataset in "$@"; do
  style=single
  [[ "$dataset" == Ref-ZOM* ]] && style=instances
  out="$OUTPUT_ROOT/$dataset"
  mkdir -p "$out"
  run_sharded evaluation/inference_refseg.py \
    "$EVAL_ROOT/_annotations/refseg/$dataset.jsonl" "$out/answer.jsonl" \
    --model_path "$MODEL_PATH" --image_root_dir "$EVAL_ROOT" \
    --prompt_style "$style" --max_new_tokens 128 \
    --min_pixels $((16 * 32 * 32)) --max_pixels $((2025 * 32 * 32))
  python3 evaluation/metrics/refseg_metric.py \
    --data_path "$out/answer.jsonl" --dataset "$dataset" --output_path "$out/eval_results.json"
done
