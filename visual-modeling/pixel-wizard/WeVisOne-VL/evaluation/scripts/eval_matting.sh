#!/usr/bin/env bash
# Image matting evaluation: MAD and MSE (x1000), Grad, Conn and SAD.
#
#   MODEL_PATH=... EVAL_ROOT=... bash evaluation/scripts/eval_matting.sh [DATASET ...]
#
# P3M-500-NP, AM-2K-test and PPM-100 use the bounding box of the ground-truth
# alpha as the box prompt. For WeRefMat, the model predicts the boxes from the
# text. Annotations are read from $EVAL_ROOT/_annotations/matting/<DATASET>.jsonl
# and are created by evaluation/prepare/matting/prepare_matting.py.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
(( $# > 0 )) || set -- P3M-500-NP AM-2K-test PPM-100  # WeRefMat is not yet released

for dataset in "$@"; do
  if [[ "$dataset" == WeRefMat ]]; then
    infer=(--box_source generated --max_new_tokens 512)
  else
    infer=(--box_source gt)
  fi
  out="$OUTPUT_ROOT/$dataset"
  mkdir -p "$out"
  run_sharded evaluation/inference_matting.py \
    "$EVAL_ROOT/_annotations/matting/$dataset.jsonl" "$out/answer.jsonl" \
    --model_path "$MODEL_PATH" --image_root_dir "$EVAL_ROOT" "${infer[@]}" \
    --min_pixels $((16 * 32 * 32)) --max_pixels $((1024 * 1024))
  python3 evaluation/metrics/matting_metric.py \
    --data_path "$out/answer.jsonl" --output_path "$out/eval_results.json"
done
