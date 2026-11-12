#!/usr/bin/env bash
# COCO val2017 detection (general detection table). Each category is queried separately.
#   MODEL_PATH=... EVAL_ROOT=... bash evaluation/scripts/eval_coco.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

COCO_GT="${COCO_GT:-$EVAL_ROOT/coco/instances_val2017.json}"
OUT="$OUTPUT_ROOT/COCO"
mkdir -p "$OUT"

run_sharded evaluation/inference_text_prompt.py \
  "$EVAL_ROOT/_annotations/box_eval/COCO.jsonl" "$OUT/answer.jsonl" \
  --model_path "$MODEL_PATH" --image_root_dir "$EVAL_ROOT" \
  --eval_type box_eval --prompt_mode single --max_new_tokens 4096

python3 evaluation/utils/convert_coco_lvis_to_standard_format.py \
  --pred_jsonl "$OUT/answer.jsonl" --gt_json "$COCO_GT" --out_tsv "$OUT/predictions.tsv"
python3 evaluation/metrics/coco_lvis_metric.py \
  --gt "$COCO_GT" --pred_tsv "$OUT/predictions.tsv" --eval_type coco \
  --output_json "$OUT/eval_results.json"
