#!/usr/bin/env bash
# LVIS v1 val detection (general detection table). Each category is queried separately.
#   MODEL_PATH=... EVAL_ROOT=... bash evaluation/scripts/eval_lvis.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LVIS_GT="${LVIS_GT:-$EVAL_ROOT/missing_annotaitons/lvis_v1_val_with_filename2.json}"
OUT="$OUTPUT_ROOT/LVIS"
mkdir -p "$OUT"

run_sharded evaluation/inference_text_prompt.py \
  "$EVAL_ROOT/_annotations/box_eval/LVIS.jsonl" "$OUT/answer.jsonl" \
  --model_path "$MODEL_PATH" --image_root_dir "$EVAL_ROOT" \
  --eval_type box_eval --prompt_mode single --max_new_tokens 4096

python3 evaluation/utils/convert_coco_lvis_to_standard_format.py \
  --pred_jsonl "$OUT/answer.jsonl" --gt_json "$LVIS_GT" --out_tsv "$OUT/predictions.tsv"
python3 evaluation/metrics/coco_lvis_metric.py \
  --gt "$LVIS_GT" --pred_tsv "$OUT/predictions.tsv" --eval_type lvis \
  --output_json "$OUT/eval_results.json"
