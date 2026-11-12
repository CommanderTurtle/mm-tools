#!/usr/bin/env bash
# Evaluation of all text-prompted box and point benchmarks except COCO and LVIS.
#
#   MODEL_PATH=... EVAL_ROOT=... bash evaluation/scripts/eval_others.sh DATASET [DATASET ...]
#
# DATASET is the name of an annotation file in $EVAL_ROOT/_annotations/box_eval/, e.g.
#   Dense200 VisDrone                                   dense detection
#   RefCOCOg_val RefCOCOg_test HumanRef                 referring detection
#   Ref-Adv-s PR-Bench DVGBench Ref-L4-all RefCOCOg-rej-val
#   HierText IC15 TotalText SROIE                       OCR
#   CountQA pixmo-count-val pixmo-count-test cloc-val cloc-test
#   KubriCount-testA KubriCount-testB MixCount-val MixCount-test
# For pointing evaluation, add the suffix "_point" to use
# $EVAL_ROOT/_annotations/point_eval/<name>.jsonl (COCO_point LVIS_point
# Dense200_point VisDrone_point HumanRef_point RefCOCOg_val_point
# RefCOCOg_test_point). PointBench and PixMo-Points-Eval are also read from
# point_eval/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
(( $# > 0 )) || { sed -n '2,16p' "$0"; exit 2; }

DEFAULT_PIXELS=(--min_pixels $((16 * 32 * 32)) --max_pixels $((2025 * 32 * 32)))
TABLE5_PIXELS=(--min_pixels $((16 * 28 * 28)) --max_pixels $((2560 * 28 * 28)))
COUNT_PIXELS=(--min_pixels 16384 --max_pixels 2073600)

for name in "$@"; do
  eval_type=box_eval
  dataset="$name"
  if [[ "$name" == *_point ]]; then
    eval_type=point_eval
    dataset="${name%_point}"
  fi
  metric=(python3 evaluation/metrics/other_metric.py)
  case "$dataset" in
    PointBench|PixMo-Points-Eval)
      eval_type=point_eval
      infer=(--prompt_style question --max_new_tokens 1024 "${COUNT_PIXELS[@]}")
      metric=(python3 evaluation/metrics/pointing_bench_metric.py
              --annotation_path "$EVAL_ROOT/_annotations/point_eval/$dataset.jsonl")
      ;;
    CountQA|pixmo-count-*|cloc-*|KubriCount-*|MixCount-*)
      infer=(--max_new_tokens 4096 "${COUNT_PIXELS[@]}")
      metric=(python3 evaluation/metrics/counting_metric.py)
      ;;
    Ref-Adv-s|PR-Bench|DVGBench|Ref-L4-all|RefCOCOg-rej-val)
      infer=(--max_new_tokens 512 "${TABLE5_PIXELS[@]}")
      ;;
    *)
      if [[ "$eval_type" == point_eval ]]; then
        infer=(--max_new_tokens 4096 "${COUNT_PIXELS[@]}")
        metric+=(--pointing)
      elif [[ "$dataset" == Dense200 || "$dataset" == VisDrone ]]; then
        infer=(--prompt_mode single --max_new_tokens 4096 "${DEFAULT_PIXELS[@]}")
      else
        infer=(--max_new_tokens 4096 "${DEFAULT_PIXELS[@]}")
      fi
      ;;
  esac

  out="$OUTPUT_ROOT/$name"
  mkdir -p "$out"
  run_sharded evaluation/inference_text_prompt.py \
    "$EVAL_ROOT/_annotations/$eval_type/$dataset.jsonl" "$out/answer.jsonl" \
    --model_path "$MODEL_PATH" --image_root_dir "$EVAL_ROOT" --eval_type "$eval_type" \
    "${infer[@]}"
  "${metric[@]}" --data_path "$out/answer.jsonl" --output_path "$out/eval_results.json"
done
