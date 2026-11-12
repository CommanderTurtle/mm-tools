"""Box F1 for COCO and LVIS, computed from a FastEval TSV file.

Precision and recall are averaged over categories (non-positive entries are
skipped), and F1 is computed from the averaged values. Results are reported at
IoU 0.50 and 0.95, and averaged over IoU 0.50:0.05:0.95.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation import fastevaluate  # noqa: E402


def safe_mean(values) -> float:
    kept = [value for value in values if value > 0]
    return float(np.mean(kept)) if kept else 0.0


def f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total > 0 else 0.0


def summarize(per_category: dict) -> dict:
    result = {}
    for suffix, key in (("", ""), ("@0.50", "50"), ("@0.95", "95")):
        precision = safe_mean(per_category[f"precision{key}"])
        recall = safe_mean(per_category[f"recall{key}"])
        name = "avg_" if not suffix else ""
        result[f"{name}precision{suffix}"] = precision
        result[f"{name}recall{suffix}"] = recall
        result[f"{name}f1{suffix}"] = f1(precision, recall)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="COCO or LVIS ground-truth JSON")
    parser.add_argument("--pred_tsv", required=True)
    parser.add_argument("--eval_type", choices=("coco", "lvis"), required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    metrics = {
        "eval_type": args.eval_type,
        **summarize(fastevaluate.evaluate(args.gt, args.pred_tsv, args.eval_type)),
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
