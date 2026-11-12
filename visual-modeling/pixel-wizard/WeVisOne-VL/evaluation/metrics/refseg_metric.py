"""cIoU and mIoU (gIoU) for RefCOCO, RefCOCO+, RefCOCOg, ReasonSeg and Ref-ZOM.

``inference_refseg.py`` saves the ``intersection``, ``union`` and ``iou`` of each
expression. An empty prediction for an empty target has an IoU of 1. The
metrics are computed as follows:

* cIoU = 100 * sum(I) / sum(U), mIoU = 100 * mean(iou), and Pr@t is the
  percentage of expressions with iou >= t.
* ReasonSeg reports mIoU as gIoU. The test split is also reported separately
  for short and long queries.
* Ref-ZOM computes cIoU and mIoU on expressions with targets (``source_type``
  "one" or "many"). Acc is the percentage of zero-target expressions with an
  empty prediction.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    intersection = sum(int(row.get("intersection", 0)) for row in rows)
    union = sum(int(row.get("union", 0)) for row in rows)
    ious = [float(row.get("iou", 0.0)) for row in rows]
    result = {
        "num_samples": n,
        "num_errors": sum(bool(row.get("error")) for row in rows),
        "cIoU": 100.0 * intersection / union if union else 0.0,
        "mIoU": 100.0 * sum(ious) / n if n else 0.0,
    }
    for t in THRESHOLDS:
        result[f"Pr@{t:.1f}"] = 100.0 * sum(iou >= t for iou in ious) / n if n else 0.0
    return result


def evaluate(rows: list[dict], dataset: str) -> dict:
    splits = defaultdict(list)
    for row in rows:
        splits[str(row.get("split", "unknown"))].append(row)
    result = {"dataset": dataset, **summarize(rows)}
    result["splits"] = {name: summarize(items) for name, items in splits.items()}

    if dataset.startswith("ReasonSeg"):
        result["gIoU"] = result["mIoU"]
        for split in result["splits"].values():
            split["gIoU"] = split["mIoU"]
        test = splits.get("test", [])
        if test:
            result["test_by_query_type"] = {
                kind: summarize([row for row in test if row.get("query_type") == kind])
                for kind in ("short", "long")
            }
    elif dataset.startswith("Ref-ZOM"):
        positives = [row for row in rows if row.get("source_type") != "zero"]
        zeros = [row for row in rows if row.get("source_type") == "zero"]
        correct = sum(not row.get("error") and int(row.get("pred_area", 0)) == 0 for row in zeros)
        result.update(
            {
                **summarize(positives),
                "num_zero_target": len(zeros),
                "Acc": 100.0 * correct / len(zeros) if zeros else 0.0,
                "groups": {
                    kind: summarize([row for row in positives if row.get("source_type") == kind])
                    for kind in ("one", "many")
                },
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="answer.jsonl from inference_refseg.py")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    with open(args.data_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = evaluate(rows, args.dataset)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("splits", "groups")}, indent=2))


if __name__ == "__main__":
    main()
