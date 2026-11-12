"""Counting metrics, where the predicted count is the number of predicted boxes.

The predicted count is the number of boxes with four coordinates in
``extracted_predictions``, regardless of category. If there are none, the valid
``<box>`` tags in the raw response are counted instead, so that truncated or
unlabeled answers are still counted. The ground-truth count is taken from
``gt_count``, ``gt["count"]`` or the number of ground-truth boxes.

EM (the metric reported in the paper) is the percentage of queries whose
predicted count equals the ground truth. RA@10 accepts ``|pred - gt| <= 0.1 * gt``.
"""

from __future__ import annotations

import argparse
import json
import math
import re

RAW_GEOM_RE = re.compile(r"<(point|box)>(.*?)(?:</\1>|$)", re.IGNORECASE | re.DOTALL)
COORD_RE = re.compile(r"<\s*([+-]?\d+(?:\.\d+)?)\s*>")


def raw_box_count(text: str) -> int:
    count = 0
    for match in RAW_GEOM_RE.finditer(text or ""):
        if match.group(1).lower() != "box":
            continue
        for segment in match.group(2).split(","):
            coords = [float(v) for v in COORD_RE.findall(segment)]
            for start in range(0, len(coords) - 3, 4):
                x0, y0, x1, y1 = coords[start : start + 4]
                count += x1 > x0 and y1 > y0
    return count


def predicted_count(row: dict) -> int:
    extracted = row.get("extracted_predictions")
    count = 0
    if isinstance(extracted, dict):
        count = sum(
            1
            for values in extracted.values()
            if isinstance(values, list)
            for value in values
            if isinstance(value, (list, tuple)) and len(value) == 4
        )
    if count == 0:
        raw = row.get("raw_response")
        if isinstance(raw, dict):
            raw = "\n".join(str(v) for v in raw.values())
        count = raw_box_count(str(raw or ""))
    return count


def gt_count(row: dict) -> int | None:
    if row.get("gt_count") is not None:
        return int(row["gt_count"])
    gt = row.get("gt")
    if not isinstance(gt, dict) or not gt:
        return None
    if "count" in gt:
        return int(gt["count"])
    if all(isinstance(items, list) for items in gt.values()):
        return sum(
            isinstance(item, (list, tuple)) and len(item) in (2, 4)
            for items in gt.values()
            for item in items
        )
    return None


def evaluate(rows: list[dict]) -> dict:
    pairs = [(gt, predicted_count(row)) for row in rows if (gt := gt_count(row)) is not None]
    n = max(len(pairs), 1)
    errors = [abs(pred - gt) for gt, pred in pairs]
    return {
        "dataset_name": rows[0].get("dataset_name") if rows else None,
        "num_samples": len(rows),
        "num_scored": len(pairs),
        "EM": 100.0 * sum(pred == gt for gt, pred in pairs) / n,
        "RA@10": 100.0 * sum(abs(pred - gt) <= 0.1 * gt for gt, pred in pairs) / n,
        "MAE": sum(errors) / n,
        "RMSE": math.sqrt(sum(e * e for e in errors) / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="answer.jsonl")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    with open(args.data_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = evaluate(rows)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
