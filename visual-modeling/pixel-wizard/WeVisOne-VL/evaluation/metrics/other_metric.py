"""Box and point F1 for detection, referring, OCR and pointing benchmarks.

For each sample, predictions are greedily matched to ground-truth boxes of the
same category (``referring_object_detection`` ignores the category) at each IoU
threshold in 0.50:0.05:0.95. Precision and recall are averaged over samples,
and F1 is computed from the averaged values at each threshold. ``avg_f1`` is the
mean of the ten F1 values. A sample without ground truth scores 1 if it has no
prediction and 0 otherwise.

For pointing, a point matches a ground-truth instance if it falls inside its
mask (or box), and each instance can be matched by at most one point.

``rejection_rate`` is the percentage of samples without a target that have no prediction.
"""

from __future__ import annotations

import argparse
import json
from statistics import mean

import numpy as np

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
# Detection categories are compared on letters only ("traffic light" == "traffic_light").
LETTER_TASKS = {"common_object_detection", "long_tailed_object_detection", "dense_object_detection"}


def task_of(row: dict, pointing: bool) -> str:
    name = str(row.get("task_name") or "referring_object_detection")
    if pointing or "pointing" in name:
        return "pointing_referring" if "referring" in name else "pointing"
    return name


def category_key(name: str, task: str) -> str:
    name = name.lower()
    if task in LETTER_TASKS:
        return "".join(ch for ch in name if ch.isalpha())
    return name.replace("_", "").replace(" ", "")


def normalize(mapping, task: str) -> dict:
    """Lowercase, then normalize keys; a later key that collides replaces the earlier one."""
    if not isinstance(mapping, dict):
        return {}
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    return {
        category_key(key, task): (value if isinstance(value, list) else [value])
        for key, value in lowered.items()
    }


def valid(values) -> list:
    return [v for v in values if v is not None and v != "None"]


def iou(a, b) -> float:
    """IoU of the first four numbers of each shape (xyxy)."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def greedy_matches(gt_boxes, pred_boxes, threshold) -> int:
    used, matches = set(), 0
    for gt_box in gt_boxes:
        best, best_index = 0.0, -1
        for index, pred_box in enumerate(pred_boxes):
            if index in used:
                continue
            value = iou(gt_box, pred_box)
            if value >= threshold and value > best:
                best, best_index = value, index
        if best_index >= 0:
            used.add(best_index)
            matches += 1
    return matches


class BoxSample:
    def __init__(self, row: dict, task: str) -> None:
        gt = normalize(row.get("gt"), task)
        pred = normalize(row.get("extracted_predictions"), task)
        self.gt_boxes = [b for v in gt.values() for b in v]
        if task == "referring_object_detection":
            self.candidates = [b for v in pred.values() for b in v]
            self.num_pred = len(self.candidates)
            self.no_target = not self.gt_boxes
        else:
            # Boxes of categories not in the ground truth are counted as false positives.
            self.candidates = [b for c in gt for b in valid(pred.get(c, []))]
            self.num_pred = sum(len(valid(v)) for v in pred.values())
            # A ground truth with categories but no boxes is not treated as "no target".
            self.no_target = not gt

    def scores(self, threshold: float) -> tuple[float, float]:
        if self.no_target:
            return (1.0, 1.0) if self.num_pred == 0 else (0.0, 0.0)
        if self.num_pred == 0 or not self.gt_boxes:
            return 0.0, 0.0
        matched = greedy_matches(self.gt_boxes, self.candidates, threshold)
        return matched / self.num_pred, matched / len(self.gt_boxes)


class PointSample:
    def __init__(self, row: dict, task: str) -> None:
        gt = {str(k).lower(): v for k, v in (row.get("gt") or {}).items()}
        pred = {str(k).lower(): v for k, v in (row.get("extracted_predictions") or {}).items()}
        if task == "pointing_referring" and gt and pred:
            pred = {next(iter(gt)): next(iter(pred.values()))}
        targets = [
            (category, point_target(mask))
            for category, masks in gt.items()
            if isinstance(masks, list)
            for mask in masks
        ]
        points = [
            (category, (float(p[0]), float(p[1])))
            for category, values in pred.items()
            if isinstance(values, list)
            for p in values
            if isinstance(p, (list, tuple)) and len(p) == 2
        ]
        self.gt_boxes = targets
        self.num_pred = len(points)
        if not targets:
            self.fixed = (1.0, 1.0) if not points else (0.0, 0.0)
        elif not points:
            self.fixed = (0.0, 0.0)
        else:
            used, matches = set(), 0
            for category, target in targets:
                for index, (pred_category, point) in enumerate(points):
                    if index not in used and pred_category == category and point_hits(point, target):
                        used.add(index)
                        matches += 1
                        break
            self.fixed = (matches / len(points), matches / len(targets))

    def scores(self, threshold: float) -> tuple[float, float]:
        return self.fixed


def point_target(value):
    if isinstance(value, dict) and "counts" in value and "size" in value:
        from pycocotools import mask as coco_mask

        rle = {"counts": value["counts"], "size": value["size"]}
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        mask = np.asarray(coco_mask.decode(rle))
        return ("mask", mask.max(axis=2) if mask.ndim == 3 else mask)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return ("box", [float(v) for v in value])
    raise ValueError(f"pointing GT must be an RLE mask or a box, got {value!r}")


def point_hits(point, target) -> bool:
    kind, value = target
    if kind == "box":
        return value[0] <= point[0] <= value[2] and value[1] <= point[1] <= value[3]
    x, y = int(point[0]), int(point[1])
    return 0 <= y < value.shape[0] and 0 <= x < value.shape[1] and bool(value[y, x] >= 0.5)


def f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def evaluate(rows: list[dict], pointing: bool = False) -> dict:
    tasks = {task_of(row, pointing) for row in rows}
    if len(tasks) != 1:
        raise ValueError(f"one answer file must hold one task, got {sorted(tasks)}")
    task = tasks.pop()
    is_point = task.startswith("pointing")
    samples = [(PointSample if is_point else BoxSample)(row, task) for row in rows]
    thresholds = (0.5,) if is_point else IOU_THRESHOLDS
    curve = []
    for threshold in thresholds:
        scores = [sample.scores(threshold) for sample in samples]
        precision = mean(p for p, _ in scores)
        recall = mean(r for _, r in scores)
        curve.append({"iou": threshold, "precision": precision, "recall": recall, "f1": f1(precision, recall)})

    result = {
        "task_name": task,
        "num_samples": len(rows),
        "avg_f1": mean(item["f1"] for item in curve),
        "precision@0.50": curve[0]["precision"],
        "recall@0.50": curve[0]["recall"],
        "f1@0.50": curve[0]["f1"],
    }
    if not is_point:
        result.update(
            {
                "precision@0.95": curve[-1]["precision"],
                "recall@0.95": curve[-1]["recall"],
                "f1@0.95": curve[-1]["f1"],
                "per_iou": curve,
            }
        )
    absent = [sample for sample in samples if not sample.gt_boxes]
    if absent:
        result["num_absent_target"] = len(absent)
        result["rejection_rate"] = sum(sample.num_pred == 0 for sample in absent) / len(absent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="answer.jsonl")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pointing", action="store_true", help="score point_eval answers")
    args = parser.parse_args()
    with open(args.data_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = evaluate(rows, pointing=args.pointing)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "per_iou"}, indent=2))


if __name__ == "__main__":
    main()
