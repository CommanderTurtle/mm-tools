"""F1@Point for PointBench and PixMo-Points-Eval.

Only annotations with ``evaluation_protocol == "point_mask_f1"`` are scored, and
each of them contains one RLE mask per target. Masks are processed in
annotation order, and each mask is matched to the first unused predicted point
inside it. A point is mapped to mask pixels as ``int(x * mask_w / image_w)``.
Precision and recall are computed per query and averaged over queries, and F1
is computed from the two averages. A query without targets scores 1 only if no
point is predicted. A missing prediction is treated as predicting no points.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from statistics import mean

import numpy as np
from pycocotools import mask as coco_mask

PROTOCOL = "point_mask_f1"


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode(rle: dict) -> np.ndarray:
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    mask = coco_mask.decode(rle)
    return np.asarray(mask.max(axis=2) if mask.ndim == 3 else mask, dtype=bool)


def points_of(row: dict) -> list[tuple[float, float]]:
    extracted = row.get("extracted_predictions") or {}
    points = []
    for values in extracted.values() if isinstance(extracted, dict) else []:
        for value in values if isinstance(values, list) else []:
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)
            ):
                points.append((float(value[0]), float(value[1])))
    return points


def score(annotation: dict, points: list) -> tuple[float, float]:
    masks = [decode(rle) for rles in (annotation.get("gt_mask") or {}).values() for rle in rles]
    if int(annotation.get("gt_count", len(masks))) != len(masks):
        raise ValueError(f"{annotation['sample_id']}: need one mask per target")
    if not masks:
        return (1.0, 1.0) if not points else (0.0, 0.0)
    if not points:
        return 0.0, 0.0
    image_w = max(1, int(annotation["image_width"]))
    image_h = max(1, int(annotation["image_height"]))
    used, matches = set(), 0
    for mask in masks:
        mask_h, mask_w = mask.shape
        for index, (x, y) in enumerate(points):
            if index in used:
                continue
            px, py = int(x * mask_w / image_w), int(y * mask_h / image_h)
            if 0 <= px < mask_w and 0 <= py < mask_h and mask[py, px]:
                used.add(index)
                matches += 1
                break
    return matches / len(points), matches / len(masks)


def summarize(scores: list[tuple[float, float]]) -> dict:
    precision = mean(p for p, _ in scores) if scores else 0.0
    recall = mean(r for _, r in scores) if scores else 0.0
    total = precision + recall
    return {
        "num_samples": len(scores),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / total if total else 0.0,
    }


def evaluate(predictions: list[dict], annotations: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in predictions}
    scored = [a for a in annotations if a.get("evaluation_protocol", PROTOCOL) == PROTOCOL]
    slices = defaultdict(list)
    scores = []
    for annotation in scored:
        result = score(annotation, points_of(by_id.get(annotation["sample_id"], {})))
        scores.append(result)
        slices[str(annotation.get("benchmark_slice"))].append(result)
    return {
        "dataset_name": annotations[0].get("dataset_name") if annotations else None,
        **summarize(scores),
        "num_unscored_annotations": len(annotations) - len(scored),
        "num_missing_predictions": sum(a["sample_id"] not in by_id for a in scored),
        "by_slice": {key: summarize(values) for key, values in sorted(slices.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="answer.jsonl")
    parser.add_argument("--annotation_path", required=True, help="pointing_bench annotation jsonl")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.data_path), read_jsonl(args.annotation_path))
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
