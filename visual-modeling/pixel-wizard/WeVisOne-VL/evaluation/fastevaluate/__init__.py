"""NumPy implementation of the ``fastevaluate`` C++ extension used by Rex-Omni.

``evaluate`` produces the same per-category precision and recall arrays as the
extension, so no compilation is required. Matching follows COCOeval for boxes,
with all detections assigned the same confidence (so the file order is the
ranking) and the area range set to "all".

* COCO: at most 100 detections per image and category; crowd ground truth is ignored.
* LVIS: federated evaluation. Detections are kept only for the positive and
  verified negative categories of each image, unmatched detections of a
  ``not_exhaustive`` category are ignored, there is no detection limit, and no
  ground truth is treated as crowd.

The precision of a category is the precision at the highest recall bin it
reaches. As in the extension, a category whose precision row is all zero takes
the value of the previous category.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
RECALL_THRESHOLDS = np.array([i * 0.01 for i in range(101)])
AREA_ALL = (0.0, 1e5**2)


def _iou(dt: np.ndarray, gt: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    """IoU of xywh boxes; a crowd GT uses the detection area as the union."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    dx1, dy1 = dt[:, 0:1], dt[:, 1:2]
    dx2, dy2 = dx1 + dt[:, 2:3], dy1 + dt[:, 3:4]
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gx1 + gt[:, 2], gy1 + gt[:, 3]
    iw = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0, None)
    ih = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0, None)
    inter = iw * ih
    dt_area = dt[:, 2:3] * dt[:, 3:4]
    gt_area = gt[:, 2] * gt[:, 3]
    union = np.where(crowd[None, :], dt_area, dt_area + gt_area - inter)
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _load_detections(path: str, name_to_id: dict, positives, negatives, not_exhaustive, lvis):
    detections = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            image_id, payload = line.rstrip("\n").split("\t", 1)
            image_id = int(image_id)
            for item in json.loads(payload):
                category = name_to_id[item["class"]]
                if lvis and category not in positives[image_id] and category not in negatives[image_id]:
                    continue
                ignored = lvis and category in not_exhaustive[image_id]
                detections[(image_id, category)].append((item["rect"], ignored))
    return detections


def _match(dt, dt_ignored, gt, gt_ignored, crowd, threshold, ious):
    """COCOeval greedy matching; returns per-detection (tp, fp) flags."""
    matched = np.zeros(len(gt), dtype=bool)
    tps, fps = [], []
    for di in range(len(dt)):
        best, match = threshold, -1
        for gi in range(len(gt)):
            if matched[gi] and not crowd[gi]:
                continue
            if match > -1 and not gt_ignored[match] and gt_ignored[gi]:
                break
            if ious[di, gi] < best:
                continue
            best, match = ious[di, gi], gi
        if match == -1:
            tps.append(0)
            fps.append(0 if dt_ignored[di] else 1)
            continue
        matched[match] = True
        tps.append(0 if gt_ignored[match] else 1)
        fps.append(0)
    return tps, fps


def _precision_curve(tps, fps, num_gt):
    tp = np.cumsum(tps, dtype=float)
    fp = np.cumsum(fps, dtype=float)
    n = len(tp)
    recall = tp / num_gt
    precision = tp / (fp + tp + np.spacing(1))
    if n:
        precision = np.maximum.accumulate(precision[::-1])[::-1]
    curve = np.zeros(len(RECALL_THRESHOLDS))
    cursor = 0
    for index, threshold in enumerate(RECALL_THRESHOLDS):
        while cursor < n and recall[cursor] <= threshold:
            cursor += 1
        if cursor >= n:
            break
        curve[index] = precision[cursor]
    return (recall[-1] if n else 0.0), curve


def evaluate(gt_json: str, pred_tsv: str, eval_type: str = "coco") -> dict[str, list[float]]:
    """Per-category precision / recall (mean over IoU 0.50:0.95, @0.50, @0.95)."""
    lvis = eval_type == "lvis"
    with open(gt_json, encoding="utf-8") as handle:
        data = json.load(handle)
    image_ids = [image["id"] for image in data["images"]]
    category_ids = [category["id"] for category in data["categories"]]
    name_to_id = {category["name"]: category["id"] for category in data["categories"]}

    gts = defaultdict(list)
    positives = defaultdict(set)
    for annotation in data["annotations"]:
        if lvis:
            annotation = dict(annotation, iscrowd=0)
        gts[(annotation["image_id"], annotation["category_id"])].append(annotation)
        positives[annotation["image_id"]].add(annotation["category_id"])
    negatives = defaultdict(set, {i["id"]: set(i.get("neg_category_ids", [])) for i in data["images"]})
    not_exhaustive = defaultdict(
        set, {i["id"]: set(i.get("not_exhaustive_category_ids", [])) for i in data["images"]}
    )
    detections = _load_detections(pred_tsv, name_to_id, positives, negatives, not_exhaustive, lvis)
    max_detections = None if lvis else 100

    num_thresholds = len(IOU_THRESHOLDS)
    recall = -np.ones((num_thresholds, len(category_ids)))
    precision = -np.ones((num_thresholds, len(RECALL_THRESHOLDS), len(category_ids)))
    for k, category in enumerate(category_ids):
        tps = [[] for _ in range(num_thresholds)]
        fps = [[] for _ in range(num_thresholds)]
        num_gt = 0
        seen = False
        for image_id in image_ids:
            gt = gts.get((image_id, category), [])
            dt = detections.get((image_id, category), [])[:max_detections]
            if not gt and not dt:
                continue
            seen = True
            dt_ignored = [flag for _, flag in dt]
            dt_boxes = np.array([rect for rect, _ in dt], dtype=float).reshape(-1, 4)
            gt_ignored = np.array(
                [
                    bool(a.get("iscrowd", 0)) or not AREA_ALL[0] <= a["area"] <= AREA_ALL[1]
                    for a in gt
                ],
                dtype=bool,
            )
            order = np.argsort(gt_ignored, kind="stable")
            gt = [gt[i] for i in order]
            gt_ignored = gt_ignored[order]
            crowd = np.array([bool(a.get("iscrowd", 0)) for a in gt], dtype=bool)
            num_gt += int((~gt_ignored).sum())
            gt_boxes = np.array([a["bbox"] for a in gt], dtype=float).reshape(-1, 4)
            ious = _iou(dt_boxes, gt_boxes, crowd)
            for t, threshold in enumerate(IOU_THRESHOLDS):
                tp, fp = _match(dt_boxes, dt_ignored, gt, gt_ignored, crowd, threshold, ious)
                tps[t] += tp
                fps[t] += fp
        if not seen or num_gt == 0:
            continue
        for t in range(num_thresholds):
            recall[t, k], precision[t, :, k] = _precision_curve(tps[t], fps[t], num_gt)

    result = defaultdict(list)
    p50 = p95 = p_carry = 0.0
    for k in range(len(category_ids)):
        p50 = _last_positive(precision[0, :, k], p50)
        p95 = _last_positive(precision[-1, :, k], p95)
        precision_sum = recall_sum = 0.0
        for t in range(num_thresholds):
            p_carry = _last_positive(precision[t, :, k], p_carry)
            precision_sum += p_carry
            recall_sum += recall[t, k]
        result["precision"].append(precision_sum / num_thresholds)
        result["recall"].append(recall_sum / num_thresholds)
        result["precision50"].append(p50)
        result["recall50"].append(recall[0, k])
        result["precision95"].append(p95)
        result["recall95"].append(recall[-1, k])
    return dict(result)


def _last_positive(row: np.ndarray, previous: float) -> float:
    for value in row[::-1]:
        if value > 0:
            return float(value)
    return previous
