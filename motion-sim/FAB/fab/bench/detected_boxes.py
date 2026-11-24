"""One-to-one detection association for evaluation; ground-truth boxes never enter body inference."""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_detector_exposure(image_rgb):
    """Bounded global gain for very dark detector inputs; no boxes, labels or corruption parameters.

    Estimate the 95th percentile on a small image. Bright inputs are unchanged. This is an opt-in
    option, off by default; amplifying sensor noise may hurt.
    """
    import cv2

    preview = cv2.resize(image_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    high = float(np.percentile(cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY), 95))
    if high >= 90:
        return image_rgb
    gain = min(4., 180. / max(high, 1.))
    table = np.clip(np.arange(256, dtype=np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.LUT(image_rgb, table)


def associate(boxes, detections, min_iou=.3):
    """Maximize valid match count, then IoU. Return detection indices (-1 for misses) and IoUs."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    detections = np.asarray(detections, np.float64).reshape(-1, 4)
    if not 0 < min_iou <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    for values in (boxes, detections):
        if not np.isfinite(values).all() or np.any(values[:, 2:] <= values[:, :2]):
            raise ValueError("Expected finite, nonempty xyxy boxes")
    indices, quality = np.full(len(boxes), -1, dtype=int), np.zeros(len(boxes))
    if not len(boxes) or not len(detections):
        return indices, quality
    area_a = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    area_b = np.prod(detections[:, 2:] - detections[:, :2], axis=1)
    wh = np.maximum(0, np.minimum(boxes[:, None, 2:], detections[None, :, 2:])
                    - np.maximum(boxes[:, None, :2], detections[None, :, :2]))
    intersection = np.prod(wh, axis=-1)
    iou = intersection / (area_a[:, None] + area_b[None, :] - intersection)
    # Dummy columns allow misses. A valid match dominates all possible secondary IoU gains.
    score = np.zeros((len(boxes), len(detections) + len(boxes)))
    score[:, :len(detections)] = np.where(iou >= min_iou, len(boxes) + 1 + iou, -1)
    rows, cols = linear_sum_assignment(-score)
    for row, col in zip(rows, cols):
        if col < len(detections) and iou[row, col] >= min_iou:
            indices[row], quality[row] = col, iou[row, col]
    return indices, quality


class DetectedBoxes:
    def __init__(self, samples, model, confidence=.25, min_iou=.3, preprocess="none"):
        import os

        os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # ultralytics never pip-installs
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT

        self.model = YOLO(model)
        self.precision = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
        self.confidence, self.min_iou = confidence, min_iou
        if preprocess not in ("none", "exposure"):
            raise ValueError("Unknown detector preprocessing")
        self.preprocess = preprocess
        self.frames = defaultdict(list)
        for sample in samples:
            self.frames[sample["image"]].append(sample)
        self.records = []

    def select(self, image_rgb, sample):
        # Corrupted inputs are seeded per sample by the existing harness; do not cache across IDs.
        if self.preprocess == "exposure":
            image_rgb = normalize_detector_exposure(image_rgb)
        output = self.model.predict(np.ascontiguousarray(image_rgb[..., ::-1]), classes=[0], imgsz=640,
                                    conf=self.confidence, verbose=False, **self.precision)[0]
        boxes = output.boxes.xyxy.float().cpu().numpy()
        people = self.frames[sample["image"]]
        indices, ious = associate([s["bbox"] for s in people], boxes, self.min_iou)
        person = next(i for i, s in enumerate(people) if s["id"] == sample["id"])
        selected = int(indices[person])
        self.records.append({"id": sample["id"], "detected": selected >= 0, "iou": float(ious[person]),
                             "detections": len(boxes), "annotated_people": len(people)})
        self.last_others = [b for i, b in enumerate(boxes) if i != selected]  # eval_hmr --mask_prompt neg
        return boxes[selected] if selected >= 0 else None
