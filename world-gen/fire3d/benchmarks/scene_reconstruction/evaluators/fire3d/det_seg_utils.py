from __future__ import annotations

from itertools import combinations
from typing import List, Sequence

import numpy as np
from scipy.spatial import ConvexHull


def _as_obb_dict(bbox: dict, largest_scale: bool = False) -> dict:
    """Normalize the OBB field names used by the repo and older annotations."""
    if "obb_world" in bbox:
        bbox = {**bbox, **bbox["obb_world"]}
    elif "world" in bbox:
        bbox = {**bbox, **bbox["world"]}

    translation = np.asarray(bbox["translation"], dtype=np.float64).reshape(3)
    if "rotation" in bbox:
        quat = np.asarray(bbox["rotation"], dtype=np.float64).reshape(4)
        quat_order = "wxyz"
    elif "rotation_quat_wxyz" in bbox:
        quat = np.asarray(bbox["rotation_quat_wxyz"], dtype=np.float64).reshape(4)
        quat_order = "wxyz"
    elif "rotation_xyzw" in bbox:
        quat = np.asarray(bbox["rotation_xyzw"], dtype=np.float64).reshape(4)
        quat_order = "xyzw"
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quat_order = "wxyz"

    scale = np.asarray(bbox.get("scale", 1.0), dtype=np.float64)
    if largest_scale:
        scale = float(np.max(scale)) if scale.ndim > 0 else float(scale)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.repeat(float(scale), 3)
    scale = scale.reshape(3)
    if np.any(scale < 0):
        raise ValueError(f"OBB scale must be non-negative, got {scale}")

    return {
        "source": bbox,
        "translation": translation,
        "rotation": _quat_to_matrix(quat, order=quat_order),
        "scale": scale,
    }


def _quat_to_matrix(quat: Sequence[float], order: str = "wxyz") -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _obb_vertices(obb: dict) -> np.ndarray:
    half = obb["scale"] * 0.5
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * half
    return local @ obb["rotation"].T + obb["translation"]


def _obb_halfspaces(obb: dict) -> tuple[np.ndarray, np.ndarray]:
    axes = obb["rotation"]
    normals = np.concatenate([axes.T, -axes.T], axis=0)
    half = obb["scale"] * 0.5
    offsets = np.concatenate(
        [
            half + axes.T @ obb["translation"],
            half - axes.T @ obb["translation"],
        ]
    )
    return normals, offsets


def _points_inside_halfspaces(
    points: np.ndarray, normals: np.ndarray, offsets: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    return np.all(points @ normals.T <= offsets + eps, axis=1)


def _dedupe_points(points: list[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    unique = []
    for point in points:
        if np.all(np.isfinite(point)) and not any(
            np.linalg.norm(point - prev) <= tol for prev in unique
        ):
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def _intersection_volume(obb_a: dict, obb_b: dict) -> float:
    normals_a, offsets_a = _obb_halfspaces(obb_a)
    normals_b, offsets_b = _obb_halfspaces(obb_b)
    normals = np.concatenate([normals_a, normals_b], axis=0)
    offsets = np.concatenate([offsets_a, offsets_b], axis=0)

    points = []
    vertices_a = _obb_vertices(obb_a)
    vertices_b = _obb_vertices(obb_b)
    points.extend(vertices_a[_points_inside_halfspaces(vertices_a, normals_b, offsets_b)])
    points.extend(vertices_b[_points_inside_halfspaces(vertices_b, normals_a, offsets_a)])

    for i, j, k in combinations(range(normals.shape[0]), 3):
        matrix = normals[[i, j, k]]
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:
            continue
        point = np.linalg.solve(matrix, offsets[[i, j, k]])
        if np.all(normals @ point <= offsets + 1e-7):
            points.append(point)

    intersection_points = _dedupe_points(points)
    if intersection_points.shape[0] < 4:
        return 0.0

    try:
        hull = ConvexHull(intersection_points)
    except Exception:
        return 0.0
    return float(max(hull.volume, 0.0))


def _obb_volume(obb: dict) -> float:
    return float(np.prod(np.maximum(obb["scale"], 0.0)))


def _obb_iou(bbox_a: dict, bbox_b: dict) -> float:
    obb_a = _as_obb_dict(bbox_a, largest_scale=True)
    obb_b = _as_obb_dict(bbox_b, largest_scale=True)
    volume_a = _obb_volume(obb_a)
    volume_b = _obb_volume(obb_b)
    if volume_a <= 0.0 or volume_b <= 0.0:
        return 0.0

    intersection = _intersection_volume(obb_a, obb_b)
    union = volume_a + volume_b - intersection
    if union <= 0.0:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def calculate_mAP(pred_bboxes: List, gt_bboxes: List, iou_threshold: float = 0.25):
    """
    every item in bboxes is a dict with four keys:
    "translation" [x, y, z].
    "rotation" [qw, qx, qy, qz].
    "scale" largest_scale: float.
    "confidence" confidence score of the prediction, only for pred_bboxes.

    return:

    mAP: mean average precision of the predictions.
    pairs: a list of matched pairs of pred and gt bboxes, each item is a tuple (pred_bbox, gt_bbox, iou).
    """
    if len(gt_bboxes) == 0:
        return 0.0, []

    pred_order = sorted(
        range(len(pred_bboxes)),
        key=lambda idx: float(
            pred_bboxes[idx].get("confidence", pred_bboxes[idx].get("source_prob", 1.0))
        ),
        reverse=True,
    )

    matched_gt = set()
    pairs = []
    true_positives = np.zeros(len(pred_order), dtype=np.float64)
    false_positives = np.zeros(len(pred_order), dtype=np.float64)

    for rank, pred_idx in enumerate(pred_order):
        pred_bbox = pred_bboxes[pred_idx]
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt_bbox in enumerate(gt_bboxes):
            if gt_idx in matched_gt:
                continue
            iou = _obb_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            true_positives[rank] = 1.0
            pairs.append((pred_bbox, gt_bboxes[best_gt_idx], best_iou))
        else:
            false_positives[rank] = 1.0

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / max(len(gt_bboxes), 1)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for idx in range(precisions.size - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])

    changed = np.where(recalls[1:] != recalls[:-1])[0]
    mAP = float(
        np.sum((recalls[changed + 1] - recalls[changed]) * precisions[changed + 1])
    )
    return mAP, pairs


def assign_point_label_from_pred_bboxes(pred_bboxes: List, points: np.ndarray):
    """
    assign a label to each point in the point cloud based on the predicted bounding boxes.
    The label is the index of the predicted bbox (count from 1) that contains the point, or 0 if the point is not contained in any bbox.

    input:
    pred_bboxes: a list of predicted bounding boxes, each item is a dict with keys:
    "translation" [x, y, z].
    "rotation" [qw, qx, qy, qz].
    "scale" largest_scale: float.
    "confidence" confidence score of the prediction.
    "index" the index of the bbox.

    points: a numpy array of shape (N, 3) representing the point cloud.

    return:
    labels: a list of labels for each point in the point cloud. a numpy array of shape (N,) where each element is the index of the predicted bbox that contains the point, or 0 if the point is not contained in any bbox.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.zeros(points.shape[0], dtype=np.int64)

    pred_order = sorted(
        range(len(pred_bboxes)),
        key=lambda idx: float(
            pred_bboxes[idx].get("confidence", pred_bboxes[idx].get("source_prob", 1.0))
        ),
        reverse=True,
    )

    for pred_idx in pred_order:
        bbox = pred_bboxes[pred_idx]
        obb = _as_obb_dict(bbox)
        local = (points - obb["translation"]) @ obb["rotation"]
        inside = np.all(np.abs(local) <= obb["scale"] * 0.5 + 1e-8, axis=1)
        unlabeled_inside = inside & (labels == 0)
        label = int(bbox.get("index", pred_idx + 1))
        labels[unlabeled_inside] = label

    return labels


def calculate_mIoU(pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int):
    """
    calculate mean IoU between the predicted labels and the ground truth labels.

    input:
    pred_labels: a numpy array of shape (N,) representing the predicted labels for each point in the point cloud.
    gt_labels: a numpy array of shape (N,) representing the ground truth labels for each point in the point cloud.
    num_classes: the number of classes (including background).

    return:
    mIoU: mean IoU between the predicted labels and the ground truth labels.
    iou_per_class: a list of IoU for each class.
    """
    pred_labels = np.asarray(pred_labels).reshape(-1)
    gt_labels = np.asarray(gt_labels).reshape(-1)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(
            f"pred_labels and gt_labels must have the same shape, got "
            f"{pred_labels.shape} and {gt_labels.shape}"
        )

    iou_per_class = []
    valid_ious = []
    for class_idx in range(num_classes):
        pred_mask = pred_labels == class_idx
        gt_mask = gt_labels == class_idx
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            iou = float("nan")
        else:
            iou = float(intersection / union)
            valid_ious.append(iou)
        iou_per_class.append(iou)

    mIoU = float(np.mean(valid_ious)) if valid_ious else float("nan")
    return mIoU, iou_per_class
