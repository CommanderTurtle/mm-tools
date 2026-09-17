
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.evaluators.fire3d.det_seg_utils import (  # noqa: E402
    assign_point_label_from_pred_bboxes,
    calculate_mAP,
    calculate_mIoU,
)
from benchmarks.scene_reconstruction.evaluators.fire3d.geometry_utils import (  # noqa: E402
    calculate_geometry_metrics,
)
from benchmarks.scene_reconstruction.evaluators.fire3d.imaginarium_utils import (  # noqa: E402
    get_obbs_and_meshes,
    get_point_cloud,
    load_pred_obbs_and_meshes,
)


DEFAULT_PRED_RESULTS_ROOT = REPO_ROOT / "outputs/benchmarks/imaginarium"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _label_colors(labels):
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.zeros((labels.shape[0], 4), dtype=np.uint8)
    colors[:, 3] = 255
    nonzero = labels > 0
    colors[nonzero, 0] = (labels[nonzero] * 37 + 83) % 255
    colors[nonzero, 1] = (labels[nonzero] * 67 + 29) % 255
    colors[nonzero, 2] = (labels[nonzero] * 97 + 151) % 255
    colors[~nonzero, :3] = 120
    return colors


def _export_labeled_cloud(path, points, labels):
    cloud = trimesh.points.PointCloud(points, colors=_label_colors(labels))
    cloud.export(path)


def _quat_to_matrix_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _bbox_to_mesh(bbox, color):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_matrix_wxyz(bbox.get("rotation", [1, 0, 0, 0]))
    transform[:3, 3] = np.asarray(bbox["translation"], dtype=np.float64)

    half = np.asarray(bbox["scale"], dtype=np.float64) * 0.5
    corners = np.array(
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
    corners = corners * half
    corners = (transform[:3, :3] @ corners.T).T + transform[:3, 3]

    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]

    radius = max(float(np.min(np.maximum(half, 1e-6))) * 0.015, 0.002)
    cylinders = []
    for start_idx, end_idx in edges:
        start = corners[start_idx]
        end = corners[end_idx]
        if np.linalg.norm(end - start) < 1e-8:
            continue
        edge_mesh = trimesh.creation.cylinder(
            radius=radius,
            sections=8,
            segment=np.stack([start, end], axis=0),
        )
        edge_mesh.visual.face_colors = color
        cylinders.append(edge_mesh)

    if len(cylinders) == 0:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(cylinders)


def _export_bbox_scene(path, bboxes, color):
    scene = trimesh.Scene()
    for bbox in bboxes:
        scene.add_geometry(_bbox_to_mesh(bbox, color))
    scene.export(path)


def _average_metrics(metrics):
    if not metrics:
        return {"CD": None, "F1": None, "NC": None}
    keys = sorted(metrics[0].keys())
    average = {}
    for key in keys:
        values = [m[key] for m in metrics if np.isfinite(m[key])]
        average[key] = float(np.mean(values)) if values else None
    return average


def _matched_pred_bboxes_with_gt_labels(matched_pairs):
    mapped = []
    for pred_bbox, gt_bbox, _ in matched_pairs:
        mapped_bbox = dict(pred_bbox)
        mapped_bbox["index"] = int(gt_bbox["index"])
        mapped.append(mapped_bbox)
    return mapped


def evaluate_scene(args):
    scene_name = args.scene_name
    pred_results_scene_dir = os.path.join(args.pred_results_root_dir, scene_name)
    pred_obbs_path = os.path.join(pred_results_scene_dir, "object_obbs.json")

    pred_bboxes, pred_meshes = load_pred_obbs_and_meshes(
        pred_results_scene_dir, pred_obbs_path
    )

    points, gt_point_labels = get_point_cloud(
        scene_name, downsample=args.point_downsample
    )
    gt_obbs_all, gt_meshes = get_obbs_and_meshes(scene_name)

    visible_gt_indices = set(int(v) for v in np.unique(gt_point_labels) if int(v) > 0)
    gt_obbs = [obb for obb in gt_obbs_all if int(obb["index"]) in visible_gt_indices]

    map_score, matched_pairs = calculate_mAP(
        pred_bboxes, gt_obbs, iou_threshold=args.iou_threshold
    )

    mapped_pred_bboxes = _matched_pred_bboxes_with_gt_labels(matched_pairs)
    pred_point_labels = assign_point_label_from_pred_bboxes(mapped_pred_bboxes, points)
    num_classes = max(visible_gt_indices | {0}) + 1 if visible_gt_indices else 1
    miou, iou_per_class = calculate_mIoU(pred_point_labels, gt_point_labels, num_classes)

    pair_results = []
    geometry_metrics = []
    for pred_bbox, gt_bbox, obb_iou in matched_pairs:
        pred_idx = int(pred_bbox["index"])
        gt_idx = int(gt_bbox["index"])
        pair_result = {
            "pred_index": pred_idx,
            "gt_index": gt_idx,
            "pred_name": pred_bbox.get("name"),
            "gt_name": gt_bbox.get("name"),
            "obb_iou": float(obb_iou),
        }

        if pred_idx in pred_meshes and gt_idx in gt_meshes:
            if args.geometry_seed is not None:
                np.random.seed(
                    int(args.geometry_seed) + pred_idx * 1_000_003 + gt_idx
                )
            metrics = calculate_geometry_metrics(
                pred_meshes[pred_idx],
                gt_meshes[gt_idx],
                n_points=args.geometry_samples,
                f1_threshold=args.f1_threshold,
            )
            pair_result["geometry"] = metrics
            geometry_metrics.append(metrics)
        else:
            pair_result["geometry"] = None
        pair_results.append(pair_result)

    results = {
        "scene_name": scene_name,
        "num_predictions": len(pred_bboxes),
        "num_gt_visible": len(gt_obbs),
        "iou_threshold": args.iou_threshold,
        "geometry_seed": args.geometry_seed,
        "mAP": map_score,
        "mIoU": miou,
        "iou_per_class": iou_per_class,
        "geometry": _average_metrics(geometry_metrics),
        "matched_pairs": pair_results,
    }

    output_dir = Path(args.output_dir or pred_results_scene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2)

    if args.export_vis:
        vis_dir = output_dir / "eval_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        _export_labeled_cloud(vis_dir / "gt_point_labels.ply", points, gt_point_labels)
        _export_labeled_cloud(vis_dir / "pred_point_labels.ply", points, pred_point_labels)
        _export_bbox_scene(vis_dir / "gt_obbs.glb", gt_obbs, [40, 180, 80, 90])
        _export_bbox_scene(vis_dir / "pred_obbs.glb", pred_bboxes, [250, 180, 20, 90])

        for pred_idx, mesh in pred_meshes.items():
            mesh.export(vis_dir / f"pred_mesh_{pred_idx:04d}.ply")
        for gt_idx in sorted(visible_gt_indices):
            if gt_idx in gt_meshes:
                gt_meshes[gt_idx].export(vis_dir / f"gt_mesh_{gt_idx:04d}.ply")

    return results, results_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument(
        "--pred_results_root_dir",
        type=str,
        default=os.environ.get(
            "FIRE3D_PREDICTIONS_ROOT", str(DEFAULT_PRED_RESULTS_ROOT)
        ),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.25)
    parser.add_argument("--point_downsample", type=int, default=1)
    parser.add_argument("--geometry_samples", type=int, default=100000)
    parser.add_argument("--geometry_seed", type=int, default=None)
    parser.add_argument(
        "--export-vis", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--f1_threshold", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    results, results_path = evaluate_scene(parse_args())
    print(json.dumps(_json_safe(results), indent=2))
    print(f"Saved results to {results_path}")
