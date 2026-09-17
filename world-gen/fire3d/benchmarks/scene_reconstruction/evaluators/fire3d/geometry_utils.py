from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _as_mesh(mesh) -> trimesh.Trimesh:

    return trimesh.util.concatenate(mesh.dump()) if isinstance(mesh, trimesh.Scene) else mesh


def _sample_mesh(mesh: trimesh.Trimesh, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0 or mesh.area <= 0.0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]
    return points.astype(np.float32), normals.astype(np.float32)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _nearest_distances_and_normals(
    src_points: np.ndarray,
    src_normals: np.ndarray,
    tgt_points: np.ndarray,
    tgt_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(tgt_points)
    distances, indices = tree.query(src_points)

    src_normals = _normalize_vectors(src_normals)
    tgt_normals = _normalize_vectors(tgt_normals)
    normal_dot = np.abs(np.sum(src_normals * tgt_normals[indices], axis=1))
    return distances.astype(np.float64), normal_dot.astype(np.float64)


# calculate chamfer distance, F1 score, Normal consistency of two trimesh meshes
def calculate_geometry_metrics(
    pred_mesh,
    gt_mesh,
    n_points: int = 100000,
    f1_threshold: float = 0.01,
):
    pred_mesh = _as_mesh(pred_mesh)
    gt_mesh = _as_mesh(gt_mesh)

    pred_points, pred_normals = _sample_mesh(pred_mesh, n_points)
    gt_points, gt_normals = _sample_mesh(gt_mesh, n_points)

    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {
            "CD": float("inf"),
            "F1": 0.0,
            "NC": 0.0,
        }

    pred_to_gt, pred_to_gt_normals = _nearest_distances_and_normals(
        pred_points, pred_normals, gt_points, gt_normals
    )
    gt_to_pred, gt_to_pred_normals = _nearest_distances_and_normals(
        gt_points, gt_normals, pred_points, pred_normals
    )

    cd = float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean()))
    precision = float(np.mean(pred_to_gt <= f1_threshold))
    recall = float(np.mean(gt_to_pred <= f1_threshold))
    f1_score = float(2.0 * precision * recall / (precision + recall + 1e-12))
    normal_consistency = float(0.5 * (pred_to_gt_normals.mean() + gt_to_pred_normals.mean()))

    return {
        "CD": cd,
        "F1": f1_score,
        "NC": normal_consistency,
    }
