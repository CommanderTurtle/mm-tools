"""Transfer predicted perception instance IDs to a denser point grid."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from plyfile import PlyData


DEFAULT_SCENE_SCALE = 24.0
DEFAULT_VOXEL_RESOLUTION = 256


def read_ply_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"].data
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Invalid or empty point PLY: {path}")
    return points


def quantized_voxel_keys(
    points: np.ndarray, *, scene_scale: float, resolution: int
) -> np.ndarray:
    """Match Voxelize.forward(): truncate positive coordinates, then clamp."""
    if scene_scale <= 0 or resolution <= 0:
        raise ValueError("scene scale and voxel resolution must be positive")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    voxel_size = float(scene_scale) / int(resolution)
    coords = np.trunc(points / voxel_size).astype(np.int64)
    coords = np.clip(coords, 0, int(resolution) - 1)
    return (
        coords[:, 0] * int(resolution) * int(resolution)
        + coords[:, 1] * int(resolution)
        + coords[:, 2]
    )


def unique_voxel_labels(
    voxel_keys: np.ndarray, point_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Collapse source labels by voxel using a deterministic majority vote.

    Perception labels are constant for points sharing an inference voxel. A
    point serialized near a voxel boundary can quantize into its neighbor when
    the PLY is read back, however. Majority voting preserves the dominant
    source assignment in that rare case; an exact tie is left unlabeled.
    """
    voxel_keys = np.asarray(voxel_keys, dtype=np.int64).reshape(-1)
    point_labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    if voxel_keys.shape != point_labels.shape:
        raise ValueError("voxel keys and point labels must have matching shapes")
    if not len(voxel_keys):
        raise ValueError("perception voxel source must not be empty")
    order = np.argsort(voxel_keys, kind="stable")
    sorted_keys = voxel_keys[order]
    sorted_labels = point_labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_keys)]
    selected_labels = np.empty(len(starts), dtype=np.int64)
    conflict_voxels = 0
    tied_voxels = 0
    minority_points = 0
    for index, (start, end) in enumerate(zip(starts, ends)):
        labels, counts = np.unique(sorted_labels[start:end], return_counts=True)
        if len(labels) == 1:
            selected_labels[index] = labels[0]
            continue
        conflict_voxels += 1
        largest_count = int(counts.max())
        winners = labels[counts == largest_count]
        minority_points += int((end - start) - largest_count)
        if len(winners) == 1:
            selected_labels[index] = winners[0]
        else:
            selected_labels[index] = -1
            tied_voxels += 1
    return sorted_keys[starts], selected_labels, {
        "num_conflict_voxels": conflict_voxels,
        "num_tied_voxels": tied_voxels,
        "num_minority_source_points": minority_points,
    }


def transfer_voxel_labels(
    source_points: np.ndarray,
    source_labels: np.ndarray,
    dense_points: np.ndarray,
    *,
    scene_scale: float = DEFAULT_SCENE_SCALE,
    resolution: int = DEFAULT_VOXEL_RESOLUTION,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign a label only when a dense point occupies a labeled source voxel."""
    source_points = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    source_labels = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    dense_points = np.asarray(dense_points, dtype=np.float64).reshape(-1, 3)
    if len(source_points) != len(source_labels):
        raise ValueError("source points and labels must have matching lengths")
    source_keys = quantized_voxel_keys(
        source_points, scene_scale=scene_scale, resolution=resolution
    )
    dense_keys = quantized_voxel_keys(
        dense_points, scene_scale=scene_scale, resolution=resolution
    )
    unique_keys, unique_labels, conflict_audit = unique_voxel_labels(
        source_keys, source_labels
    )
    positions = np.searchsorted(unique_keys, dense_keys)
    safe_positions = np.minimum(positions, len(unique_keys) - 1)
    matched = (positions < len(unique_keys)) & (
        unique_keys[safe_positions] == dense_keys
    )
    transferred = np.full(len(dense_points), -1, dtype=np.int64)
    transferred[matched] = unique_labels[positions[matched]]
    return transferred, {
        "num_source_points": int(len(source_points)),
        "num_source_voxels": int(len(unique_keys)),
        "num_dense_points": int(len(dense_points)),
        "num_dense_points_in_source_voxels": int(np.count_nonzero(matched)),
        "num_dense_points_assigned": int(np.count_nonzero(transferred >= 0)),
        **conflict_audit,
    }


def load_perception_voxel_source(
    perception_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load source model points, labels, and the raw-to-model **rigid** transform.

    This used to assume raw -> model was a pure translation, which held only
    while every dataloader normalised by subtracting the min corner. The
    single_image loader can also canonicalise the room yaw by 0/90/180/270
    degrees, in which case `preprocess_transform = T_norm @ R_yaw` and the
    relation is a rotation plus a translation. The translation-only check
    rejected exactly the scenes whose chosen yaw was non-zero -- 10 of 20 in
    the prior aligned run -- so the relation is always recovered as a full rigid
    transform (Kabsch, no scale, no reflection) and validated on its residual.
    """
    model_points = read_ply_xyz(perception_dir / "pred_points.ply")
    raw_points = read_ply_xyz(perception_dir / "raw_pred_points.ply")
    with (perception_dir / "point_instance_masks.pkl").open("rb") as handle:
        labels = np.asarray(pickle.load(handle)["pred"], dtype=np.int64).reshape(-1)
    if len(model_points) != len(raw_points) or len(model_points) != len(labels):
        raise ValueError(f"Source point/label count mismatch: {perception_dir}")

    raw_centroid = raw_points.mean(axis=0)
    model_centroid = model_points.mean(axis=0)
    covariance = (raw_points - raw_centroid).T @ (model_points - model_centroid)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt          # applied on the right: p @ rotation
    raw_to_model = np.eye(4)
    raw_to_model[:3, :3] = rotation.T
    raw_to_model[:3, 3] = model_centroid - raw_centroid @ rotation
    predicted = raw_points @ rotation + raw_to_model[:3, 3]
    residual = float(np.max(np.abs(predicted - model_points)))
    if residual > 1e-4:
        raise ValueError(
            "Raw/model perception points are not related by a rigid transform: "
            f"{residual}"
        )
    return model_points, labels, raw_to_model, residual
