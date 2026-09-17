"""Snap PBR condition points onto the inferred shape surface, per object.

In training, the PBR flow's condition points are aligned with the shape sparse
voxels by construction: both come from the same GT object. At inference the
condition points come from the perception cloud, while the shape is *generated*
-- so the two are misaligned by whatever the perception OBB, the depth, and the
generation disagree on. This moves each condition point to the nearest vertex of
the decoded 512-resolution shape (its canonical dual vertex, one per active
voxel), restoring the training-time contract that condition points lie on the
shape surface.

The snap happens in each object's canonical [-0.5, 0.5] cube and is mapped back
to the scene frame, so the flow model's own canonicalisation reproduces the
snapped positions untouched. Points whose nearest shape vertex is farther than
``max_distance`` are left unmoved: a point that far off the generated shape is
more likely mis-segmented background than a misaligned surface sample, and
dragging it onto the object would attach a wrong DINO feature to the surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEBUG_TARGET_CAP = 60_000


def _canonicalise(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1))], axis=1
    )
    return (homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def snap_condition_points(
    *,
    points: np.ndarray,
    instance_ids: np.ndarray,
    scene_to_object: np.ndarray,
    targets_by_local_id: dict[int, np.ndarray],
    object_names: list[str] | None = None,
    max_distance: float = 0.15,
    fraction: float = 1.0,
    min_target_vertices: int = 16,
    tree_target_cap: int = 500_000,
    debug_dir: Path | None = None,
    seed: int = 20260903,
) -> tuple[np.ndarray, dict]:
    """Return (snapped scene points, audit). Only listed objects move.

    fraction 1.0 places a point exactly on its nearest shape vertex; smaller
    values move it that fraction of the way. Distances are in canonical units
    (the object cube is 1.0 across; one 512-voxel is ~0.002).
    """

    from scipy.spatial import cKDTree

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if max_distance <= 0:
        raise ValueError(f"max_distance must be positive, got {max_distance}")

    rng = np.random.default_rng(seed)
    snapped = np.asarray(points, dtype=np.float32).copy()
    audit: dict = {"max_distance": float(max_distance), "fraction": float(fraction),
                   "objects": []}
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for local_id, targets in sorted(targets_by_local_id.items()):
        name = (
            object_names[local_id]
            if object_names is not None and local_id < len(object_names)
            else f"object_{local_id:04d}"
        )
        entry = {"local_id": int(local_id), "object_name": name}
        mask = np.asarray(instance_ids) == local_id
        count = int(mask.sum())
        targets = np.asarray(targets, dtype=np.float64)
        if count == 0 or targets.shape[0] < min_target_vertices:
            entry.update({"num_points": count, "num_moved": 0,
                          "skipped": "no_points" if count == 0 else "too_few_shape_vertices"})
            audit["objects"].append(entry)
            continue

        transform = np.asarray(scene_to_object[local_id], dtype=np.float64)
        canonical = _canonicalise(points[mask], transform)

        tree_targets = targets
        if targets.shape[0] > tree_target_cap:
            keep = rng.choice(targets.shape[0], tree_target_cap, replace=False)
            tree_targets = targets[keep]
        tree = cKDTree(tree_targets)
        distances, indices = tree.query(canonical, k=1, workers=-1)
        moved = distances <= max_distance
        moved_canonical = canonical.copy()
        moved_canonical[moved] = canonical[moved] + fraction * (
            tree_targets[indices[moved]] - canonical[moved]
        )
        back = _canonicalise(moved_canonical, np.linalg.inv(transform))
        snapped[mask] = back.astype(np.float32)

        entry.update({
            "num_points": count,
            "num_moved": int(moved.sum()),
            "moved_fraction": float(moved.mean()),
            "nn_distance_before_mean": float(distances.mean()),
            "nn_distance_before_p95": float(np.percentile(distances, 95)),
            "displacement_mean": float(
                fraction * distances[moved].mean()) if moved.any() else 0.0,
            "displacement_max": float(
                fraction * distances[moved].max()) if moved.any() else 0.0,
            "num_shape_vertices": int(targets.shape[0]),
        })
        audit["objects"].append(entry)

        if debug_dir is not None:
            sample = targets
            if sample.shape[0] > DEBUG_TARGET_CAP:
                keep = rng.choice(sample.shape[0], DEBUG_TARGET_CAP, replace=False)
                sample = sample[keep]
            np.savez_compressed(
                debug_dir / f"{name}.npz",
                cond_before=canonical.astype(np.float32),
                cond_after=moved_canonical.astype(np.float32),
                nn_distance=distances.astype(np.float32),
                moved=moved,
                shape_vertices=sample.astype(np.float32),
            )

    rows = [o for o in audit["objects"] if "moved_fraction" in o]
    audit["num_objects_snapped"] = len(rows)
    audit["scene_moved_fraction"] = (
        float(np.mean([o["moved_fraction"] for o in rows])) if rows else 0.0
    )
    audit["scene_nn_distance_before_mean"] = (
        float(np.mean([o["nn_distance_before_mean"] for o in rows])) if rows else 0.0
    )
    if debug_dir is not None:
        (debug_dir / "snap_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    return snapped, audit
