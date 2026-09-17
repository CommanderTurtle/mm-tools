"""Robust rectangular-room prior for background reconstruction conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoomBox:
    """A z-up box represented by its center, box-to-scene rotation, and radii."""

    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray
    yaw_degrees: float
    trim_quantile: float

    def to_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "rotation_box_to_scene": self.rotation.tolist(),
            "half_extents": self.half_extents.tolist(),
            "full_extents": (2.0 * self.half_extents).tolist(),
            "yaw_degrees": float(self.yaw_degrees),
            "trim_quantile": float(self.trim_quantile),
        }


def _validate_points(points: np.ndarray, *, minimum: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    if len(points) < minimum:
        raise ValueError(f"Need at least {minimum} points, got {len(points)}")
    if not np.isfinite(points).all():
        raise ValueError("Room-box points must be finite")
    return points


def fit_z_up_room_box(
    points: np.ndarray,
    *,
    trim_quantile: float = 0.01,
    yaw_samples: int = 180,
) -> RoomBox:
    """Fit a robust minimum-area XY rectangle with independent robust Z bounds."""
    points = _validate_points(points, minimum=8)
    if not 0.0 <= trim_quantile < 0.25:
        raise ValueError("trim_quantile must be in [0, 0.25)")
    if yaw_samples <= 0:
        raise ValueError("yaw_samples must be positive")

    lower_q, upper_q = trim_quantile, 1.0 - trim_quantile
    best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for yaw in np.linspace(0.0, 0.5 * np.pi, yaw_samples, endpoint=False):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        local_xy = points[:, :2] @ rotation[:2, :2]
        bounds = np.quantile(local_xy, [lower_q, upper_q], axis=0)
        extents = bounds[1] - bounds[0]
        area = float(np.prod(extents))
        candidate = (area, float(yaw), rotation, bounds, extents)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    assert best is not None
    _, yaw, rotation, xy_bounds, xy_extents = best
    z_bounds = np.quantile(points[:, 2], [lower_q, upper_q])
    full_extents = np.r_[xy_extents, z_bounds[1] - z_bounds[0]]
    if np.any(full_extents <= 1e-6):
        raise ValueError(f"Degenerate fitted room extents: {full_extents.tolist()}")
    local_center = np.r_[xy_bounds.mean(axis=0), z_bounds.mean()]
    center = local_center @ rotation.T
    return RoomBox(
        center=center,
        rotation=rotation,
        half_extents=0.5 * full_extents,
        yaw_degrees=float(np.degrees(yaw)),
        trim_quantile=float(trim_quantile),
    )


def distance_to_room_box_surface(points: np.ndarray, box: RoomBox) -> np.ndarray:
    """Return exact Euclidean distance to the six finite faces of a box."""
    points = _validate_points(points, minimum=1)
    local = (points - box.center) @ box.rotation
    face_delta = np.abs(local) - box.half_extents
    outside = np.maximum(face_delta, 0.0)
    distances = np.linalg.norm(outside, axis=1)
    inside = np.all(face_delta <= 0.0, axis=1)
    distances[inside] = np.min(-face_delta[inside], axis=1)
    return distances


def fit_room_box_isotropic_canonical_transform(
    points: np.ndarray,
    box: RoomBox,
    *,
    margin: float = 0.01,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a training-compatible isotropic transform enclosing room points.

    The reconstruction flows were trained with one scalar object scale, so the
    room is not independently stretched along xyz. The fitted room center and
    yaw are retained, while the scalar scale expands enough to place every
    retained room-shell point inside ``[-0.5 + margin, 0.5 - margin]^3``.

    Returns the scene-to-canonical transform and a serializable audit record.
    """
    points = _validate_points(points, minimum=1)
    if not 0.0 <= margin < 0.5:
        raise ValueError("canonical margin must be in [0, 0.5)")

    local = (points - box.center) @ box.rotation
    required_half_extents = np.max(np.abs(local), axis=0)
    usable_half_extent = 0.5 - float(margin)
    scalar_scale = float(np.max(required_half_extents) / usable_half_extent)
    if not np.isfinite(scalar_scale) or scalar_scale <= 1e-6:
        raise ValueError(f"Degenerate canonical room scale: {scalar_scale}")

    object_to_scene = np.eye(4, dtype=np.float64)
    object_to_scene[:3, :3] = box.rotation * scalar_scale
    object_to_scene[:3, 3] = box.center
    scene_to_object = np.linalg.inv(object_to_scene)

    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    canonical = (homogeneous @ scene_to_object.T)[:, :3]
    maximum_absolute_coordinate = float(np.max(np.abs(canonical)))
    tolerance = 1e-9
    if maximum_absolute_coordinate > usable_half_extent + tolerance:
        raise RuntimeError(
            "Room canonical fit failed to enclose its input points: "
            f"{maximum_absolute_coordinate} > {usable_half_extent}"
        )

    return scene_to_object.astype(np.float32), {
        "mode": "room_box_isotropic_enclose",
        "canonical_margin": float(margin),
        "usable_half_extent": usable_half_extent,
        "center": box.center.tolist(),
        "rotation_box_to_scene": box.rotation.tolist(),
        "required_half_extents": required_half_extents.tolist(),
        "isotropic_scale": scalar_scale,
        "object_to_scene": object_to_scene.tolist(),
        "scene_to_object": scene_to_object.tolist(),
        "canonical_bounds": [
            canonical.min(axis=0).tolist(),
            canonical.max(axis=0).tolist(),
        ],
        "maximum_absolute_coordinate": maximum_absolute_coordinate,
        "num_points": int(len(points)),
        "num_outside_unit_box": int(
            np.count_nonzero(np.any(np.abs(canonical) > 0.5, axis=1))
        ),
    }


def filter_background_instance_near_room_box(
    points: np.ndarray,
    instance_ids: np.ndarray,
    *,
    background_instance_id: int | None = None,
    distance_threshold: float = 0.12,
    trim_quantile: float = 0.01,
    yaw_samples: int = 180,
    minimum_points: int = 512,
    adaptive_max_distance: float = 0.25,
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove background-ID points far from a fitted room-box surface.

    Non-background IDs are copied exactly. If the fixed distance contains fewer
    than ``minimum_points``, the threshold grows only as far as
    ``adaptive_max_distance`` so FPS has useful support without admitting
    arbitrary interior clutter.
    """
    points = _validate_points(points, minimum=1)
    instance_ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    if len(points) != len(instance_ids):
        raise ValueError("points and instance_ids must have matching lengths")
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    if minimum_points < 1:
        raise ValueError("minimum_points must be positive")
    if adaptive_max_distance < distance_threshold:
        raise ValueError(
            "adaptive_max_distance must be at least distance_threshold"
        )

    valid_ids, counts = np.unique(instance_ids[instance_ids >= 0], return_counts=True)
    if not len(valid_ids):
        raise ValueError("Cannot select a background from empty instance IDs")
    if background_instance_id is None:
        background_instance_id = int(valid_ids[int(np.argmax(counts))])
    background_mask = instance_ids == int(background_instance_id)
    background_points = points[background_mask]
    if len(background_points) < 8:
        raise ValueError(
            f"Background instance {background_instance_id} has only "
            f"{len(background_points)} points"
        )

    box = fit_z_up_room_box(
        background_points,
        trim_quantile=trim_quantile,
        yaw_samples=yaw_samples,
    )
    distances = distance_to_room_box_surface(background_points, box)
    required = min(int(minimum_points), len(background_points))
    effective_threshold = float(distance_threshold)
    fixed_count = int(np.count_nonzero(distances <= effective_threshold))
    if fixed_count < required:
        required_distance = float(np.partition(distances, required - 1)[required - 1])
        effective_threshold = min(
            max(effective_threshold, required_distance),
            float(adaptive_max_distance),
        )

    retained_background = distances <= effective_threshold + 1e-12
    retained_count = int(np.count_nonzero(retained_background))
    if retained_count == 0:
        raise ValueError(
            f"Room-box prior retained no points for instance {background_instance_id}"
        )
    keep = np.ones(len(points), dtype=bool)
    keep[np.flatnonzero(background_mask)] = retained_background
    filtered_ids = instance_ids.copy()
    filtered_ids[~keep] = -1
    return filtered_ids, {
        "schema": "ff_background_room_box_prior_v1",
        "background_local_instance_id": int(background_instance_id),
        "source_background_points": int(len(background_points)),
        "fixed_distance_threshold": float(distance_threshold),
        "effective_distance_threshold": float(effective_threshold),
        "adaptive_max_distance": float(adaptive_max_distance),
        "minimum_requested_points": int(minimum_points),
        "minimum_effective_points": int(required),
        "fixed_threshold_points": fixed_count,
        "retained_background_points": retained_count,
        "rejected_background_points": int(len(background_points) - retained_count),
        "minimum_points_met": bool(retained_count >= required),
        "surface_distance_quantiles": {
            str(quantile): float(np.quantile(distances, quantile))
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
        },
        "box": box.to_dict(),
    }
