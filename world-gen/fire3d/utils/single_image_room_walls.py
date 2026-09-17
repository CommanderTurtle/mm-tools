"""Room-wall estimation and yaw canonicalisation for single_image clouds.

The wall estimator is described in
`design/20260902_2330-0500_single_image_room_wall_estimation.md`. It
lives here rather than under `eval/` because the dataloader needs it; the
figure-drawing front end stays in `eval/perception/estimate_single_image_room_walls.py`.

Two pieces:

`estimate_walls` -- ScanNet++ rasterises the XY projection and takes the outer
contour, which recovers the *visible footprint* of a single-view frustum rather
than its walls. Here `aligned_pcd.ply` is an organized (H/2, W/2) lattice, so
surface normals come from finite differences and a wall is a vertical surface
that additionally (a) spans most of the floor-to-ceiling height, which drops
chair backs and sofa sides, and (b) has the scene on one side of it within its
own span, which no interior plane satisfies.

`best_axis_aligned_yaw` -- pick one of 0/90/180/270 degrees about the vertical.
A multiple of 90 keeps axis-parallel walls axis-parallel, so the only thing it
changes is *which* side of the bounding box each wall ends up on once
`point_normalize` shifts the cloud's min corner to the origin. Choosing the yaw
that pulls the walls onto the x and y axes makes the normalized input the
network sees consistent across scenes: walls at the origin planes instead of
wherever the camera happened to look.
"""

from __future__ import annotations

import numpy as np

WALL_NORMAL_MAX_Z = 0.35
HORIZONTAL_NORMAL_MIN_Z = 0.85
CELL_SIZE = 0.05
WALL_MIN_POINTS = 4
WALL_MIN_HEIGHT_FRACTION = 0.55
WALL_MIN_LENGTH = 0.60
WALL_ONE_SIDED_FRACTION = 0.93
WALL_SIDE_TOLERANCE = 0.10
WALL_SPAN_MARGIN = 0.20
WALL_MERGE_DISTANCE = 0.25
WALL_MIN_SIDE_POINTS = 1000
YAW_CHOICES = (0, 90, 180, 270)
# Scores closer than this are treated as equal, and the smallest yaw wins.
TIE_TOLERANCE = 1e-4


def organized_normals(points: np.ndarray, height: int, width: int) -> np.ndarray:
    """Per-point unit normals from the organized (H, W) point lattice."""

    grid = points.reshape(height, width, 3)
    du = np.zeros_like(grid)
    dv = np.zeros_like(grid)
    du[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    dv[1:-1, :] = grid[2:, :] - grid[:-2, :]
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.clip(norm, 1e-9, None))
    normals[norm[..., 0] < 1e-9] = 0.0
    return normals.reshape(-1, 3)


def rasterise(points: np.ndarray, cell_size: float = CELL_SIZE) -> dict:
    """XY occupancy plus the row/col of every point."""

    xy = points[:, :2]
    origin = xy.min(axis=0) - 2.0 * cell_size
    extent = xy.max(axis=0) - origin + 2.0 * cell_size
    width = int(np.ceil(extent[0] / cell_size)) + 1
    height = int(np.ceil(extent[1] / cell_size)) + 1
    cols = np.clip(((xy[:, 0] - origin[0]) / cell_size).astype(np.int64), 0, width - 1)
    rows = np.clip(((xy[:, 1] - origin[1]) / cell_size).astype(np.int64), 0, height - 1)
    counts = np.zeros((height, width), np.int64)
    np.add.at(counts, (rows, cols), 1)
    return dict(origin=origin, cell_size=cell_size, counts=counts,
                rows=rows, cols=cols, shape=(height, width))


def wall_segments(grid, points, is_wall_point, *, floor_z, ceiling_z,
                  cell_size=CELL_SIZE, min_points=WALL_MIN_POINTS,
                  min_height_fraction=WALL_MIN_HEIGHT_FRACTION,
                  min_length=WALL_MIN_LENGTH,
                  one_sided_fraction=WALL_ONE_SIDED_FRACTION,
                  side_tolerance=WALL_SIDE_TOLERANCE,
                  span_margin=WALL_SPAN_MARGIN,
                  merge_distance=WALL_MERGE_DISTANCE,
                  min_side_points=WALL_MIN_SIDE_POINTS):
    """Room walls among the vertical surfaces, plus their line segments."""

    import cv2
    from scipy import ndimage

    room_height = max(ceiling_z - floor_z, 1e-6)
    rows, cols, shape = grid["rows"], grid["cols"], grid["shape"]
    z = points[:, 2]

    span_max = np.full(shape, -np.inf)
    span_min = np.full(shape, np.inf)
    counts = np.zeros(shape, np.int64)
    wr, wc, wz = rows[is_wall_point], cols[is_wall_point], z[is_wall_point]
    np.maximum.at(span_max, (wr, wc), wz)
    np.minimum.at(span_min, (wr, wc), wz)
    np.add.at(counts, (wr, wc), 1)
    span = np.where(counts > 0, span_max - span_min, 0.0)

    mask = (span >= min_height_fraction * room_height) & (counts >= min_points)
    mask = ndimage.binary_closing(mask, iterations=1)
    labels, count = ndimage.label(mask)
    if count:
        min_cells = max(3, int(min_length / cell_size))
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, np.arange(1, count + 1))
        keep = {i + 1 for i, size in enumerate(sizes) if size >= min_cells}
        mask = np.isin(labels, list(keep)) if keep else np.zeros_like(mask)

    segments = []
    if mask.any():
        min_cells = max(3, int(round(min_length / cell_size)))
        lines = cv2.HoughLinesP(
            mask.astype(np.uint8) * 255, rho=1, theta=np.pi / 180,
            threshold=min_cells, minLineLength=min_cells,
            maxLineGap=max(2, min_cells // 2),
        )
        xy = points[:, :2]
        if lines is not None:
            for x0, y0, x1, y1 in lines.reshape(-1, 4):
                p0 = grid["origin"] + (np.array([x0, y0]) + 0.5) * cell_size
                p1 = grid["origin"] + (np.array([x1, y1]) + 0.5) * cell_size
                direction = p1 - p0
                length = float(np.linalg.norm(direction))
                if length < min_length:
                    continue
                normal = np.array([-direction[1], direction[0]]) / length
                delta = xy - p0
                signed = delta @ normal
                along = delta @ (direction / length)
                within = (along >= -span_margin) & (along <= length + span_margin)
                off_plane = within & (np.abs(signed) > side_tolerance)
                denominator = int(off_plane.sum())
                if denominator < min_side_points:
                    continue
                positive = int((off_plane & (signed > side_tolerance)).sum())
                inside = max(positive, denominator - positive) / denominator
                if inside < one_sided_fraction:
                    continue
                segments.append({"start": p0.tolist(), "end": p1.tolist(),
                                 "length_m": length,
                                 "one_sided_fraction": float(inside),
                                 "points_off_plane": denominator})

    kept = []
    for seg in sorted(segments, key=lambda s: -s["length_m"]):
        p0, p1 = np.asarray(seg["start"]), np.asarray(seg["end"])
        direction = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        duplicate = False
        for other in kept:
            q0, q1 = np.asarray(other["start"]), np.asarray(other["end"])
            other_dir = (q1 - q0) / max(np.linalg.norm(q1 - q0), 1e-9)
            if abs(float(direction @ other_dir)) < 0.98:
                continue
            offset = abs(float((p0 - q0) @ np.array([-other_dir[1], other_dir[0]])))
            if offset < merge_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(seg)
    return mask, kept


def estimate_walls(points: np.ndarray, grid_height: int, grid_width: int) -> dict:
    """Full wall estimate for one organized single_image cloud."""

    z = points[:, 2]
    floor_z = float(np.percentile(z, 1))
    ceiling_z = float(np.percentile(z, 99))
    normals = organized_normals(points, grid_height, grid_width)
    valid = np.linalg.norm(normals, axis=1) > 0.5
    n_z = np.abs(normals[:, 2])
    is_vertical = valid & (n_z <= WALL_NORMAL_MAX_Z)
    is_horizontal = valid & (n_z >= HORIZONTAL_NORMAL_MIN_Z)

    grid = rasterise(points)
    mask, segments = wall_segments(grid, points, is_vertical,
                                   floor_z=floor_z, ceiling_z=ceiling_z)
    return {
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "room_height_m": ceiling_z - floor_z,
        "segments": segments,
        "wall_cell_mask": mask,
        "grid": grid,
        "is_vertical": is_vertical,
        "is_horizontal": is_horizontal,
        "is_floor": is_horizontal & (z <= floor_z + 0.10),
        "is_ceiling": is_horizontal & (z >= ceiling_z - 0.10),
    }


def yaw_matrix(degrees: float) -> np.ndarray:
    """4x4 rotation about the vertical (z) axis."""

    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[0, 0], matrix[0, 1] = cos, -sin
    matrix[1, 0], matrix[1, 1] = sin, cos
    return matrix


def _wall_axis_distance(segments, points_xy: np.ndarray) -> float:
    """Length-weighted distance from each wall to the axis it is parallel to.

    After `point_normalize` the cloud's min corner sits at the origin, so a wall
    lying on the min side is at coordinate ~0 (distance ~0 from its axis) and one
    on the max side is a whole room away. Weighting by segment length keeps a
    long wall from being outvoted by a short stub.
    """

    if not segments:
        return float("inf")
    offset = points_xy.min(axis=0)
    total_weight = 0.0
    total = 0.0
    for seg in segments:
        p0 = np.asarray(seg["start"], float) - offset
        p1 = np.asarray(seg["end"], float) - offset
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        # the perpendicular coordinate is the wall's distance to its own axis:
        # a wall running along x sits at some y, and vice versa
        if abs(direction[0]) >= abs(direction[1]):
            distance = float(abs(0.5 * (p0[1] + p1[1])))   # parallel to x -> its y
        else:
            distance = float(abs(0.5 * (p0[0] + p1[0])))   # parallel to y -> its x
        total += distance * length
        total_weight += length
    return total / total_weight if total_weight else float("inf")


def best_axis_aligned_yaw(points: np.ndarray, segments) -> dict:
    """Pick the 0/90/180/270 yaw that pulls the walls onto the x and y axes."""

    scores = {}
    for yaw in YAW_CHOICES:
        rotation = yaw_matrix(yaw)[:2, :2]
        rotated_xy = points[:, :2] @ rotation.T
        rotated = []
        for seg in segments:
            rotated.append({
                "start": (rotation @ np.asarray(seg["start"], float)).tolist(),
                "end": (rotation @ np.asarray(seg["end"], float)).tolist(),
                "length_m": seg["length_m"],
            })
        scores[yaw] = _wall_axis_distance(rotated, rotated_xy)
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    # Ties are the common case, not the exception: a scene with walls on both
    # min sides scores identically for two yaws. Comparing raw floats then picks
    # on ~1e-16 of rounding noise, so the "canonical" yaw could flip between
    # runs. Quantise first and break ties toward the smallest rotation.
    best = (
        min(finite, key=lambda yaw: (round(finite[yaw] / TIE_TOLERANCE), yaw))
        if finite
        else 0
    )
    ranked = sorted(finite, key=lambda yaw: (round(finite[yaw] / TIE_TOLERANCE), yaw))
    tied = [
        yaw for yaw in ranked
        if round(finite[yaw] / TIE_TOLERANCE) == round(finite[best] / TIE_TOLERANCE)
    ]
    return {
        "yaw_degrees": int(best),
        "scores": {int(k): float(v) for k, v in scores.items()},
        "tied_yaws": [int(y) for y in tied],
        "num_segments": len(segments),
        "applied": bool(finite),
    }
