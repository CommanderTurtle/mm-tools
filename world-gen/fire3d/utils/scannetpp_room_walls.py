"""Room-wall polygon estimation and outlier pruning for ScanNet++ clouds.

ScanNet++ depth here comes from a monocular depth estimator, so the projected
cloud carries flyer points that land outside the real walls. Projected to the
XY plane, the walls form connected chains of straight lines -- usually one
rectangle, sometimes an L-shape or several chains when the capture spans more
than one space.

Recovery and pruning:

1. rasterise the XY projection into an occupancy grid;
2. keep cells with enough points (flyers are sparse), close small gaps, fill
   the interior (the floor makes the room interior dense);
3. keep every connected component that is large enough to be a real space --
   a single capture can legitimately contain more than one room, so only the
   largest component alone is not safe;
4. smooth each component (closing + hole fill) so the boundary follows the
   outer walls rather than furniture-level concavities, then extract its outer
   contour and simplify it with Douglas-Peucker into the wall polyline;
5. rasterise the simplified polygons, dilate by a safety margin, and prune
   every point whose XY cell falls outside.

The pruning is deliberately conservative: interior concavities are filled, so
only points beyond the outer walls are removed. Everything is parameterised
and nothing here changes default dataloader behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class RoomWallEstimate:
    polygons_xy: list                 # list of (M_i, 2) wall polygons, closed implicitly
    keep_mask: np.ndarray             # (N,) bool per input point
    occupancy: np.ndarray             # (H, W) raw occupancy counts (debug)
    room_mask: np.ndarray             # (H, W) bool, union of filled simplified polygons
    keep_region: np.ndarray           # (H, W) bool, room_mask dilated by margin
    grid_origin: np.ndarray           # (2,) world xy of cell (0, 0) corner
    cell_size: float
    stats: dict = field(default_factory=dict)


def estimate_room_walls(
    points_xy: np.ndarray,
    *,
    cell_size: float = 0.06,
    min_points_per_cell: int = 3,
    closing_iterations: int = 2,
    smooth_closing_iterations: int = 3,
    component_min_area_m2: float = 1.5,
    component_min_fraction: float = 0.05,
    simplify_epsilon_m: float = 0.20,
    margin_m: float = 0.15,
) -> RoomWallEstimate:
    """Estimate the wall polygons of one capture and a per-point keep mask."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"points_xy must be (N, 2), got {points_xy.shape}")

    pad = 2.0 * cell_size + margin_m
    origin = points_xy.min(axis=0) - pad
    extent = points_xy.max(axis=0) - origin + pad
    width = int(np.ceil(extent[0] / cell_size)) + 1
    height = int(np.ceil(extent[1] / cell_size)) + 1

    cols = np.clip(((points_xy[:, 0] - origin[0]) / cell_size).astype(np.int64), 0, width - 1)
    rows = np.clip(((points_xy[:, 1] - origin[1]) / cell_size).astype(np.int64), 0, height - 1)
    occupancy = np.zeros((height, width), dtype=np.int64)
    np.add.at(occupancy, (rows, cols), 1)

    dense = occupancy >= int(min_points_per_cell)
    if closing_iterations > 0:
        dense = ndimage.binary_closing(dense, iterations=closing_iterations)
    filled = ndimage.binary_fill_holes(dense)

    labels, count = ndimage.label(filled)
    if count == 0:
        raise ValueError("Occupancy grid is empty; no room region found")
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    cell_area = cell_size * cell_size
    largest = float(sizes.max())
    keep_labels = [
        index + 1
        for index, size in enumerate(sizes)
        if size * cell_area >= component_min_area_m2
        and size >= component_min_fraction * largest
    ]
    if not keep_labels:
        keep_labels = [int(np.argmax(sizes)) + 1]

    polygons_xy = []
    room_mask = np.zeros_like(filled, dtype=np.uint8)
    epsilon_cells = max(1.0, simplify_epsilon_m / cell_size)
    for label_id in keep_labels:
        component = labels == label_id
        # smooth the boundary so it follows the outer walls, and refill any
        # concavity mouths the smoothing closes off
        if smooth_closing_iterations > 0:
            component = ndimage.binary_closing(
                component, iterations=smooth_closing_iterations
            )
            component = ndimage.binary_fill_holes(component)
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, epsilon_cells, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue
        cv2.fillPoly(room_mask, [approx.reshape(-1, 1, 2)], 1)
        polygons_xy.append(origin[None, :] + (approx.astype(np.float64) + 0.5) * cell_size)
    if not polygons_xy:
        raise ValueError("Every wall polygon degenerated below 3 vertices")

    room_mask = room_mask.astype(bool)
    margin_cells = max(1, int(round(margin_m / cell_size)))
    keep_region = ndimage.binary_dilation(room_mask, iterations=margin_cells)
    keep_mask = keep_region[rows, cols]

    stats = {
        "num_points": int(points_xy.shape[0]),
        "num_kept": int(keep_mask.sum()),
        "num_pruned": int((~keep_mask).sum()),
        "pruned_fraction": float((~keep_mask).mean()),
        "num_components": len(polygons_xy),
        "wall_vertices_per_component": [int(p.shape[0]) for p in polygons_xy],
        "grid_shape": [int(height), int(width)],
        "cell_size": float(cell_size),
        "min_points_per_cell": int(min_points_per_cell),
        "smooth_closing_iterations": int(smooth_closing_iterations),
        "component_min_area_m2": float(component_min_area_m2),
        "component_min_fraction": float(component_min_fraction),
        "simplify_epsilon_m": float(simplify_epsilon_m),
        "margin_m": float(margin_m),
    }
    return RoomWallEstimate(
        polygons_xy=polygons_xy,
        keep_mask=keep_mask,
        occupancy=occupancy,
        room_mask=room_mask,
        keep_region=keep_region,
        grid_origin=origin,
        cell_size=float(cell_size),
        stats=stats,
    )


def keep_mask_for_points(estimate: RoomWallEstimate, points_xy: np.ndarray) -> np.ndarray:
    """Look up the keep region for arbitrary points in the estimate's frame.

    The estimate can be built from one sampling stride and applied to another;
    points falling outside the grid are pruned.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    cols = np.floor((points_xy[:, 0] - estimate.grid_origin[0]) / estimate.cell_size).astype(np.int64)
    rows = np.floor((points_xy[:, 1] - estimate.grid_origin[1]) / estimate.cell_size).astype(np.int64)
    height, width = estimate.keep_region.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    keep = np.zeros(points_xy.shape[0], dtype=bool)
    keep[inside] = estimate.keep_region[rows[inside], cols[inside]]
    return keep


def wall_line_rotation(
    polygons_xy: list,
    *,
    tolerance_deg: float = 5.0,
    min_segment_length_m: float = 0.3,
    search_step_deg: float = 0.05,
) -> tuple[float, dict]:
    """Yaw that maximises the number of near-axis-parallel wall lines.

    Every wall segment (from every chain) longer than ``min_segment_length_m``
    votes. A segment counts as aligned at yaw ``phi`` when its rotated
    direction is within ``tolerance_deg`` of the x or y axis. The count is the
    primary objective; total aligned length breaks ties. The winning yaw is
    then refined by the length-weighted mean angular residual of its aligned
    segments, and reported in ``[-45, 45)`` degrees so the scene stays close to
    its original orientation.
    """
    starts = np.concatenate([np.asarray(p, dtype=np.float64) for p in polygons_xy], axis=0)
    ends = np.concatenate(
        [np.roll(np.asarray(p, dtype=np.float64), -1, axis=0) for p in polygons_xy], axis=0
    )
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    long_enough = lengths >= float(min_segment_length_m)
    if not long_enough.any():
        long_enough = lengths > 0
    angles_deg = np.degrees(np.arctan2(vectors[long_enough, 1], vectors[long_enough, 0])) % 90.0
    seg_lengths = lengths[long_enough]

    def alignment(phi_deg: float) -> tuple[int, float, np.ndarray]:
        rotated = (angles_deg + phi_deg) % 90.0
        residual = np.minimum(rotated, 90.0 - rotated)
        aligned = residual <= float(tolerance_deg)
        return int(aligned.sum()), float(seg_lengths[aligned].sum()), aligned

    candidates = np.arange(0.0, 90.0, float(search_step_deg))
    best_phi, best_count, best_length = 0.0, -1, -1.0
    for phi in candidates:
        count, total, _ = alignment(float(phi))
        if count > best_count or (count == best_count and total > best_length):
            best_phi, best_count, best_length = float(phi), count, total

    # refine: cancel the length-weighted mean signed residual of aligned lines
    _, _, aligned = alignment(best_phi)
    rotated = (angles_deg[aligned] + best_phi) % 90.0
    signed = np.where(rotated > 45.0, rotated - 90.0, rotated)
    if aligned.any():
        best_phi = best_phi - float(
            np.average(signed, weights=seg_lengths[aligned])
        )
    count, total, aligned = alignment(best_phi)

    # report the mod-90 representative closest to zero rotation
    phi = (best_phi + 45.0) % 90.0 - 45.0
    diagnostics = {
        "rotation_deg": float(phi),
        "num_segments": int(angles_deg.size),
        "num_aligned": int(count),
        "aligned_length_m": float(total),
        "total_length_m": float(seg_lengths.sum()),
        "tolerance_deg": float(tolerance_deg),
        "min_segment_length_m": float(min_segment_length_m),
        "segment_aligned": aligned,
        "segment_kept": long_enough,
    }
    return float(np.radians(phi)), diagnostics
