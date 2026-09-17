"""Shared contract for fair, object-centric ShapeR inference caches.

The cache deliberately excludes complete ground-truth meshes.  It contains only
multiview RGB bytes, sparse visible points/projections/depths, and annotated pose and
size metadata needed to put an object into the coordinate system used by FF models.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCHEMA = "ff_shaper_object_input_v1"
MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "mesh_vertices",
        "mesh_faces",
        "vertices",
        "faces",
        "gt_mesh",
        "meshes_canonical",
        "meshes_world",
    }
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write immutable content-addressed bytes without exposing partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def image_identity(encoded: bytes) -> tuple[str, str, int, int]:
    """Return content SHA-256, extension, height, and width for an encoded image."""
    digest = hashlib.sha256(encoded).hexdigest()
    with Image.open(io.BytesIO(encoded)) as image:
        width, height = image.size
        fmt = (image.format or "JPEG").upper()
    extension = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
    }.get(fmt, f".{fmt.lower()}")
    return digest, extension, int(height), int(width)


def validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported ShapeR object cache schema: {metadata.get('schema')!r}")
    leaked = FORBIDDEN_INFERENCE_KEYS.intersection(metadata)
    if leaked:
        raise ValueError(f"Inference metadata contains forbidden GT geometry keys: {sorted(leaked)}")
    required = {
        "sample_id",
        "scene_id",
        "bounds",
        "T_model_world",
        "image_relpaths",
        "image_hw",
        "observations_relpath",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise KeyError(f"ShapeR object metadata is missing keys: {missing}")


def read_manifest(cache_root: Path) -> list[dict[str, Any]]:
    cache_root = Path(cache_root)
    manifest_path = cache_root / MANIFEST_NAME
    entries = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("schema") != SCHEMA:
                raise ValueError(
                    f"Bad schema in {manifest_path}:{line_number}: {entry.get('schema')!r}"
                )
            entries.append(entry)
    return entries


def load_cached_object(
    cache_root: Path, entry_or_metadata_path: dict[str, Any] | str | Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_root = Path(cache_root)
    if isinstance(entry_or_metadata_path, dict):
        metadata_path = cache_root / entry_or_metadata_path["metadata_relpath"]
    else:
        metadata_path = Path(entry_or_metadata_path)
        if not metadata_path.is_absolute():
            metadata_path = cache_root / metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata)
    observations_path = cache_root / metadata["observations_relpath"]
    with np.load(observations_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_observation_arrays(metadata, arrays)
    return metadata, arrays


def validate_observation_arrays(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    required = {
        "points_model",
        "view_points_model",
        "view_uv",
        "view_depth",
        "view_offsets",
        "Ts_camera_model",
        "camera_params",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"Observation archive is missing keys: {missing}")
    points = np.asarray(arrays["view_points_model"])
    uv = np.asarray(arrays["view_uv"])
    depth = np.asarray(arrays["view_depth"])
    offsets = np.asarray(arrays["view_offsets"])
    num_views = len(metadata["image_relpaths"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"view_points_model must be [N,3], got {points.shape}")
    if uv.shape != (points.shape[0], 2):
        raise ValueError(f"view_uv must be [N,2], got {uv.shape} for {points.shape[0]} points")
    if depth.shape != (points.shape[0],):
        raise ValueError(f"view_depth must be [N], got {depth.shape}")
    if offsets.shape != (num_views + 1,):
        raise ValueError(f"view_offsets must have {num_views + 1} values, got {offsets.shape}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != points.shape[0]:
        raise ValueError("view_offsets do not cover the concatenated observations")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("view_offsets must be monotonic")
    if np.asarray(arrays["Ts_camera_model"]).shape != (num_views, 4, 4):
        raise ValueError("Ts_camera_model shape does not match the cached view count")


def select_view_indices(view_offsets: np.ndarray, max_views: int | None) -> np.ndarray:
    offsets = np.asarray(view_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    available = np.flatnonzero(counts > 0)
    if max_views is None or max_views <= 0 or available.size <= int(max_views):
        return available.astype(np.int64)
    # Prefer views with more sparse support; restore capture order for DINO batching.
    ranked = available[np.argsort(-counts[available], kind="stable")[: int(max_views)]]
    return np.sort(ranked).astype(np.int64)


def select_view_indices_by_feature_support(
    arrays: dict[str, np.ndarray],
    image_hw: np.ndarray,
    *,
    feature_stride: int,
    max_views: int | None,
) -> np.ndarray:
    """Select views by unique valid DINO-grid cells rather than raw point count.

    Sparse ShapeR observations often contain many points that collapse into the same
    feature cell.  Ranking by raw observations overstates those views and can exclude
    a view with broader image evidence.  Ties preserve capture order.
    """
    stride = int(feature_stride)
    if stride <= 0:
        raise ValueError(f"feature_stride must be positive, got {feature_stride}")
    offsets = np.asarray(arrays["view_offsets"], dtype=np.int64)
    uv = np.asarray(arrays["view_uv"], dtype=np.float32)
    depth = np.asarray(arrays["view_depth"], dtype=np.float32)
    image_hw = np.asarray(image_hw, dtype=np.int64)
    num_views = len(offsets) - 1
    if image_hw.shape != (num_views, 2):
        raise ValueError(f"image_hw must be [{num_views},2], got {image_hw.shape}")
    support = np.zeros((num_views,), dtype=np.int64)
    for view in range(num_views):
        start, end = int(offsets[view]), int(offsets[view + 1])
        if end <= start:
            continue
        height, width = (int(value) for value in image_hw[view])
        grid_h, grid_w = height // stride, width // stride
        uv_view = uv[start:end]
        depth_view = depth[start:end]
        gx = np.floor(uv_view[:, 0] / stride).astype(np.int64)
        gy = np.floor(uv_view[:, 1] / stride).astype(np.int64)
        valid = (
            np.isfinite(uv_view).all(axis=1)
            & np.isfinite(depth_view)
            & (depth_view > 0)
            & (gx >= 0)
            & (gx < grid_w)
            & (gy >= 0)
            & (gy < grid_h)
        )
        if np.any(valid):
            support[view] = np.unique(gy[valid] * grid_w + gx[valid]).size
    available = np.flatnonzero(support > 0)
    if max_views is None or max_views <= 0 or available.size <= int(max_views):
        return available.astype(np.int64)
    ranked = available[np.argsort(-support[available], kind="stable")[: int(max_views)]]
    return np.sort(ranked).astype(np.int64)


def zbuffer_observations_to_feature_grid(
    arrays: dict[str, np.ndarray],
    image_hw: np.ndarray,
    *,
    feature_stride: int,
    view_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Select the nearest visible 3D point in each DINO/AnyUp feature cell.

    Returned ``feature_indices`` index the flattened feature tensor produced from the
    selected RGB stack, not the original all-view stack.
    """
    stride = int(feature_stride)
    if stride <= 0:
        raise ValueError(f"feature_stride must be positive, got {feature_stride}")
    offsets = np.asarray(arrays["view_offsets"], dtype=np.int64)
    points = np.asarray(arrays["view_points_model"], dtype=np.float32)
    uv = np.asarray(arrays["view_uv"], dtype=np.float32)
    depth = np.asarray(arrays["view_depth"], dtype=np.float32)
    image_hw = np.asarray(image_hw, dtype=np.int64)
    if view_indices is None:
        selected_views = np.arange(len(offsets) - 1, dtype=np.int64)
    else:
        selected_views = np.asarray(list(view_indices), dtype=np.int64)
    if selected_views.size == 0:
        return {
            "points_model": np.zeros((0, 3), dtype=np.float32),
            "feature_indices": np.zeros((0,), dtype=np.int64),
            "source_view_indices": np.zeros((0,), dtype=np.int64),
            "feature_yx": np.zeros((0, 2), dtype=np.int32),
            "depth": np.zeros((0,), dtype=np.float32),
        }

    selected_shapes = image_hw[selected_views]
    if np.any(selected_shapes != selected_shapes[0]):
        raise ValueError(
            "Selected ShapeR RGB views must share one resolution for batched DINO inference"
        )
    height, width = (int(value) for value in selected_shapes[0])
    grid_height, grid_width = height // stride, width // stride
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"Feature stride {stride} is too large for image shape {(height, width)}")
    cells_per_view = grid_height * grid_width

    output_points = []
    output_features = []
    output_views = []
    output_yx = []
    output_depth = []
    for local_view, source_view in enumerate(selected_views.tolist()):
        start, end = int(offsets[source_view]), int(offsets[source_view + 1])
        if end <= start:
            continue
        uv_view = uv[start:end]
        depth_view = depth[start:end]
        gx = np.floor(uv_view[:, 0] / stride).astype(np.int64)
        gy = np.floor(uv_view[:, 1] / stride).astype(np.int64)
        valid = (
            np.isfinite(uv_view).all(axis=1)
            & np.isfinite(depth_view)
            & (depth_view > 0)
            & (gx >= 0)
            & (gx < grid_width)
            & (gy >= 0)
            & (gy < grid_height)
        )
        if not np.any(valid):
            continue
        source_indices = np.arange(start, end, dtype=np.int64)[valid]
        gx, gy, depth_valid = gx[valid], gy[valid], depth_view[valid]
        cell = gy * grid_width + gx
        # Primary key is cell; secondary key is increasing depth.
        order = np.lexsort((depth_valid, cell))
        cell_sorted = cell[order]
        first = np.r_[True, cell_sorted[1:] != cell_sorted[:-1]]
        chosen_local = order[first]
        chosen_source = source_indices[chosen_local]
        chosen_cell = cell[chosen_local]
        chosen_gy = gy[chosen_local]
        chosen_gx = gx[chosen_local]
        output_points.append(points[chosen_source])
        output_features.append(local_view * cells_per_view + chosen_cell)
        output_views.append(np.full(chosen_source.shape, source_view, dtype=np.int64))
        output_yx.append(np.stack([chosen_gy, chosen_gx], axis=1).astype(np.int32))
        output_depth.append(depth[chosen_source])

    if not output_points:
        return {
            "points_model": np.zeros((0, 3), dtype=np.float32),
            "feature_indices": np.zeros((0,), dtype=np.int64),
            "source_view_indices": np.zeros((0,), dtype=np.int64),
            "feature_yx": np.zeros((0, 2), dtype=np.int32),
            "depth": np.zeros((0,), dtype=np.float32),
        }
    return {
        "points_model": np.concatenate(output_points, axis=0).astype(np.float32, copy=False),
        "feature_indices": np.concatenate(output_features).astype(np.int64, copy=False),
        "source_view_indices": np.concatenate(output_views).astype(np.int64, copy=False),
        "feature_yx": np.concatenate(output_yx, axis=0).astype(np.int32, copy=False),
        "depth": np.concatenate(output_depth).astype(np.float32, copy=False),
    }


def observations_to_feature_grid_without_zbuffer(
    arrays: dict[str, np.ndarray],
    image_hw: np.ndarray,
    *,
    feature_stride: int,
    view_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Retain every projected depth observation and only attach its DINO cell.

    This is the correct adapter for a dense depth + instance-mask input.  Multiple
    depth pixels are intentionally allowed to reference the same DINO feature cell:
    geometry is defined by depth/mask resolution, while DINO is merely sampled to
    attach an appearance feature to each already-selected 3D point.
    """
    stride = int(feature_stride)
    if stride <= 0:
        raise ValueError(f"feature_stride must be positive, got {feature_stride}")
    offsets = np.asarray(arrays["view_offsets"], dtype=np.int64)
    points = np.asarray(arrays["view_points_model"], dtype=np.float32)
    uv = np.asarray(arrays["view_uv"], dtype=np.float32)
    depth = np.asarray(arrays["view_depth"], dtype=np.float32)
    image_hw = np.asarray(image_hw, dtype=np.int64)
    selected_views = (
        np.arange(len(offsets) - 1, dtype=np.int64)
        if view_indices is None
        else np.asarray(list(view_indices), dtype=np.int64)
    )
    empty = {
        "points_model": np.zeros((0, 3), dtype=np.float32),
        "feature_indices": np.zeros((0,), dtype=np.int64),
        "source_view_indices": np.zeros((0,), dtype=np.int64),
        "feature_yx": np.zeros((0, 2), dtype=np.int32),
        "depth": np.zeros((0,), dtype=np.float32),
    }
    if selected_views.size == 0:
        return empty
    selected_shapes = image_hw[selected_views]
    if np.any(selected_shapes != selected_shapes[0]):
        raise ValueError(
            "Selected ShapeR RGB views must share one resolution for batched DINO inference"
        )
    height, width = (int(value) for value in selected_shapes[0])
    grid_height, grid_width = height // stride, width // stride
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"Feature stride {stride} is too large for image shape {(height, width)}")
    cells_per_view = grid_height * grid_width

    output_points: list[np.ndarray] = []
    output_features: list[np.ndarray] = []
    output_views: list[np.ndarray] = []
    output_yx: list[np.ndarray] = []
    output_depth: list[np.ndarray] = []
    for local_view, source_view in enumerate(selected_views.tolist()):
        start, end = int(offsets[source_view]), int(offsets[source_view + 1])
        if end <= start:
            continue
        uv_view = uv[start:end]
        depth_view = depth[start:end]
        gx = np.floor(uv_view[:, 0] / stride).astype(np.int64)
        gy = np.floor(uv_view[:, 1] / stride).astype(np.int64)
        valid = (
            np.isfinite(uv_view).all(axis=1)
            & np.isfinite(depth_view)
            & (depth_view > 0)
            & (gx >= 0)
            & (gx < grid_width)
            & (gy >= 0)
            & (gy < grid_height)
        )
        if not np.any(valid):
            continue
        source_indices = np.arange(start, end, dtype=np.int64)[valid]
        gx, gy = gx[valid], gy[valid]
        cells = gy * grid_width + gx
        output_points.append(points[source_indices])
        output_features.append(local_view * cells_per_view + cells)
        output_views.append(np.full(source_indices.shape, source_view, dtype=np.int64))
        output_yx.append(np.stack([gy, gx], axis=1).astype(np.int32))
        output_depth.append(depth[source_indices])
    if not output_points:
        return empty
    return {
        "points_model": np.concatenate(output_points, axis=0).astype(np.float32, copy=False),
        "feature_indices": np.concatenate(output_features).astype(np.int64, copy=False),
        "source_view_indices": np.concatenate(output_views).astype(np.int64, copy=False),
        "feature_yx": np.concatenate(output_yx, axis=0).astype(np.int32, copy=False),
        "depth": np.concatenate(output_depth).astype(np.float32, copy=False),
    }


def geometry_fps_preselect_rasterized_points(
    rasterized: dict[str, np.ndarray],
    max_points: int,
    *,
    initial_voxel_resolution: int = 64,
    max_candidate_points: int = 65536,
) -> dict[str, np.ndarray]:
    """Cap dense observations by 3D coverage, independent of DINO cell layout.

    A nearest-depth representative is first retained per adaptive 3D voxel.  A
    deterministic farthest-point pass then selects the requested count.  Every
    aligned raster field (including DINO feature indices) follows the chosen rows.
    """
    points = np.asarray(rasterized["points_model"], dtype=np.float32)
    count = int(points.shape[0])
    max_points = int(max_points)
    if max_points <= 0 or count <= max_points:
        return dict(rasterized)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_model must be [N,3], got {points.shape}")
    depth = np.asarray(rasterized["depth"], dtype=np.float32)
    if depth.shape != (count,):
        raise ValueError(f"depth must be [N], got {depth.shape} for {count} points")

    lower = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lower, 1e-8)
    resolution = max(2, int(initial_voxel_resolution))
    representatives = np.empty((0,), dtype=np.int64)
    while True:
        xyz = np.floor((points - lower) / span * resolution).astype(np.int64)
        xyz = np.clip(xyz, 0, resolution - 1)
        voxel = (xyz[:, 0] * resolution + xyz[:, 1]) * resolution + xyz[:, 2]
        rows = np.arange(count, dtype=np.int64)
        order = np.lexsort((rows, depth, voxel))
        sorted_voxel = voxel[order]
        first = np.r_[True, sorted_voxel[1:] != sorted_voxel[:-1]]
        representatives = order[first]
        if len(representatives) >= max_points or resolution >= 256:
            break
        resolution *= 2
    extrema = np.unique(
        np.concatenate([np.argmin(points, axis=0), np.argmax(points, axis=0)])
    ).astype(np.int64, copy=False)
    if len(representatives) > int(max_candidate_points):
        # The voxel codes are spatially distributed; a stable, even subsample keeps
        # the subsequent FPS cost bounded without consulting image/DINO cells.
        positions = np.linspace(
            0, len(representatives) - 1, int(max_candidate_points), dtype=np.int64
        )
        representatives = representatives[positions]
    # Preserve exact object extents even when the nearest-depth representative of
    # the boundary voxel lies slightly inside that voxel.
    representatives = np.unique(np.concatenate([extrema, representatives]))
    candidate_points = points[representatives].astype(np.float64, copy=False)
    if len(representatives) <= max_points:
        chosen = representatives
    else:
        selected = np.empty((max_points,), dtype=np.int64)
        center = candidate_points.mean(axis=0)
        selected[0] = int(np.argmax(np.sum((candidate_points - center) ** 2, axis=1)))
        min_distance = np.sum(
            (candidate_points - candidate_points[selected[0]]) ** 2, axis=1
        )
        for index in range(1, max_points):
            selected[index] = int(np.argmax(min_distance))
            distance = np.sum(
                (candidate_points - candidate_points[selected[index]]) ** 2, axis=1
            )
            np.minimum(min_distance, distance, out=min_distance)
        chosen = representatives[selected]

    output: dict[str, np.ndarray] = {}
    for key, value in rasterized.items():
        array = np.asarray(value)
        output[key] = array[chosen] if array.ndim > 0 and array.shape[0] == count else value
    return output


def geometry_hybrid_preselect_rasterized_points(
    rasterized: dict[str, np.ndarray],
    max_points: int,
    *,
    fps_fraction: float = 0.25,
    seed: int = 0,
    initial_voxel_resolution: int = 64,
) -> dict[str, np.ndarray]:
    """Mix 3D coverage points with density-preserving random depth pixels.

    Pure FPS overweights extreme interpolation pixels.  This sampler reserves a
    small portion for 3D coverage and draws the rest uniformly from the complete
    depth+mask cloud, preserving its underlying confidence/density distribution.
    Selection never uses DINO cells or GT geometry.
    """
    points = np.asarray(rasterized["points_model"])
    count = int(points.shape[0])
    max_points = int(max_points)
    if max_points <= 0 or count <= max_points:
        return dict(rasterized)
    fraction = float(fps_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fps_fraction must be in [0,1], got {fps_fraction}")
    fps_count = min(max_points, max(0, int(round(max_points * fraction))))
    marked = dict(rasterized)
    marked["_source_row"] = np.arange(count, dtype=np.int64)
    if fps_count:
        fps_result = geometry_fps_preselect_rasterized_points(
            marked,
            fps_count,
            initial_voxel_resolution=initial_voxel_resolution,
        )
        fps_rows = np.asarray(fps_result["_source_row"], dtype=np.int64)
    else:
        fps_rows = np.empty((0,), dtype=np.int64)
    random_count = max_points - len(fps_rows)
    available = np.ones((count,), dtype=bool)
    available[fps_rows] = False
    available_rows = np.flatnonzero(available)
    rng = np.random.default_rng(int(seed))
    random_rows = rng.choice(available_rows, size=random_count, replace=False)
    chosen = np.concatenate([fps_rows, random_rows.astype(np.int64, copy=False)])
    output: dict[str, np.ndarray] = {}
    for key, value in rasterized.items():
        array = np.asarray(value)
        output[key] = array[chosen] if array.ndim > 0 and array.shape[0] == count else value
    return output


def geometry_random_preselect_rasterized_points(
    rasterized: dict[str, np.ndarray],
    max_points: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select a stable uniform subset without replacement.

    The caller supplies a seed derived from the evaluation seed and sample ID.  A
    local NumPy generator deliberately keeps this selection independent of global
    Python/NumPy/Torch RNG consumption by model loading or earlier objects.  Chosen
    rows are sorted so the retained observations preserve their original ordering.
    """
    points = np.asarray(rasterized["points_model"])
    count = int(points.shape[0])
    max_points = int(max_points)
    if max_points <= 0 or count <= max_points:
        return dict(rasterized)
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(
        rng.choice(count, size=max_points, replace=False).astype(np.int64, copy=False)
    )
    output: dict[str, np.ndarray] = {}
    for key, value in rasterized.items():
        array = np.asarray(value)
        output[key] = array[chosen] if array.ndim > 0 and array.shape[0] == count else value
    return output


def deduplicate_rasterized_points(
    rasterized: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Keep one nearest-view DINO observation for each exact sparse 3D point.

    ShapeR stores sparse points together with every view in which they are visible.
    Consequently, the rasterized conditioning sequence may contain the same 3D point
    several times with different image features. This helper preserves first-seen
    3D-point order, but uses the observation with minimum positive camera depth as the
    representative so its DINO feature comes from the closest valid view.
    """
    points = np.asarray(rasterized["points_model"])
    count = int(points.shape[0])
    if count == 0:
        return dict(rasterized)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_model must be [N,3], got {points.shape}")
    depth = np.asarray(rasterized["depth"])
    if depth.shape != (count,):
        raise ValueError(f"depth must be [N], got {depth.shape} for {count} points")

    _, first_indices, inverse = np.unique(
        points, axis=0, return_index=True, return_inverse=True
    )
    observation_indices = np.arange(count, dtype=np.int64)
    # Primary key: point group; secondary: nearest depth; tertiary: stable row.
    ranked = np.lexsort((observation_indices, depth, inverse))
    ranked_groups = inverse[ranked]
    group_first = np.r_[True, ranked_groups[1:] != ranked_groups[:-1]]
    chosen_by_sorted_group = ranked[group_first]
    stable_group_order = np.argsort(first_indices, kind="stable")
    chosen = chosen_by_sorted_group[stable_group_order]

    output: dict[str, np.ndarray] = {}
    for key, value in rasterized.items():
        array = np.asarray(value)
        output[key] = array[chosen] if array.ndim > 0 and array.shape[0] == count else value
    return output


def object_model_points_to_ff_condition(
    points_model: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Embed one metric model-frame object in an FF-compatible positive scene frame.

    The returned transform maps the synthetic scene coordinates into the object's
    ``[-0.5, 0.5]^3`` canonical frame, matching the training/inference contract.
    """
    points_model = np.asarray(points_model, dtype=np.float32).reshape(-1, 3)
    bounds = np.asarray(bounds, dtype=np.float32).reshape(3)
    full_scale = float(2.0 * np.max(bounds))
    if not np.isfinite(full_scale) or full_scale <= 0:
        raise ValueError(f"Invalid ShapeR object bounds: {bounds.tolist()}")
    translation = np.full((3,), full_scale * 0.5, dtype=np.float32)
    points_scene = points_model + translation
    scene_to_object = np.eye(4, dtype=np.float32)
    scene_to_object[:3, :3] /= full_scale
    scene_to_object[:3, 3] = -translation / full_scale
    # Sparse SLAM points may lie slightly outside the annotated complete-mesh OBB.
    # Keep the authoritative scale here; the model's existing
    # filter_points_in_object_unit_box() drops those noisy outliers together with their
    # aligned image features.
    return points_scene, scene_to_object, full_scale


def canonicalize_model_points_yaw(
    points_model: np.ndarray,
    bounds: np.ndarray,
    yaw_deg: float,
    *,
    rescale_rotated_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate model points so the estimated front maps to canonical +Y.

    The transform convention matches ``annotate_obb_front.py``. When rescaling is
    enabled, the new conservative half-extents are calculated by rotating the eight
    annotated OBB corners, avoiding accidental unit-cube pruning at arbitrary yaw.
    """
    points = np.asarray(points_model, dtype=np.float32).reshape(-1, 3)
    half_extents = np.asarray(bounds, dtype=np.float32).reshape(3)
    yaw = np.deg2rad(float(yaw_deg))
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    model_to_canonical = np.eye(4, dtype=np.float32)
    model_to_canonical[:3, :3] = rotation
    canonical_to_model = np.eye(4, dtype=np.float32)
    canonical_to_model[:3, :3] = rotation.T
    rotated_points = points @ rotation.T
    rotated_bounds = (
        np.abs(rotation) @ half_extents if rescale_rotated_bounds else half_extents.copy()
    )
    return rotated_points, rotated_bounds, model_to_canonical, canonical_to_model


def load_rgb_stack(
    cache_root: Path,
    metadata: dict[str, Any],
    view_indices: Iterable[int],
    *,
    roll_degrees: Iterable[float] | None = None,
) -> np.ndarray:
    cache_root = Path(cache_root)
    indices = np.asarray(list(view_indices), dtype=np.int64)
    rolls = (
        np.zeros((len(indices),), dtype=np.float64)
        if roll_degrees is None
        else np.asarray(list(roll_degrees), dtype=np.float64)
    )
    if rolls.shape != (len(indices),):
        raise ValueError(f"roll_degrees must contain {len(indices)} values, got {rolls.shape}")
    images = []
    for view_index, roll in zip(indices.tolist(), rolls.tolist()):
        path = cache_root / metadata["image_relpaths"][view_index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if abs(float(roll)) > 1e-7:
                image = image.rotate(
                    float(roll),
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(0, 0, 0),
                )
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
    if not images:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)
    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("Selected cached images do not have a common RGB resolution")
    return np.stack(images, axis=0)


def project_model_point(
    point_model: np.ndarray,
    T_camera_model: np.ndarray,
    camera_params: np.ndarray,
) -> np.ndarray:
    """Project one model-frame point with the cached perspective camera."""
    point_h = np.r_[np.asarray(point_model, dtype=np.float64).reshape(3), 1.0]
    camera_h = np.asarray(T_camera_model, dtype=np.float64).reshape(4, 4) @ point_h
    if abs(float(camera_h[3])) < 1e-12:
        raise ValueError("Invalid homogeneous camera projection")
    camera = camera_h[:3] / camera_h[3]
    if not np.isfinite(camera).all() or float(camera[2]) <= 0:
        raise ValueError("Model-frame roll probe is not in front of the camera")
    params = np.asarray(camera_params, dtype=np.float64)
    if params.shape in ((4, 4), (16,)):
        intrinsic = params.reshape(4, 4)
    elif params.shape in ((3, 3), (9,)):
        intrinsic = np.eye(4, dtype=np.float64)
        intrinsic[:3, :3] = params.reshape(3, 3)
    else:
        raise ValueError(
            "Expected 3x3, flat-9, 4x4, or flat-16 perspective intrinsics, "
            f"got {params.shape}"
        )
    return np.asarray(
        [
            intrinsic[0, 0] * camera[0] / camera[2] + intrinsic[0, 2],
            intrinsic[1, 1] * camera[1] / camera[2] + intrinsic[1, 2],
        ],
        dtype=np.float64,
    )


def camera_roll_to_upright_degrees(
    T_camera_model: np.ndarray,
    camera_params: np.ndarray,
    *,
    up_step: float = 0.25,
) -> float:
    """Return the in-plane image rotation that maps model +Z to image-up.

    Cached ShapeR perspective views inherit wearable-camera roll. FF's training
    renderers use gravity-upright images and the flow context has no camera-ray input,
    so callers may opt into this deterministic rectification before DINO extraction.
    """
    origin_uv = project_model_point(np.zeros(3), T_camera_model, camera_params)
    up_uv = project_model_point(
        np.asarray([0.0, 0.0, float(up_step)], dtype=np.float64),
        T_camera_model,
        camera_params,
    )
    delta = up_uv - origin_uv
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) < 1e-8:
        raise ValueError("Projected model +Z direction is degenerate")
    angle = float(np.degrees(np.arctan2(delta[0], -delta[1])))
    return float((angle + 180.0) % 360.0 - 180.0)


def rotate_uv_same_canvas(
    uv: np.ndarray,
    angle_degrees: float,
    image_hw: Iterable[int],
) -> np.ndarray:
    """Apply PIL.Image.rotate's visual CCW transform to pixel coordinates."""
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    height, width = (int(value) for value in image_hw)
    center = np.asarray([(width - 1.0) * 0.5, (height - 1.0) * 0.5], dtype=np.float64)
    angle = np.radians(float(angle_degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    # Image coordinates are x-right/y-down. Positive PIL angles are visually CCW.
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return ((points - center) @ rotation.T + center).astype(np.float32)


def upright_rectified_observations(
    arrays: dict[str, np.ndarray],
    image_hw: np.ndarray,
    view_indices: Iterable[int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Rotate selected view UVs consistently with gravity-upright RGB images."""
    selected = np.asarray(list(view_indices), dtype=np.int64)
    image_hw = np.asarray(image_hw, dtype=np.int64)
    offsets = np.asarray(arrays["view_offsets"], dtype=np.int64)
    transformed = dict(arrays)
    transformed_uv = np.asarray(arrays["view_uv"], dtype=np.float32).copy()
    rolls = []
    for view_index in selected.tolist():
        roll = camera_roll_to_upright_degrees(
            arrays["Ts_camera_model"][view_index],
            arrays["camera_params"][view_index],
        )
        start, end = (int(value) for value in offsets[view_index : view_index + 2])
        transformed_uv[start:end] = rotate_uv_same_canvas(
            transformed_uv[start:end], roll, image_hw[view_index]
        )
        rolls.append(roll)
    transformed["view_uv"] = transformed_uv
    return transformed, np.asarray(rolls, dtype=np.float32)
