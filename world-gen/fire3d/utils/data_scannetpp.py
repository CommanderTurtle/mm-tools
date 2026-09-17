import numpy as np
import torch
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = Path(__file__).resolve().parents[1]
from utils.read_frames import (
    read_rgbs,
    read_depths,
    read_depths_pi3,
    read_masks_v2,
    read_cameras,
    read_cameras_scannetpp,
)
from utils.transforms import point_augment, point_normalize
from utils.project import project_depth_to_points
from utils.scannetpp_room_walls import (
    estimate_room_walls,
    keep_mask_for_points,
    wall_line_rotation,
)

DEFAULT_ROOT_DIR = os.environ.get(
    "FF_SCANNETPP_ROOT",
    str(REPO_ROOT / "data/scannetpp"),
)
dataset_name = "scannetpp"
ROTATION_DOWNSAMPLE = 16
DEFAULT_DEPTH_SUBDIR = "depth"
DEFAULT_DEPTH_INVALID_POLICY = "nearest_fill"
# Room-wall outlier pruning + wall-line yaw alignment. Wall alignment is on by
# default and can be disabled completely with FF_SCANNETPP_WALL_ALIGN=0. The
# older FF_SCANNETPP_WALL_PRUNE=0 behavior remains available: it keeps the
# legacy min-area-AABB rotation while disabling room-polygon pruning.


def scannetpp_root() -> str:
    """Current dataset root, allowing protocols to configure it after import."""

    return os.environ.get("FF_SCANNETPP_ROOT", DEFAULT_ROOT_DIR)


def scannetpp_depth_subdir() -> str:
    """Depth directory selected for every ScanNet++ pipeline stage.

    Keep the historical ``depth`` directory as the default.  Ablation
    protocols may select a sibling directory, but never an absolute or nested
    path that could escape the scene root.
    """

    value = os.environ.get("FF_SCANNETPP_DEPTH_SUBDIR", DEFAULT_DEPTH_SUBDIR)
    if value in {"", ".", ".."} or os.path.isabs(value) or os.path.basename(value) != value:
        raise ValueError(
            "FF_SCANNETPP_DEPTH_SUBDIR must be one relative directory name, "
            f"got {value!r}"
        )
    return value


def depth_invalid_policy() -> str:
    """How invalid/confidence-rejected depths enter projected point clouds."""

    value = os.environ.get(
        "FF_SCANNETPP_DEPTH_INVALID_POLICY", DEFAULT_DEPTH_INVALID_POLICY
    )
    if value not in {"nearest_fill", "drop"}:
        raise ValueError(
            "FF_SCANNETPP_DEPTH_INVALID_POLICY must be 'nearest_fill' or "
            f"'drop', got {value!r}"
        )
    return value


def depth_confidence_threshold() -> float | None:
    """Optional threshold for stored sigmoid Pi3 confidence probabilities."""

    raw = os.environ.get("FF_SCANNETPP_DEPTH_CONFIDENCE_THRESHOLD")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "FF_SCANNETPP_DEPTH_CONFIDENCE_THRESHOLD must be a float in [0,1], "
            f"got {raw!r}"
        ) from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            "FF_SCANNETPP_DEPTH_CONFIDENCE_THRESHOLD must be in [0,1], "
            f"got {raw!r}"
        )
    return value


def wall_alignment_default() -> bool:
    return os.environ.get("FF_SCANNETPP_WALL_ALIGN", "1").lower() not in {
        "0", "false", ""
    }


def wall_prune_default() -> bool:
    return os.environ.get("FF_SCANNETPP_WALL_PRUNE", "1").lower() not in {
        "0", "false", ""
    }


def load_scannetpp_data():
    root = scannetpp_root()
    depth_subdir = scannetpp_depth_subdir()
    scenes_dir = os.path.join(root, 'scenes')
    current_whitelist_path = os.path.join(root, "whitelist.txt")

    scene_ids = sorted(os.listdir(scenes_dir))

    # read the whitelist when one is published; otherwise use every scene present.
    # FF_SCANNETPP_WHITELIST=0 ignores it: the shared root is actively being
    # extended and a partial whitelist can drop scenes mid-experiment.
    use_whitelist = os.environ.get("FF_SCANNETPP_WHITELIST", "1") != "0"
    if use_whitelist and os.path.isfile(current_whitelist_path):
        with open(current_whitelist_path, 'r') as f:
            whitelist = [s.strip() for s in f.read().splitlines() if s.strip()]
        print(f"Using {len(whitelist)} scenes from the whitelist")
    else:
        whitelist = scene_ids
        print(f"No whitelist at {current_whitelist_path}; using all {len(whitelist)} scenes under {scenes_dir}")

    data_list = []

    for scene_id in tqdm(scene_ids, desc="Loading ScanNet++ data"):
        if scene_id not in whitelist:
            continue
        data_dict = {
            'scene_id': scene_id,
            "data_name": f"{dataset_name}_{scene_id}",
            'camera_path': os.path.join(scenes_dir, scene_id, f'camera.json'),
            'frames_dir': os.path.join(scenes_dir, scene_id, f'rgb'),
            'depth_dir': os.path.join(scenes_dir, scene_id, depth_subdir),
        }
        if os.path.exists(data_dict['camera_path']) \
            and os.path.exists(data_dict['frames_dir']) \
            and os.path.exists(data_dict['depth_dir']):
            data_list.append(data_dict)

    return data_list

def _rotation_transform_from_points(points):
    """
    points: numpy array of shape (N, 3)
    return: (center, angle) where center is shape (2,) and angle is in radians

    implementation details: we want the points cloud when projected to 2d xy plane, it should be as closed as possible to a AABB rectangle.

    Steps:
    1. points_xy = points[:, :2]
    2. rotate candidate angles: np.linspace(0, 2*np.pi, 360, endpoint=False)
    3. for each candidate, rotate points around center of pcd in 2d, and calculate the AABB rectangle area
    4. the candidate with the smallest AABB rectangle area is the best rotation angle
    """
    points = np.asarray(points)
    if points.shape[0] == 0:
        return np.zeros(2, dtype=np.float64), 0.0
    xy = points[:, :2].astype(np.float64, copy=False)
    center = xy.mean(axis=0)
    x = xy[:, 0] - center[0]
    y = xy[:, 1] - center[1]

    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    best_angle = 0.0
    best_area = np.inf
    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        xr = c * x - s * y
        yr = s * x + c * y
        area = float((xr.max() - xr.min()) * (yr.max() - yr.min()))
        if area < best_area:
            best_area = area
            best_angle = theta

    return center, best_angle


def rotate_points(points, center=None, angle=None, return_transform=False):
    """
    points: numpy array of shape (N, 3)
    return: numpy array of shape (N, 3)
    """
    points = np.asarray(points)
    if points.shape[0] == 0:
        if return_transform:
            return points, (np.zeros(2, dtype=np.float64), 0.0)
        return points

    if center is None or angle is None:
        center, angle = _rotation_transform_from_points(points)
    else:
        center = np.asarray(center, dtype=np.float64)

    dtype = points.dtype
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(points, dtype=np.float64)
    out[:, 2:] = points[:, 2:].astype(np.float64, copy=False)
    dx = points[:, 0].astype(np.float64, copy=False) - center[0]
    dy = points[:, 1].astype(np.float64, copy=False) - center[1]
    out[:, 0] = center[0] + c * dx - s * dy
    out[:, 1] = center[1] + s * dx + c * dy
    out = out.astype(dtype, copy=False)
    if return_transform:
        return out, (center, angle)
    return out

def default_max_frames() -> int:
    """Frame cap for scannetpp scenes; FF_SCANNETPP_MAX_FRAMES overrides.

    The shared roots carry up to 300 frames per scene, which is also the default
    cap. Reading the override here -- and nowhere else -- keeps perception,
    reconstruction, and camera sampling locked to the same frames.
    """

    value = int(os.environ.get("FF_SCANNETPP_MAX_FRAMES", "300"))
    if value <= 0:
        raise ValueError(f"FF_SCANNETPP_MAX_FRAMES must be positive, got {value}")
    return value


def load_scene_frames(data_dict, max_num_frames=None, parallel_io=True):
    """Read one scene's cameras/RGB/depth and apply the frame-count cap.

    Shared so that reconstruction sees exactly the frames perception saw.
    ``max_num_frames=None`` resolves to ``default_max_frames()``.
    """
    if max_num_frames is None:
        max_num_frames = default_max_frames()
    if max_num_frames <= 0:
        raise ValueError(f"max_num_frames must be positive, got {max_num_frames}")
    intrinsics, c2ws, (height, width) = read_cameras_scannetpp(data_dict['camera_path'])
    rgbs = read_rgbs(data_dict['frames_dir'], height, width, parallel=parallel_io, ext="png")
    depths = read_depths_pi3(
        data_dict['depth_dir'],
        height,
        width,
        parallel=parallel_io,
        confidence_threshold=depth_confidence_threshold(),
    )

    n = rgbs.shape[0]
    if n > max_num_frames:
        frame_idx = np.linspace(0, n - 1, num=max_num_frames)
        frame_idx = np.rint(frame_idx).astype(np.int64)
        frame_idx = np.clip(frame_idx, 0, n - 1)
        intrinsics = intrinsics[frame_idx]
        c2ws = c2ws[frame_idx]
        rgbs = rgbs[frame_idx]
        depths = depths[frame_idx]
    return rgbs, depths, intrinsics, c2ws, (height, width)


def scene_wall_alignment(
    rgbs,
    depths,
    intrinsics,
    c2ws,
    *,
    wall_alignment=None,
    wall_prune=None,
):
    """Resolve the shared scene yaw and rotation centre.

    Both perception and reconstruction call this so they share one frame; the
    reference grid is always ROTATION_DOWNSAMPLE regardless of the sampling
    stride the caller ultimately uses. Disabling wall alignment returns an
    identity transform without projecting a reference cloud or estimating walls.

    When wall alignment is enabled, wall polygons drive pruning and yaw. The
    legacy min-area-AABB yaw remains the deterministic fallback when estimation
    fails, and is also selected explicitly by FF_SCANNETPP_WALL_PRUNE=0.
    """
    if wall_alignment is None:
        wall_alignment = wall_alignment_default()
    if wall_prune is None:
        wall_prune = wall_prune_default()
    if not wall_alignment:
        return None, np.zeros(2, dtype=np.float64), 0.0, {
            "enabled": False,
            "method": "disabled",
            "rotation_deg": 0.0,
            "wall_pruning": False,
        }

    reference_points, _ = project_depth_to_points(
        rgbs,
        depths,
        intrinsics,
        c2ws,
        downsample=ROTATION_DOWNSAMPLE,
        invalid_policy=depth_invalid_policy(),
    )
    if not wall_prune:
        rotation_center, rotation_angle = _rotation_transform_from_points(
            reference_points
        )
        return None, rotation_center, rotation_angle, {
            "enabled": True,
            "method": "min_area_aabb",
            "rotation_deg": float(np.degrees(rotation_angle)),
            "wall_pruning": False,
        }

    try:
        estimate = estimate_room_walls(reference_points[:, :2])
        rotation_angle, rotation_diag = wall_line_rotation(estimate.polygons_xy)
        reference_keep = keep_mask_for_points(estimate, reference_points[:, :2])
        rotation_center = reference_points[reference_keep][:, :2].mean(axis=0)
        rotation_diag = {
            **rotation_diag,
            "enabled": True,
            "method": "wall_lines",
            "wall_pruning": True,
        }
    except Exception as exc:
        print(
            f"[data_scannetpp] wall alignment failed ({exc}); "
            "falling back to min-area-AABB rotation without pruning"
        )
        estimate = None
        rotation_center, rotation_angle = _rotation_transform_from_points(
            reference_points
        )
        rotation_diag = {
            "enabled": True,
            "method": "min_area_aabb_fallback",
            "rotation_deg": float(np.degrees(rotation_angle)),
            "wall_pruning": False,
            "wall_error": str(exc),
        }
    return estimate, rotation_center, rotation_angle, rotation_diag


def get_inference_data(
    data_list,
    idx,
    image_downsample=1,
    wall_alignment=None,
    wall_prune=None,
):

    if wall_alignment is None:
        wall_alignment = wall_alignment_default()
    if wall_prune is None:
        wall_prune = wall_prune_default()
    if not wall_alignment:
        wall_prune = False
    max_num_frames = default_max_frames()

    data_dict = data_list[idx]
    camera_path = data_dict['camera_path']
    frames_dir = data_dict['frames_dir']
    depth_dir = data_dict['depth_dir']

    rgbs, depths, intrinsics, c2ws, _ = load_scene_frames(
        data_dict, max_num_frames=max_num_frames
    )

    invalid_policy = depth_invalid_policy()
    confidence_threshold = depth_confidence_threshold()
    points, points_rgbs, depth_grid_keep = project_depth_to_points(
        rgbs, depths, intrinsics, c2ws,
        downsample=image_downsample,
        invalid_policy=invalid_policy,
        return_grid_keep_mask=True,
    )

    # Estimate the room-wall chains on the reference grid, prune points outside
    # the walls (monocular-depth flyers), and pick the yaw that maximises the
    # number of near-axis-parallel wall lines. If wall alignment is disabled,
    # skip estimation, pruning, and rotation together.
    wall_info = None
    grid_keep_mask = depth_grid_keep.copy()
    pruned_points = np.zeros((0, 3), dtype=points.dtype)
    estimate, rotation_center, rotation_angle, rotation_diag = scene_wall_alignment(
        rgbs,
        depths,
        intrinsics,
        c2ws,
        wall_alignment=wall_alignment,
        wall_prune=wall_prune,
    )
    if estimate is not None:
        keep = keep_mask_for_points(estimate, points[:, :2])
        pruned_points = points[~keep]
        points = points[keep]
        points_rgbs = points_rgbs[keep]
        # eval_perception computes DINO features on the full projected grid, so
        # it needs the full-length keep mask to drop the same entries.
        combined_keep = np.zeros(grid_keep_mask.shape, dtype=bool)
        combined_keep[depth_grid_keep] = keep
        grid_keep_mask = combined_keep
        wall_info = {
            "estimate_stats": estimate.stats,
            "rotation": rotation_diag,
        }

    # alignment of rotation
    if wall_alignment:
        points = rotate_points(points, center=rotation_center, angle=rotation_angle)

    points, norm_transform = point_normalize(points)

    if wall_info is not None:
        translation = np.asarray(norm_transform, dtype=np.float64)[:3, 3]
        pruned_final = rotate_points(
            pruned_points.astype(np.float64), center=rotation_center, angle=rotation_angle
        ) + translation[None, :]
        polygons_final = [
            rotate_points(np.asarray(poly, dtype=np.float64),
                          center=rotation_center, angle=rotation_angle)
            + translation[None, :2]
            for poly in estimate.polygons_xy
        ]
        # per-polygon aligned flags in original segment order
        counts = [np.asarray(poly).shape[0] for poly in estimate.polygons_xy]
        kept_flags = wall_info["rotation"]["segment_kept"]
        aligned_full = np.zeros(int(np.sum(counts)), dtype=bool)
        aligned_full[kept_flags] = wall_info["rotation"]["segment_aligned"]
        offsets = np.cumsum([0] + counts)
        wall_info.update({
            "rotation_center": rotation_center,
            "rotation_angle": float(rotation_angle),
            "polygons_final": polygons_final,
            "segment_aligned_per_polygon": [
                aligned_full[offsets[i]:offsets[i + 1]] for i in range(len(counts))
            ],
            "segment_kept_per_polygon": [
                np.asarray(kept_flags)[offsets[i]:offsets[i + 1]] for i in range(len(counts))
            ],
            "pruned_points_final": pruned_final,
        })

    result = {
        "points": points,
        "points_rgbs": points_rgbs,
        "rgbs": rgbs,
        "data_name": f'{data_dict["scene_id"]}',
        "preprocess_transform": norm_transform,
        "scene_alignment": rotation_diag,
        "wall_prune_info": wall_info,
        "depth_filter": {
            "invalid_policy": invalid_policy,
            "confidence_threshold": confidence_threshold,
            "source_grid_points": int(depth_grid_keep.size),
            "valid_depth_grid_points": int(depth_grid_keep.sum()),
            "retained_grid_points": int(grid_keep_mask.sum()),
        },
    }
    if not grid_keep_mask.all():
        result["mask"] = grid_keep_mask
    return result

if __name__ == "__main__":
    data_list = load_scannetpp_data()
    print(f"Total number of scenes: {len(data_list)}")

    save_dir = "./vis/scannetpp/"
    os.makedirs(save_dir, exist_ok=True)
    import open3d as o3d
    for idx in tqdm(range(len(data_list)), desc="Saving scenes", total=len(data_list)):
        data_dict = get_inference_data(data_list, idx, image_downsample=32)
        # save points and colors with open3d
        points = data_dict["points"]
        colors = data_dict["points_rgbs"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(save_dir, f"{idx:04d}_{data_dict['data_name']}.ply"), pcd)
