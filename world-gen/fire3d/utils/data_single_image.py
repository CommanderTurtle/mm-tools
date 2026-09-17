import os
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from utils.single_image_room_walls import (
    best_axis_aligned_yaw,
    estimate_walls,
    yaw_matrix,
)
from utils.transforms import point_normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = os.environ.get("FF_SINGLE_IMAGE_ROOT", str(REPO_ROOT / "data/single_image"))
dataset_name = "single_image"
CAMERA_FLIP = np.diag([1.0, -1.0, -1.0])

# Optional room-yaw canonicalisation before normalisation: rotate by
# 0/90/180/270 about the vertical so estimated walls land on the x and y axes
# once point_normalize moves the min corner to the origin. It is OFF by default;
# FF_SINGLE_IMAGE_WALL_ALIGN=1 or wall_align=True opts in. See
# utils/single_image_room_walls.py.
_WALL_ALIGN_CACHE: dict = {}


def single_image_root() -> str:
    """Current dataset root, allowing protocols to configure it after import."""

    return os.environ.get("FF_SINGLE_IMAGE_ROOT", DEFAULT_ROOT_DIR)


def wall_align_default() -> bool:
    return os.environ.get("FF_SINGLE_IMAGE_WALL_ALIGN", "0") not in {
        "0", "false", "False", ""
    }


def load_single_image_data():
    root = single_image_root()
    data_root = os.path.join(root, "data")
    with open(os.path.join(root, "single_image_valid.txt"), "r") as f:
        whitelist = f.read().strip("\n").splitlines()
    whitelist = [s.strip() for s in whitelist if s.strip()]
    available = set(os.listdir(data_root))
    data_list = sorted([s for s in whitelist if s in available])
    print(f"Using {len(data_list)} scenes from the whitelist")
    return data_list


def scene_camera(scene: str, *, native_resolution: bool = False) -> dict:
    """Return the input camera without importing legacy orchestration code."""

    root = Path(single_image_root())
    published = root / "data" / scene / "camera.json"
    if published.is_file():
        camera = json.loads(published.read_text(encoding="utf-8"))
        if camera.get("schema") != "fire3d_single_image_camera_v1":
            raise ValueError(f"Unsupported camera schema: {published}")
        return camera["native" if native_resolution else "reconstruction"]

    source = root / scene.lstrip("0")
    annotation = json.loads(
        (source / f"annotation_{scene}.json").read_text(encoding="utf-8")
    )
    intrinsics = np.asarray(annotation["camera_intrinsics"], dtype=np.float64)
    rotation = np.asarray(annotation["camera_pose_rot"], dtype=np.float64) @ CAMERA_FLIP
    eye = np.asarray(annotation["camera_pose_tran"], dtype=np.float64)
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    up = rotation @ np.array([0.0, -1.0, 0.0])
    full_height, full_width = np.load(
        source / f"depth_{scene}.npy", mmap_mode="r"
    ).shape
    frame = {
        "eye": eye.tolist(),
        "lookat": (eye + forward).tolist(),
        "up": up.tolist(),
    }
    if native_resolution:
        return {
            "frame": frame,
            "forward": forward.tolist(),
            "K": intrinsics.tolist(),
            "width": int(full_width),
            "height": int(full_height),
            "crop_top": 0,
            "crop_left": 0,
            "source_size": [int(full_height), int(full_width)],
        }
    grid_height, grid_width = full_height // 2, full_width // 2
    crop_height, crop_width = (grid_height // 16) * 16, (grid_width // 16) * 16
    top = (grid_height - crop_height) // 2
    left = (grid_width - crop_width) // 2
    return {
        "frame": frame,
        "forward": forward.tolist(),
        "K": [
            [intrinsics[0, 0], 0.0, intrinsics[0, 2] - 2 * left],
            [0.0, intrinsics[1, 1], intrinsics[1, 2] - 2 * top],
            [0.0, 0.0, 1.0],
        ],
        "width": 2 * crop_width,
        "height": 2 * crop_height,
        "crop_top": 2 * top,
        "crop_left": 2 * left,
        "source_size": [int(full_height), int(full_width)],
    }


def resolve_wall_yaw(points: np.ndarray, height: int, width: int, cache_key):
    """Yaw in {0, 90, 180, 270} that pulls the estimated walls onto x/y.

    A multiple of 90 degrees about the vertical keeps axis-parallel walls
    axis-parallel; all it changes is which side of the bounding box each wall
    lands on once `point_normalize` moves the min corner to the origin. Picking
    the yaw that minimises the length-weighted wall-to-axis distance therefore
    puts the walls on the origin planes for every scene, instead of wherever the
    camera happened to be pointing.
    """

    if cache_key in _WALL_ALIGN_CACHE:
        return _WALL_ALIGN_CACHE[cache_key]
    estimate = estimate_walls(points, height, width)
    decision = best_axis_aligned_yaw(points, estimate["segments"])
    decision["floor_z"] = estimate["floor_z"]
    decision["ceiling_z"] = estimate["ceiling_z"]
    _WALL_ALIGN_CACHE[cache_key] = decision
    return decision


def get_inference_data(data_list, idx, image_downsample=16, wall_align=None):
    data_item = data_list[idx]
    data_dir = os.path.join(single_image_root(), "data", data_item)
    if wall_align is None:
        wall_align = wall_align_default()

    rgb_path = os.path.join(data_dir, "rgb.jpeg")
    pcd_path = os.path.join(data_dir, "aligned_pcd.ply")

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]

    pcd = trimesh.load(pcd_path, process=False)
    points = np.asarray(pcd.vertices, dtype=np.float32)

    point_height, point_width = height // 2, width // 2
    assert points.shape[0] == point_height * point_width, (
        f"pcd has {points.shape[0]} points, expected "
        f"{point_height * point_width} for rgb {height}x{width}"
    )
    wall_decision = None
    if wall_align:
        # estimate on the full organized lattice, before any subsampling: the
        # normals the wall test needs come from the (H/2, W/2) grid structure
        wall_decision = resolve_wall_yaw(points, point_height, point_width, data_item)
        rotation = yaw_matrix(wall_decision["yaw_degrees"])
        points = points @ rotation[:3, :3].T

    points = points.reshape(point_height, point_width, 3)

    points_rgbs = cv2.resize(
        rgb,
        (point_width, point_height),
        interpolation=cv2.INTER_AREA,
    )

    stride = max(1, image_downsample // 2)
    downsampled_height = height // image_downsample
    downsampled_width = width // image_downsample
    points = points[::stride, ::stride, :][
        :downsampled_height, :downsampled_width
    ].reshape(-1, 3)
    points_rgbs = points_rgbs[::stride, ::stride, :][
        :downsampled_height, :downsampled_width
    ].reshape(-1, 3)

    ok = np.isfinite(points).all(axis=-1)
    points = points[ok]
    points_rgbs = points_rgbs[ok]

    points, norm_transform = point_normalize(points)
    if wall_decision is not None:
        # preprocess_transform must map RAW WORLD -> normalized, because callers
        # invert it to push predictions back (eval_perception uses
        # np.linalg.inv(preprocess_transform)). The yaw therefore composes on
        # the right of the normalisation, not after it.
        norm_transform = norm_transform @ yaw_matrix(wall_decision["yaw_degrees"])
    rgbs = rgb[None, ...]

    print("Loaded single image data:")
    print(f"  points shape: {points.shape}")
    print(f"  points_rgbs shape: {points_rgbs.shape}")
    print(f"  rgbs shape: {rgbs.shape}")
    print(f"  data_name: {data_item.replace('/', '_')}")

    payload = {
        "points": points,
        "points_rgbs": points_rgbs,
        "rgbs": rgbs,
        "data_name": data_item.replace("/", "_"),
        "preprocess_transform": norm_transform,
    }
    if wall_decision is not None:
        payload["wall_alignment"] = wall_decision
        print(f"  wall-aligned yaw: {wall_decision['yaw_degrees']} deg "
              f"(scores {wall_decision['scores']})")
    return payload
