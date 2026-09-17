from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


SCENE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def clean_scene_id(value: object) -> str:
    scene = SCENE_ID.sub("-", str(value or "local-scene").strip()).strip(".-")
    if not scene:
        scene = "local-scene"
    return scene[:96]


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for item in archive.infolist():
        normalized = Path(item.filename.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Unsafe path in input bundle: {item.filename}")
        unix_mode = item.external_attr >> 16
        if unix_mode and (unix_mode & 0o170000) == 0o120000:
            raise ValueError(f"Symlinks are not accepted in input bundles: {item.filename}")
        members.append(item)
    return members


def extract_bundle(bundle: Path, destination: Path) -> tuple[Path, Path, Path | None]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        members = _safe_members(archive)
        archive.extractall(destination, members)

    def unique(names: set[str], required: bool = True) -> Path | None:
        found = [path for path in destination.rglob("*") if path.is_file() and path.name.lower() in names]
        if not found and required:
            raise ValueError(f"Bundle is missing one of: {', '.join(sorted(names))}")
        if len(found) > 1:
            raise ValueError(f"Bundle contains more than one {sorted(names)} candidate.")
        return found[0] if found else None

    rgb = unique({"rgb.jpeg", "rgb.jpg", "rgb.png", "image.jpeg", "image.jpg", "image.png"})
    pcd = unique({"aligned_pcd.ply", "pointcloud.ply", "points.ply"})
    camera = unique({"camera.json"}, required=False)
    assert rgb is not None and pcd is not None
    return rgb, pcd, camera


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if width < 64 or height < 64 or width % 2 or height % 2:
        raise ValueError("Fire3D RGB inputs must be even-sized and at least 64×64 pixels.")
    return width, height


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(65536).decode("ascii", errors="ignore")
    match = re.search(r"^element\s+vertex\s+(\d+)\s*$", header, re.MULTILINE)
    if not match or "end_header" not in header:
        raise ValueError("The aligned point cloud is not a readable PLY with a vertex count.")
    return int(match.group(1))


def camera_payload(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> dict[str, Any]:
    if min(fx, fy) <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    grid_height, grid_width = height // 2, width // 2
    crop_height, crop_width = (grid_height // 16) * 16, (grid_width // 16) * 16
    top = (grid_height - crop_height) // 2
    left = (grid_width - crop_width) // 2
    frame = {"eye": [0.0, 0.0, 0.0], "lookat": [0.0, 0.0, 1.0], "up": [0.0, -1.0, 0.0]}
    native = {
        "frame": frame,
        "forward": [0.0, 0.0, 1.0],
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "width": width,
        "height": height,
        "crop_top": 0,
        "crop_left": 0,
        "source_size": [height, width],
    }
    reconstruction = {
        **native,
        "K": [[fx, 0.0, cx - 2 * left], [0.0, fy, cy - 2 * top], [0.0, 0.0, 1.0]],
        "width": 2 * crop_width,
        "height": 2 * crop_height,
        "crop_top": 2 * top,
        "crop_left": 2 * left,
    }
    return {"schema": "fire3d_single_image_camera_v1", "native": native, "reconstruction": reconstruction}


def write_camera(path: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema") != "fire3d_single_image_camera_v1":
        raise ValueError("camera.json must use fire3d_single_image_camera_v1.")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stage_scene(
    rgb: Path,
    pointcloud: Path,
    destination: Path,
    scene_id: str,
    *,
    camera: Path | None = None,
    intrinsics: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    scene_id = clean_scene_id(scene_id)
    width, height = image_dimensions(rgb)
    expected = (height // 2) * (width // 2)
    actual = ply_vertex_count(pointcloud)
    if actual != expected:
        raise ValueError(
            f"aligned_pcd.ply has {actual:,} vertices; RGB {width}×{height} requires "
            f"exactly {expected:,} organized H/2×W/2 points."
        )
    scene = destination / "single_image" / "data" / scene_id
    scene.mkdir(parents=True, exist_ok=True)
    with Image.open(rgb) as image:
        image.convert("RGB").save(scene / "rgb.jpeg", format="JPEG", quality=97, subsampling=0)
    shutil.copy2(pointcloud, scene / "aligned_pcd.ply")
    if camera:
        payload = json.loads(camera.read_text(encoding="utf-8"))
    else:
        fx, fy, cx, cy = intrinsics or (
            1.2 * max(width, height),
            1.2 * max(width, height),
            width / 2,
            height / 2,
        )
        payload = camera_payload(width, height, fx, fy, cx, cy)
    write_camera(scene / "camera.json", payload)
    root = destination / "single_image"
    (root / "single_image_valid.txt").write_text(scene_id + "\n", encoding="utf-8")
    return {
        "scene_id": scene_id,
        "width": width,
        "height": height,
        "point_count": actual,
        "dataset_root": str(destination),
        "camera": payload,
    }


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            depth = np.asarray(image)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("Depth input must be a single-channel PNG/TIFF or a 2-D NPY array.")
    return np.asarray(depth, dtype=np.float32)


def prepare_rgbd(
    rgb: Path,
    depth_path: Path,
    destination: Path,
    scene_id: str,
    *,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    minimum_depth: float,
    maximum_depth: float,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    width, height = image_dimensions(rgb)
    depth = _load_depth(depth_path) * float(depth_scale)
    target_width, target_height = width // 2, height // 2
    depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    u, v = np.meshgrid(np.arange(target_width), np.arange(target_height))
    source_u = (u.astype(np.float32) + 0.5) * 2.0 - 0.5
    source_v = (v.astype(np.float32) + 0.5) * 2.0 - 0.5
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z >= minimum_depth) & (z <= maximum_depth)
    x = (source_u - float(cx)) * z / float(fx)
    y = (source_v - float(cy)) * z / float(fy)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    points[~valid.reshape(-1)] = np.nan
    if int(valid.sum()) < 512:
        raise ValueError("Fewer than 512 valid metric depth samples remain after filtering.")
    cloud = destination / "prepared-aligned-pcd.ply"
    vertices = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(cloud)
    staged = stage_scene(
        rgb,
        cloud,
        destination / "dataset",
        scene_id,
        intrinsics=(fx, fy, cx, cy),
    )
    staged.update(
        {
            "valid_depth_points": int(valid.sum()),
            "depth_scale": float(depth_scale),
            "depth_range_m": [float(minimum_depth), float(maximum_depth)],
        }
    )
    return staged


def package_dataset(dataset_root: Path, destination: Path) -> Path:
    root = dataset_root / "single_image"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return destination


def make_protocol(template: Path, destination: Path, controls: dict[str, Any]) -> Path:
    payload = deepcopy(json.loads(template.read_text(encoding="utf-8")))
    payload["name"] = "fire3d_single_image_mmtools_5090_v1"
    payload["status"] = "local-optimized"
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["description"] = (
        "Fire3D single-image release semantics with BF16 flow residency and "
        "batch-one bounds for a 32 GiB RTX 5090."
    )
    payload["input_rgb"]["loader_options"]["wall_alignment"] = bool(controls.get("wall_alignment", False))
    reconstruction = payload["reconstruction"]
    flow = reconstruction["flow"]
    flow.update(
        {
            "inference_num_steps": int(controls.get("inference_steps", 12)),
            "pbr_guidance_strength": float(controls.get("pbr_guidance", 3.0)),
            "seed": int(controls.get("seed", 20260717)),
            "max_cond_len": int(controls.get("max_condition_points", 1024)),
            "pbr_max_cond_len": int(controls.get("pbr_condition_points", 512)),
            "object_batch_size": 1,
            "ss_shape_object_batch_size": 1,
            "pbr_object_batch_size": 1,
        }
    )
    reconstruction["conditioning"]["anyup_frame_batch_size"] = 1
    reconstruction["geometry_stage"]["mesh_decode_batch_size"] = 1
    reconstruction["appearance_decode"]["object_chunk_size"] = 1
    mesh = reconstruction["mesh_postprocess"]
    mesh.update(
        {
            "object_batch_size": 1,
            "uv_cpu_workers": int(controls.get("uv_workers", 4)),
            "texture_size": int(controls.get("texture_size", 512)),
            "decimation_target": int(controls.get("decimation_target", 100000)),
            "projection_raster_instance_batch_size": 1,
        }
    )
    reconstruction["execution"].update(
        {
            "model_cache": False,
            "read_workers": int(controls.get("read_workers", 2)),
            "detailed_profile": bool(controls.get("detailed_profile", True)),
        }
    )
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def default_intrinsics(controls: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    focal = float(controls.get("focal_length", 1.2 * max(width, height)))
    return (
        float(controls.get("focal_x", focal)),
        float(controls.get("focal_y", focal)),
        float(controls.get("principal_x", width / 2)),
        float(controls.get("principal_y", height / 2)),
    )
