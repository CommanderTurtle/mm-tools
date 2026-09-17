import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torchvision.transforms import v2
from tqdm import tqdm
from utils.project import project_depth_to_world_patch_geometry_with_instance_mask

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def load_scene_metadata(room_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    camera_data = load_json(room_dir / "cameras.json")
    frames = sorted(camera_data["frames"], key=lambda frame: frame["frame_name"])
    intrinsics = np.asarray(
        [frame["depth_intrinsics"] for frame in frames],
        dtype=np.float32,
    )
    c2ws = np.asarray(
        [frame["camera_to_world"] for frame in frames],
        dtype=np.float32,
    )[:, :3, :]
    return frames, intrinsics, c2ws


def load_rgbs_by_frames(room_dir: Path, frames: list[dict]) -> np.ndarray:
    rgbs = []
    for frame in frames:
        image = cv2.imread(str(room_dir / "frames" / frame["frame_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(room_dir / "frames" / frame["frame_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgbs.append(image)
    return np.stack(rgbs, axis=0)


def load_depths_by_frames(room_dir: Path, frames: list[dict]) -> np.ndarray:
    depths = []
    for frame in frames:
        depth_archive = np.load(room_dir / "depth" / frame["depth_name"])
        if "depth" in depth_archive:
            depth = depth_archive["depth"]
        elif "arr_0" in depth_archive:
            depth = depth_archive["arr_0"]
        else:
            raise KeyError(f"Unsupported depth archive keys: {list(depth_archive.keys())}")
        depths.append(depth.astype(np.float32))
    return np.stack(depths, axis=0)



def points_in_polygon_xy(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    poly_x = polygon_xy[:, 0]
    poly_y = polygon_xy[:, 1]

    x1 = poly_x
    y1 = poly_y
    x2 = np.roll(poly_x, -1)
    y2 = np.roll(poly_y, -1)

    y_between = (y1 > y[:, None]) != (y2 > y[:, None])
    denom = y2 - y1
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    x_intersections = (x2 - x1) * (y[:, None] - y1) / denom + x1
    crossings = y_between & (x[:, None] < x_intersections)
    return np.count_nonzero(crossings, axis=1) % 2 == 1


def compute_polygon_centroid(polygon_xy: np.ndarray) -> np.ndarray:
    return polygon_xy.mean(axis=0)


def compute_prune_mask(points: np.ndarray, room_boundary: dict) -> np.ndarray:
    polygon_xy = np.asarray(room_boundary["polygon_xy"], dtype=np.float32)
    centroid_xy = compute_polygon_centroid(polygon_xy)
    padded_polygon_xy = centroid_xy[None, :] + 1.1 * (polygon_xy - centroid_xy[None, :])
    z_min = float(room_boundary.get("z_min", -np.inf))
    z_max = float(room_boundary.get("z_max", np.inf))

    points_flat = points.reshape(-1, 3)
    inside_xy = points_in_polygon_xy(points_flat[:, :2], padded_polygon_xy)
    inside_z = (points_flat[:, 2] >= (z_min - 1e-5)) & (points_flat[:, 2] <= (z_max + 1e-5))
    inside_mask = inside_xy & inside_z
    return (~inside_mask).reshape(points.shape[:2])

def load_masks_by_frames(room_dir: Path, frames: list[dict]) -> np.ndarray:
    masks = []
    for frame in frames:
        mask_archive = np.load(room_dir / "masks" / frame["mask_name"])
        if "mask" in mask_archive:
            mask = mask_archive["mask"]
        elif "arr_0" in mask_archive:
            mask = mask_archive["arr_0"]
        else:
            raise KeyError(f"Unsupported mask archive keys: {list(mask_archive.keys())}")
        masks.append(mask.astype(np.int32))
    return np.stack(masks, axis=0)

def load_data(room_dir_str):
    room_dir = Path(room_dir_str)
    boxes_data = load_json(room_dir / "oriented_bboxes.json")
    room_boundary = load_json(room_dir / "room_boundary.json")
    frames, intrinsics, c2ws = load_scene_metadata(room_dir)
    rgbs = load_rgbs_by_frames(room_dir, frames)
    depths = load_depths_by_frames(room_dir, frames)
    masks = load_masks_by_frames(room_dir, frames)

    num_frames, rgb_h, rgb_w, _ = rgbs.shape
    depth_h, depth_w = int(depths.shape[1]), int(depths.shape[2])
    if (depth_h, depth_w) != (rgb_h, rgb_w):
        sx = rgb_w / depth_w
        sy = rgb_h / depth_h
        depths = np.stack(
            [cv2.resize(depths[i], (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST) for i in range(num_frames)],
            axis=0,
        )
        masks = np.stack(
            [cv2.resize(masks[i], (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST) for i in range(num_frames)],
            axis=0,
        )
        intrinsics = intrinsics.copy()
        intrinsics[:, 0, :] *= sx
        intrinsics[:, 1, :] *= sy

    points, _, _, _ = (
        project_depth_to_world_patch_geometry_with_instance_mask(
            depths,
            intrinsics,
            c2ws,
            masks,
            downsample=1,
        )
    )

    prune_mask = compute_prune_mask(points, room_boundary)
    points = points.reshape(num_frames, rgb_h, rgb_w, 3)
    prune_mask = prune_mask.reshape(num_frames, rgb_h, rgb_w)

    return rgbs, points, prune_mask
