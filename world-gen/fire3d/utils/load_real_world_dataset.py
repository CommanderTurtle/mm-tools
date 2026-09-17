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



def load_scene_metadata(scene_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    camera_data = load_json(scene_dir / "cameras.json")
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

def quaternion_wxyz_to_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    quat_xyzw = np.asarray(
        [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ],
        dtype=np.float64,
    )
    return Rotation.from_quat(quat_xyzw).as_matrix()

def compute_prune_mask(points: np.ndarray, bg_box: dict, bg_pad_ratio: float) -> np.ndarray:
    center = np.asarray(bg_box["center"], dtype=np.float32)
    scale = np.asarray(bg_box["scale"], dtype=np.float32) * (1.0 + bg_pad_ratio)
    rotation = quaternion_wxyz_to_matrix(bg_box["rotation_quaternion_wxyz"]).astype(np.float32)
    deltas = points - center[None, None, :]
    local_points = deltas @ rotation
    half_extents = scale[None, None, :] * 0.5
    inside_mask = np.all(np.abs(local_points) <= (half_extents + 1e-6), axis=-1)
    return ~inside_mask

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

def load_data(scene_dir_str):
    scene_dir = Path(scene_dir_str)
    frames, intrinsics, c2ws = load_scene_metadata(scene_dir)
    rgbs = load_rgbs_by_frames(scene_dir, frames)
    depths = load_depths_by_frames(scene_dir, frames)
    masks = load_masks_by_frames(scene_dir, frames)

    boxes_data = load_json(scene_dir / "oriented_bboxes.json")
    boxes = boxes_data["boxes"]
    bg_box = next(box for box in boxes if int(box["instance_id"]) == 0)

    num_frames, rgb_h, rgb_w, _ = rgbs.shape

    # Resize depth and masks directly to the DINO patch grid (output_h x output_w) and scale
    # intrinsics accordingly, then project with downsample=1 so geometry matches DINO 1-to-1.
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
        intrinsics[:, 0, :] *= sx  # fx, cx
        intrinsics[:, 1, :] *= sy  # fy, cy

    points, _, _, _ = (
        project_depth_to_world_patch_geometry_with_instance_mask(
            depths,
            intrinsics,
            c2ws,
            masks,
            downsample=1,
        )
    )


    prune_mask = compute_prune_mask(points, bg_box, 0.1)
    points = points.reshape(num_frames, rgb_h, rgb_w, 3)
    prune_mask = prune_mask.reshape(num_frames, rgb_h, rgb_w)
    return rgbs, points, prune_mask