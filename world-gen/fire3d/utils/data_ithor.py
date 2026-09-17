import numpy as np
import torch
import os
import pickle
import trimesh
from pathlib import Path
from utils.exact_camera_rgb import resolve_frames_dir
from utils.read_frames import (
    read_rgbs,
    read_depths,
    read_masks_v2,
    read_cameras,
)
from utils.transforms import point_augment, point_normalize
from utils.project import project_depth_to_points
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
data_dir = os.environ.get("FIRE3D_ITHOR_ROOT", str(_REPO_ROOT / "data" / "ithor"))
whitelist_path = os.path.join(data_dir, "whitelist.txt")
num_videos_per_scene = 1
dataset_name = "ithor"

def load_ithor_data(data_root=None):

    root = data_root or data_dir
    renders_dir = os.path.join(root, 'renders')
    transforms_dir = os.path.join(root, 'transforms')

    scene_ids = sorted(os.listdir(renders_dir))
    active_whitelist_path = os.path.join(root, 'whitelist.txt') if data_root else whitelist_path
    with open(active_whitelist_path, 'r') as f:
        whitelist = f.read().strip("\n").splitlines()
    scene_ids = [scene_id for scene_id in scene_ids if scene_id in whitelist]
    print(f"Using {len(scene_ids)} scenes from the whitelist")

    data_list = []

    for scene_id in tqdm(scene_ids, desc="Loading ITHOR data"):
        for video_i in range(num_videos_per_scene):
            data_dict = {
                'scene_id': scene_id,
                'video_id': video_i,
                "data_name": f"{dataset_name}_{scene_id}_{video_i}",
                'camera_path': os.path.join(renders_dir, scene_id, f'{video_i}.json'),
                'frames_dir': resolve_frames_dir(
                    os.path.join(renders_dir, scene_id, f'{video_i}_frames'),
                    dataset_subdir='ithor',
                    scene_id=scene_id,
                    video_id=video_i,
                ),
                'masks_dir': os.path.join(renders_dir, scene_id, f'{video_i}_masks'),
                'depth_dir': os.path.join(renders_dir, scene_id, f'{video_i}_depth'),
                'transforms_path': os.path.join(transforms_dir, f'{scene_id}.pkl'),
            }
            if os.path.exists(data_dict['camera_path']) \
                and os.path.exists(data_dict['frames_dir']) \
                and os.path.exists(data_dict['masks_dir']) \
                and os.path.exists(data_dict['depth_dir']) \
                and os.path.exists(data_dict['transforms_path']):
                data_list.append(data_dict)

    return data_list


def _load_gt_obbs(transforms_path, preprocess_transform):
    with open(transforms_path, 'rb') as f:
        objects_transforms = pickle.load(f)

    R = np.asarray(preprocess_transform[:3, :3], dtype=np.float64)
    t = np.asarray(preprocess_transform[:3, 3], dtype=np.float64)

    gt_obbs = []
    for latent_name in sorted(objects_transforms.keys()):
        if not latent_name.startswith("object_"):
            continue
        object_i = int(latent_name.split("_")[-1])
        obj_data = objects_transforms[latent_name]
        scale = float(obj_data["scale"])
        angles = obj_data["angles"]
        trans = np.asarray(obj_data["trans"], dtype=np.float64)

        new_trans = R @ trans + t
        quat = trimesh.transformations.quaternion_from_euler(
            angles[0], angles[1], angles[2], axes="sxyz"
        )
        gt_obbs.append({
            "index": object_i,
            "name": latent_name,
            "translate": [float(v) for v in new_trans],
            "rotation": [float(v) for v in quat],
            "scale": scale,
        })
    return gt_obbs


def get_inference_data(data_list, idx, image_downsample=1):
    data_dict = data_list[idx]
    camera_path = data_dict['camera_path']
    frames_dir = data_dict['frames_dir']
    depth_dir = data_dict['depth_dir']
    masks_dir = data_dict['masks_dir']
    transforms_path = data_dict['transforms_path']

    parallel_io = True

    intrinsics, c2ws, (height, width) = read_cameras(camera_path)

    rgbs = read_rgbs(frames_dir, height, width, parallel=parallel_io)
    depths = read_depths(depth_dir, height, width, parallel=parallel_io)
    masks, _ = read_masks_v2(masks_dir, height, width, parallel=parallel_io)

    points, points_rgbs = project_depth_to_points(
        rgbs, depths, intrinsics, c2ws,
        downsample=image_downsample,
    )

    H_ds = height // image_downsample
    W_ds = width // image_downsample
    masks_ds = masks[:, ::image_downsample, ::image_downsample][:, :H_ds, :W_ds]
    points_instance_masks = masks_ds.reshape(-1).astype(np.int64)

    points, norm_transform = point_normalize(points)

    gt_obbs = _load_gt_obbs(transforms_path, norm_transform)

    return {
        "points": points,
        "points_rgbs": points_rgbs,
        "rgbs": rgbs,
        "data_name": f'{data_dict["scene_id"]}_{data_dict["video_id"]}',
        "preprocess_transform": norm_transform,
        "points_instance_masks": points_instance_masks,
        "gt_obbs": gt_obbs,
        "intrinsics": intrinsics,
        "c2ws": c2ws,
        "height": height,
        "width": width,
        "depths": depths,
    }
