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
data_dir = os.environ.get(
    "FIRE3D_IMAGINARIUM_ROOT",
    str(_REPO_ROOT / "data" / "imaginarium"),
)
num_videos_per_scene = 1
dataset_name = "Imaginarium"

def load_imaginarium_data(data_root=None):

    root = data_root or data_dir
    renders_dir = os.path.join(root, 'renders')
    transforms_dir = os.path.join(root, 'transforms')

    scene_ids = sorted(os.listdir(renders_dir))
    whitelist_path = os.path.join(root, "whitelist.txt")
    if os.path.isfile(whitelist_path):
        with open(whitelist_path, "r") as handle:
            whitelist = {line.strip() for line in handle if line.strip()}
        scene_ids = [scene_id for scene_id in scene_ids if scene_id in whitelist]
        print(f"Using {len(scene_ids)} scenes from the release whitelist")

    data_list = []

    for scene_id in tqdm(scene_ids, desc="Loading Imaginarium data"):
        for video_i in range(num_videos_per_scene):
            data_dict = {
                'scene_id': scene_id,
                'video_id': video_i,
                "data_name": f"{dataset_name}_{scene_id}_{video_i}",
                'camera_path': os.path.join(renders_dir, scene_id, f'{video_i}.json'),
                'frames_dir': resolve_frames_dir(
                    os.path.join(renders_dir, scene_id, f'{video_i}_frames'),
                    dataset_subdir='Imaginarium',
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


def _apply_camera_pose_override(
    *,
    camera_override_path,
    intrinsics,
    c2ws,
    rgbs,
    depths,
    masks,
):
    """Select matched frames and replace GT c2w poses from a benchmark NPZ."""
    override = np.load(camera_override_path, allow_pickle=False)
    required = {
        "image_names",
        "frame_indices",
        "gt_c2w",
        "aligned_predicted_c2w",
    }
    missing = sorted(required.difference(override.files))
    if missing:
        raise KeyError(
            f"Camera override {camera_override_path} is missing keys: {missing}"
        )

    frame_indices = np.asarray(override["frame_indices"], dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError(
            f"Camera override frame_indices must be a non-empty vector: "
            f"{camera_override_path}"
        )
    if np.any(frame_indices < 0) or np.any(frame_indices >= len(c2ws)):
        raise IndexError(
            f"Camera override frame indices are outside [0, {len(c2ws)}): "
            f"{camera_override_path}"
        )
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise ValueError(
            f"Camera override contains duplicate frame indices: {camera_override_path}"
        )

    expected_gt = np.asarray(override["gt_c2w"], dtype=np.float64)
    aligned_c2ws = np.asarray(
        override["aligned_predicted_c2w"], dtype=np.float64
    )
    if expected_gt.shape != (len(frame_indices), 4, 4):
        raise ValueError(
            f"Unexpected gt_c2w shape {expected_gt.shape}: {camera_override_path}"
        )
    if aligned_c2ws.shape != expected_gt.shape:
        raise ValueError(
            f"Unexpected aligned_predicted_c2w shape {aligned_c2ws.shape}: "
            f"{camera_override_path}"
        )

    source_gt = np.asarray(c2ws[frame_indices], dtype=np.float64)
    gt_residual = float(
        np.max(np.abs(source_gt - expected_gt[:, :3, :]))
    )
    if gt_residual > 1e-5:
        raise ValueError(
            "Camera override GT poses do not match the selected Imaginarium "
            f"frames (max residual {gt_residual:.3e}): {camera_override_path}"
        )

    return {
        "intrinsics": intrinsics[frame_indices],
        "c2ws": aligned_c2ws[:, :3, :],
        "rgbs": rgbs[frame_indices],
        "depths": depths[frame_indices],
        "masks": masks[frame_indices],
        "metadata": {
            "path": os.path.abspath(camera_override_path),
            "image_names": np.asarray(override["image_names"]).astype(str).tolist(),
            "frame_indices": frame_indices.tolist(),
            "num_source_frames": int(len(c2ws)),
            "num_selected_frames": int(len(frame_indices)),
            "gt_pose_max_residual": gt_residual,
        },
    }


def get_inference_data(
    data_list,
    idx,
    image_downsample=1,
    camera_override_path=None,
):
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

    camera_override_metadata = None
    if camera_override_path is not None:
        overridden = _apply_camera_pose_override(
            camera_override_path=camera_override_path,
            intrinsics=intrinsics,
            c2ws=c2ws,
            rgbs=rgbs,
            depths=depths,
            masks=masks,
        )
        intrinsics = overridden["intrinsics"]
        c2ws = overridden["c2ws"]
        rgbs = overridden["rgbs"]
        depths = overridden["depths"]
        masks = overridden["masks"]
        camera_override_metadata = overridden["metadata"]

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
        "camera_pose_override": camera_override_metadata,
    }
