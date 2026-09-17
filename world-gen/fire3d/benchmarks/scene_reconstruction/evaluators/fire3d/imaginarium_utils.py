
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.interpolate import griddata


REPO_ROOT = Path(__file__).resolve().parents[4]
gt_root_dir = os.environ.get(
    "FF_IMAGINARIUM_ROOT", str(REPO_ROOT / "data/evaluation/imaginarium")
)


def as_mesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        return trimesh.util.concatenate(mesh_or_scene.dump())
    return trimesh.load(mesh_or_scene, force="mesh", process=False)


def load_glb(mesh_path):
    mesh_or_scene = trimesh.load(mesh_path, force="scene", process=False)
    mesh = as_mesh(mesh_or_scene)
    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rot_x_90)
    # mesh.apply_scale(0.001)
    return mesh


def _normalize_pred_obb(entry, index):
    source = dict(entry)
    if "obb_world" in source:
        obb = dict(source["obb_world"])
    elif "world" in source:
        obb = dict(source["world"])
    else:
        obb = source

    rotation = obb.get("rotation", obb.get("rotation_quat_wxyz"))
    if rotation is None and "rotation_xyzw" in obb:
        x, y, z, w = obb["rotation_xyzw"]
        rotation = [w, x, y, z]
    if rotation is None:
        rotation = [1.0, 0.0, 0.0, 0.0]

    bbox = {
        "translation": obb["translation"],
        "rotation": rotation,
        "scale": obb["scale"],
        "confidence": source.get("confidence", source.get("source_prob", 1.0)),
        "index": int(source.get("index", index)),
        "name": source.get("name", f"pred_{index:04d}"),
        "category": source.get("category", source.get("source_object_name")),
        "source": source,
    }
    if "T_world_object" in obb:
        bbox["T_world_object"] = obb["T_world_object"]
    return bbox


def load_pred_obbs_and_meshes(pred_results_scene_dir, pred_obbs_path):
    with open(pred_obbs_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    pred_entries = pred_data.get("objects", pred_data)
    pred_bboxes = []
    pred_meshes = {}

    for pred_i, entry in enumerate(pred_entries, start=1):
        bbox = _normalize_pred_obb(entry, pred_i)
        pred_bboxes.append(bbox)

        mesh_candidates = []
        if bbox.get("name"):
            mesh_candidates.extend(
                [
                    Path(pred_results_scene_dir) / f"{bbox['name']}.ply",
                    Path(pred_results_scene_dir) / f"{bbox['name']}.glb",
                    Path(pred_results_scene_dir) / f"{bbox['name']}.obj",
                ]
            )
        if entry.get("mesh_path"):
            mesh_candidates.append(Path(entry["mesh_path"]))

        mesh_path = next((path for path in mesh_candidates if path.exists()), None)
        if mesh_path is not None:
            pred_meshes[bbox["index"]] = as_mesh(
                trimesh.load(mesh_path, force="mesh", process=False)
            )

    return pred_bboxes, pred_meshes


def _read_single_rgb(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def read_rgbs(
    frames_dir, height=None, width=None, parallel=False, max_workers=None, ext="jpg"
):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(f".{ext}")])
    filepaths = [os.path.join(frames_dir, f) for f in frame_files]
    n_frames = len(filepaths)

    if height is not None and width is not None:
        result = np.empty((n_frames, height, width, 3), dtype=np.float32)

        def _read_into(args):
            idx, fp = args
            result[idx] = _read_single_rgb(fp)

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_read_into, enumerate(filepaths)))
        else:
            for idx, fp in enumerate(filepaths):
                result[idx] = _read_single_rgb(fp)
        return result

    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rgb_images = list(executor.map(_read_single_rgb, filepaths))
    else:
        rgb_images = [_read_single_rgb(fp) for fp in filepaths]

    return np.stack(rgb_images)


def _read_single_depth(filepath):
    depth_archive = np.load(filepath)
    if "depth" in depth_archive:
        return depth_archive["depth"].astype(np.float32)
    if "arr_0" in depth_archive:
        return depth_archive["arr_0"].astype(np.float32)
    raise KeyError(f"Unsupported depth archive keys: {list(depth_archive.keys())}")


def read_depths(depth_dir, height=None, width=None, parallel=False, max_workers=None):
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npz")])
    filepaths = [os.path.join(depth_dir, f) for f in depth_files]
    n_frames = len(filepaths)

    if height is not None and width is not None:
        result = np.empty((n_frames, height, width), dtype=np.float32)

        def _read_into(args):
            idx, fp = args
            result[idx] = _read_single_depth(fp)

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_read_into, enumerate(filepaths)))
        else:
            for idx, fp in enumerate(filepaths):
                result[idx] = _read_single_depth(fp)
        return result

    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            depth_images = list(executor.map(_read_single_depth, filepaths))
    else:
        depth_images = [_read_single_depth(fp) for fp in filepaths]

    return np.stack(depth_images)



def read_cameras(camera_path):
    with open(camera_path, "r") as f:
        camera_data = json.load(f)

    K = np.array(camera_data["K"])
    width = camera_data["width"]
    height = camera_data["height"]
    frames = camera_data["frames"]
    num_frames = len(frames)

    intrinsics = K.reshape(1, 3, 3).repeat(num_frames, axis=0)
    c2ws = []
    for frame_data in frames:
        eye = np.array(frame_data["eye"])
        lookat = np.array(frame_data["lookat"])
        up_vec = np.array(frame_data["up"])

        forward = lookat - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up_vec)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        R_mat = np.column_stack([right, -up, forward])

        extrinsic_c2w = np.eye(4)
        extrinsic_c2w[:3, :3] = R_mat
        extrinsic_c2w[:3, 3] = eye
        c2ws.append(extrinsic_c2w[:3, :])

    c2ws = np.stack(c2ws)
    return intrinsics, c2ws, (height, width)


def _read_single_mask_v2(filepath):
    mask_archive = np.load(filepath)
    if "mask" in mask_archive:
        return mask_archive["mask"].astype(np.int32)
    if "arr_0" in mask_archive:
        return mask_archive["arr_0"].astype(np.int32)
    raise KeyError(f"Unsupported mask archive keys: {list(mask_archive.keys())}")


def _read_single_mask_with_unique_v2(filepath):
    mask = _read_single_mask_v2(filepath)
    return mask, np.unique(mask)


def read_masks_v2(masks_dir, height=None, width=None, parallel=False, max_workers=None):
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npz")])
    filepaths = [os.path.join(masks_dir, f) for f in mask_files]
    n_frames = len(filepaths)

    all_unique_sets = []
    if height is not None and width is not None:
        result = np.empty((n_frames, height, width), dtype=np.int32)

        def _read_into(args):
            idx, fp = args
            mask, unique_vals = _read_single_mask_with_unique_v2(fp)
            result[idx] = mask
            return unique_vals

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                all_unique_sets = list(executor.map(_read_into, enumerate(filepaths)))
        else:
            for idx, fp in enumerate(filepaths):
                all_unique_sets.append(_read_into((idx, fp)))

        existing_indices = np.unique(np.concatenate(all_unique_sets)).tolist()
        return result, existing_indices

    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_read_single_mask_with_unique_v2, filepaths))
    else:
        results = [_read_single_mask_with_unique_v2(fp) for fp in filepaths]

    mask_images = [r[0] for r in results]
    all_unique_sets = [r[1] for r in results]
    existing_indices = np.unique(np.concatenate(all_unique_sets)).tolist()
    return np.stack(mask_images), existing_indices



def project_depth_to_points(rgbs, depth, masks, intrinsics, c2ws, downsample=8):
    N_im, H, W = depth.shape

    H_ds = H // downsample
    W_ds = W // downsample

    depth_ds = depth[:, ::downsample, ::downsample]
    rgbs_ds = rgbs[:, ::downsample, ::downsample, :]
    masks_ds = masks[:, ::downsample, ::downsample]

    depth_ds = depth_ds[:, :H_ds, :W_ds]
    rgbs_ds = rgbs_ds[:, :H_ds, :W_ds, :]
    masks_ds = masks_ds[:, :H_ds, :W_ds]

    invalid_depth_max = 50.0
    invalid_mask = (
        np.isinf(depth_ds)
        | np.isnan(depth_ds)
        | (depth_ds > invalid_depth_max)
        | (depth_ds <= 0)
    )
    if invalid_mask.any():
        for i in range(N_im):
            valid = ~invalid_mask[i]
            if not valid.any():
                depth_ds[i] = 0
                continue
            if invalid_mask[i].sum() == 0:
                continue
            points_valid = np.column_stack(np.where(valid))
            values_valid = depth_ds[i][valid]
            points_invalid = np.column_stack(np.where(invalid_mask[i]))
            filled = griddata(
                points_valid,
                values_valid,
                points_invalid,
                method="nearest",
                fill_value=0.0,
            )
            depth_ds[i][invalid_mask[i]] = filled

    half_ds = downsample // 2
    y_ds = np.arange(H_ds, dtype=np.float32) * downsample + half_ds
    x_ds = np.arange(W_ds, dtype=np.float32) * downsample + half_ds

    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    z_cam = depth_ds
    x_cam = (x_ds[None, None, :] - cx) * z_cam / fx
    y_cam = (y_ds[None, :, None] - cy) * z_cam / fy

    R = c2ws[:, :, :3]
    t = c2ws[:, :, 3]

    cam_coords = np.stack([x_cam, y_cam, z_cam], axis=-1)
    cam_flat = cam_coords.reshape(N_im, -1, 3).transpose(0, 2, 1)
    world_flat = np.matmul(R, cam_flat) + t[:, :, None]

    world_coords = world_flat.transpose(0, 2, 1).reshape(-1, 3)
    rgbs_ds = rgbs_ds.reshape(-1, 3)
    masks_ds = masks_ds.reshape(-1)

    return world_coords, rgbs_ds, masks_ds




def get_point_cloud(scene_name, downsample=1):
    # get scene point cloud from loading rgbs, depth, and cameras
    gt_renders_dir = os.path.join(gt_root_dir, "renders", scene_name)
    camera_path = os.path.join(gt_renders_dir, "0.json")
    frames_dir = os.path.join(gt_renders_dir, "0_frames")
    depth_dir = os.path.join(gt_renders_dir, "0_depth")

    intrinsics, c2ws, (height, width) = read_cameras(camera_path)

    rgbs = read_rgbs(frames_dir, height, width, parallel=True, max_workers=8)
    depths = read_depths(depth_dir, height, width, parallel=True, max_workers=8)
    masks, _ = read_masks_v2(
        os.path.join(gt_renders_dir, "0_masks"),
        height,
        width,
        parallel=True,
        max_workers=8,
    )

    points, points_rgbs, points_masks = project_depth_to_points(
        rgbs,
        depths,
        masks,
        intrinsics,
        c2ws,
        downsample=downsample,
    )

    return points, points_masks


def get_obbs_and_meshes(scene_name):
    gt_scene_obbs_path = os.path.join(gt_root_dir, "transforms", scene_name + ".pkl")
    gt_scene_meshes_dir = os.path.join(gt_root_dir, "scenes", scene_name)

    with open(gt_scene_obbs_path, "rb") as f:
        objects_transforms = pickle.load(f)

    latent_names = sorted(list(objects_transforms.keys()))

    gt_obbs = []
    gt_meshes = {}

    for latent_name in latent_names:
        if not latent_name.startswith("object_"):
            continue

        object_i = int(latent_name.split("_")[-1])
        obj_data = objects_transforms[latent_name]
        scale = obj_data["scale"]
        angles = obj_data["angles"]
        trans = obj_data["trans"]

        mesh_i = load_glb(os.path.join(gt_scene_meshes_dir, f"{latent_name}.glb"))

        total_transform = trimesh.transformations.compose_matrix(
            scale=[scale, scale, scale],
            shear=None,
            angles=[angles[0], angles[1], angles[2]],
            translate=[trans[0], trans[1], trans[2]],
            perspective=None,
        )
        mesh_i.apply_transform(total_transform)

        quat = trimesh.transformations.quaternion_from_euler(
            angles[0], angles[1], angles[2], axes="sxyz"
        )
        gt_obb = {
            "translation": [float(v) for v in trans],
            "rotation": np.asarray(quat, dtype=np.float64).astype(float).tolist(),
            "scale": [float(scale), float(scale), float(scale)],
            "index": object_i,
            "name": latent_name,
            "transform_angles": [float(v) for v in angles],
        }
        gt_obbs.append(gt_obb)
        gt_meshes[object_i] = mesh_i

    return gt_obbs, gt_meshes
