"""Shared detection post-processing: pose decoding, box corners, NMS."""
from pathlib import Path

import numpy as np
import torch
import trimesh

from utils.discrete import continue_transform, discrete_transform
from utils.constants import SCALE_MIN, SCALE_MAX, NUM_BINS
from pytorch3d.ops import box3d_overlap
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from utils.discrete import continue_transform
import pickle
import os

# Unit box corners in [-0.5, 0.5]^3, used for get_box_corners
BOX_CORNER_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float32,
) - 0.5  # [8, 3]
BOX_CORNER_VERTICES_HOMO = np.concatenate(
    [BOX_CORNER_VERTICES, np.ones((8, 1), dtype=np.float32)], axis=-1
)  # [8, 4]


def get_box_corners(pos_bins, angle_bins, scale_bins):
    """Convert pose bins (pos, angle, scale) to 3D box corners [N, 8, 3]."""
    num_objects = pos_bins.shape[0]
    all_box_corners = []

    for obj_i in range(num_objects):
        d_trans = pos_bins[obj_i].reshape(3)
        d_angles = angle_bins[obj_i].reshape(3)
        d_scale = scale_bins[obj_i].reshape(1)
        d_trans = [float(x) for x in d_trans]
        d_angles = [float(x) for x in d_angles]
        d_scale = float(d_scale)
        scale, angles, trans = continue_transform(d_scale, d_angles, d_trans)
        scale = max(scale, np.power(2e-4, 1 / 3))

        composed_transform = trimesh.transformations.compose_matrix(
            scale=[scale, scale, scale],
            shear=None,
            angles=[angles[0], angles[1], angles[2]],
            translate=[trans[0], trans[1], trans[2]],
            perspective=None,
        )
        composed_transform = np.array(composed_transform).astype(np.float32)

        box_corner_vertices_homo_obj_i = BOX_CORNER_VERTICES_HOMO @ composed_transform.T
        box_corner_vertices_obj_i = box_corner_vertices_homo_obj_i[:, :3]
        all_box_corners.append(box_corner_vertices_obj_i)

    return np.stack(all_box_corners, axis=0)  # [num_objects, 8, 3]


def _pose_bins_to_world_matrix(pos_bins_row, angle_bins_row, scale_bins_row):
    """4x4 world transform mapping local unit cube [-0.5, 0.5]^3 to predicted box."""
    d_trans = pos_bins_row.reshape(3)
    d_angles = angle_bins_row.reshape(3)
    d_scale = scale_bins_row.reshape(1)
    d_trans = [float(x) for x in d_trans]
    d_angles = [float(x) for x in d_angles]
    d_scale = float(d_scale)
    scale, angles, trans = continue_transform(d_scale, d_angles, d_trans)
    scale = max(scale, np.power(2e-4, 1 / 3))
    composed_transform = trimesh.transformations.compose_matrix(
        scale=[scale, scale, scale],
        shear=None,
        angles=[angles[0], angles[1], angles[2]],
        translate=[trans[0], trans[1], trans[2]],
        perspective=None,
    )
    return np.asarray(composed_transform, dtype=np.float64)


def _robust_axis_extent(pts):
    """Approximate object span: 2 * max axis half-extent (95th pct from median), like mask AABB."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0:
        return 0.0
    center = np.median(pts, axis=0)
    hx = np.percentile(np.abs(pts[:, 0] - center[0]), 95)
    hy = np.percentile(np.abs(pts[:, 1] - center[1]), 95)
    hz = np.percentile(np.abs(pts[:, 2] - center[2]), 95)
    eps = 1e-6
    return float(2.0 * max(max(hx, eps), max(hy, eps), max(hz, eps)))


def process_pred_poses_w_seg(
    pos_bin_logits,
    angle_bin_logits,
    scale_bin_logits,
    valid_logits,
    pred_masks,
    threshold=0.5,
    seg_feats=None,
):
    """
    Filter by valid_logits, decode bins, keep only instances with at least one assigned point.
    Returns:
        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, valid_logits, pred_instance_ids
    """
    valid_indices = valid_logits.sigmoid() > threshold

    if valid_indices.sum() == 0:
        return None, None, None, None, None, None, None

    pos_bin_logits = pos_bin_logits[valid_indices]
    angle_bin_logits = angle_bin_logits[valid_indices]
    scale_bin_logits = scale_bin_logits[valid_indices]
    valid_logits = valid_logits[valid_indices]
    pred_pos_bins = torch.argmax(pos_bin_logits, dim=-1)
    pred_angle_bins = torch.argmax(angle_bin_logits, dim=-1)
    pred_scale_bins = torch.argmax(scale_bin_logits, dim=-1)
    pred_masks = pred_masks[valid_indices]

    pred_instance_ids = torch.argmax(pred_masks, dim=0)
    pred_existing_instance_ids = torch.unique(pred_instance_ids)

    pred_pos_bins = pred_pos_bins[pred_existing_instance_ids]
    pred_angle_bins = pred_angle_bins[pred_existing_instance_ids]
    pred_scale_bins = pred_scale_bins[pred_existing_instance_ids]
    valid_logits = valid_logits[pred_existing_instance_ids]
    pred_masks = pred_masks[pred_existing_instance_ids]
    pred_instance_ids = torch.argmax(pred_masks, dim=0)
    if seg_feats is not None:
        seg_feats = seg_feats[valid_indices]
        seg_feats = seg_feats[pred_existing_instance_ids]

        return (
            pred_pos_bins,
            pred_angle_bins,
            pred_scale_bins,
            pred_masks,
            valid_logits,
            pred_instance_ids,
            seg_feats,
        )
    else:
        return (
            pred_pos_bins,
            pred_angle_bins,
            pred_scale_bins,
            pred_masks,
            valid_logits,
            pred_instance_ids,
        )


def assign_point_instances_with_background(
    foreground_masks,
    background_mask=None,
):
    """Assign point labels, reserving label 0 for an optional background.

    Historical checkpoints have no background_mask and retain their original
    zero-based foreground labels. Learned-background checkpoints jointly
    argmax `[background, foreground...]`, producing background label 0 and
    foreground labels 1..K.
    """
    foreground_masks = np.asarray(foreground_masks)
    if foreground_masks.ndim != 2:
        raise ValueError(
            "foreground_masks must have shape [num_foreground, num_points]"
        )
    num_points = foreground_masks.shape[1]
    if background_mask is None:
        if foreground_masks.shape[0] == 0:
            return np.full((num_points,), -1, dtype=np.int32)
        return np.argmax(foreground_masks, axis=0).astype(np.int32)

    background_mask = np.asarray(background_mask).reshape(1, -1)
    if background_mask.shape[1] != num_points:
        raise ValueError(
            "background/foreground point counts differ: "
            f"{background_mask.shape[1]} vs {num_points}"
        )
    all_masks = np.concatenate(
        [background_mask, foreground_masks], axis=0
    )
    return np.argmax(all_masks, axis=0).astype(np.int32)


def foreground_winner_indices(foreground_masks, background_mask):
    """Return foreground rows that win at least one joint point assignment."""
    instance_ids = assign_point_instances_with_background(
        foreground_masks, background_mask
    )
    foreground_ids = np.unique(instance_ids[instance_ids > 0]) - 1
    return foreground_ids.astype(np.int64, copy=False)


def nms_process(
    pos_bins,
    angle_bins,
    scale_bins,
    masks,
    valid_logits,
    points,
    feats,
    iou_threshold=0.5,
    seg_feats=None,
):
    """
    Non-maximum suppression using 3D box IoU. Order by sigmoid(valid_logits), keep
    boxes that do not overlap kept boxes above iou_threshold.
    Returns:
        pos_bins, angle_bins, scale_bins, masks, valid_logits
    """
    num_pred = pos_bins.shape[0]
    if num_pred == 0:
        return pos_bins, angle_bins, scale_bins, masks, valid_logits

    pos_bins = np.asarray(pos_bins)
    angle_bins = np.asarray(angle_bins)
    scale_bins = np.asarray(scale_bins)
    masks = np.asarray(masks)
    valid_logits = np.asarray(valid_logits, dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-valid_logits))

    box_corners = get_box_corners(pos_bins, angle_bins, scale_bins)
    box_corners_t = torch.from_numpy(box_corners).float()
    _, iou_matrix = box3d_overlap(box_corners_t, box_corners_t)
    iou_matrix = iou_matrix.cpu().numpy()

    order = np.argsort(-scores)
    keep = []
    for i in order:
        if all(iou_matrix[i, j] <= iou_threshold for j in keep):
            keep.append(i)
    keep = np.array(keep)

    if seg_feats is not None:
        return (
            pos_bins[keep],
            angle_bins[keep],
            scale_bins[keep],
            masks[keep],
            valid_logits[keep],
            seg_feats[keep],
        )
    else:
        return (
            pos_bins[keep],
            angle_bins[keep],
            scale_bins[keep],
            masks[keep],
            valid_logits[keep],
        )


def remove_unreasonable_preds_by_scale(
    pred_pos_bins,
    pred_angle_bins,
    pred_scale_bins,
    pred_masks,
    seg_feats,
    points,
    scale_too_large_ratio=1.5,
    min_points_for_scale_check=3,
):
    """
    Drop predictions when (1) no segmented instance points lie inside the predicted
    oriented box, or (2) the decoded box scale is much larger than the robust
    spatial extent of instance points inside the box (requires at least
    min_points_for_scale_check such points).
    """
    pred_pos_bins = np.asarray(pred_pos_bins)
    pred_angle_bins = np.asarray(pred_angle_bins)
    pred_scale_bins = np.asarray(pred_scale_bins)
    pred_masks = np.asarray(pred_masks)
    points = np.asarray(points, dtype=np.float64)

    num_pred = pred_pos_bins.shape[0]
    if num_pred == 0:
        return pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, seg_feats

    instance_ids = np.argmax(pred_masks, axis=0)
    n_pts = points.shape[0]
    ones = np.ones((n_pts, 1), dtype=np.float64)
    pts_h = np.hstack([points, ones])

    keep = []
    for i in range(num_pred):
        M = _pose_bins_to_world_matrix(
            pred_pos_bins[i], pred_angle_bins[i], pred_scale_bins[i]
        )
        inv_m = np.linalg.inv(M)
        local = (inv_m @ pts_h.T).T[:, :3]
        inside_box = np.all(np.abs(local) <= 0.5 + 1e-5, axis=1)
        belongs = instance_ids == i
        in_box_instance = inside_box & belongs
        if not np.any(in_box_instance):
            continue

        d_trans = np.asarray(pred_pos_bins[i]).reshape(3)
        d_angles = np.asarray(pred_angle_bins[i]).reshape(3)
        d_scale = np.asarray(pred_scale_bins[i]).reshape(1)
        d_trans = [float(x) for x in d_trans]
        d_angles = [float(x) for x in d_angles]
        d_scale = float(d_scale)
        pred_scale, _, _ = continue_transform(d_scale, d_angles, d_trans)
        pred_scale = max(pred_scale, np.power(2e-4, 1 / 3))

        n_in = int(np.count_nonzero(in_box_instance))
        if n_in >= min_points_for_scale_check:
            extent = _robust_axis_extent(points[in_box_instance])
            if extent > 0 and pred_scale > scale_too_large_ratio * extent:
                continue

        keep.append(i)

    keep = np.array(keep, dtype=np.int64)
    seg_feats = np.asarray(seg_feats)
    return (
        pred_pos_bins[keep],
        pred_angle_bins[keep],
        pred_scale_bins[keep],
        pred_masks[keep],
        seg_feats[keep],
    )

def _box_corners_from_aabb(center, half_extents):
    """
    Build 8 box corners from AABB center and half-extents (hx, hy, hz).
    Returns array of shape (8, 3).
    """
    hx, hy, hz = half_extents[0], half_extents[1], half_extents[2]
    # 8 corners: center + [±hx, ±hy, ±hz]
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners = center + signs * np.array([hx, hy, hz], dtype=np.float64)
    return corners


def _aabb_from_mask_points(points, instance_ids, idx):
    """
    Compute non-uniform axis-aligned bounding box from points and a single instance mask.
    Center = median xyz of masked points. Half-extents (per axis) = 95th percentile
    of absolute deviation from center along each axis.
    Returns (center, half_extents) where half_extents = (hx, hy, hz), or None if no points in mask.
    """
    points = np.asarray(points, dtype=np.float64)
    in_mask = instance_ids == idx
    if not np.any(in_mask):
        return None
    pts = points[in_mask]
    center = np.median(pts, axis=0)
    hx = np.percentile(np.abs(pts[:, 0] - center[0]), 95)
    hy = np.percentile(np.abs(pts[:, 1] - center[1]), 95)
    hz = np.percentile(np.abs(pts[:, 2] - center[2]), 95)
    eps = 1e-6
    hx = max(hx, eps)
    hy = max(hy, eps)
    hz = max(hz, eps)
    half_extents = np.array([hx, hy, hz], dtype=np.float64)
    return center, half_extents


def nms_process_from_masks(
    pos_bins,
    angle_bins,
    scale_bins,
    masks,
    valid_logits,
    points,
    feats,
    iou_threshold=0.5,
):
    """
    Non-maximum suppression using 3D box IoU. Recomputes non-uniform AABB from
    points and masks (center = median of masked points; per-axis half-extents =
    95th percentile of absolute deviation from center along each axis). NMS uses
    these non-uniform boxes for IoU. The stored scale_bins use a single uniform
    scale = 2 * max(hx, hy, hz) (max of the three axis extents). Order by
    sigmoid(valid_logits), keep boxes that do not overlap kept boxes above
    iou_threshold.
    Returns:
        pos_bins, angle_bins, scale_bins, masks, valid_logits
    """
    num_pred = pos_bins.shape[0]
    if num_pred == 0:
        return pos_bins, angle_bins, scale_bins, masks, valid_logits

    pos_bins = np.asarray(pos_bins)
    angle_bins = np.asarray(angle_bins)
    scale_bins = np.asarray(scale_bins)
    masks = np.asarray(masks)
    points = np.asarray(points)
    valid_logits = np.asarray(valid_logits, dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-valid_logits))

    new_pos_bins = []
    new_angle_bins = []
    new_scale_bins = []
    box_corners_list = []
    instance_ids = masks.argmax(axis=0)
    for i in range(num_pred):
        aabb = _aabb_from_mask_points(points, instance_ids, i)
        if aabb is None:
            print(f"No points in mask {i}")
            new_pos_bins.append(np.asarray(pos_bins[i]).reshape(3))
            new_angle_bins.append(np.asarray(angle_bins[i]).reshape(3))
            sc = np.asarray(scale_bins[i]).reshape(-1)
            new_scale_bins.append(sc[0] if sc.size else 0)
            box_corners_list.append(
                get_box_corners(
                    pos_bins[i : i + 1],
                    angle_bins[i : i + 1],
                    scale_bins[i : i + 1],
                )[0]
            )
        else:
            center, half_extents = aabb
            half_extents = np.clip(half_extents, SCALE_MAX / NUM_BINS, SCALE_MAX)
            box_corners_list.append(_box_corners_from_aabb(center, half_extents))
            uniform_scale = float(2.0 * np.max(half_extents))
            uniform_scale = np.clip(uniform_scale, SCALE_MAX / NUM_BINS, SCALE_MAX)
            d_scale, d_angles, d_trans = discrete_transform(
                uniform_scale, [0.0, 0.0, 0.0], center.tolist()
            )
            new_scale_bins.append(d_scale)
            new_angle_bins.append(np.array(d_angles))
            new_pos_bins.append(np.array(d_trans))
    scale_bins = np.array(new_scale_bins).reshape(-1, 1)
    angle_bins = np.array(new_angle_bins)
    pos_bins = np.array(new_pos_bins)

    box_corners = np.stack(box_corners_list, axis=0)
    box_corners_t = torch.from_numpy(box_corners).float()
    _, iou_matrix = box3d_overlap(box_corners_t, box_corners_t)
    iou_matrix = iou_matrix.cpu().numpy()

    order = np.argsort(-scores)
    keep = []
    for i in order:
        if all(iou_matrix[i, j] <= iou_threshold for j in keep):
            keep.append(i)
    keep = np.array(keep)

    return (
        pos_bins[keep],
        angle_bins[keep],
        scale_bins[keep],
        masks[keep],
        valid_logits[keep],
    )



def visualize_feats_with_pca(
    points,  # [num_points, 3]
    feats,   # [num_points, d_model]
    save_path,  # str
    pca_transform_fn=None,
    return_pca_transform_fn=False,
):
    """
    Save a point cloud where RGB is PCA(feats) projected to 3 channels.
    Optionally return a PCA transform function that can be reused.
    """
    points_np = np.asarray(points)
    feats_np = np.asarray(feats)

    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {points_np.shape}")
    if feats_np.ndim != 2:
        raise ValueError(f"Expected feats shape [N, C], got {feats_np.shape}")
    if points_np.shape[0] != feats_np.shape[0]:
        raise ValueError(
            f"points and feats must have same N, got {points_np.shape[0]} and {feats_np.shape[0]}"
        )
    if points_np.shape[0] == 0:
        return (None if return_pca_transform_fn else None)

    if pca_transform_fn is None:
        from sklearn.decomposition import PCA

        n_components = min(3, feats_np.shape[1])
        pca = PCA(n_components=n_components)
        feats_pca = pca.fit_transform(feats_np)

        if n_components < 3:
            pad = np.zeros((feats_pca.shape[0], 3 - n_components), dtype=feats_pca.dtype)
            feats_pca = np.hstack([feats_pca, pad])

        min_vals = feats_pca.min(axis=0, keepdims=True)
        max_vals = feats_pca.max(axis=0, keepdims=True)

        def pca_transform_fn(input_feats):
            input_feats_np = np.asarray(input_feats)
            if input_feats_np.ndim != 2:
                raise ValueError(f"Expected input_feats shape [N, C], got {input_feats_np.shape}")
            if input_feats_np.shape[1] != feats_np.shape[1]:
                raise ValueError(
                    f"Expected input_feats C={feats_np.shape[1]}, got {input_feats_np.shape[1]}"
                )
            feats_pca_local = pca.transform(input_feats_np)
            if n_components < 3:
                pad = np.zeros(
                    (feats_pca_local.shape[0], 3 - n_components),
                    dtype=feats_pca_local.dtype,
                )
                feats_pca_local = np.hstack([feats_pca_local, pad])
            denom_local = max_vals - min_vals
            denom_local[denom_local == 0] = 1
            colors_local = (feats_pca_local - min_vals) / denom_local
            return np.clip(colors_local, 0.0, 1.0).astype(np.float64)

    colors = pca_transform_fn(feats_np)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)

    if return_pca_transform_fn:
        return pca_transform_fn

def _create_arrow_mesh_along_z(length, radius=0.02):
    """
    Creates a trimesh arrow (cylinder shaft + cone head) along +Z from origin.
    Arrow tip is at (0, 0, length). Used for XYZ orientation arrows.
    """
    shaft_ratio = 0.75
    shaft_len = length * shaft_ratio
    cone_len = length * (1.0 - shaft_ratio)
    cone_radius = radius * 2.0  # head slightly wider than shaft
    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_len)
    # Trimesh cylinder is centered; move so base at origin
    shaft.apply_translation([0, 0, shaft_len / 2.0])
    cone = trimesh.creation.cone(radius=cone_radius, height=cone_len)
    # Cone has base at z=0, apex at z=cone_len; move so base at shaft_len
    cone.apply_translation([0, 0, shaft_len])
    arrow = trimesh.util.concatenate([shaft, cone])
    return arrow


def create_bbox_mesh_frame(pos, euler, s, w, color=None):
    """
    Generates a 3D triangle mesh of a bounding box frame plus XYZ orientation
    arrows from the center (RGB = X, Y, Z).

    Parameters:
    pos (list or array): [x, y, z] center position of the bounding box.
    euler (list or array): [rx, ry, rz] Euler angles in radians.
    s (float): Scale factor (distance between the centers of parallel edges).
    w (float): Width/thickness of the frame edges.

    Returns:
    trimesh.Trimesh: The combined mesh of the bounding box frame and arrows.
    """
    meshes = []
    half_s = s / 2.0

    # Helper function to create and position a single edge cuboid
    def add_edge(center, extents):
        # Create the edge as a box (cuboid)
        edge = trimesh.creation.box(extents=extents)
        # Move it to the correct local center position
        edge.apply_translation(center)
        meshes.append(edge)

    # 1. Create the 12 edges in local coordinates (centered at origin)

    # 4 edges parallel to the X-axis
    for y in [-half_s, half_s]:
        for z in [-half_s, half_s]:
            add_edge([0, y, z], [s + w, w, w])

    # 4 edges parallel to the Y-axis
    for x in [-half_s, half_s]:
        for z in [-half_s, half_s]:
            add_edge([x, 0, z], [w, s + w, w])

    # 4 edges parallel to the Z-axis
    for x in [-half_s, half_s]:
        for y in [-half_s, half_s]:
            add_edge([x, y, 0], [w, w, s + w])

    # 2. Combine all 12 separate edge meshes into a single mesh object
    frame_mesh = trimesh.util.concatenate(meshes)

    # Apply per-instance color to bbox frame only (before arrows are concatenated).
    if color is not None:
        color = np.asarray(color).reshape(-1)[:3]
        if color.max() <= 1.0:
            color = color * 255.0
        color_rgba = np.array(
            [int(color[0]), int(color[1]), int(color[2]), 255], dtype=np.uint8
        )
        frame_mesh.visual.vertex_colors = np.tile(
            color_rgba, (len(frame_mesh.vertices), 1)
        )
    else:
        frame_mesh.visual.vertex_colors = np.tile(
            np.array([128, 128, 128, 255], dtype=np.uint8), (len(frame_mesh.vertices), 1)
        )

    # 3. Orientation arrows: +X (red), +Y (green), +Z (blue), length proportional to scale
    arrow_length = s * 0.5
    arrow_radius = max(w * 1.5, 0.005)
    x_arrow = _create_arrow_mesh_along_z(arrow_length, arrow_radius)
    # +Z -> +X: rotate +90 deg around Y
    x_arrow.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    n_arrow_verts = len(x_arrow.vertices)
    x_arrow.visual.vertex_colors = np.tile(
        np.array([[255, 0, 0, 255]], dtype=np.uint8), (n_arrow_verts, 1)
    )
    y_arrow = _create_arrow_mesh_along_z(arrow_length, arrow_radius)
    # +Z -> +Y: rotate -90 deg around X
    y_arrow.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    y_arrow.visual.vertex_colors = np.tile(
        np.array([[0, 255, 0, 255]], dtype=np.uint8), (n_arrow_verts, 1)
    )
    z_arrow = _create_arrow_mesh_along_z(arrow_length, arrow_radius)
    z_arrow.visual.vertex_colors = np.tile(
        np.array([[0, 0, 255, 255]], dtype=np.uint8), (n_arrow_verts, 1)
    )
    arrows_mesh = trimesh.util.concatenate([x_arrow, y_arrow, z_arrow])
    frame_mesh = trimesh.util.concatenate([frame_mesh, arrows_mesh])

    # 4. Compute the transformation matrix
    # 'sxyz' assumes static (extrinsic) xyz axes. Change to 'rxyz' if your
    # euler angles use rotating (intrinsic) axes.
    transform = trimesh.transformations.euler_matrix(
        euler[0], euler[1], euler[2], axes='sxyz'
    )

    # Inject the translation (position) into the 4x4 transformation matrix
    transform[:3, 3] = pos

    # 5. Apply rotation and translation to the fully assembled frame and arrows
    frame_mesh.apply_transform(transform)

    return frame_mesh

def visualize_pred_poses_w_seg(
    pos_bins, # [num_valid, 3]
    angle_bins, # [num_valid, 3]
    scale_bins, # [num_valid, 1]
    masks, # [num_valid, N_points]
    points, # [N_points, 3]
    # points_rgbs, # [N_points, 3]
    feats, # [N_points, C]
    save_pcd_path, # str
    save_box_path, # str
    save_feat_pts_path, # str
    background_mask=None, # [N_points], optional learned-background score
):
    num_valid = pos_bins.shape[0]
    if num_valid == 0:
        return
    instance_ids = assign_point_instances_with_background(
        masks, background_mask
    ).reshape(-1)
    if background_mask is None:
        instance_colors = np.random.rand(num_valid, 3)
        point_colors = instance_colors[instance_ids]
        existing_instance_ids = np.unique(instance_ids)
        object_instance_ids = instance_ids
    else:
        # Joint labels reserve 0 for background and use 1..K for foreground.
        instance_colors = np.random.rand(num_valid + 1, 3)
        instance_colors[0] = np.array([0.7, 0.7, 0.7])
        point_colors = instance_colors[instance_ids]
        existing_instance_ids = (
            np.unique(instance_ids[instance_ids > 0]) - 1
        )
        object_instance_ids = instance_ids - 1

    # save the points and colors first with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(save_pcd_path, pcd)



    box_meshes = []
    feat_pts_dict = {}

    for obj_i in existing_instance_ids:
        d_trans = pos_bins[obj_i].reshape(3)
        d_angles = angle_bins[obj_i].reshape(3)
        d_scale = scale_bins[obj_i].reshape(1)
        d_trans = [float(item) for item in d_trans]
        d_angles = [float(item) for item in d_angles]
        d_scale = float(d_scale)
        scale, angles, trans = continue_transform(d_scale, d_angles, d_trans)
        color_index = (
            obj_i if background_mask is None else obj_i + 1
        )
        box_mesh = create_bbox_mesh_frame(
            trans, angles, scale, 0.01, color=instance_colors[color_index]
        )
        box_meshes.append(box_mesh)

        feats_obj_i = feats[object_instance_ids == obj_i]
        points_obj_i = points[object_instance_ids == obj_i]
        feat_pts_dict[obj_i] = {
            "transform": {
                "scale": scale,
                "angles": angles,
                "trans": trans,
            },
            "points": points_obj_i,
            "feats": feats_obj_i,
        }
    if box_meshes:
        box_mesh = trimesh.util.concatenate(box_meshes)
        box_mesh.export(save_box_path)
    if save_feat_pts_path is not None:
        with open(save_feat_pts_path, "wb") as f:
            pickle.dump(feat_pts_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def visualize_pred_poses_w_seg_rgbs(
    pos_bins, # [num_valid, 3]
    angle_bins, # [num_valid, 3]
    scale_bins, # [num_valid, 1]
    masks, # [num_valid, N_points]
    points, # [N_points, 3]
    rgbs, # [N_points, 3]
    save_pcd_rgb_path, # str
    save_pcd_instance_path,
    save_box_path, # str
):
    num_valid = pos_bins.shape[0]
    if num_valid == 0:
        return
    instance_ids = np.argmax(masks, axis=0).reshape(-1) # [N_points]

    instance_colors = np.random.rand(num_valid, 3)
    point_colors = instance_colors[instance_ids]

    existing_instance_ids = np.unique(instance_ids)

    # save the points and colors first with open3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    o3d.io.write_point_cloud(save_pcd_rgb_path, pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(save_pcd_instance_path, pcd)



    box_meshes = []
    feat_pts_dict = {}

    for obj_i in existing_instance_ids:
        d_trans = pos_bins[obj_i].reshape(3)
        d_angles = angle_bins[obj_i].reshape(3)
        d_scale = scale_bins[obj_i].reshape(1)
        d_trans = [float(item) for item in d_trans]
        d_angles = [float(item) for item in d_angles]
        d_scale = float(d_scale)
        scale, angles, trans = continue_transform(d_scale, d_angles, d_trans)
        box_mesh = create_bbox_mesh_frame(
            trans, angles, scale, 0.01, color=instance_colors[obj_i]
        )
        box_meshes.append(box_mesh)


    box_mesh = trimesh.util.concatenate(box_meshes)
    box_mesh.export(save_box_path)



def visualize_pred_poses_w_seg_continuous(
    trans,
    angles,
    scales,
    masks,
    points,
    feats,
    save_pcd_path,
    save_box_path,
    save_feat_pts_path,
):
    num_valid = trans.shape[0]
    if num_valid == 0:
        return

    instance_ids = np.argmax(masks, axis=0).reshape(-1)
    instance_colors = np.random.rand(num_valid, 3)
    point_colors = instance_colors[instance_ids]
    existing_instance_ids = np.unique(instance_ids)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(save_pcd_path, pcd)

    box_meshes = []
    feat_pts_dict = {}

    for obj_i in existing_instance_ids:
        obj_trans = np.asarray(trans[obj_i]).reshape(3)
        obj_angles = np.asarray(angles[obj_i]).reshape(3)
        obj_scale = float(np.asarray(scales[obj_i]).reshape(-1)[0])
        box_mesh = create_bbox_mesh_frame(
            obj_trans, obj_angles, obj_scale, 0.01, color=instance_colors[obj_i]
        )
        box_meshes.append(box_mesh)

        feats_obj_i = feats[instance_ids == obj_i]
        points_obj_i = points[instance_ids == obj_i]
        feat_pts_dict[obj_i] = {
            "transform": {
                "scale": obj_scale,
                "angles": [float(x) for x in obj_angles],
                "trans": [float(x) for x in obj_trans],
            },
            "points": points_obj_i,
            "feats": feats_obj_i,
        }

    box_mesh = trimesh.util.concatenate(box_meshes)
    box_mesh.export(save_box_path)
    if save_feat_pts_path is not None:
        with open(save_feat_pts_path, "wb") as f:
            pickle.dump(feat_pts_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def make_transform():
    """
    Create a transform for numpy RGB arrays with values in (0, 1).

    Returns:
        A transform function that takes numpy array and returns normalized tensor
    """
    def transform_fn(tensor):
        """
        Transform numpy array with values in (0, 1) to normalized tensor.

        Args:
            rgb_array: numpy array of shape [n, h, w] or [n, h, w, 3] with values in (0, 1)
                      If [n, h, w], assumes grayscale and will be converted to RGB by repeating channels

        Returns:
            Tensor of shape [n, 3, resize_size, resize_size] normalized for ImageNet
        """
        # Convert numpy to tensor
        tensor = torch.from_numpy(tensor).float()

        # Handle shape [n, h, w] - add channel dimension and repeat for RGB
        if tensor.ndim == 3:
            # [n, h, w] -> [n, h, w, 3] by repeating the channel
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif tensor.ndim == 4:
            # Already [n, h, w, 3]
            pass
        else:
            raise ValueError(f"Expected input shape [n, h, w] or [n, h, w, 3], got {tensor.shape}")

        # Convert from [n, h, w, 3] to [n, 3, h, w] (channels first)
        tensor = tensor.permute(0, 3, 1, 2)

        # Normalize with ImageNet stats
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        tensor = normalize(tensor)

        return tensor

    return transform_fn


def visualize_valid_logits(
    valid_logits,  # [N], raw logits
    save_path,  # str, png path
):
    logits = np.asarray(valid_logits).reshape(-1)
    if logits.size == 0:
        return

    # Convert raw logits to probabilities.
    scores = 1.0 / (1.0 + np.exp(-logits))

    # Largest first, smallest last.
    sorted_scores = np.sort(scores)[::-1]
    x = np.arange(sorted_scores.shape[0], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, sorted_scores, linewidth=1.8, color="tab:blue")
    ax.set_title("Sorted Valid Logits")
    ax.set_xlabel("Sorted Index")
    ax.set_ylabel("Sigmoid Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def get_dino_model(model_path=None, repo_dir=None):
    if model_path is None:
        model_path = os.environ.get(
            "FIRE3D_DINOV3_WEIGHTS",
            str(
                Path(__file__).resolve().parents[1]
                / "checkpoints"
                / "Fire3D"
                / "external"
                / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            ),
        )
    if repo_dir is None:
        repo_dir = os.environ.get("FIRE3D_DINOV3_REPO")
    if not repo_dir:
        raise ValueError(
            "DINOv3 source checkout is required; pass repo_dir or set "
            "FIRE3D_DINOV3_REPO."
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=model_path)
    model.to(device)
    model.eval()
    transform_fn = make_transform()
    return model, transform_fn


def visualize_results(
    points,
    points_rgbs,
    colors,
    results_dict,
    vis_save_dir,
    valid_score_threshold,
    nms_iou_threshold,
    save_feats=False,
    visualize_point_dino_feats_with_pca=False,
):

    points = points[0].cpu().numpy()
    feats = colors[0].cpu().numpy()  # [num_points, C]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(points_rgbs)
    save_pcd_path = os.path.join(vis_save_dir, "pred_points.ply")
    o3d.io.write_point_cloud(save_pcd_path.replace(".ply", "_rgbs.ply"), pcd)

    if visualize_point_dino_feats_with_pca:
        visualize_feats_with_pca(
            points=points,
            feats=feats,
            save_path=os.path.join(vis_save_dir, "pred_points_dino_pca_features.ply"),
            return_pca_transform_fn=False,
        )

    voxel_inverse_indices = results_dict["voxel_inverse_indices"] # [num_points]

    pos_bin_logits = results_dict["pos_bin_logits"]
    angle_bin_logits = results_dict["angle_bin_logits"]
    scale_bin_logits = results_dict["scale_bin_logits"]
    valid_logits = results_dict["valid_logits"]
    original_valid_logits = valid_logits.clone()[0].cpu().numpy() # [MAX_SCENE_OBJECTS]
    pred_masks_logits = results_dict["pred_masks_logits"] # [B=1, MAX_SCENE_OBJECTS, num_voxels]
    background_mask = None
    if "background_pred_masks_logits" in results_dict:
        background_mask = (
            results_dict["background_pred_masks_logits"]
            .sigmoid()[:, :, voxel_inverse_indices][0, 0]
            .detach().cpu().numpy()
        )
    encoded_seg_context_feats = results_dict["encoded_seg_context_feats"] # [B=1, num_voxels, d_model]
    seg_feats = results_dict["seg_feats"] # [B=1, num_voxels, seg_feat_dim]
    context_memory = results_dict["context_memory"] # [B=1, context_length, d_model]
    context_coords = results_dict["context_coords"] # [B=1, context_length, 3]
    pred_masks = pred_masks_logits.sigmoid()
    pred_masks = pred_masks[:, :, voxel_inverse_indices] # [B=1, MAX_SCENE_OBJECTS, num_points]
    encoded_seg_context_feats = encoded_seg_context_feats[:, voxel_inverse_indices, :] # [B=1, num_points, d_model]

    pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, pred_valid_logits, _, seg_feats = process_pred_poses_w_seg(
        pos_bin_logits=pos_bin_logits,
        angle_bin_logits=angle_bin_logits,
        scale_bin_logits=scale_bin_logits,
        valid_logits=valid_logits,
        pred_masks=pred_masks,
        threshold=valid_score_threshold,
        seg_feats=seg_feats,
    )
    if pred_pos_bins is None:
        print("No valid predictions found")
        return False


    pred_pos_bins = pred_pos_bins.cpu().numpy()  # [num_pred_objects, 3]
    pred_angle_bins = pred_angle_bins.cpu().numpy()  # [num_pred_objects, 3]
    pred_scale_bins = pred_scale_bins.cpu().numpy()  # [num_pred_objects, 1]
    pred_masks = pred_masks.cpu().numpy()  # [num_pred_objects, num_points]
    pred_valid_logits = pred_valid_logits.cpu().numpy()  # [num_pred_objects]
    encoded_seg_context_feats = encoded_seg_context_feats[0].cpu().numpy() # [full_context_length, d_model]
    seg_feats = seg_feats.cpu().numpy() # [num_pred_objects, seg_feat_dim]
    context_memory = context_memory[0].cpu().numpy() # [context_length, d_model]
    context_coords = context_coords[0].cpu().numpy() # [context_length, 3]



    pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, _, seg_feats = nms_process(
        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, pred_valid_logits, points, feats,
        iou_threshold=nms_iou_threshold,
        seg_feats=seg_feats,
    )

    num_pred = pred_pos_bins.shape[0]
    if num_pred == 0:
        print("No valid predictions found after NMS")
        return False

    pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, seg_feats = remove_unreasonable_preds_by_scale(
        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, seg_feats, points
    )

    num_pred = pred_pos_bins.shape[0]
    if num_pred == 0:
        print("No valid predictions found after removing unreasonable predictions")
        return False


    visualize_pred_poses_w_seg(
        pos_bins=pred_pos_bins,
        angle_bins=pred_angle_bins,
        scale_bins=pred_scale_bins,
        masks=pred_masks,
        points=points,
        # points_rgbs=points_rgbs,
        feats=feats,
        save_pcd_path=save_pcd_path,
        save_box_path=os.path.join(vis_save_dir, "pred_boxes.ply"),
        save_feat_pts_path=os.path.join(vis_save_dir, "pred_feat_pts.pkl") if save_feats else None,
        background_mask=background_mask,
    )

    # pca_transform_fn = visualize_feats_with_pca(
    #     points=points, # [num_points, 3]
    #     feats=encoded_seg_context_feats, # [num_points, d_model]
    #     save_path=os.path.join(eval_iter_dir, "encoded_seg_context_feats.ply"),
    #     return_pca_transform_fn=True,
    # )

    # visualize_feats_with_pca(
    #     points=points, # [num_points, 3]
    #     feats=seg_feats_per_point, # [num_points, d_model]
    #     pca_transform_fn=pca_transform_fn,
    #     save_path=os.path.join(eval_iter_dir, "seg_feats_per_point.ply"),
    # )

    # visualize_feats_with_pca(
    #     points=context_coords, # [context_length, 3]
    #     feats=context_memory, # [context_length, d_model]
    #     save_path=os.path.join(eval_iter_dir, "context_memory.ply"),
    #     return_pca_transform_fn=False,
    # )

    # # visualize the original valid logits
    # visualize_valid_logits(
    #     valid_logits=original_valid_logits,
    #     save_path=os.path.join(eval_iter_dir, "valid_logits.png"),
    # )

    return True
