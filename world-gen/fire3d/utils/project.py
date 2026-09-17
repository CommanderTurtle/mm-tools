import numpy as np
from scipy.stats import mode
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F


DEPTH_INVALID_POLICIES = ("nearest_fill", "drop")


def _validate_depth_invalid_policy(policy):
    if policy not in DEPTH_INVALID_POLICIES:
        raise ValueError(
            f"depth invalid policy must be one of {DEPTH_INVALID_POLICIES}, "
            f"got {policy!r}"
        )


def _fill_invalid_depth_nearest(depth, invalid_mask):
    """Fill invalid pixels in-place, preserving the historical projection path."""

    for frame_index in range(depth.shape[0]):
        valid = ~invalid_mask[frame_index]
        if not valid.any():
            depth[frame_index] = 0
            continue
        if not invalid_mask[frame_index].any():
            continue
        points_valid = np.column_stack(np.where(valid))
        values_valid = depth[frame_index][valid]
        points_invalid = np.column_stack(np.where(invalid_mask[frame_index]))
        filled = griddata(
            points_valid,
            values_valid,
            points_invalid,
            method="nearest",
            fill_value=0.0,
        )
        depth[frame_index][invalid_mask[frame_index]] = filled

def downsample_all(rgbs, depths, masks, intrinsics, height, width, factor):
    if factor <= 0:
        raise ValueError(f"downsample factor must be positive, got {factor}")

    if factor == 1:
        return rgbs, depths, masks, intrinsics, height, width

    height_ds = height // factor
    width_ds = width // factor

    rgbs_ds = (
        F.interpolate(
            torch.from_numpy(rgbs).permute(0, 3, 1, 2),
            size=(height_ds, width_ds),
            mode="bilinear",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .numpy()
        .astype(rgbs.dtype, copy=False)
    )

    if depths is not None:
        depths_ds = depths[:, ::factor, ::factor]
        depths_ds = depths_ds[:, :height_ds, :width_ds]
    else:
        depths_ds = None
    if masks is not None:
        masks_ds = masks[:, ::factor, ::factor]
        masks_ds = masks_ds[:, :height_ds, :width_ds]
    else:
        masks_ds = None

    intrinsics_ds = intrinsics.copy()
    intrinsics_ds[:, 0, 0] /= factor
    intrinsics_ds[:, 1, 1] /= factor
    intrinsics_ds[:, 0, 2] /= factor
    intrinsics_ds[:, 1, 2] /= factor

    return rgbs_ds, depths_ds, masks_ds, intrinsics_ds, height_ds, width_ds

def downsample_valid_masks(masks, height, width, factor):
    if factor <= 0:
        raise ValueError(f"downsample factor must be positive, got {factor}")

    if factor == 1:
        return masks

    height_ds = height // factor
    width_ds = width // factor

    masks_ds = masks[:, ::factor, ::factor]
    masks_ds = masks_ds[:, :height_ds, :width_ds]

    return masks_ds



def project_depth_to_points(
    rgbs,
    depth,
    intrinsics,
    c2ws,
    downsample=8,
    invalid_policy="nearest_fill",
    return_grid_keep_mask=False,
):
    """
    Project depth maps to 3D world coordinates.

    rgbs: (N_im, H, W, 3) np.array
    depths: (N_im, H, W) np.array
    intrinsics: (N_im, 3, 3) np.array
    c2ws: (N_im, 3, 4) np.array
    downsample: int, downsample factor for all the images
    invalid_policy: ``nearest_fill`` preserves the historical behavior;
        ``drop`` omits invalid samples instead of inventing depth.
    return_grid_keep_mask: Return the full flattened DINO-grid validity mask.

    Returns:
        points (N_im * H//downsample * W//downsample, 3) np.array
        rgbs (N_im * H//downsample * W//downsample, 3) np.array
    """
    N_im, H, W = depth.shape

    # Compute downsampled dimensions
    H_ds = H // downsample
    W_ds = W // downsample

    # Downsample depth maps and rgbs by taking every `downsample`-th pixel
    depth_ds = depth[:, ::downsample, ::downsample].copy()  # (N_im, H_ds, W_ds)
    rgbs_ds = rgbs[:, ::downsample, ::downsample, :]  # (N_im, H_ds, W_ds, 3)

    depth_ds = depth_ds[:, :H_ds, :W_ds]
    rgbs_ds = rgbs_ds[:, :H_ds, :W_ds, :]

    # print(f"depth_ds shape: {depth_ds.shape}, rgbs_ds shape: {rgbs_ds.shape}")

    _validate_depth_invalid_policy(invalid_policy)

    # Historical behavior fills invalid depth. The drop path retains a mask and
    # sets rejected samples to zero only so projection stays finite before filter.
    invalid_depth_max = 50.0
    invalid_mask = np.isinf(depth_ds) | np.isnan(depth_ds) | (depth_ds > invalid_depth_max) | (depth_ds <= 0)
    grid_keep_mask = (~invalid_mask).reshape(-1)
    if invalid_policy == "nearest_fill":
        if invalid_mask.any():
            _fill_invalid_depth_nearest(depth_ds, invalid_mask)
        grid_keep_mask = np.ones(grid_keep_mask.shape, dtype=bool)
    else:
        depth_ds[invalid_mask] = 0.0

    # 1. Create Meshgrid at downsampled resolution (no repeat needed - use broadcasting)
    # Map to original image coordinates (center of downsampled pixels)
    half_ds = downsample // 2
    y_ds = (np.arange(H_ds, dtype=np.float32) * downsample + half_ds)  # (H_ds,)
    x_ds = (np.arange(W_ds, dtype=np.float32) * downsample + half_ds)  # (W_ds,)

    # 2. Extract Intrinsic params (keep original intrinsics since grid is in original coords)
    fx = intrinsics[:, 0, 0][:, None, None]  # (N_im, 1, 1)
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    # 3. Backproject to Camera Coordinates using broadcasting
    # x_ds: (W_ds,) broadcasts to (N_im, H_ds, W_ds)
    # y_ds: (H_ds,) needs reshape to (H_ds, 1) to broadcast correctly
    z_cam = depth_ds  # (N_im, H_ds, W_ds)
    x_cam = (x_ds[None, None, :] - cx) * z_cam / fx  # (N_im, H_ds, W_ds)
    y_cam = (y_ds[None, :, None] - cy) * z_cam / fy  # (N_im, H_ds, W_ds)

    # 4. Transform to World Coordinates using batched matmul (BLAS optimized)
    R = c2ws[:, :, :3]  # (N_im, 3, 3)
    t = c2ws[:, :, 3]   # (N_im, 3)

    # Stack camera coordinates: (N_im, H_ds, W_ds, 3)
    cam_coords = np.stack([x_cam, y_cam, z_cam], axis=-1)
    # print(f"cam_coords shape: {cam_coords.shape}")

    # Reshape for batched matmul: (N_im, H_ds*W_ds, 3) -> (N_im, 3, H_ds*W_ds)
    cam_flat = cam_coords.reshape(N_im, -1, 3).transpose(0, 2, 1)  # (N_im, 3, H_ds*W_ds)

    # Batched matmul: (N_im, 3, 3) @ (N_im, 3, H_ds*W_ds) -> (N_im, 3, H_ds*W_ds)
    world_flat = np.matmul(R, cam_flat) + t[:, :, None]

    # Reshape back: (N_im, 3, H_ds*W_ds) -> (N_im, H_ds, W_ds, 3)
    world_coords = world_flat.transpose(0, 2, 1).reshape(-1, 3)
    rgbs_ds = rgbs_ds.reshape(-1, 3)

    if invalid_policy == "drop":
        world_coords = world_coords[grid_keep_mask]
        rgbs_ds = rgbs_ds[grid_keep_mask]
    if return_grid_keep_mask:
        return world_coords, rgbs_ds, grid_keep_mask
    return world_coords, rgbs_ds

def project_depth_to_points_patch_average(depth, intrinsics, c2ws, downsample=16):
    """
    Project depth maps to 3D world coordinates using patch-wise averaging.

    depth: (N_im, H, W) np.array
    intrinsics: (N_im, 3, 3) np.array
    c2ws: (N_im, 3, 4) np.array
    downsample: int, patch size (e.g., 14 or 16 for ViT/DINO)

    Returns:
        points: (N_im * H_ds * W_ds, 3) np.array
    """
    N_im, H, W = depth.shape
    H_ds = H // downsample
    W_ds = W // downsample


    # 0. Handle invalid/outlier depth values
    invalid_depth_max = 1000.0
    invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > invalid_depth_max) | (depth <= 0)
    # We keep the points for now to maintain the grid structure,
    # but you might want to filter them out later.
    # depth[invalid_mask] = 0

    if invalid_mask.any():
        for i in range(N_im):
            valid = ~invalid_mask[i]
            if not valid.any():
                depth[i] = 0
                continue
            n_invalid = invalid_mask[i].sum()
            if n_invalid == 0:
                continue
            # (row, col) of valid and invalid pixels
            points_valid = np.column_stack(np.where(valid))
            values_valid = depth[i][valid]
            points_invalid = np.column_stack(np.where(invalid_mask[i]))
            filled = griddata(points_valid, values_valid, points_invalid, method="nearest", fill_value=0.0)
            depth[i][invalid_mask[i]] = filled

    # 1. Patch-wise Depth Averaging
    # Reshape to (N_im, H_ds, downsample, W_ds, downsample)
    # Then mean over the patch dimensions (axes 2 and 4)
    depth_patches = depth[:N_im, :H_ds*downsample, :W_ds*downsample].reshape(
        N_im, H_ds, downsample, W_ds, downsample
    )
    depth_ds = depth_patches.mean(axis=(2, 4))  # (N_im, H_ds, W_ds)

    # 3. Create Grid in original pixel coordinates
    # We use the center of each patch to align with the DINO feature receptive field
    offset = (downsample - 1) / 2.0
    y_coords = np.arange(H_ds) * downsample + offset
    x_coords = np.arange(W_ds) * downsample + offset

    # 4. Extract Intrinsic params
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    # 5. Backproject to Camera Coordinates
    # x_coords: (W_ds,) broadcasts, y_coords: (H_ds,) reshaped for broadcasting
    z_cam = depth_ds
    x_cam = (x_coords[None, None, :] - cx) * z_cam / fx
    y_cam = (y_coords[None, :, None] - cy) * z_cam / fy

    cam_coords = np.stack([x_cam, y_cam, z_cam], axis=-1) # (N_im, H_ds, W_ds, 3)

    # 6. Transform to World Coordinates
    R = c2ws[:, :, :3]  # (N_im, 3, 3)
    t = c2ws[:, :, 3]   # (N_im, 3)

    # Flatten for batched multiplication: (N_im, N_pts, 3) -> (N_im, 3, N_pts)
    cam_flat = cam_coords.reshape(N_im, -1, 3).transpose(0, 2, 1)

    # World = R @ Cam + t
    world_flat = np.matmul(R, cam_flat) + t[:, :, None]

    # Reshape back to (Total_Points, 3)
    world_coords = world_flat.transpose(0, 2, 1).reshape(-1, 3)

    return world_coords



def project_depth_to_points_patch_average_with_instance_mask(depth, intrinsics, c2ws, instance_mask, downsample=16):
    """
    Project depth maps to 3D world coordinates using patch-wise averaging.

    depth: (N_im, H, W) np.array
    intrinsics: (N_im, 3, 3) np.array
    c2ws: (N_im, 3, 4) np.array
    instance_mask: (N_im, H, W) np.array (int32, 0: background, 1: instance 1, 2: instance 2, ...)
    downsample: int, patch size (e.g., 14 or 16 for ViT/DINO)

    Returns:
        points: (N_im * H_ds * W_ds, 3) np.array
        instance_ids: (N_im * H_ds * W_ds,) np.array (int32, 0: background, 1: instance 1, 2: instance 2, ...)
    """
    world_coords, _, _, instance_ids = project_depth_to_world_patch_geometry_with_instance_mask(
        depth, intrinsics, c2ws, instance_mask, downsample=downsample
    )
    return world_coords.reshape(-1, 3), instance_ids.reshape(-1)


def project_depth_to_world_patch_geometry_with_instance_mask(
    depth,
    intrinsics,
    c2ws,
    instance_mask,
    downsample=16,
    invalid_policy="nearest_fill",
    return_grid_keep_mask=False,
):
    """
    Compute per-patch depth geometry that can later reconstruct world-space points.

    Returns:
        world_coords: (N_im, H_ds * W_ds, 3)
        depth_ds: (N_im, H_ds * W_ds)
        ray_dirs_world: (N_im, H_ds * W_ds, 3)
        instance_ids: (N_im, H_ds * W_ds)
        grid_keep_mask: Optional (N_im, H_ds * W_ds) validity mask. Geometry
            remains dense so callers can apply this mask to matching features.
    """
    depth = depth.copy()

    N_im, H, W = depth.shape
    H_ds = H // downsample
    W_ds = W // downsample

    _validate_depth_invalid_policy(invalid_policy)
    invalid_depth_max = 50.0
    invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > invalid_depth_max) | (depth <= 0)

    if invalid_policy == "nearest_fill":
        if invalid_mask.any():
            _fill_invalid_depth_nearest(depth, invalid_mask)
    else:
        depth[invalid_mask] = 0.0

    depth_patches = depth[:N_im, :H_ds * downsample, :W_ds * downsample].reshape(
        N_im, H_ds, downsample, W_ds, downsample
    )
    # Keep patch geometry internally consistent with the ray direction below.
    # We sample depth/mask from one integer pixel inside each patch, so the ray
    # must pass through the same pixel. Using the geometric half-patch center
    # for even patch sizes creates a deterministic half-pixel offset.
    center_idx = downsample // 2
    depth_ds = depth_patches[:, :, center_idx, :, center_idx]

    invalid_crop = invalid_mask[
        :N_im, :H_ds * downsample, :W_ds * downsample
    ]
    invalid_patches = invalid_crop.reshape(
        N_im, H_ds, downsample, W_ds, downsample
    )
    sampled_valid = ~invalid_patches[:, :, center_idx, :, center_idx]
    if invalid_policy == "nearest_fill":
        sampled_valid = np.ones(sampled_valid.shape, dtype=bool)

    mask_crop = instance_mask[:N_im, :H_ds * downsample, :W_ds * downsample]
    mask_patches = mask_crop.reshape(N_im, H_ds, downsample, W_ds, downsample)
    inst_ds = mask_patches[:, :, center_idx, :, center_idx]

    offset = float(center_idx)
    y_coords = np.arange(H_ds, dtype=np.float32) * downsample + offset
    x_coords = np.arange(W_ds, dtype=np.float32) * downsample + offset

    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    # print(f"N_im: {N_im}, H_ds: {H_ds}, W_ds: {W_ds}, x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, fx shape: {fx.shape}, fy shape: {fy.shape}, cx shape: {cx.shape}, cy shape: {cy.shape}")

    x_dir = np.broadcast_to((x_coords[None, None, :] - cx) / fx, (N_im, H_ds, W_ds))
    y_dir = np.broadcast_to((y_coords[None, :, None] - cy) / fy, (N_im, H_ds, W_ds))
    z_dir = np.ones((N_im, H_ds, W_ds), dtype=np.float32)

    ray_dirs_cam = np.stack([x_dir, y_dir, z_dir], axis=-1)

    R = c2ws[:, :, :3]
    t = c2ws[:, :, 3]

    ray_dirs_cam_flat = ray_dirs_cam.reshape(N_im, -1, 3).transpose(0, 2, 1)
    ray_dirs_world = np.matmul(R, ray_dirs_cam_flat).transpose(0, 2, 1)

    depth_ds_flat = depth_ds.reshape(N_im, -1)
    world_coords = t[:, None, :] + ray_dirs_world * depth_ds_flat[..., None]
    instance_ids = inst_ds.reshape(N_im, -1)

    if return_grid_keep_mask:
        return (
            world_coords,
            depth_ds_flat,
            ray_dirs_world,
            instance_ids,
            sampled_valid.reshape(N_im, -1),
        )
    return world_coords, depth_ds_flat, ray_dirs_world, instance_ids
