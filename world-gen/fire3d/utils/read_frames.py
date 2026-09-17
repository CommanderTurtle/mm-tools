import os
import numpy as np
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor
import cv2

GLOBAL_MEAN = np.array([0.485, 0.456, 0.406])
GLOBAL_STD = np.array([0.229, 0.224, 0.225])

def apply_global_standardization(colors):
    # colors: (N, 3) in range [0, 1]
    return (colors - GLOBAL_MEAN) / GLOBAL_STD


def _read_single_rgb(filepath):
    """Helper function to read a single RGB image using OpenCV (faster than PIL)."""
    # cv2.imread reads as BGR, convert to RGB
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def read_rgbs(frames_dir, height=None, width=None, parallel=False, max_workers=None, ext="jpg"):
    """
    Read RGB images from a directory.

    Args:
        frames_dir: Directory containing .jpg files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)

    Returns:
        Stacked numpy array of shape (N, H, W, 3)
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(f'.{ext}')])
    filepaths = [os.path.join(frames_dir, f) for f in frame_files]
    n_frames = len(filepaths)

    # Pre-allocate if dimensions are known (faster than np.stack)
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

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rgb_images = list(executor.map(_read_single_rgb, filepaths))
    else:
        rgb_images = [_read_single_rgb(fp) for fp in filepaths]

    return np.stack(rgb_images)



def _read_single_depth(filepath):
    """Helper function to read a single depth file."""
    depth_archive = np.load(filepath)

    if "depth" in depth_archive:
        depth = depth_archive["depth"].astype(np.float32)
    elif "arr_0" in depth_archive:
        depth = depth_archive["arr_0"].astype(np.float32)

    else:
        raise KeyError(f"Unsupported depth archive keys: {list(depth_archive.keys())}")

    return depth


def read_depths(depth_dir, height=None, width=None, parallel=False, max_workers=None):
    """
    Read depth files from a directory.

    Args:
        depth_dir: Directory containing .npz files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)

    Returns:
        Stacked numpy array of shape (N, H, W)
    """
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npz')])
    filepaths = [os.path.join(depth_dir, f) for f in depth_files]
    n_frames = len(filepaths)

    # Pre-allocate if dimensions are known (faster than np.stack)
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

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            depth_images = list(executor.map(_read_single_depth, filepaths))
    else:
        depth_images = [_read_single_depth(fp) for fp in filepaths]

    return np.stack(depth_images)

def _read_single_depth_pi3(filepath, confidence_threshold=None):
    """Read one Pi3-style depth archive and encode rejected pixels as invalid."""

    with np.load(filepath) as depth_dict:
        depth = np.asarray(depth_dict["depth"], dtype=np.float32).copy()
        valid = np.asarray(depth_dict["valid"], dtype=np.bool_).copy()
        if depth.shape != valid.shape:
            raise ValueError(
                f"Pi3 depth/valid shape mismatch in {filepath}: "
                f"{depth.shape} != {valid.shape}"
            )
        if confidence_threshold is not None:
            if "confidence" not in depth_dict:
                raise KeyError(
                    f"Pi3 confidence threshold {confidence_threshold} was requested, "
                    f"but {filepath} has no 'confidence' array"
                )
            confidence = np.asarray(depth_dict["confidence"], dtype=np.float32)
            if confidence.shape != depth.shape:
                raise ValueError(
                    f"Pi3 depth/confidence shape mismatch in {filepath}: "
                    f"{depth.shape} != {confidence.shape}"
                )
            finite_confidence = np.isfinite(confidence)
            if finite_confidence.any():
                minimum = float(confidence[finite_confidence].min())
                maximum = float(confidence[finite_confidence].max())
                if minimum < 0.0 or maximum > 1.0:
                    raise ValueError(
                        f"Pi3 confidence in {filepath} must contain sigmoid "
                        f"probabilities in [0,1], got [{minimum}, {maximum}]"
                    )
            valid &= finite_confidence & (confidence >= confidence_threshold)

    valid &= np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = 1e6
    return depth



def read_depths_pi3(
    depth_dir,
    height=None,
    width=None,
    parallel=False,
    max_workers=None,
    confidence_threshold=None,
):
    """
    Read depth files from a directory.

    Args:
        depth_dir: Directory containing .npz files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)
        confidence_threshold: Optional threshold for a stored sigmoid-probability
            ``confidence`` array. Archives without confidence fail explicitly.

    Returns:
        Stacked numpy array of shape (N, H, W)
    """
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npz')])
    filepaths = [os.path.join(depth_dir, f) for f in depth_files]
    n_frames = len(filepaths)

    # Pre-allocate if dimensions are known (faster than np.stack)
    if height is not None and width is not None:
        result = np.empty((n_frames, height, width), dtype=np.float32)

        def _read_into(args):
            idx, fp = args
            result[idx] = _read_single_depth_pi3(
                fp, confidence_threshold=confidence_threshold
            )

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_read_into, enumerate(filepaths)))
        else:
            for idx, fp in enumerate(filepaths):
                result[idx] = _read_single_depth_pi3(
                    fp, confidence_threshold=confidence_threshold
                )
        return result

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            depth_images = list(
                executor.map(
                    lambda path: _read_single_depth_pi3(
                        path, confidence_threshold=confidence_threshold
                    ),
                    filepaths,
                )
            )
    else:
        depth_images = [
            _read_single_depth_pi3(
                fp, confidence_threshold=confidence_threshold
            )
            for fp in filepaths
        ]

    return np.stack(depth_images)



def _read_single_mask(filepath):
    """Helper function to read a single mask image using OpenCV (faster than PIL)."""
    # Read as grayscale for mask
    mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return mask.astype(np.int32)


def _read_single_mask_with_unique(fp):
    """Read a single mask and return both the mask and its unique values."""
    mask = _read_single_mask(fp)
    unique_vals = np.unique(mask)
    return mask, unique_vals


def read_masks(masks_dir, height=None, width=None, parallel=False, max_workers=None):
    """
    Read mask images from a directory.

    Args:
        masks_dir: Directory containing .png files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)

    Returns:
        Stacked numpy array of shape (N, H, W)
        existing_indices: list of unique object ids in the masks
    """
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    filepaths = [os.path.join(masks_dir, f) for f in mask_files]
    n_frames = len(filepaths)

    all_unique_sets = []

    # Pre-allocate if dimensions are known (faster than np.stack)
    if height is not None and width is not None:
        result = np.empty((n_frames, height, width), dtype=np.int32)

        def _read_into(args):
            idx, fp = args
            mask, unique_vals = _read_single_mask_with_unique(fp)
            result[idx] = mask
            return unique_vals

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                all_unique_sets = list(executor.map(_read_into, enumerate(filepaths)))
        else:
            for idx, fp in enumerate(filepaths):
                unique_vals = _read_into((idx, fp))
                all_unique_sets.append(unique_vals)

        # Combine all unique values into a global unique list
        existing_indices = np.unique(np.concatenate(all_unique_sets)).tolist()
        return result, existing_indices

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_read_single_mask_with_unique, filepaths))
        mask_images = [r[0] for r in results]
        all_unique_sets = [r[1] for r in results]
    else:
        mask_images = []
        for fp in filepaths:
            mask, unique_vals = _read_single_mask_with_unique(fp)
            mask_images.append(mask)
            all_unique_sets.append(unique_vals)

    # Combine all unique values into a global unique list
    existing_indices = np.unique(np.concatenate(all_unique_sets)).tolist()
    return np.stack(mask_images), existing_indices

def _read_single_mask_v2(filepath):
    """Helper function to read a single mask image using OpenCV (faster than PIL)."""
    # Read as grayscale for mask
    mask_archive = np.load(filepath)

    if "mask" in mask_archive:
        mask = mask_archive["mask"].astype(np.int32)
    elif "arr_0" in mask_archive:
        mask = mask_archive["arr_0"].astype(np.int32)
    else:
        raise KeyError(f"Unsupported mask archive keys: {list(mask_archive.keys())}")

    return mask


def _read_single_mask_with_unique_v2(fp):
    """Read a single mask and return both the mask and its unique values."""
    mask = _read_single_mask_v2(fp)
    unique_vals = np.unique(mask)
    return mask, unique_vals

def read_masks_v2(masks_dir, height=None, width=None, parallel=False, max_workers=None):
    """
    Read mask images from a directory.

    Args:
        masks_dir: Directory containing .npz files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)

    Returns:
        Stacked numpy array of shape (N, H, W)
        existing_indices: list of unique object ids in the masks
    """
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npz')])
    filepaths = [os.path.join(masks_dir, f) for f in mask_files]
    n_frames = len(filepaths)

    all_unique_sets = []

    # Pre-allocate if dimensions are known (faster than np.stack)
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
                unique_vals = _read_into((idx, fp))
                all_unique_sets.append(unique_vals)

        # Combine all unique values into a global unique list
        existing_indices = np.unique(np.concatenate(all_unique_sets)).tolist()
        return result, existing_indices

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_read_single_mask_with_unique_v2, filepaths))
        mask_images = [r[0] for r in results]
        all_unique_sets = [r[1] for r in results]
    else:
        mask_images = []
        for fp in filepaths:
            mask, unique_vals = _read_single_mask_with_unique_v2(fp)
            mask_images.append(mask)
            all_unique_sets.append(unique_vals)

    # Combine all unique values into a global unique list
    existing_indices = np.unique(np.concatenate(all_unique_sets)).tolist()
    return np.stack(mask_images), existing_indices

def read_masks_v2_with_per_frame_existing_ids(masks_dir, height=None, width=None, parallel=False, max_workers=None):
    """
    Read mask images from a directory.

    Args:
        masks_dir: Directory containing .npz files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)

    Returns:
        Stacked numpy array of shape (N, H, W)
        existing_indices: bool array of shape (N, max_instance_id + 1)
    """
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npz')])
    filepaths = [os.path.join(masks_dir, f) for f in mask_files]
    n_frames = len(filepaths)

    all_unique_sets = [None] * n_frames

    def _build_existing_indices(per_frame_unique_vals):
        max_instance_id = 0
        for unique_vals in per_frame_unique_vals:
            if unique_vals.size > 0:
                max_instance_id = max(max_instance_id, int(unique_vals.max()))

        existing_indices = np.zeros((n_frames, max_instance_id + 1), dtype=bool)
        for frame_idx, unique_vals in enumerate(per_frame_unique_vals):
            existing_indices[frame_idx, unique_vals.astype(np.int64)] = True
        return existing_indices

    # Pre-allocate if dimensions are known (faster than np.stack)
    if height is not None and width is not None:
        result = np.empty((n_frames, height, width), dtype=np.int32)

        def _read_into(args):
            idx, fp = args
            mask, unique_vals = _read_single_mask_with_unique_v2(fp)
            result[idx] = mask
            return idx, unique_vals

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, unique_vals in executor.map(_read_into, enumerate(filepaths)):
                    all_unique_sets[idx] = unique_vals
        else:
            for idx, fp in enumerate(filepaths):
                read_idx, unique_vals = _read_into((idx, fp))
                all_unique_sets[read_idx] = unique_vals

        existing_indices = _build_existing_indices(all_unique_sets)
        return result, existing_indices

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_read_single_mask_with_unique_v2, filepaths))
    else:
        results = [_read_single_mask_with_unique_v2(fp) for fp in filepaths]

    mask_images = [r[0] for r in results]
    all_unique_sets = [r[1] for r in results]
    existing_indices = _build_existing_indices(all_unique_sets)
    return np.stack(mask_images), existing_indices



def read_cameras(camera_path):
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)

    K = np.array(camera_data['K'])
    width = camera_data['width']
    height = camera_data['height']
    frames = camera_data['frames']
    num_frames = len(frames)

    intrinsics = K.reshape(1, 3, 3).repeat(num_frames, axis=0)
    c2ws = []
    for frame_i in range(num_frames):
        frame_data = frames[frame_i]
        eye = np.array(frame_data['eye'])
        lookat = np.array(frame_data['lookat'])
        up_vec = np.array(frame_data['up'])

        forward = lookat - eye
        forward = forward / np.linalg.norm(forward)

        # right = forward × up (perpendicular to both)
        right = np.cross(forward, up_vec)
        right = right / np.linalg.norm(right)

        # Recompute up to ensure orthonormality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Build rotation matrix: columns are [right, -up, forward]
        R_mat = np.column_stack([right, -up, forward])

        # Build camera-to-world transform (extrinsic)
        # R_mat transforms from camera to world: world_vec = R_mat @ cam_vec
        extrinsic_c2w = np.eye(4)
        extrinsic_c2w[:3, :3] = R_mat
        extrinsic_c2w[:3, 3] = eye
        extrinsic_c2w = extrinsic_c2w[:3, :]
        c2ws.append(extrinsic_c2w)

    c2ws = np.stack(c2ws)
    return intrinsics, c2ws, (height, width)



def read_cameras_scannetpp(camera_path):
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)

    width = camera_data['width']
    height = camera_data['height']

    poses_c2w = camera_data['poses_c2w']
    intrinsics_K = camera_data['intrinsics_K']

    num_frames = len(poses_c2w)

    c2ws = np.array(poses_c2w).reshape(num_frames, 4, 4)[:, :3, :]
    intrinsics = np.array(intrinsics_K).reshape(num_frames, 3, 3)


    # intrinsics = K.reshape(1, 3, 3).repeat(num_frames, axis=0)
    # c2ws = []
    # for frame_i in range(num_frames):
    #     frame_data = frames[frame_i]
    #     eye = np.array(frame_data['eye'])
    #     lookat = np.array(frame_data['lookat'])
    #     up_vec = np.array(frame_data['up'])

    #     forward = lookat - eye
    #     forward = forward / np.linalg.norm(forward)

    #     # right = forward × up (perpendicular to both)
    #     right = np.cross(forward, up_vec)
    #     right = right / np.linalg.norm(right)

    #     # Recompute up to ensure orthonormality
    #     up = np.cross(right, forward)
    #     up = up / np.linalg.norm(up)

    #     # Build rotation matrix: columns are [right, -up, forward]
    #     R_mat = np.column_stack([right, -up, forward])

    #     # Build camera-to-world transform (extrinsic)
    #     # R_mat transforms from camera to world: world_vec = R_mat @ cam_vec
    #     extrinsic_c2w = np.eye(4)
    #     extrinsic_c2w[:3, :3] = R_mat
    #     extrinsic_c2w[:3, 3] = eye
    #     extrinsic_c2w = extrinsic_c2w[:3, :]
    #     c2ws.append(extrinsic_c2w)

    # c2ws = np.stack(c2ws)
    return intrinsics, c2ws, (height, width)



def _read_single_prune_mask(filepath):
    """Helper function to read a single prune mask file."""
    return np.load(filepath)['mask'].astype(np.bool_)


def read_prune_masks(prune_masks_dir, height=None, width=None, parallel=False, max_workers=None):
    """
    Read prune mask files from a directory.

    Args:
        depth_dir: Directory containing .npz files
        height: Image height (if known, enables pre-allocation for speed)
        width: Image width (if known, enables pre-allocation for speed)
        parallel: If True, use multi-threading for parallel file reading
        max_workers: Number of worker threads (default: None, uses ThreadPoolExecutor default)

    Returns:
        Stacked numpy array of shape (N, H, W)
    """
    prune_mask_files = sorted([f for f in os.listdir(prune_masks_dir) if f.endswith('.npz')])
    filepaths = [os.path.join(prune_masks_dir, f) for f in prune_mask_files]
    n_frames = len(filepaths)

    # Pre-allocate if dimensions are known (faster than np.stack)
    if height is not None and width is not None:
        result = np.empty((n_frames, height, width), dtype=np.bool_)

        def _read_into(args):
            idx, fp = args
            result[idx] = _read_single_prune_mask(fp)

        if parallel and n_frames > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_read_into, enumerate(filepaths)))
        else:
            for idx, fp in enumerate(filepaths):
                result[idx] = _read_single_prune_mask(fp)
        return result

    # Fallback to np.stack if dimensions unknown
    if parallel and n_frames > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            prune_masks = list(executor.map(_read_single_prune_mask, filepaths))
    else:
        prune_masks = [_read_single_prune_mask(fp) for fp in filepaths]

    return np.stack(prune_masks)
