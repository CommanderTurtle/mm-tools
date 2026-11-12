import numpy as np
from scipy.spatial.transform import Rotation as R

from .discretize_utils import (
    TRANS_VECTORS_4T4R,
    ROTATE_ANGLES_4T4R,
    TRANS_VECTORS_6T6R,
    ROTATE_ANGLES_6T6R,
)
from .discretize_utils import discretize_4t4r, discretize_6t6r, discretize_4t4r_robust


def discretize_poses_old(
    pose_dict,
    tps=False,
    mode="4t4r",
    downsample_ratio=4,
    normalize_displacement=True,
    absolute_threshold=None,
    relative_threshold=None,
):
    """
    Convert pose dictionary to model input format with optional normalized displacement.

    Displacement normalization: each 3D displacement vector is normalized by the
    average displacement magnitude of the entire clip (computed across all consecutive
    frame pairs). This ensures scale-invariant discretization across different videos.

    Args:
        pose_dict: dict {frame_idx: {"extrinsic": ndarray(4,4), "intrinsic": ndarray(3,3)}}
        tps: if True, only record translation when there's no rotation (filter coupled motion)
        mode: "4t4r" or "6t6r" - discretization mode
            - 4t4r: 4 translations (forward/backward/right/left) + 4 rotations (yaw/pitch)
            - 6t6r: 6 translations (+up/down) + 6 rotations (+roll)
        downsample_ratio: int - temporal downsampling ratio (default: 4)
        normalize_displacement: bool - if True, normalize c2w translation by average displacement
                                       (default: True for scale-invariant representation)

    Returns:
        w2cs: np.ndarray [T, 4, 4]
        intrinsics: np.ndarray [T, 3, 3] (normalized)
        action_labels: np.ndarray [T]
    """
    mode = mode.lower()
    assert mode in ["4t4r", "4t4r_robust", "6t6r"], (
        f"Unknown mode: {mode}. Use '4t4r' or '6t6r'."
    )

    pose_keys = sorted(pose_dict.keys(), key=lambda x: int(x))
    num_frames = len(pose_keys)

    # Extract c2w and intrinsic matrices (support both "intrinsic" and "K" keys)
    c2ws = np.stack([np.array(pose_dict[k]["extrinsic"]) for k in pose_keys]).copy()
    intrinsic_key = "intrinsic" if "intrinsic" in pose_dict[pose_keys[0]] else "K"
    intrinsics = np.stack(
        [np.array(pose_dict[k][intrinsic_key]) for k in pose_keys]
    ).copy()

    # Normalize intrinsics: fx/width, fy/height, cx=cy=0.5
    # Guard against invalid cx/cy that would cause inf/NaN.
    if not np.isfinite(intrinsics).all():
        raise ValueError("Invalid intrinsics: contains NaN/Inf before normalization.")

    denom_x = intrinsics[:, 0, 2] * 2.0
    denom_y = intrinsics[:, 1, 2] * 2.0
    denom_x = np.maximum(np.abs(denom_x), 1e-6)
    denom_y = np.maximum(np.abs(denom_y), 1e-6)
    intrinsics[:, 0, 0] /= denom_x
    intrinsics[:, 1, 1] /= denom_y
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5

    # Compute relative c2w
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0] = c2ws[0]
    relative_c2w[1:] = np.linalg.inv(c2ws[:-1]) @ c2ws[1:]

    # Compute average displacement magnitude for normalization
    # ΔPc_t = R_t @ (P_{t+1} - P_t), which equals relative_c2w[t, :3, 3]
    displacement_norms = np.linalg.norm(relative_c2w[1:, :3, 3], axis=1)
    avg_displacement = (
        np.mean(displacement_norms) if len(displacement_norms) > 0 else 1.0
    )
    avg_displacement = max(avg_displacement, 1e-6)  # avoid division by zero

    # Optionally normalize c2w and w2c translation by average displacement
    # This ensures scale-invariant trajectory representation across different videos
    if normalize_displacement:
        c2ws[:, :3, 3] /= avg_displacement
    else:
        avg_displacement = 1.0

    w2cs = np.linalg.inv(c2ws)
    if not np.isfinite(w2cs).all():
        raise ValueError("Invalid poses: w2cs contains NaN/Inf after normalization.")

    # Discretize motion into action labels based on mode
    if mode == "4t4r":
        action_labels = discretize_4t4r(relative_c2w, avg_displacement, num_frames, tps)
    elif mode == "4t4r_robust":
        # Only override the robust thresholds that were explicitly provided;
        # otherwise keep discretize_4t4r_robust's calibrated defaults.
        robust_kwargs = {}
        if absolute_threshold is not None:
            robust_kwargs["absolute_threshold"] = absolute_threshold
        if relative_threshold is not None:
            robust_kwargs["relative_threshold"] = relative_threshold
        action_labels = discretize_4t4r_robust(
            relative_c2w, avg_displacement, num_frames, tps, **robust_kwargs
        )
    else:  # 6t6r
        action_labels = discretize_6t6r(relative_c2w, avg_displacement, num_frames, tps)

    if downsample_ratio > 1:
        w2cs = w2cs[::downsample_ratio]
        intrinsics = intrinsics[::downsample_ratio]
        action_labels = action_labels[::downsample_ratio]

    return w2cs, intrinsics, action_labels


def discretize_poses(
    pose_dict, tps=False, mode="4t4r", downsample_ratio=4, normalize_displacement=True
):
    """
    Convert pose dictionary to model input format with optional normalized displacement.

    Displacement normalization: each 3D displacement vector is normalized by the
    average displacement magnitude of the entire clip (computed across all consecutive
    frame pairs). This ensures scale-invariant discretization across different videos.

    Args:
        pose_dict: dict {frame_idx: {"extrinsic": ndarray(4,4), "intrinsic": ndarray(3,3)}}
        tps: if True, only record translation when there's no rotation (filter coupled motion)
        mode: "4t4r" or "6t6r" - discretization mode
            - 4t4r: 4 translations (forward/backward/right/left) + 4 rotations (yaw/pitch)
            - 6t6r: 6 translations (+up/down) + 6 rotations (+roll)
        downsample_ratio: int - temporal downsampling ratio (default: 4)
        normalize_displacement: bool - if True, normalize c2w translation by average displacement
                                       (default: True for scale-invariant representation)

    Returns:
        c2ws: np.ndarray [T, 4, 4]
        intrinsics: np.ndarray [T, 3, 3] (normalized)
        action_labels: np.ndarray [T]
    """

    _, _, action_labels_old = discretize_poses_old(
        pose_dict,
        tps=tps,
        mode=mode,
        downsample_ratio=downsample_ratio,
        normalize_displacement=normalize_displacement,
    )

    mode = mode.lower()
    assert mode in ["4t4r", "4t4r_robust", "6t6r"], (
        f"Unknown mode: {mode}. Use '4t4r' or '6t6r'."
    )

    pose_keys = sorted(pose_dict.keys(), key=lambda x: int(x))
    num_frames = len(pose_keys)

    # Extract c2w and intrinsic matrices (support both "intrinsic" and "K" keys)
    c2ws = np.stack([np.array(pose_dict[k]["extrinsic"]) for k in pose_keys]).copy()
    intrinsic_key = "intrinsic" if "intrinsic" in pose_dict[pose_keys[0]] else "K"
    intrinsics = np.stack(
        [np.array(pose_dict[k][intrinsic_key]) for k in pose_keys]
    ).copy()

    # Normalize intrinsics: fx/width, fy/height, cx=cy=0.5
    # Guard against invalid cx/cy that would cause inf/NaN.
    if not np.isfinite(intrinsics).all():
        raise ValueError("Invalid intrinsics: contains NaN/Inf before normalization.")

    # Compute relative c2w
    cam_to_orgin = 0
    target_cam_c2w = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -cam_to_orgin], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    rel = target_cam_c2w @ np.linalg.inv(c2ws[0])

    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0] = target_cam_c2w
    relative_c2w[1:] = rel @ c2ws[1:]

    c2ws = relative_c2w
    if not np.isfinite(c2ws).all():
        raise ValueError("Invalid poses: w2cs contains NaN/Inf after normalization.")

    # Compute average displacement magnitude for normalization
    # ΔPc_t = R_t @ (P_{t+1} - P_t), which equals relative_c2w[t, :3, 3]
    displacement_norms = np.linalg.norm(relative_c2w[1:, :3, 3], axis=1)
    avg_displacement = (
        np.mean(displacement_norms) if len(displacement_norms) > 0 else 1.0
    )
    avg_displacement = max(avg_displacement, 1e-6)  # avoid division by zero

    # Discretize motion into action labels based on mode
    if mode == "4t4r":
        action_labels = discretize_4t4r(relative_c2w, avg_displacement, num_frames, tps)
    elif mode == "4t4r_robust":
        action_labels = discretize_4t4r_robust(
            relative_c2w, avg_displacement, num_frames, tps
        )
    else:  # 6t6r
        action_labels = discretize_6t6r(relative_c2w, avg_displacement, num_frames, tps)

    if downsample_ratio > 1:
        # c2ws = c2ws[::downsample_ratio]
        # intrinsics = intrinsics[::downsample_ratio]
        action_labels = action_labels[::downsample_ratio]

    return c2ws, intrinsics, action_labels_old


def reconstruct_trajectory(
    action_labels, initial_c2w=None, trans_scale=0.08, rot_scale=2.0, mode="4t4r"
):
    """
    Reconstruct camera trajectory from discretized action labels.

    Args:
        action_labels: np.ndarray [T] - discretized action labels
        initial_c2w: np.ndarray [4, 4] - initial camera-to-world matrix (default: identity)
        trans_scale: float - translation magnitude per step
        rot_scale: float - rotation angle scale (in degrees)
        mode: "4t4r" or "6t6r" - discretization mode

    Returns:
        c2ws: np.ndarray [T, 4, 4] - reconstructed camera-to-world matrices
        positions: np.ndarray [T, 3] - camera positions in world coordinates
    """
    mode = mode.lower()
    assert mode in ["4t4r", "6t6r"], f"Unknown mode: {mode}. Use '4t4r' or '6t6r'."

    num_frames = len(action_labels)

    if initial_c2w is None:
        initial_c2w = np.eye(4)

    c2ws = np.zeros((num_frames, 4, 4))
    c2ws[0] = initial_c2w

    # Select appropriate lookup tables based on mode
    if mode == "4t4r":
        trans_vectors = TRANS_VECTORS_4T4R
        rotate_angles = ROTATE_ANGLES_4T4R
        num_rot_labels = 9
    else:  # 6t6r
        trans_vectors = TRANS_VECTORS_6T6R
        rotate_angles = ROTATE_ANGLES_6T6R
        num_rot_labels = 13

    for i in range(1, num_frames):
        # Decode action label
        trans_label = action_labels[i] // num_rot_labels
        rotate_label = action_labels[i] % num_rot_labels

        # Get translation vector (in local camera frame)
        trans_vec = trans_vectors.get(trans_label, np.array([0, 0, 0])).astype(float)
        if np.linalg.norm(trans_vec) > 0:
            trans_vec = trans_vec / np.linalg.norm(trans_vec) * trans_scale

        # Get rotation angles
        rot_angles_vec = (
            rotate_angles.get(rotate_label, np.array([0, 0, 0])).astype(float)
            * rot_scale
        )

        # Build relative transformation matrix
        # Rotation: euler angles (pitch, yaw, roll) in degrees
        rot_matrix = R.from_euler("xyz", rot_angles_vec, degrees=True).as_matrix()

        relative_c2w = np.eye(4)
        relative_c2w[:3, :3] = rot_matrix
        relative_c2w[:3, 3] = trans_vec

        # Apply relative transformation: C_i = C_{i-1} @ relative
        c2ws[i] = c2ws[i - 1] @ relative_c2w

    # Extract positions
    positions = c2ws[:, :3, 3]

    return c2ws, positions


# ============== Backward Compatibility Wrappers ==============
def discretize_poses_worldplay(pose_dict, tps=False, mode="4t4r"):
    """Wrapper for discretize_poses (backward compatibility)."""
    w2cs, intrinsics, action_labels = discretize_poses(pose_dict, tps=tps, mode=mode)
    return w2cs, intrinsics, action_labels


def reconstruct_trajectory_worldplay(
    action_labels, initial_c2w=None, trans_scale=0.08, rot_scale=2.0, mode="4t4r"
):
    """Wrapper for reconstruct_trajectory (backward compatibility)."""
    c2ws, positions = reconstruct_trajectory(
        action_labels,
        initial_c2w=initial_c2w,
        trans_scale=trans_scale,
        rot_scale=rot_scale,
        mode=mode,
    )
    return c2ws, positions


def discretize_poses_relic(pose_dict, tps=False, mode="6t6r"):
    w2cs, intrinsics, action_labels = discretize_poses(pose_dict, tps=tps, mode=mode)
    return w2cs, intrinsics, action_labels


def reconstruct_trajectory_relic(
    action_labels, initial_c2w=None, trans_scale=0.08, rot_scale=2.0, mode="6t6r"
):
    c2ws, positions = reconstruct_trajectory(
        action_labels,
        initial_c2w=initial_c2w,
        trans_scale=trans_scale,
        rot_scale=rot_scale,
        mode=mode,
    )
    return c2ws, positions
