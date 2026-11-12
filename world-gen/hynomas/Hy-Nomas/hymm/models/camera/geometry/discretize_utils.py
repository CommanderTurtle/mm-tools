import numpy as np
from scipy.spatial.transform import Rotation as R


# ============== 4T4R Mode ==============
# 4 translations: forward, backward, right, left
# 4 rotations: yaw_right, yaw_left, pitch_up, pitch_down
ONEHOT_TO_LABEL_4T4R = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}
LABEL_TO_ONEHOT_4T4R = {v: k for k, v in ONEHOT_TO_LABEL_4T4R.items()}

# Translation action vectors for 4T4R (in camera local frame: +X=right, +Y=down, +Z=forward)
TRANS_VECTORS_4T4R = {
    0: np.array([0, 0, 0]),  # none
    1: np.array([0, 0, 1]),  # forward
    2: np.array([0, 0, -1]),  # backward
    3: np.array([1, 0, 0]),  # right
    4: np.array([-1, 0, 0]),  # left
    5: np.array([1, 0, 1]),  # forward+right
    6: np.array([-1, 0, 1]),  # forward+left
    7: np.array([1, 0, -1]),  # backward+right
    8: np.array([-1, 0, -1]),  # backward+left
}

# Rotation action angles for 4T4R in degrees (x=pitch, y=yaw, z=roll)
# OpenCV coords: +X=right, +Y=down, +Z=forward
ROTATE_ANGLES_4T4R = {
    0: np.array([0, 0, 0]),  # none
    1: np.array([0, 1, 0]),  # yaw right (positive Y)
    2: np.array([0, -1, 0]),  # yaw left (negative Y)
    3: np.array([-1, 0, 0]),  # pitch up (negative X)
    4: np.array([1, 0, 0]),  # pitch down (positive X)
    5: np.array([-1, 1, 0]),  # yaw right + pitch up
    6: np.array([1, 1, 0]),  # yaw right + pitch down
    7: np.array([-1, -1, 0]),  # yaw left + pitch up
    8: np.array([1, -1, 0]),  # yaw left + pitch down
}


# ============== 6T6R Mode ==============
# 6 translations: forward, backward, right, left, up, down
# 6 rotations: yaw_right, yaw_left, pitch_up, pitch_down, roll_left, roll_right
ONEHOT_TO_LABEL_6T6R = {
    (0, 0, 0, 0, 0, 0): 0,
    (1, 0, 0, 0, 0, 0): 1,
    (0, 1, 0, 0, 0, 0): 2,
    (0, 0, 1, 0, 0, 0): 3,
    (0, 0, 0, 1, 0, 0): 4,
    (0, 0, 0, 0, 1, 0): 5,
    (0, 0, 0, 0, 0, 1): 6,
    # Combined translations (common combinations)
    (1, 0, 1, 0, 0, 0): 7,  # forward+right
    (1, 0, 0, 1, 0, 0): 8,  # forward+left
    (0, 1, 1, 0, 0, 0): 9,  # backward+right
    (0, 1, 0, 1, 0, 0): 10,  # backward+left
    (1, 0, 0, 0, 1, 0): 11,  # forward+up
    (1, 0, 0, 0, 0, 1): 12,  # forward+down
}
LABEL_TO_ONEHOT_6T6R = {v: k for k, v in ONEHOT_TO_LABEL_6T6R.items()}

# Translation action vectors for 6T6R (in camera local frame: +X=right, +Y=down, +Z=forward)
TRANS_VECTORS_6T6R = {
    0: np.array([0, 0, 0]),  # none
    1: np.array([0, 0, 1]),  # forward
    2: np.array([0, 0, -1]),  # backward
    3: np.array([1, 0, 0]),  # right
    4: np.array([-1, 0, 0]),  # left
    5: np.array([0, -1, 0]),  # up (negative Y in OpenCV)
    6: np.array([0, 1, 0]),  # down (positive Y in OpenCV)
    7: np.array([1, 0, 1]),  # forward+right
    8: np.array([-1, 0, 1]),  # forward+left
    9: np.array([1, 0, -1]),  # backward+right
    10: np.array([-1, 0, -1]),  # backward+left
    11: np.array([0, -1, 1]),  # forward+up
    12: np.array([0, 1, 1]),  # forward+down
}

# Rotation action angles for 6T6R in degrees (x=pitch, y=yaw, z=roll)
ROTATE_ANGLES_6T6R = {
    0: np.array([0, 0, 0]),  # none
    1: np.array([0, 1, 0]),  # yaw right (positive Y)
    2: np.array([0, -1, 0]),  # yaw left (negative Y)
    3: np.array([-1, 0, 0]),  # pitch up (negative X)
    4: np.array([1, 0, 0]),  # pitch down (positive X)
    5: np.array([0, 0, -1]),  # roll left (negative Z, counter-clockwise)
    6: np.array([0, 0, 1]),  # roll right (positive Z, clockwise)
    # Combined rotations (common combinations)
    7: np.array([-1, 1, 0]),  # yaw right + pitch up
    8: np.array([1, 1, 0]),  # yaw right + pitch down
    9: np.array([-1, -1, 0]),  # yaw left + pitch up
    10: np.array([1, -1, 0]),  # yaw left + pitch down
    11: np.array([0, 1, -1]),  # yaw right + roll left
    12: np.array([0, 1, 1]),  # yaw right + roll right
}


def one_hot_to_label_4t4r(one_hot: np.ndarray) -> np.ndarray:
    """Convert one-hot array [N, 4] to label array [N] for 4T4R mode."""
    return np.array(
        [ONEHOT_TO_LABEL_4T4R.get(tuple(row), 0) for row in one_hot], dtype=np.int32
    )


def one_hot_to_label_6t6r(one_hot: np.ndarray) -> np.ndarray:
    """Convert one-hot array [N, 6] to label array [N] for 6T6R mode."""
    return np.array(
        [ONEHOT_TO_LABEL_6T6R.get(tuple(row), 0) for row in one_hot], dtype=np.int32
    )


def discretize_4t4r(relative_c2w, avg_displacement, num_frames, tps):
    """Discretize poses using 4T4R mode (forward/backward/right/left + yaw/pitch)."""
    trans_one_hot = np.zeros((num_frames, 4), dtype=np.int32)
    rotate_one_hot = np.zeros((num_frames, 4), dtype=np.int32)

    for i in range(1, num_frames):
        # Translation direction (normalized by average displacement)
        move_dir = relative_c2w[i, :3, 3] / avg_displacement
        move_norm = np.linalg.norm(move_dir)

        # Rotation angles (euler xyz in degrees)
        rot_angles = R.from_matrix(relative_c2w[i, :3, :3]).as_euler(
            "xyz", degrees=True
        )

        # Record translation (optionally filter if rotating in tps mode)
        # After normalization, move_norm ~ 1.0 for average displacement
        # Use threshold relative to normalized scale (0.1 means 10% of average displacement)
        if move_norm > 0.01:
            if not tps or (abs(rot_angles[0]) < 0.05 and abs(rot_angles[1]) < 0.05):
                angles_deg = np.degrees(np.arccos(np.clip(move_dir / move_norm, -1, 1)))
                if angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward
                if angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        # Record rotation
        # OpenCV: +Y=down, positive Y rotation = yaw right
        if rot_angles[1] > 0.05:
            rotate_one_hot[i, 0] = 1  # yaw right
        elif rot_angles[1] < -0.05:
            rotate_one_hot[i, 1] = 1  # yaw left
        # OpenCV coords: +X=right, +Y=down, +Z=forward
        # Pitch rotation around X-axis (right-hand rule): positive X rotation = pitch up (camera looks up)
        if rot_angles[0] > 0.05:
            rotate_one_hot[i, 2] = 1  # pitch up (positive X rotation)
        elif rot_angles[0] < -0.05:
            rotate_one_hot[i, 3] = 1  # pitch down (negative X rotation)

    # Convert to action labels: trans_label * 9 + rotate_label
    trans_labels = one_hot_to_label_4t4r(trans_one_hot)
    rotate_labels = one_hot_to_label_4t4r(rotate_one_hot)
    action_labels = trans_labels * 9 + rotate_labels

    return action_labels


def discretize_6t6r(relative_c2w, avg_displacement, num_frames, tps):
    """Discretize poses using 6T6R mode (forward/backward/right/left/up/down + yaw/pitch/roll)."""
    trans_one_hot = np.zeros((num_frames, 6), dtype=np.int32)
    rotate_one_hot = np.zeros((num_frames, 6), dtype=np.int32)

    for i in range(1, num_frames):
        # Translation direction (normalized by average displacement)
        move_dir = relative_c2w[i, :3, 3] / avg_displacement
        move_norm = np.linalg.norm(move_dir)

        # Rotation angles (euler xyz in degrees)
        rot_angles = R.from_matrix(relative_c2w[i, :3, :3]).as_euler(
            "xyz", degrees=True
        )

        # Record translation (optionally filter if rotating in tps mode)
        if move_norm > 0.1:
            if not tps or (
                abs(rot_angles[0]) < 0.05
                and abs(rot_angles[1]) < 0.05
                and abs(rot_angles[2]) < 0.05
            ):
                angles_deg = np.degrees(np.arccos(np.clip(move_dir / move_norm, -1, 1)))
                # Z-axis: forward/backward
                if angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward
                # X-axis: right/left
                if angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left
                # Y-axis: up/down (OpenCV: +Y=down)
                if angles_deg[1] > 120:
                    trans_one_hot[i, 4] = 1  # up (negative Y direction)
                elif angles_deg[1] < 60:
                    trans_one_hot[i, 5] = 1  # down (positive Y direction)

        # Record rotation
        # Yaw (Y-axis rotation)
        if rot_angles[1] > 0.05:
            rotate_one_hot[i, 0] = 1  # yaw right
        elif rot_angles[1] < -0.05:
            rotate_one_hot[i, 1] = 1  # yaw left
        # Pitch (X-axis rotation): positive X rotation = pitch up (camera looks up)
        if rot_angles[0] > 0.05:
            rotate_one_hot[i, 2] = 1  # pitch up
        elif rot_angles[0] < -0.05:
            rotate_one_hot[i, 3] = 1  # pitch down
        # Roll (Z-axis rotation)
        if rot_angles[2] < -0.05:
            rotate_one_hot[i, 4] = 1  # roll left (counter-clockwise)
        elif rot_angles[2] > 0.05:
            rotate_one_hot[i, 5] = 1  # roll right (clockwise)

    # Convert to action labels: trans_label * 13 + rotate_label
    # (13 possible translation labels, 13 possible rotation labels)
    trans_labels = one_hot_to_label_6t6r(trans_one_hot)
    rotate_labels = one_hot_to_label_6t6r(rotate_one_hot)
    action_labels = trans_labels * 13 + rotate_labels

    return action_labels


# ============== Robust Discretization Functions (with dual thresholds) ==============


def discretize_4t4r_robust(
    relative_c2w,
    avg_displacement,
    num_frames,
    tps,
    relative_threshold=0.2,
    absolute_threshold=0.02,
    rotation_threshold_deg=0.05,
):
    """
    Discretize poses using 4T4R mode with dual thresholds for robust motion detection.

    This function addresses two common issues:
    1. Static/near-static scenes: After normalization, tiny motions get amplified → false actions
    2. Large-scale scenes: Small relative motions have large absolute displacements → false actions

    Solution: Require BOTH relative and absolute thresholds to be exceeded.

    Args:
        relative_c2w: [T, 4, 4] relative camera-to-world matrices
        avg_displacement: Average displacement magnitude for normalization
        num_frames: Number of frames
        tps: If True, only record translation when there's no rotation
        relative_threshold: Minimum relative displacement (fraction of avg_displacement, default 0.1)
        absolute_threshold: Minimum absolute displacement in world units (default 0.001)
        rotation_threshold_deg: Minimum rotation angle in degrees to detect (default 0.5)

    Returns:
        action_labels: [T] discretized action labels
    """
    trans_one_hot = np.zeros((num_frames, 4), dtype=np.int32)
    rotate_one_hot = np.zeros((num_frames, 4), dtype=np.int32)

    for i in range(1, num_frames):
        # Raw translation (before normalization) - for absolute threshold check
        raw_translation = relative_c2w[i, :3, 3]
        raw_displacement = np.linalg.norm(raw_translation)

        # Translation direction (normalized by average displacement)
        move_dir = raw_translation / avg_displacement
        move_norm = np.linalg.norm(move_dir)

        # Rotation angles (euler xyz in degrees)
        rot_angles = R.from_matrix(relative_c2w[i, :3, :3]).as_euler(
            "xyz", degrees=True
        )

        # Check if motion is significant using DUAL thresholds:
        # 1. Relative motion must be significant (compared to avg displacement)
        # 2. Absolute motion must be significant (not just noise in static scenes)
        has_significant_translation = (move_norm > relative_threshold) and (
            raw_displacement > absolute_threshold
        )

        # Record translation (optionally filter if rotating in tps mode)
        if has_significant_translation:
            if not tps or (
                abs(rot_angles[0]) < rotation_threshold_deg
                and abs(rot_angles[1]) < rotation_threshold_deg
            ):
                angles_deg = np.degrees(np.arccos(np.clip(move_dir / move_norm, -1, 1)))
                if angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward
                if angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        # Record rotation (with configurable threshold)
        # OpenCV: +Y=down, positive Y rotation = yaw right
        if rot_angles[1] > rotation_threshold_deg:
            rotate_one_hot[i, 0] = 1  # yaw right
        elif rot_angles[1] < -rotation_threshold_deg:
            rotate_one_hot[i, 1] = 1  # yaw left
        # OpenCV coords: +X=right, +Y=down, +Z=forward
        # Pitch rotation around X-axis (right-hand rule): positive X rotation = pitch up
        if rot_angles[0] > rotation_threshold_deg:
            rotate_one_hot[i, 2] = 1  # pitch up (positive X rotation)
        elif rot_angles[0] < -rotation_threshold_deg:
            rotate_one_hot[i, 3] = 1  # pitch down (negative X rotation)

    # Convert to action labels: trans_label * 9 + rotate_label
    trans_labels = one_hot_to_label_4t4r(trans_one_hot)
    rotate_labels = one_hot_to_label_4t4r(rotate_one_hot)
    action_labels = trans_labels * 9 + rotate_labels

    return action_labels


def discretize_6t6r_robust(
    relative_c2w,
    avg_displacement,
    num_frames,
    tps,
    relative_threshold=0.1,
    absolute_threshold=0.001,
    rotation_threshold_deg=0.5,
):
    """
    Discretize poses using 6T6R mode with dual thresholds for robust motion detection.

    This function addresses two common issues:
    1. Static/near-static scenes: After normalization, tiny motions get amplified → false actions
    2. Large-scale scenes: Small relative motions have large absolute displacements → false actions

    Solution: Require BOTH relative and absolute thresholds to be exceeded.

    Args:
        relative_c2w: [T, 4, 4] relative camera-to-world matrices
        avg_displacement: Average displacement magnitude for normalization
        num_frames: Number of frames
        tps: If True, only record translation when there's no rotation
        relative_threshold: Minimum relative displacement (fraction of avg_displacement, default 0.1)
        absolute_threshold: Minimum absolute displacement in world units (default 0.001)
        rotation_threshold_deg: Minimum rotation angle in degrees to detect (default 0.5)

    Returns:
        action_labels: [T] discretized action labels
    """
    trans_one_hot = np.zeros((num_frames, 6), dtype=np.int32)
    rotate_one_hot = np.zeros((num_frames, 6), dtype=np.int32)

    for i in range(1, num_frames):
        # Raw translation (before normalization) - for absolute threshold check
        raw_translation = relative_c2w[i, :3, 3]
        raw_displacement = np.linalg.norm(raw_translation)

        # Translation direction (normalized by average displacement)
        move_dir = raw_translation / avg_displacement
        move_norm = np.linalg.norm(move_dir)

        # Rotation angles (euler xyz in degrees)
        rot_angles = R.from_matrix(relative_c2w[i, :3, :3]).as_euler(
            "xyz", degrees=True
        )

        # Check if motion is significant using DUAL thresholds
        has_significant_translation = (move_norm > relative_threshold) and (
            raw_displacement > absolute_threshold
        )

        # Record translation (optionally filter if rotating in tps mode)
        if has_significant_translation:
            if not tps or (
                abs(rot_angles[0]) < rotation_threshold_deg
                and abs(rot_angles[1]) < rotation_threshold_deg
                and abs(rot_angles[2]) < rotation_threshold_deg
            ):
                angles_deg = np.degrees(np.arccos(np.clip(move_dir / move_norm, -1, 1)))
                # Z-axis: forward/backward
                if angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward
                # X-axis: right/left
                if angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left
                # Y-axis: up/down (OpenCV: +Y=down)
                if angles_deg[1] > 120:
                    trans_one_hot[i, 4] = 1  # up (negative Y direction)
                elif angles_deg[1] < 60:
                    trans_one_hot[i, 5] = 1  # down (positive Y direction)

        # Record rotation (with configurable threshold)
        # Yaw (Y-axis rotation)
        if rot_angles[1] > rotation_threshold_deg:
            rotate_one_hot[i, 0] = 1  # yaw right
        elif rot_angles[1] < -rotation_threshold_deg:
            rotate_one_hot[i, 1] = 1  # yaw left
        # Pitch (X-axis rotation): positive X rotation = pitch up
        if rot_angles[0] > rotation_threshold_deg:
            rotate_one_hot[i, 2] = 1  # pitch up
        elif rot_angles[0] < -rotation_threshold_deg:
            rotate_one_hot[i, 3] = 1  # pitch down
        # Roll (Z-axis rotation)
        if rot_angles[2] < -rotation_threshold_deg:
            rotate_one_hot[i, 4] = 1  # roll left (counter-clockwise)
        elif rot_angles[2] > rotation_threshold_deg:
            rotate_one_hot[i, 5] = 1  # roll right (clockwise)

    # Convert to action labels: trans_label * 13 + rotate_label
    trans_labels = one_hot_to_label_6t6r(trans_one_hot)
    rotate_labels = one_hot_to_label_6t6r(rotate_one_hot)
    action_labels = trans_labels * 13 + rotate_labels

    return action_labels
