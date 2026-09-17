from utils.constants import (
    NUM_BINS, POS_MIN, POS_MAX, SCALE_MIN, SCALE_MAX,
    EULER_X_MIN, EULER_X_MAX,
    EULER_Y_MIN, EULER_Y_MAX,
    EULER_Z_MIN, EULER_Z_MAX,
)
import numpy as np
import torch

def _euler_to_rotation_matrix(angles):
    """Convert Euler angles to rotation matrix using numpy.

    Uses extrinsic XYZ convention (static axes), matching trimesh's 'sxyz'.
    R = Rz @ Ry @ Rx
    """
    x, y, z = angles

    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    # R = Rz @ Ry @ Rx (extrinsic XYZ)
    R = np.array([
        [cy*cz, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [cy*sz, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ])
    return R


def _rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles.

    Uses extrinsic XYZ convention (static axes), matching trimesh's 'sxyz'.
    Extracts angles from R = Rz @ Ry @ Rx.
    """
    # R[2,0] = -sin(y)
    sy = -R[2, 0]

    # Check for gimbal lock
    if np.abs(sy) < 1.0 - 1e-6:
        y = np.arcsin(np.clip(sy, -1.0, 1.0))
        # R[2,1] = cy*sx, R[2,2] = cy*cx
        x = np.arctan2(R[2, 1], R[2, 2])
        # R[1,0] = cy*sz, R[0,0] = cy*cz
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: set z = 0
        y = np.pi / 2 * np.sign(sy)
        z = 0.0
        if sy > 0:
            # y = +pi/2: R[0,1] = sx, R[1,1] = cx
            x = np.arctan2(R[0, 1], R[1, 1])
        else:
            # y = -pi/2: R[0,1] = -sx, R[1,1] = cx
            x = np.arctan2(-R[0, 1], R[1, 1])

    return [x, y, z]

def formalize_euler_angles(angles):
    R = _euler_to_rotation_matrix(angles)
    angles = _rotation_matrix_to_euler(R)
    return angles


def _quantize(val, v_min, v_max):
    """Standard linear binning function - returns bin index as int."""
    val_norm = (val - v_min) / (v_max - v_min)
    bin_idx = int(val_norm * (NUM_BINS - 1))
    return max(0, min(NUM_BINS - 1, bin_idx))


def _dequantize(bin_idx, v_min, v_max):
    """Convert bin index back to continuous value."""
    val_norm = bin_idx / (NUM_BINS - 1)
    return val_norm * (v_max - v_min) + v_min


def discrete_transform(scale, angles, trans):
    """
    Convert continuous transform parameters to discrete bin indices.

    Args:
        scale: float - uniform scale
        angles: list of 3 floats - Euler angles in radians
        trans: list of 3 floats - translation

    Returns:
        d_scale: int (long bin index) - scale bin index
        d_angles: list of 3 ints (long bin indices) - Euler angles bin indices
            Order: [x, y, z]
        d_trans: list of 3 ints (long bin indices) - translation bin indices
    """
    # Quantize scale
    d_scale = _quantize(scale, SCALE_MIN, SCALE_MAX)
    # Formalize euler angles
    angles = formalize_euler_angles(angles)
    # Quantize Euler angles
    d_angle_x = _quantize(angles[0], EULER_X_MIN, EULER_X_MAX)
    d_angle_y = _quantize(angles[1], EULER_Y_MIN, EULER_Y_MAX)
    d_angle_z = _quantize(angles[2], EULER_Z_MIN, EULER_Z_MAX)
    d_angles = [d_angle_x, d_angle_y, d_angle_z]
    # Quantize translation
    d_trans = [_quantize(t, POS_MIN, POS_MAX) for t in trans]

    return d_scale, d_angles, d_trans


def _quantize_batch(vals, v_min, v_max):
    """Batched quantization - vectorized."""
    vals_norm = (vals - v_min) / (v_max - v_min)
    bin_idx = (vals_norm * (NUM_BINS - 1)).astype(np.int64)
    return np.clip(bin_idx, 0, NUM_BINS - 1)


def _euler_to_rotation_matrix_batch(angles):
    """Batched: Euler angles (N, 3) -> rotation matrices (N, 3, 3)."""
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]

    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    N = angles.shape[0]
    R = np.empty((N, 3, 3), dtype=angles.dtype)

    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx

    return R


def _rotation_matrix_to_euler_batch(R):
    """Batched: rotation matrices (N, 3, 3) -> Euler angles (N, 3)."""
    N = R.shape[0]
    angles = np.empty((N, 3), dtype=R.dtype)

    sy = -R[:, 2, 0]
    non_gimbal = np.abs(sy) < 1.0 - 1e-6

    y = np.arcsin(np.clip(sy, -1.0, 1.0))
    x = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    z = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    gimbal_pos = (~non_gimbal) & (sy > 0)
    gimbal_neg = (~non_gimbal) & (sy <= 0)

    if np.any(gimbal_pos):
        y[gimbal_pos] = np.pi / 2
        z[gimbal_pos] = 0.0
        x[gimbal_pos] = np.arctan2(R[gimbal_pos, 0, 1], R[gimbal_pos, 1, 1])

    if np.any(gimbal_neg):
        y[gimbal_neg] = -np.pi / 2
        z[gimbal_neg] = 0.0
        x[gimbal_neg] = np.arctan2(-R[gimbal_neg, 0, 1], R[gimbal_neg, 1, 1])

    angles[:, 0] = x
    angles[:, 1] = y
    angles[:, 2] = z

    return angles


def formalize_euler_angles_batch(angles):
    """Batched: formalize Euler angles (N, 3) -> (N, 3)."""
    R = _euler_to_rotation_matrix_batch(angles)
    return _rotation_matrix_to_euler_batch(R)


def discrete_transform_batch(scales, angles, trans):
    """
    Batched version: convert multiple continuous transforms to discrete bin indices.

    Args:
        scales: (N,) array of scales
        angles: (N, 3) array of Euler angles in radians
        trans: (N, 3) array of translations

    Returns:
        d_scales: (N,) int array - scale bin indices
        d_angles: (N, 3) int array - Euler angles bin indices
        d_trans: (N, 3) int array - translation bin indices
    """
    scales = np.asarray(scales)
    angles = np.asarray(angles)
    trans = np.asarray(trans)

    # Quantize scales
    d_scales = _quantize_batch(scales, SCALE_MIN, SCALE_MAX)

    # Formalize and quantize Euler angles
    angles = formalize_euler_angles_batch(angles)
    angle_mins = np.array([EULER_X_MIN, EULER_Y_MIN, EULER_Z_MIN])
    angle_maxs = np.array([EULER_X_MAX, EULER_Y_MAX, EULER_Z_MAX])
    d_angles = _quantize_batch(angles, angle_mins, angle_maxs)

    # Quantize translations
    d_trans = _quantize_batch(trans, POS_MIN, POS_MAX)

    return d_scales, d_angles, d_trans


def continue_transform(d_scale, d_angles, d_trans):
    """
    Convert discrete bin indices back to continuous transform parameters.

    Args:
        d_scale: int - scale bin index
        d_angles: list of 3 ints - Euler angles bin indices
            Order: [x, y, z]
        d_trans: list of 3 ints - translation bin indices

    Returns:
        scale: float - uniform scale
        angles: list of 3 floats - Euler angles in radians
        trans: list of 3 floats - translation
    """
    # Dequantize scale
    scale = _dequantize(d_scale, SCALE_MIN, SCALE_MAX)
    # Dequantize Euler angles
    angle_x = _dequantize(d_angles[0], EULER_X_MIN, EULER_X_MAX)
    angle_y = _dequantize(d_angles[1], EULER_Y_MIN, EULER_Y_MAX)
    angle_z = _dequantize(d_angles[2], EULER_Z_MIN, EULER_Z_MAX)
    angles = [angle_x, angle_y, angle_z]
    # Dequantize translation
    trans = [_dequantize(t, POS_MIN, POS_MAX) for t in d_trans]

    return scale, angles, trans

def _dequantize_batch(bin_idx, v_min, v_max):
    """Batched dequantization: bin indices -> continuous values."""
    val_norm = np.asarray(bin_idx, dtype=np.float64) / (NUM_BINS - 1)
    return val_norm * (v_max - v_min) + v_min


def continue_transform_batch(d_scale, d_angles, d_trans):
    """
    Convert discrete bin indices back to continuous transform parameters.

    Args:
        d_scale: (N,) int array - scale bin indices
        d_angles: (N, 3) int array - Euler angles bin indices
        d_trans: (N, 3) int array - translation bin indices

    Returns:
        scales: (N,) array of scales
        angles: (N, 3) array of Euler angles in radians
        trans: (N, 3) array of translations
    """
    d_scale = np.asarray(d_scale)
    d_scale = d_scale.clip(min=1)
    d_angles = np.asarray(d_angles)
    d_trans = np.asarray(d_trans)

    scales = _dequantize_batch(d_scale, SCALE_MIN, SCALE_MAX)

    angle_mins = np.array([EULER_X_MIN, EULER_Y_MIN, EULER_Z_MIN])
    angle_maxs = np.array([EULER_X_MAX, EULER_Y_MAX, EULER_Z_MAX])
    angles = _dequantize_batch(d_angles, angle_mins, angle_maxs)

    trans = _dequantize_batch(d_trans, POS_MIN, POS_MAX)

    return scales, angles, trans

# Module-level cache for angle bound tensors (avoids repeated tensor creation)
_ANGLE_BOUNDS_CACHE = {}

def _get_angle_bounds(device):
    """Get cached angle min/max tensors for the given device."""
    if device not in _ANGLE_BOUNDS_CACHE:
        angle_min = torch.tensor([EULER_X_MIN, EULER_Y_MIN, EULER_Z_MIN], dtype=torch.float32, device=device)
        angle_max = torch.tensor([EULER_X_MAX, EULER_Y_MAX, EULER_Z_MAX], dtype=torch.float32, device=device)
        _ANGLE_BOUNDS_CACHE[device] = (angle_min, angle_max)
    return _ANGLE_BOUNDS_CACHE[device]

# @torch.compile
def continue_transform_torch(d_scales, d_angles, d_transs):
    """
    Convert discrete bin indices back to continuous transform parameters.

    Args:
        d_scales: (B, max_num_objects, 1) int - scale bin indices
        d_angles: (B, max_num_objects, 3) int - Euler angles bin indices
        d_transs: (B, max_num_objects, 3) int - translation bin indices

    Returns:
        scales: (B, max_num_objects, 1) float - continuous scale
        angles: (B, max_num_objects, 3) float - continuous Euler angles
        transs: (B, max_num_objects, 3) float - continuous translation
    """
    scales = _dequantize(d_scales, SCALE_MIN, SCALE_MAX)
    transs = _dequantize(d_transs, POS_MIN, POS_MAX)

    # Use cached tensors for angle bounds (avoids slow new_tensor calls)
    angle_min, angle_max = _get_angle_bounds(d_angles.device)
    angles = d_angles.float() / (NUM_BINS - 1) * (angle_max - angle_min) + angle_min

    return scales, angles, transs

def get_bin_centers(device, dtype):
    """
    Get the bin centers for the scale, angles, and translation.
    Return:
        scale_centers: [num_bins] float tensor.
        angle_centers: [num_bins, 3] float tensor.
        pos_centers: [num_bins, 3] float tensor.
    """
    scale_centers = torch.linspace(SCALE_MIN, SCALE_MAX, NUM_BINS, device=device, dtype=dtype)
    angle_x_centers = torch.linspace(EULER_X_MIN, EULER_X_MAX, NUM_BINS, device=device, dtype=dtype)
    angle_y_centers = torch.linspace(EULER_Y_MIN, EULER_Y_MAX, NUM_BINS, device=device, dtype=dtype)
    angle_z_centers = torch.linspace(EULER_Z_MIN, EULER_Z_MAX, NUM_BINS, device=device, dtype=dtype)
    trans_centers = torch.linspace(POS_MIN, POS_MAX, NUM_BINS, device=device, dtype=dtype)

    pos_centers = torch.stack([trans_centers, trans_centers, trans_centers], dim=-1)
    angle_centers = torch.stack([angle_x_centers, angle_y_centers, angle_z_centers], dim=-1)

    return scale_centers, angle_centers, pos_centers