import numpy as np
import trimesh

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Toggle to enable debug assertions comparing numpy vs trimesh implementations
_DEBUG_TRANSFORM = False

# Threshold for using numba (smaller arrays are faster with numpy)
_NUMBA_THRESHOLD = 10000


def _euler_to_rotation_matrix_sxyz(angles):
    """Convert Euler angles to rotation matrix using extrinsic XYZ (sxyz) convention.
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


def _rotation_matrix_to_euler_sxyz(R):
    """Convert rotation matrix to Euler angles using extrinsic XYZ (sxyz) convention.
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


def transform_6d_from_transform_np(scale, angles, trans, transform):
    """
    Pure numpy implementation of transform_6d_from_transform.
    Uses extrinsic XYZ (sxyz) Euler convention matching trimesh.

    inputs:
        scale: float
        angles: list of 3 floats (Euler angles in 'sxyz' convention)
        trans: list of 3 floats
        transform: 4x4 transform matrix

    returns:
        scale: float
        angles: list of 3 floats
        trans: list of 3 floats
    """
    transform = np.asarray(transform)
    angles = np.asarray(angles)
    trans = np.asarray(trans)

    # Build rotation matrix from Euler angles (sxyz convention)
    R = _euler_to_rotation_matrix_sxyz(angles)

    # Build original 4x4 transform: M = T @ R @ S
    # Where S is uniform scale, R is rotation, T is translation
    # This is equivalent to: M[:3,:3] = scale * R, M[:3,3] = trans
    original_transform = np.eye(4)
    original_transform[:3, :3] = scale * R
    original_transform[:3, 3] = trans

    # Apply additional transform
    new_transform = transform @ original_transform

    # Extract the upper-left 3x3 (contains scale * rotation)
    M = new_transform[:3, :3]

    # Extract scale from column norms (should be uniform)
    scale_x = np.linalg.norm(M[:, 0])
    scale_y = np.linalg.norm(M[:, 1])
    scale_z = np.linalg.norm(M[:, 2])
    new_scale = (scale_x + scale_y + scale_z) / 3.0  # Average for uniform scale

    # Extract rotation by removing scale
    if new_scale > 1e-8:
        R_new = M / new_scale
    else:
        R_new = np.eye(3)

    # Convert rotation matrix back to Euler angles
    new_angles = _rotation_matrix_to_euler_sxyz(R_new)

    # Extract translation
    new_trans = new_transform[:3, 3].tolist()

    return float(new_scale), new_angles, new_trans


def _euler_to_rotation_matrix_sxyz_batch(angles):
    """Batched: Convert Euler angles to rotation matrices.

    Args:
        angles: (N, 3) array of Euler angles [x, y, z]
    Returns:
        R: (N, 3, 3) rotation matrices
    """
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


def _rotation_matrix_to_euler_sxyz_batch(R):
    """Batched: Convert rotation matrices to Euler angles.

    Args:
        R: (N, 3, 3) rotation matrices
    Returns:
        angles: (N, 3) Euler angles [x, y, z]
    """
    N = R.shape[0]
    angles = np.empty((N, 3), dtype=R.dtype)

    sy = -R[:, 2, 0]

    # Non-gimbal lock case (most common)
    non_gimbal = np.abs(sy) < 1.0 - 1e-6

    # Handle non-gimbal case
    y = np.arcsin(np.clip(sy, -1.0, 1.0))
    x = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    z = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    # Handle gimbal lock cases (rare, but need to handle)
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

def get_transform_matrix_batch(scales, angles, trans):
    """
    Batched version: get transform matrices for multiple 6d poses + scales at once.

    Args:
        scales: (N,) array of scales
        angles: (N, 3) array of Euler angles
        trans: (N, 3) array of translations

    Returns:
        transforms: (N, 4, 4) transform matrix of the 6d poses + scales
    """
    N = scales.shape[0]
    scales = np.asarray(scales)
    angles = np.asarray(angles)
    trans = np.asarray(trans)

    # Build rotation matrices from Euler angles: (N, 3, 3)
    R = _euler_to_rotation_matrix_sxyz_batch(angles)

    # Build original 4x4 transforms: (N, 4, 4)
    # M[:3,:3] = scale * R, M[:3,3] = trans
    transforms = np.zeros((N, 4, 4), dtype=np.float64)
    transforms[:, :3, :3] = scales[:, None, None] * R
    transforms[:, :3, 3] = trans
    transforms[:, 3, 3] = 1.0

    return transforms


def transform_6d_from_transform_batch(scales, angles, trans, transform):
    """
    Batched version: transform multiple 6d poses + scales at once.

    Args:
        scales: (N,) array of scales
        angles: (N, 3) array of Euler angles
        trans: (N, 3) array of translations
        transform: (4, 4) transform matrix to apply

    Returns:
        new_scales: (N,) array
        new_angles: (N, 3) array
        new_trans: (N, 3) array
    """
    N = scales.shape[0]
    transform = np.asarray(transform)
    scales = np.asarray(scales)
    angles = np.asarray(angles)
    trans = np.asarray(trans)

    # Build rotation matrices from Euler angles: (N, 3, 3)
    R = _euler_to_rotation_matrix_sxyz_batch(angles)

    # Build original 4x4 transforms: (N, 4, 4)
    # M[:3,:3] = scale * R, M[:3,3] = trans
    original_transforms = np.zeros((N, 4, 4), dtype=np.float64)
    original_transforms[:, :3, :3] = scales[:, None, None] * R
    original_transforms[:, :3, 3] = trans
    original_transforms[:, 3, 3] = 1.0

    # Apply transform: new_transform = transform @ original_transform
    # Use einsum for batched matrix multiply: (4,4) @ (N,4,4) -> (N,4,4)
    new_transforms = np.einsum('ij,njk->nik', transform, original_transforms)

    # Extract upper-left 3x3: (N, 3, 3)
    M = new_transforms[:, :3, :3]

    # Extract scales from column norms
    scale_cols = np.linalg.norm(M, axis=1)  # (N, 3)
    new_scales = scale_cols.mean(axis=1)  # (N,)

    # Extract rotation by removing scale
    R_new = M / np.maximum(new_scales[:, None, None], 1e-8)

    # Convert rotation matrices back to Euler angles
    new_angles = _rotation_matrix_to_euler_sxyz_batch(R_new)

    # Extract translation
    new_trans = new_transforms[:, :3, 3]

    return new_scales, new_angles, new_trans


def point_augment(
    points,
    augment=True,
    augment_rotation=True,
    augment_scale=False,
    continuous_yaw=False,
    continuous_yaw_degrees=(-180.0, 180.0),
    discrete_yaw_degrees=(0.0, 90.0, 180.0, 270.0),
):
    """
    points: numpy array of shape (N, 3)

    todo:
    1. get the center of the points
    2. along the line cross the center and parallel to the z axis, add a random rotation (0-2pi) around the line
    2. along the line cross the center and parallel to the x axis, add a random rotation (-10deg to 10deg) around the line
    3. along the line cross the center and parallel to the y axis, add a random rotation (-10deg to 10deg) around the line
    returns:
        points: numpy array of shape (N, 3)
        augment_transform: 4x4 transform matrix
    """
    if augment:
        # Get the center of the points
        center = np.mean(points, axis=0)

        # Generate random rotation angles
        if augment_rotation:
            horizonal_rotation_limit = 0.0
            if continuous_yaw:
                if len(continuous_yaw_degrees) != 2:
                    raise ValueError(
                        "continuous_yaw_degrees must contain [min, max]"
                    )
                yaw_min, yaw_max = (
                    float(continuous_yaw_degrees[0]),
                    float(continuous_yaw_degrees[1]),
                )
                if yaw_min > yaw_max:
                    raise ValueError(
                        "continuous_yaw_degrees must satisfy min <= max"
                    )
                angle_z = np.deg2rad(np.random.uniform(yaw_min, yaw_max))
            else:
                if not discrete_yaw_degrees:
                    raise ValueError(
                        "discrete_yaw_degrees must contain at least one choice"
                    )
                angle_z = np.deg2rad(
                    float(np.random.choice(discrete_yaw_degrees))
                )
            angle_x = np.random.uniform(-np.deg2rad(horizonal_rotation_limit), np.deg2rad(horizonal_rotation_limit))  # -10 to 10 degrees around x-axis
            angle_y = np.random.uniform(-np.deg2rad(horizonal_rotation_limit), np.deg2rad(horizonal_rotation_limit))  # -10 to 10 degrees around y-axis
        else:
            angle_z = 0.0  # no rotation around z-axis


        # Build rotation matrices
        # Rotation around z-axis
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [0,   0,  1]
        ])

        # Rotation around x-axis
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        Rx = np.array([
            [1,  0,   0],
            [0, cx, -sx],
            [0, sx,  cx]
        ])

        # Rotation around y-axis
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        Ry = np.array([
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0, cy]
        ])

        # Combined rotation: R = Ry @ Rx @ Rz (apply z first, then x, then y)
        R = Ry @ Rx @ Rz

        # # Generate a random scaling factor between 0.9 and 1.1
        # scale = np.random.uniform(0.9, 1.1)

        # Generate a random scaling factor between 0.75 and 1.25
        if augment_scale:
            scale = np.random.uniform(0.75, 1.25)
        else:
            scale = 1.0

        # Combine scaling with rotation: S_R = scale * R
        S_R = scale * R

        # Build the 4x4 transform matrix for scaling and rotation around the center:
        # 1. Translate to origin (subtract center)
        # 2. Apply scaling
        # 3. Apply rotation
        # 4. Translate back (add center)
        # T = T_translate_back @ S @ R @ T_translate_to_origin
        # Which simplifies to: T[:3,:3] = scale * R, T[:3,3] = center - (scale * R) @ center

        # Compute translation offset once (avoid redundant computation)
        translation = center - S_R @ center

        augment_transform = np.eye(4)
        augment_transform[:3, :3] = S_R
        augment_transform[:3, 3] = translation

        # Apply the transform to points
        # Use points @ S_R.T instead of (S_R @ points.T).T to avoid transpose copies
        points_augmented = points @ S_R.T + translation

    else:
        points_augmented = points
        augment_transform = np.eye(4)

    return points_augmented, augment_transform


if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _min_xyz_serial(points):
        """Serial min for small arrays or first pass."""
        N = points.shape[0]
        min_x = points[0, 0]
        min_y = points[0, 1]
        min_z = points[0, 2]
        for i in range(1, N):
            if points[i, 0] < min_x:
                min_x = points[i, 0]
            if points[i, 1] < min_y:
                min_y = points[i, 1]
            if points[i, 2] < min_z:
                min_z = points[i, 2]
        return min_x, min_y, min_z

    @njit(parallel=True, cache=True, fastmath=True)
    def _min_and_subtract_numba(points, out):
        """
        Fused parallel min + subtract in single pass.
        Uses chunked parallel reduction for min, then parallel subtract.
        """
        N = points.shape[0]
        num_threads = 8  # Reasonable default for most systems
        chunk_size = (N + num_threads - 1) // num_threads

        # Thread-local mins stored in array
        local_mins = np.empty((num_threads, 3), dtype=points.dtype)

        # First pass: each chunk computes local min in parallel
        for t in prange(num_threads):
            start = t * chunk_size
            end = min(start + chunk_size, N)
            if start < N:
                min_x = points[start, 0]
                min_y = points[start, 1]
                min_z = points[start, 2]
                for i in range(start + 1, end):
                    if points[i, 0] < min_x:
                        min_x = points[i, 0]
                    if points[i, 1] < min_y:
                        min_y = points[i, 1]
                    if points[i, 2] < min_z:
                        min_z = points[i, 2]
                local_mins[t, 0] = min_x
                local_mins[t, 1] = min_y
                local_mins[t, 2] = min_z
            else:
                local_mins[t, 0] = np.inf
                local_mins[t, 1] = np.inf
                local_mins[t, 2] = np.inf

        # Reduce local mins (serial, but only num_threads iterations)
        global_min_x = local_mins[0, 0]
        global_min_y = local_mins[0, 1]
        global_min_z = local_mins[0, 2]
        for t in range(1, num_threads):
            if local_mins[t, 0] < global_min_x:
                global_min_x = local_mins[t, 0]
            if local_mins[t, 1] < global_min_y:
                global_min_y = local_mins[t, 1]
            if local_mins[t, 2] < global_min_z:
                global_min_z = local_mins[t, 2]

        # Second pass: parallel subtract
        for i in prange(N):
            out[i, 0] = points[i, 0] - global_min_x
            out[i, 1] = points[i, 1] - global_min_y
            out[i, 2] = points[i, 2] - global_min_z

        return global_min_x, global_min_y, global_min_z


def point_normalize(points):
    """
    points: numpy array of shape (N, 3)

    todo: points - points_xyz_min

    returns:
        points: numpy array of shape (N, 3)
        augment_transform: 4x4 transform matrix
    """
    N = points.shape[0]

    if _HAS_NUMBA and N > _NUMBA_THRESHOLD:
        # Fused parallel min + subtract
        points_normalized = np.empty_like(points)
        min_x, min_y, min_z = _min_and_subtract_numba(points, points_normalized)
        points_xyz_min = np.array([min_x, min_y, min_z])
    else:
        # Numpy fallback for small arrays
        points_xyz_min = points.min(axis=0)
        points_normalized = points - points_xyz_min

    # Build the 4x4 transform matrix
    norm_transform = np.array([
        [1.0, 0.0, 0.0, -points_xyz_min[0]],
        [0.0, 1.0, 0.0, -points_xyz_min[1]],
        [0.0, 0.0, 1.0, -points_xyz_min[2]],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return points_normalized, norm_transform



def transform_6d_from_transform(scale, angles, trans, transform):
    """
    transform a 6d pose + 1d scale according to an additional transform
    inputs:
        scale: float
        angles: list of 3 floats (Euler angles in trimesh 'sxyz' convention)
        trans: list of 3 floats
        transform: 4x4 transform matrix

    returns:
        scale: float
        angles: list of 3 floats
        trans: list of 3 floats
    """
    # Use fast numpy implementation
    new_scale_np, new_angles_np, new_trans_np = transform_6d_from_transform_np(scale, angles, trans, transform)

    if _DEBUG_TRANSFORM:
        # Compare with trimesh implementation
        transform = np.array(transform)

        # Build the original 4x4 transform matrix using trimesh's convention
        # trimesh.transformations.compose_matrix uses 'sxyz' (static/extrinsic XYZ) Euler convention
        # The order is: M = T @ R @ S (scale first, then rotate, then translate)
        original_transform = trimesh.transformations.compose_matrix(
            scale=[scale, scale, scale],
            angles=angles,
            translate=trans
        )

        # Apply the additional transform: new_transform = transform @ original_transform
        new_transform = transform @ original_transform

        # Decompose the new transform back into scale, angles, trans
        # trimesh.transformations.decompose_matrix returns: scale, shear, angles, translate, perspective
        new_scale_vec, _, new_angles_trimesh, new_trans_trimesh, _ = trimesh.transformations.decompose_matrix(new_transform)

        # Extract uniform scale (should be uniform, take the first component)
        new_scale_trimesh = float(new_scale_vec[0])

        # Debug assertions
        atol = 1e-5
        assert np.isclose(new_scale_np, new_scale_trimesh, atol=atol), \
            f"Scale mismatch: np={new_scale_np}, trimesh={new_scale_trimesh}"
        assert np.allclose(new_angles_np, new_angles_trimesh, atol=atol), \
            f"Angles mismatch: np={new_angles_np}, trimesh={list(new_angles_trimesh)}"
        assert np.allclose(new_trans_np, new_trans_trimesh, atol=atol), \
            f"Trans mismatch: np={new_trans_np}, trimesh={list(new_trans_trimesh)}"

    return new_scale_np, list(new_angles_np), new_trans_np
