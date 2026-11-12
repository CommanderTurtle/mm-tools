"""Inference-time angle scaling for already framewise-relative camera poses."""

import numpy as np
from scipy.spatial.transform import Rotation


def scale_framewise_rotations(
    poses: np.ndarray, scale: float = 1.0,
) -> np.ndarray:
    """Scale local axis-angle increments, leaving translations unchanged.

    Call after resampling and framewise normalization, before model dtype
    conversion/embedding. This changes the camera condition, not the source
    absolute trajectory; its pitch limits and position closure no longer apply.
    Unit gains return the input without a matrix round trip. Otherwise a copy
    is returned. The same gain applies to every axis and direction, preserving
    the rotation axis. Scaled step angles must stay below pi.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("trajectory_rotation_scale must be finite and positive")
    if scale == 1.0:
        return poses
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("Rotation scaling requires nonempty [T,4,4] local poses")
    if not np.issubdtype(poses.dtype, np.floating) or not np.isfinite(poses).all():
        raise ValueError("Rotation scaling requires finite floating-point poses")
    rotations = poses[:, :3, :3].astype(np.float64)
    if not (
        np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-5, rtol=0)
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-5, rtol=0)
    ):
        raise ValueError("Rotation scaling requires valid SO(3) input matrices")
    rotvecs = Rotation.from_matrix(rotations).as_rotvec()
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_rotvecs = rotvecs * scale
        max_angle = float(np.linalg.norm(scaled_rotvecs, axis=1).max())
    if not np.isfinite(max_angle) or max_angle >= np.pi - 1e-7:
        raise ValueError(
            "Rotation gains make a local step reach/exceed 180 degrees; "
            "reduce trajectory_rotation_scale "
            "to avoid axis-angle wrapping"
        )
    result = poses.copy()
    result[:, :3, :3] = Rotation.from_rotvec(scaled_rotvecs).as_matrix()
    return result
