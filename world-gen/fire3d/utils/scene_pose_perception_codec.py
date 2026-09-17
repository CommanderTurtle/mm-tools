"""Pose codec for the signed-coordinate det/seg v2 branch.

This module deliberately does not change :mod:`utils.discrete`.  The legacy
detector uses positions in ``[0, 24]``; this branch supervises object centers
in ``[-12, 12]`` while preserving the existing Euler and direct-scale bins.
Keeping the codec isolated makes the old training path byte-for-byte
unchanged.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from utils.constants import (
    EULER_X_MAX,
    EULER_X_MIN,
    EULER_Y_MAX,
    EULER_Y_MIN,
    EULER_Z_MAX,
    EULER_Z_MIN,
    NUM_BINS,
    SCALE_MAX,
    SCALE_MIN,
)
from utils.discrete import (
    _euler_to_rotation_matrix_batch,
    _rotation_matrix_to_euler_batch,
)
from utils.loss import euler_angles_to_matrix, rotation_matrix_geodesic_distance


SIGNED_POS_MIN = -12.0
SIGNED_POS_MAX = 12.0


def _quantize_np(values: np.ndarray, low, high, num_bins: int = NUM_BINS):
    """Linearly quantize and reject non-finite values before clipping."""

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("pose target contains NaN or Inf")
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    normalized = (values - low) / (high - low)
    return np.clip(
        np.floor(normalized * (num_bins - 1) + 1e-8), 0, num_bins - 1
    ).astype(np.int64)


def _dequantize_torch(indices: torch.Tensor, low, high, num_bins=NUM_BINS):
    low = torch.as_tensor(low, dtype=torch.float32, device=indices.device)
    high = torch.as_tensor(high, dtype=torch.float32, device=indices.device)
    return indices.float() / float(num_bins - 1) * (high - low) + low


def formalize_euler_np(angles: np.ndarray) -> np.ndarray:
    """Canonicalize extrinsic XYZ Euler angles through their SO(3) matrix."""

    angles = np.asarray(angles, dtype=np.float64).reshape(-1, 3)
    return _rotation_matrix_to_euler_batch(_euler_to_rotation_matrix_batch(angles))


def encode_signed_pose_np(
    translations: np.ndarray,
    euler_xyz: np.ndarray,
    scales: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode signed centers, XYZ Euler rotations, and *direct* uniform scale."""

    translations = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    euler_xyz = formalize_euler_np(euler_xyz)
    scales = np.asarray(scales, dtype=np.float64).reshape(-1, 1)
    if np.any(translations < SIGNED_POS_MIN) or np.any(translations > SIGNED_POS_MAX):
        raise ValueError("object center lies outside signed [-12, 12]^3 domain")
    pos_bins = _quantize_np(
        translations, SIGNED_POS_MIN, SIGNED_POS_MAX, num_bins
    )
    angle_bins = _quantize_np(
        euler_xyz,
        [EULER_X_MIN, EULER_Y_MIN, EULER_Z_MIN],
        [EULER_X_MAX, EULER_Y_MAX, EULER_Z_MAX],
        num_bins,
    )
    # Scale is intentionally not logarithmically transformed in this branch.
    scale_bins = _quantize_np(scales, SCALE_MIN, SCALE_MAX, num_bins)
    return pos_bins, angle_bins, scale_bins


def decode_signed_pose_torch(pos_bins, angle_bins, scale_bins, num_bins=NUM_BINS):
    """Inverse of :func:`encode_signed_pose_np` for targets or argmax bins."""

    translations = _dequantize_torch(
        pos_bins, SIGNED_POS_MIN, SIGNED_POS_MAX, num_bins
    )
    angles = _dequantize_torch(
        angle_bins,
        [EULER_X_MIN, EULER_Y_MIN, EULER_Z_MIN],
        [EULER_X_MAX, EULER_Y_MAX, EULER_Z_MAX],
        num_bins,
    )
    scales = _dequantize_torch(scale_bins, SCALE_MIN, SCALE_MAX, num_bins)
    return scales, angles, translations


def signed_bin_centers(device=None):
    """Return FP32 centers used by the differentiable Hungarian soft argmax."""

    pos = torch.linspace(SIGNED_POS_MIN, SIGNED_POS_MAX, NUM_BINS, device=device)
    scale = torch.linspace(SCALE_MIN, SCALE_MAX, NUM_BINS, device=device)
    angle = torch.stack(
        [
            torch.linspace(EULER_X_MIN, EULER_X_MAX, NUM_BINS, device=device),
            torch.linspace(EULER_Y_MIN, EULER_Y_MAX, NUM_BINS, device=device),
            torch.linspace(EULER_Z_MIN, EULER_Z_MAX, NUM_BINS, device=device),
        ],
        dim=0,
    )
    return pos.float(), angle.float(), scale.float()


def local_up_quarter_turn_equivalent_rotations(rotation: torch.Tensor):
    """Return ``R @ Rz(k*pi/2)`` for object-local up-axis equivalence.

    Right multiplication is essential.  Left multiplication would rotate
    around the scene/world Z axis and is wrong once an object is tilted.
    """

    angles = torch.arange(4, device=rotation.device, dtype=rotation.dtype)
    angles = angles * (math.pi / 2.0)
    c, s = torch.cos(angles), torch.sin(angles)
    zero, one = torch.zeros_like(angles), torch.ones_like(angles)
    rz = torch.stack(
        [
            torch.stack([c, -s, zero], dim=-1),
            torch.stack([s, c, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )
    return torch.matmul(rotation.unsqueeze(-3), rz)


def matrix_to_euler_xyz_torch(rotation: torch.Tensor) -> torch.Tensor:
    """Convert ``R=Rz@Ry@Rx`` matrices to canonical XYZ Euler angles."""

    sy = -rotation[..., 2, 0]
    regular = sy.abs() < 1.0 - 1e-6
    y = torch.asin(sy.clamp(-1.0, 1.0))
    x_regular = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    z_regular = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    y_lock = torch.sign(sy) * (math.pi / 2.0)
    x_pos = torch.atan2(rotation[..., 0, 1], rotation[..., 1, 1])
    x_neg = torch.atan2(-rotation[..., 0, 1], rotation[..., 1, 1])
    x_lock = torch.where(sy > 0, x_pos, x_neg)
    x = torch.where(regular, x_regular, x_lock)
    y = torch.where(regular, y, y_lock)
    z = torch.where(regular, z_regular, torch.zeros_like(z_regular))
    return torch.stack([x, y, z], dim=-1)


def local_up_equivalent_angle_bins(angle_bins: torch.Tensor, num_bins=NUM_BINS):
    """Build four full XYZ-bin targets for local-up quarter-turn symmetry."""

    _, angles, _ = decode_signed_pose_torch(
        torch.zeros_like(angle_bins),
        angle_bins,
        torch.zeros(*angle_bins.shape[:-1], 1, device=angle_bins.device),
        num_bins,
    )
    rotations = euler_angles_to_matrix(angles)
    candidates = matrix_to_euler_xyz_torch(
        local_up_quarter_turn_equivalent_rotations(rotations)
    )
    lows = candidates.new_tensor([EULER_X_MIN, EULER_Y_MIN, EULER_Z_MIN])
    highs = candidates.new_tensor([EULER_X_MAX, EULER_Y_MAX, EULER_Z_MAX])
    bins = torch.floor(
        (candidates - lows) / (highs - lows) * float(num_bins - 1) + 1e-6
    ).long().clamp_(0, num_bins - 1)
    bins[..., 0, :] = angle_bins.long()  # exact zero-turn discrete target
    return bins


def local_up_geodesic(angle_pred: torch.Tensor, angle_gt: torch.Tensor):
    """Minimum SO(3) distance over local-up 0/90/180/270-degree variants."""

    pred_rotation = euler_angles_to_matrix(angle_pred)
    gt_rotation = euler_angles_to_matrix(angle_gt)
    candidates = local_up_quarter_turn_equivalent_rotations(gt_rotation)
    return rotation_matrix_geodesic_distance(
        pred_rotation.unsqueeze(-3), candidates
    ).min(dim=-1).values


@torch.no_grad()
def signed_pose_metrics(
    pos_logits,
    angle_logits,
    scale_logits,
    target_pos_bins,
    target_angle_bins,
    target_scale_bins,
    valid_mask,
):
    """Report physical errors after Hungarian-resorted argmax decoding."""

    count = target_pos_bins.shape[1]
    pred_pos_bins = pos_logits[:, :count].argmax(dim=-1)
    pred_angle_bins = angle_logits[:, :count].argmax(dim=-1)
    pred_scale_bins = scale_logits[:, :count].argmax(dim=-1)
    pred_scale, pred_angle, pred_pos = decode_signed_pose_torch(
        pred_pos_bins, pred_angle_bins, pred_scale_bins
    )
    gt_scale, gt_angle, gt_pos = decode_signed_pose_torch(
        target_pos_bins, target_angle_bins, target_scale_bins
    )
    weights = valid_mask.float()
    denominator = weights.sum().clamp_min(1.0)
    translation_l1 = ((pred_pos - gt_pos).abs().mean(dim=-1) * weights).sum()
    scale_l1 = ((pred_scale - gt_scale).abs().squeeze(-1) * weights).sum()
    rotation = (local_up_geodesic(pred_angle, gt_angle) * weights).sum()
    raw_rotation = (
        rotation_matrix_geodesic_distance(
            euler_angles_to_matrix(pred_angle), euler_angles_to_matrix(gt_angle)
        )
        * weights
    ).sum()
    return {
        "translation_l1_world": translation_l1 / denominator,
        "scale_l1_world": scale_l1 / denominator,
        "rotation_geodesic_degrees": rotation / denominator * (180.0 / math.pi),
        "rotation_geodesic_raw_degrees": raw_rotation / denominator * (180.0 / math.pi),
    }
