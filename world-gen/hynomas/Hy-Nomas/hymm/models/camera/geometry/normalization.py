"""Camera geometry shared by public inference and camera conditioning."""

from typing import List

import numpy as np


def _intrinsics_4_to_3x3(params: np.ndarray) -> np.ndarray:
    """[fx, fy, cx, cy] -> 3x3 intrinsics matrix."""
    if params.ndim == 1:
        params = params.reshape(1, -1)
    N = params.shape[0]
    K = np.zeros((N, 3, 3), dtype=params.dtype)
    K[:, 0, 0] = params[:, 0]
    K[:, 1, 1] = params[:, 1]
    K[:, 0, 2] = params[:, 2]
    K[:, 1, 2] = params[:, 3]
    K[:, 2, 2] = 1.0
    return K


def _se3_inverse_np(T: np.ndarray) -> np.ndarray:
    """Analytical SE(3) inverse for a stack of 4x4 homogeneous transforms.

    For ``T = [[R, t], [0, 1]]`` (R orthonormal, t 3-vector), returns
    ``T^-1 = [[R^T, -R^T t], [0, 1]]``. Faster and numerically nicer than
    ``np.linalg.inv`` because it exploits the SE(3) block structure.

    Args:
        T: ``[..., 4, 4]`` homogeneous transforms (last two dims must be 4x4).

    Returns:
        Same shape ``[..., 4, 4]`` with the inverse of each transform.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]  # [..., 3, 1]
    R_inv = np.swapaxes(R, -1, -2)  # R^T
    t_inv = -np.matmul(R_inv, t)  # [..., 3, 1]
    out = np.zeros_like(T)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:4] = t_inv
    out[..., 3, 3] = 1.0
    return out


def _framewise_relative_np(integral: np.ndarray) -> np.ndarray:
    """Convert integral-relative poses to framewise (relative-to-prev-frame) poses.

    Given ``integral[i] = inv(p[ref]) @ p[i]`` (where ``ref`` is the segment
    start, so ``integral[ref] == I``), returns:

        out[ref] = I
        out[i]   = inv(integral[i-1]) @ integral[i]   for i > ref

    This is exactly the framewise step in
    ``hymm.models.camera.camera_modules.compute_relative_poses`` (with the
    standard SE(3) algebra simplification giving
    ``out[i] = inv(p[i-1]) @ p[i]`` regardless of the chosen ref).

    Args:
        integral: ``[N, 4, 4]`` integral-relative poses with ``integral[0] == I``.

    Returns:
        ``[N, 4, 4]`` framewise poses with ``out[0] == I``.
    """
    if integral.shape[0] <= 1:
        return integral.copy()
    out = np.zeros_like(integral)
    out[0] = integral[0]  # I (kept)
    out[1:] = np.matmul(_se3_inverse_np(integral[:-1]), integral[1:])
    return out


_DISPLACEMENT_QUANTILE = 0.99


_MIN_DISPLACEMENT_SCALE = 1e-6


def _displacement_scale(translations: np.ndarray) -> float:
    norms = np.linalg.norm(translations, axis=1)
    raw_scale = (
        float(np.quantile(norms, _DISPLACEMENT_QUANTILE)) if len(norms) > 0 else 0.0
    )
    if not np.isfinite(raw_scale) or raw_scale < 0.0:
        raise ValueError(
            f"displacement scale must be finite and non-negative, got {raw_scale}"
        )
    return raw_scale


def _assert_normalized_displacement(
    translations: np.ndarray,
    raw_scale: float,
) -> None:
    assert np.isfinite(translations).all(), (
        "normalized camera translations must be finite"
    )
    if raw_scale >= _MIN_DISPLACEMENT_SCALE and len(translations) > 0:
        normalized_p99 = float(
            np.quantile(
                np.linalg.norm(translations, axis=1),
                _DISPLACEMENT_QUANTILE,
            )
        )
        assert np.isclose(normalized_p99, 1.0, rtol=1e-5, atol=1e-7), (
            f"normalized displacement p99 must be 1.0, got {normalized_p99}"
        )


def _normalize_poses_per_segment(
    poses: np.ndarray,
    segment_boundaries: List[int],
    normalize_relative_to_first_frame: bool = True,
    normalize_framewise: bool = True,
    normalize_displacement: bool = True,
    normalize_displacement_per_video: bool = False,
) -> np.ndarray:
    """Per-segment c2w normalization, mirroring the model-side ``compute_relative_poses``.

    Within each segment ``[c_j, c_{j+1})``, three progressive steps (each
    requires the previous one):

      1. First-frame: re-anchor to segment's first frame (``out[c_j] == I``).
      2. Framewise: convert integral-relative to relative-to-prev-frame.
      3. Displacement: divide translations by either each segment's 99th
         percentile norm or one full-video 99th percentile norm.

    When step 1 is disabled, raw poses are returned unchanged.

    Args:
        poses: [N, 4, 4] c2w matrices.
        segment_boundaries: camera-frame indices, length ``num_segments + 1``,
            starts at 0 and ends at N.
        normalize_relative_to_first_frame: enable step 1.
        normalize_framewise: enable step 2 (requires step 1).
        normalize_displacement: enable step 3 (requires step 2).
        normalize_displacement_per_video: when step 3 is enabled, use one
            99th-percentile translation norm over every segment instead of a
            separate percentile for each segment.

    Returns:
        ``[N, 4, 4]`` normalized poses.
    """
    assert not normalize_displacement_per_video or normalize_displacement, (
        "normalize_displacement_per_video requires normalize_displacement"
    )
    assert not normalize_displacement or normalize_framewise, (
        "normalize_displacement requires normalize_framewise"
    )
    assert not normalize_framewise or normalize_relative_to_first_frame, (
        "normalize_framewise requires normalize_relative_to_first_frame"
    )

    if not normalize_relative_to_first_frame:
        return poses.copy()

    out = np.zeros_like(poses)
    target_cam_c2w = np.eye(4, dtype=poses.dtype)

    # Step 1: per-segment first-frame normalization.
    for c0, c1 in zip(segment_boundaries[:-1], segment_boundaries[1:]):
        if c1 <= c0:
            continue
        rel = target_cam_c2w @ np.linalg.inv(poses[c0])
        out[c0] = target_cam_c2w
        if c1 - c0 > 1:
            out[c0 + 1 : c1] = rel @ poses[c0 + 1 : c1]

        # Step 2: per-segment framewise (relative-to-prev-frame).
        if normalize_framewise and c1 - c0 > 1:
            out[c0:c1] = _framewise_relative_np(out[c0:c1])

            # Step 3a: per-segment displacement normalization.
            if (
                normalize_displacement
                and not normalize_displacement_per_video
                and c1 - c0 > 1
            ):
                translations = out[c0 + 1 : c1, :3, 3]
                raw_scale = _displacement_scale(translations)
                scale = max(raw_scale, _MIN_DISPLACEMENT_SCALE)
                assert np.isfinite(scale) and scale > 0.0
                out[c0:c1, :3, 3] /= scale
                _assert_normalized_displacement(
                    out[c0 + 1 : c1, :3, 3],
                    raw_scale,
                )

    # Step 3b: one translation scale shared by the complete video. Segment
    # starts remain identity/zero, while relative scale across segments is
    # preserved.
    if (
        normalize_framewise
        and normalize_displacement
        and normalize_displacement_per_video
        and len(out) > 1
    ):
        translation_chunks = [
            out[c0 + 1 : c1, :3, 3]
            for c0, c1 in zip(
                segment_boundaries[:-1],
                segment_boundaries[1:],
            )
            if c1 - c0 > 1
        ]
        translations = (
            np.concatenate(translation_chunks, axis=0)
            if translation_chunks
            else out[:0, :3, 3]
        )
        raw_scale = _displacement_scale(translations)
        scale = max(raw_scale, _MIN_DISPLACEMENT_SCALE)
        assert np.isfinite(scale) and scale > 0.0
        out[:, :3, 3] /= scale
        _assert_normalized_displacement(
            translations / scale,
            raw_scale,
        )
    return out
