"""CPU-only camera preprocessing shared by demo and precomputed NPZ export.

Only NumPy/SciPy and the existing camera discretizer are imported here.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FORMAT = "demo_camera_precomputed_v1"
NORMALIZATION = {
    "normalize_relative_to_first_frame": True,
    "normalize_framewise": True,
    "normalize_displacement": True,
    "normalize_displacement_per_video": True,
}


def normalize_poses(
    poses,
    segment_boundaries,
    normalize_relative_to_first_frame=True,
    normalize_framewise=True,
    normalize_displacement=True,
    normalize_displacement_per_video=False,
):
    """Same SE(3)/p99 algorithm as the training per-segment normalizer."""
    assert not normalize_displacement_per_video or normalize_displacement
    assert not normalize_displacement or normalize_framewise
    assert not normalize_framewise or normalize_relative_to_first_frame
    if not normalize_relative_to_first_frame:
        return poses.copy()
    out = np.zeros_like(poses)
    identity = np.eye(4, dtype=poses.dtype)

    def scale_translations(translations):
        raw = (
            float(np.quantile(np.linalg.norm(translations, axis=1), 0.99))
            if len(translations)
            else 0.0
        )
        if not np.isfinite(raw) or raw < 0:
            raise ValueError("Invalid displacement scale")
        return max(raw, 1e-6)

    for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
        if end <= start:
            continue
        relative = identity @ np.linalg.inv(poses[start])
        out[start] = identity
        if end - start > 1:
            out[start + 1 : end] = relative @ poses[start + 1 : end]
        if normalize_framewise and end - start > 1:
            integral = out[start:end]
            inverse = np.zeros_like(integral[:-1])
            rotation_inverse = np.swapaxes(integral[:-1, :3, :3], -1, -2)
            translation_inverse = -np.matmul(rotation_inverse, integral[:-1, :3, 3:4])
            inverse[:, :3, :3] = rotation_inverse
            inverse[:, :3, 3:4] = translation_inverse
            inverse[:, 3, 3] = 1.0
            out[start + 1 : end] = np.matmul(inverse, integral[1:])
            if normalize_displacement and not normalize_displacement_per_video:
                out[start:end, :3, 3] /= scale_translations(out[start + 1 : end, :3, 3])
    if (
        normalize_framewise
        and normalize_displacement
        and normalize_displacement_per_video
        and len(out) > 1
    ):
        chunks = [
            out[start + 1 : end, :3, 3]
            for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:])
            if end - start > 1
        ]
        translations = np.concatenate(chunks) if chunks else out[:0, :3, 3]
        out[:, :3, 3] /= scale_translations(translations)
    return out


def resample(c2ws, intrinsics, target):
    n = len(c2ws)
    if n < 1 or target < 1:
        raise ValueError("Camera timeline must be nonempty")
    if n == target:
        return c2ws.copy(), intrinsics.copy()
    if n == 1:
        return np.repeat(c2ws, target, axis=0), np.repeat(intrinsics, target, axis=0)
    src, dst = np.linspace(0, 1, n), np.linspace(0, 1, target)
    poses = np.repeat(np.eye(4)[None], target, axis=0)
    poses[:, :3, :3] = Slerp(src, Rotation.from_matrix(c2ws[:, :3, :3]))(
        dst
    ).as_matrix()
    poses[:, :3, 3] = np.stack(
        [np.interp(dst, src, c2ws[:, axis, 3]) for axis in range(3)], axis=1
    )
    flat = intrinsics.reshape(n, -1)
    ks = np.stack(
        [np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1
    )
    return poses.astype(c2ws.dtype), ks.reshape(target, 3, 3).astype(intrinsics.dtype)


def prepare_turn(camera, num_frames, flags=None):
    """Return the demo's float32, pre-model tensors for ONE original turn.

    Exactly follows inline conversion -> discretize -> canonical resampling ->
    per-request normalization. Inference applies camera gains after this step.
    Currently supports one segment per turn, as in the supplied camera export.
    """
    from .discretize_poses import discretize_poses

    flags = dict(NORMALIZATION if flags is None else flags)
    count = (int(num_frames) - 1) // 4 * 4 + 1
    if count < 1:
        raise ValueError("Invalid turn frame count")
    if not camera:
        return {
            "norm_extrinsics": np.repeat(
                np.eye(4, dtype=np.float32)[None], count, axis=0
            ),
            "norm_intrinsics": np.repeat(
                np.eye(3, dtype=np.float32)[None], count, axis=0
            ),
            "action_labels": np.zeros(count, dtype=np.int64),
        }
    camera = json.loads(camera) if isinstance(camera, str) else camera
    pose_dict = {}
    for index, frame in enumerate(camera["frames"]):
        pos = np.array([frame["position"][a] for a in "xyz"], dtype=np.float64)
        quat = np.array([frame["quaternion"][a] for a in "xyzw"], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isfinite(pos).all() or not np.isfinite(quat).all() or norm < 1e-8:
            raise ValueError("Invalid recorded position/quaternion")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quat / norm).as_matrix() @ np.diag(
            [1.0, -1.0, -1.0]
        )
        pose[:3, 3] = pos
        fov = float(frame.get("fov", 55.0))
        if not 0 < fov < 180:
            raise ValueError("Invalid FOV")
        focal = 0.5 / np.tan(np.radians(fov) / 2)
        k = np.array([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1.0]])
        pose_dict[str(index)] = {"extrinsic": pose.tolist(), "K": k.tolist()}
    if not pose_dict:
        raise ValueError("Empty camera recording")
    poses, ks, labels = discretize_poses(
        pose_dict, mode="4t4r_robust", downsample_ratio=1, normalize_displacement=False
    )
    if len(poses) != count:
        poses, ks = resample(poses, ks, count)
        _, _, labels = discretize_poses(
            {i: {"extrinsic": poses[i], "intrinsic": ks[i]} for i in range(count)},
            mode="4t4r_robust",
            downsample_ratio=1,
            normalize_displacement=False,
        )
    if flags["normalize_relative_to_first_frame"]:
        poses = normalize_poses(poses.astype(np.float64), [0, count], **flags).astype(
            np.float32
        )
    if not np.isfinite(poses).all() or np.abs(poses).max() > 1e6:
        raise ValueError("Invalid normalized camera magnitude")
    return {
        "norm_extrinsics": poses.astype(np.float32),
        "norm_intrinsics": ks.astype(np.float32),
        "action_labels": labels.astype(np.int64),
    }


def load_precomputed(data, num_frames, segment_boundaries, flags):
    """Fail closed on changed settings/timeline; never normalize or resample."""
    if str(data["camera_format"].item()) != FORMAT:
        raise ValueError("Unsupported precomputed camera format")
    recorded_flags = json.loads(str(data["normalization_json"].item()))
    if recorded_flags != flags:
        raise ValueError(
            f"Precomputed camera normalization mismatch: stored={recorded_flags}, runtime={flags}"
        )
    bounds = data["segment_boundaries_pixel"].tolist()
    if bounds != list(segment_boundaries or []) or bounds[-1] != num_frames:
        raise ValueError(
            "Precomputed camera requires its original offline segment boundaries/length"
        )
    offsets = data["turn_camera_offsets"].tolist()
    lengths = data["turn_num_frames"].tolist()
    enabled = data["turn_has_camera"].tolist()
    n = len(data["norm_extrinsics"])
    if (
        len(bounds) != len(lengths) + 1
        or len(offsets) != len(bounds)
        or len(enabled) != len(lengths)
        or offsets[0] != 0
        or offsets[-1] != n
        or bounds[0] != 0
    ):
        raise ValueError("Invalid precomputed turn metadata")
    for i, frames in enumerate(lengths):
        count = (int(frames) - 1) // 4 * 4 + 1
        latent_count = (int(frames) - 1) // 4 + 1
        if frames <= 0 or offsets[i + 1] - offsets[i] != count:
            raise ValueError("Invalid precomputed camera turn length")
        expected_pixels = 4 * latent_count if i < len(lengths) - 1 else frames
        if bounds[i + 1] - bounds[i] != expected_pixels:
            raise ValueError(
                "Precomputed camera latent mapping does not match offline timeline"
            )
    result = {}
    for key, shape in (
        ("norm_extrinsics", (n, 4, 4)),
        ("norm_intrinsics", (n, 3, 3)),
        ("action_labels", (n,)),
    ):
        value = data[key].copy()
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid precomputed {key}")
        if key == "norm_extrinsics" and np.abs(value).max() > 1e6:
            raise ValueError("Precomputed camera magnitude exceeds safety limit")
        if key == "action_labels" and (
            not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)
        ):
            raise ValueError("Invalid precomputed action labels")
        result[key] = value
    result.update(
        turn_camera_offsets=offsets, turn_has_camera=enabled, num_frames=num_frames
    )
    return result
