"""Non-AR bidirectional multi-prompt video generation pipeline (one_shot)."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as R, Slerp

from hymm.utils.camera_cfg import (
    camera_guidance_is_active,
    combine_camera_text_cfg,
)
from hymm.utils.camera_utils import (
    canonical_camera_frame_count,
    pixel_to_camera_boundaries,
)
from hymm.pipeline.base import BasePipeline, align_to
from hymm.pipeline.solvers.euler import EulerSolver


def get_dynamic_guidance_scale(
    step, total_step, guidance_scale_min, guidance_scale_max, guidance_scale_max_step
):
    if step < guidance_scale_max_step:
        return guidance_scale_min + (guidance_scale_max - guidance_scale_min) * (
            step / guidance_scale_max_step
        )
    else:
        return guidance_scale_max - (guidance_scale_max - guidance_scale_min) * (
            step - guidance_scale_max_step
        ) / (total_step - guidance_scale_max_step)


def get_cfg_scale_for_step(cfg_schedule, step, total_steps, cfg_scale, **kwargs):
    """Return the CFG scale for a given denoising step based on schedule type.

    Supported schedules:
        None / "constant"  – fixed ``cfg_scale`` every step (default).
        "skip_first"       – cfg_scale=1.0 on step 0, then ``cfg_scale``.
        "skip_first_n"     – cfg_scale=1.0 for the first ``n`` steps (via
                             ``cfg_skip_steps`` kwarg, default 1).
        "cosine"           – cosine decay from ``cfg_scale`` → 1.0.
        "linear_ramp"      – linear decay from ``cfg_scale`` → 1.0.
        "linear_peak"      – legacy triangle schedule (wraps
                             ``get_dynamic_guidance_scale``).
    """
    import math

    if cfg_schedule is None or cfg_schedule == "constant":
        return cfg_scale

    if cfg_schedule == "skip_first":
        return 1.0 if step == 0 else cfg_scale

    if cfg_schedule == "skip_first_n":
        n = kwargs.get("cfg_skip_steps", 1)
        return 1.0 if step < n else cfg_scale

    if cfg_schedule == "cosine":
        progress = step / max(total_steps - 1, 1)
        return 1.0 + (cfg_scale - 1.0) * (1 + math.cos(math.pi * progress)) / 2

    if cfg_schedule == "linear_ramp":
        progress = step / max(total_steps - 1, 1)
        return cfg_scale - (cfg_scale - 1.0) * progress

    if cfg_schedule == "linear_peak":
        peak_step = kwargs.get("cfg_scale_max_step", total_steps // 2)
        cfg_min = kwargs.get("cfg_scale_min", 1.0)
        cfg_max = kwargs.get("cfg_scale_max", cfg_scale)
        return get_dynamic_guidance_scale(
            step, total_steps, cfg_min, cfg_max, peak_step
        )

    raise ValueError(f"Unknown cfg_schedule: {cfg_schedule!r}")


# ============================================================================
# Camera trajectory utilities
# ============================================================================

_TRAJECTORIES_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "camera"
    / "geometry"
    / "trajectories"
)


def _inline_camera_frames_to_pose_dict(payload: object) -> Optional[Dict[str, Dict]]:
    """Convert camera_vis-style inline JSON into the pipeline pose format."""
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
    elif isinstance(payload, list):
        frames = payload
    else:
        return None

    pose_dict: Dict[str, Dict] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Invalid inline camera frame at index {index}")
        position = frame.get("position")
        quaternion = frame.get("quaternion")
        if not isinstance(position, dict) or not isinstance(quaternion, dict):
            raise ValueError(
                f"Inline camera frame {index} must contain position and quaternion"
            )
        try:
            translation = np.array(
                [position["x"], position["y"], position["z"]],
                dtype=np.float64,
            )
            quat_xyzw = np.array(
                [
                    quaternion["x"],
                    quaternion["y"],
                    quaternion["z"],
                    quaternion["w"],
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid inline camera frame {index}") from exc
        if not np.isfinite(translation).all() or not np.isfinite(quat_xyzw).all():
            raise ValueError(f"Non-finite inline camera frame at index {index}")
        quat_norm = np.linalg.norm(quat_xyzw)
        if quat_norm < 1e-8:
            raise ValueError(f"Zero-length quaternion at inline camera frame {index}")

        # Browser recordings use Three.js camera coordinates
        # (+X right, +Y up, camera forward -Z). The model camera path loader
        # expects OpenCV camera coordinates (+X right, +Y down, +Z forward).
        # Keep the world-space position and flip the local Y/Z basis vectors.
        rotation_three = R.from_quat(quat_xyzw / quat_norm).as_matrix()
        rotation = rotation_three @ np.diag([1.0, -1.0, -1.0])
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation

        fov = float(frame.get("fov", 55.0))
        focal = 0.5 / np.tan(np.radians(fov) / 2.0)
        intrinsic = np.array(
            [[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        pose_dict[str(index)] = {
            "extrinsic": extrinsic.tolist(),
            "K": intrinsic.tolist(),
        }
    return pose_dict or None


def _resample_camera_poses(
    c2ws: np.ndarray,
    intrinsics: np.ndarray,
    target_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothly resample camera poses onto an evenly spaced target timeline.

    Translation and intrinsics use component-wise linear interpolation.
    Rotations use quaternion SLERP, which is the rotation-space equivalent of
    linear interpolation and avoids the duplicate-pose / large-jump pattern
    produced by nearest-index resampling.
    """
    source_frames = len(c2ws)
    target_frames = int(target_frames)
    if source_frames <= 0:
        raise ValueError("camera trajectory must contain at least one frame")
    if target_frames <= 0:
        raise ValueError(
            f"target camera frame count must be positive, got {target_frames}"
        )
    if intrinsics.shape[0] != source_frames:
        raise ValueError(
            "camera extrinsics/intrinsics length mismatch: "
            f"{source_frames} != {intrinsics.shape[0]}"
        )
    if source_frames == target_frames:
        return c2ws.copy(), intrinsics.copy()
    if source_frames == 1:
        return (
            np.repeat(c2ws, target_frames, axis=0),
            np.repeat(intrinsics, target_frames, axis=0),
        )

    source_time = np.linspace(0.0, 1.0, source_frames, dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)

    rotations = Slerp(
        source_time,
        R.from_matrix(np.asarray(c2ws[:, :3, :3], dtype=np.float64)),
    )(target_time).as_matrix()

    translations = np.stack(
        [np.interp(target_time, source_time, c2ws[:, axis, 3]) for axis in range(3)],
        axis=1,
    )
    flat_intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(
        source_frames, -1
    )
    interpolated_intrinsics = np.stack(
        [
            np.interp(target_time, source_time, flat_intrinsics[:, index])
            for index in range(flat_intrinsics.shape[1])
        ],
        axis=1,
    ).reshape((target_frames,) + intrinsics.shape[1:])

    resampled_c2ws = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        target_frames,
        axis=0,
    )
    resampled_c2ws[:, :3, :3] = rotations
    resampled_c2ws[:, :3, 3] = translations
    return (
        resampled_c2ws.astype(c2ws.dtype, copy=False),
        interpolated_intrinsics.astype(intrinsics.dtype, copy=False),
    )


def _resample_demo_camera_segments(c2ws, intrinsics, pixel_boundaries):
    """Align exported demo turns independently; never interpolate across turns.

    Pixel-to-camera boundary mapping is not a uniform whole-clip resampling.
    Crossing turn boundaries can leak translation into rotation-only turns and
    cause enormous displacement-normalized spikes.
    """
    bounds = [int(value) for value in pixel_boundaries]
    if (
        len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != len(c2ws)
        or any(a >= b for a, b in zip(bounds[:-1], bounds[1:]))
    ):
        raise ValueError("Invalid demo NPZ segment_boundaries_pixel")
    camera_bounds = pixel_to_camera_boundaries(bounds, len(c2ws))
    if len(camera_bounds) != len(bounds):
        raise ValueError(
            "Demo camera segments are too short for the VAE camera timeline"
        )
    pose_parts, intrinsic_parts = [], []
    for start, end, cstart, cend in zip(
        bounds[:-1], bounds[1:], camera_bounds[:-1], camera_bounds[1:]
    ):
        poses, ks = _resample_camera_poses(
            c2ws[start:end], intrinsics[start:end], cend - cstart
        )
        pose_parts.append(poses)
        intrinsic_parts.append(ks)
    return (
        np.concatenate(pose_parts),
        np.concatenate(intrinsic_parts),
        camera_bounds,
    )


def prepare_camera_inputs(
    trajectory_source: str,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    normalize_relative_to_first_frame: bool = False,
    normalize_framewise: bool = False,
    normalize_displacement: bool = False,
    normalize_displacement_per_video: bool = False,
    segment_boundaries_pixel: Optional[List[int]] = None,
    translation_scale: Optional[float] = None,
    rotation_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Prepare camera control tensors from a trajectory source.

    Args:
        trajectory_source: Predefined trajectory name (e.g. "pan_left"),
            path to a trajectory JSON file, or path to a .npz file.
        num_frames: Number of *pixel* frames in the generated video.
            For .npz files this is ignored — the trajectory length is used directly.
        device: Target torch device.
        dtype: Target torch dtype.
        normalize_relative_to_first_frame: Mirror the training dataloader's
            ``WM1MCameraDataset`` setting. When True, re-anchor poses so that
            the (per-segment) first frame becomes identity. Critical for
            train/eval distribution match: without this, a model trained on
            first-frame-normalized poses sees an out-of-distribution input
            domain (raw absolute c2ws).
        normalize_framewise: When True, after first-frame normalization,
            convert integral-relative poses to relative-to-prev-frame poses.
            Only applied if ``normalize_relative_to_first_frame=True``.
        normalize_displacement: When True, divide each segment's translations
            by its 99th-percentile translation magnitude. Only applied if
            ``normalize_relative_to_first_frame=True``.
        normalize_displacement_per_video: When True, use one translation
            99th-percentile norm across the complete video instead of one per segment.
            Per-segment first-frame and framewise normalization are unchanged.
        segment_boundaries_pixel: Per-segment boundaries in pixel-frame units
            (length ``num_segments + 1``, starting at 0, ending at the camera
            trajectory length). When provided AND consistent with the loaded
            trajectory length, normalization runs **per segment** (matches
            training). When ``None``
            or inconsistent (e.g. NPZ overrode num_frames), falls back to
            whole-clip normalization. Ignored entirely if
            ``normalize_relative_to_first_frame=False``.
        translation_scale: Positive inference gain applied to translations AFTER
            normalization, including displacement normalization. Defaults to 1.0.
            This is the only translation gain; trajectory metadata is ignored.
        rotation_scale: Positive gain on local rotation angles AFTER framewise
            normalization: Exp(scale * Log(delta_R)). Defaults to 1.0
            (disabled). Translations and action labels are unchanged.

    Returns:
        Dict with keys: norm_extrinsics [1,T,4,4], norm_intrinsics [1,T,3,3],
        action_labels [1,T], action_onehot [1,T,8], num_frames (int).
    """
    translation_scale = 1.0 if translation_scale is None else float(translation_scale)
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError("trajectory_translation_scale must be finite and positive")
    rotation_scale = float(rotation_scale)
    if not np.isfinite(rotation_scale) or rotation_scale <= 0:
        raise ValueError("trajectory_rotation_scale must be finite and positive")
    if rotation_scale != 1.0 and not (
        normalize_relative_to_first_frame and normalize_framewise
    ):
        raise ValueError(
            "Rotation gains require normalize_relative_to_first_frame=true "
            "and normalize_framewise=true; do not scale absolute poses"
        )
    from hymm.models.camera.geometry.rotation_scale import scale_framewise_rotations
    from hymm.models.camera.geometry.discretize_poses import discretize_poses
    from hymm.models.camera.geometry.action_labels import action_labels_to_onehot
    from hymm.models.camera.geometry.generate_trajectories import TRAJECTORY_GENERATORS

    # Reuse the dataloader's per-segment normalization helper so train and eval
    # share **exactly** the same SE(3) math. If this drifts the model will see
    # an OOD pose distribution at inference time.
    from hymm.models.camera.geometry.normalization import _normalize_poses_per_segment

    from hymm.models.camera.geometry.demo_precomputed import (
        load_precomputed,
        prepare_turn,
    )

    normalization_flags = {
        "normalize_relative_to_first_frame": normalize_relative_to_first_frame,
        "normalize_framewise": normalize_framewise,
        "normalize_displacement": normalize_displacement,
        "normalize_displacement_per_video": normalize_displacement_per_video,
    }

    def apply_camera_gains(extrinsics):
        if translation_scale != 1.0:
            extrinsics = extrinsics.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                extrinsics[:, :3, 3] *= translation_scale
            if not np.isfinite(extrinsics[:, :3, 3]).all():
                raise ValueError(
                    "trajectory_translation_scale produced nonfinite translations"
                )
            logger.info(
                f"  Translation gain applied: {translation_scale:.4f}x (post-normalize)"
            )
        if rotation_scale != 1.0:
            extrinsics = scale_framewise_rotations(extrinsics, rotation_scale)
            logger.info(
                f"  Local rotation gain: {rotation_scale:.4f}x (post-normalize)"
            )
        return extrinsics

    def as_camera_tensors(arrays):
        labels = arrays["action_labels"]
        extrinsics = apply_camera_gains(arrays["norm_extrinsics"])
        result = {
            "norm_extrinsics": torch.from_numpy(extrinsics)
            .float()
            .unsqueeze(0)
            .to(device=device, dtype=dtype),
            "norm_intrinsics": torch.from_numpy(arrays["norm_intrinsics"])
            .float()
            .unsqueeze(0)
            .to(device=device, dtype=dtype),
            "action_labels": torch.from_numpy(labels)
            .long()
            .unsqueeze(0)
            .to(device=device),
            "action_onehot": torch.from_numpy(
                action_labels_to_onehot(labels, mode="4t4r")
            )
            .float()
            .unsqueeze(0)
            .to(device=device, dtype=dtype),
            "num_frames": arrays.get("num_frames", num_frames),
        }
        for key in ("turn_camera_offsets", "turn_has_camera"):
            if key in arrays:
                result[key] = arrays[key]
        return result

    if trajectory_source.endswith(".npz") and os.path.isfile(trajectory_source):
        with np.load(trajectory_source, allow_pickle=False) as precomputed:
            if "camera_format" in precomputed:
                arrays = load_precomputed(
                    precomputed,
                    num_frames,
                    segment_boundaries_pixel,
                    normalization_flags,
                )
                logger.info(
                    "Loaded precomputed demo camera turns: no resampling, "
                    "normalization is skipped; inference gains are applied next."
                )
                return as_camera_tensors(arrays)
    # The common demo case: one recorded camera turn, one local segment.
    # Share the exact CPU path with the offline exporter. Multi-segment inline
    # requests and legacy trajectories retain their original handling below.
    if trajectory_source.lstrip().startswith("{"):
        inline = json.loads(trajectory_source)
        if isinstance(inline.get("frames"), list) and (
            segment_boundaries_pixel is None
            or list(segment_boundaries_pixel) == [0, num_frames]
        ):
            arrays = prepare_turn(inline, num_frames, normalization_flags)
            return as_camera_tensors(arrays)

    # 1. Load or generate trajectory
    is_npz = os.path.isfile(trajectory_source) and trajectory_source.endswith(".npz")
    demo_pixel_boundaries = None
    demo_camera_boundaries = None

    if is_npz:
        # NPZ file: load c2ws and intrinsics directly (same as old camera_trainer.py)
        # Trajectory length determines video length — no resampling.
        data = np.load(trajectory_source)
        c2ws = data["c2ws"].copy()  # [T, 4, 4]
        intrinsics = np.asarray(data["intrinsics"]).copy()
        # Accept both constructed-bench [T,3,3] and vipe-export formats:
        # shared [fx,fy,cx,cy] / [1,4] / [T,4]. Matches WM1MCameraDataset.
        from hymm.models.camera.geometry.normalization import _intrinsics_4_to_3x3

        if intrinsics.ndim == 1:
            intrinsics = intrinsics.reshape(1, -1)
        if intrinsics.ndim == 2 and intrinsics.shape[-1] == 4:
            intrinsics = _intrinsics_4_to_3x3(intrinsics)
        elif not (intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3)):
            raise ValueError(
                f"Unsupported intrinsics shape {tuple(intrinsics.shape)} in "
                f"{trajectory_source!r}; expected [T,3,3] or [fx,fy,cx,cy]/[T,4]."
            )
        if intrinsics.shape[0] == 1 and len(c2ws) > 1:
            intrinsics = np.broadcast_to(
                intrinsics, (len(c2ws),) + intrinsics.shape[1:]
            ).copy()
        elif intrinsics.shape[0] != len(c2ws):
            raise ValueError(
                f"c2ws/intrinsics length mismatch in {trajectory_source!r}: "
                f"c2ws={len(c2ws)} intrinsics={intrinsics.shape[0]}"
            )
        num_frames = len(c2ws)
        if "segment_boundaries_pixel" in data:
            demo_pixel_boundaries = data["segment_boundaries_pixel"].tolist()
        logger.info(
            f"Loaded camera trajectory from npz: {trajectory_source} "
            f"(c2ws={c2ws.shape}, intrinsics={intrinsics.shape}, "
            f"overriding num_frames={num_frames})"
        )

        # Compute action_labels from raw c2ws (only for action_onehot, not used in extrinsics_attn)
        pose_dict = {
            i: {"extrinsic": c2ws[i], "intrinsic": intrinsics[i]}
            for i in range(len(c2ws))
        }
        _, _, action_labels = discretize_poses(
            pose_dict,
            mode="4t4r_robust",
            downsample_ratio=1,
            normalize_displacement=False,
        )
    else:
        # JSON file / predefined name / trajectory generator
        inline_pose_dict = None
        stripped_source = trajectory_source.lstrip()
        if stripped_source.startswith("{") or stripped_source.startswith("["):
            try:
                inline_payload = json.loads(trajectory_source)
                inline_pose_dict = _inline_camera_frames_to_pose_dict(inline_payload)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid inline camera trajectory JSON") from exc

        if inline_pose_dict is not None:
            pose_dict = inline_pose_dict
            logger.info(f"Loaded inline camera trajectory ({len(pose_dict)} frames)")
        elif os.path.isfile(trajectory_source):
            with open(trajectory_source, "r") as f:
                pose_dict = json.load(f)
            logger.info(
                f"Loaded camera trajectory from file: {trajectory_source} ({len(pose_dict)} frames)"
            )
        elif (_TRAJECTORIES_DIR / f"{trajectory_source}.json").is_file():
            traj_path = _TRAJECTORIES_DIR / f"{trajectory_source}.json"
            with open(traj_path, "r") as f:
                pose_dict = json.load(f)
            logger.info(
                f"Loaded predefined trajectory: {trajectory_source} ({len(pose_dict)} frames)"
            )
        elif trajectory_source in TRAJECTORY_GENERATORS:
            pose_dict = TRAJECTORY_GENERATORS[trajectory_source](num_frames=num_frames)
            logger.info(
                f"Generated trajectory on-the-fly: {trajectory_source} ({len(pose_dict)} frames)"
            )
        else:
            raise FileNotFoundError(
                f"Camera trajectory '{trajectory_source}' not found. "
                f"Provide a valid file path, or one of the predefined names: "
                f"{list(TRAJECTORY_GENERATORS.keys())}"
            )

        # Discretize poses -> c2ws, intrinsics, action_labels
        c2ws, intrinsics, action_labels = discretize_poses(
            pose_dict,
            mode="4t4r_robust",
            downsample_ratio=1,
            normalize_displacement=False,
        )
        logger.info(
            f"  Discretized: c2ws={c2ws.shape}, intrinsics={intrinsics.shape}, "
            f"action_labels={action_labels.shape}, unique_labels={len(np.unique(action_labels))}"
        )

    # Align camera conditioning to the first-frame-aware VAE timeline used in
    # training. A video with L latent frames has (L - 1) * 4 + 1 canonical
    # camera frames; _preprocess_camera then repeats camera frame 0 three times
    # and groups every four matrices into exactly L camera tokens.
    target_camera_frames = canonical_camera_frame_count(num_frames)
    traj_len = len(action_labels)
    if traj_len != target_camera_frames or demo_pixel_boundaries is not None:
        if demo_pixel_boundaries is not None:
            c2ws, intrinsics, demo_camera_boundaries = (
                _resample_demo_camera_segments(
                    c2ws, intrinsics, demo_pixel_boundaries
                )
            )
        else:
            c2ws, intrinsics = _resample_camera_poses(
                c2ws,
                intrinsics,
                target_camera_frames,
            )
        resampled_pose_dict = {
            index: {
                "extrinsic": c2ws[index],
                "intrinsic": intrinsics[index],
            }
            for index in range(target_camera_frames)
        }
        _, _, action_labels = discretize_poses(
            resampled_pose_dict,
            mode="4t4r_robust",
            downsample_ratio=1,
            normalize_displacement=False,
        )
        logger.info(
            f"  Training-aligned camera interpolation: {traj_len} -> "
            f"{target_camera_frames} frames (video remains {num_frames} frames)"
        )
    traj_len = len(action_labels)
    action_onehot = action_labels_to_onehot(action_labels, mode="4t4r")

    # ========== Pose normalization (mirrors training dataloader) ==========
    # The training dataloader (`WM1MCameraDataset._vipe_to_camera_tensors`)
    # applies first-frame / framewise / displacement normalization in c2w
    # space before feeding poses to the model. If inference skips this, the
    # model sees an out-of-distribution pose domain and camera control quality
    # degrades. Match the dataloader's pipeline exactly here.
    if normalize_relative_to_first_frame:
        traj_len = len(c2ws)
        # Validate provided segment boundaries. They must:
        #   - have length >= 2 (at least one segment)
        #   - start at 0 and end at traj_len (cover the whole clip)
        #   - be strictly monotonically increasing
        # Pixel boundaries follow the output-video timeline. Convert them to
        # the canonical camera timeline via the same latent boundary mapping
        # used by WM1MCameraDataset: pixel//4 -> max(0, 4*latent - 3).
        boundaries: List[int]
        if demo_camera_boundaries is not None:
            boundaries = demo_camera_boundaries
            logger.info(
                f"  Demo NPZ per-segment camera alignment: "
                f"pixel_boundaries={demo_pixel_boundaries} -> camera_boundaries={boundaries}"
            )
        elif segment_boundaries_pixel is not None:
            pixel_sb = [int(b) for b in segment_boundaries_pixel]
            valid_pixel = (
                len(pixel_sb) >= 2
                and pixel_sb[0] == 0
                and pixel_sb[-1] == num_frames
                and all(pixel_sb[i] < pixel_sb[i + 1] for i in range(len(pixel_sb) - 1))
            )
            sb = pixel_to_camera_boundaries(pixel_sb, num_frames) if valid_pixel else []
            valid_camera = (
                len(sb) == len(pixel_sb)
                and sb[0] == 0
                and sb[-1] == traj_len
                and all(sb[i] < sb[i + 1] for i in range(len(sb) - 1))
            )
            if valid_camera:
                boundaries = sb
                if len(sb) > 2:
                    logger.info(
                        f"  Camera per-segment normalization: "
                        f"pixel_boundaries={pixel_sb} -> camera_boundaries={sb}"
                    )
            else:
                logger.info(
                    f"  Camera per-segment boundaries {segment_boundaries_pixel} "
                    f"inconsistent with video length {num_frames} or canonical "
                    f"camera length {traj_len}; falling back to whole-clip normalization."
                )
                boundaries = [0, traj_len]
        else:
            boundaries = [0, traj_len]

        # _normalize_poses_per_segment requires float64 to avoid LU pivoting
        # garbage on near-singular inputs (matches dataloader's promotion).
        c2ws_norm = _normalize_poses_per_segment(
            c2ws.astype(np.float64),
            segment_boundaries=boundaries,
            normalize_framewise=normalize_framewise,
            normalize_displacement=normalize_displacement,
            normalize_displacement_per_video=normalize_displacement_per_video,
        )

        # Defensive sanity checks (mirror dataloader's `_vipe_to_camera_tensors`).
        # Inference can't gracefully demote to text-only like training does,
        # so a hard error here is the right behavior — better fail loudly than
        # silently produce garbage videos from rank-deficient pose math.
        if not np.isfinite(c2ws_norm).all():
            raise ValueError(
                "Camera normalization produced NaN/Inf c2ws. "
                f"trajectory_source={trajectory_source!r}"
            )
        max_abs = float(np.max(np.abs(c2ws_norm)))
        if max_abs > 1e6:
            raise ValueError(
                f"Camera normalization produced c2ws magnitude {max_abs:.3e} "
                f"exceeding 1e6 threshold (rank-deficient input suspected). "
                f"trajectory_source={trajectory_source!r}"
            )

        c2ws = c2ws_norm.astype(np.float32)
        logger.info(
            f"  Camera normalization applied: "
            f"first_frame={normalize_relative_to_first_frame}, "
            f"framewise={normalize_framewise}, "
            f"displacement={normalize_displacement}, "
            f"segments={len(boundaries) - 1}"
        )
    else:
        logger.info(
            "  Camera normalization SKIPPED (normalize_relative_to_first_frame=False). "
            "Model will receive raw c2ws — verify this matches training config!"
        )

    # 5. Apply inference gains once, after normalization and before embedding.
    c2ws = apply_camera_gains(c2ws)

    # 6. Convert to tensors with batch dim
    norm_extrinsics = (
        torch.from_numpy(c2ws).float().unsqueeze(0).to(device=device, dtype=dtype)
    )
    norm_intrinsics = (
        torch.from_numpy(intrinsics).float().unsqueeze(0).to(device=device, dtype=dtype)
    )
    action_labels_t = (
        torch.from_numpy(action_labels).long().unsqueeze(0).to(device=device)
    )
    action_onehot_t = (
        torch.from_numpy(action_onehot)
        .float()
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )

    logger.info(
        f"  Camera tensors: extrinsics={norm_extrinsics.shape}, "
        f"intrinsics={norm_intrinsics.shape}, "
        f"action_labels={action_labels_t.shape}, action_onehot={action_onehot_t.shape}"
    )

    return {
        "norm_extrinsics": norm_extrinsics,
        "norm_intrinsics": norm_intrinsics,
        "action_labels": action_labels_t,
        "action_onehot": action_onehot_t,
        "num_frames": num_frames,
    }


class MultiPromptBidirectionalPipeline(BasePipeline):
    """
    Multi-Prompt Bidirectional Video Generation Pipeline.

    Encodes each prompt independently, then passes them as a 4D text tensor
    [B, N, S, C] to CPTHYNomasDiffusionTransformer which handles per-chunk
    text modulation via magi_mp / magi_mp_causal attention.
    """

    def __init__(
        self,
        args,
        device: torch.device,
        dit_model_path: str = None,
        world_mesh=None,
    ):
        logger.info("Initializing Multi-Prompt Bidirectional Pipeline...")
        super().__init__(
            args, device, dit_model_path, world_mesh=world_mesh
        )

        logger.info("Pipeline initialized successfully!")

    def _encode_prompts(
        self,
        prompts: List[str],
        data_type: str = "video",
    ) -> dict:
        """Encode multiple prompts independently, returning lists of embeddings."""
        all_states = []
        all_masks = []
        for prompt in prompts:
            states, mask = self._encode_single_prompt(prompt, data_type=data_type)
            all_states.append(states)
            all_masks.append(mask)

        return {
            "all_states": all_states,
            "all_masks": all_masks,
        }

    def _inference(
        self,
        prompts: List[str],
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        flow_shift: float,
        seed: int,
        negative_prompt: str,
        prompt_video_boundaries: Optional[List[int]] = None,
        use_dynamic_cfg_scale: bool = False,
        cfg_scale_min: float = 1.0,
        cfg_scale_max: float = 8.0,
        cfg_scale_max_step: int = 16,
    ):
        """Run bidirectional inference on a list of prompts."""
        # Process prompts sequentially (batch_size = 1)
        outputs = []
        for prompt in prompts:
            # Encode text
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)

            # Set seed
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Prepare inputs
            latent_shape = (1, self.in_channels, num_frames, height // 8, width // 8)
            latents = torch.randn(latent_shape, generator=generator, device=self.device)

            # Encode prompt
            if self.text_encoder is not None:
                text_embeds = self.text_encoder(text_input_ids)[0]
            else:
                text_embeds = None

            # Run diffusion
            output = self.diffusion_pipe(
                latents=latents,
                text_embeds=text_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                flow_shift=flow_shift,
                generator=generator,
                negative_prompt=negative_prompt,
                prompt_video_boundaries=prompt_video_boundaries,
                use_dynamic_cfg_scale=use_dynamic_cfg_scale,
                cfg_scale_min=cfg_scale_min,
                cfg_scale_max=cfg_scale_max,
                cfg_scale_max_step=cfg_scale_max_step,
            )
            outputs.append(output)

        return outputs

    @torch.no_grad()
    def inference(
        self,
        prompts: List[str],
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float = 6.0,
        camera_guidance_scale: float = 1.0,
        num_inference_steps: int = 50,
        flow_shift: float = 5.0,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        prompt_video_boundaries: Optional[List[int]] = None,
        use_dynamic_cfg_scale: bool = False,
        cfg_scale_min: float = 1.0,
        cfg_scale_max: float = 8.0,
        cfg_scale_max_step: int = 16,
        cfg_schedule: Optional[str] = None,
        cfg_skip_steps: int = 0,
        camera_trajectory: Optional[str] = None,
        segment_text_drops: Optional[List[bool]] = None,
        segment_has_camera: Optional[List[bool]] = None,
        trajectory_translation_scale: Optional[float] = None,
        trajectory_rotation_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate video conditioned on multiple prompts (non-AR, bidirectional).

        segment_text_drops:
            Optional length-``num_prompts`` list of bools. ``True`` at index i
            replaces ``prompts[i]`` with ``""`` BEFORE text encoding so that
            segment runs as text-uncond. Used by the joint bench when
            ``joint_caption_constraint == "empty"``.
        segment_has_camera:
            Optional length-``num_prompts`` list of bools. Builds a
            ``[B, num_prompts]`` ``has_camera_mask`` that gates camera
            cross-attention output per segment (zero for ``False`` entries,
            keep otherwise). Used by the joint bench when
            ``segment_control == "prompt"`` (camera off on text-controlled
            half). When ``None`` (or all ``True``), camera attention runs
            normally on every segment — current default behaviour.
        """
        num_prompts = len(prompts)
        batch_size = 1

        # ---- Per-segment text drop (joint bench: joint_caption_constraint=="empty") ----
        if segment_text_drops is not None:
            assert len(segment_text_drops) == num_prompts, (
                f"segment_text_drops length {len(segment_text_drops)} "
                f"!= num_prompts {num_prompts}"
            )
            new_prompts: List[str] = []
            dropped: List[int] = []
            for i, p in enumerate(prompts):
                if segment_text_drops[i]:
                    new_prompts.append("")
                    dropped.append(i)
                else:
                    new_prompts.append(p)
            prompts = new_prompts
            if dropped:
                logger.info(
                    f"  Per-segment text drop applied at indices {dropped} "
                    f"(replaced with empty string for text-uncond)"
                )

        # Multi-segment text requires segment-aware attention (e.g. magi_mp_causal);
        # flash3 does not support segment-level causal masking and will silently ignore
        # per-chunk text modulation, producing incorrect results.
        attn_mode = getattr(self.args.dit, "attn_mode", "")
        if "flash3" in attn_mode:
            raise ValueError(
                f"Multi-segment text is incompatible with attn_mode='{attn_mode}'. "
                f"flash3 does not support segment-level causal masking required by multi-prompt "
                f"text modulation. Use 'magi_mp_causal' or similar attention mode instead."
            )

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        generator = torch.Generator(device="cpu").manual_seed(
            seed if seed is not None else 42
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)

        patch_size = self.transformer.patch_size
        in_channels = self.transformer.config.in_channels
        out_channels = self.transformer.config.out_channels

        latent_frames = (num_frames - 1) // 4 + 1

        reorg_token = (
            "888" in self.args.dit.model_type or "16168" in self.args.dit.model_type
        )
        if reorg_token and latent_frames > 1:
            latent_frames = (latent_frames + 1) // 2

        latent_h = target_height // self.s_ratio
        latent_w = target_width // self.s_ratio

        tt = latent_frames // patch_size[0]
        th = latent_h // patch_size[1]
        tw = latent_w // patch_size[2]

        logger.info(f"Multi-Prompt Bidirectional Inference:")
        logger.info(f"  Prompts: {num_prompts}")
        logger.info(
            f"  Resolution: {target_height}x{target_width}, frames={num_frames}"
        )
        logger.info(f"  Latent: T={latent_frames}, H={latent_h}, W={latent_w}")
        logger.info(f"  Patched: tt={tt}, th={th}, tw={tw}")
        logger.info(f"  Attn mode: {self.args.dit.attn_mode}")
        if cfg_schedule and cfg_schedule != "constant":
            logger.info(f"  CFG schedule: {cfg_schedule}  (base cfg={guidance_scale})")
        elif use_dynamic_cfg_scale:
            logger.info(
                f"  Dynamic CFG: min={cfg_scale_min}, max={cfg_scale_max}, "
                f"peak_step={cfg_scale_max_step}/{num_inference_steps}"
            )

        # Snapshot pixel-frame boundaries BEFORE the in-place TT conversion
        # below. Camera normalization downstream (per-segment, in c2w space)
        # operates on pixel-frame indices, matching the dataloader's
        # `segment_boundaries_camera` semantics in `WM1MCameraDataset`.
        prompt_video_boundaries_pixel: Optional[List[int]] = (
            list(prompt_video_boundaries)
            if prompt_video_boundaries is not None
            else None
        )

        if prompt_video_boundaries is not None:
            if prompt_video_boundaries[-1] != tt:
                ps_t = patch_size[0]
                tt_boundaries = []
                # NOTE: match training dataloader's formula exactly
                # (wm_1m_torch_dataset.py:2664-2665):
                #     pixel_frame = int(end_t * fps)
                #     latent_idx  = pixel_frame // vae_temporal_ratio  (floor)
                # Previously inference used (b-1)//4 + 1 (ceil-ish), which
                # produced off-by-one latent boundaries vs training for most
                # event end_times and caused magi_mp_causal to attend to the
                # wrong frames.
                for b in prompt_video_boundaries:
                    lf = b // 4  # floor, matches training
                    tt_boundaries.append(lf // ps_t)
                tt_boundaries[0] = 0
                tt_boundaries[-1] = tt
                # Clamp intermediate values so they stay within [0, tt)
                for i in range(1, len(tt_boundaries) - 1):
                    tt_boundaries[i] = min(
                        tt_boundaries[i], tt - (len(tt_boundaries) - 1 - i)
                    )
                # Ensure monotonically increasing
                for i in range(1, len(tt_boundaries)):
                    if tt_boundaries[i] <= tt_boundaries[i - 1]:
                        tt_boundaries[i] = tt_boundaries[i - 1] + 1
                # Re-enforce last boundary (the loop above may have pushed it)
                tt_boundaries[-1] = tt
                logger.info(
                    f"  Converted pixel frame boundaries to tt boundaries: "
                    f"{prompt_video_boundaries} -> {tt_boundaries}  "
                    f"(patch_size_t={ps_t}, tt={tt})"
                )
                prompt_video_boundaries = tt_boundaries

            assert len(prompt_video_boundaries) == num_prompts + 1, (
                f"prompt_video_boundaries length ({len(prompt_video_boundaries)}) must be num_prompts+1 ({num_prompts + 1})"
            )
            assert prompt_video_boundaries[0] == 0, (
                f"prompt_video_boundaries must start at 0, got {prompt_video_boundaries[0]}"
            )
            assert prompt_video_boundaries[-1] == tt, (
                f"prompt_video_boundaries must end at tt={tt}, got {prompt_video_boundaries[-1]}"
            )
        else:
            assert tt % num_prompts == 0, (
                f"tt ({tt}) must be divisible by num_prompts ({num_prompts}). "
                f"Adjust --num-frames or provide --prompt-video-boundaries."
            )
            frames_per_prompt = tt // num_prompts
            prompt_video_boundaries = [
                i * frames_per_prompt for i in range(num_prompts + 1)
            ]

        logger.info(
            f"  prompt_video_boundaries (frame-level): {prompt_video_boundaries}"
        )

        logger.info("Encoding text prompts...")
        has_camera_module = (
            getattr(self.transformer, "camera_injection_mode", None) is not None
        )
        has_camera_condition = (
            camera_trajectory is not None
            and has_camera_module
            and (segment_has_camera is None or any(segment_has_camera))
        )
        do_text_cfg = guidance_scale > 1.0
        do_camera_cfg = camera_guidance_is_active(
            camera_guidance_scale,
            has_camera_condition,
        )
        guidance_branches = 3 if do_camera_cfg else (2 if do_text_cfg else 1)
        needs_guidance = guidance_branches > 1
        enc = self._encode_prompts(prompts)

        text_states_list = enc["all_states"]
        text_masks_list = enc["all_masks"]

        text_states_4d = torch.stack(
            [s.squeeze(0) for s in text_states_list], dim=0
        ).unsqueeze(0)
        text_masks_3d = torch.stack(
            [m.squeeze(0) for m in text_masks_list], dim=0
        ).unsqueeze(0)

        if needs_guidance:
            # Match training: each unconditional segment is encoded independently.
            neg_list_states = []
            neg_list_masks = []
            for _ in range(num_prompts):
                ns, nm = self._encode_single_prompt(
                    negative_prompt or "",
                    data_type="video",
                    is_uncond=True,
                )
                neg_list_states.append(ns.squeeze(0))
                neg_list_masks.append(nm.squeeze(0))
            neg_states_4d = torch.stack(neg_list_states, dim=0).unsqueeze(0)
            neg_masks_3d = torch.stack(neg_list_masks, dim=0).unsqueeze(0)
            if do_camera_cfg:
                text_states_4d = torch.cat(
                    [neg_states_4d, neg_states_4d, text_states_4d], dim=0
                )
                text_masks_3d = torch.cat(
                    [neg_masks_3d, neg_masks_3d, text_masks_3d], dim=0
                )
            else:
                text_states_4d = torch.cat([neg_states_4d, text_states_4d], dim=0)
                text_masks_3d = torch.cat([neg_masks_3d, text_masks_3d], dim=0)

        latents = torch.randn(
            [batch_size, out_channels, latent_frames, latent_h, latent_w],
            generator=generator,
            dtype=self.target_dtype,
        ).to(device=self.device)

        # Create zero vision_states placeholder
        vision_states = torch.zeros(
            batch_size,
            1,
            self.transformer.hidden_size,
            device=self.device,
            dtype=self.target_dtype,
        )
        if needs_guidance:
            vision_states = vision_states.repeat(guidance_branches, 1, 1)

        # ========== Camera control: prepare tensors ==========
        camera_kwargs = {}
        logger.info(
            f"  Camera injection mode (model): {getattr(self.transformer, 'camera_injection_mode', None)}"
        )
        if camera_trajectory is not None:
            if not has_camera_module:
                logger.warning(
                    f"  camera_trajectory='{camera_trajectory}' is set, but model has "
                    f"camera_injection_mode=None. Camera control will have NO effect! "
                    f"Make sure model.camera.use_camera_control=true in your config "
                    f"and the checkpoint contains camera module weights."
                )
            # Pose-normalization flags: prefer inference.* overrides, fall
            # back to root-level (which is where _flatten_cfg promotes the
            # dataset.* fields when an inference yaml inherits from a camera
            # training yaml). Default False to preserve legacy inference
            # behavior for yamls that don't inherit a camera training config.
            inf_cfg = getattr(self.args, "inference", None)

            def _cam_flag(name: str, default: bool = False) -> bool:
                if inf_cfg is not None and getattr(inf_cfg, name, None) is not None:
                    return bool(getattr(inf_cfg, name))
                return bool(getattr(self.args, name, default))

            norm_first = _cam_flag("normalize_relative_to_first_frame", False)
            norm_fw = _cam_flag("normalize_framewise", False)
            norm_disp = _cam_flag("normalize_displacement", False)
            norm_disp_per_video = _cam_flag("normalize_displacement_per_video", False)

            raw_scale = trajectory_translation_scale
            if raw_scale is None:
                raw_scale = getattr(inf_cfg, "trajectory_translation_scale", None)
            traj_scale = float(raw_scale) if raw_scale is not None else None
            raw_rotation_scale = trajectory_rotation_scale
            if raw_rotation_scale is None:
                raw_rotation_scale = getattr(inf_cfg, "trajectory_rotation_scale", 1.0)
            rotation_scale = (
                float(raw_rotation_scale) if raw_rotation_scale is not None else 1.0
            )
            if (
                rotation_scale != 1.0
                and getattr(self.transformer, "camera_injection_mode", None)
                != "extrinsics_attn"
            ):
                raise ValueError("Rotation gains currently require extrinsics_attn")
            logger.info(
                f"  Camera config: normalize_first={norm_first}, framewise={norm_fw}, "
                f"displacement={norm_disp}, "
                f"displacement_per_video={norm_disp_per_video}, "
                f"trajectory_translation_scale={traj_scale}, "
                f"trajectory_rotation_scale={rotation_scale}"
            )

            cam = prepare_camera_inputs(
                trajectory_source=camera_trajectory,
                num_frames=num_frames,
                device=self.device,
                dtype=self.target_dtype,
                normalize_relative_to_first_frame=norm_first,
                normalize_framewise=norm_fw,
                normalize_displacement=norm_disp,
                normalize_displacement_per_video=norm_disp_per_video,
                segment_boundaries_pixel=prompt_video_boundaries_pixel,
                translation_scale=traj_scale,
                rotation_scale=rotation_scale,
            )
            if "turn_camera_offsets" in cam:
                raise ValueError(
                    "Precomputed demo camera turns currently require seg_ar + extrinsics_attn"
                )
            # NPZ trajectories override num_frames to match trajectory length
            if cam["num_frames"] != num_frames:
                logger.info(
                    f"  Camera trajectory overrides num_frames: {num_frames} -> {cam['num_frames']}"
                )
                num_frames = cam["num_frames"]
                latent_frames = (num_frames - 1) // 4 + 1
                if reorg_token and latent_frames > 1:
                    latent_frames = (latent_frames + 1) // 2
                tt = latent_frames // patch_size[0]
                logger.info(f"  Updated latent dims: T={latent_frames}, tt={tt}")
                latents = torch.randn(
                    [batch_size, out_channels, latent_frames, latent_h, latent_w],
                    generator=generator,
                    dtype=self.target_dtype,
                ).to(device=self.device)
                # Recompute prompt_video_boundaries with new tt
                frames_per_prompt = tt // num_prompts
                prompt_video_boundaries = [
                    i * frames_per_prompt for i in range(num_prompts + 1)
                ]
                prompt_video_boundaries[-1] = tt
                logger.info(
                    f"  Updated prompt_video_boundaries: {prompt_video_boundaries}"
                )

            # Keep camera tensors batch-aligned with the active guidance branches.
            if needs_guidance:
                camera_kwargs = {
                    "norm_extrinsics": cam["norm_extrinsics"].repeat(
                        guidance_branches, 1, 1, 1
                    ),
                    "norm_intrinsics": cam["norm_intrinsics"].repeat(
                        guidance_branches, 1, 1, 1
                    ),
                    "action_labels": cam["action_labels"].repeat(guidance_branches, 1),
                    "action_onehot": cam["action_onehot"].repeat(
                        guidance_branches, 1, 1
                    ),
                }
            else:
                camera_kwargs = {k: v for k, v in cam.items() if k != "num_frames"}

            # ---- Per-segment camera-off gating (joint bench:
            #      segment_control == "prompt" → has_camera = False on that
            #      segment). Builds a [B, num_prompts] mask that ActionModule /
            #      residual_norm path expand to a token-level gate using
            #      prompt_video_boundaries.
            if segment_has_camera is not None:
                assert len(segment_has_camera) == num_prompts, (
                    f"segment_has_camera length {len(segment_has_camera)} "
                    f"!= num_prompts {num_prompts}"
                )
                cond_mask = [bool(x) for x in segment_has_camera]
            else:
                cond_mask = [True] * num_prompts

            if do_camera_cfg:
                all_camera_off = [False] * num_prompts
                mask_t = torch.tensor(
                    [all_camera_off, cond_mask, cond_mask],
                    dtype=torch.bool,
                    device=self.device,
                )
                camera_kwargs["has_camera_mask"] = mask_t
                logger.info(
                    f"  Separate camera CFG mask: base={all_camera_off}, "
                    f"camera=text={cond_mask} (shape={tuple(mask_t.shape)})"
                )
            elif do_text_cfg:
                mask_t = torch.tensor(
                    [cond_mask, cond_mask],
                    dtype=torch.bool,
                    device=self.device,
                )  # [2, N]: text CFG shares the same camera condition.
                camera_kwargs["has_camera_mask"] = mask_t
                logger.info(
                    f"  Text-only CFG camera mask: uncond=cond={cond_mask} "
                    f"(shape={tuple(mask_t.shape)})"
                )
            elif not all(cond_mask):
                mask_t = torch.tensor(
                    [cond_mask],
                    dtype=torch.bool,
                    device=self.device,
                )  # [1, N]
                camera_kwargs["has_camera_mask"] = mask_t
                logger.info(
                    f"  Per-segment camera mask: {cond_mask} "
                    f"(shape={tuple(mask_t.shape)})"
                )
            logger.info(f"  Camera control enabled: trajectory='{camera_trajectory}'")
        elif has_camera_module:
            logger.warning(
                f"  Model has camera_injection_mode='{self.transformer.camera_injection_mode}' "
                f"but no camera_trajectory was provided. Camera modules will receive None inputs."
            )

        solver_type = getattr(self.args.inference, "solver_type", "euler")
        if solver_type != "euler":
            raise ValueError(
                f"Unsupported inference solver: {solver_type!r}; expected 'euler'"
            )
        scheduler = EulerSolver(
            shift=flow_shift,
            reverse=True,
            solver="euler",
        )
        logger.info("  Solver: Euler")
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps, device=self.device
        )
        timesteps = scheduler.timesteps

        extra_kwargs = {
            "prompt_video_boundaries": prompt_video_boundaries,
        }
        # inference_chunk_size controls attention granularity:
        #   0: segment-level causal (segment j sees segments 0..j)
        #   1: full bidirectional (all frames see each other)
        #  >1: chunk-level (uniform groups of chunk_size latent frames)
        chunk_size = getattr(self.args, "inference_chunk_size", 1)
        if chunk_size is None:
            chunk_size = 1
        extra_kwargs["chunk_size"] = chunk_size

        # ---- Previous-segment last-frame video restriction (one_shot) ----
        # When enabled (and segment-level causal, chunk_size==0), segment j's
        # video AND text queries attend their own segment's video in full plus
        # only the FINAL latent frame of each previous segment (video K-ranges
        # only; text->text unchanged). Mirrors the training flag
        # prev_seg_last_frame_only.
        if chunk_size == 0 and self._prev_seg_last_frame_only():
            extra_kwargs["prev_seg_last_frame_only"] = True
            logger.info("  prev_seg_last_frame_only ON (video+text->video restricted)")

        # ---- Cross- and intra-segment action attention masking (one_shot) ----
        # Cross-segment text-tail masking is independent from the optional
        # intra-segment per-action partition:
        #   - cross enabled: other segments expose their text head only
        #   - intra enabled: action-k video/text sees head + action-k text only
        #   - intra disabled: a segment sees its own complete text block
        # Intra masking remains segment-level only (chunk_size == 0).
        # Coordinates: text offsets are segment-relative token coords; video
        # windows are ABSOLUTE video-token coords (seg-relative latent + segment
        # start, x tokens_per_frame; tt == latent since patch_size_t == 1).
        cross_segment_text_head_only = self._cross_segment_text_head_only_enabled()
        intra_seg_active = chunk_size == 0 and self._intra_seg_enabled()
        extra_kwargs["cross_segment_text_head_only"] = cross_segment_text_head_only
        if cross_segment_text_head_only or intra_seg_active:
            tpf = int(th * tw)
            pvb_tt = prompt_video_boundaries  # tt (== latent) units, len num_prompts+1
            pos_masks = text_masks_3d[-1]  # positive-sample per-block valid text
            txt_action_start_pb: List[int] = []
            txt_action_offsets_pb: List[Optional[List[int]]] = []
            vid_action_boundaries_pb: List[Optional[List[List[int]]]] = []
            for j in range(num_prompts):
                seg_prompt = prompts[j] if j < len(prompts) else ""
                a_start = (
                    self._compute_action_start(seg_prompt)
                    if cross_segment_text_head_only
                    else -1
                )
                offs = None
                vwin_abs = None
                if intra_seg_active:
                    offs = self._compute_action_text_offsets(seg_prompt)
                    if offs is not None and j < pos_masks.shape[0]:
                        valid_j = int(pos_masks[j].sum().item())
                        offs = [min(int(o), valid_j) for o in offs]
                    seg_ls, seg_le = int(pvb_tt[j]), int(pvb_tt[j + 1])
                    rel = None
                    try:
                        rel = self._compute_action_video_windows(
                            seg_prompt,
                            seg_ls,
                            seg_le,
                        )
                    except ValueError as _e:
                        logger.warning(
                            f"  [seg {j}] intra-seg video windows disabled: {_e}"
                        )
                    if rel is not None:
                        vwin_abs = [
                            [
                                (int(vs) + seg_ls) * tpf,
                                (int(ve) + seg_ls) * tpf,
                                int(aidx),
                            ]
                            for vs, ve, aidx in rel
                        ]
                txt_action_start_pb.append(int(a_start))
                txt_action_offsets_pb.append(offs)
                vid_action_boundaries_pb.append(vwin_abs)
            if cross_segment_text_head_only:
                extra_kwargs["txt_action_start_per_block"] = txt_action_start_pb
            if intra_seg_active:
                extra_kwargs["txt_action_offsets_per_block"] = txt_action_offsets_pb
                extra_kwargs["vid_action_boundaries_per_block"] = (
                    vid_action_boundaries_pb
                )
            logger.info(
                f"  action text masks: cross_segment_head_only="
                f"{cross_segment_text_head_only}, intra_segment={intra_seg_active}, "
                f"action_starts={txt_action_start_pb}, offsets={txt_action_offsets_pb}, "
                f"vid_windows={vid_action_boundaries_pb}"
            )

        logger.info(f"  inference_chunk_size: {chunk_size}")
        logger.info(f"  text_states shape: {text_states_4d.shape}")
        logger.info(f"  text_masks shape: {text_masks_3d.shape}")
        logger.info(
            f"  extra_kwargs: prompt_video_boundaries={prompt_video_boundaries}, chunk_size={chunk_size}"
        )
        logger.info(f"  latents shape: {latents.shape}")

        logger.info(f"Starting denoising loop ({num_inference_steps} steps)...")
        start_time = time.time()

        with self._dit_denoise_residency():
            for i, t in enumerate(timesteps):
                t_expand = t.expand(batch_size).to(self.device)

                latent_input = latents

                if latent_input.shape[1] < in_channels:
                    pad = torch.zeros(
                        [
                            *latent_input.shape[:1],
                            in_channels - latent_input.shape[1],
                            *latent_input.shape[2:],
                        ],
                        device=latent_input.device,
                        dtype=latent_input.dtype,
                    )
                    latent_input = torch.cat([latent_input, pad], dim=1)

                if needs_guidance:
                    latent_input = latent_input.repeat(guidance_branches, 1, 1, 1, 1)
                    t_expand = t_expand.repeat(guidance_branches)

                with torch.autocast(device_type="cuda", dtype=self.target_dtype):
                    output = self.transformer(
                        hidden_states=latent_input,
                        timestep=t_expand,
                        text_states=text_states_4d,
                        encoder_attention_mask=text_masks_3d,
                        vision_states=vision_states,
                        guidance=None,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                        **camera_kwargs,
                    )
                    noise_pred = output[0]

                if needs_guidance:
                    if cfg_schedule and cfg_schedule != "constant":
                        guidance_scale_step = get_cfg_scale_for_step(
                            cfg_schedule,
                            i,
                            num_inference_steps,
                            guidance_scale,
                            cfg_skip_steps=cfg_skip_steps,
                            cfg_scale_min=cfg_scale_min,
                            cfg_scale_max=cfg_scale_max,
                            cfg_scale_max_step=cfg_scale_max_step,
                        )
                    elif use_dynamic_cfg_scale:
                        guidance_scale_step = get_dynamic_guidance_scale(
                            i,
                            num_inference_steps,
                            cfg_scale_min,
                            cfg_scale_max,
                            cfg_scale_max_step,
                        )
                    else:
                        guidance_scale_step = guidance_scale
                    if do_camera_cfg:
                        noise_pred_base, noise_pred_camera, noise_pred_text = (
                            noise_pred.chunk(3)
                        )
                        noise_pred = combine_camera_text_cfg(
                            noise_pred_base,
                            noise_pred_camera,
                            noise_pred_text,
                            camera_guidance_scale,
                            guidance_scale_step,
                        )
                    else:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale_step * (
                            noise_pred_cond - noise_pred_uncond
                        )

                latents = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents,
                    return_dict=False,
                )[0].to(self.target_dtype)

                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(f"  Step {i + 1}/{num_inference_steps}")

        elapsed = time.time() - start_time
        logger.info(f"Denoising complete in {elapsed:.2f}s")

        logger.info("Decoding latents to video...")
        return self._decode_latents(latents, num_frames)
