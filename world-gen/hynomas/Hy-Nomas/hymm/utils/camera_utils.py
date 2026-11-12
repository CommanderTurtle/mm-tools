"""
Utilities for camera control training and inference.
"""

import random
from typing import List, Sequence

import torch

# Camera-specific module name prefixes (params in these modules remain trainable when model.dit.freeze=True)
_CAMERA_MODULE_PREFIXES = (
    "control_adapter",
    "patch_embedding_camctrl",
    "c2ws_hidden_states_layer1",
    "c2ws_hidden_states_layer2",
    "camera_embed",
    "action_in",
    "action_modules",
    "cam_scale_layer",
    "cam_shift_layer",
    "extrinsics_embedding_camctrl",
)


def is_camera_param(name: str) -> bool:
    """Return True iff parameter name belongs to a camera injection module."""
    return any(prefix in name for prefix in _CAMERA_MODULE_PREFIXES)


def freeze_base_dit_for_camera(model, logger=None):
    """
    Freeze base DiT parameters, keeping only camera injection modules trainable.
    Triggered by model.camera.freeze_dit_for_camera=True.
    Camera modules: control_adapter, patch_embedding_camctrl, c2ws layers, camera_embed,
    action_in, cam_scale/cam_shift (for shift variant), etc.
    """
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if is_camera_param(name):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    if logger:
        logger.info(
            f"camera.freeze_dit_for_camera: frozen {frozen_count / 1e6:.2f}M params, "
            f"trainable {trainable_count / 1e6:.2f}M params (camera modules only)"
        )


def ensure_camera_params_fp32(
    model, *, role: str, logger=None, force: bool = False
) -> None:
    """Force and verify fp32 *master parameter storage* for camera modules.

    Camera modules are part of the parent DiT's root FSDP2 unit (they are not
    separately sharded).  FSDP mixed precision may still use a lower-precision
    all-gather/forward view, but the persistent local/master parameters tracked
    by the optimizer or frozen model must remain fp32 for every role.
    """
    camera_params = [
        (name, param)
        for name, param in model.named_parameters()
        if is_camera_param(name)
    ]
    if not camera_params:
        return

    if force:
        with torch.no_grad():
            for _name, param in camera_params:
                if param.dtype != torch.float32:
                    param.data = param.data.float()

    bad = [
        (name, str(param.dtype))
        for name, param in camera_params
        if param.dtype != torch.float32
    ]
    if bad:
        preview = ", ".join(f"{name}:{dtype}" for name, dtype in bad[:20])
        raise RuntimeError(
            f"{role} camera master parameters must all be fp32; "
            f"found {len(bad)} non-fp32 parameter(s): {preview}"
        )
    if logger is not None:
        logger.info(
            f"[camera] {role}: verified {len(camera_params)} camera parameter "
            "tensors use fp32 master storage"
        )


def assert_camera_local_params_fp32(model, *, role: str, logger=None) -> None:
    """Verify fp32 local/master shards after FSDP2 wrapping."""
    bad = []
    count = 0
    for name, param in model.named_parameters():
        if not is_camera_param(name):
            continue
        count += 1
        local = param._local_tensor if hasattr(param, "_local_tensor") else param
        if local.dtype != torch.float32:
            bad.append((name, str(local.dtype)))
    if bad:
        preview = ", ".join(f"{name}:{dtype}" for name, dtype in bad[:20])
        raise RuntimeError(
            f"{role} camera FSDP2 local/master shards must all be fp32; "
            f"found {len(bad)} non-fp32 parameter(s): {preview}"
        )
    if count and logger is not None:
        logger.info(
            f"[camera] {role}: verified {count} FSDP2 local/master camera "
            "parameter shards remain fp32"
        )


def sample_segment_camera_keep(
    *,
    num_segments: int,
    dropout_p: float,
    min_on: int,
    global_step: int,
    sample_idx: int,
) -> List[bool]:
    """Decide which segments stay camera-ON for one multi-segment camera sample.

    Ported verbatim from ``unified.py::_sample_segment_camera_keep`` so the
    ddmd / ddmd_adv / pcm camera path matches the authoritative per-segment
    dropout behavior. Returns a length-``num_segments`` bool list (True = keep
    camera, False = drop / treat as prompt-controlled). Decisions use a
    deterministic per-(global_step, sample_idx) RNG so every SP rank in a group
    derives the identical mask.

    Each segment is dropped independently with prob ``dropout_p``, then two hard
    constraints are enforced: at least one dropped AND at least ``min_on`` kept.

    IMPORTANT: callers MUST only invoke this when ``dropout_p > 0`` so the RNG
    stream is never touched in the default (p=0) path. The guard below is a
    second line of defense, not a substitute for the caller-side gate.
    """
    if num_segments <= 1 or dropout_p <= 0.0:
        return [True] * max(num_segments, 1)
    min_on = max(1, int(min_on))
    if num_segments <= min_on:
        return [True] * num_segments

    rng = random.Random()
    rng.seed(
        (int(global_step) * 1_000_003 + int(sample_idx) * 9_176 + 12_345) & 0x7FFFFFFF
    )

    keep = [rng.random() >= dropout_p for _ in range(num_segments)]

    if all(keep):
        keep[rng.randrange(num_segments)] = False

    n_on = sum(keep)
    if n_on < min_on:
        off_idx = [j for j, k in enumerate(keep) if not k]
        for j in rng.sample(off_idx, min_on - n_on):
            keep[j] = True

    return keep


def freeze_camera_modules(model, logger=None):
    """Freeze only student/generator camera-control parameters.

    This is the inverse of ``freeze_base_dit_for_camera``: base DiT parameter
    flags are left untouched, while every parameter identified by
    ``is_camera_param`` is forced to ``requires_grad=False``. Call this before
    FSDP wrapping and optimizer construction so all optimizer implementations
    naturally exclude the frozen camera parameters.
    """
    frozen_count = 0
    frozen_tensors = 0
    for name, param in model.named_parameters():
        if not is_camera_param(name):
            continue
        param.requires_grad = False
        frozen_count += param.numel()
        frozen_tensors += 1
    if logger:
        logger.info(
            f"camera.freeze_camera_modules=True: froze "
            f"{frozen_count / 1e6:.3f}M student camera params "
            f"across {frozen_tensors} tensors; base DiT flags unchanged"
        )
        if frozen_tensors == 0:
            logger.warning(
                "camera.freeze_camera_modules=True but no camera parameters "
                "were found (check use_camera_control / injection_mode)."
            )


def print_camera_param_summary(model, logger, max_listed: int = 20) -> None:
    """Log a one-shot summary of trainable camera-module params at startup.

    Prints total numel + tensor count, plus the first ``max_listed`` parameter
    names so it's obvious which modules are wired in (or missing).
    """
    cam_params = [
        (n, p.numel())
        for n, p in model.named_parameters()
        if p.requires_grad and is_camera_param(n)
    ]
    total_cam = sum(numel for _, numel in cam_params)
    logger.info(
        f"[camera] trainable camera params: {total_cam / 1e6:.3f}M "
        f"across {len(cam_params)} tensors"
    )
    if total_cam == 0:
        # freeze_camera_modules=True freezes every camera tensor on purpose, so
        # distinguish "intentionally frozen" from "never initialized".
        if any(is_camera_param(n) for n, _ in model.named_parameters()):
            logger.info("[camera] all camera parameters are intentionally frozen.")
        else:
            logger.warning(
                "[camera] no camera parameters detected — camera module "
                "may not be initialized (check model.camera.use_camera_control / "
                "injection_mode)."
            )
        return
    for name, numel in cam_params[:max_listed]:
        logger.info(f"  {name}: {numel}")
    if len(cam_params) > max_listed:
        logger.info(f"  ... ({len(cam_params) - max_listed} more tensors)")


def canonical_camera_frame_count(
    num_video_frames: int,
    temporal_ratio: int = 4,
) -> int:
    """Return the training-aligned camera length for a video frame count."""
    num_video_frames = int(num_video_frames)
    temporal_ratio = int(temporal_ratio)
    if num_video_frames <= 0:
        raise ValueError(f"num_video_frames must be positive, got {num_video_frames}")
    if temporal_ratio <= 0:
        raise ValueError(f"temporal_ratio must be positive, got {temporal_ratio}")

    latent_frames = (num_video_frames - 1) // temporal_ratio + 1
    return (latent_frames - 1) * temporal_ratio + 1


def pixel_to_camera_boundaries(
    pixel_boundaries: Sequence[int],
    num_video_frames: int,
    temporal_ratio: int = 4,
) -> List[int]:
    """Map video-frame boundaries to training-aligned camera-frame boundaries."""
    if len(pixel_boundaries) < 2:
        raise ValueError("pixel_boundaries must contain at least start and end")

    target_camera_frames = canonical_camera_frame_count(
        num_video_frames, temporal_ratio
    )
    offset = temporal_ratio - 1
    camera_boundaries = []
    for boundary in pixel_boundaries:
        latent_boundary = max(0, int(boundary) // temporal_ratio)
        camera_boundary = (
            max(0, latent_boundary * temporal_ratio - offset)
            if latent_boundary > 0
            else 0
        )
        camera_boundaries.append(min(camera_boundary, target_camera_frames))

    camera_boundaries[0] = 0
    camera_boundaries[-1] = target_camera_frames
    for i in range(1, len(camera_boundaries)):
        if camera_boundaries[i] <= camera_boundaries[i - 1]:
            camera_boundaries[i] = min(
                camera_boundaries[i - 1] + 1, target_camera_frames
            )

    if camera_boundaries[-1] != target_camera_frames or any(
        camera_boundaries[i] >= camera_boundaries[i + 1]
        for i in range(len(camera_boundaries) - 1)
    ):
        return [0, target_camera_frames]
    return camera_boundaries
