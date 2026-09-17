"""Camera-consistent augmentation for isolated object RGB/depth renders.

The transform intentionally owns RGB, metric depth, foreground mask, and
intrinsics together.  Cropping RGB alone would misalign DINO cells and the 3D
points reconstructed from depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectRenderAugmentResult:
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray
    metadata: dict[str, Any]


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list/tuple, got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError(f"Invalid {name}: {value!r}")
    return lo, hi


def _validate_inputs(rgb, depth, mask, intrinsics) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth shape {depth.shape} does not match rgb {rgb.shape[:2]}")
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match rgb {rgb.shape[:2]}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3,3], got {intrinsics.shape}")
    if not np.isfinite(rgb).all() or not np.isfinite(intrinsics).all():
        raise ValueError("rgb and intrinsics must be finite")


def _identity_result(rgb, depth, mask, intrinsics, reason: str) -> ObjectRenderAugmentResult:
    return ObjectRenderAugmentResult(
        rgb=rgb,
        depth=depth,
        mask=mask,
        intrinsics=intrinsics,
        metadata={"applied": False, "reason": reason},
    )


def _sample_crop(mask: np.ndarray, rng: np.random.Generator, config: Mapping[str, Any]):
    height, width = mask.shape
    crop_scale = _pair(config.get("crop_scale", (0.6, 1.0)), "crop_scale")
    aspect_ratio = _pair(config.get("aspect_ratio", (0.8, 1.25)), "aspect_ratio")
    if crop_scale[0] <= 0 or crop_scale[1] > 1:
        raise ValueError("crop_scale must lie in (0,1]")
    if aspect_ratio[0] <= 0:
        raise ValueError("aspect_ratio must be positive")

    foreground = mask.astype(bool, copy=False)
    foreground_count = int(foreground.sum())
    if foreground_count == 0:
        return (0, 0, width, height, 0, True)

    ys, xs = np.nonzero(foreground)
    object_cx = 0.5 * (float(xs.min()) + float(xs.max()))
    object_cy = 0.5 * (float(ys.min()) + float(ys.max()))
    center_jitter = float(config.get("center_jitter", 0.2))
    min_retained = float(config.get("min_foreground_retained", 0.65))
    max_attempts = int(config.get("max_crop_attempts", 10))
    if not 0 <= min_retained <= 1:
        raise ValueError("min_foreground_retained must be in [0,1]")
    if max_attempts < 1:
        raise ValueError("max_crop_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        scale = float(rng.uniform(*crop_scale))
        log_aspect = float(rng.uniform(np.log(aspect_ratio[0]), np.log(aspect_ratio[1])))
        aspect = float(np.exp(log_aspect))
        crop_w = int(np.clip(round(width * scale * np.sqrt(aspect)), 1, width))
        crop_h = int(np.clip(round(height * scale / np.sqrt(aspect)), 1, height))

        cx = object_cx + float(rng.uniform(-center_jitter, center_jitter)) * crop_w
        cy = object_cy + float(rng.uniform(-center_jitter, center_jitter)) * crop_h
        x0 = int(np.clip(round(cx - crop_w * 0.5), 0, width - crop_w))
        y0 = int(np.clip(round(cy - crop_h * 0.5), 0, height - crop_h))
        retained = int(foreground[y0 : y0 + crop_h, x0 : x0 + crop_w].sum())
        if retained / foreground_count >= min_retained:
            return (x0, y0, crop_w, crop_h, attempt, False)

    return (0, 0, width, height, max_attempts, True)


def _random_background(
    height: int,
    width: int,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, str, float]:
    keep_white = float(config.get("keep_white_background_probability", 0.10))
    solid = float(config.get("solid_background_probability", 0.45))
    noisy = float(config.get("noisy_background_probability", 0.45))
    probs = np.asarray([keep_white, solid, noisy], dtype=np.float64)
    if (probs < 0).any() or float(probs.sum()) <= 0:
        raise ValueError("Background probabilities must be non-negative with positive sum")
    probs /= probs.sum()
    mode = str(rng.choice(np.asarray(["white", "solid", "noisy"]), p=probs))

    if mode == "white":
        return np.ones((height, width, 3), dtype=np.float32), mode, 0.0

    base = rng.uniform(0.0, 1.0, size=(1, 1, 3)).astype(np.float32)
    background = np.broadcast_to(base, (height, width, 3)).copy()
    if mode == "solid":
        return background, mode, 0.0

    noise_std = _pair(config.get("background_noise_std", (0.0, 0.08)), "background_noise_std")
    if noise_std[0] < 0:
        raise ValueError("background_noise_std must be non-negative")
    std = float(rng.uniform(*noise_std))
    if std > 0:
        background += rng.normal(0.0, std, size=background.shape).astype(np.float32)
        if rng.random() < float(config.get("low_frequency_noise_probability", 0.5)):
            grid_h = max(2, min(8, height))
            grid_w = max(2, min(8, width))
            low = rng.normal(0.0, std, size=(grid_h, grid_w, 3)).astype(np.float32)
            background += cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(background, 0.0, 1.0), mode, std


def augment_object_render(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, Any] | None,
) -> ObjectRenderAugmentResult:
    """Apply a mask-aware crop/resize and replace only background RGB pixels.

    With the augmentation disabled or not sampled, the original array objects
    are returned.  This makes the disabled path free of resampling or numeric
    changes.
    """

    _validate_inputs(rgb, depth, mask, intrinsics)
    config = {} if config is None else config
    if not bool(config.get("enabled", False)):
        return _identity_result(rgb, depth, mask, intrinsics, "disabled")
    probability = float(config.get("probability", 1.0))
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    if rng.random() >= probability:
        return _identity_result(rgb, depth, mask, intrinsics, "probability")

    in_h, in_w = rgb.shape[:2]
    out_h = int(config.get("output_height", in_h))
    out_w = int(config.get("output_width", in_w))
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output dimensions must be positive")

    x0, y0, crop_w, crop_h, attempts, fallback = _sample_crop(mask, rng, config)
    slices = np.s_[y0 : y0 + crop_h, x0 : x0 + crop_w]
    rgb_crop = np.ascontiguousarray(rgb[slices])
    depth_crop = np.ascontiguousarray(depth[slices])
    mask_crop = np.ascontiguousarray(mask[slices])

    rgb_resized = cv2.resize(rgb_crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(depth_crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.resize(
        mask_crop.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    if rgb_resized.ndim == 2:
        rgb_resized = rgb_resized[..., None]

    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)
    image_transform = np.asarray(
        [[sx, 0.0, -sx * x0], [0.0, sy, -sy * y0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsics_resized = image_transform @ intrinsics.astype(np.float32, copy=False)

    background, background_mode, noise_std = _random_background(out_h, out_w, rng, config)
    rgb_augmented = rgb_resized.astype(np.float32, copy=True)
    rgb_augmented[~mask_resized] = background[~mask_resized]
    rgb_augmented = np.clip(rgb_augmented, 0.0, 1.0)

    metadata = {
        "applied": True,
        "crop_xywh": [int(x0), int(y0), int(crop_w), int(crop_h)],
        "input_hw": [int(in_h), int(in_w)],
        "output_hw": [int(out_h), int(out_w)],
        "resize_xy": [sx, sy],
        "crop_attempts": int(attempts),
        "identity_crop_fallback": bool(fallback),
        "foreground_retained": float(mask_crop.astype(bool).sum()) / max(float(mask.astype(bool).sum()), 1.0),
        "background_mode": background_mode,
        "background_noise_std": float(noise_std),
        "image_transform": image_transform.tolist(),
    }
    return ObjectRenderAugmentResult(
        rgb=rgb_augmented,
        depth=depth_resized.astype(np.float32, copy=False),
        mask=mask_resized,
        intrinsics=intrinsics_resized.astype(np.float32, copy=False),
        metadata=metadata,
    )
