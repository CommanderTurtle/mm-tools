"""Deterministic image corruptions for robustness variants of the benchmarks (e.g. 3dpw_dark, emdb1_dark).

Low light follows the unprocess -> under-expose -> sensor noise -> reprocess model of synthetic low-light
benchmarks (Brooks et al. 2019 "Unprocessing images for learned raw denoising"; LOL / ExLPose-style synthesis):
sRGB is linearised, exposure is cut 10-50x, Poisson-Gaussian noise is added in linear space with per-channel gains
(white balance), and the result is gamma-encoded and quantised to 8 bits. This differs from the training
augmentation (train/train_fab.py low_light: sRGB gain / gamma / additive noise), so the benchmark does not reward
fitting one corruption generator. Parameters come from the sample id, so every method sees the same image.
"""

from __future__ import annotations

import zlib

import numpy as np


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def low_light(image: np.ndarray, sample_id: str) -> np.ndarray:
    """uint8 RGB -> uint8 RGB low-light version, deterministic in sample_id."""
    rng = np.random.default_rng(zlib.crc32(sample_id.encode()))
    lin = _srgb_to_linear(image.astype(np.float32) / 255.0)
    wb = rng.uniform(0.75, 1.25, 3).astype(np.float32)  # white-balance gains (undone after the noise)
    raw = lin / wb / rng.uniform(10.0, 50.0)
    shot, read = rng.uniform(0.002, 0.012), rng.uniform(0.0005, 0.003)
    raw = raw + rng.normal(0.0, 1.0, raw.shape).astype(np.float32) * np.sqrt(shot * np.maximum(raw, 0) + read ** 2)
    out = _linear_to_srgb(np.clip(raw * wb, 0, 1))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


LOW_LIGHT_LEVELS = {  # FAB-Hard synthetic low light: exposure cut, shot / read noise ranges
    "moderate": ((8.0, 15.0), (0.002, 0.006), (0.0005, 0.0015)),
    "hard": ((25.0, 50.0), (0.004, 0.010), (0.001, 0.003)),
    "extreme": ((80.0, 150.0), (0.008, 0.016), (0.002, 0.005)),
}


def low_light_level(image: np.ndarray, sample_id: str, level: str) -> np.ndarray:
    """low_light with the exposure / noise ranges of one LOW_LIGHT_LEVELS difficulty, deterministic in sample_id."""
    (e0, e1), (s0, s1), (r0, r1) = LOW_LIGHT_LEVELS[level]
    rng = np.random.default_rng(zlib.crc32(f"{level}/{sample_id}".encode()))
    lin = _srgb_to_linear(image.astype(np.float32) / 255.0)
    wb = rng.uniform(0.75, 1.25, 3).astype(np.float32)
    raw = lin / wb / rng.uniform(e0, e1)
    shot, read = rng.uniform(s0, s1), rng.uniform(r0, r1)
    raw = raw + rng.normal(0.0, 1.0, raw.shape).astype(np.float32) * np.sqrt(shot * np.maximum(raw, 0) + read ** 2)
    out = _linear_to_srgb(np.clip(raw * wb, 0, 1))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def motion_blur(image: np.ndarray, sample_id: str) -> np.ndarray:
    """Linear motion blur (fast motion / slow shutter): a line kernel of 1.5-3.5 % of the image's longer side at a random
    angle, deterministic in sample_id."""
    import cv2

    rng = np.random.default_rng(zlib.crc32(f"blur/{sample_id}".encode()))
    n = max(3, int(round(rng.uniform(0.015, 0.035) * max(image.shape[:2]))) | 1)
    k = np.zeros((n, n), np.float32)
    k[n // 2] = 1.0
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1.0), (n, n))
    return cv2.filter2D(image, -1, k / max(k.sum(), 1e-6))


def low_resolution(image: np.ndarray, sample_id: str) -> np.ndarray:
    """Low resolution (distant / surveillance cameras): downscaled so the longer side is 240-360 px (area), then upscaled
    back (bilinear) so the geometry and intrinsics are unchanged, deterministic in sample_id."""
    import cv2

    rng = np.random.default_rng(zlib.crc32(f"lowres/{sample_id}".encode()))
    H, W = image.shape[:2]
    s = rng.uniform(240, 360) / max(H, W)
    small = cv2.resize(image, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {"dark": low_light, "blur": motion_blur, "lowres": low_resolution}
