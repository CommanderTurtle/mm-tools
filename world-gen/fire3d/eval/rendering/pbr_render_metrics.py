"""Shared image metrics for canonical PBR render comparisons.

The benchmark intentionally aggregates in two levels: views are averaged into
one object score, then object scores are averaged into sample/subset scores.
This prevents objects with more successfully rendered views from receiving
more weight.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


METRIC_KEYS = (
    "l1",
    "rmse",
    "full_psnr",
    "full_ssim",
    "crop_psnr",
    "crop_ssim",
    "intersection_psnr",
    "gt_foreground_psnr",
    "mask_iou",
    "gt_coverage",
    "pred_coverage",
)


def load_rgba_composited(
    path: str | Path,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one RGBA render, composite it deterministically, and return mask."""
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    background = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = rgba[..., :3] * alpha + background * (1.0 - alpha)
    return rgb, alpha[..., 0] > float(alpha_threshold)


def _bbox(mask: np.ndarray, padding: int = 4) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    return (
        slice(max(0, int(ys.min()) - padding), min(mask.shape[0], int(ys.max()) + padding + 1)),
        slice(max(0, int(xs.min()) - padding), min(mask.shape[1], int(xs.max()) + padding + 1)),
    )


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def _ssim(pred: np.ndarray, target: np.ndarray) -> float:
    # skimage defaults to a 7x7 window, while a benchmark crop can legitimately
    # be smaller for a distant object. Use the largest supported odd window.
    # SSIM is undefined below 3 pixels, so the finite-only aggregation ignores
    # that view instead of aborting the complete benchmark.
    spatial_extent = min(int(pred.shape[0]), int(pred.shape[1]))
    if spatial_extent < 3:
        return math.nan
    win_size = min(7, spatial_extent if spatial_extent % 2 else spatial_extent - 1)
    return float(
        structural_similarity(
            target,
            pred,
            data_range=1.0,
            channel_axis=-1,
            win_size=win_size,
        )
    )


def image_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, float]:
    """Compute the frozen PBR benchmark metrics for one matched render view."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected matching RGB images, got pred={pred.shape}, target={target.shape}")
    if pred_mask.shape != target_mask.shape or pred_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"Expected matching image masks, got pred={pred_mask.shape}, target={target_mask.shape}"
        )
    difference = pred.astype(np.float64) - target.astype(np.float64)
    union = pred_mask | target_mask
    intersection = pred_mask & target_mask
    crop = _bbox(union)
    intersection_mse = float(np.mean(difference[intersection] ** 2)) if intersection.any() else math.nan
    target_foreground_mse = (
        float(np.mean(difference[target_mask] ** 2)) if target_mask.any() else math.nan
    )
    return {
        "l1": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "full_psnr": _psnr(pred, target),
        "full_ssim": _ssim(pred, target),
        "crop_psnr": _psnr(pred[crop], target[crop]),
        "crop_ssim": _ssim(pred[crop], target[crop]),
        "intersection_psnr": (
            -10.0 * math.log10(max(intersection_mse, 1e-12))
            if math.isfinite(intersection_mse)
            else math.nan
        ),
        "gt_foreground_psnr": (
            -10.0 * math.log10(max(target_foreground_mse, 1e-12))
            if math.isfinite(target_foreground_mse)
            else math.nan
        ),
        "mask_iou": float(intersection.sum() / max(int(union.sum()), 1)),
        "gt_coverage": float(target_mask.mean()),
        "pred_coverage": float(pred_mask.mean()),
    }


def finite_mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def aggregate_metric_rows(
    rows: Iterable[dict[str, Any]],
    metric_keys: Iterable[str] = METRIC_KEYS,
) -> dict[str, float | None]:
    rows = list(rows)
    return {key: finite_mean(row.get(key) for row in rows) for key in metric_keys}


def aggregate_objects_two_level(
    object_rows: list[dict[str, Any]],
    *,
    sample_key: str = "sample_id",
    metric_key: str = "metrics",
) -> dict[str, Any]:
    """Return object-weighted and sample-weighted scores.

    Each object row must already contain the mean across that object's views.
    The sample-weighted result first averages objects within each sample, then
    averages sample means. This is the protocol used for the 25+15 report.
    """
    object_weighted = {
        key: finite_mean(row[metric_key].get(key) for row in object_rows) for key in METRIC_KEYS
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in object_rows:
        grouped.setdefault(str(row[sample_key]), []).append(row)
    sample_rows = [
        {
            "sample_id": sample_id,
            "num_objects": len(rows),
            "metrics": {
                key: finite_mean(row[metric_key].get(key) for row in rows) for key in METRIC_KEYS
            },
        }
        for sample_id, rows in sorted(grouped.items())
    ]
    sample_weighted = {
        key: finite_mean(row["metrics"].get(key) for row in sample_rows) for key in METRIC_KEYS
    }
    return {
        "num_objects": len(object_rows),
        "num_samples": len(sample_rows),
        "object_weighted": object_weighted,
        "sample_weighted": sample_weighted,
        "samples": sample_rows,
    }
