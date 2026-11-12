"""Image matting metrics: MAD, MSE, Grad, Conn and SAD.

``compute_matting_metrics`` compares a predicted alpha with a uint8 ground-truth
alpha, using the Adobe definitions (Gaussian-derivative gradient with sigma 1.4,
and connectivity with a threshold step of 0.1). ``inference_matting.py`` saves
these values for each sample. This script averages them for each dataset,
reports MAD and MSE multiplied by 1000, and computes ``overall`` and any
``--group`` as averages weighted by the number of samples.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

KEYS = ("mad", "mse", "grad", "conn", "sad")


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _gauss_gradient(image: np.ndarray, sigma: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    half = int(np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * 1e-2))))
    coords = np.arange(-half, half + 1, dtype=np.float64)
    kernel = _gauss(coords[:, None], sigma) * (-coords[None, :] * _gauss(coords[None, :], sigma) / sigma**2)
    kernel /= np.sqrt(np.square(np.abs(kernel)).sum())
    gx = cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(image, cv2.CV_64F, kernel.T, borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1


def compute_matting_metrics(prediction: np.ndarray, alpha_u8: np.ndarray) -> dict[str, float]:
    target = alpha_u8.astype(np.float64) / 255.0
    pred = np.clip(prediction.astype(np.float64), 0.0, 1.0)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/alpha shape mismatch: {pred.shape} vs {target.shape}")
    difference = pred - target

    target_x, target_y = _gauss_gradient(target)
    pred_x, pred_y = _gauss_gradient(pred)
    grad = np.square(np.hypot(pred_x, pred_y) - np.hypot(target_x, target_y)).sum()

    level = np.full(target.shape, -1.0)
    for threshold in np.arange(0.1, 1.01, 0.1):
        omega = _largest_component((pred >= threshold) & (target >= threshold))
        level[(level < 0) & ~omega] = threshold - 0.1
    level[level < 0] = 1.0
    pred_delta, target_delta = pred - level, target - level
    pred_phi = 1.0 - pred_delta * (pred_delta >= 0.15)
    target_phi = 1.0 - target_delta * (target_delta >= 0.15)

    absolute = np.abs(difference)
    return {
        "mad": float(absolute.mean()),
        "mse": float(np.square(difference).mean()),
        "grad": float(grad / 1000.0),
        "conn": float(np.abs(pred_phi - target_phi).sum() / 1000.0),
        "sad": float(absolute.sum() / 1000.0),
    }


def _table(means: dict[str, float], count: int) -> dict:
    return {
        "count": count,
        "MAD": means["mad"] * 1000.0,
        "MSE": means["mse"] * 1000.0,
        "Grad": means["grad"],
        "Conn": means["conn"],
        "SAD": means["sad"],
    }


def _weighted(datasets: dict[str, dict], names: list[str]) -> dict:
    total = sum(datasets[name]["count"] for name in names)
    result = {"count": total}
    for key in ("MAD", "MSE", "Grad", "Conn", "SAD"):
        result[key] = sum(datasets[name][key] * datasets[name]["count"] for name in names) / total
    return result


def evaluate(rows: list[dict], groups: dict[str, list[str]] | None = None) -> dict:
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    datasets = {}
    for name, items in by_dataset.items():
        means = {key: float(np.mean([row["metrics"][key] for row in items])) for key in KEYS}
        datasets[name] = {
            **_table(means, len(items)),
            "valid_predictions": sum(bool(row.get("valid_prediction")) for row in items),
        }
    result = {"datasets": datasets, "overall": _weighted(datasets, list(datasets))}
    for group, names in (groups or {}).items():
        present = [name for name in names if name in datasets]
        if present:
            result[group] = _weighted(datasets, present)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="answer.jsonl from inference_matting.py")
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=DS1,DS2",
        help="extra sample-weighted aggregate over the listed datasets",
    )
    args = parser.parse_args()
    groups = {}
    for spec in args.group:
        name, _, members = spec.partition("=")
        groups[name] = [m for m in members.split(",") if m]
    with open(args.data_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = evaluate(rows, groups)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
