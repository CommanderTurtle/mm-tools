from __future__ import annotations

import cv2
import numpy as np


def decode_image(data: bytes, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if image is None:
        raise ValueError("Could not decode image data.")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            maximum = np.iinfo(image.dtype).max
            image = np.clip(image.astype(np.float32) * (255.0 / maximum), 0, 255).astype(np.uint8)
        else:
            image = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("Could not encode PNG output.")
    return encoded.tobytes()


def prepare_mask(mask: np.ndarray, shape: tuple[int, int], grow: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 3] if mask.shape[2] == 4 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 24, 255, 0).astype(np.uint8)
    if grow > 0:
        size = 2 * grow + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask


def remove_object(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "telea",
    radius: float = 5,
    grow: int = 5,
    feather: int = 4,
) -> np.ndarray:
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] == 4 else None
    rgb = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    binary = prepare_mask(mask, rgb.shape[:2], grow)
    if not np.any(binary):
        raise ValueError("The mask is empty. Brush or select an object first.")
    algorithm = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    repaired = cv2.inpaint(rgb, binary, max(1.0, float(radius)), algorithm)
    if feather > 0:
        k = feather * 2 + 1
        blend = cv2.GaussianBlur(binary, (k, k), 0).astype(np.float32)[:, :, None] / 255.0
        repaired = np.clip(repaired * blend + rgb * (1.0 - blend), 0, 255).astype(np.uint8)
    if alpha is not None:
        repaired = np.dstack([repaired, alpha])
    return repaired


def fuzzy_mask(image: np.ndarray, x: int, y: int, tolerance: int) -> np.ndarray:
    rgb = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = rgb.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Fuzzy-selection point is outside the image.")
    flood = np.zeros((height + 2, width + 2), np.uint8)
    working = rgb.copy()
    delta = (tolerance, tolerance, tolerance)
    cv2.floodFill(working, flood, (x, y), (255, 255, 255), delta, delta, cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    return flood[1:-1, 1:-1]
