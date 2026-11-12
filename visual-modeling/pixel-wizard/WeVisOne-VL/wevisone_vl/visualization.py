"""Prediction parsing, overlays, alpha mattes, and cutout export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv
from PIL import Image, ImageFont

if TYPE_CHECKING:
    from .engine import InferenceResult


REF_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
    re.DOTALL | re.IGNORECASE,
)
COLORS = (
    (255, 92, 124),
    (47, 212, 123),
    (64, 156, 255),
    (255, 184, 48),
    (176, 92, 255),
    (0, 204, 196),
)
SV_PALETTE = sv.ColorPalette.from_hex(
    [
        "#FF5C7C",
        "#2FD47B",
        "#409CFF",
        "#FFB830",
        "#B05CFF",
        "#00CCC4",
        "#FF7849",
        "#38BDF8",
    ]
)


def parse_predictions(text: str, width: int, height: int) -> dict[str, list[dict[str, Any]]]:
    """Parse WeVisOne token coordinates into pixel-space annotations."""
    text = (text or "").split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0]
    predictions: dict[str, list[dict[str, Any]]] = {}
    for label, shape, body in REF_RE.findall(text):
        label = label.strip() or "object"
        bucket = predictions.setdefault(label, [])
        size = 2 if shape.lower() == "point" else 4
        for segment in body.split(","):
            values = [int(value) for value in re.findall(r"<(\d{1,4})>", segment)]
            for offset in range(0, len(values) - size + 1, size):
                chunk = values[offset : offset + size]
                if size == 2:
                    coords = [chunk[0] / 1000 * width, chunk[1] / 1000 * height]
                else:
                    coords = [
                        chunk[0] / 1000 * width,
                        chunk[1] / 1000 * height,
                        chunk[2] / 1000 * width,
                        chunk[3] / 1000 * height,
                    ]
                    if coords[2] <= coords[0] or coords[3] <= coords[1]:
                        continue
                bucket.append({"type": shape.lower(), "coords": coords})
        if not bucket:
            predictions.pop(label, None)
    return predictions


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def combined_alpha(masks: list[np.ndarray], width: int, height: int) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.float32)
    for mask in masks:
        current = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if current.shape != (height, width):
            current = np.asarray(
                Image.fromarray(np.uint8(current * 255), mode="L").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        alpha = np.maximum(alpha, current)
    return alpha


def render_overlay(
    image: Image.Image,
    result: InferenceResult,
    *,
    mask_opacity: float = 0.5,
    mask_threshold: float | None = None,
) -> Image.Image:
    """Overlay continuous mattes and draw every parsed box/point."""
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32).copy()
    alpha = combined_alpha(result.masks, image.width, image.height)
    if mask_threshold is not None:
        alpha = (alpha >= float(mask_threshold)).astype(np.float32)
    if alpha.max() > 0:
        color = np.asarray(COLORS[0], dtype=np.float32)
        weight = (alpha * max(0.0, min(1.0, mask_opacity)))[..., None]
        array = array * (1.0 - weight) + color * weight
    canvas = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
    line_width = max(2, round(image.width / 400))
    predictions = parse_predictions(result.response, image.width, image.height)

    box_xyxy: list[list[float]] = []
    box_class_ids: list[int] = []
    box_labels: list[str] = []
    for label_index, (label, annotations) in enumerate(predictions.items()):
        for annotation in annotations:
            if annotation["type"] == "box":
                box_xyxy.append(annotation["coords"])
                box_class_ids.append(label_index)
                box_labels.append(label)

    if box_xyxy:
        detections = sv.Detections(
            xyxy=np.asarray(box_xyxy, dtype=np.float32),
            class_id=np.asarray(box_class_ids, dtype=int),
        )
        scene = np.array(canvas, dtype=np.uint8, copy=True)
        box_annotator = sv.BoxAnnotator(color=SV_PALETTE, thickness=line_width)
        label_annotator = sv.LabelAnnotator(
            color=SV_PALETTE,
            text_color=sv.Color.BLACK,
            text_scale=max(0.45, min(0.85, image.width / 1500)),
            text_thickness=max(1, round(line_width / 2)),
            text_padding=max(4, line_width * 2),
            smart_position=True,
        )
        scene = box_annotator.annotate(scene=scene, detections=detections)
        scene = label_annotator.annotate(scene=scene, detections=detections, labels=box_labels)
        canvas = Image.fromarray(scene)

    point_xyxy: list[list[float]] = []
    point_class_ids: list[int] = []
    for label_index, (label, annotations) in enumerate(predictions.items()):
        for annotation in annotations:
            if annotation["type"] == "point":
                x, y = annotation["coords"]
                point_xyxy.append([x, y, x, y])
                point_class_ids.append(label_index)

    if point_xyxy:
        point_detections = sv.Detections(
            xyxy=np.asarray(point_xyxy, dtype=np.float32),
            class_id=np.asarray(point_class_ids, dtype=int),
        )
        scene = np.array(canvas, dtype=np.uint8, copy=True)
        radius = max(6, round(image.width / 85))
        dot_annotator = sv.DotAnnotator(
            color=SV_PALETTE,
            radius=radius,
            outline_thickness=max(2, round(radius / 3)),
            outline_color=sv.Color.WHITE,
        )
        scene = dot_annotator.annotate(scene=scene, detections=point_detections)
        canvas = Image.fromarray(scene)
    return canvas


def make_cutout(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgba = np.dstack(
        (
            np.asarray(image.convert("RGB"), dtype=np.uint8),
            np.uint8(np.clip(alpha, 0.0, 1.0) * 255),
        )
    )
    return Image.fromarray(rgba, mode="RGBA")


def save_result(
    output_dir: str | Path,
    image: Image.Image,
    result: InferenceResult,
) -> dict[str, Path]:
    """Save text, structured metadata, overlays, and lossless alpha outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    image = image.convert("RGB")

    response_path = output_dir / "response.txt"
    response_path.write_text(result.response + "\n", encoding="utf-8")
    metadata_path = output_dir / "result.json"
    metadata_path.write_text(
        json.dumps(
            {
                "prompt": result.prompt,
                "response": result.response,
                "latency_s": result.latency_s,
                "seg_boxes_normalized_xyxy": result.boxes,
                "mask_files": [f"masks/mask_{i:03d}.png" for i in range(len(result.masks))],
                "predictions": parse_predictions(result.response, image.width, image.height),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    overlay_path = output_dir / "overlay.png"
    render_overlay(image, result).save(overlay_path)
    alpha = combined_alpha(result.masks, image.width, image.height)
    alpha_path = output_dir / "alpha.png"
    Image.fromarray(np.uint8(np.clip(alpha, 0.0, 1.0) * 255), mode="L").save(alpha_path)
    cutout_path = output_dir / "cutout.png"
    make_cutout(image, alpha).save(cutout_path)
    for index, mask in enumerate(result.masks):
        Image.fromarray(np.uint8(np.clip(mask, 0.0, 1.0) * 255), mode="L").save(
            masks_dir / f"mask_{index:03d}.png"
        )
    return {
        "response": response_path,
        "metadata": metadata_path,
        "overlay": overlay_path,
        "alpha": alpha_path,
        "cutout": cutout_path,
    }
