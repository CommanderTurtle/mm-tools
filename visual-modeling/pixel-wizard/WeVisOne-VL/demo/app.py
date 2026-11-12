#!/usr/bin/env python3
"""Gradio web demo for the released WeVisOne-VL checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import html
import inspect
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _proxy_key in ("no_proxy", "NO_PROXY"):
    _existing_no_proxy = [p.strip() for p in os.environ.get(_proxy_key, "").split(",") if p.strip()]
    for _local_host in ("localhost", "127.0.0.1", "0.0.0.0"):
        if _local_host not in _existing_no_proxy:
            _existing_no_proxy.append(_local_host)
    os.environ[_proxy_key] = ",".join(_existing_no_proxy)

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from wevisone_vl.engine import DEFAULT_MODEL_ID, InferenceResult
    from wevisone_vl.prompts import build_prompt as build_engine_prompt
except ImportError:
    import importlib.util

    DEFAULT_MODEL_ID = "tencent/WeVisOne-VL-4B"

    @dataclass
    class InferenceResult:  # type: ignore[no-redef]
        response: str
        prompt: str
        boxes: list[list[float]] = field(default_factory=list)
        masks: list[np.ndarray] = field(default_factory=list)
        latency_s: float = 0.0

    _prompts_spec = importlib.util.spec_from_file_location(
        "wevisone_vl_prompts", PROJECT_ROOT / "wevisone_vl" / "prompts.py"
    )
    assert _prompts_spec and _prompts_spec.loader
    _prompts_mod = importlib.util.module_from_spec(_prompts_spec)
    _prompts_spec.loader.exec_module(_prompts_mod)
    build_engine_prompt = _prompts_mod.build_prompt

EXAMPLES_IMAGE_DIR = DEMO_DIR / "examples" / "images"

# ---------------------------------------------------------------------------
# Task taxonomy (aligned with the WeVisOne-VL paper)
#   Part 1: Object-level Perception (6 tasks)
#   Part 2: Pixel-level Perception  (2 tasks: Segmentation & Matting)
# ---------------------------------------------------------------------------
OBJECT_TASKS = ("referring", "detection", "counting", "point", "ocr", "caption")
PIXEL_TASKS = ("segmentation", "matting")
ALL_TASKS = OBJECT_TASKS + PIXEL_TASKS

TASK_LABELS: Dict[str, str] = {
    "referring": "Grounding",
    "detection": "Dense Detection",
    "counting": "Counting",
    "point": "Pointing",
    "ocr": "OCR",
    "caption": "Caption",
    "segmentation": "Segmentation",
    "matting": "Matting",
}

COUNTING_MODES = ["Box Counting", "Point Counting"]
OCR_MODES = ["All Text", "Text Grounding"]
CAPTION_MODES = [
    "Detailed Caption (with Boxes)",
    "One-Sentence Caption (with Boxes)",
    "Free-form / VQA",
]
CAPTION_PRESET_PROMPTS: Dict[str, str] = {
    "Detailed Caption (with Boxes)": "describe the image with bounding boxes in detail",
    "One-Sentence Caption (with Boxes)": "describe the image with bounding boxes in one sentence.",
    "Free-form / VQA": "What is happening in this photo?",
}

DEFAULT_SUBTASK: Dict[str, str] = {
    "referring": "",
    "detection": "",
    "counting": "Box Counting",
    "point": "",
    "ocr": "All Text",
    "caption": "Detailed Caption (with Boxes)",
    "segmentation": "",
    "matting": "",
}

DEFAULT_QUERIES: Dict[Tuple[str, str], str] = {
    ("referring", ""): "boys holding microphone",
    ("detection", ""): "sofa\npillow\ncoffee table\nbook\nlamp\nvase",
    ("counting", "Box Counting"): "white chess pieces",
    ("counting", "Point Counting"): "white chess pieces",
    ("point", ""): "dogs",
    ("ocr", "All Text"): "",
    ("ocr", "Text Grounding"): "COFFEE",
    ("caption", "Detailed Caption (with Boxes)"): CAPTION_PRESET_PROMPTS["Detailed Caption (with Boxes)"],
    ("caption", "One-Sentence Caption (with Boxes)"): CAPTION_PRESET_PROMPTS["One-Sentence Caption (with Boxes)"],
    ("caption", "Free-form / VQA"): CAPTION_PRESET_PROMPTS["Free-form / VQA"],
    ("segmentation", ""): "parasol with alternating rings of light green and dark green with flowers",
    ("matting", ""): "person",
}

SEG_DISPLAY_MODES = ["Overlay", "Binary Mask", "Cutout"]
MATTING_DISPLAY_MODES = ["Green Screen", "Overlay", "Alpha Matte", "Cutout"]
MATTING_GREEN_RGB = (0.0, 214.0, 90.0)

GALLERY_EXAMPLES: List[Dict[str, str]] = [
    {
        "title": "Grounding · Singers",
        "image": "boys.jpg",
        "task": "referring",
        "subtask": "",
        "query": "boys holding microphone",
    },
    {
        "title": "Dense Detection · Living Room",
        "image": "livingroom.jpg",
        "task": "detection",
        "subtask": "",
        "query": "sofa\npillow\ncoffee table\nbook\nlamp\nvase",
    },
    {
        "title": "Counting · Chess Pieces",
        "image": "chess.jpg",
        "task": "counting",
        "subtask": "Box Counting",
        "query": "white chess pieces",
    },
    {
        "title": "Pointing · Dogs",
        "image": "dogs.jpg",
        "task": "point",
        "subtask": "",
        "query": "dogs",
    },
    {
        "title": "OCR · Coffee Sign",
        "image": "coffee_sign.jpg",
        "task": "ocr",
        "subtask": "All Text",
        "query": "",
    },
    {
        "title": "Caption · Grounded Description",
        "image": "livingroom.jpg",
        "task": "caption",
        "subtask": "Detailed Caption (with Boxes)",
        "query": "describe the image with bounding boxes in detail",
    },
    {
        "title": "Segmentation · Parasol",
        "image": "parasol.jpg",
        "task": "segmentation",
        "subtask": "",
        "query": "parasol with alternating rings of light green and dark green with flowers",
    },
    {
        "title": "Matting · Portrait",
        "image": "seabreeze.jpg",
        "task": "matting",
        "subtask": "",
        "query": "person",
    },
]

# ---------------------------------------------------------------------------
# Styling (unified sans-serif font, minimal copy)
# ---------------------------------------------------------------------------
UI_FONT_STACK = [
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "Noto Sans",
    "sans-serif",
]

UI_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 8px 28px 40px !important;
}
.wv-hero {
    text-align: center;
    padding: 32px 16px 20px;
}
.wv-hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0;
}
.wv-hero .wv-grad {
    background: linear-gradient(90deg, #6366f1, #8b5cf6 50%, #06b6d4);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.wv-hero p {
    margin: 12px auto 0;
    max-width: 720px;
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--body-text-color-subdued);
}
.wv-links {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}
.wv-links a {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--border-color-primary);
    background: var(--block-background-fill);
    color: var(--body-text-color) !important;
    font-size: 0.88rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: border-color .15s, transform .15s;
}
.wv-links a:hover {
    border-color: var(--color-accent);
    transform: translateY(-1px);
}

.wv-card {
    background: #ffffff !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 20px !important;
    padding: 22px 24px !important;
    gap: 16px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 10px 30px -12px rgba(15, 23, 42, .12);
}
.dark .wv-card {
    background: var(--neutral-900) !important;
}
.wv-card .block {
    box-shadow: none !important;
}
.wv-main {
    gap: 24px !important;
    align-items: stretch;
}
.wv-section {
    gap: 14px !important;
    margin-top: 24px;
}
.wv-title {
    margin: 0 !important;
}
.wv-title h3 {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}
.wv-title h3 .wv-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    font-size: .8rem;
    color: #fff;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
}

.wv-task-bar {
    gap: 28px !important;
    flex-wrap: wrap !important;
    align-items: flex-start;
}
.wv-task-col {
    gap: 10px !important;
    flex: 0 1 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
.wv-task-label p {
    margin: 0 !important;
    font-size: .76rem !important;
    font-weight: 700 !important;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
}
.wv-chip-row {
    gap: 8px !important;
    flex-wrap: wrap !important;
}
button.wv-chip {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: max-content !important;
    white-space: nowrap !important;
    border-radius: 999px !important;
    padding: 8px 18px !important;
    font-size: .92rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: all .15s ease;
}
button.wv-chip.secondary {
    background: transparent !important;
    border: 1px solid var(--border-color-primary) !important;
    color: var(--body-text-color) !important;
}
button.wv-chip.secondary:hover {
    border-color: var(--color-accent) !important;
    color: var(--color-accent) !important;
}
button.wv-chip.primary {
    border: 1px solid transparent !important;
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    box-shadow: 0 6px 16px -6px rgba(99, 102, 241, .6) !important;
}
button.wv-run {
    border-radius: 14px !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    letter-spacing: .02em;
    background: linear-gradient(90deg, #6366f1, #8b5cf6 55%, #06b6d4) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 10px 24px -10px rgba(99, 102, 241, .7) !important;
}
button.wv-run:hover {
    filter: brightness(1.06);
}

.wv-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
}
.wv-stat {
    display: inline-flex;
    align-items: baseline;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    background: var(--background-fill-secondary);
    font-size: .88rem;
    color: var(--body-text-color-subdued);
}
.wv-stat b {
    font-size: 1rem;
    color: var(--body-text-color);
}
.wv-stat.wv-stat-hero b {
    font-size: 1.15rem;
    color: var(--color-accent);
}
.wv-legend {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid var(--border-color-primary);
    font-size: .85rem;
}
.wv-legend i {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
}
.wv-legend span {
    color: var(--body-text-color-subdued);
}
.wv-placeholder {
    font-size: .9rem;
    color: var(--body-text-color-subdued);
}

.wv-example-gallery .thumbnail-item {
    aspect-ratio: 4 / 3 !important;
    border-radius: 14px !important;
    overflow: hidden;
    transition: transform .15s ease, box-shadow .15s ease;
}
.wv-example-gallery .thumbnail-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -12px rgba(15, 23, 42, .35);
}
"""

HERO_HTML = """
<div class="wv-hero">
  <h1>WeVisOne-VL</h1>
  <h1 class="wv-grad" style="font-size:1.5rem;margin-top:6px;">Unified Visual Perception from Objects to Pixels</h1>
  <p>One model for grounding, detection, counting, pointing, OCR, grounded captioning,
     segmentation and matting. Pick a task, upload an image, and run.</p>
  <div class="wv-links">
    <a href="https://github.com/Tencent/WeVisOne-VL" target="_blank">GitHub</a>
    <a href="https://huggingface.co/tencent/WeVisOne-VL-4B" target="_blank">🤗 WeVisOne-VL-4B</a>
    <a href="https://huggingface.co/tencent/WeVisOne-VL-2B" target="_blank">🤗 WeVisOne-VL-2B</a>
  </div>
</div>
"""

EMPTY_SUMMARY_HTML = '<div class="wv-placeholder">Run the model to see results here.</div>'


def _section_title(step: int, text: str) -> gr.Markdown:
    return gr.Markdown(
        f'### <span class="wv-step">{step}</span>{text}', elem_classes=["wv-title"]
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def resolve_engine_task(task: str, subtask: str) -> str:
    task = (task or "referring").strip().lower()
    subtask = (subtask or "").strip()
    if task == "grounding":
        return "referring"
    if task in ("dense detection", "dense_detection"):
        return "detection"
    if task == "counting":
        return "count_point" if subtask == "Point Counting" else "count_box"
    if task == "ocr":
        return "ocr_grounding" if subtask == "Text Grounding" else "ocr"
    if task == "caption":
        return "freeform"
    return task


def build_task_prompt(task: str, subtask: str, query: str) -> str:
    engine_task = resolve_engine_task(task, subtask)
    text_query = (query or "").strip()
    if task == "caption":
        if not text_query:
            text_query = CAPTION_PRESET_PROMPTS.get(
                subtask, CAPTION_PRESET_PROMPTS["Detailed Caption (with Boxes)"]
            )
        return text_query
    if engine_task != "ocr" and not text_query:
        default_q = DEFAULT_QUERIES.get((task, subtask)) or DEFAULT_QUERIES.get((task, ""), "object")
        text_query = default_q
    return build_engine_prompt(engine_task, text_query)


# ---------------------------------------------------------------------------
# Prediction parsing & visualization
# ---------------------------------------------------------------------------
_REF_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
    re.DOTALL | re.IGNORECASE,
)
_COUNT_RE = re.compile(r"Total number:\s*(\d+)", re.IGNORECASE)

_PALETTE: List[Tuple[int, int, int]] = [
    (255, 92, 124),
    (47, 212, 123),
    (64, 156, 255),
    (255, 184, 48),
    (176, 92, 255),
    (0, 204, 196),
    (255, 112, 67),
    (120, 200, 80),
]
_BOX_COLOR_AUTO = "Auto by Category"
_BOX_COLOR_CUSTOM = "Custom"
_BOX_COLOR_PRESETS: Dict[str, Optional[Tuple[int, int, int]]] = {
    _BOX_COLOR_AUTO: None,
    "Pink": (255, 92, 124),
    "Green": (47, 212, 123),
    "Blue": (64, 156, 255),
    "Amber": (255, 184, 48),
    "Purple": (176, 92, 255),
    "Cyan": (0, 204, 196),
    "White": (255, 255, 255),
    _BOX_COLOR_CUSTOM: None,
}


def _color_for(text: str) -> Tuple[int, int, int]:
    return _PALETTE[abs(hash(text)) % len(_PALETTE)]


def _parse_hex_color(value: Any, default: Tuple[int, int, int] = (255, 92, 124)) -> Tuple[int, int, int]:
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    s = str(value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) >= 6:
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            pass
    return default


def _resolve_box_color(preset: str, custom_hex: str, category: str) -> Tuple[int, int, int]:
    if preset == _BOX_COLOR_CUSTOM:
        return _parse_hex_color(custom_hex)
    rgb = _BOX_COLOR_PRESETS.get(preset)
    if rgb is not None:
        return rgb
    return _color_for(category)


def _load_font(size: int) -> ImageFont.ImageFont:
    size = max(8, int(size))
    for path in (
        "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _display_scale(image: Image.Image, target_w: float = 800.0) -> float:
    longest = max(int(image.width), int(image.height), 1)
    return max(0.75, min(longest / target_w, 5.0))


def _contrast_ink(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    y = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (255, 255, 255) if y < 186 else (20, 24, 32)


def _draw_tag(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    *,
    above: bool = True,
) -> None:
    text = str(text or "")
    if not text:
        return
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x = max(5, int(round(th * 0.5)))
    pad_y = max(3, int(round(th * 0.3)))
    box_h = th + pad_y * 2
    box_w = tw + pad_x * 2
    top = y - box_h if above and y > box_h + 1 else y
    draw.rounded_rectangle(
        [x, top, x + box_w, top + box_h], radius=max(3, box_h // 3), fill=color
    )
    draw.text(
        (x + pad_x - bbox[0], top + pad_y - bbox[1]),
        text,
        fill=_contrast_ink(color),
        font=font,
    )


def _draw_point(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    tag: str,
    font: ImageFont.ImageFont,
    scale: float,
    draw_width: int,
    show_labels: bool,
) -> None:
    r = max(8, int(round((8 + max(draw_width, 1)) * scale)))
    ring = max(2, int(round(2.5 * scale)))
    draw.ellipse([x - r - ring, y - r - ring, x + r + ring, y + r + ring], fill="white", outline="white")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    inner = max(3, r // 3)
    draw.ellipse([x - inner, y - inner, x + inner, y + inner], fill="white", outline="white")
    if show_labels and tag:
        _draw_tag(draw, tag, x + r + int(4 * scale), y - r, color, font, above=True)


def parse_predictions_with_masks(
    text: str,
    width: int,
    height: int,
    masks: Sequence[np.ndarray],
) -> Dict[str, List[Dict[str, Any]]]:
    clean = (text or "").split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0]
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    mask_idx = 0

    for label, shape, body in _REF_RE.findall(clean):
        cat = (label or "object").strip() or "object"
        bucket = predictions.setdefault(cat, [])
        kind = shape.lower()
        step = 2 if kind == "point" else 4
        for segment in (body or "").split(","):
            values = [int(v) for v in re.findall(r"<(\d{1,4})>", segment)]
            has_seg = bool(re.search(r"<(?:OBJ|SEG)>", segment or "", flags=re.IGNORECASE))
            for offset in range(0, len(values) - step + 1, step):
                chunk = values[offset : offset + step]
                if step == 2:
                    coords = [chunk[0] / 1000.0 * width, chunk[1] / 1000.0 * height]
                else:
                    coords = [
                        chunk[0] / 1000.0 * width,
                        chunk[1] / 1000.0 * height,
                        chunk[2] / 1000.0 * width,
                        chunk[3] / 1000.0 * height,
                    ]
                    if coords[2] <= coords[0] or coords[3] <= coords[1]:
                        continue
                item: Dict[str, Any] = {"type": kind, "coords": coords}
                if has_seg and mask_idx < len(masks):
                    item["mask"] = masks[mask_idx]
                    mask_idx += 1
                bucket.append(item)
        if not bucket:
            predictions.pop(cat, None)

    if mask_idx == 0 and masks:
        flat_boxes = [
            ann
            for anns in predictions.values()
            for ann in anns
            if ann.get("type") == "box"
        ]
        for ann, m in zip(flat_boxes, masks):
            ann["mask"] = m
    return predictions


def extract_count(text: str, predictions: Dict[str, List[Dict[str, Any]]], *, counting: bool = False) -> Optional[int]:
    m = _COUNT_RE.search(text or "")
    if m:
        return int(m.group(1))
    n = sum(len(v) for v in predictions.values())
    if n:
        return n
    if counting:
        nums = re.findall(r"-?\d+", text or "")
        if nums:
            return int(nums[-1])
    return None


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    img = Image.fromarray(
        np.clip(mask * 255.0 if mask.dtype != np.uint8 else mask, 0, 255).astype(np.uint8),
        mode="L",
    ).resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _combined_alpha(
    predictions: Dict[str, List[Dict[str, Any]]],
    raw_masks: Sequence[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    combined = np.zeros((height, width), dtype=np.float32)
    found = False
    for anns in predictions.values():
        for ann in anns:
            raw_mask = ann.get("mask")
            if raw_mask is None:
                continue
            mask = np.asarray(raw_mask, dtype=np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = np.clip(mask, 0.0, 1.0)
            combined = np.maximum(combined, _resize_mask(mask, width, height))
            found = True
    if not found:
        for raw_mask in raw_masks:
            mask = np.clip(np.asarray(raw_mask, dtype=np.float32), 0.0, 1.0)
            combined = np.maximum(combined, _resize_mask(mask, width, height))
    return combined


def _overlay_masks(
    image: Image.Image,
    predictions: Dict[str, List[Dict[str, Any]]],
    raw_masks: Sequence[np.ndarray],
    *,
    box_color_preset: str,
    box_color_hex: str,
    mask_alpha: float,
    mask_threshold: float,
    mask_mode: str = "binary",
) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    alpha = float(max(0.0, min(1.0, mask_alpha)))
    if alpha <= 0:
        return image.copy()
    painted = False
    for category, anns in predictions.items():
        color = _resolve_box_color(box_color_preset, box_color_hex, category)
        for ann in anns:
            raw_mask = ann.get("mask")
            if raw_mask is None:
                continue
            mask = np.asarray(raw_mask, dtype=np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = np.clip(mask, 0.0, 1.0)
            mask = _resize_mask(mask, width, height)
            if mask_mode == "matting":
                keep = mask > 0.0
                weight = (mask * alpha)[..., None]
            else:
                keep = mask >= float(mask_threshold)
                weight = (keep.astype(np.float32) * alpha)[..., None]
            if not np.any(keep):
                continue
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            arr = np.where(keep[..., None], arr * (1.0 - weight) + color_arr * weight, arr)
            painted = True
    if not painted and raw_masks:
        combined = _combined_alpha(predictions, raw_masks, width, height)
        if mask_mode == "matting":
            keep = combined > 0.0
            weight = (combined * alpha)[..., None]
        else:
            keep = combined >= float(mask_threshold)
            weight = (keep.astype(np.float32) * alpha)[..., None]
        if np.any(keep):
            color_arr = np.array(_PALETTE[0], dtype=np.float32).reshape(1, 1, 3)
            arr = np.where(keep[..., None], arr * (1.0 - weight) + color_arr * weight, arr)
            painted = True
    if not painted:
        return image.copy()
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_from_cache(
    cache: Optional[Dict[str, Any]],
    font_size: int = 16,
    draw_width: int = 3,
    show_labels: bool = True,
    box_color_preset: str = _BOX_COLOR_AUTO,
    box_color_hex: str = "#ff5c7c",
    show_boxes: bool = True,
    show_masks: bool = True,
    mask_alpha: float = 0.45,
    mask_threshold: float = 0.5,
    pixel_display: str = "Overlay",
) -> Optional[Image.Image]:
    if not cache:
        return None
    arr = cache.get("pil")
    if arr is None:
        return None
    pil = arr.convert("RGB") if isinstance(arr, Image.Image) else Image.fromarray(np.asarray(arr)).convert("RGB")
    preds: Dict[str, List[Dict[str, Any]]] = cache.get("preds") or {}
    raw_masks: List[np.ndarray] = cache.get("masks") or []
    task: str = cache.get("task") or "detection"
    is_pixel_task = task in PIXEL_TASKS
    width, height = pil.size

    if is_pixel_task and (preds or raw_masks):
        alpha = _combined_alpha(preds, raw_masks, width, height)
        effective_alpha = alpha if task == "matting" else (alpha >= float(mask_threshold)).astype(np.float32)
        alpha_u8 = np.clip(effective_alpha * 255.0, 0, 255).astype(np.uint8)

        if pixel_display in ("Alpha Matte", "Binary Mask"):
            return Image.fromarray(alpha_u8, mode="L")
        if pixel_display == "Green Screen":
            fg = np.asarray(pil, dtype=np.float32)
            green = np.empty_like(fg)
            green[...] = MATTING_GREEN_RGB
            w3 = effective_alpha[..., None]
            comp = fg * w3 + green * (1.0 - w3)
            return Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), mode="RGB")
        if pixel_display == "Cutout":
            rgba = pil.convert("RGBA")
            rgba.putalpha(Image.fromarray(alpha_u8, mode="L"))
            return rgba

    vis = pil.copy()
    if is_pixel_task and show_masks and (preds or raw_masks):
        vis = _overlay_masks(
            vis,
            preds,
            raw_masks,
            box_color_preset=box_color_preset,
            box_color_hex=box_color_hex,
            mask_alpha=float(mask_alpha),
            mask_threshold=float(mask_threshold),
            mask_mode=("matting" if task == "matting" else "binary"),
        )

    if not preds:
        return vis

    if show_boxes and not is_pixel_task:
        fill_layer = Image.new("RGBA", vis.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill_layer)
        for category, anns in preds.items():
            color = _resolve_box_color(box_color_preset, box_color_hex, category)
            for ann in anns:
                coords = ann.get("coords") or []
                if ann.get("type", "box") == "box" and len(coords) == 4:
                    x0, y0, x1, y1 = [int(c) for c in coords]
                    if x1 > x0 and y1 > y0:
                        fill_draw.rectangle([x0, y0, x1, y1], fill=(*color, 38))
        vis = Image.alpha_composite(vis.convert("RGBA"), fill_layer).convert("RGB")

    draw = ImageDraw.Draw(vis)
    scale = _display_scale(vis)
    font = _load_font(max(11, int(round(int(font_size) * scale))))
    lw = max(1, int(round(max(int(draw_width), 1) * scale)))

    for category, anns in preds.items():
        color = _resolve_box_color(box_color_preset, box_color_hex, category)
        tag = str(category or "").strip() or "object"
        for ann in anns:
            kind = ann.get("type", "box")
            coords = ann.get("coords") or []
            if kind == "box" and len(coords) == 4:
                x0, y0, x1, y1 = [int(c) for c in coords]
                if x1 <= x0 or y1 <= y0:
                    continue
                if show_boxes:
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
                if show_labels and show_boxes:
                    _draw_tag(draw, tag, x0, y0, color, font)
            elif kind == "point" and len(coords) == 2:
                x, y = int(coords[0]), int(coords[1])
                _draw_point(draw, x, y, color, tag, font, scale, int(draw_width), bool(show_labels))
    return vis


# ---------------------------------------------------------------------------
# Mock engine for UI preview without loading GPU weights
# ---------------------------------------------------------------------------
class MockEngine:
    """Stand-in engine used with ``--mock`` so the UI can be tested without a GPU."""

    model_name_or_path = f"{DEFAULT_MODEL_ID} (mock)"
    backend = "mock"

    def predict(
        self,
        image: Image.Image,
        *,
        task: str = "freeform",
        query: str | Sequence[str] = "",
        prompt: Optional[str] = None,
        decode_matting: bool = True,
        **generation_kwargs: Any,
    ) -> InferenceResult:
        del generation_kwargs
        time.sleep(0.25)
        w, h = image.size
        actual_prompt = (prompt or "").strip() or build_engine_prompt(task, query)
        first_label = (
            (query.splitlines()[0].strip() if isinstance(query, str) and query.strip() else "object")
            if query
            else "object"
        )
        if task in {"point", "count_point"}:
            resp = (
                f"<ref>{first_label}</ref><point><320><460>,<640><510></point>"
                + ("\nTotal number: 2" if task == "count_point" else "")
            )
            return InferenceResult(response=resp, prompt=actual_prompt, latency_s=0.25)
        if task == "count_box":
            resp = (
                f"<ref>{first_label}</ref><box><180><260><420><740>,<540><280><810><760></box>\n"
                "Total number: 2"
            )
            return InferenceResult(response=resp, prompt=actual_prompt, latency_s=0.25)
        if task in {"segmentation", "matting"} and decode_matting:
            yy, xx = np.mgrid[:h, :w]
            dist2 = ((xx - w * 0.5) / (w * 0.26)) ** 2 + ((yy - h * 0.52) / (h * 0.34)) ** 2
            matte = np.clip(1.25 - dist2, 0.0, 1.0).astype(np.float32)
            resp = f"<ref>{first_label}</ref><box><240><180><760><860><OBJ></box>"
            return InferenceResult(
                response=resp,
                prompt=actual_prompt,
                boxes=[[0.24, 0.18, 0.76, 0.86]],
                masks=[matte],
                latency_s=0.25,
            )
        if task == "freeform":
            resp = (
                f"In this scene, <ref>main subject</ref><box><160><180><840><860></box> "
                "is clearly visible in the foreground."
            )
            return InferenceResult(response=resp, prompt=actual_prompt, latency_s=0.25)
        resp = f"<ref>{first_label}</ref><box><180><200><780><840></box>"
        return InferenceResult(response=resp, prompt=actual_prompt, latency_s=0.25)


# ---------------------------------------------------------------------------
# Gradio UI helpers
# ---------------------------------------------------------------------------
def _to_pil(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (str, Path)) and Path(image).is_file():
        return Image.open(str(image)).convert("RGB")
    return None


def subtask_ui_for(task: str, subtask: Optional[str] = None) -> Dict[str, Any]:
    if task == "counting":
        val = subtask if subtask in COUNTING_MODES else DEFAULT_SUBTASK["counting"]
        return gr.update(visible=True, choices=COUNTING_MODES, value=val, label="Mode")
    if task == "ocr":
        val = subtask if subtask in OCR_MODES else DEFAULT_SUBTASK["ocr"]
        return gr.update(visible=True, choices=OCR_MODES, value=val, label="Mode")
    if task == "caption":
        val = subtask if subtask in CAPTION_MODES else DEFAULT_SUBTASK["caption"]
        return gr.update(visible=True, choices=CAPTION_MODES, value=val, label="Preset")
    return gr.update(visible=False)


def query_ui_for(task: str, subtask: str, value: Optional[str] = None) -> Dict[str, Any]:
    if value is None:
        value = DEFAULT_QUERIES.get((task, subtask), DEFAULT_QUERIES.get((task, ""), ""))
    if task == "referring":
        return gr.update(
            visible=True,
            value=value,
            lines=2,
            label="Grounding Expression",
            placeholder="boys holding microphone",
        )
    if task == "detection":
        return gr.update(
            visible=True,
            value=value,
            lines=4,
            label="Categories (one per line)",
            placeholder="sofa\npillow\ncoffee table",
        )
    if task == "counting":
        return gr.update(
            visible=True,
            value=value,
            lines=1,
            label="Target Category",
            placeholder="white chess pieces",
        )
    if task == "point":
        return gr.update(
            visible=True,
            value=value,
            lines=2,
            label="Pointing Target",
            placeholder="dogs",
        )
    if task == "ocr":
        if subtask == "Text Grounding":
            return gr.update(
                visible=True,
                value=value,
                lines=1,
                label="Text to Locate",
                placeholder="COFFEE",
            )
        return gr.update(visible=False, value="")
    if task == "caption":
        return gr.update(
            visible=True,
            value=value,
            lines=2,
            label="Instruction / Question",
            placeholder="describe the image with bounding boxes in detail",
        )
    if task in PIXEL_TASKS:
        return gr.update(
            visible=True,
            value=value,
            lines=2,
            label="Referring Expression",
            placeholder="parasol with alternating rings of light green and dark green with flowers",
        )
    return gr.update(visible=True, value=value, lines=2, label="Query")


def pixel_display_ui_for(task: str) -> Dict[str, Any]:
    """Only Pixel-level tasks (Segmentation & Matting) show overlay/matte display controls."""
    if task == "segmentation":
        return gr.update(
            visible=True,
            choices=SEG_DISPLAY_MODES,
            value="Overlay",
            label="Display Mode",
        )
    if task == "matting":
        return gr.update(
            visible=True,
            choices=MATTING_DISPLAY_MODES,
            value="Green Screen",
            label="Display Mode",
        )
    return gr.update(visible=False)


def task_button_updates(active_task: str) -> List[Dict[str, Any]]:
    return [
        gr.update(variant="primary" if active_task == t else "secondary")
        for t in ALL_TASKS
    ]


def _style_kwargs(target: str) -> Dict[str, Any]:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=UI_FONT_STACK,
    ).set(
        body_background_fill="#f5f6fa",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_50",
        block_background_fill_dark="*neutral_800",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_border_color_dark="*neutral_700",
        block_shadow="none",
        block_radius="14px",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_text_color="*neutral_500",
        block_label_text_weight="600",
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        block_title_text_color="*neutral_600",
        block_title_text_color_dark="*neutral_300",
        block_title_text_weight="600",
        block_title_padding="0",
        block_label_padding="4px 8px",
        input_background_fill="#ffffff",
        input_background_fill_dark="*neutral_800",
        input_border_color="*neutral_200",
        input_shadow="none",
        input_shadow_focus="0 0 0 3px rgba(99, 102, 241, .18)",
        input_border_color_focus="*primary_400",
        layout_gap="16px",
    )
    blocks_takes_css = "css" in inspect.signature(gr.Blocks.__init__).parameters
    return {"css": UI_CSS, "theme": theme} if (target == "blocks") == blocks_takes_css else {}


def build_demo(engine: Any) -> gr.Blocks:
    engine_lock = threading.Lock()
    initial_example = GALLERY_EXAMPLES[0]
    initial_task = initial_example["task"]
    initial_subtask = initial_example["subtask"]
    initial_query = initial_example["query"]
    initial_prompt = build_task_prompt(initial_task, initial_subtask, initial_query)
    initial_img_path = EXAMPLES_IMAGE_DIR / initial_example["image"]
    initial_image = Image.open(initial_img_path).convert("RGB") if initial_img_path.is_file() else None

    gallery_items: List[Tuple[str, str]] = []
    for ex in GALLERY_EXAMPLES:
        p = EXAMPLES_IMAGE_DIR / ex["image"]
        if p.is_file():
            gallery_items.append((str(p), ex["title"]))

    with gr.Blocks(title="WeVisOne-VL Demo", **_style_kwargs("blocks")) as demo:
        gr.HTML(HERO_HTML)

        raw_image = gr.State(initial_image)
        vis_cache = gr.State(None)
        task_state = gr.State(initial_task)

        with gr.Column(elem_classes=["wv-card"]):
            _section_title(1, "Choose a task")
            with gr.Row(elem_classes=["wv-task-bar"]):
                with gr.Column(elem_classes=["wv-task-col"]):
                    gr.Markdown("Object-level Perception", elem_classes=["wv-task-label"])
                    with gr.Row(elem_classes=["wv-chip-row"]):
                        chip = {"size": "md", "scale": 0, "min_width": 0, "elem_classes": ["wv-chip"]}
                        btn_referring = gr.Button("Grounding", variant="primary", **chip)
                        btn_detection = gr.Button("Dense Detection", variant="secondary", **chip)
                        btn_counting = gr.Button("Counting", variant="secondary", **chip)
                        btn_point = gr.Button("Pointing", variant="secondary", **chip)
                        btn_ocr = gr.Button("OCR", variant="secondary", **chip)
                        btn_caption = gr.Button("Caption", variant="secondary", **chip)
                with gr.Column(elem_classes=["wv-task-col"]):
                    gr.Markdown("Pixel-level Perception", elem_classes=["wv-task-label"])
                    with gr.Row(elem_classes=["wv-chip-row"]):
                        btn_segmentation = gr.Button("Segmentation", variant="secondary", **chip)
                        btn_matting = gr.Button("Matting", variant="secondary", **chip)

        task_buttons = [
            btn_referring,
            btn_detection,
            btn_counting,
            btn_point,
            btn_ocr,
            btn_caption,
            btn_segmentation,
            btn_matting,
        ]

        with gr.Row(equal_height=False, elem_classes=["wv-main", "wv-section"]):
            with gr.Column(scale=1, elem_classes=["wv-card"]):
                _section_title(2, "Image & query")
                input_image = gr.Image(
                    value=initial_image,
                    label="Input Image",
                    show_label=False,
                    type="pil",
                    height=440,
                )
                subtask = gr.Dropdown(
                    choices=COUNTING_MODES,
                    value=COUNTING_MODES[0],
                    allow_custom_value=True,
                    label="Mode",
                    visible=False,
                )
                query = gr.Textbox(
                    label="Grounding Expression",
                    value=initial_query,
                    lines=2,
                    placeholder="boys holding microphone",
                )
                with gr.Accordion("Prompt sent to the model (editable)", open=False):
                    prompt_preview = gr.Textbox(
                        show_label=False,
                        value=initial_prompt,
                        interactive=True,
                        lines=3,
                    )
                run_btn = gr.Button("Run", variant="primary", size="lg", elem_classes=["wv-run"])

                with gr.Accordion("Generation Parameters", open=False):
                    temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="top_p")
                    top_k = gr.Slider(1, 100, value=50, step=1, label="top_k")
                    max_tokens = gr.Slider(64, 8192, value=4096, step=64, label="max_new_tokens")
                    repetition_penalty = gr.Slider(
                        1.0, 1.5, value=1.05, step=0.01, label="repetition_penalty"
                    )

                with gr.Accordion("Visualization Options", open=False):
                    with gr.Row():
                        font_size = gr.Slider(10, 40, value=16, step=1, label="Font Size")
                        draw_width = gr.Slider(1, 10, value=3, step=1, label="Line Width")
                    with gr.Row():
                        show_boxes = gr.Checkbox(value=True, label="Show Boxes")
                        show_labels = gr.Checkbox(value=True, label="Show Labels")
                    with gr.Row():
                        box_color_preset = gr.Dropdown(
                            choices=list(_BOX_COLOR_PRESETS.keys()),
                            value=_BOX_COLOR_AUTO,
                            label="Color Palette",
                        )
                        box_color_hex = gr.ColorPicker(value="#ff5c7c", label="Custom Color")
                    # Pixel-only visualization controls (hidden for Object-level tasks)
                    with gr.Group(visible=False) as pixel_viz_group:
                        show_masks = gr.Checkbox(value=True, label="Show Mask Overlay")
                        mask_alpha = gr.Slider(
                            0.05, 0.95, value=0.45, step=0.05, label="Overlay Opacity"
                        )
                        mask_threshold = gr.Slider(
                            0.1,
                            0.9,
                            value=0.5,
                            step=0.05,
                            label="Segmentation Threshold",
                        )

            with gr.Column(scale=1, elem_classes=["wv-card"]):
                _section_title(3, "Result")
                output_image = gr.Image(
                    label="Result",
                    show_label=False,
                    height=440,
                    interactive=False,
                )
                # Only visible when task is Segmentation or Matting
                pixel_display = gr.Radio(
                    choices=SEG_DISPLAY_MODES,
                    value="Overlay",
                    label="Display Mode",
                    visible=False,
                )
                summary_md = gr.HTML(EMPTY_SUMMARY_HTML)
                with gr.Accordion("Model Output", open=True):
                    raw_out = gr.Textbox(
                        show_label=False,
                        lines=6,
                        max_lines=16,
                        interactive=False,
                    )

        with gr.Column(elem_classes=["wv-card", "wv-section"]):
            gr.Markdown("### Examples", elem_classes=["wv-title"])
            gr.Markdown(
                "Click an example to load the image, task and query, then press **Run**.",
                elem_classes=["wv-placeholder"],
            )
            example_gallery = gr.Gallery(
                value=gallery_items,
                label="Examples",
                show_label=False,
                columns=4,
                height="auto",
                object_fit="cover",
                allow_preview=False,
                elem_classes=["wv-example-gallery"],
            )

        # -------------------------------------------------------------------
        # Event callbacks
        # -------------------------------------------------------------------
        def switch_task(target_task: str):
            sub = DEFAULT_SUBTASK.get(target_task, "")
            default_q = DEFAULT_QUERIES.get(
                (target_task, sub), DEFAULT_QUERIES.get((target_task, ""), "")
            )
            prompt_text = build_task_prompt(target_task, sub, default_q)
            is_pixel = target_task in PIXEL_TASKS
            return (
                target_task,
                subtask_ui_for(target_task, sub),
                query_ui_for(target_task, sub, value=default_q),
                prompt_text,
                pixel_display_ui_for(target_task),
                gr.update(visible=is_pixel),
                gr.update(visible=(target_task == "segmentation")),
                *task_button_updates(target_task),
            )

        def on_subtask_change(task: str, sub: str, current_query: str):
            task = task or "referring"
            if task not in ("counting", "ocr", "caption"):
                return gr.update(), gr.update()
            sub = sub or ""
            default_q = DEFAULT_QUERIES.get((task, sub))
            new_q = default_q if default_q is not None else current_query
            prompt_text = build_task_prompt(task, sub, new_q)
            return (
                query_ui_for(task, sub, value=new_q),
                prompt_text,
            )

        def on_query_change(task: str, sub: str, current_query: str):
            return build_task_prompt(task or "referring", sub or "", current_query or "")

        def on_image_change(img: Optional[Image.Image]):
            return _to_pil(img)

        def run_inference(
            image: Optional[Image.Image],
            cached_raw_image: Optional[Image.Image],
            task: str,
            sub: str,
            q: str,
            user_prompt: str,
            temp: float,
            tp: float,
            tk: int,
            max_tok: int,
            rep_pen: float,
            f_size: int,
            d_width: int,
            s_boxes: bool,
            s_labels: bool,
            c_preset: str,
            c_hex: str,
            s_masks: bool,
            m_alpha: float,
            m_thresh: float,
            p_display: str,
        ):
            pil = _to_pil(image) or _to_pil(cached_raw_image)
            if pil is None:
                return None, "Please upload or select an image first.", "", None

            task = (task or "referring").strip().lower()
            sub = (sub or "").strip()
            engine_task = resolve_engine_task(task, sub)
            is_pixel = task in PIXEL_TASKS

            actual_prompt = (user_prompt or "").strip()
            if not actual_prompt:
                actual_prompt = build_task_prompt(task, sub, q)

            try:
                with engine_lock:
                    result: InferenceResult = engine.predict(
                        pil,
                        task=engine_task,
                        query=q or "",
                        prompt=actual_prompt,
                        decode_matting=is_pixel,
                        max_new_tokens=int(max_tok),
                        temperature=float(temp),
                        top_p=float(tp),
                        top_k=int(tk),
                        repetition_penalty=float(rep_pen),
                    )
            except Exception as exc:
                return (
                    None,
                    f"Inference error: {exc}",
                    f'<div class="wv-placeholder" style="color:#e11d48;"><b>Error:</b> {html.escape(str(exc))}</div>',
                    None,
                )

            preds = parse_predictions_with_masks(
                result.response, pil.width, pil.height, result.masks
            )
            count = extract_count(result.response, preds, counting=(task == "counting"))
            cache = {
                "pil": np.array(pil.convert("RGB")),
                "preds": preds,
                "masks": result.masks,
                "task": task,
            }
            vis = (
                render_from_cache(
                    cache,
                    font_size=int(f_size),
                    draw_width=int(d_width),
                    show_labels=bool(s_labels),
                    box_color_preset=c_preset,
                    box_color_hex=c_hex,
                    show_boxes=bool(s_boxes),
                    show_masks=bool(s_masks),
                    mask_alpha=float(m_alpha),
                    mask_threshold=float(m_thresh),
                    pixel_display=p_display,
                )
                or pil
            )

            n_inst = sum(len(v) for v in preds.values())
            n_mask = len(result.masks)
            model_tag = getattr(engine, "model_name_or_path", DEFAULT_MODEL_ID)
            header = (
                f"[{model_tag} | {TASK_LABELS.get(task, task)}]  "
                f"latency={result.latency_s:.2f}s  "
                f"instances={n_inst}"
                + (f"  mattes={n_mask}" if is_pixel else "")
                + f"\nPrompt: {result.prompt}\n"
                + ("-" * 56)
                + "\n"
            )

            def stat(label: str, value: Any, hero: bool = False) -> str:
                cls = "wv-stat wv-stat-hero" if hero else "wv-stat"
                return f'<span class="{cls}">{html.escape(label)} <b>{html.escape(str(value))}</b></span>'

            stats: List[str] = []
            if task == "counting":
                stats.append(stat("Count", count if count is not None else "N/A", hero=True))
            if is_pixel:
                stats.append(stat("Masks", n_mask, hero=True))
            if n_inst:
                stats.append(stat("Instances", n_inst, hero=not (is_pixel or task == "counting")))
            stats.append(stat("Latency", f"{result.latency_s:.2f}s"))
            for cat, anns in preds.items():
                r, g, b = _resolve_box_color(c_preset, c_hex, cat)
                stats.append(
                    f'<span class="wv-legend"><i style="background:rgb({r},{g},{b})"></i>'
                    f"{html.escape(cat)} <span>×{len(anns)}</span></span>"
                )
            summary = f'<div class="wv-stats">{"".join(stats)}</div>'

            return vis, header + result.response, summary, cache

        def redraw_from_cache(
            cache: Optional[Dict[str, Any]],
            f_size: int,
            d_width: int,
            s_boxes: bool,
            s_labels: bool,
            c_preset: str,
            c_hex: str,
            s_masks: bool,
            m_alpha: float,
            m_thresh: float,
            p_display: str,
        ):
            vis = render_from_cache(
                cache,
                font_size=int(f_size),
                draw_width=int(d_width),
                show_labels=bool(s_labels),
                box_color_preset=c_preset,
                box_color_hex=c_hex,
                show_boxes=bool(s_boxes),
                show_masks=bool(s_masks),
                mask_alpha=float(m_alpha),
                mask_threshold=float(m_thresh),
                pixel_display=p_display,
            )
            if vis is None:
                return gr.update()
            return vis

        def on_select_example(evt: gr.SelectData):
            idx = evt.index
            if isinstance(idx, (list, tuple)):
                idx = idx[0]
            if idx is None or idx >= len(GALLERY_EXAMPLES):
                return [gr.update() for _ in range(9 + len(ALL_TASKS))]
            ex = GALLERY_EXAMPLES[idx]
            task = ex["task"]
            sub = ex["subtask"]
            q = ex["query"]
            img_path = EXAMPLES_IMAGE_DIR / ex["image"]
            img = Image.open(img_path).convert("RGB") if img_path.is_file() else None
            prompt_text = build_task_prompt(task, sub, q)
            is_pixel = task in PIXEL_TASKS
            return (
                img,
                img,
                task,
                subtask_ui_for(task, sub),
                query_ui_for(task, sub, value=q),
                prompt_text,
                pixel_display_ui_for(task),
                gr.update(visible=is_pixel),
                gr.update(visible=(task == "segmentation")),
                *task_button_updates(task),
            )

        # Wire task buttons
        task_switch_outputs = [
            task_state,
            subtask,
            query,
            prompt_preview,
            pixel_display,
            pixel_viz_group,
            mask_threshold,
            *task_buttons,
        ]
        for _task_name, _btn in zip(ALL_TASKS, task_buttons):
            _btn.click(
                fn=lambda t=_task_name: switch_task(t),
                inputs=[],
                outputs=task_switch_outputs,
            )

        # Use .input (user-only) instead of .change so programmatic task switches
        # never overwrite prompt_preview with stale task_state.
        subtask.input(
            fn=on_subtask_change,
            inputs=[task_state, subtask, query],
            outputs=[query, prompt_preview],
        )
        query.input(
            fn=on_query_change,
            inputs=[task_state, subtask, query],
            outputs=prompt_preview,
        )
        input_image.change(
            fn=on_image_change,
            inputs=input_image,
            outputs=raw_image,
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                input_image,
                raw_image,
                task_state,
                subtask,
                query,
                prompt_preview,
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
                font_size,
                draw_width,
                show_boxes,
                show_labels,
                box_color_preset,
                box_color_hex,
                show_masks,
                mask_alpha,
                mask_threshold,
                pixel_display,
            ],
            outputs=[output_image, raw_out, summary_md, vis_cache],
        )

        viz_controls = [
            font_size,
            draw_width,
            show_boxes,
            show_labels,
            box_color_preset,
            box_color_hex,
            show_masks,
            mask_alpha,
            mask_threshold,
            pixel_display,
        ]
        for _ctrl in viz_controls:
            _ctrl.change(
                fn=redraw_from_cache,
                inputs=[vis_cache, *viz_controls],
                outputs=output_image,
            )

        example_gallery.select(
            fn=on_select_example,
            inputs=[],
            outputs=[
                input_image,
                raw_image,
                task_state,
                subtask,
                query,
                prompt_preview,
                pixel_display,
                pixel_viz_group,
                mask_threshold,
                *task_buttons,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeVisOne-VL Gradio Demo")
    parser.add_argument(
        "--model",
        "--model_path",
        dest="model",
        default=os.environ.get("WEVISONE_MODEL", DEFAULT_MODEL_ID),
    )
    parser.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--server-name", "--server_name", dest="server_name", default="0.0.0.0")
    parser.add_argument(
        "--server-port",
        "--server_port",
        dest="server_port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--mock", action="store_true", help="skip model loading; return mock outputs for UI preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        engine: Any = MockEngine()
    else:
        from wevisone_vl import WeVisOneVL

        engine = WeVisOneVL(args.model, backend=args.backend, device=args.device)

    demo = build_demo(engine)
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(EXAMPLES_IMAGE_DIR)],
        **_style_kwargs("launch"),
    )


if __name__ == "__main__":
    main()
