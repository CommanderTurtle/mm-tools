from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _inside(root: Path, value: str | Path, base: Path | None = None) -> Path | None:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base or root) / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if candidate.is_file() and root in candidate.parents else None


def _asset_path(root: Path, parse_root: Path, element: dict[str, Any]) -> str | None:
    for key in (
        "rendered_image_path",
        "extracted_image_uri",
        "extracted_image_path",
        "image_path",
        "canvas_image_uri",
    ):
        value = element.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidate = _inside(root, value, parse_root)
        if candidate:
            return candidate.relative_to(root).as_posix()
    return None


def build_editor_document(
    *, root: Path, parse_path: Path, canvas_width: int, canvas_height: int
) -> dict[str, Any]:
    raw = json.loads(parse_path.read_text(encoding="utf-8"))
    elements = raw.get("elements", raw) if isinstance(raw, dict) else raw
    if not isinstance(elements, list):
        elements = []
    layers: list[dict[str, Any]] = []
    for index, raw_element in enumerate(elements):
        if not isinstance(raw_element, dict):
            continue
        element = dict(raw_element)
        bbox = element.get("bbox", [0, 0, canvas_width, canvas_height])
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, canvas_width, canvas_height]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        layer_type = str(element.get("type") or element.get("action_type") or "image")
        layer_id = str(element.get("id") or f"layer-{index + 1}")
        font_color = element.get("font_color") or "#ffffff"
        if isinstance(font_color, dict):
            font_color = font_color.get("hex") or "#ffffff"
        layers.append(
            {
                "id": layer_id,
                "name": str(element.get("name") or element.get("label") or layer_id),
                "type": layer_type,
                "visible": True,
                "locked": False,
                "opacity": 1.0,
                "z": index,
                "x": x1,
                "y": y1,
                "width": max(1.0, x2 - x1),
                "height": max(1.0, y2 - y1),
                "rotation": float(element.get("angle_deg", element.get("angle", 0)) or 0),
                "asset_path": _asset_path(root, parse_path.parent, element),
                "text": {
                    "content": str(element.get("content") or ""),
                    "font_family": str(element.get("font_family") or "sans-serif"),
                    "font_size": float(element.get("font_size_px") or max(12, y2 - y1)),
                    "color": str(font_color),
                    "bold": bool(element.get("font_bold", element.get("bold", False))),
                    "italic": bool(element.get("font_italic", element.get("italic", False))),
                    "font_file_path": str(element.get("font_file_path") or ""),
                },
                "source": element,
            }
        )
    return {
        "version": 1,
        "canvas": {"width": canvas_width, "height": canvas_height, "background": "transparent"},
        "layers": layers,
        "source": parse_path.relative_to(root).as_posix(),
    }


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result)) if math.isfinite(result) else default


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    canvas = document.get("canvas", {})
    width = int(_number(canvas.get("width"), 1024, 1, 16384))
    height = int(_number(canvas.get("height"), 1024, 1, 16384))
    raw_layers = document.get("layers", [])
    if not isinstance(raw_layers, list) or len(raw_layers) > 1000:
        raise ValueError("Editable documents may contain at most 1,000 layers.")
    layers = []
    for index, value in enumerate(raw_layers):
        if not isinstance(value, dict):
            continue
        layer = dict(value)
        layer["id"] = re.sub(r"[^A-Za-z0-9_.-]", "-", str(layer.get("id") or f"layer-{index}"))[:120]
        layer["name"] = str(layer.get("name") or layer["id"])[:240]
        layer["visible"] = bool(layer.get("visible", True))
        layer["locked"] = bool(layer.get("locked", False))
        layer["opacity"] = _number(layer.get("opacity"), 1, 0, 1)
        layer["z"] = int(_number(layer.get("z"), index, -100000, 100000))
        layer["x"] = _number(layer.get("x"), 0, -32768, 32768)
        layer["y"] = _number(layer.get("y"), 0, -32768, 32768)
        layer["width"] = _number(layer.get("width"), 1, 1, 32768)
        layer["height"] = _number(layer.get("height"), 1, 1, 32768)
        layer["rotation"] = _number(layer.get("rotation"), 0, -36000, 36000)
        asset_path = layer.get("asset_path")
        layer["asset_path"] = str(asset_path) if asset_path else None
        text = layer.get("text", {}) if isinstance(layer.get("text"), dict) else {}
        layer["text"] = {
            "content": str(text.get("content") or "")[:100000],
            "font_family": str(text.get("font_family") or "sans-serif")[:240],
            "font_size": _number(text.get("font_size"), 16, 1, 4096),
            "color": str(text.get("color") or "#ffffff")[:64],
            "bold": bool(text.get("bold", False)),
            "italic": bool(text.get("italic", False)),
            "font_file_path": str(text.get("font_file_path") or "")[:4096],
        }
        layers.append(layer)
    return {
        "version": 1,
        "canvas": {"width": width, "height": height, "background": "transparent"},
        "layers": layers,
        "source": str(document.get("source") or ""),
    }


def _font(root: Path, text: dict[str, Any]) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(1, int(text.get("font_size", 16)))
    font_path = text.get("font_file_path")
    if font_path:
        candidate = _inside(root, str(font_path), root)
        if candidate:
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    for fallback in ("DejaVuSans-Bold.ttf" if text.get("bold") else "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(fallback, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_layer(root: Path, canvas_size: tuple[int, int], layer: dict[str, Any]) -> Image.Image:
    width = max(1, int(round(layer["width"])))
    height = max(1, int(round(layer["height"])))
    if str(layer.get("type", "")).lower() == "text":
        piece = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(piece)
        draw.multiline_text(
            (0, 0),
            layer["text"]["content"],
            fill=layer["text"]["color"],
            font=_font(root, layer["text"]),
            spacing=max(1, int(layer["text"]["font_size"] * 0.2)),
        )
    else:
        source = _inside(root, layer.get("asset_path") or "", root)
        if source:
            with Image.open(source) as opened:
                piece = opened.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
        else:
            piece = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if layer["opacity"] < 1:
        alpha = piece.getchannel("A").point(lambda value: round(value * layer["opacity"]))
        piece.putalpha(alpha)
    rotation = float(layer.get("rotation", 0))
    if rotation:
        piece = piece.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    x = int(round(layer["x"] - (piece.width - width) / 2))
    y = int(round(layer["y"] - (piece.height - height) / 2))
    canvas.alpha_composite(piece, (x, y))
    return canvas


def export_document(root: Path, document: dict[str, Any]) -> dict[str, str]:
    document = validate_document(document)
    canvas_size = (document["canvas"]["width"], document["canvas"]["height"])
    export_root = root / "editable_export"
    layers_root = export_root / "layers"
    if export_root.exists():
        shutil.rmtree(export_root)
    layers_root.mkdir(parents=True)
    (root / "editable.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    composite = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    for index, layer in enumerate(sorted(document["layers"], key=lambda item: item["z"])):
        if not layer["visible"]:
            continue
        rendered = _render_layer(root, canvas_size, layer)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", layer["name"]).strip(".-") or layer["id"]
        rendered.save(layers_root / f"{index:04d}-{safe_name[:80]}.png")
        composite = Image.alpha_composite(composite, rendered)
    composite_path = export_root / "composite.png"
    composite.save(composite_path)
    package_path = root / "editable-layer-package.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.write(root / "editable.json", "editable.json")
        package.write(composite_path, "composite.png")
        for path in sorted(layers_root.glob("*.png")):
            package.write(path, f"layers/{path.name}")
    return {
        "document": "editable.json",
        "composite": composite_path.relative_to(root).as_posix(),
        "package": package_path.relative_to(root).as_posix(),
    }
