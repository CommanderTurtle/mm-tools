"""Versioned presentation-render protocol loading for unified rendering."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

RENDER_PROTOCOL_DIR = Path(__file__).resolve().parent / "render_protocols"
RENDER_PROTOCOL_SCHEMA = "ff_unified_render_protocol_v1"
BASELINE_RENDER_PROTOCOL = "render_sep5_v0"
RENDER_PROTOCOLS = ("geometry", "texture")


def available_render_protocols() -> tuple[str, ...]:
    return tuple(path.stem for path in sorted(RENDER_PROTOCOL_DIR.glob("*.json")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_path(identifier: str | Path, relative_to: Path | None = None) -> Path:
    value = str(identifier)
    candidate = Path(value)
    filename = candidate.name if candidate.suffix else f"{candidate.name}.json"
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = []
        if relative_to is not None:
            candidates.append(relative_to / filename)
        candidates.extend((candidate, RENDER_PROTOCOL_DIR / filename))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ValueError(
        f"Unknown render protocol {value!r}; choose one of "
        f"{available_render_protocols()} or pass a JSON path"
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_render_protocol(source: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if source in stack:
        cycle = " -> ".join(path.name for path in (*stack, source))
        raise ValueError(f"Render protocol inheritance cycle: {cycle}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    parent = payload.get("extends")
    if parent is not None:
        parent_source = _protocol_path(parent, source.parent)
        inherited = _load_render_protocol(parent_source, (*stack, source))
        payload = _deep_merge(inherited, payload)
    validate_render_protocol(payload, source)
    return payload


def load_render_protocol(identifier: str | Path) -> tuple[dict[str, Any], Path]:
    source = _protocol_path(identifier)
    return _load_render_protocol(source, ()), source


def validate_render_protocol(
    protocol: dict[str, Any], source: Path | None = None
) -> None:
    location = f" in {source}" if source is not None else ""
    if protocol.get("schema") != RENDER_PROTOCOL_SCHEMA:
        raise ValueError(f"Unsupported render protocol schema{location}")
    if not protocol.get("name"):
        raise ValueError(f"Render protocol has no name{location}")

    render = protocol.get("render")
    if not isinstance(render, dict):
        raise ValueError(f"Render protocol has no render object{location}")
    protocols = render.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError(f"render.protocols must be a nonempty list{location}")
    unknown = sorted(set(protocols) - set(RENDER_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown render outputs {unknown}{location}")
    if render.get("engine") != "BLENDER_EEVEE_NEXT":
        raise ValueError(f"Only BLENDER_EEVEE_NEXT is currently supported{location}")
    if not isinstance(render.get("samples"), int) or render["samples"] <= 0:
        raise ValueError(f"render.samples must be a positive integer{location}")

    style = render.get("style")
    if not isinstance(style, dict) or not style.get("name"):
        raise ValueError(f"render.style must name a style{location}")
    applies_to = style.get("applies_to")
    if not isinstance(applies_to, list):
        raise ValueError(f"render.style.applies_to must be a list{location}")
    unknown = sorted(set(applies_to) - set(RENDER_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown style targets {unknown}{location}")
    if not isinstance(style.get("parameters"), dict):
        raise ValueError(f"render.style.parameters must be an object{location}")

    background = render.get("background")
    required_backgrounds = (
        "gt_geometry",
        "gt_texture",
        "reconstruction_geometry",
        "reconstruction_texture",
    )
    if not isinstance(background, dict) or any(
        not isinstance(background.get(key), bool) for key in required_backgrounds
    ):
        raise ValueError(
            f"render.background must define boolean {required_backgrounds}{location}"
        )

    geometry = render.get("geometry_material")
    if not isinstance(geometry, dict) or geometry.get("palette") not in {
        "legacy",
        "diverse",
    }:
        raise ValueError(f"Unsupported geometry material palette{location}")
    gray = geometry.get("background_rgba")
    if not isinstance(gray, list) or len(gray) != 4:
        raise ValueError(f"geometry background_rgba must have four values{location}")

    output = render.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"render.output must be an object{location}")
    expected_output = {
        "file_format": "PNG",
        "color_mode": "RGB",
        "color_depth": "8",
        "film_transparent": False,
    }
    mismatched = {
        key: (output.get(key), expected)
        for key, expected in expected_output.items()
        if output.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Unsupported render output settings {mismatched}{location}")
    empty = output.get("empty_background_rgb")
    if not isinstance(empty, list) or len(empty) != 3:
        raise ValueError(f"empty_background_rgb must have three values{location}")

    camera = render.get("camera")
    if not isinstance(camera, dict) or camera.get("type") != "PERSP":
        raise ValueError(f"Only perspective cameras are currently supported{location}")
    if float(camera.get("clip_start", 0.0)) <= 0.0:
        raise ValueError(f"camera.clip_start must be positive{location}")
    if float(camera.get("clip_end", 0.0)) <= float(camera["clip_start"]):
        raise ValueError(f"camera.clip_end must exceed clip_start{location}")


def validate_render_scope(protocol: dict[str, Any], dataset: str, source: str) -> None:
    scope = protocol.get("scope") or {}
    datasets = scope.get("datasets")
    if datasets and dataset not in datasets:
        raise ValueError(
            f"Render protocol {protocol['name']} supports datasets {datasets!r}, "
            f"not {dataset!r}"
        )
    sources = scope.get("sources")
    if sources and source not in sources:
        raise ValueError(
            f"Render protocol {protocol['name']} supports sources {sources!r}, "
            f"not {source!r}"
        )


def render_protocol_config(protocol: dict[str, Any]) -> dict[str, Any]:
    """Translate a render protocol into the unified renderer's config shape."""

    validate_render_protocol(protocol)
    render = protocol["render"]
    background = render["background"]
    geometry = render["geometry_material"]
    output = render["output"]
    style = render["style"]
    return {
        "protocols": list(render["protocols"]),
        "render": {
            "engine": render["engine"],
            "recipe": render["recipe"],
            "render_profile": render["render_profile"],
            "samples": int(render["samples"]),
            "background": bool(background["reconstruction_texture"]),
            "gt_geometry_background": bool(background["gt_geometry"]),
            "gt_texture_background": bool(background["gt_texture"]),
            "reconstruction_geometry_background": bool(
                background["reconstruction_geometry"]
            ),
            "reconstruction_texture_background": bool(
                background["reconstruction_texture"]
            ),
            "geometry_palette": geometry["palette"],
            "gray_rgba": list(geometry["background_rgba"]),
            "style": style["name"],
            "style_parameters": dict(style["parameters"]),
            "style_applies_to": list(style["applies_to"]),
            "camera": dict(render["camera"]),
            "output": dict(output),
            "empty_background_rgb": list(output["empty_background_rgb"]),
        },
    }


def render_protocol_identity(
    requested: str | Path, protocol: dict[str, Any], source: Path
) -> dict[str, Any]:
    resolved = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "requested_name": str(requested),
        "resolved_name": protocol["name"],
        "source": str(source),
        "source_sha256": _sha256(source),
        "resolved_sha256": hashlib.sha256(resolved).hexdigest(),
        "extends": protocol.get("extends"),
        "schema": protocol["schema"],
        "version": protocol.get("version"),
        "status": protocol.get("status"),
    }
