"""Parse WeVisOne-VL answers with coordinate tokens into pixel coordinates.

The model answers with Rex-Omni style tags::

    <ref>person</ref><box><12><40><380><990>,<500><42><880><991></box>
    <ref>cup</ref><point><512><640></point>

Coordinates are bins from ``<0>`` to ``<999>``, converted to pixels as
``bin / 1000 * width`` (or height). One tag may contain several shapes
separated by commas, and the closing tag may be missing if generation is
truncated.
"""

from __future__ import annotations

import re

REF_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
    re.DOTALL,
)
BARE_RE = re.compile(
    r"<(box|point|polygon)>(.*?)(?:</\1>|<(?:box|point|polygon)>|$)",
    re.DOTALL,
)
FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)
BIN_RE = re.compile(r"<(\d+)>")


def parse_prediction(text: str, width: int, height: int, *, pointing: bool = False) -> dict:
    """Return ``{category: [shape, ...]}`` in absolute pixels.

    Boxes are ``[x0, y0, x1, y1]`` (degenerate boxes are dropped), points are
    ``[x, y]`` and polygons are flat ``[x0, y0, x1, y1, ...]`` lists. A ``<box>``
    answer stays a four-number box in pointing mode, so it never earns point
    credit. ``pointing`` additionally accepts bare tags without ``<ref>``.
    """
    text = text.split("<|im_end|>")[0]
    result: dict[str, list] = {}
    matches = REF_RE.findall(text)
    if matches:
        for category, shape, body in matches:
            _add_shapes(result, category.strip(), shape, body, width, height)
        return result
    if pointing:
        fence = FENCE_RE.search(text)
        if fence:
            text = fence.group(1)
        for shape, body in BARE_RE.findall(text):
            _add_shapes(result, "", shape, body, width, height)
    return result


def _add_shapes(
    result: dict[str, list], category: str, shape: str, body: str, width: int, height: int
) -> None:
    shapes = []
    for segment in body.strip().rstrip(",").split(","):
        values = [int(value) for value in BIN_RE.findall(segment)]
        if shape == "point":
            for k in range(0, len(values) - 1, 2):
                shapes.append([values[k] / 1000.0 * width, values[k + 1] / 1000.0 * height])
        elif shape == "polygon":
            polygon = []
            for k in range(0, len(values) - 1, 2):
                polygon += [values[k] / 1000.0 * width, values[k + 1] / 1000.0 * height]
            if polygon:
                shapes.append(polygon)
        else:
            for k in range(0, len(values) - 3, 4):
                x0, y0, x1, y1 = values[k : k + 4]
                if x1 > x0 and y1 > y0:
                    shapes.append(
                        [
                            x0 / 1000.0 * width,
                            y0 / 1000.0 * height,
                            x1 / 1000.0 * width,
                            y1 / 1000.0 * height,
                        ]
                    )
    result.setdefault(category, []).extend(shapes)
