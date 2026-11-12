"""Task prompts used by the released WeVisOne-VL checkpoint."""

from __future__ import annotations

from typing import Iterable, Sequence


TASKS = (
    "detection",
    "referring",
    "point",
    "count_box",
    "count_point",
    "segmentation",
    "matting",
    "ocr",
    "ocr_grounding",
    "visual_prompt",
    "freeform",
)


def _labels(query: str | Iterable[str]) -> list[str]:
    if isinstance(query, str):
        return [item.strip() for item in query.splitlines() if item.strip()]
    return [str(item).strip() for item in query if str(item).strip()]


def _text(query: str | Iterable[str]) -> str:
    if isinstance(query, str):
        return query.strip()
    return " ".join(str(item).strip() for item in query if str(item).strip())


def _phrase(query: str | Iterable[str]) -> str:
    text = _text(query)
    return text if not text or text[-1] in ".?!" else text + "."


def _box_tokens(boxes: Sequence[Sequence[float]], width: int, height: int) -> str:
    encoded = []
    for box in boxes:
        if len(box) < 4 or width <= 0 or height <= 0:
            continue
        values = (
            float(box[0]) / width,
            float(box[1]) / height,
            float(box[2]) / width,
            float(box[3]) / height,
        )
        bins = [max(0, min(1000, int(value * 1000))) for value in values]
        encoded.append("".join(f"<{value}>" for value in bins))
    return ",".join(encoded)


def build_prompt(
    task: str,
    query: str | Iterable[str] = "",
    *,
    reference_boxes: Sequence[Sequence[float]] | None = None,
    image_size: tuple[int, int] = (0, 0),
) -> str:
    """Build the exact instruction family used for checkpoint inference.

    Multiple detection labels can be passed as a list of strings
    or separated by newlines. A referring expression remains a single phrase
    even when it contains commas. Reference boxes are pixel-space
    ``[x1, y1, x2, y2]`` coordinates.
    """
    task = task.strip().lower()
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(TASKS)}")
    labels = _labels(query)
    text_query = _text(query)

    if task == "detection":
        return "Locate all the instances that match the following description: " + ". ".join(labels) + "."
    if task == "referring":
        return "Locate the instances that match the following description: " + _phrase(query)
    if task == "point":
        return "point to: " + ". ".join(labels)
    if task == "count_box":
        return f"How many {text_query or 'objects'} in the image? Please count them with boxes."
    if task == "count_point":
        return f"How many {text_query or 'objects'} in the image?"
    if task in {"segmentation", "matting"}:
        return (
            "Locate the instances that match the following description and "
            "provide the alpha mattes: " + _phrase(query)
        )
    if task == "ocr":
        return "detect all the word in the box format"
    if task == "ocr_grounding":
        return "Locate the following text in the image: " + text_query
    if task == "visual_prompt":
        width, height = image_size
        coords = _box_tokens(reference_boxes or [], width, height)
        visual_prompt = f"<ref>object 1</ref><box>{coords}</box>"
        return (
            "Locate all similar objects in the image based on the reference boxes: "
            + visual_prompt
        )
    return text_query
