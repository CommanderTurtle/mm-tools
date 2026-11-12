#!/usr/bin/env python3
"""Open-vocabulary object detection example using WeVisOne-VL."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image

from wevisone_vl import WeVisOneVL
from wevisone_vl.visualization import save_result

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_path = os.environ.get("WEVISONE_MODEL", "tencent/WeVisOne-VL-4B")
    examples_dir = Path(__file__).resolve().parent
    image_path = examples_dir / "images" / "livingroom.jpg"
    output_dir = Path("outputs/detection_livingroom")

    # Load the image first so a bad path fails before the expensive model load.
    if not image_path.is_file():
        logger.error("Example image not found: %s", image_path)
        return 1
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        logger.error("Cannot read image %s: %s", image_path, exc)
        return 1

    try:
        model = WeVisOneVL(
            model_path,
            backend="transformers",  # or "vllm" for faster inference
            device="auto",
            dtype="bfloat16",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.exception("Failed to initialize model %r: %s", model_path, exc)
        return 1

    categories = [
        "sofa",
        "pillow",
        "coffee table",
        "book",
        "lamp",
        "vase",
    ]
    try:
        result = model.predict(
            image,
            task="detection",
            query=categories,
            max_new_tokens=4096,
            temperature=0.0,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Inference failed: %s", exc)
        return 1

    print(f"Raw response: {result.response}")
    try:
        paths = save_result(output_dir, image, result)
    except OSError as exc:
        logger.exception("Cannot write results to %s: %s", output_dir, exc)
        return 1
    print(f"Saved visualization to: {paths['overlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
