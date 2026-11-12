"""Command-line inference entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from PIL import Image

from .engine import DEFAULT_MODEL_ID, WeVisOneVL
from .prompts import TASKS
from .visualization import save_result

logger = logging.getLogger(__name__)

EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeVisOne-VL image inference")
    parser.add_argument("--model", default=os.environ.get("WEVISONE_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", choices=TASKS, default="freeform")
    parser.add_argument("--query", default="")
    parser.add_argument("--prompt", help="Use a raw prompt instead of the task template")
    parser.add_argument(
        "--reference-boxes",
        type=json.loads,
        default=None,
        help='Pixel-space JSON boxes, e.g. "[[10,20,100,200]]"',
    )
    parser.add_argument("--output-dir", default="outputs/prediction")
    parser.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--decoder-batch-size", type=int, default=16)
    parser.add_argument("--revision")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    """Check user-supplied values before any heavy work starts.

    Returns the resolved image path. Raises ``ValueError`` with a message that is
    safe to show to the user.
    """
    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        raise ValueError(f"--image is not an existing file: {image_path}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    if args.decoder_batch_size <= 0:
        raise ValueError(f"--decoder-batch-size must be positive, got {args.decoder_batch_size}")
    if args.temperature < 0:
        raise ValueError(f"--temperature must be non-negative, got {args.temperature}")
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir exists and is not a directory: {output_dir}")
    return image_path


def _load_image(path: Path) -> Image.Image:
    # ``convert`` returns a fully loaded copy, so the file handle can be closed.
    with Image.open(path) as source:
        return source.convert("RGB")


def main() -> int:
    """Run inference and return a process exit code (0 on success)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        image_path = validate_args(args)
    except ValueError as exc:
        logger.error("Invalid arguments: %s", exc)
        return EXIT_USAGE

    try:
        image = _load_image(image_path)
    except (OSError, ValueError) as exc:  # includes UnidentifiedImageError
        logger.error("Cannot read image %s: %s", image_path, exc)
        logger.debug("Image load failure", exc_info=True)
        return EXIT_FAILURE

    try:
        engine = WeVisOneVL(
            args.model,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            decoder_batch_size=args.decoder_batch_size,
        )
    except KeyboardInterrupt:
        logger.error("Interrupted while loading the model")
        return EXIT_INTERRUPTED
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.exception("Failed to initialize model %r: %s", args.model, exc)
        return EXIT_FAILURE

    try:
        result = engine.predict(
            image,
            task=args.task,
            query=args.query,
            prompt=args.prompt,
            reference_boxes=args.reference_boxes,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except KeyboardInterrupt:
        logger.error("Interrupted during inference")
        return EXIT_INTERRUPTED
    except (OSError, RuntimeError, ValueError) as exc:  # includes CUDA out-of-memory
        logger.exception("Inference failed: %s", exc)
        return EXIT_FAILURE

    try:
        paths = save_result(args.output_dir, image, result)
    except OSError as exc:
        logger.exception("Cannot write results to %s: %s", args.output_dir, exc)
        # The prediction itself succeeded, so still show it.
        print(result.response)
        return EXIT_FAILURE

    print(result.response)
    print(f"\nSaved results to {paths['metadata'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
