"""Image matting inference for WeVisOne-VL, with MAD, MSE, Grad, Conn and SAD per sample.

The box that prompts the mask decoder comes from one of two sources:

* ``gt`` (P3M-500-NP, AM-2K-test, PPM-100): the bounding box of the ground-truth
  alpha (pixels above 127) is written into the response
  ``<ref>expr</ref><box><x1><y1><x2><y2><OBJ></box>``, and the unrounded box is
  passed to the decoder (single-instance prompt).
* ``generated`` (WeRefMat): the model predicts all matching boxes and ``<OBJ>``
  tokens from the text (instances prompt), and the predicted mattes are merged
  by pixel-wise max. If the response contains no ``<OBJ>``, the prediction is an
  all-zero alpha.

Annotation format (see ``prepare/matting/prepare_matting.py``)::

    {"sample_id", "dataset", "image_path", "expression",
     "alpha_path" | "mattes": [path | {"path", "alpha_box_xywh"?}], "alpha_box_xywh"?}

``alpha_box_xywh`` gives the position of a cropped alpha in the full image.
Multiple ``mattes`` are merged by pixel-wise max into one ground-truth alpha.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics.matting_metric import compute_matting_metrics  # noqa: E402
from evaluation.utils.prompt_builder import TMPL_MATTE  # noqa: E402
from wevisone_vl.engine import WeVisOneVL, parse_seg_boxes  # noqa: E402

GT_BOX_ALPHA_THRESHOLD = 127


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_jsonl_path", required=True)
    parser.add_argument("--image_root_dir", default="", help="prefix for relative image / alpha paths")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--box_source", choices=("gt", "generated"), required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="generated mode only")
    parser.add_argument("--min_pixels", type=int, default=16 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--batch_size", type=int, default=8, help="images per generate() call")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    return parser.parse_args()


def _read_alpha(path: str, root: str) -> np.ndarray:
    with Image.open(os.path.join(root, path)) as opened:
        return np.asarray(opened.convert("L"), dtype=np.uint8)


def _paste(canvas: np.ndarray, alpha: np.ndarray, box_xywh) -> None:
    if box_xywh is None:
        if alpha.shape != canvas.shape:
            raise ValueError(f"alpha {alpha.shape} vs image {canvas.shape} without alpha_box_xywh")
        np.maximum(canvas, alpha, out=canvas)
        return
    x, y, w, h = (int(round(float(v))) for v in box_xywh)
    if alpha.shape != (h, w) or x < 0 or y < 0 or x + w > canvas.shape[1] or y + h > canvas.shape[0]:
        raise ValueError(f"alpha crop {alpha.shape} does not fit box {box_xywh} in {canvas.shape}")
    region = canvas[y : y + h, x : x + w]
    np.maximum(region, alpha, out=region)


def read_sample(entry: dict, root: str) -> tuple[Image.Image, np.ndarray]:
    with Image.open(os.path.join(root, entry["image_path"])) as opened:
        image = opened.convert("RGB")
    alpha = np.zeros((image.height, image.width), dtype=np.uint8)
    if entry.get("mattes"):
        for matte in entry["mattes"]:
            if isinstance(matte, str):
                matte = {"path": matte}
            _paste(alpha, _read_alpha(matte["path"], root), matte.get("alpha_box_xywh"))
    else:
        _paste(alpha, _read_alpha(entry["alpha_path"], root), entry.get("alpha_box_xywh"))
    return image, alpha


def tight_alpha_box(alpha: np.ndarray) -> list[float]:
    """Normalized xyxy box of alpha > 127 (Matting-Anything convention)."""
    ys, xs = np.where(alpha > GT_BOX_ALPHA_THRESHOLD)
    if not len(xs):
        return [0.0, 0.0, 1.0, 1.0]
    h, w = alpha.shape
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


def box_response(expression: str, box: list[float]) -> str:
    coords = "".join(f"<{max(0, min(1000, round(v * 1000)))}>" for v in box)
    return f"<ref>{expression}</ref><box>{coords}<OBJ></box>"


def main() -> None:
    args = get_args()
    with open(args.test_jsonl_path, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    end = len(entries) if args.end_idx < 0 else min(args.end_idx, len(entries))
    entries = entries[args.start_idx : end]
    engine = WeVisOneVL(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attention_implementation=args.attn_implementation,
    )
    gt_boxes = args.box_source == "gt"
    template = TMPL_MATTE["single" if gt_boxes else "instances"]
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as output:
        for start in tqdm(range(0, len(entries), args.batch_size), desc="matting"):
            batch = entries[start : start + args.batch_size]
            samples = [read_sample(entry, args.image_root_dir) for entry in batch]
            prompts = [template.format(expression=entry["expression"]) for entry in batch]
            if gt_boxes:
                forced = [[tight_alpha_box(alpha)] for _, alpha in samples]
                responses = [box_response(e["expression"], b[0]) for e, b in zip(batch, forced)]
            else:
                forced = [None] * len(batch)
                responses = engine.generate_text_batch(
                    [image for image, _ in samples],
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )

            for entry, (image, alpha), prompt, response, boxes in zip(batch, samples, prompts, responses, forced):
                masks = []
                if boxes is not None or parse_seg_boxes(response):
                    boxes, masks = engine.decode_masks(image, prompt, response, boxes=boxes)
                if not masks:
                    prediction = np.zeros(alpha.shape, dtype=np.float32)
                elif gt_boxes:
                    prediction = masks[0]
                else:
                    prediction = np.max(np.stack(masks), axis=0)
                row = {
                    "dataset": entry["dataset"],
                    "sample_id": entry["sample_id"],
                    "image_path": entry["image_path"],
                    "expression": entry["expression"],
                    "box_source": args.box_source,
                    "response": response,
                    "pred_boxes01": boxes or [],
                    "valid_prediction": bool(masks),
                    "metrics": compute_matting_metrics(prediction, alpha),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
