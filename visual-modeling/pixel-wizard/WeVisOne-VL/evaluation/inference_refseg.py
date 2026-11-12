"""Referring segmentation inference for WeVisOne-VL (RefCOCO/+/g, ReasonSeg, Ref-ZOM).

The model generates ``<ref>..</ref><box>..<OBJ></box>`` with greedy decoding.
The response is then fed back to decode one probability map for each ``<OBJ>``.
The maps are resized to the original image size, merged by pixel-wise max and
thresholded at 0.5. For each expression, the intersection and union with the
ground truth (excluding ``ignore_rle`` pixels) are saved for
``metrics/refseg_metric.py``.

Annotation format (see ``prepare/refseg/prepare_refseg.py``)::

    {"id", "image_path", "expression", "gt_rle", "split",
     "ignore_rle"?, "source_type"?, "is_sentence"?}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.utils.prompt_builder import TMPL_MATTE  # noqa: E402
from wevisone_vl.engine import WeVisOneVL, parse_seg_boxes  # noqa: E402

THRESHOLD = 0.5


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_jsonl_path", required=True)
    parser.add_argument("--image_root_dir", default="")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument(
        "--prompt_style",
        choices=tuple(TMPL_MATTE),
        default="single",
        help="instances for Ref-ZOM (zero / one / many targets), single otherwise",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_pixels", type=int, default=16 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=2025 * 32 * 32)
    parser.add_argument("--batch_size", type=int, default=16, help="images per generate() call")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--save_masks", action="store_true", help="store the predicted mask as COCO RLE")
    return parser.parse_args()


def decode_rle(rle: dict) -> np.ndarray:
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    mask = coco_mask.decode(rle)
    return np.asarray(mask.any(axis=2) if mask.ndim == 3 else mask, dtype=bool)


def mask_metrics(pred: np.ndarray | None, gt: np.ndarray, ignore: np.ndarray | None) -> dict:
    valid = np.ones_like(gt) if ignore is None else ~ignore
    gt = gt & valid
    pred = np.zeros_like(gt) if pred is None else pred & valid
    intersection = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return {
        "intersection": intersection,
        "union": union,
        "iou": intersection / union if union else 1.0,
        "pred_area": int(pred.sum()),
        "gt_area": int(gt.sum()),
    }


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
    template = TMPL_MATTE[args.prompt_style]
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as output:
        for start in tqdm(range(0, len(entries), args.batch_size), desc="refseg"):
            batch = []
            for entry in entries[start : start + args.batch_size]:
                gt = decode_rle(entry["gt_rle"])
                ignore = decode_rle(entry["ignore_rle"]) if entry.get("ignore_rle") else None
                try:
                    with Image.open(os.path.join(args.image_root_dir, entry["image_path"])) as opened:
                        image = opened.convert("RGB")
                    if gt.shape != (image.height, image.width):
                        raise ValueError(f"mask {gt.shape} vs image {(image.height, image.width)}")
                    batch.append((entry, gt, ignore, image, None))
                except Exception as exc:
                    batch.append((entry, gt, ignore, None, f"{type(exc).__name__}: {exc}"))

            ready = [item for item in batch if item[3] is not None]
            prompts = [template.format(expression=item[0]["expression"]) for item in ready]
            responses = engine.generate_text_batch(
                [item[3] for item in ready],
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.0,
            )
            answers = {id(item[0]): (p, r) for item, p, r in zip(ready, prompts, responses)}

            for entry, gt, ignore, image, error in batch:
                prompt, response = answers.get(id(entry), ("", ""))
                boxes, pred = [], None
                if image is not None and parse_seg_boxes(response):
                    boxes, masks = engine.decode_masks(image, prompt, response)
                    if masks:
                        pred = np.max(np.stack(masks), axis=0) >= THRESHOLD
                is_sentence = entry.get("is_sentence")
                row = {
                    "id": entry.get("id"),
                    "image": entry["image_path"],
                    "split": entry.get("split", "unknown"),
                    "source_type": entry.get("source_type"),
                    "query_type": {True: "long", False: "short"}.get(is_sentence),
                    "expression": entry["expression"],
                    "response": response,
                    "pred_boxes01": boxes,
                    "valid_prediction": pred is not None or entry.get("source_type") == "zero",
                    **mask_metrics(pred, gt, ignore),
                }
                if error:
                    row.update(valid_prediction=False, error=error)
                if args.save_masks and pred is not None:
                    rle = coco_mask.encode(np.asfortranarray(pred.astype(np.uint8)))
                    row["pred_rle"] = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
