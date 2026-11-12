"""Convert the referring segmentation benchmarks to the ``refseg`` format.

The script reads the downloaded releases from ``--raw_root/<NAME>`` (see
README.md) and writes ``--eval_root/_annotations/refseg/<DATASET>.jsonl``.
RefCOCO/+/g and Ref-ZOM use the COCO 2014 images in
``--eval_root/coco/{train2014,val2014}``. For ReasonSeg, ``--eval_root/ReasonSeg``
is linked to the release folder.

Each row has the format ``{id, image_path, expression, gt_rle, split,
dataset_name}``. ReasonSeg additionally contains ``ignore_rle`` and
``is_sentence``, and ReasonSeg and Ref-ZOM contain ``source_type``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import pickletools
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.prepare.common import json_rle, link_directory, rle_to_json, write_jsonl

REFCOCO = {
    "RefCOCO_seg": ("refcoco", ("val", "testA", "testB")),
    "RefCOCO+_seg": ("refcoco+", ("val", "testA", "testB")),
    "RefCOCOg_seg": ("refcocog", ("val", "test")),
}


def refcoco(dataset: str):
    stem, splits = REFCOCO[dataset]

    def convert(raw: Path, eval_root: Path) -> Iterable[dict]:
        for split in splits:
            with (raw / "RefCOCO" / f"{stem}_{split}.json").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    (target,) = record["objects"]
                    yield {
                        "id": record["id"],
                        "image_path": f"coco/train2014/{record['image']}",
                        "expression": str(target["label"]).strip(),
                        "gt_rle": target["rle"],
                        "split": split,
                        "dataset_name": dataset,
                    }
    return convert


def _rasterize(annotation: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Target/ignore masks exactly as the LISA ReasonSeg evaluator draws them.

    Polygons are painted from largest to smallest; labels containing "ignore" paint
    ignore pixels and "flag" shapes are skipped.
    """
    import cv2

    polygons = []
    for shape in annotation.get("shapes") or []:
        label = str(shape.get("label") or "")
        points = np.asarray(shape.get("points") or [], dtype=np.int32)
        if label.lower() == "flag" or points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(canvas, [points], True, 1, 1)
        cv2.fillPoly(canvas, [points], 1)
        polygons.append((int(canvas.sum()), label, points))
    categorical = np.zeros((height, width), dtype=np.uint8)
    for index in np.argsort([area for area, _, _ in polygons])[::-1]:
        _, label, points = polygons[int(index)]
        value = 255 if "ignore" in label.lower() else 1
        cv2.polylines(categorical, [points], True, value, 1)
        cv2.fillPoly(categorical, [points], value)
    return categorical == 1, categorical == 255


def reasonseg(split: str, dataset: str):
    def convert(raw: Path, eval_root: Path) -> Iterable[dict]:
        from PIL import Image

        link_directory(eval_root / "ReasonSeg", raw / "ReasonSeg")
        for annotation_path in sorted((raw / "ReasonSeg" / split).glob("*.json")):
            image_path = annotation_path.with_suffix(".jpg")
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                annotation = json.loads(annotation_path.read_text(encoding="cp1252"))
            text = " ".join(str(annotation["text"][0]).split())
            is_sentence = bool(annotation.get("is_sentence"))
            with Image.open(image_path) as image:
                target, ignore = _rasterize(annotation, *image.size)
            yield {
                "id": f"reasonseg-{split}-{annotation_path.stem}",
                "image_path": f"ReasonSeg/{split}/{image_path.name}",
                "expression": text if is_sentence else f"What is {text} in this image?",
                "gt_rle": json_rle(target),
                "split": split,
                "dataset_name": dataset,
                "ignore_rle": json_rle(ignore),
                "source_type": "reasoning",
                "is_sentence": is_sentence,
            }
    return convert


def _load_primitive_pickle(path: Path) -> Any:
    """Load a pickle only if it contains plain data (no class/function opcodes)."""
    forbidden = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX",
                 "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}
    with path.open("rb") as stream:
        for operation, _, position in pickletools.genops(stream):
            if operation.name in forbidden:
                raise ValueError(f"unsafe opcode {operation.name} at byte {position} in {path}")
    with path.open("rb") as stream:
        return pickle.load(stream)


def refzom(raw: Path, eval_root: Path) -> Iterable[dict]:
    """One row per test sentence; all referred instances merged into one mask (empty for "zero")."""
    from pycocotools import mask as mask_utils

    source = raw / "Ref-ZOM"
    refs = [ref for ref in _load_primitive_pickle(source / "Ref_ZOM.p") if ref.get("split") == "test"]
    instances = json.loads((source / "instances.json").read_text(encoding="utf-8"))
    annotations = {int(item["id"]): item for item in instances["annotations"]}
    images = {int(item["id"]): item for item in instances["images"]}
    for ref in sorted(refs, key=lambda ref: int(ref["ref_id"])):
        image = images[int(ref["image_id"])]
        height, width = int(image["height"]), int(image["width"])
        file_name = str(ref.get("file_name") or image["file_name"])
        coco_split = file_name.split("_", 2)[1]
        ann_ids = ref.get("ann_id") or []
        rles = []
        for ann_id in ann_ids if isinstance(ann_ids, list) else [ann_ids]:
            segmentation = annotations[int(ann_id)]["segmentation"]
            if isinstance(segmentation, list) or isinstance(segmentation.get("counts"), list):
                encoded = mask_utils.frPyObjects(segmentation, height, width)
                rles.extend(encoded if isinstance(encoded, list) else [encoded])
            else:
                rles.append(segmentation)
        if rles:
            gt_rle = rle_to_json(mask_utils.merge(rles, intersect=False))
        else:
            gt_rle = json_rle(np.zeros((height, width), dtype=np.uint8))
        for sentence in ref.get("sentences") or []:
            yield {
                "id": f"ref-zom-test-{int(sentence['sent_id']):06d}",
                "image_path": f"coco/{coco_split}/{file_name}",
                "expression": str(sentence.get("sent") or sentence.get("raw") or "").strip(),
                "gt_rle": gt_rle,
                "split": "test",
                "dataset_name": "Ref-ZOM_seg_test",
                "source_type": str(ref.get("source") or "").lower(),
            }


CONVERTERS = {
    **{name: refcoco(name) for name in REFCOCO},
    "ReasonSeg_seg_val": reasonseg("val", "ReasonSeg_seg_val"),
    "ReasonSeg_seg_test": reasonseg("test", "ReasonSeg_seg_test"),
    "Ref-ZOM_seg_test": refzom,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", type=Path, required=True, help="folder holding the downloaded releases")
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(CONVERTERS), choices=list(CONVERTERS))
    args = parser.parse_args()
    for dataset in args.datasets:
        output = args.eval_root / "_annotations" / "refseg" / f"{dataset}.jsonl"
        write_jsonl(output, CONVERTERS[dataset](args.raw_root, args.eval_root))


if __name__ == "__main__":
    main()
