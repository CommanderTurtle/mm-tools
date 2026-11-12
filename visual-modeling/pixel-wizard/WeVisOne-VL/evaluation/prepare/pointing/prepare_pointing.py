"""Convert PointBench and PixMo-Points-Eval to the ``point_eval`` format.

The script reads the downloaded releases from ``--raw_root/<NAME>`` (see
README.md), saves images to ``--eval_root/<NAME>/``, and writes annotations to
``--eval_root/_annotations/point_eval/<NAME>.jsonl``.

Each row contains its own ``question`` and one RLE mask per target in
``gt_mask["target"]``. A predicted point is correct if it falls inside a mask.
Only rows with ``evaluation_protocol == "point_mask_f1"`` are scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from evaluation.prepare import pixmo_cache
from evaluation.prepare.common import (
    extract_members,
    iter_parquet,
    json_rle,
    sha256_bytes,
    sha256_file,
    write_bytes,
    write_jsonl,
)

POINTBENCH_ARCHIVES = {
    "selected_images.zip": "a82914086ddcc28d33a90aebb707cb99a5e0d91af5beab4159384ac4e1e0b6e9",
    "selected_masks.zip": "a2c93869e659df6ddce5966fa660f0d9058237ed5d261c0664ae683cd11e2ed8",
}
PIXMO_PARQUET = "data/test-00000-of-00001.parquet"
PIXMO_PARQUET_SHA256 = "5a7bf84b8b06ccde4deff7857923686cae3540ecc39212183c010b3a9275c3e0"
PIXMO_ROWS = 436


def row(dataset: str, sample_id: str, image_path: str, size: tuple[int, int], question: str,
        protocol: str, task_family: str, benchmark_slice: str, masks: list[np.ndarray]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "question": question.strip(),
        "dataset_name": dataset,
        "task_name": "referring_object_detection",
        "task_family": task_family,
        "evaluation_protocol": protocol,
        "benchmark_slice": benchmark_slice,
        "categories": ["target"],
        "gt": {"target": []},
        "gt_mask": {"target": [json_rle(mask) for mask in masks]},
        "gt_count": len(masks),
        "image_width": int(size[0]),
        "image_height": int(size[1]),
    }


def _file_index(root: Path) -> dict[str, list[Path]]:
    """Map file names to paths, also under their repaired UTF-8 spelling.

    Two names in the PointBench zips are UTF-8 without the zip UTF-8 flag, so they
    extract as CP437; ``data.json`` refers to them by the intended name.
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        names = {path.name}
        try:
            names.add(path.name.encode("cp437").decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        for name in names | {unicodedata.normalize("NFC", name) for name in names}:
            if path not in index.setdefault(name, []):
                index[name].append(path)
    return index


def _resolve(index: dict[str, list[Path]], name: str, category: str) -> Path:
    candidates = index.get(name) or index.get(unicodedata.normalize("NFC", name), [])
    candidates = [path for path in candidates if category in path.parts] or candidates
    if len(candidates) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve {name!r}: {candidates}")
    return candidates[0]


def pointbench(raw: Path, eval_root: Path) -> Iterable[dict]:
    source, files = raw / "PointBench", eval_root / "PointBench" / "files"
    for archive, digest in POINTBENCH_ARCHIVES.items():
        if sha256_file(source / archive) != digest:
            raise ValueError(f"{source / archive}: sha256 differs from the pinned revision")
        extract_members(source / archive, files)
    references = {}
    with (source / "pixmo_metadata.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("points") and record["points"] != "[]":
                references[record["image_filename"]] = json.loads(record["points"])
    index = _file_index(files)
    for row_index, record in enumerate(json.loads((source / "data.json").read_text(encoding="utf-8"))):
        category = str(record["category"])
        image_file = _resolve(index, record["image_filename"], category)
        with Image.open(image_file) as image:
            width, height = image.size
        with Image.open(_resolve(index, record["mask_filename"], category)) as image:
            values = np.asarray(image)
        mask = values > 127 if values.ndim == 2 else np.any(values > 127, axis=2)
        if mask.shape != (height, width):
            print(f"PointBench row {row_index}: mask/image size mismatch upstream, skipped", flush=True)
            continue
        question = str(record["user_input"]).strip()
        if category == "steerable":
            coords = ", ".join(f"[{float(p['x']) * width / 100:.1f}, {float(p['y']) * height / 100:.1f}]"
                               for p in references[record["image_filename"]])
            question += (f"\nThe image contains an existing original point at pixel coordinates: {coords}."
                         "\nThe query refers to this existing point.")
        protocol = "point_counting_union_region" if category == "counting" else "point_mask_f1"
        item = row("PointBench", f"pointbench-{row_index:04d}", image_file.relative_to(eval_root).as_posix(),
                   (width, height), question, protocol, "general_pointing", category, [mask])
        item["gt_count"] = int(record.get("count", 1))
        yield item


def _pixmo_slice(count: int) -> str:
    if count <= 1:
        return ["zero", "single"][count]
    return "low_freq" if count <= 3 else "med_freq" if count <= 10 else "high_freq"


def pixmo_points_eval(raw: Path, eval_root: Path) -> Iterable[dict]:
    parquet = raw / "PixMo-Points-Eval" / PIXMO_PARQUET
    if sha256_file(parquet) != PIXMO_PARQUET_SHA256:
        raise ValueError(f"{parquet}: sha256 differs from the pinned revision")
    records = list(iter_parquet([parquet], columns=["image_url", "image_sha256", "label", "masks"]))
    if len(records) != PIXMO_ROWS:
        raise ValueError(f"expected {PIXMO_ROWS} rows, got {len(records)}")
    image_of = {r["image_url"]: f"PixMo-Points-Eval/images-official/{r['image_sha256'].lower()}.img" for r in records}
    expected = {r["image_url"]: r["image_sha256"].lower() for r in records}
    pending = [url for url, path in image_of.items()
               if not (eval_root / path).is_file() or sha256_file(eval_root / path) != expected[url]]
    if pending:
        fetched = pixmo_cache.fetch(pending, accept=lambda url, payload: sha256_bytes(payload) == expected[url])
        missing = [url for url in pending if url not in fetched]
        if missing:
            raise FileNotFoundError(f"{len(missing)} PixMo-Points-Eval images unavailable, e.g. {missing[:3]}")
        for url in pending:
            write_bytes(fetched[url], eval_root / image_of[url])
    for index, record in enumerate(records):
        image_path = image_of[record["image_url"]]
        with Image.open(eval_root / image_path) as image:
            size = image.size
        # Queries without a target are released with one all-zero placeholder mask.
        masks = [mask for value in record["masks"] if np.any(mask := np.asarray(value, dtype=bool))]
        words = str(record["label"]).lower().split()
        if words and words[0] in {"a", "the"}:
            words = words[1:]
        yield row("PixMo-Points-Eval", f"pixmo_points_eval-test-{index}", image_path, size,
                  f"Point to {' '.join(words)}.", "point_mask_f1", "general_and_counting_pointing",
                  _pixmo_slice(len(masks)), masks)


CONVERTERS = {"PointBench": pointbench, "PixMo-Points-Eval": pixmo_points_eval}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", type=Path, required=True, help="folder holding the downloaded releases")
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(CONVERTERS), choices=list(CONVERTERS))
    args = parser.parse_args()
    for dataset in args.datasets:
        output = args.eval_root / "_annotations" / "point_eval" / f"{dataset}.jsonl"
        write_jsonl(output, CONVERTERS[dataset](args.raw_root, args.eval_root))


if __name__ == "__main__":
    main()
