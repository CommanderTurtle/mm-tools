"""Convert the referring detection benchmarks to the ``box_eval`` format.

The script reads the downloaded releases from ``--raw_root/<NAME>`` (see
README.md), saves images to ``--eval_root/<NAME>/images``, and writes
annotations to ``--eval_root/_annotations/box_eval/<DATASET>.jsonl``.

Each row has the format ``{image_path, dataset_name, task_name, categories:
[expression], gt: {expression: [[x0, y0, x1, y1]]}, sample_id}``. Expressions
without a target have an empty box list.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from evaluation.prepare.common import (
    extract_members,
    image_suffix,
    iter_parquet,
    read_jsonl,
    write_bytes,
    write_jsonl,
)


def row(dataset: str, image_path: str, expression: str, boxes: list, sample_id: str) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "dataset_name": dataset,
        "task_name": "referring_object_detection",
        "categories": [expression],
        "gt": {expression: boxes},
        "sample_id": sample_id,
    }


def ref_adv_s(raw: Path, eval_root: Path) -> Iterable[dict]:
    for record in iter_parquet((raw / "Ref-Adv-s" / "data").glob("*.parquet")):
        index = int(record["row_idx"])
        image_path = f"Ref-Adv-s/images/{index:04d}{image_suffix(record['image'])}"
        write_bytes(record["image"]["bytes"], eval_root / image_path)
        box = [float(value) for value in record["solution"]]
        yield row("Ref-Adv-s", image_path, str(record["normal_caption"]), [box], f"ref-adv-s-{index:04d}")


def pr_bench(raw: Path, eval_root: Path) -> Iterable[dict]:
    source = raw / "PR-Bench"
    records = read_jsonl(source / "annotation.jsonl")
    extract_members(source / "images.zip", eval_root / "PR-Bench", {f"images/{r['image_path']}" for r in records})
    for index, record in enumerate(records):
        rejected = str(record["task_type"]).lower() == "reject"
        boxes = [] if rejected else [[float(value) for value in record["bbox"]]]
        yield row("PR-Bench", f"PR-Bench/images/{record['image_path']}", str(record["expression"]), boxes,
                  f"pr-bench-{index:04d}")


def dvgbench(raw: Path, eval_root: Path) -> Iterable[dict]:
    source = raw / "DVGBench"
    records = read_jsonl(source / "dvg_test.jsonl")
    members = {f"images/{r['dataset']}/{r['image_id']}" for r in records}
    extract_members(source / "images.zip", eval_root / "DVGBench", members)
    # question_id is not unique across subsets, so the row index is used as the ID.
    for index, record in enumerate(records):
        box = [float(value) for value in record["bbox"]]
        image_path = f"DVGBench/images/{record['dataset']}/{record['image_id']}"
        yield row("DVGBench", image_path, str(record["question"]), [box], f"dvgbench-{index:04d}")


def ref_l4(raw: Path, eval_root: Path) -> Iterable[dict]:
    source = raw / "Ref-L4"
    records = list(iter_parquet([source / "ref-l4-val.parquet", source / "ref-l4-test.parquet"]))
    records.sort(key=lambda record: int(record["id"]))
    extract_members(source / "images.tar.gz", eval_root / "Ref-L4" / "images", {r["file_name"] for r in records})
    for record in records:
        x, y, w, h = (float(value) for value in record["bbox"])
        yield row("Ref-L4-all", f"Ref-L4/images/{record['file_name']}", str(record["caption"]).strip(),
                  [[x, y, x + w, y + h]], f"ref-l4-{int(record['id']):08d}")


def refcocog_rej(raw: Path, eval_root: Path) -> Iterable[dict]:
    records = iter_parquet((raw / "RefCOCOg_rej" / "data").glob("validation-*.parquet"))
    for record in records:
        number = int(record["id"])
        image_path = f"RefCOCOg-rej/images/{number:05d}{image_suffix(record['image'])}"
        write_bytes(record["image"]["bytes"], eval_root / image_path)
        yield row("RefCOCOg-rej-val", image_path, str(record["rejected_exp"]).strip(), [],
                  f"refcocog-rej-val-{number:05d}")


CONVERTERS = {
    "Ref-Adv-s": ref_adv_s,
    "PR-Bench": pr_bench,
    "DVGBench": dvgbench,
    "Ref-L4-all": ref_l4,
    "RefCOCOg-rej-val": refcocog_rej,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", type=Path, required=True, help="folder holding the downloaded releases")
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(CONVERTERS), choices=list(CONVERTERS))
    args = parser.parse_args()
    for dataset in args.datasets:
        output = args.eval_root / "_annotations" / "box_eval" / f"{dataset}.jsonl"
        write_jsonl(output, CONVERTERS[dataset](args.raw_root, args.eval_root))


if __name__ == "__main__":
    main()
