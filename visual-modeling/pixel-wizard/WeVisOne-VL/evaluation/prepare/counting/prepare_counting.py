"""Convert the counting benchmarks to the ``box_eval`` format.

The script reads the downloaded releases from ``--raw_root/<NAME>`` (see
README.md), saves images to ``--eval_root/<DATASET>/images``, and writes
annotations to ``--eval_root/_annotations/box_eval/<DATASET>.jsonl``.

Each row has the format ``{image_path, dataset_name, task_name: "counting",
categories: [phrase], gt: {phrase: [[x0, y0, x1, y1]]}, gt_count, sample_id,
qa_index}``. Only ``gt_count`` is used for scoring. ``gt`` contains the boxes
provided by the release, which may be empty.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evaluation.prepare import pixmo_cache
from evaluation.prepare.common import iter_parquet, link_directory, save_jpeg, write_bytes, write_jsonl


def row(dataset: str, image_path: str, label: str, boxes: list, gt_count: int,
        image_index: int | str, qa_index: int) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "dataset_name": dataset,
        "task_name": "counting",
        "categories": [label],
        "gt": {label: boxes},
        "gt_count": int(gt_count),
        "sample_id": f"{image_index:06d}_{qa_index}" if isinstance(image_index, int) else f"{image_index}_{qa_index}",
        "qa_index": qa_index,
    }


def count_phrase(question: str) -> str:
    """Turn a CountQA question into the noun phrase used by the counting prompt."""
    q = re.sub(r"^,\s*", "", " ".join(str(question).strip().split()))
    match = re.search(r"(?i)\bhow many\s+(.+)$", q)
    if match:
        phrase = match.group(1).strip()
        prefix = re.sub(r"^,\s*", "", q[: match.start()].strip()).rstrip(".!?,").strip()
        phrase = re.sub(r"(?i)\bare there\b", "", phrase)
        phrase = re.sub(r"(?i)\bis there\b", "", phrase)
        phrase = re.sub(r"(?i)\s*(?:do you see|can you see|are present)\s*$", "", phrase).strip()
        phrase = re.sub(r"(?i)^total number of\s+", "", phrase).strip()
        phrase = re.sub(r"(?i)^(.+?)\s+are\s+", r"\1 ", phrase, count=1).strip()  # "tiles are on X" -> "tiles on X"
        phrase = re.sub(r"(?i)\s+in the image\s*$", "", phrase).strip()
        phrase = re.sub(r"\s+", " ", phrase.rstrip("?.!").strip()).strip()
        if not phrase:  # "If you combine all the chess pieces, how many are there?"
            subject = re.search(r"(?i)\b(?:all (?:of )?(?:the )?|the )?([A-Za-z][\w\s\-/]+?)\s*$", prefix)
            return (subject.group(1).strip() if subject else prefix) or "objects"
        return f"{phrase} ({prefix})" if prefix else phrase
    match = re.search(r"(?i)\b(?:what is (?:the )?(?:total )?number of|count(?: the)?)\s+(.+)$", q)
    if match:
        phrase = re.sub(r"(?i)\s+in the image\s*$", "", match.group(1).strip().rstrip("?.!").strip()).strip()
        return phrase or q.rstrip("?.!").strip()
    return q.rstrip("?.!").strip()


def countqa(raw: Path, eval_root: Path) -> Iterable[dict]:
    for image_index, record in enumerate(iter_parquet((raw / "CountQA" / "data").glob("*.parquet"))):
        image_path = f"CountQA/images/{image_index:06d}.jpg"
        save_jpeg(record["image"]["bytes"], eval_root / image_path)
        for qa_index, (question, answer) in enumerate(zip(record["questions"], record["answers"])):
            yield row("CountQA", image_path, count_phrase(question), [], int(str(answer).strip()),
                      image_index, qa_index)


def pixmo_count(split: str, dataset: str):
    def convert(raw: Path, eval_root: Path) -> Iterable[dict]:
        records = list(iter_parquet((raw / "pixmo-count" / "data").glob(f"{split}-*.parquet"),
                                    columns=["image_url", "label", "count"]))
        targets = {f"{dataset}/images/{index:06d}.jpg": record["image_url"] for index, record in enumerate(records)}
        pending = {path: url for path, url in targets.items() if not (eval_root / path).is_file()}
        # Some hosts now serve images with different bytes but identical pixels, so the
        # published sha256 is not checked. The cache is used only for unavailable URLs.
        fetched = pixmo_cache.fetch(pending.values()) if pending else {}
        missing = [path for path, url in pending.items() if url not in fetched]
        if missing:
            raise FileNotFoundError(f"{dataset}: {len(missing)} images unavailable, e.g. {missing[:3]}")
        for path, url in pending.items():
            write_bytes(fetched[url], eval_root / path)
        for index, record in enumerate(records):
            label = str(record.get("label") or "").strip() or "objects"
            yield row(dataset, f"{dataset}/images/{index:06d}.jpg", label, [], int(record["count"]), index, 0)
    return convert


def cloc(split: str, dataset: str):
    def convert(raw: Path, eval_root: Path) -> Iterable[dict]:
        source = raw / "CLOC"
        records = json.loads((source / "annotations" / f"{split}_split_expanded_by_class.json").read_text())
        for folder in sorted({record["image_path"].split("/", 1)[0] for record in records.values()}):
            link_directory(eval_root / "CLOC" / folder, source / "data" / folder)
        for key in sorted(records, key=lambda key: (0, int(key)) if key.isdigit() else (1, key)):
            record = records[key]
            label = str(record["selected_classes"][0]).strip()
            target = record["selected_annotation"].get(label) or {}
            boxes = [[float(value) for value in box[:4]] for box in target.get("bbox") or []]
            gt_count = len(target.get("point") or []) or len(boxes)
            yield row(dataset, f"CLOC/{record['image_path']}", label, boxes, gt_count,
                      str(int(key)) if key.isdigit() else key, 0)
    return convert


def kubricount(split: str, dataset: str):
    def convert(raw: Path, eval_root: Path) -> Iterable[dict]:
        source = raw / "KubriCount"
        items = json.loads((source / split / "extracted_metadata.json").read_text())
        image_ids = list(dict.fromkeys(str(item["image_id"]) for item in items))
        path_of = {image_id: f"{dataset}/images/{index:06d}.jpg" for index, image_id in enumerate(image_ids)}
        _extract_kubric_images(source, split, eval_root, path_of)
        qa_count: dict[int, int] = defaultdict(int)
        for item in items:
            image_path = path_of[str(item["image_id"])]
            image_index = int(Path(image_path).stem)
            qa_index = qa_count[image_index]
            qa_count[image_index] += 1
            boxes = []
            for corners in item.get("box_examples_coordinates") or []:
                xs, ys = [float(p[0]) for p in corners], [float(p[1]) for p in corners]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
            label = str(item.get("category") or "").strip() or "objects"
            gt_count = int(item.get("count") or 0) or len(item.get("points") or []) or len(boxes)
            yield row(dataset, image_path, label, boxes, gt_count, image_index, qa_index)
    return convert


def _extract_kubric_images(source: Path, split: str, eval_root: Path, path_of: dict[str, str]) -> None:
    shard_of = {}
    for line in (source / "metadata" / f"{split}_pass_scenes.jsonl").read_text().splitlines():
        scene = json.loads(line)
        shard_of[str(scene["path_in_dataset"]).rstrip("/")] = source / scene["shard"]
    by_shard: dict[Path, set[str]] = defaultdict(set)
    for image_id, image_path in path_of.items():
        if not (eval_root / image_path).is_file():
            by_shard[shard_of[str(Path(image_id).parent)]].add(image_id)
    for shard, wanted in by_shard.items():
        print(f"extracting {len(wanted)} images from {shard}", flush=True)
        with tarfile.open(shard) as bundle:
            for member in bundle:
                if member.isfile() and member.name in wanted:
                    save_jpeg(bundle.extractfile(member).read(), eval_root / path_of[member.name])
                    wanted.discard(member.name)
                    if not wanted:
                        break
        if wanted:
            raise FileNotFoundError(f"{len(wanted)} images absent from {shard}")


def mixcount(split: str, dataset: str):
    def convert(raw: Path, eval_root: Path) -> Iterable[dict]:
        records = iter_parquet((raw / "MixCount" / "parquet_data").glob(f"{split}-*.parquet"))
        for image_index, record in enumerate(records):
            image_path = f"{dataset}/images/{image_index:06d}.jpg"
            save_jpeg(record["image"]["bytes"], eval_root / image_path)
            names = [str(name).strip() for name in record.get("class_names") or []]
            counts = [int(count) for count in record.get("class_counts") or []]
            objects = record.get("objects") or {}
            boxes_of: dict[int, list] = defaultdict(list)
            for (x, y, w, h), category in zip(objects.get("bbox") or [], objects.get("category") or []):
                boxes_of[int(category)].append([float(x), float(y), float(x) + float(w), float(y) + float(h)])
            for qa_index in range(max(len(names), int(record.get("num_classes") or 0))):
                label = names[qa_index] if qa_index < len(names) and names[qa_index] else f"class_{qa_index}"
                boxes = boxes_of.get(qa_index, [])
                gt_count = counts[qa_index] if qa_index < len(counts) else len(boxes)
                yield row(dataset, image_path, label, boxes, gt_count, image_index, qa_index)
    return convert


CONVERTERS = {
    "CountQA": countqa,
    "pixmo-count-val": pixmo_count("validation", "pixmo-count-val"),
    "pixmo-count-test": pixmo_count("test", "pixmo-count-test"),
    "cloc-val": cloc("val", "cloc-val"),
    "cloc-test": cloc("test", "cloc-test"),
    "KubriCount-testA": kubricount("testA", "KubriCount-testA"),
    "KubriCount-testB": kubricount("testB", "KubriCount-testB"),
    "MixCount-val": mixcount("val", "MixCount-val"),
    "MixCount-test": mixcount("test", "MixCount-test"),
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
