"""Convert ``answer.jsonl`` to the FastEval TSV format (``image_id \\t [detections]``).

Only boxes of categories that are annotated as present in the image are kept
(positive-only protocol). All boxes are assigned the same confidence, so the
order in the answer is the ranking.
"""

from __future__ import annotations

import argparse
import json
import os


def norm_category(name) -> str:
    return str(name).lower().replace("_", " ").strip()


def convert(pred_jsonl: str, gt_json: str, out_tsv: str) -> None:
    with open(gt_json, encoding="utf-8") as handle:
        data = json.load(handle)
    id_to_name = {int(c["id"]): c["name"] for c in data["categories"]}
    name_to_id = {}
    for category_id, name in id_to_name.items():
        name_to_id[name] = category_id
        name_to_id[norm_category(name)] = category_id
    image_ids = {os.path.basename(i["file_name"]): int(i["id"]) for i in data["images"]}
    positives: dict[int, set[int]] = {}
    for annotation in data["annotations"]:
        positives.setdefault(int(annotation["image_id"]), set()).add(int(annotation["category_id"]))

    detections: dict[int, list[dict]] = {}
    with open(pred_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            basename = os.path.basename(row["image_path"])
            if basename not in image_ids:
                raise ValueError(f"{basename} is not in {gt_json}")
            image_id = image_ids[basename]
            allowed = positives.get(image_id, set())
            entries = []
            for name, shapes in (row.get("extracted_predictions") or {}).items():
                category_id = name_to_id.get(norm_category(name), name_to_id.get(str(name)))
                if category_id is None or category_id not in allowed:
                    continue
                for shape in shapes:
                    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
                        continue
                    x1, y1, x2, y2 = shape
                    entries.append(
                        {"class": id_to_name[category_id], "conf": 0.2, "rect": [x1, y1, x2 - x1, y2 - y1]}
                    )
            detections[image_id] = entries

    os.makedirs(os.path.dirname(os.path.abspath(out_tsv)), exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as handle:
        for image_id, entries in detections.items():
            handle.write(f"{image_id}\t{json.dumps(entries, ensure_ascii=False)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--out_tsv", required=True)
    args = parser.parse_args()
    convert(args.pred_jsonl, args.gt_json, args.out_tsv)


if __name__ == "__main__":
    main()
