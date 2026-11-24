"""Merge bench2d-format folders into one benchmark (no image copies: files are referenced relative to the
merged root). Used for the low-light benchmark = real dark images (human-corrected labels) + synthetic low
light bench2d. Re-run whenever a member's labels change.

    python -m bench.bench2d.merge_sets --out $DATA_ROOT/dark_v1 \
        --sets $DATA_ROOT/dark_pose:real_dark $DATA_ROOT/bench2d_v2_dark:syn_
(":prefix" ending in "_" prefixes the member's own subsets, otherwise it replaces them)
"""
import argparse
import json
import os


def main(args):
    os.makedirs(args.out, exist_ok=True)
    images, members = [], []
    for spec in args.sets:
        root, tag = spec.rsplit(":", 1)
        ann = json.load(open(os.path.join(root, "annotations.json")))
        rel = os.path.relpath(root, args.out)
        for r in ann["images"]:
            sub = tag + r["subset"] if tag.endswith("_") else tag
            images.append(dict(r, id=f"{sub}/{r['id'].split('/', 1)[-1]}", file=os.path.join(rel, r["file"]), subset=sub))
        members.append(dict(root=root, tag=tag, labels=ann["meta"].get("version"), images=len(ann["images"])))
    meta = dict(name=os.path.basename(args.out.rstrip("/")), members=members, note=args.note)
    json.dump(dict(meta=meta, images=images), open(os.path.join(args.out, "annotations.json"), "w"))
    print("wrote", args.out, len(images), "images", sorted({r["subset"] for r in images}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--note", default="low-light hand benchmark: real dark photos + synthetic low light (bench/bench2d/make_dark.py)")
    main(p.parse_args())
