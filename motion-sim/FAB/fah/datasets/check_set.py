"""Check a locally built set (bench2d v2 from bench.bench2d.build, its low-light copy from bench.bench2d.make_dark) against
the fingerprints of the copy the paper was scored on: same image ids per subset, the same labels per image (MD5 of each
record's hands), and optionally the same image bytes (SHA-256, first 16 hex digits).

    python datasets/check_set.py --ann $DATA_ROOT/bench2d_v2/annotations.json --ref datasets/bench2d_v2/fingerprints.json
    python datasets/check_set.py --ann $DATA_ROOT/bench2d_v2_dark/annotations.json --ref datasets/low_light/fingerprints.json --images
    python datasets/check_set.py --ann <annotations.json> --make <fingerprints.json> [--images]   (writes the reference)
"""
import argparse
import hashlib
import json
import os
from collections import Counter


def label_md5(rec):
    hands = [[bool(h["is_right"]), h["kp2d"], h["vis"], h["bbox"]] for h in rec["hands"]]
    return hashlib.md5(json.dumps(hands, separators=(",", ":")).encode()).hexdigest()[:16]


def fingerprints(ann_path, images):
    root = os.path.dirname(ann_path)
    out = {}
    for r in json.load(open(ann_path))["images"]:
        f = dict(subset=r["subset"], labels=label_md5(r))
        if images:
            f["image"] = hashlib.sha256(open(os.path.join(root, r["file"]), "rb").read()).hexdigest()[:16]
        out[r["id"]] = f
    return out


def main(args):
    mine = fingerprints(args.ann, args.images)
    if args.make:
        json.dump(mine, open(args.make, "w"), separators=(",", ":"), sort_keys=True)
        print(f"wrote {len(mine)} fingerprints to {args.make}")
        return
    ref = json.load(open(args.ref))
    count = Counter()
    for k, f in ref.items():
        m = mine.get(k)
        if m is None:
            count[(f["subset"], "missing")] += 1
            continue
        count[(f["subset"], "labels same" if m["labels"] == f["labels"] else "labels differ")] += 1
        if args.images and "image" in f:
            count[(f["subset"], "image same" if m["image"] == f["image"] else "image differs")] += 1
    for k in mine.keys() - ref.keys():
        count[(mine[k]["subset"], "extra")] += 1
    for (s, what), n in sorted(count.items()):
        print(f"{s:14s} {what:14s} {n:6d}")
    bad = sum(n for (s, w), n in count.items() if w in ("missing", "extra", "labels differ"))
    print("OK: same images and labels as the paper's copy" if not bad else f"{bad} records missing / extra / with other labels")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True)
    p.add_argument("--ref", default="")
    p.add_argument("--make", default="")
    p.add_argument("--images", action="store_true", help="also compare image bytes (SHA-256)")
    main(p.parse_args())
