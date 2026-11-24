"""Fetch a subset of Dome-Objaverse (512x512 RGBA renders, 48 views per object).

Only the object images are needed: the teacher produces every training target itself, so
the normal/depth maps and the captions are ignored.  Objects are taken round-robin across
the 160 top-level shards so the subset spans the whole set rather than one corner of it.

    python fetch_dome.py --objects 3000 --out data/dome_objaverse
"""
import argparse
import collections
import os
import random

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(os.path.dirname(__file__), "..", "data", "dome_files.txt"))
    ap.add_argument("--objects", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = [l.strip() for l in open(a.index) if l.strip()]
    by_shard = collections.defaultdict(list)
    for f in files:
        by_shard[f.split("/")[1]].append(f)
    rng = random.Random(a.seed)
    for v in by_shard.values():
        rng.shuffle(v)

    picked, shards = [], sorted(by_shard, key=int)
    while len(picked) < a.objects:
        progressed = False
        for s in shards:
            if by_shard[s] and len(picked) < a.objects:
                picked.append(by_shard[s].pop())
                progressed = True
        if not progressed:
            break
    print(f"[fetch] {len(picked)} objects over {len(shards)} shards -> {a.out}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    snapshot_download("zeyuanyin/Dome-Objaverse", repo_type="dataset", local_dir=a.out,
                      allow_patterns=picked, max_workers=a.workers)
    have = sum(len(fs) for _, _, fs in os.walk(a.out) if any(f.endswith(".parquet") for f in fs))
    print(f"[fetch] done: {have} parquet files under {a.out}", flush=True)


if __name__ == "__main__":
    main()
