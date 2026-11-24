"""Fetch Dome-Objaverse objects the student never trained on and extract one input view each.

    python fetch_unseen.py --objects 102 --out data/dome_unseen --inputs test_inputs/dome_unseen

Objects already under the training root are excluded; the rest are picked round-robin across
the 160 shards (a different seed from the training fetch).  The input is a fixed view (v13: the
~30-degree elevation ring, three-quarter azimuth), saved as RGBA so Idol uses the real alpha.
"""
import argparse
import collections
import glob
import io
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(os.path.dirname(__file__), "..", "data", "dome_files.txt"))
    ap.add_argument("--train_root", default="data/dome_objaverse")
    ap.add_argument("--objects", type=int, default=102)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--view", type=int, default=13)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    seen = {os.path.relpath(p, a.train_root) for p in glob.glob(os.path.join(a.train_root, "dome_objaverse", "*", "*.parquet"))}
    files = [l.strip() for l in open(a.index) if l.strip() and l.strip() not in seen]
    by_shard = collections.defaultdict(list)
    for f in files:
        by_shard[f.split("/")[1]].append(f)
    rng = random.Random(a.seed)
    for v in by_shard.values():
        rng.shuffle(v)
    picked, shards = [], sorted(by_shard, key=int)
    while len(picked) < a.objects:
        for s in shards:
            if by_shard[s] and len(picked) < a.objects:
                picked.append(by_shard[s].pop())
    assert not set(picked) & seen
    print(f"[unseen] {len(seen)} training objects excluded; fetching {len(picked)}", flush=True)
    snapshot_download("zeyuanyin/Dome-Objaverse", repo_type="dataset", local_dir=a.out,
                      allow_patterns=picked, max_workers=4)
    os.makedirs(a.inputs, exist_ok=True)
    n = 0
    for f in picked:
        t = pq.read_table(os.path.join(a.out, f), columns=["view_id", "image_png"])
        ids = t.column("view_id").to_pylist()
        im = Image.open(io.BytesIO(t.column("image_png")[ids.index(a.view)].as_py())).convert("RGBA")
        shard, oid = f.split("/")[1], os.path.splitext(os.path.basename(f))[0]
        im.save(os.path.join(a.inputs, f"dome-{shard}-{oid}.png"))
        n += 1
    print(f"[unseen] wrote {n} input views to {a.inputs}", flush=True)


if __name__ == "__main__":
    main()
