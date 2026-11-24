"""Held-out labelled crops for the part evaluations.

A crop is held out when its source image falls in train.finetune's validation split (crc32 of the source id modulo
VAL_MOD == 0; never trained on) and its part label was made by an accepted route (the labels clipped to a hand mask).
The paper's crops are listed in bench/splits (parts_heldout.json: part head, 200 per pool; dense_parts.json:
dense points, 300 per pool; parts_hard.json: reviewed difficult HOT3D crops); `sample` is the rule they were drawn
with (one seeded permutation over the pools in order), `from_split` checks a split against a pool directory.

Split file: {"version": ..., "pools": {pool: {"part_dir": <label dir in the pool>, "mask_dir": <hand-mask dir, part
head only>, "files": [crop file, ...]}}}."""
import json
import os
import zlib

import numpy as np

VAL_MOD = 50  # = train.finetune VAL_MOD
ROUTES = ("mesh+mask", "model+mask")  # train.part_masks (GT mesh) / train.part_labels_model (model mesh), clipped to a hand mask


def is_heldout(record):
    """train.finetune's validation split: 1 / VAL_MOD of the source images (a video / sequence id before ':')."""
    return zlib.crc32(record.get("src", record["file"]).split(":")[0].encode()) % VAL_MOD == 0


def label_routes(pool_dir, part_dir):
    """crop file -> label route, from the label index files <part_dir>_index.*.jsonl."""
    routes = {}
    for f in sorted(os.listdir(pool_dir)):
        if f.startswith(part_dir + "_index.") and f.endswith(".jsonl"):
            for q in map(json.loads, open(os.path.join(pool_dir, f))):
                routes[q["file"]] = q.get("route")
    return routes


def candidates(crops, pool, part_dir, routes=ROUTES):
    """Held-out records of a pool with a label of an accepted route, in index order."""
    P = os.path.join(crops, pool)
    rt = label_routes(P, part_dir)
    return [r for r in map(json.loads, open(os.path.join(P, "index.jsonl"))) if rt.get(r["file"]) in routes and is_heldout(r)]


def sample(crops, pools, part_dirs, n, routes=ROUTES, seed=0):
    """pool -> n random held-out records (one permutation stream over the pools in the given order)."""
    rng = np.random.default_rng(seed)
    out = {}
    for pool in pools:
        recs = candidates(crops, pool, part_dirs.get(pool, "masks_parts"), routes)
        out[pool] = [recs[i] for i in rng.permutation(len(recs))[:n]]
    return out


def load_split(path):
    split = json.load(open(path))
    if "pools" not in split:
        raise ValueError(f"{path}: no 'pools'")
    return split


def from_split(crops, split, pools=None, part_dirs=None, mask_dirs=None, routes=ROUTES):
    """pool -> (part_dir, mask_dir, records) for exactly the split's crops (in the split's order). Raises if a listed
    crop is absent from the pool, outside the held-out split, without an accepted label, or listed twice."""
    part_dirs, mask_dirs = part_dirs or {}, mask_dirs or {}
    names = list(split["pools"]) if pools is None else list(pools)
    extra = set(names) - set(split["pools"])
    if extra:
        raise ValueError(f"pools not in the split: {sorted(extra)}")
    out = {}
    for pool in names:
        entry = split["pools"][pool]
        pdir = part_dirs.get(pool, entry.get("part_dir", "masks_parts"))
        mdir = mask_dirs.get(pool, entry.get("mask_dir", "masks"))
        files = entry["files"]
        if len(set(files)) != len(files):
            raise ValueError(f"{pool}: duplicate crops in the split would reweight the evaluation")
        by = {r["file"]: r for r in candidates(crops, pool, pdir, routes)}
        missing = [f for f in files if f not in by]
        if missing:
            raise ValueError(f"{pool}: {len(missing)} listed crops are absent or fail the held-out / label-route filters: {missing[:5]}")
        out[pool] = (pdir, mdir, [by[f] for f in files])
    return out


def write_split(path, selection, part_dirs, mask_dirs=None, **meta):
    """selection: pool -> records (sample / from_split order)."""
    mask_dirs = mask_dirs or {}
    pools = {p: dict(part_dir=part_dirs.get(p, "masks_parts"), **({"mask_dir": mask_dirs[p]} if p in mask_dirs else {}),
                     files=[r["file"] for r in recs]) for p, recs in selection.items()}
    json.dump(dict(meta, pools=pools), open(path, "w"), indent=1)
