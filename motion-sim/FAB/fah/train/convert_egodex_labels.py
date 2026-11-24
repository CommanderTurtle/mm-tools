"""Rewrite the EgoDex ARKit 3D labels into the MANO joint convention (correction fit by train/egodex_mano_offsets.py)
as new crop pools <pool>_mano that share the original images and masks through relative symlinks; the original
pools are not modified.

    python -m train.convert_egodex_labels --root $DATA_ROOT/crops --pools egodex_test,egodex_part1 \
        --corr $CKPT_ROOT/egodex_mano/arkit_to_mano_ref_mesh.json --pack
-> <root>/<pool>_mano/{index.jsonl, BASE_POOL, <episode> -> ../<pool>/<episode>, masks, masks_index.*.jsonl}
   --pack: $FAH_WORK/data/crop_pools/<pool>_mano.tar (index + links only), label / mask QC copies in
   $FAH_WORK/qc/<pool>_mano/ and $FAH_WORK/qc/masks/; extract the BASE_POOL next to it before training.
"""
import argparse
import glob
import json
import os
import shutil
import tarfile

import numpy as np

from train.egodex_mano_offsets import apply

WORK = os.environ.get("FAH_WORK", "output")  # packed pools / QC files


def mask_entries(pool):
    """Top-level entries (masks dir, masks_index files) of the pool's SAM masks tar."""
    t = os.path.join(WORK, "data", "crop_pools", f"{pool}_masks.tar")
    if not os.path.exists(t):
        return []
    with tarfile.open(t) as tf:
        rel = [n.split("/", 1)[1] for n in tf.getnames() if n.count("/") == 1]
    return [r for r in rel if r == "masks" or r.startswith("masks_index")]


def reproject(r, j):
    """2D labels follow converted 3D joints (crop pixels, same visibility flags)."""
    if r.get("kp2d") is not None and r.get("K") is not None:
        K, M, kp = np.asarray(r["K"]), np.asarray(r["M"]), np.asarray(r["kp2d"], np.float64)
        uv = j[:, :2] / np.clip(j[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
        kp[:, :2] = uv @ M[:, :2].T + M[:, 2]
        r["kp2d"] = kp.round(2).tolist()


def write_pool(root, pool, name, recs, pack):
    """Converted pool <root>/<name>: new index + BASE_POOL + relative links to everything else of <root>/<pool>;
    pack: tar (index + links) in $FAH_WORK/data/crop_pools and copies of the pool's label / mask QC files."""
    src, dst = os.path.join(root, pool), os.path.join(root, name)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "index.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    open(os.path.join(dst, "BASE_POOL"), "w").write(pool + "\n")
    links = {e for e in os.listdir(src) if e != "index.jsonl" and not e.startswith(("qc.", "qc_", "mask_qc"))} | set(mask_entries(pool))
    for e in sorted(links):
        p = os.path.join(dst, e)
        if not os.path.lexists(p):
            os.symlink(os.path.join("..", pool, e), p)
    for q in glob.glob(os.path.join(src, "qc.jsonl")) + glob.glob(os.path.join(src, "qc.[0-9]*.jsonl")) + glob.glob(os.path.join(src, "mask_qc.jsonl")):
        shutil.copy(q, dst)  # QC keep-lists are keyed by crop file: still valid
    print(f"{pool} -> {name}: {len(recs)} records, {len(links)} links", flush=True)
    if pack:
        out = os.path.join(WORK, "data", "crop_pools", f"{name}.tar")
        with tarfile.open(out + ".tmp", "w") as tf:
            tf.add(dst, arcname=name)  # symlinks are stored as links
        os.replace(out + ".tmp", out)
        qs = glob.glob(os.path.join(WORK, "qc", pool, "qc.[0-9]*.jsonl"))
        if qs:
            os.makedirs(os.path.join(WORK, "qc", name), exist_ok=True)
            for q in qs:
                shutil.copy(q, os.path.join(WORK, "qc", name))
        mq = os.path.join(WORK, "qc", "masks", f"{pool}_mask_qc.jsonl")
        if os.path.exists(mq):
            shutil.copy(mq, os.path.join(WORK, "qc", "masks", f"{name}_mask_qc.jsonl"))
        print(f"packed {out} (+ {len(qs)} label QC files)", flush=True)


def convert(args, pool, corr):
    delta = np.asarray(corr["delta"])
    ridge = None if not (args.ridge and corr.get("ridge")) else {k: np.asarray(v) for k, v in corr["ridge"].items() if k in ("W", "b")}
    recs = [json.loads(l) for l in open(os.path.join(args.root, pool, "index.jsonl"))]
    has3d = [i for i, r in enumerate(recs) if r.get("joints3d") is not None]
    J = np.asarray([recs[i]["joints3d"] for i in has3d], np.float64)
    Jc = np.concatenate([apply(J[i:i + 50000], delta, ridge) for i in range(0, len(J), 50000)]) if len(J) else J
    tag = f"arkit->mano {os.path.basename(args.corr)}{' +ridge' if ridge is not None else ''}"
    for i, j in zip(has3d, Jc):
        recs[i]["joints3d"] = j.round(5).tolist()
        reproject(recs[i], j)
        recs[i]["label_conv"] = tag
    shift = np.linalg.norm(Jc - J, axis=-1).mean() * 1000 if len(J) else 0
    print(f"{pool}: {len(has3d)} / {len(recs)} records converted, mean joint shift {shift:.1f} mm", flush=True)
    write_pool(args.root, pool, pool + args.suffix, recs, args.pack)


def main(args):
    corr = json.load(open(args.corr))
    for pool in args.pools.split(","):
        convert(args, pool, corr)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="$DATA_ROOT/crops")
    p.add_argument("--pools", default="egodex_test,egodex_part1")
    p.add_argument("--corr", required=True, help="arkit_to_mano_*.json from train.egodex_mano_offsets fit")
    p.add_argument("--ridge", action="store_true", help="also apply the ridge term (if fitted)")
    p.add_argument("--suffix", default="_mano")
    p.add_argument("--pack", action="store_true", help="write $FAH_WORK/data/crop_pools/<pool>_mano.tar + QC copies")
    main(p.parse_args())
