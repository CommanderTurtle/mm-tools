"""Label / mask check lists of the released recipe (the qc/ folder next to the weights): checks every file against
train/assets/qc_lists.json (sha256) and writes it uncompressed where train.finetune reads it, <crops>/<pool>/qc.<i>.jsonl
(train.qc_crops label check: keeps err2d < 0.12) and <crops>/<pool>/mask_qc.jsonl (train.qc_masks: drops masks with
kp_in < 0.5 or mesh_iou < 0.2). The lists name crops by their pool-relative file names, as written by the builders with the
documented arguments.

    python -m train.fetch_qc --src <weights>/qc --crops $DATA_ROOT/crops     # or the folder's URL
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "qc_lists.json")


def fetch(src, crops, manifest=MANIFEST, pools=None):
    files = json.load(open(manifest))["files"]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, meta in sorted(files.items()):
            pool = name.split("/")[0]
            if pools and pool not in pools:
                continue
            if src.startswith(("http://", "https://")):
                path = os.path.join(tmp, name.replace("/", "__"))
                urllib.request.urlretrieve(f"{src.rstrip('/')}/{name}", path)
            else:
                path = os.path.join(src, name)
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if sha != meta["sha256"]:
                raise ValueError(f"{name}: sha256 {sha} != {meta['sha256']}")
            if not os.path.isdir(os.path.join(crops, pool)):
                print(f"skip {name}: no pool {os.path.join(crops, pool)}")
                continue
            dst = os.path.join(crops, name[:-len(".gz")])
            with gzip.open(path, "rb") as f, open(dst, "wb") as g:
                shutil.copyfileobj(f, g)
            n = sum(1 for _ in open(dst))
            if n != meta["rows"]:
                raise ValueError(f"{dst}: {n} rows != {meta['rows']}")
            out.append(dst)
            print(f"{dst}: {n} rows")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="the qc folder (<pool>/<name>.jsonl.gz) or its URL")
    p.add_argument("--crops", required=True, help="crop root ($DATA_ROOT/crops) with the pools as named in --sources")
    p.add_argument("--pools", default="", help="comma-separated subset (default: every pool in the manifest)")
    a = p.parse_args()
    fetch(a.src, a.crops, pools=set(filter(None, a.pools.split(","))))
