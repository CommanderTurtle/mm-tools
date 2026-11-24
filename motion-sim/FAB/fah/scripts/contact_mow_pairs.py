"""MOW test under HACO's protocol, with paired deltas: per arm the per-image P / R / F1 at the arm's 35 mm threshold (undefined
-> 0, every test image has contact), their mean (F1, HACO's MOW metric) and the pooled vertex F1; paired arm-minus-arm deltas
with an image bootstrap (MOW test images are independent; 1,000 samples, seed 3: the same samples for every arm).

--mow: MOW with HACO's preprocessed labels (splits/test.json, preprocessed_data/test/contact_data/<id>.npy: 778 values, > 0.5 =
contact), fetched by each user from MOW and HACO's release; --preds: the directory of pred_<tag>.npz (prob: N x 778, rows in
splits/test.json order), e.g. the HACO adapter's or an FAH contact head's predictions.

    python scripts/contact_mow_pairs.py --mow $DATA_ROOT/mow --preds $DATA_ROOT/contact/test/mow \
        --arms fah=fahL-contact@0.58,haco=haco@0.52 --pairs fah-haco --out mow_pairs.json
"""
import argparse
import json
import os

import numpy as np


def mow_labels(root):
    ids = json.load(open(os.path.join(root, "splits", "test.json")))
    lab = np.stack([np.load(os.path.join(root, "preprocessed_data", "test", "contact_data", f"{i}.npy")) > 0.5 for i in ids])
    return ids, lab


def per_image(prob, lab, thr):
    pb = prob >= thr
    tp, fp, fn = (pb & lab).sum(1), (pb & ~lab).sum(1), (~pb & lab).sum(1)
    P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    R = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-9), 0.0), (tp, fp, fn)


def main(a):
    ids, lab = mow_labels(a.mow)
    arms = {}
    for x in a.arms.split(","):
        name, spec = x.split("=")
        tag, thr = spec.split("@")
        z = np.load(os.path.join(a.preds, f"pred_{tag}.npz"))
        arms[name] = (spec, *per_image(z["prob"].astype(np.float32), lab, float(thr)))
    rng = np.random.default_rng(3)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(a.boot)]
    pooled = lambda c, b: 2 * c[0][b].sum() / max(2 * c[0][b].sum() + c[1][b].sum() + c[2][b].sum(), 1)
    ci = lambda x: [round(float(np.mean(x)), a.ndp), [round(float(v), a.ndp) for v in np.percentile(x, [2.5, 97.5])]]
    out = dict(images=len(ids), boot=a.boot, rows={}, pairs={})
    for k, (spec, f1, c) in arms.items():
        out["rows"][k] = dict(spec=spec, F1=float(f1.mean()), F1_ci=ci([f1[b].mean() for b in boots]), pooledF1=float(pooled(c, slice(None))))
        print("mow", k, spec, json.dumps(out["rows"][k]), flush=True)
    for p in [x for x in a.pairs.split(",") if x]:
        u, v = p.split("-")
        out["pairs"][p] = dict(F1=ci([arms[u][1][b].mean() - arms[v][1][b].mean() for b in boots]),
                               pooledF1=ci([pooled(arms[u][2], b) - pooled(arms[v][2], b) for b in boots]))
        print("mow", p, json.dumps(out["pairs"][p]), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mow", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "mow"))
    p.add_argument("--preds", required=True, help="directory of pred_<tag>.npz")
    p.add_argument("--arms", required=True, help="name=tag@thr,...")
    p.add_argument("--pairs", default="", help="u-v,... (arm names)")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--out", default="")
    p.add_argument("--ndp", type=int, default=4, help="decimals of the CIs / paired deltas")
    main(p.parse_args())
