"""Paired sequence-bootstrap confidence intervals for the contact tables (1,000 resamples of the sequences, AP on the first 50).

    gt     GT-box test sets (bench.contact_eval predict / score files): arms 'name=pred_tag@threshold' scored on the same hands;
           per arm pooled vertex F1, AP, per-hand F1 (HACO's protocol) and hand-flag F1, each with its CI, and every arm minus the
           first one (paired: the same resampled sequences).
    video  full-pipeline rows (bench.contact_eval video --out <arm>/video_<dataset>.json -> video_<dataset>_rows.npz): per arm the
           image output (--image s1c: the calibrated head's own map, or s1) and the video output (s4: the blend + FAH-Video flag;
           s2: the video model alone) with pooled vertex F1, AP, per-hand F1, flag F1, onset / release / stationary vertex F1,
           hover / free false-contact rates and flag switches per 100 hand-frames; video - image per arm and arm - first arm.

Sequences: HOT3D = the recording (index.jsonl sequence_id), DexYCB = date_time (all camera views of a capture together),
others = the clip (gt: the src sequence). --subset nochair drops HOI4D Chair (category C20), clean keeps Kettle / Knife / Mug
(C12, C13, C2).

    python -m bench.contact_ci gt --set hot3d_bench --arms haco=haco_haco_hand2x@0.56,fah=fah@cal0.68 --vidx $WEIGHTS/fah/model.pt --out ci_hot3d.json
    python -m bench.contact_ci video --dataset hoi4d --set hoi4d_test --subset nochair --arms fah=$RUNS/fah_video --out ci_hoi4d.json

gt seed groups: an arm may join several runs with '+' (name=tag_s0@cal0.68+tag_s1@cal0.68+...); its value
per resample is the mean over its runs (the same resampled sequences for every run), its point value the mean of the runs';
--pairs adds arm differences besides every arm minus the first.

    python -m bench.contact_ci gt --set hot3d_bench --vidx $WEIGHTS/fah/model.pt --out ci_geo.json --pairs dense-mano,mano-image \
        --arms image=nogeo_s0@cal0.68+nogeo_s1@cal0.68+nogeo_s2@cal0.68,mano=mano_s0@cal0.68+...,dense=dense_s0@cal0.68+...
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.contact_eval import TEST, ap_score  # noqa: E402

NAMES = ["vF1", "AP", "hF1", "flagF1", "onset_vF1", "release_vF1", "stationary_vF1", "hover_fp", "free_fp", "flips_per100"]


def sequences(test_root, set_name, files):
    """Bootstrap unit per row: HOT3D recording (index.jsonl sequence_id), DexYCB date_time, else the clip dir."""
    if set_name.startswith("hot3d"):
        m = {}
        for line in open(os.path.join(test_root, set_name, "index.jsonl")):
            r = json.loads(line)
            m[r["file"].split("/")[0]] = r.get("sequence_id") or r["file"].split("/")[0]
        return np.asarray([m.get(f.split("/")[0], f.split("/")[0]) for f in files])
    if set_name.startswith("dexycb"):
        return np.asarray(["_".join(f.split("/")[0].split("_")[:2]) for f in files])
    return np.asarray([f.rsplit("/", 1)[0] if "/" in f else f for f in files])


def subset_rows(clips, sub):
    if sub == "clean":  # HOI4D Kettle (C12) / Knife (C13) / Mug (C2)
        return np.array([re.search(r"_C(\d+)_", c).group(1) in ("12", "13", "2") for c in clips])
    if sub == "nochair":  # HOI4D without Chair (C20)
        return np.array([re.search(r"_C(\d+)_", c).group(1) != "20" for c in clips])
    return np.ones(len(clips), bool)


def boots_of(seq, idx, n, seed):
    us = np.unique(seq[idx])
    by = {u: idx[seq[idx] == u] for u in us}
    rng = np.random.default_rng(seed)
    return us, [np.concatenate([by[u] for u in rng.choice(us, len(us))]) for _ in range(n)]


DIGITS = 4  # --digits


def ci(d):
    d = d[np.isfinite(d)]
    return [round(float(d.mean()), DIGITS), [round(float(x), DIGITS) for x in np.percentile(d, [2.5, 97.5])]] if len(d) else [float("nan"), [float("nan")] * 2]


def gt(a):
    root = os.path.join(a.test_root, a.set)
    meta = [json.loads(line) for line in open(os.path.join(root, "contact.jsonl"))]
    files = [m["file"] for m in meta]
    lab = np.load(os.path.join(root, "sdf.npy")).astype(np.float32) < 10
    key = {}
    for line in open(os.path.join(root, "index.jsonl")):  # the row's sequence: sequence_id, else the src's sequence
        q = json.loads(line)
        key[q["file"]] = q.get("sequence_id") or q.get("src", q["file"]).split(":")[0]
    seq = np.asarray([key[f] for f in files])
    objs = a.objs.split(",") if a.objs else {"clean": ["Kettle", "Knife", "Mug"]}.get(a.subset)
    rows = np.asarray([(m.get("obj") in objs) if objs else not (a.subset == "nochair" and m.get("obj") == "Chair") for m in meta])
    gtf = lab.any(1)
    arms, runs, cnt, prob = [], {}, {}, {}
    for item in a.arms.split(","):
        name, rest = item.split("=", 1)
        arms.append(name)
        runs[name] = []
        for k, spec in enumerate(rest.split("+")):  # seed group: several runs of one arm
            tag, thr = spec.rsplit("@", 1)
            z = np.load(os.path.join(root, f"pred_{tag}.npz"))
            assert z["files"].tolist() == files, f"{tag}: prediction rows must follow contact.jsonl"
            if thr.startswith("cal"):  # an uncalibrated head's predictions as the calibrated head outputs them, scored at 0.5
                from bench.contact_eval import posthoc_cal

                p, thr = posthoc_cal(z["prob_nn"], float(thr[3:]), a.vidx), "0.5"
            else:
                p = z["prob"].astype(np.float32)
            pb = p >= float(thr)
            r = (name, k)
            runs[name].append(r)
            prob[r] = p
            cnt[r] = ((pb & lab).sum(1), (pb & ~lab).sum(1), (~pb & lab).sum(1), z["hand"] > 0.5)

    def met(n, sel):
        tp, fp, fn, hf = (x[sel] for x in cnt[n])
        g = gtf[sel]
        P, R = tp[g] / np.maximum(tp[g] + fp[g], 1), tp[g] / np.maximum(tp[g] + fn[g], 1)
        return np.array([2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1), np.mean(2 * P * R / np.maximum(P + R, 1e-9)),
                         2 * (hf & g).sum() / max(hf.sum() + g.sum(), 1)])

    idx = np.nonzero(rows)[0]
    us, boots = boots_of(seq, idx, a.boot, a.seed)
    B = {n: np.mean([[met(r, b) for b in boots] for r in runs[n]], 0) for n in arms}  # seed groups: mean over the runs per resample
    APb = {n: np.mean([[ap_score(prob[r][b].ravel(), lab[b].ravel()) for b in boots[:a.ap_boot]] for r in runs[n]], 0) for n in arms}
    out = dict(set=a.set, subset=a.objs or a.subset or "all", hands=len(idx), sequences=int(len(us)), arms={})
    for n in arms:
        full = np.mean([met(r, idx) for r in runs[n]], 0)
        r = {m: round(float(full[i]), DIGITS) for i, m in enumerate(["vF1", "hF1", "flagF1"])}
        r["AP"] = round(float(np.mean([ap_score(prob[x][idx].ravel(), lab[idx].ravel()) for x in runs[n]])), DIGITS)
        r["runs"] = len(runs[n])
        r.update({f"{m}_ci": ci(B[n][:, i])[1] for i, m in enumerate(["vF1", "hF1", "flagF1"])})
        r["AP_ci"] = ci(APb[n])[1]
        if n != arms[0]:
            r[f"minus_{arms[0]}"] = {m: ci(B[n][:, i] - B[arms[0]][:, i]) for i, m in enumerate(["vF1", "hF1", "flagF1"])}
            r[f"minus_{arms[0]}"]["AP"] = ci(APb[n] - APb[arms[0]])
        out["arms"][n] = r
    for pr in [x for x in a.pairs.split(",") if x]:  # extra arm differences, e.g. dense-mano
        x, y = pr.split("-")
        out.setdefault("pairs", {})[pr] = {m: ci(B[x][:, i] - B[y][:, i]) for i, m in enumerate(["vF1", "hF1", "flagF1"])}
        out["pairs"][pr]["AP"] = ci(APb[x] - APb[y])
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out), flush=True)


def video(a):
    Z, names = {}, []
    for item in a.arms.split(","):
        name, d = item.split("=", 1)
        Z[name] = np.load(os.path.join(d, f"video_{a.dataset}_rows.npz"))
        names.append(name)
    z0 = Z[names[0]]
    clips = z0["clip"]
    for n in names:
        assert (Z[n]["clip"] == clips).all() and (Z[n]["gt_any"] == z0["gt_any"]).all(), f"{n}: rows differ from {names[0]}"
    sel = np.nonzero(subset_rows(clips, a.subset))[0]
    if a.set.startswith("hot3d"):
        seq = sequences(a.test_root, a.set, [c + "/x" for c in clips])
    elif a.dataset == "dexycb":
        seq = np.asarray(["_".join(c.split("_")[:2]) for c in clips])
    else:
        seq = np.asarray(clips)
    us, boots = boots_of(seq, sel, a.boot, a.seed)
    g = z0["gt_any"]

    def met(z, k, fk, s, ap=True):
        tp, fp, fn, hf = z[k][s, 0], z[k][s, 1], z[k][s, 2], z[k][s, 3] > 0
        gg = g[s]
        P, R = tp[gg] / np.maximum(tp[gg] + fp[gg], 1), tp[gg] / np.maximum(tp[gg] + fn[gg], 1)
        r = [2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1),
             ap_score(z[f"prob_{k}"][s].astype(np.float32).ravel(), z["lab"][s].ravel()) if ap else np.nan,
             np.mean(2 * P * R / np.maximum(P + R, 1e-9)), 2 * (hf & gg).sum() / max(hf.sum() + gg.sum(), 1)]
        for c in ("onset", "release", "stationary"):
            cs = z[f"cat_{c}"][s]
            r.append(2 * tp[cs].sum() / max(2 * tp[cs].sum() + fp[cs].sum() + fn[cs].sum(), 1))
        for c in ("hover", "free"):
            cs = z[f"cat_{c}"][s]
            r.append(fp[cs].sum() / max(cs.sum() * 778, 1))
        r.append(100 * z[fk][s].sum() / max(len(s), 1))
        return np.array(r, np.float64)

    res = dict(dataset=a.dataset, subset=a.subset, sequences=int(len(us)), hand_frames=int(len(sel)), arms={})
    B = {}
    for n in names:
        z, r = Z[n], {}
        for out_name, k, fk in (("image", a.image, "flip1"), ("video", "s4", "flip2"), ("video_alone", "s2", "flip2")):
            r[out_name] = dict(zip(NAMES, np.round(met(z, k, fk, sel), 4).tolist()))
            bb = np.array([met(z, k, fk, b, ap=False) for b in boots])
            bb[:a.ap_boot, 1] = [ap_score(z[f"prob_{k}"][b].astype(np.float32).ravel(), z["lab"][b].ravel()) for b in boots[:a.ap_boot]]
            B[(n, out_name)] = bb
        D = B[(n, "video")] - B[(n, "image")]
        r["video-image"] = {m: ci(D[:a.ap_boot if m == "AP" else None, i]) for i, m in enumerate(NAMES)}
        if n != names[0]:
            D = B[(n, "video")] - B[(names[0], "video")]
            r[f"video-{names[0]}"] = {m: ci(D[:a.ap_boot if m == "AP" else None, i]) for i, m in enumerate(NAMES)}
        res["arms"][n] = r
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["gt", "video"])
    p.add_argument("--test_root", default=TEST)
    p.add_argument("--set", default="", help="gt: the test set; video: its name for the HOT3D sequence ids (hot3d_bench)")
    p.add_argument("--dataset", default="", help="video: the rows' dataset (video_<dataset>_rows.npz)")
    p.add_argument("--arms", required=True, help="gt: name=pred_tag@thr (or @cal<thr>: post-hoc calibrated, needs --vidx; '+' joins "
                   "the runs of a seed group),...; "
                   "video: name=<dir of video_<dataset>_rows.npz>,...")
    p.add_argument("--vidx", default="", help="gt @cal arms: a checkpoint with the head's dense point indices (model.pt)")
    p.add_argument("--subset", default="", help="nochair | clean (HOI4D; gt: by the rows' contact.jsonl obj, video: by clip category)")
    p.add_argument("--objs", default="", help="gt: only the rows of these objects (contact.jsonl obj), e.g. Kettle,Knife,Mug,ToyCar")
    p.add_argument("--image", default="s1c", help="video: the image row (s1c = the calibrated head's own map, s1 = the debiased map)")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--ap_boot", type=int, default=50)
    p.add_argument("--seed", type=int, default=-1, help="resampling seed (default: 2 for gt, 5 for video, the paper's)")
    p.add_argument("--pairs", default="", help="gt: extra arm differences x-y,... (besides every arm minus the first)")
    p.add_argument("--digits", type=int, default=4, help="decimals of the stored values / intervals")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    DIGITS = a.digits
    a.seed = a.seed if a.seed >= 0 else (2 if a.cmd == "gt" else 5)
    dict(gt=gt, video=video)[a.cmd](a)
