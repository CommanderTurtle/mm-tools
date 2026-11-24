"""Finger-part correctness of the dense surface points on held-out labelled crops (bench.parts.dense; the paper's
supplementary dense-point finger assignment).

Each arm is a model run on every crop with the crop's label box (right hand); its raw dense points, their MANO parts and
the head's visibility are scored against the crop's reference part labels. Every arm is also scored with the first
arm's visibility (ref_*), and each later arm gets a paired bootstrap against each earlier one (95 % interval; --boot_unit
clip resamples whole source videos). Predictions are cached per arm (<out_dir>/preds_<arm>.pkl, checked against the
model / options / crops they were made with); --preds scores stored predictions without a model.

    python scripts/dense_part_eval.py --crops $DATA_ROOT/crops --arms fah=fah --geometry --out_dir output/dense/fah
    python scripts/dense_part_eval.py --crops $DATA_ROOT/crops --arms a=runs/a/final.pt,b=runs/b/final.pt \
        --method fah_pipeline --method_args '<json>' --out_dir output/dense/ab
    python scripts/dense_part_eval.py --crops $DATA_ROOT/crops --preds ctl=x/preds_ctl.pkl,new=x/preds_new.pkl --out_dir y
Arms: name=<FAH model> (fah-s | fah-b | fah | fah-h, --method fah) or name=<checkpoint> (finetune_ckpt).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.parts import dense, heldout  # noqa: E402
from train.build_crops import crop_box  # noqa: E402
from train.part_masks import mano_parts  # noqa: E402

DEFAULT_SPLIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "splits", "dense_parts.json")
MODELS = ("fah-s", "fah-b", "fah", "fah-h")


def arm_args(spec, args):
    a = dict(json.loads(os.path.expandvars(args.method_args)))
    if spec in MODELS:
        if args.method != "fah":
            raise ValueError(f"arm {spec}: model names need --method fah")
        a["model"] = spec
    else:
        a["finetune_ckpt"] = os.path.realpath(os.path.expandvars(spec))
    a.update(pose_trt="", verify=False)  # the PyTorch head (dense points + visibility logits)
    return a


def infer(arm, spec, sel, args):
    """Raw head outputs per crop: dense points (crop px), visibility logits, view side (crop px)."""
    import torch

    from bench.run import load_method

    m = load_method(args.method, arm_args(spec, args))
    m = getattr(m, "method", m)  # fah.api.FAH wraps a FAHMethod
    est, kh = m.est, m.est.model.keypoint_head
    nb, nv = (int(x) for x in kh.dense.tolist())
    vidx = kh.dense_vidx.cpu().numpy() if getattr(kh, "dense_vidx", None) is not None else np.arange(778)
    pid = mano_parts(args.mano_dir, with_vpart=True)[2][vidx].astype(np.int64)
    cap, run = [], est._run_crops
    est._run_crops = lambda crops: cap.append(run(crops)) or cap[-1]  # the last pass (recrop) is the hand's output
    torch.backends.cuda.matmul.allow_tf32 = m.tf32
    K = np.array([[640, 0, 320], [0, 640, 320], [0, 0, 1]], np.float32)
    S = est.input_size[0]
    out = dict(arm=arm, spec=spec, pid=pid, vidx=vidx, pools={}, files={p: [r["file"] for r in recs] for p, (_, recs) in sel.items()})
    for pool, (_, recs) in sel.items():
        res = []
        for r in recs:
            img = cv2.imread(os.path.join(args.crops, pool, r["file"]))
            if img is None:
                res.append(None)
                continue
            cap.clear()
            est.process_image(img, boxes=crop_box(r)[None], is_right=np.ones(1, bool), cam_int=K)
            o = cap[-1]
            A = np.asarray(o["affine_trans"][0], np.float64)
            uv = (np.asarray(o["pt_uv"][0], np.float64) * S - A[:, 2]) @ np.linalg.inv(A[:, :2]).T
            vl = np.asarray(o["pt_vis"][0], np.float32)[21 + nb:21 + nb + nv] if "pt_vis" in o else np.full(nv, 10.0, np.float32)
            res.append(dict(file=r["file"], uv=uv[21 + nb:21 + nb + nv].astype(np.float32), vis=vl, side=float(S / A[0, 0])))
        out["pools"][pool] = res
        print(arm, pool, len(res), flush=True)
    del m, est
    torch.cuda.empty_cache()
    return out


def inference_signature(spec, sel, args):
    """Prediction caches depend on the model, inference options and crop geometry, not the scoring labels."""
    a = arm_args(spec, args)
    model = a.get("finetune_ckpt")
    stat = os.stat(model) if model else None
    geometry = {pool: [{k: r[k] for k in ("file", "box", "M", "hw", "is_right") if k in r} for r in recs] for pool, (_, recs) in sel.items()}
    return dict(version=1, method=args.method, method_args=a, checkpoint=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) if stat else None,
                crops=os.path.realpath(args.crops), mano_dir=os.path.realpath(args.mano_dir),
                geometry_sha256=hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest())


def selection(args):
    part_dirs = dict(x.split(":") for x in args.part_dirs.split(",")) if args.part_dirs else {}
    pools = args.pools.split(",") if args.pools else None
    if args.sample:
        if pools is None:
            raise ValueError("--sample needs --pools")
        return {p: (part_dirs.get(p, "masks_parts"), r) for p, r in heldout.sample(args.crops, pools, part_dirs, args.sample).items()}
    return {p: (pdir, recs) for p, (pdir, _, recs) in heldout.from_split(args.crops, heldout.load_split(args.split), pools, part_dirs).items()}


def load_predictions(args, sel):
    preds, arms = {}, []
    for a, pf in (x.split("=", 1) for x in args.preds.split(",") if x):  # stored predictions, no model
        p = pickle.load(open(pf, "rb"))
        for pool, (_, recs) in sel.items():
            if p["files"].get(pool) != [r["file"] for r in recs]:
                raise ValueError(f"{pf}: its {pool} predictions are for other crops than the selection")
        preds[a] = p
        arms.append(a)
    for a, spec in (x.split("=", 1) for x in args.arms.split(",") if x):
        pf = os.path.join(args.out_dir, f"preds_{a}.pkl")
        signature = inference_signature(spec, sel, args)
        if os.path.exists(pf):
            p = pickle.load(open(pf, "rb"))
            if p.get("inference_signature") != signature:
                raise ValueError(f"Prediction cache {pf} was made with another model / options / crops; use a new --out_dir")
        else:
            p = infer(a, spec, sel, args)
            p["inference_signature"] = signature
            pickle.dump(p, open(pf, "wb"))
        preds[a] = p
        arms.append(a)
    if not arms:
        raise ValueError("no --arms / --preds")
    return preds, arms


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    sel = selection(args)
    preds, arms = load_predictions(args, sel)
    ref = arms[0]
    for a in arms[1:]:
        for key in ("pid", "vidx"):
            if key not in preds[ref] or key not in preds[a] or not np.array_equal(preds[a][key], preds[ref][key]):
                raise ValueError(f"{a}: dense point correspondence ({key}) differs from {ref}; "
                                 "reference visibility and shared GT geometry require the same ordered points")
    lab_of = lambda pool, pdir, r: cv2.imread(os.path.join(args.crops, pool, pdir, os.path.splitext(r["file"])[0] + ".png"), cv2.IMREAD_GRAYSCALE)
    geos = {}
    if args.geometry:  # model-independent GT geometry of the GT-mesh pools' crops
        faces, fpart = mano_parts(args.mano_dir)
        p0 = preds[ref]
        vidx = np.asarray(p0["vidx"]) if "vidx" in p0 else np.arange(len(p0["pid"]))
        for pool, (pdir, recs) in sel.items():
            if not all("verts_file" in r for r in recs):
                continue
            geos[pool] = []
            for r in recs:
                lab = lab_of(pool, pdir, r)
                geos[pool].append(None if lab is None else dense.gt_geometry(r, os.path.dirname(os.path.join(args.crops, pool, r["file"])), lab,
                                                                            vidx, p0["pid"], faces, fpart))
    hands, hands_ref = {}, {}
    for a in arms:
        hands[a], hands_ref[a] = {}, {}
        for pool, (pdir, recs) in sel.items():
            hs, hr = [], []
            for i, (r, p, q) in enumerate(zip(recs, preds[a]["pools"][pool], preds[ref]["pools"][pool])):
                lab = lab_of(pool, pdir, r)
                geo = geos[pool][i] if pool in geos else None
                if p is None or q is None or lab is None or (pool in geos and geo is None):
                    hs.append(None)
                    hr.append(None)
                    continue
                hs.append(dense.score_hand(lab, p["uv"], p["vis"] > 0, preds[a]["pid"], p["side"], geo=geo))
                hr.append(dense.score_hand(lab, p["uv"], q["vis"] > 0, preds[a]["pid"], p["side"]))
            hands[a][pool], hands_ref[a][pool] = hs, hr
    pools = list(sel)
    report = {}
    for a in arms:
        rep = dict(model=preds[a].get("spec", preds[a].get("ckpt")))
        for pool in pools:
            hs = [h for h in hands[a][pool] if h is not None]
            hr = [h for h in hands_ref[a][pool] if h is not None]
            rep[pool] = dense.metrics(dense.stack(hs))
            rep[pool].update({"ref_" + k: v for k, v in dense.metrics(dense.stack(hr)).items() if k in dense.KEYS})
        rep.update(dense.group_report(rep, hands[a], pools))
        report[a] = rep
        print(a, json.dumps({g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep[g].items()
                                 if k in ("n", *dense.KEYS, "part_acc_tips", "vis_frac")} for g in (*dense.GROUPS, "all") if g in rep}), flush=True)
    common = lambda a, b, p: [i for i in range(len(sel[p][1])) if hands[a][p][i] is not None and hands[b][p][i] is not None]
    report["bootstrap"] = {}
    for i in range(1, len(arms)):
        for j in range(i):
            a, b = arms[j], arms[i]
            for g, ps in [*dense.GROUPS.items(), ("all", tuple(pools))]:
                ps = [p for p in ps if p in pools]
                if not ps:
                    continue
                ha = {p: dense.stack([hands[a][p][k] for k in common(a, b, p)]) for p in ps}
                hb = {p: dense.stack([hands[b][p][k] for k in common(a, b, p)]) for p in ps}
                cl = {p: [sel[p][1][k].get("src", sel[p][1][k]["file"]).split(":")[0] for k in common(a, b, p)] for p in ps} \
                    if args.boot_unit == "clip" else None  # train.finetune's split unit (a video / sequence)
                report["bootstrap"][f"{b}-{a}:{g}"] = dense.bootstrap(ha, hb, ps, B=args.boot, clips=cl)
    json.dump(report, open(os.path.join(args.out_dir, "dense_part_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", required=True, help="crop pools (train.build_*_crops) with their part labels")
    p.add_argument("--arms", default="", help="name=<FAH model or checkpoint>,... (the first arm is the reference for ref_* / bootstrap)")
    p.add_argument("--preds", default="", help="name=<preds_*.pkl>,... stored predictions, scored before the --arms (no model)")
    p.add_argument("--method", default="fah", choices=("fah", "fah_pipeline"))
    p.add_argument("--method_args", default='{"mesh": "none", "tips": false}')
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--pools", default="", help="subset of the split's pools (default: all)")
    p.add_argument("--part_dirs", default="", help="pool:dir,... override the split's label directories")
    p.add_argument("--sample", type=int, default=0, help="instead of the split: n held-out crops per pool by the split's rule")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--boot_unit", choices=("hand", "clip"), default="hand", help="paired bootstrap resampling unit; clip = whole videos")
    p.add_argument("--geometry", action="store_true", help="GT-mesh breakdowns (crossings, occlusion, hand size) on pools with verts_file")
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
