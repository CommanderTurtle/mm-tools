"""Part-head quality on held-out labelled crops (bench.parts.seg; the paper's supplementary
part-segmentation table, released heads).

The method runs on each stored 640 px crop twice: with the crop's label box as the only detection (the part map vs the
pool's reference labels) and with its own detector (the detected hand's mask and parts vs the pool's SAM hand mask and
the labels). Crops: a split file (default bench/splits/parts_heldout.json, the paper's 200 per pool) or --sample n
(the rule the split was drawn with). --save_preds keeps every crop's part maps / detector masks; --from_preds recomputes
the metrics from them without a model.

    python scripts/part_seg_eval.py --crops $DATA_ROOT/crops --method fah \
        --method_args '{"model": "fah", "mesh": "none", "tips": false, "parts": true}' \
        --pools hot3d_lab_mesh,dexycb_lab,oakink2_ego_lab --save_preds output/parts/fah --out output/parts/fah_3d.json
    python scripts/part_seg_eval.py --crops $DATA_ROOT/crops --from_preds output/parts/fah --out x.json
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.parts import heldout, seg  # noqa: E402
from train.build_crops import crop_box  # noqa: E402

DEFAULT_SPLIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "splits", "parts_heldout.json")


def pipeline(name, method_args):
    """The bench.methods.fah.FAHMethod behind --method fah (model and mode) or fah_pipeline (full arguments), with the
    settings the evaluation needs: no pose verification, the detector's instance masks, the raw part head unless
    parts_in_mask is set explicitly."""
    from bench.run import load_method

    a = dict(method_args)
    a.setdefault("parts_in_mask", False)  # raw head quality, measurable independently of the mask
    a.update(verify=False, det_masks=True)
    if name == "fah":
        a.setdefault("parts", True)
    m = load_method(name, a)
    m = getattr(m, "method", m)  # fah.api.FAH wraps a FAHMethod
    if getattr(m, "schedule", None) is not None:
        raise ValueError("part_seg_eval needs the image pipeline (det_every 1)")
    return m, bool(a["parts_in_mask"])


def selection(args):
    part_dirs = dict(x.split(":") for x in args.part_dirs.split(",")) if args.part_dirs else {}
    mask_dirs = dict(x.split(":") for x in args.mask_dirs.split(",")) if args.mask_dirs else {}
    pools = args.pools.split(",") if args.pools else None
    if args.sample:
        if pools is None:
            raise ValueError("--sample needs --pools")
        recs = heldout.sample(args.crops, pools, part_dirs, args.sample)
        return {p: (part_dirs.get(p, "masks_parts"), mask_dirs.get(p, "masks"), r) for p, r in recs.items()}
    return heldout.from_split(args.crops, heldout.load_split(args.split), pools, part_dirs, mask_dirs)


def main(args):
    sel = selection(args)
    if args.from_preds:
        stored = {p: pickle.load(open(os.path.join(args.from_preds, f"{p}.pkl"), "rb")) for p in sel}
        parts_in_mask = {p: s["parts_in_mask"] for p, s in stored.items()}
    else:
        method_args = json.loads(os.path.expandvars(args.method_args))
        m, pim = pipeline(args.method, method_args)
        det, box_det = m.detector, seg.BoxDetector(m.detector)
        stored, parts_in_mask = {}, {}
        for pool, (pdir, mdir, recs) in sel.items():
            P = os.path.join(args.crops, pool)
            preds = []
            for r in recs:
                img = cv2.imread(os.path.join(P, r["file"]))
                stem = os.path.splitext(r["file"])[0]
                if img is None or not os.path.exists(os.path.join(P, pdir, stem + ".png")):
                    preds.append(None)
                    continue
                preds.append(seg.predict_crop(m, box_det, det, img, crop_box(r), os.path.exists(os.path.join(P, mdir, stem + ".png"))))
            m.detector = m.est.detector = det
            stored[pool] = dict(version=1, files=[r["file"] for r in recs], preds=preds, parts_in_mask=pim, method=args.method,
                                method_args=method_args)
            parts_in_mask[pool] = pim
            if args.save_preds:
                os.makedirs(args.save_preds, exist_ok=True)
                pickle.dump(stored[pool], open(os.path.join(args.save_preds, f"{pool}.pkl"), "wb"))
    rep = {}
    for pool, (pdir, mdir, recs) in sel.items():
        P = os.path.join(args.crops, pool)
        st = stored[pool]
        if st["files"] != [r["file"] for r in recs]:
            raise ValueError(f"{pool}: stored predictions are for other crops than the selection")
        acc = seg.new_acc()
        for r, pred in zip(recs, st["preds"]):
            if pred is None:
                continue
            stem = os.path.splitext(r["file"])[0]
            lab = seg.read_label(os.path.join(P, pdir, stem + ".png"))
            if lab is None:
                continue
            mp = os.path.join(P, mdir, stem + ".png")
            sam = seg.read_label(mp) if os.path.exists(mp) else None
            seg.score_crop(acc, lab, sam, pred)
        rep[pool] = seg.report(acc, parts_in_mask[pool])
        print(pool, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep[pool].items() if not k.endswith("iou_per_class")}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", required=True, help="crop pools (train.build_*_crops) with their part labels and hand masks")
    p.add_argument("--method", default="fah", choices=("fah", "fah_pipeline"))
    p.add_argument("--method_args", default='{"model": "fah", "mesh": "none", "tips": false, "parts": true}')
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--pools", default="", help="subset of the split's pools (default: all)")
    p.add_argument("--part_dirs", default="", help="pool:dir,... override the split's label directories")
    p.add_argument("--mask_dirs", default="", help="pool:dir,... override the split's hand-mask directories")
    p.add_argument("--sample", type=int, default=0, help="instead of the split: n held-out crops per pool by the split's rule")
    p.add_argument("--save_preds", default="", help="directory: <pool>.pkl with every crop's part maps / detector masks")
    p.add_argument("--from_preds", default="", help="directory written by --save_preds: metrics only, no model")
    p.add_argument("--out", default="")
    main(p.parse_args())
