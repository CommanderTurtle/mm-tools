"""Score stored bench2d-format predictions against the CURRENT labels of a set whose labels evolve
(hard-pose / dark sets: model-consensus labels, then human-corrected): the GT stored in the prediction pickles at run
time is replaced by the hands of --ann before running bench.bench2d.evaluate.

    python -m bench.hardpose.evaluate --ann data/hard_pose_v1/annotations.json \
        --runs output/hardpose/hp-fah:joints2d,output/hardpose/hp-wilor:joints2d_crop --out output/hardpose/metrics.json
"""
import argparse
import glob
import json
import os
import pickle
import tempfile

import numpy as np

from bench.bench2d.evaluate import evaluate_dir, match
from bench.common import box_iou


def main(args):
    ann = json.load(open(args.ann))
    gt = {r["id"]: [dict(is_right=h["is_right"], joints2d=np.asarray(h["kp2d"], np.float32), vis=np.asarray(h["vis"]),
                         bbox=np.asarray(h["bbox"], np.float32), joints3d=None, verts3d=None) for h in r["hands"]
                    if h.get("status") not in ("needs_label", "ignore")]
          for r in ann["images"]}
    ignore = {r["id"]: [np.asarray(h["bbox"], np.float32) for h in r["hands"] if h.get("status") in ("needs_label", "ignore")]
              for r in ann["images"]}  # not (yet) labelled: neither scored nor counted as false positives
    res = {"labels": ann["meta"].get("version"), "runs": {}}
    for spec in args.runs.split(","):
        d, key = (spec.split(":") + ["joints2d"])[:2]
        with tempfile.TemporaryDirectory() as tmp:
            for path in glob.glob(os.path.join(d, "*.pkl")):
                clip = pickle.load(open(path, "rb"))
                for fr in clip["frames"]:
                    fr["gt"] = gt.get(fr["key"], [])
                    ig = ignore.get(fr["key"], [])
                    if ig:  # scored hands are matched first; only the remaining predictions inside an ignored box are exempt
                        kp = fr["pred"] if key == "joints3d_fit" else [dict(p, joints2d=p.get(key, p.get("joints2d"))) for p in fr["pred"]]
                        hit = set(match(kp, fr["gt"])[0].values())
                        fr["pred"] = [p for i, p in enumerate(fr["pred"]) if i in hit or p.get("bbox") is None
                                      or all(box_iou(np.asarray(p["bbox"], np.float32), b) < 0.3 for b in ig)]
                    fr["fully_annotated"] = True
                pickle.dump(clip, open(os.path.join(tmp, os.path.basename(path)), "wb"))
            m = evaluate_dir(tmp, key)
        name = os.path.basename(d.rstrip("/")) + ("" if key == "joints2d" else f":{key}")
        res["runs"][name] = m
        a = m["all"]
        print(f"{name:40s} recall {a['recall']:.3f} prec {a['precision'] or 0:.3f} PCK@.05 {a['pck05'] or 0:.3f} "
              f"PCK@.1 {a['pck10'] or 0:.3f} AUC {a['auc'] or 0:.3f} tipPCK@.1 {a['tip_pck10'] or 0:.3f} "
              f"PCK@.1 incl.miss {a['pck10_all']:.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default="")
    main(p.parse_args())
