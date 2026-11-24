"""Assemble a FAB bundle: one frozen body + the heads trained on it in separate frozen-body runs (train_fab.py
--contact_only / --vis_only / --g1_only / --parts_only with --init <body>). Every head checkpoint must equal the body
outside its own head (bit-equal shared tensors). The bundle holds the body's weights (EMA folded in), each head's
weights, and:
  args       the body's args, plus the heads' structure args (parts / mask) and head_args / bundle provenance
  body_args  the body's args unchanged: FABPredictor derives every body-path default from them
             (fab/predictor.body_args: foot target, ...), whatever the heads' runs used

    python scripts/build_bundle.py --body runs/fab-l-body/model.pt --contact runs/fab-l-contact/model.pt \
        --vis runs/fab-l-visibility/model.pt --g1 runs/fab-l-g1/model.pt --parts runs/fab-l-parts/model.pt \
        --out runs/fab-l-bundle/model.pt [--check_images "frames/*.jpg"]

--check_images (a GPU): after writing, check_bundle() runs FABPredictor on those images and requires (1) the bundle's
body outputs to equal the body run's, (2) each head's outputs to equal its own checkpoint's, (3) the outputs other than
the part maps to be identical with and without the part head (max |diff| <= --tol).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEADS = {"vis": ("vis_head.",), "contact": ("contact_head.",), "g1": ("g1_head.",), "parts": ("dense_head.part_",),
         "mask": ("dense_head.mask_",)}
HEAD_ARGS = {"parts": ("dense_parts", "part_depth"), "mask": ("mask_query", "mask_depth")}
CONFIG = ("dense_head.p25d",)  # config buffers a head run may set differently (its --dense_25d training mix): the body's kept
# what each head reads besides itself: G1 = the MHR decoder's pose token + keypoints only (decoder_model.py), so its run's
# dense head (not loaded there: re-initialised, frozen) is not compared; the others read the dense head's features
READS_DENSE = {"vis": True, "contact": True, "g1": False, "parts": True, "mask": True}


def weights(st):
    return {**st["model"], **(st.get("ema") or {})}


BODY_KEYS = ("mesh_smpl_fit", "mesh_mhr", "mesh_smpl", "keypoints2d", "joints_h36m14", "smpl_pose", "smpl_betas", "cam_t")
HEAD_KEYS = {"contact": ("contact",), "vis": ("keypoint_visibility",), "g1": ("g1_q", "g1_root")}


def check_bundle(bundle, body, heads, images, tol=1e-4, max_images=12):
    """bundle / body: checkpoint paths; heads: {name: head checkpoint} (contact / vis / g1 / parts); images: list of
    image paths (people in view). Boxes come from the released detector once and are reused for every model.
    -> {check: (max |diff|, ok)}; raises AssertionError listing the failures."""
    import tempfile

    import cv2
    import numpy as np
    import torch

    from fab.predictor import FABPredictor

    imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in list(images)[:max_images]]
    det = FABPredictor(bundle, fast=False)
    boxes = []
    for im in imgs:
        pp = det.predict(im)
        boxes.append(np.stack([np.asarray(q["box"], np.float32) for q in pp]) if pp else None)
    del det
    torch.cuda.empty_cache()
    keep = [i for i, b in enumerate(boxes) if b is not None]
    if not keep:
        raise ValueError("check_bundle: no person detected in the check images")
    imgs, boxes = [imgs[i] for i in keep], [boxes[i] for i in keep]

    def run(ck, **kw):
        fab = FABPredictor(ck, detector="", fast=False, **kw)
        outs = [fab.predict(im, None, b) for im, b in zip(imgs, boxes)]
        del fab
        torch.cuda.empty_cache()
        return outs

    def maxdiff(a_, b_, keys):
        d = {}
        for k in keys:
            vals = [float(np.abs(np.asarray(x[k], np.float64) - np.asarray(y[k], np.float64)).max())
                    for pa, pb in zip(a_, b_) for x, y in zip(pa, pb) if k in x and k in y]
            if vals:
                d[k] = max(vals)
        return d

    res = {}
    # (1) body path: the visibility head legitimately gates the 2D foot weights; held off so only the body path is compared
    ref = run(bundle, foot_2d_vis=False)
    for k, v in maxdiff(run(body, foot_2d_vis=False), ref, BODY_KEYS).items():
        res[f"body:{k}"] = (v, v <= tol)
    ref = run(bundle)
    for name, ck in heads.items():
        if name not in HEAD_KEYS:
            continue
        for k, v in maxdiff(run(ck), ref, HEAD_KEYS[name]).items():
            res[f"{name}:{k}"] = (v, v <= tol)
    if heads.get("parts"):  # (3) the part head changes nothing else
        st = torch.load(bundle, map_location="cpu", weights_only=False)
        st["model"] = {k: v for k, v in st["model"].items() if not k.startswith("dense_head.part_")}
        with tempfile.TemporaryDirectory() as d:
            nop = Path(d) / "model.pt"
            torch.save(st, nop)
            other = sorted({k for pp in ref for q in pp for k in q if not k.startswith("parts")})
            for k, v in maxdiff(run(nop), ref, other).items():
                res[f"parts-off:{k}"] = (v, v <= tol)
    for k, (v, ok) in res.items():
        print(f"check {k}: max |diff| {v:.2e} {'OK' if ok else 'FAIL'}", flush=True)
    bad = [k for k, (_, ok) in res.items() if not ok]
    assert not bad, f"bundle checks failed: {bad}"
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", required=True)
    for h in HEADS:
        p.add_argument(f"--{h}", default="", help=f"head checkpoint trained on the frozen body ({', '.join(HEADS[h])}*)")
    p.add_argument("--out", required=True)
    p.add_argument("--check_images", default="", help="glob of images with people: run check_bundle() after writing (GPU)")
    p.add_argument("--tol", type=float, default=1e-4, help="check_bundle tolerance (m / probability / rad)")
    a = p.parse_args()
    body = torch.load(a.body, map_location="cpu", weights_only=False)
    w = weights(body)
    ta = dict(body.get("args", {}))
    out_args, head_args = dict(ta), {}
    all_heads = tuple(x for v in HEADS.values() for x in v)
    for h, pre in HEADS.items():
        f = getattr(a, h)
        if not f:
            continue
        st = torch.load(f, map_location="cpu", weights_only=False)
        hw = weights(st)
        shared = [k for k in hw if k in w and not k.startswith(all_heads) and k not in CONFIG]
        diff = lambda k: hw[k].shape != w[k].shape or not torch.equal(hw[k].to(w[k].dtype), w[k])  # noqa: E731
        if not READS_DENSE[h]:
            nd = sum(diff(k) for k in shared if k.startswith("dense_head."))
            if nd:
                print(f"{h}: {nd} dense-head tensors differ in {f} (not read by this head: ignored)", flush=True)
            shared = [k for k in shared if not k.startswith("dense_head.")]
        bad = [k for k in shared if diff(k)]
        if bad:
            raise ValueError(f"{f}: differs from the body outside the heads ({len(bad)} tensors, e.g. {bad[:3]}) -- "
                             "not trained on this frozen body")
        mine = {k: v for k, v in hw.items() if k.startswith(pre)}
        if not mine:
            raise ValueError(f"{f}: no {pre} weights")
        w = {k: v for k, v in w.items() if not k.startswith(pre)}  # the body's own copy of the head (e.g. g1) replaced
        w.update(mine)
        hargs = dict(st.get("args", {}))
        for k in HEAD_ARGS.get(h, ()):
            out_args[k] = hargs.get(k, out_args.get(k))
        head_args[h] = {"checkpoint": f, "step": st.get("step"), "args": hargs}
        print(f"{h}: {len(mine)} tensors from {f} ({len(shared)} shared tensors equal to the body)", flush=True)
    out_args["head_args"] = head_args
    out_args["bundle"] = {"body": a.body, **{h: v["checkpoint"] for h, v in head_args.items()}}
    out = {"model": w, "ema": None, "step": body.get("step"), "args": out_args, "body_args": ta}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print("wrote", a.out, flush=True)
    if a.check_images:
        import glob

        check_bundle(a.out, a.body, {h: getattr(a, h) for h in HEADS if getattr(a, h)}, sorted(glob.glob(a.check_images)), a.tol)


if __name__ == "__main__":
    main()
