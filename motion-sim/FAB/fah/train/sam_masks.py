"""SAM 2.1 pseudo instance masks for a crop pool (train.build_crops layout), prompted per hand with its
box (mapped into the 640 crop) + up to N visible GT keypoints as positive points (labelled pools).

    python -m train.sam_masks --pool $DATA_ROOT/crops/coco_wb --shard 0 --num_shards 4
-> <pool>/masks/<file>.png (uint8 0/255 at crop resolution) + <pool>/masks_index.<shard>.jsonl (file, score)

--model sam3: SAM 3 (facebookresearch/sam3, its interactive SAM 1 task head; sam3 env) with the same prompts, e.g.
    python -m train.sam_masks --pool $DATA_ROOT/crops/coco_wb --model sam3 --ckpt sam3.pt --out_dir masks_sam3 --ref_dir masks
-> <pool>/masks_sam3/<file>.png + <pool>/masks_sam3/masks_index.<shard>.jsonl; every index line also carries the mask QC
   kp_in (visible keypoints inside the 3 px dilated mask, >= 3 visible), fill (mask area / box area) and, with --ref_dir,
   iou_ref / kp_in_ref (IoU with the pool's existing mask of that crop, e.g. the SAM 2.1 one, and that mask's kp_in)
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

# keypoints used as positive point prompts (wrist, MCPs, finger tips), OpenPose order
PROMPT_JOINTS = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]


def crop_box(r):
    """The hand box in crop pixels (the record's image box, mirrored like the crop for left hands)."""
    x1, y1, x2, y2 = r["box"]
    if not r["is_right"]:
        w = r["hw"][1]
        x1, x2 = w - 1 - x2, w - 1 - x1
    M = np.asarray(r["M"], np.float64)
    p = np.array([[x1, y1], [x2, y2]]) @ M[:, :2].T + M[:, 2]
    return p.ravel()


class Sam3Predictor:
    """SAM 3's interactive (SAM 1 task) head behind SAM2ImagePredictor's batch interface."""

    def __init__(self, ckpt="", compile=False):
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.model = build_sam3_image_model(checkpoint_path=ckpt or None, load_from_HF=not ckpt, enable_inst_interactivity=True,
                                            compile=compile)
        self.proc, self.image, self.state = Sam3Processor(self.model), Image.fromarray, None

    def set_image_batch(self, imgs):
        self.state = self.proc.set_image_batch([self.image(i) for i in imgs])

    def predict_batch(self, **kw):
        return self.model.predict_inst_batch(self.state, **kw)


def mask_qc(m, box, kp, ref=None):
    """kp_in: visible keypoints inside the 3 px dilated mask (>= 3 visible); fill: mask area / box area; iou_ref: IoU
    with a reference mask of the same crop."""
    x1, y1, x2, y2 = box
    vis = kp[:, 2] > 0
    kp_in = None
    if vis.sum() >= 3:
        md = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        xy = np.clip(kp[vis, :2].round().astype(int), 0, [m.shape[1] - 1, m.shape[0] - 1])
        kp_in = float(md[xy[:, 1], xy[:, 0]].mean())
    q = dict(kp_in=kp_in, fill=float(m.sum() / max((x2 - x1) * (y2 - y1), 1.0)))
    if ref is not None:
        q["iou_ref"] = float((m & ref).sum() / max((m | ref).sum(), 1))
        q["kp_in_ref"] = mask_qc(ref, box, kp)["kp_in"]
    return q


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.model == "sam3":
        predictor = Sam3Predictor(args.ckpt, args.compile)
    else:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(build_sam2_hf(args.model, device="cuda"))
    recs = [json.loads(l) for l in open(os.path.join(args.pool, "index.jsonl"))]
    if args.qc_keep:  # only crops whose labels passed train.qc_crops (EgoDex), every stride-th of them
        import glob

        qf = glob.glob(os.path.join(args.pool, "qc.jsonl")) + glob.glob(os.path.join(args.pool, "qc.[0-9]*.jsonl"))
        keep = {q["file"] for f in qf for q in map(json.loads, open(f))
                if q["err2d"] < args.qc_keep}
        recs = [r for r in recs if r["file"] in keep]
    if args.ref_only:  # the crops the reference masks were made for (e.g. EgoDex: its SAM 2.1 run's --qc_keep / --stride subset)
        import glob

        ri = os.path.join(args.pool, "masks_index.*.jsonl") if args.ref_dir == "masks" else os.path.join(args.pool, args.ref_dir, "..", "masks_index.*.jsonl")
        have = {json.loads(l)["file"] for f in glob.glob(ri) for l in open(f)}
        recs = [r for r in recs if r["file"] in have]
    recs = recs[:: args.stride][args.shard :: args.num_shards]
    out_dir = os.path.join(args.pool, args.out_dir)  # a directory other than "masks" keeps them out of the training mask losses
    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir if args.out_dir != "masks" else args.pool, f"masks_index.{args.shard}.jsonl")
    done = set()
    if os.path.exists(idx_path):
        done = {json.loads(l)["file"] for l in open(idx_path)}
    recs = [r for r in recs if r["file"] not in done]
    fout = open(idx_path, "a")
    print(f"{len(recs)} crops to label ({len(done)} done)", flush=True)
    for b in range(0, len(recs), args.batch):
        chunk = recs[b : b + args.batch]
        imgs = [cv2.imread(os.path.join(args.pool, r["file"]))[:, :, ::-1].copy() for r in chunk]
        boxes, pts, lbls, kps = [], [], [], []
        for r in chunk:
            boxes.append(crop_box(r)[None])
            kp = np.asarray(r.get("kp2d", np.zeros((21, 3))), np.float64)
            if not kp[:, 2].any() and "joints3d" in r and "K" in r:  # 3D-only labels: their projection into the crop
                j = np.asarray(r["joints3d"], np.float64) @ np.asarray(r["K"], np.float64).T
                M = np.asarray(r["M"], np.float64)
                uv = (j[:, :2] / j[:, 2:3]) @ M[:, :2].T + M[:, 2]
                kp = np.concatenate([uv, ((uv >= 0) & (uv < 640)).all(1, keepdims=True)], 1)
            kps.append(kp)
            sel = [j for j in PROMPT_JOINTS if kp[j, 2] > 0][: args.max_points]
            if sel:
                pts.append(kp[sel, :2][None])
                lbls.append(np.ones((1, len(sel)), np.int32))
            else:
                pts.append(None)
                lbls.append(None)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image_batch(imgs)
            masks, scores, _ = predictor.predict_batch(point_coords_batch=pts, point_labels_batch=lbls, box_batch=boxes,
                                                        multimask_output=False)
        for r, m, s, box, kp in zip(chunk, masks, scores, boxes, kps):
            m = np.asarray(m).reshape(-1, *np.asarray(m).shape[-2:])[0] > 0
            path = os.path.join(out_dir, os.path.splitext(r["file"])[0] + ".png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, m.astype(np.uint8) * 255)
            rec = dict(file=r["file"], score=float(np.asarray(s).ravel()[0]), area=int(m.sum()))
            if args.model == "sam3" or args.ref_dir:
                ref = cv2.imread(os.path.join(args.pool, args.ref_dir, os.path.splitext(r["file"])[0] + ".png"), 0) if args.ref_dir else None
                rec.update(mask_qc(m, box[0], kp, None if ref is None else ref > 127))
            fout.write(json.dumps(rec) + "\n")
        if (b // args.batch) % 50 == 0:
            fout.flush()
            print(f"{b + len(chunk)}/{len(recs)}", flush=True)
    fout.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--model", default="facebook/sam2.1-hiera-large", help="SAM 2.1 HF model, or sam3 (SAM 3, the sam3 env)")
    p.add_argument("--ckpt", default="", help="sam3: sam3.pt (default: facebook/sam3 from the HF hub)")
    p.add_argument("--ref_dir", default="", help="existing masks of the pool to compare with (iou_ref), e.g. masks")
    p.add_argument("--ref_only", action="store_true", help="only the crops listed in the reference masks' index")
    p.add_argument("--compile", action="store_true", help="sam3: torch.compile the image model (~15 % faster encoder)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_points", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--qc_keep", type=float, default=0.0, help="keep crops with qc.jsonl / qc.<i>.jsonl err2d below this")
    p.add_argument("--stride", type=int, default=1, help="label every stride-th crop")
    p.add_argument("--out_dir", default="masks", help="mask directory inside the pool (e.g. sam_parts/masks: part-label use only)")
    main(p.parse_args())
