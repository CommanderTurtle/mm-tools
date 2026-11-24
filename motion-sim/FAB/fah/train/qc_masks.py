"""Quality check of a crop pool's hand instance masks (SAM pseudo-masks): per crop,
  kp_in    fraction of visible GT 2D keypoints inside the (3 px dilated) mask   (low: mask misses hand parts)
  mesh_iou IoU between the mask and the silhouette of a model's mesh (MANO-topology vertices rendered with the
           template faces; a model-independent check is kp_in)                    (low: wrong instance / occlusion)
  arm      fraction of mask pixels beyond the wrist line (perpendicular to wrist -> middle MCP, 0.25 hand
           lengths behind the wrist)                                               (high: arm included)
Writes <pool>/mask_qc.jsonl (read by train.finetune) and review sheets at quantiles of each score (named by --tag).

    python -m train.qc_masks --root $DATA_ROOT/crops --pool coco_wb --ckpt <finetune ckpt> --num 3000
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from bench.bench2d.gallery import draw_hand
from train.finetune import LabeledCrops, ManoJoints, Trainable, make_batch


def wrist_arm_fraction(mask, kp, vis, margin=0.25):
    if vis[0] <= 0 or vis[9] <= 0:
        return None
    d = kp[9, :2] - kp[0, :2]
    L = np.linalg.norm(d)
    if L < 4:
        return None
    d = d / L
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    s = (np.stack([xs, ys], 1) - kp[0, :2]) @ d
    return float((s < -margin * L).mean())


@torch.no_grad()
def main(args):
    from bench.common import ManoLayer
    from fah import HandEstimator, load_hand_model
    from fah.estimator import load_joint_regressor
    from fah.student import StudentEncoder
    from fah.vis import render_instance_masks

    dev = torch.device("cuda")
    model, cfg = load_hand_model(args.checkpoint, device=dev, backbone_dtype=torch.bfloat16, verbose=False)
    est = HandEstimator(model, cfg, input_size=args.size)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict({k[6:]: v for k, v in ck["trainable"].items() if k.startswith("model.")}, strict=False)
    enc = None
    if ck.get("student_name"):
        enc = StudentEncoder(ck["student_name"], pretrained=False, size=ck["args"].get("student_size") or None)
        enc.load_state_dict({k[8:]: v for k, v in ck["trainable"].items() if k.startswith("encoder.")})
        enc = enc.to(dev).eval()
    trainable = Trainable(model, enc).to(dev).eval()
    n_full = model.head_pose_hand.mhr.character_torch.mesh.rest_vertices.shape[0]
    W778 = load_joint_regressor("fah/assets/mhr_to_mano778_regressor.npz", n_full)
    cols = np.nonzero(W778.any(0))[0]
    remap = -np.ones(n_full, dtype=np.int64)
    remap[est.hand_vert_idx] = np.arange(len(est.hand_vert_idx))
    Wd, idx_d = torch.as_tensor(W778[:, cols], device=dev), torch.as_tensor(remap[cols], device=dev)
    faces = ManoLayer(args.mano_dir).faces[True]
    ds = LabeledCrops(args.root, [args.pool], size=args.size, aug=False, split="all")
    has_m = [i for i, r in enumerate(ds.items) if "mask_path" in r]
    rng = np.random.default_rng(0)
    idx = rng.choice(has_m, min(args.num, len(has_m)), replace=False) if args.num else np.asarray(has_m)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=32, num_workers=6)
    S = enc.input_size if enc is not None else args.size
    rows, k = [], 0
    for b in loader:
        b = {kk: v.to(dev) for kk, v in b.items()}
        x = b["img"].float() / 255.0
        ratio = S / args.size
        bs = dict(b, A=b["A"] * ratio)
        bs["img_f"] = F.interpolate(x, size=(S, S), mode="bilinear", antialias=True) if S != args.size else x
        out = trainable(make_batch(bs, S), bs["img_f"])
        verts = torch.einsum("vk,bkc->bvc", Wd, out["pred_vertices"][:, idx_d]) + out["pred_cam_t"][:, None]
        for i in range(len(verts)):
            K, A = b["K"][i].cpu().numpy(), b["A"][i].cpu().numpy()
            Kv = np.array([[A[0, 0] * K[0, 0], 0, A[0, 0] * K[0, 2] + A[0, 2]], [0, A[1, 1] * K[1, 1], A[1, 1] * K[1, 2] + A[1, 2]], [0, 0, 1]])
            sil = render_instance_masks([(verts[i].cpu().numpy(), faces)], Kv, (args.size, args.size)) > 0
            m = b["mask"][i].cpu().numpy() > 127
            kp = b["kp2d"][i].cpu().numpy()
            vis = kp[:, 2] > 0
            md = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            xy = np.clip(kp[vis, :2].round().astype(int), 0, args.size - 1)
            kp_in = float(md[xy[:, 1], xy[:, 0]].mean()) if vis.any() else None
            inter, union = (sil & m).sum(), (sil | m).sum()
            rows.append(dict(di=int(idx[k]), file=ds.items[idx[k]]["file"], kp_in=kp_in,
                             mesh_iou=float(inter / max(union, 1)), arm=wrist_arm_fraction(m, kp, kp[:, 2])))
            k += 1
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.root, args.pool, "mask_qc.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({kk: v for kk, v in r.items() if kk != "di"}) + "\n")
    summ = {"pool": args.pool, "n": len(rows)}
    for key in ("kp_in", "mesh_iou", "arm"):
        v = np.array([r[key] for r in rows if r[key] is not None])
        if not len(v):
            continue
        summ[key] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(os.path.join(args.out, f"mask_qc_{args.pool}_{args.tag}.json"), "w"), indent=1)
    for key in ("kp_in", "mesh_iou", "arm"):  # review sheets: lowest / median / highest of each score
        order = sorted([r for r in rows if r[key] is not None], key=lambda r: r[key])
        if not order:
            continue
        for name, sel in (("low", order[:16]), ("mid", order[len(order) // 2:len(order) // 2 + 16]), ("high", order[-16:])):
            tiles = []
            for r in sel:
                v = ds[r["di"]]
                img = np.ascontiguousarray(v["img"].permute(1, 2, 0).numpy()[:, :, ::-1])
                m = v["mask"].numpy() > 127
                t = img.copy()
                t[m] = (0.5 * t[m] + 0.5 * np.array([255, 120, 40])).astype(np.uint8)
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(t, cnt, -1, (255, 160, 60), 2)
                kp = v["kp2d"].numpy()
                draw_hand(t, kp[:, :2], kp[:, 2] > 0, thick=2)
                cv2.putText(t, f"{key} {r[key]:.2f}", (4, 18), 0, 0.55, (0, 255, 255), 1)
                tiles.append(cv2.resize(t, (256, 256)))
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
            cv2.imwrite(os.path.join(args.out, f"mask_qc_{args.pool}_{key}_{name}.jpg"), sheet)
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", default="qc")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--num", type=int, default=3000)
    p.add_argument("--out", default="output/qc")
    main(p.parse_args())
