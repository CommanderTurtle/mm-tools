"""Label quality check for a crop pool (e.g. EgoDex ARKit labels): run a trained model on every (or a sampled
subset of) crop and score the pool's 2D / root-relative 3D labels against the predictions. Writes per-crop scores
(<pool>/qc.jsonl, shards <pool>/qc.<i>.jsonl; read by train.finetune), a summary and review sheets (named by --tag) at
agreement quantiles (label = green, model = magenta).

    python -m train.qc_crops --root $DATA_ROOT/crops --pool egodex_part1 --ckpt <finetune ckpt> --num 20000
Keep-list for training: crops with err2d < thresh (fraction of the hand size) -> train.finetune --qc_keep.
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
from train import finetune
from train.finetune import LabeledCrops, ManoJoints, Trainable, make_batch, project


@torch.no_grad()
def main(args):
    from fah import HandEstimator, load_hand_model
    from fah.student import StudentEncoder

    dev = torch.device("cuda")
    model, cfg = load_hand_model(args.checkpoint, device=dev, backbone_dtype=torch.bfloat16, verbose=False)
    est = HandEstimator(model, cfg, input_size=args.size)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k[6:]: v for k, v in ck["trainable"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    enc = None
    if ck.get("student_name"):
        enc = StudentEncoder(ck["student_name"], pretrained=False, size=ck["args"].get("student_size") or None)
        enc.load_state_dict({k[8:]: v for k, v in ck["trainable"].items() if k.startswith("encoder.")})
        enc = enc.to(dev).eval()
    trainable = Trainable(model, enc).to(dev).eval()
    mano = ManoJoints(est).to(dev)
    finetune.LABEL_QC = 0  # score every crop, not only the ones a previous QC kept
    ds = LabeledCrops(args.root, [args.pool], size=args.size, aug=False, split="all")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), min(args.num, len(ds)), replace=False) if args.num else np.arange(len(ds))
    si, sn = map(int, args.shard.split("/"))
    idx = idx[si::sn]
    sfx = f".{si}" if sn > 1 else ""
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=64, num_workers=6)
    S = enc.input_size if enc is not None else args.size
    rows, k = [], 0
    for b in loader:
        b = {kk: v.to(dev) for kk, v in b.items()}
        x = b["img"].float() / 255.0
        ratio = S / args.size
        bs = dict(b, A=b["A"] * ratio)
        bs["img_f"] = F.interpolate(x, size=(S, S), mode="bilinear", antialias=True) if S != args.size else x
        out = trainable(make_batch(bs, S), bs["img_f"])
        j, uv = mano(out, b["K"], bs["A"])
        uv = uv / ratio  # back to label (args.size view) pixels
        lab = b["kp2d"].clone()
        no2d = (lab[..., 2].sum(1) == 0) & (b["has3d"] > 0)  # 3D-only labels (EgoDex ARKit): use their projection
        if no2d.any():
            p2 = project(b["j3d"][no2d], b["K"][no2d], b["A"][no2d])
            inv = (p2[..., 0] >= 0) & (p2[..., 0] < args.size) & (p2[..., 1] >= 0) & (p2[..., 1] < args.size)
            lab[no2d] = torch.cat([p2, inv[..., None].float()], -1)
        vis = lab[..., 2] > 0
        size = (lab[..., :2].amax(1) - lab[..., :2].amin(1)).amax(-1).clamp(min=8.0)
        e2 = ((uv - lab[..., :2]).norm(dim=-1) * vis).sum(1) / vis.sum(1).clamp(min=1) / size
        rel, g_rel = j - j[:, :1], b["j3d"] - b["j3d"][:, :1]
        e3 = (rel - g_rel).norm(dim=-1).mean(1) * 1000
        for i in range(len(e2)):
            r = ds.items[idx[k]]
            rows.append(dict(di=int(idx[k]), file=r["file"], err2d=float(e2[i]), err3d_mm=float(e3[i]) if b["has3d"][i] > 0 else None,
                             n_vis=int(vis[i].sum()), pred2d=uv[i].cpu().numpy().round(1).tolist(),
                             lab2d=lab[i].cpu().numpy().round(1).tolist()))
            k += 1
    e = np.array([r["err2d"] for r in rows])
    summ = dict(pool=args.pool, n=len(rows), err2d_median=float(np.median(e)),
                frac_below={t: float((e < t).mean()) for t in (0.05, 0.1, 0.15, 0.2, 0.3)},
                err3d_median_mm=float(np.median([r["err3d_mm"] for r in rows if r["err3d_mm"] is not None] or [np.nan])))
    print(json.dumps(summ, indent=1))
    with open(os.path.join(args.root, args.pool, f"qc{sfx}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({kk: v for kk, v in r.items() if kk not in ("pred2d", "lab2d", "di")}) + "\n")
    os.makedirs(args.out, exist_ok=True)
    json.dump(summ, open(os.path.join(args.out, f"qc_{args.pool}_{args.tag}{sfx}_summary.json"), "w"), indent=1)
    if si > 0:
        return
    # review sheets: 24 crops per agreement quantile
    order = np.argsort(e)
    for q in (0.1, 0.5, 0.8, 0.9, 0.97):
        sel = order[int(q * len(order)):int(q * len(order)) + 24]
        tiles = []
        for si in sel:
            r = rows[si]
            v = ds[r["di"]]
            img = np.ascontiguousarray(v["img"].permute(1, 2, 0).numpy()[:, :, ::-1])
            lab = np.asarray(r["lab2d"])
            t = img.copy()
            draw_hand(t, lab[:, :2], lab[:, 2] > 0, thick=2)
            p = img.copy()
            draw_hand(p, np.asarray(r["pred2d"]), thick=2)
            tile = np.concatenate([t, p], 1)
            cv2.putText(tile, f"err {r['err2d']:.3f}", (4, 18), 0, 0.55, (0, 255, 255), 1)
            tiles.append(cv2.resize(tile, (384, 192)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        cv2.imwrite(os.path.join(args.out, f"qc_{args.pool}_{args.tag}_q{int(q * 100):02d}.jpg"), sheet)
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", default="qc")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--num", type=int, default=20000)
    p.add_argument("--out", default="output/qc")
    p.add_argument("--shard", default="0/1", help="i/n: score every n-th sampled crop starting at i")
    main(p.parse_args())
