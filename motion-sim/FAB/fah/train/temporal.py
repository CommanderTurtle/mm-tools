"""Temporal refiner for video: a small transformer over a per-hand track of per-frame predictions (any per-frame
model, from bench.run pickles) that refines root-relative joints and the root position and fills frames where
the detector missed the hand. Offline (bidirectional, whole clip) or online (causal, last frame of the window).

Tracks are built from the predictions alone (one track per predicted side, highest score wins), as at test time;
training targets come from the GT hand each prediction matches (box IoU, as in bench.evaluate), and for frames
without a prediction from the GT hand of that side, so the model learns to fill misses.

    python -m train.temporal train --batch "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --train $DATA_ROOT/video_preds/hot3d/train-kpC-det800 --out <dir> [--causal]
    python -m train.temporal apply --ckpt <dir>/model.pt --pred_dir $RESULTS/hot3d/kpC-det800 \
        --out_dir $RESULTS/hot3d/kpC-det800-temporal
"""
import argparse
import glob
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.common import box_iou, match_hands

D_IN = 1 + 1 + 63 + 42 + 3  # present, score, root-relative joints (dm), normalised 2D joints, root (x/z, y/z, log z)


def frame_feats(p, K):
    j = np.asarray(p["joints3d"], np.float64)
    rel = (j - j[:1]) * 10.0
    uv = np.asarray(p["joints2d"], np.float64)
    nuv = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]], 1)
    z = max(j[0, 2], 1e-3)
    root = np.array([j[0, 0] / z, j[0, 1] / z, np.log(z)])
    return np.concatenate([[1.0, p.get("score", 1.0)], rel.ravel(), nuv.ravel(), root])


def root_to_xyz(r):
    z = np.exp(r[..., 2])
    return np.stack([r[..., 0] * z, r[..., 1] * z, z], -1)


def build_tracks(clip, with_gt=True):
    """-> {side: dict(x (T, D_IN), present (T,), pred_idx (T,), gt_rel (T, 21, 3), gt_root (T, 3), has_gt (T,))}"""
    T = len(clip["frames"])
    tracks = {}
    for side in (True, False):
        x = np.zeros((T, D_IN), np.float32)
        tok = None  # per-hand latent of the per-frame model (if its predictions carry "token")
        pidx = -np.ones(T, np.int64)
        g_rel, g_root, has = np.zeros((T, 21, 3), np.float32), np.zeros((T, 3), np.float32), np.zeros(T, np.float32)
        for t, fr in enumerate(clip["frames"]):
            K = np.asarray(fr["K"], np.float64)
            cand = [(p.get("score", 0), k) for k, p in enumerate(fr["pred"])
                    if bool(p["is_right"]) == side and p.get("joints3d") is not None and np.isfinite(p["joints3d"]).all()]
            if cand:
                k = max(cand)[1]
                x[t] = frame_feats(fr["pred"][k], K)
                pidx[t] = k
                if fr["pred"][k].get("token") is not None:
                    if tok is None:
                        tok = np.zeros((T, len(fr["pred"][k]["token"])), np.float16)
                    tok[t] = fr["pred"][k]["token"]
            if not with_gt:
                continue
            g = None
            if pidx[t] >= 0:  # the GT hand this prediction matches (box IoU, any side)
                ious = [(box_iou(fr["pred"][pidx[t]]["bbox"], h["bbox"]), h) for h in fr["gt"]]
                ious = [a for a in ious if a[0] >= 0.3]
                g = max(ious, key=lambda a: a[0])[1] if ious else None
            else:
                same = [h for h in fr["gt"] if bool(h["is_right"]) == side]
                g = same[0] if same else None
            if g is not None:
                gj = np.asarray(g["joints3d"], np.float64)
                g_rel[t] = (gj - gj[:1]) * 10.0
                z = max(gj[0, 2], 1e-3)
                g_root[t] = [gj[0, 0] / z, gj[0, 1] / z, np.log(z)]
                has[t] = 1.0
        tracks[side] = dict(x=x, present=x[:, 0].copy(), pred_idx=pidx, gt_rel=g_rel, gt_root=g_root, has_gt=has)
        if tok is not None:
            tracks[side]["tok"] = tok
    return tracks


class TemporalRefiner(nn.Module):
    def __init__(self, dim=256, layers=4, heads=4, max_len=64, causal=False, tok_dim=0):
        super().__init__()
        self.causal = causal
        self.inp = nn.Sequential(nn.Linear(D_IN, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, dim)) if tok_dim else None
        self.pos = nn.Parameter(torch.randn(max_len, dim) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.1, batch_first=True,
                                                                    norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 63 + 3))

    def forward(self, x, tok=None):
        """x (B, T, D_IN) [+ tok (B, T, tok_dim)] -> rel (B, T, 21, 3) dm, root (B, T, 3): residuals on present
        frames, full on missing."""
        B, T, _ = x.shape
        h = self.inp(x) + self.pos[:T]
        if self.tok is not None and tok is not None:
            h = h + self.tok(tok.float()) * x[..., :1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1) if self.causal else None
        # is_causal: what nn.TransformerEncoder detects from this mask anyway, without its host sync (CUDA graphs)
        o = self.out(self.enc(h, mask=mask, is_causal=True if self.causal else None))
        m = x[..., :1]
        rel = x[..., 2:65].reshape(B, T, 21, 3) * m[..., None] + o[..., :63].reshape(B, T, 21, 3)
        root = x[..., 107:110] * m + o[..., 63:]
        return rel, root


def pa_error(p, g):
    """Per-frame PA-MPJPE (mm) of (..., 21, 3) joints in dm, batched similarity Procrustes."""
    P, G = p.reshape(-1, 21, 3), g.reshape(-1, 21, 3)
    Pc, Gc = P - P.mean(1, keepdim=True), G - G.mean(1, keepdim=True)
    with torch.no_grad():  # alignment without gradient (SVD backward is unstable for near-degenerate frames)
        U, S, Vh = torch.linalg.svd(Pc.transpose(1, 2) @ Gc)
        d = torch.sign(torch.linalg.det(U @ Vh))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
        R = U @ D @ Vh
        s = (S * D.diagonal(dim1=1, dim2=2)).sum(-1) / Pc.pow(2).sum((1, 2)).clamp(min=1e-8)
    al = s[:, None, None] * Pc @ R + G.mean(1, keepdim=True)
    return ((al - G).norm(dim=-1).mean(-1) * 100).reshape(p.shape[:-2])


def windows(tracks_list, T, strides=(1,)):
    out = []
    for tr in tracks_list:
        n = len(tr["x"])
        for k in strides:  # k > 1: every k-th frame = the same motion k x faster
            if n < T * k:  # shorter tracks (e.g. DexYCB) would give ragged windows
                continue
            for s in range(0, max(n - T * k, 0) + 1, T * k // 2):
                sl = slice(s, s + T * k, k)
                if tr["has_gt"][sl].sum() >= 4:
                    out.append({k_: v[sl] for k_, v in tr.items() if k_ != "pred_idx"})
    return out


def load_tracks(pred_dir):
    tr = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.pkl"))):
        clip = pickle.load(open(f, "rb"))
        tr += [v for v in build_tracks(clip).values() if v["has_gt"].sum() >= 4]
    return tr


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracks = [t for d in args.train.split(",") for t in load_tracks(d)]  # comma-separated track dirs
    rng = np.random.default_rng(0)
    rng.shuffle(tracks)
    nval = max(len(tracks) // 20, 1)
    strides = tuple(int(k) for k in args.strides.split(","))
    val, trn = windows(tracks[:nval], args.T), windows(tracks[nval:], args.T, strides)
    print(f"{len(tracks)} tracks -> {len(trn)} train / {len(val)} val windows", flush=True)
    stack = lambda ws, k: torch.as_tensor(np.stack([w[k] for w in ws]))
    keys = ("x", "gt_rel", "gt_root", "has_gt") + (("tok",) if args.use_tok else ())
    data = {k: stack(trn, k).to(dev) for k in keys}
    vdata = {k: stack(val, k).to(dev) for k in keys}
    tok_dim = data["tok"].shape[-1] if args.use_tok else 0
    model = TemporalRefiner(dim=args.dim, layers=args.layers, max_len=args.T, causal=args.causal, tok_dim=tok_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    def loss_fn(d, aug):
        x = d["x"].clone()
        if aug and args.geo_aug:  # where / how far the track is must not matter: mirror, image-plane shift, depth
            d = dict(d, gt_rel=d["gt_rel"].clone(), gt_root=d["gt_root"].clone())
            B, T = x.shape[:2]
            m = x[..., :1]
            flip = (torch.rand(B, 1, 1, device=dev) < 0.5).float() * -2 + 1
            x[..., 2:65] = (x[..., 2:65].reshape(B, T, 21, 3) * torch.stack([flip, torch.ones_like(flip), torch.ones_like(flip)], -1)).reshape(B, T, 63)
            nuv = x[..., 65:107].reshape(B, T, 21, 2) * torch.cat([flip, torch.ones_like(flip)], -1)[..., None, :]
            rn = x[..., 107:109] * torch.cat([flip, torch.ones_like(flip)], -1)
            d["gt_rel"][..., 0] *= flip
            d["gt_root"][..., 0] *= flip[..., 0]
            # in-plane rotation about the optical axis (exact for camera-frame joints and normalised 2D alike)
            th = (torch.rand(B, 1, device=dev) * 2 - 1) * args.geo_rot
            def rot2(a_, b_):  # rotate (a_, b_) by th; a_ / b_ (B, ...) of any rank
                c, s_ = (f(th).view(B, *([1] * (a_.dim() - 1))) for f in (torch.cos, torch.sin))
                return c * a_ - s_ * b_, s_ * a_ + c * b_
            rj = x[..., 2:65].reshape(B, T, 21, 3)
            rx, ry = rot2(rj[..., 0], rj[..., 1])
            x[..., 2:65] = torch.stack([rx, ry, rj[..., 2]], -1).reshape(B, T, 63)
            nx, ny = rot2(nuv[..., 0], nuv[..., 1])
            nuv = torch.stack([nx, ny], -1)
            qx, qy = rot2(rn[..., 0], rn[..., 1])
            rn = torch.stack([qx, qy], -1)
            gx, gy = rot2(d["gt_rel"][..., 0], d["gt_rel"][..., 1])
            d["gt_rel"] = torch.stack([gx, gy, d["gt_rel"][..., 2]], -1)
            hx, hy = rot2(d["gt_root"][..., 0], d["gt_root"][..., 1])
            d["gt_root"] = torch.stack([hx, hy, d["gt_root"][..., 2]], -1)
            dxy = (torch.rand(B, 1, 2, device=dev) * 2 - 1) * args.geo_shift
            dlz = (torch.rand(B, 1, 1, device=dev) * 2 - 1) * args.geo_depth
            nuv = (rn + dxy)[:, :, None] + (nuv - rn[:, :, None]) * torch.exp(-dlz)[..., None]
            x[..., 65:107] = nuv.reshape(B, T, 42) * m
            x[..., 107:109] = (rn + dxy) * m
            x[..., 109:110] = (x[..., 109:110] + dlz) * m
            d["gt_root"][..., :2] += dxy
            d["gt_root"][..., 2:] += dlz
        if aug:  # simulate missed detections and jitter
            drop = (torch.rand(x.shape[:2], device=dev) < args.p_drop)[..., None]
            x = torch.where(drop, torch.zeros_like(x), x)
            x[..., 2:] += torch.randn_like(x[..., 2:]) * args.noise * x[..., :1]
        rel, root = model(x, d.get("tok"))
        w = d["has_gt"]
        e_rel = (rel - d["gt_rel"]).norm(dim=-1).mean(-1) * 100  # mm
        gr, pr = root_to_xyz(d["gt_root"]), root_to_xyz(root)
        e_root = (pr - gr).norm(dim=-1) * 1000
        n = w.sum().clamp(min=1)
        e_rel = (e_rel * w).sum() / n
        if args.w_pa:  # articulation after similarity alignment (what PA-MPJPE scores), frames with GT + input
            sel = (w > 0) & (x[..., 0] > 0)
            e_rel = e_rel + args.w_pa * pa_error(rel[sel], d["gt_rel"][sel]).mean()
        if args.w_acc:  # match the GT acceleration of the root-relative joints (smooth but not over-smoothed)
            acc = lambda r: r[:, 2:] - 2 * r[:, 1:-1] + r[:, :-2]
            wa = w[:, 2:] * w[:, 1:-1] * w[:, :-2]
            e_acc = (acc(rel) - acc(d["gt_rel"])).norm(dim=-1).mean(-1) * 100
            e_rel = e_rel + args.w_acc * (e_acc * wa).sum() / wa.sum().clamp(min=1)
        return e_rel, (e_root * w).sum() / n, x

    def root_to_xyz(r):
        z = r[..., 2].exp()
        return torch.stack([r[..., 0] * z, r[..., 1] * z, z], -1)

    for step in range(args.steps):
        idx = torch.randint(0, len(data["x"]), (args.batch,), device=dev)
        lr_, lroot, _ = loss_fn({k: v[idx] for k, v in data.items()}, True)
        loss = lr_ + args.w_root * lroot
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vr, vroot, _ = loss_fn(vdata, False)
                # per-frame input error on the same frames (present + GT) for reference
                m = vdata["has_gt"] * vdata["x"][..., 0]
                base = ((vdata["x"][..., 2:65].reshape(*m.shape, 21, 3) - vdata["gt_rel"]).norm(dim=-1).mean(-1) * 100 * m).sum() / m.sum()
            model.train()
            print(f"step {step} loss {loss.item():.2f} | val rel {vr.item():.2f} mm (per-frame {base.item():.2f}) root {vroot.item():.1f} mm", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(args)), os.path.join(args.out, "model.pt"))


@torch.no_grad()
def apply(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    tok_dim = ck["model"]["tok.1.weight"].shape[1] if "tok.1.weight" in ck["model"] else 0
    model = TemporalRefiner(dim=a["dim"], layers=a["layers"], max_len=a["T"], causal=a["causal"], tok_dim=tok_dim).eval()
    model.load_state_dict(ck["model"])
    model.to(dev)
    T = a["T"]
    os.makedirs(args.out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.pred_dir, "*.pkl"))):
        clip = pickle.load(open(f, "rb"))
        tracks = build_tracks(clip, with_gt=False)
        n = len(clip["frames"])
        new = [[] for _ in range(n)]
        used = [set() for _ in range(n)]
        for side, tr in tracks.items():
            x = torch.as_tensor(tr["x"], device=dev)
            tk = torch.as_tensor(tr["tok"], device=dev) if tok_dim and "tok" in tr else None
            sub = lambda a_, b_: None if tk is None else tk[None, a_:b_]
            rel = np.zeros((n, 21, 3), np.float32)
            root = np.zeros((n, 3), np.float32)
            if a["causal"]:  # online: each frame from the window ending at it
                for t in range(n):
                    s = max(0, t - T + 1)
                    r, ro = model(x[None, s:t + 1], sub(s, t + 1))
                    rel[t], root[t] = r[0, -1].cpu().numpy(), ro[0, -1].cpu().numpy()
            else:  # offline: overlapping windows, keep the central part
                starts = list(range(0, max(n - T, 0) + 1, T // 2))
                if starts[-1] + T < n:
                    starts.append(n - T)
                for s in starts:
                    r, ro = model(x[None, s:s + T], sub(s, s + T))
                    lo = 0 if s == 0 else T // 4
                    hi = T if s + T >= n else 3 * T // 4
                    rel[s + lo:s + hi], root[s + lo:s + hi] = r[0, lo:hi].cpu().numpy(), ro[0, lo:hi].cpu().numpy()
            # a track exists where the hand was predicted within +-gap frames (no hallucination far from detections)
            pres = tr["present"] > 0
            if a["causal"]:  # online: only past detections may open a filled frame
                near = np.array([pres[max(0, t - args.gap):t + 1].any() for t in range(n)])
            else:
                near = np.convolve(pres.astype(float), np.ones(2 * args.gap + 1), "same") > 0 if args.gap else pres
            for t in range(n):
                if not near[t]:
                    continue
                fr = clip["frames"][t]
                K = np.asarray(fr["K"], np.float64)
                r_t = rel[t] / 10.0
                if tr["present"][t] > 0 and args.rel_blend < 1:  # keep (part of) the per-frame articulation
                    r_t = args.rel_blend * r_t + (1 - args.rel_blend) * tr["x"][t, 2:65].reshape(21, 3) / 10.0
                j = r_t + root_to_xyz(root[t])
                uv = j[:, :2] / j[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
                k = tr["pred_idx"][t]
                if k >= 0:
                    p = dict(fr["pred"][k])
                    used[t].add(k)
                    if p.get("verts3d") is not None:  # mesh follows the refined root
                        p["verts3d"] = p["verts3d"] - p["joints3d"][:1] + j[:1]
                else:
                    pad = 0.15 * (uv.max(0) - uv.min(0)).max()
                    p = dict(is_right=side, score=0.0, verts3d=None,
                             bbox=np.concatenate([uv.min(0) - pad, uv.max(0) + pad]).astype(np.float32))
                    if any(box_iou(p["bbox"], q["bbox"]) > 0.3 for q in fr["pred"]):  # hand present, side flipped
                        continue
                p.update(joints3d=j.astype(np.float32), joints2d=uv.astype(np.float32))
                new[t].append(p)
        for t, fr in enumerate(clip["frames"]):
            fr["pred"] = new[t] + [p for k, p in enumerate(fr["pred"]) if k not in used[t] and
                                   not any(bool(p["is_right"]) == bool(q["is_right"]) for q in new[t])]
        clip["method"] = clip["method"] + "+temporal"
        pickle.dump(clip, open(os.path.join(args.out_dir, os.path.basename(f)), "wb"))
    print("wrote", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--T", type=int, default=32)
    t.add_argument("--dim", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--causal", action="store_true")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, required=True, help="training batch size; choose for your setup")
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--w_root", type=float, default=0.1)
    t.add_argument("--p_drop", type=float, default=0.1)
    t.add_argument("--noise", type=float, default=0.01)
    t.add_argument("--w_pa", type=float, default=0.0, help="PA-MPJPE term on the refined articulation")
    t.add_argument("--w_acc", type=float, default=0.0, help="acceleration-matching term")
    t.add_argument("--use_tok", action="store_true", help="also feed the per-frame model's per-hand latent")
    t.add_argument("--geo_aug", action="store_true", help="mirror / shift / depth-scale whole tracks (position invariance)")
    t.add_argument("--geo_shift", type=float, default=0.3, help="max image-plane shift (normalised coords)")
    t.add_argument("--geo_depth", type=float, default=0.4, help="max log-depth offset")
    t.add_argument("--geo_rot", type=float, default=3.1416, help="max in-plane rotation (rad)")
    t.add_argument("--strides", default="1", help="training windows at these frame strides (faster motion; val stays at 1)")
    t2 = sub.add_parser("apply")
    t2.add_argument("--ckpt", required=True)
    t2.add_argument("--pred_dir", required=True)
    t2.add_argument("--out_dir", required=True)
    t2.add_argument("--gap", type=int, default=2, help="fill missed frames within this many frames of a detection")
    t2.add_argument("--rel_blend", type=float, default=0.5, help="weight of the refined articulation (0: root only)")
    a = p.parse_args()
    train(a) if a.cmd == "train" else apply(a)
