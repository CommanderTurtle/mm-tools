"""Learned mesh fitter: one forward pass from a dense keypoint model's prediction (MANO-topology surface + 21 joints)
to a MANO mesh, trained on the model's own predictions over training clips against the GT meshes (so it also learns
the prediction's systematic errors). The hand is canonicalised by the closed-form per-part Procrustes pose
(fah/mano_procrustes.py: root rotation R0 + flat-hand pose p0); an MLP predicts the pose residual, the shape, a
small rotation and a translation; fah.fast_mano.FastMano gives the mesh.

    python -m train.mesh_fitter extract --runs $RESULTS/hot3d/train-fah,$RESULTS/dexycb/train-fah --out $DATA_ROOT/lmf/train.npz
    python -m train.mesh_fitter train --batch "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --data $DATA_ROOT/lmf/train.npz --out $CKPT_ROOT/mesh_fitter/lmf
    python -m train.mesh_fitter apply --ckpt .../model.pt --run $R/hot3d/image --out $R/hot3d/image-lmf
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fah.fast_mano import FastMano, rodrigues  # noqa: E402
from fah.mano_procrustes import ManoProcrustes  # noqa: E402

SUB = np.arange(0, 778, 4)  # 195 surface points as input


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def canon(pm, v, j, right):
    """Right-hand frame (mirror left hands), closed-form init: R0 (3, 3), p0 (45,), the wrist."""
    m = np.array([1.0 if right else -1.0, 1.0, 1.0])
    v, j = np.asarray(v, np.float64) * m, np.asarray(j, np.float64) * m
    R, _ = pm.rotations(v)
    R0, p0 = pm.local_pose(R)
    return v, j, R0, p0, m


def features(v, j, R0):
    """Canonical input: subset surface + joints, wrist-centred, in the root frame (R0^T)."""
    w = j[..., :1, :]
    return np.concatenate([((v[..., SUB, :] - w) @ R0), ((j - w) @ R0)], -2).reshape(*v.shape[:-2], -1)


def extract(args):
    pm = ManoProcrustes(args.mano_dir)
    X, P0, R0s, W, GV, GJ = [], [], [], [], [], []
    for run in args.runs.split(","):
        files = sorted(glob.glob(os.path.join(run, "*.pkl")))[::args.every_clip]
        for f in files:
            d = pickle.load(open(f, "rb"))
            for fr in d["frames"][::args.every_frame]:
                for g in fr["gt"]:
                    if g.get("verts3d") is None:
                        continue
                    best, bp = 0.3, None
                    for p in fr["pred"]:
                        if bool(p["is_right"]) == bool(g["is_right"]) and p.get("verts3d") is not None:
                            u = box_iou(p["bbox"], g["bbox"])
                            if u > best:
                                best, bp = u, p
                    if bp is None:
                        continue
                    v, j, R0, p0, m = canon(pm, bp["verts3d"], bp["joints3d"], bool(g["is_right"]))
                    X.append(features(v, j, R0).astype(np.float32)), P0.append(p0.astype(np.float32)), R0s.append(R0.astype(np.float32))
                    W.append(j[0].astype(np.float32))
                    GV.append((np.asarray(g["verts3d"], np.float64) * m).astype(np.float32)), GJ.append((np.asarray(g["joints3d"], np.float64) * m).astype(np.float32))
            print(f, len(X), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, x=np.stack(X), p0=np.stack(P0), R0=np.stack(R0s), w=np.stack(W), gv=np.stack(GV), gj=np.stack(GJ))
    print(len(X), "samples ->", args.out)


class Fitter(nn.Module):
    def __init__(self, d_in, hidden=1024, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in + 45, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 45 + 10 + 3 + 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, p0):
        o = self.net(torch.cat([x * 10.0, p0], -1))
        return p0 + o[:, :45], o[:, 45:55], o[:, 55:58] * 0.1, o[:, 58:61] * 0.01


def mesh(mano, fit, x, p0, R0, w):
    pose, betas, dr, dt = fit(x, p0)
    v, j = mano(pose, betas)
    R = R0 @ rodrigues(dr)
    return v @ R.mT + w[:, None] + dt[:, None], j @ R.mT + w[:, None] + dt[:, None]


def train(args):
    dev = torch.device("cuda")
    d = np.load(args.data)
    n = len(d["x"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    val, tr = perm[:2000], perm[2000:]
    T = {k: torch.as_tensor(d[k], device=dev) for k in ("x", "p0", "R0", "w", "gv", "gj")}
    mano = FastMano(args.mano_dir, dev)
    fit = Fitter(T["x"].shape[1], args.hidden, args.dropout).to(dev)
    opt = torch.optim.AdamW(fit.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tr) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    os.makedirs(args.out, exist_ok=True)

    def evaluate(idx):
        with torch.no_grad():
            errs = []
            for s in range(0, len(idx), 1024):
                b = torch.as_tensor(idx[s:s + 1024], device=dev)
                v, j = mesh(mano, fit, T["x"][b], T["p0"][b], T["R0"][b], T["w"][b])
                g = T["gv"][b]
                vc, gc = v - v.mean(1, keepdim=True), g - g.mean(1, keepdim=True)  # translation-aligned vertex error
                errs.append((vc - gc).norm(dim=-1).mean(1))
            return torch.cat(errs).mean().item() * 1000

    print(f"{len(tr)} train / {len(val)} val; init val vertex error {evaluate(val):.2f} mm", flush=True)
    step = 0
    for ep in range(args.epochs):
        rng.shuffle(tr)
        for s in range(0, len(tr) - args.batch + 1, args.batch):
            b = torch.as_tensor(tr[s:s + args.batch], device=dev)
            x = T["x"][b]
            if args.noise > 0:  # input noise (canonical frame, m): other prediction error patterns than the training clips'
                x = x + torch.randn_like(x) * args.noise
            v, j = mesh(mano, fit, x, T["p0"][b], T["R0"][b], T["w"][b])
            gv, gj = T["gv"][b], T["gj"][b]
            loss = (v - gv).abs().sum(-1).mean() + (j - gj).abs().sum(-1).mean() \
                + ((v - v.mean(1, keepdim=True)) - (gv - gv.mean(1, keepdim=True))).abs().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        print(f"epoch {ep}: loss {loss.item() * 1000:.2f}, val vertex error {evaluate(val):.2f} mm", flush=True)
        torch.save(dict(model=fit.state_dict(), d_in=T["x"].shape[1], hidden=args.hidden), os.path.join(args.out, "model.pt"))


def apply(args):
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    fit = Fitter(ck["d_in"], ck.get("hidden", 1024)).to(dev).eval()
    fit.load_state_dict(ck["model"])
    mano = FastMano(args.mano_dir, dev)
    pm = ManoProcrustes(args.mano_dir)
    files = [args.run] if args.run.endswith(".pkl") else sorted(glob.glob(os.path.join(args.run, "*.pkl")))
    os.makedirs(os.path.dirname(args.out) if args.out.endswith(".pkl") else args.out, exist_ok=True)
    for f in files:
        dd = pickle.load(open(f, "rb"))
        items = []
        for fr in dd["frames"]:
            for p in fr["pred"]:
                v = p.get("verts3d_dense", p.get("verts3d"))
                if v is None or len(v) != 778 or not np.isfinite(v).all():
                    continue
                vv, jj, R0, p0, m = canon(pm, v, p["joints3d"], bool(p["is_right"]))
                items.append((p, v, features(vv, jj, R0), p0, R0, jj[0], m))
        for s in range(0, len(items), 2048):
            ch = items[s:s + 2048]
            T = lambda k: torch.as_tensor(np.stack([c[k] for c in ch]), dtype=torch.float32, device=dev)
            with torch.no_grad():
                v, _ = mesh(mano, fit, T(2), T(3), T(4), T(5))
            v = v.cpu().numpy()
            for c, vk in zip(ch, v):
                c[0]["verts3d_dense"], c[0]["verts3d"], c[0]["mesh_param"] = c[1], (vk * c[6]).astype(np.float32), True
        pickle.dump(dd, open(args.out if args.out.endswith(".pkl") else os.path.join(args.out, os.path.basename(f)), "wb"))
        print(os.path.basename(f), len(items), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--every_clip", type=int, default=1)
    e.add_argument("--every_frame", type=int, default=2)
    t = sp.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, required=True, help="training batch size; choose for your setup")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=1024)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--noise", type=float, default=0.0, help="std (m) of Gaussian noise on the canonical input points")
    a = sp.add_parser("apply")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--out", required=True)
    for q in (e, t, a):
        q.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    args = p.parse_args()
    {"extract": extract, "train": train, "apply": apply}[args.cmd](args)
