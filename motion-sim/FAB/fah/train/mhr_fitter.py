"""Learned one-pass initialiser of the MHR-Hand fit (fah/mhr_fit.py MHRInitNet; the MHR counterpart of
train/mesh_fitter.py). Trained on a dense keypoint model's predictions over training clips (HOT3D train clips, DexYCB s0
train sequences; benchmark sequences asserted absent) against the GT MANO meshes: from the closed-form init (per-part
Procrustes) and the canonical prediction it predicts a residual on the hand pose / scale / shape, a small rotation and a
translation, so that the MHR hand -- through the differentiable rig and the MHR -> MANO regressors -- lands on the GT
surface and joints (it also learns the prediction's systematic errors). At run time: closed-form init -> this net ->
Levenberg-Marquardt steps (MHRHandFitter(init_net=...)).

    python -m train.mhr_fitter extract --runs $RESULTS/hot3d/train-fah,$RESULTS/dexycb/train-fah --out $CKPT_ROOT/mhr_fitter/data/train.npz
    python -m train.mhr_fitter train --batch "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --data .../train.npz --vreg fah/assets/mhr_to_mano778.npz --out $CKPT_ROOT/mhr_fitter/mhr_fitter
-> <out>/model.pt (= the release's shared/mhr_fitter.pt), <out>/report.json
"""
import argparse
import glob
import json
import os
import pickle
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fah.mhr_fit import VREG, MHRHandFitter, MHRInitNet  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def clip_sequences(crops="/data/crops"):
    """HOT3D clip -> sequence (the hot3d_lab_mesh index); DexYCB clips are named <sequence>_<camera serial>."""
    m = {}
    for line in open(os.path.join(crops, "hot3d_lab_mesh", "index.jsonl")):
        r = json.loads(line)
        m.setdefault(r["src"].split(":")[0], r["sequence_id"])
    return lambda ds, clip: m[clip] if ds == "hot3d" else "_".join(clip.split("_")[:2])


def benchmark_sequences():
    sp = lambda f: json.load(open(os.path.join(REPO, "bench", "splits", f)))
    return dict(hot3d=set(sp("hot3d_bench.json")["hawor_val_sequences"]), dexycb=set(sp("dexycb_s0_test.json")["test_sequences"]))


def extract(args):
    seq_of, bench = clip_sequences(args.crops), benchmark_sequences()
    keys = ("v", "j", "gv", "gj")
    out = {k: [] for k in keys + ("w", "right", "clip")}
    report = {}
    for run in args.runs.split(","):
        ds = "hot3d" if "/hot3d/" in run else "dexycb"
        files = sorted(glob.glob(os.path.join(run, "*.pkl")))[::args.every_clip]
        seqs = {seq_of(ds, os.path.basename(f)[:-4]) for f in files}
        overlap = sorted(seqs & bench[ds])
        report[run] = dict(clips=len(files), sequences=len(seqs), benchmark_overlap=len(overlap))
        assert not overlap, (run, overlap[:5])  # training clips only
        for f in files:
            d = pickle.load(open(f, "rb"))
            for fr in d["frames"][::args.every_frame]:
                for g in fr["gt"]:
                    if g.get("verts3d") is None:
                        continue
                    best, bp = 0.3, None
                    for p in fr["pred"]:
                        if bool(p["is_right"]) == bool(g["is_right"]) and p.get("verts3d") is not None and len(p["verts3d"]) == 778:
                            u = box_iou(p["bbox"], g["bbox"])
                            if u > best:
                                best, bp = u, p
                    if bp is None or not np.isfinite(bp["verts3d"]).all():
                        continue
                    m = np.array([1.0 if g["is_right"] else -1.0, 1.0, 1.0])
                    w = np.asarray(bp["joints3d"], np.float64)[0] * m
                    for k, x in zip(keys, (bp["verts3d"], bp["joints3d"], g["verts3d"], g["joints3d"])):
                        out[k].append((np.asarray(x, np.float64) * m - w).astype(np.float16))  # wrist-relative: float16 keeps ~0.06 mm
                    out["w"].append(w.astype(np.float32)), out["right"].append(bool(g["is_right"])), out["clip"].append(f"{ds}/{d['clip']}")
            print(f, len(out["w"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **{k: np.stack(v) for k, v in out.items()})
    json.dump(report, open(os.path.splitext(args.out)[0] + "_overlap.json", "w"), indent=1)
    print(len(out["w"]), "hands ->", args.out, report)


def forward(fit, net, f, x0, R0):
    """Learned residual -> regressed MANO-topology mesh / joints (wrist-centred camera frame), differentiable."""
    x, R = fit.apply_init(net(f), x0, R0)
    v, _ = fit.rig.verts(x, fit.B, "f")
    q = torch.cat([fit.rig.regress(v, fit.B, "Wv"), fit.rig.regress(v, fit.B, "Wj")], 1)
    q = (q[:, :, None, :] * R[:, None]).sum(-1) + x[:, None, fit.sl_t]
    return q[:, :778], q[:, 778:]


def train(args):
    from fah import load_hand_model
    from fah.mhr_fit import build_asset

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    d = np.load(args.data)
    n = len(d["w"])
    held = np.array([zlib.crc32(c.encode()) % 100 < 3 for c in d["clip"]])  # ~3 % of the clips held out
    tr, val = np.nonzero(~held)[0], np.nonzero(held)[0]
    T = {k: torch.as_tensor(d[k].astype(np.float32), device=dev) for k in ("v", "j", "gv", "gj")}
    model, _ = load_hand_model(args.checkpoint, device="cpu", verbose=False)
    fit = MHRHandFitter(build_asset(model.head_pose_hand, vreg=args.vreg), dev, graph=False, compile=False)
    X0, R0, F = [], [], []
    with torch.no_grad():  # closed-form init + features of every sample
        for s in range(0, n, 2048):
            tV, tJ = T["v"][s:s + 2048], T["j"][s:s + 2048]
            x0, r0 = fit._init(tV)
            X0.append(x0), R0.append(r0), F.append(fit.features(tV, tJ, x0, r0))
    X0, R0, F = torch.cat(X0), torch.cat(R0), torch.cat(F)
    npt = (len(fit.B["sub"]) + 21) * 3
    net = MHRInitNet(F.shape[1], fit.n, args.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    os.makedirs(args.out, exist_ok=True)

    def evaluate(idx, use_net=True):
        errs = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                b = torch.as_tensor(idx[s:s + 1024], device=dev)
                f = F[b] if use_net else F[b] * 0
                v, _ = forward(fit, net if use_net else (lambda z: torch.zeros(len(z), fit.n, device=dev)), f, X0[b], R0[b])
                g = T["gv"][b]
                errs.append(((v - v.mean(1, keepdim=True)) - (g - g.mean(1, keepdim=True))).norm(dim=-1).mean(1))
        return torch.cat(errs).mean().item() * 1000

    rep = dict(train=int(len(tr)), val=int(len(val)), val_closed_form_mm=evaluate(val, False))
    print(f"{len(tr)} train / {len(val)} val hands; closed-form init val vertex error {rep['val_closed_form_mm']:.2f} mm", flush=True)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        rng.shuffle(tr)
        for s in range(0, len(tr) - args.batch + 1, args.batch):
            b = torch.as_tensor(tr[s:s + args.batch], device=dev)
            f = F[b]
            if args.noise > 0:  # input noise on the canonical points (m): other prediction error patterns than the training clips'
                f = torch.cat([f[:, :npt] + torch.randn_like(f[:, :npt]) * args.noise * 10.0, f[:, npt:]], 1)
            v, j = forward(fit, net, f, X0[b], R0[b])
            gv, gj = T["gv"][b], T["gj"][b]
            loss = (v - gv).abs().sum(-1).mean() + (j - gj).abs().sum(-1).mean() \
                + ((v - v.mean(1, keepdim=True)) - (gv - gv.mean(1, keepdim=True))).abs().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        e = evaluate(val)
        print(f"epoch {ep}: loss {loss.item() * 1000:.2f}, val vertex error {e:.2f} mm", flush=True)
        ck = dict(model=net.state_dict(), d_in=int(F.shape[1]), d_out=int(fit.n), hidden=args.hidden, vreg=os.path.basename(args.vreg),
                  sub=fit.B["sub"].cpu().numpy(), args=vars(args))
        torch.save(ck, os.path.join(args.out, "model.pt"))
    rep["val_net_mm"] = e
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--runs", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--crops", default="/data/crops")
    e.add_argument("--every_clip", type=int, default=1)
    e.add_argument("--every_frame", type=int, default=2)
    t = sp.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    t.add_argument("--vreg", default=VREG, help="MHR -> MANO-778 vertex regressor of the loss (the one the fitter ships with)")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, required=True, help="training batch size; choose for your setup")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=1024)
    t.add_argument("--noise", type=float, default=0.003, help="std (m) of Gaussian noise on the canonical input points")
    args = p.parse_args()
    {"extract": extract, "train": train}[args.cmd](args)
