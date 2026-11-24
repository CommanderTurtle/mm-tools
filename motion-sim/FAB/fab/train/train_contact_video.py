"""Video contact model: a causal GRU over a person track's per-frame contact-head
outputs (train/extract_contact_tracks.py) -> per-frame contact logits = the image head's logits + a zero-initialised
residual, so it starts as the per-frame model.

Per-frame input: the contact head's hidden features, its logits, the 12 contact keypoints in the camera frame relative to
the ankle midpoint, their camera-frame velocities (m/s, from the native frame times), the ankle midpoint's velocity and
the frame interval. Training windows of --T frames with random frame strides (faster apparent motion / lower frame rates).
Loss: masked BCE against the world-GT labels.

    python -m train.train_contact_video --tracks "<track dir>,<track dir>" --val "<track dir>" --out <run dir>
    python -m train.train_contact_video --eval_ckpt $FAB_WEIGHTS/shared/contact_video.pt --test "<track dirs>" --pred_out <dir> \
        --test_cams <cams dir>      (world mode runs this, scripts/fab_world.sh)
"""

from __future__ import annotations

import argparse
import glob
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def check_cams(T, pts, name, T0=None):
    """The camera convention the video head was trained in (ViPE's): camera -> world, the first camera's frame
    (T_world_cam[0] = I), OpenCV axes (y down). A gravity-aligned world (z up, T0 != I) would go in silently and cut
    static recall a lot, hence these checks. T (n, 4, 4) of the track's frames, pts (n, 12, 3) the contact keypoints in
    the camera frame (rows 0 / 1 ankles, 8 / 9 knees), T0 the path's first camera (default T[0]). Raises ValueError when:
      T0 != I                  (the path is not re-expressed in its first camera: inv(T0) @ T_t, contact_clips prep;
                                ViPE's own float32 T0 is within 2e-3)
      improper rotations       (not orthonormal / det -1)
      legs pointing down       (the median shin, knee - ankle, has y > 0.3 in the world: an upside-down world)
    A world -> camera path is not detectable from the track alone (shin-direction jitter / drift does not separate the
    two directions on moving cameras); contact_clips prep checks recon paths against the recon body instead.
    -> dict of the measured values."""
    T = np.asarray(T, np.float64)
    R = T[:, :3, :3]
    err0 = float(np.abs((T[0] if T0 is None else np.asarray(T0, np.float64)) - np.eye(4)).max())
    if err0 > 1e-2:
        raise ValueError(f"{name}: T_world_cam[0] != I (max |T0 - I| {err0:.3g}); re-express the path in its first camera")
    orth = float(np.abs(np.einsum("tji,tjk->tik", R, R) - np.eye(3)).max())
    if orth > 1e-2 or (np.linalg.det(R) < 0).any():
        raise ValueError(f"{name}: T_world_cam rotations are not proper rotations (orthonormality error {orth:.3g})")
    pts = np.asarray(pts, np.float64)
    shin = (pts[:, [8, 9]] - pts[:, [0, 1]]).mean(1)
    shin /= np.linalg.norm(shin, axis=-1, keepdims=True).clip(1e-9)
    up = np.median(np.einsum("tij,tj->ti", R, shin), 0)
    if up[1] > 0.3:
        raise ValueError(f"{name}: the legs point down in the camera world (median shin {np.round(up, 2)}); expected y down")
    return dict(T0_err=err0, orth_err=orth, shin_up=np.round(up, 3).tolist())


def world_points(t, cams, stem):
    """Keypoints to the world frame of an estimated camera (a world-camera dump: frame, T_world_cam, fix = the
    metric body scale the cameras assume); tracks without a camera file stay in their camera frame."""
    for name in (stem, f"emdb2_{stem}"):
        f = Path(cams) / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            row = {int(x): i for i, x in enumerate(z["frame"])}
            T = z["T_world_cam"].astype(np.float64)
            fix = float(z["fix"]) if "fix" in z.files else 1.0
            ix = [row.get(int(fr)) for fr in t["frame"]]
            keep = np.array([i is not None for i in ix])
            pts = t["pts"].astype(np.float64) * fix
            Tk = T[[i for i in ix if i is not None]]
            check_cams(Tk, pts[keep], f.name, T0=T[0])
            pts[keep] = np.einsum("tij,tkj->tki", Tk[:, :3, :3], pts[keep]) + Tk[:, None, :3, 3]
            return pts.astype(np.float32)
    return t["pts"]


def load_tracks(dirs, cams=""):
    """dirs: comma-separated track dirs, each optionally "<dir>@k" = its tracks k times (window sampling is uniform over
    tracks); cams: camera dumps applied to the tracks that have one (the others stay in their camera frame)."""
    out = []
    for d in [x for x in dirs.split(",") if x]:
        rep = 1
        if "@" in Path(d).name:
            d, rep = d.rsplit("@", 1)
            rep = int(rep)
        n0 = len(out)
        for f in sorted(glob.glob(f"{d}/*.npz")):
            z = np.load(f)
            t = {k: z[k] for k in ("logits", "feat", "pts", "time", "c", "valid")}
            if cams:  # world-frame keypoints from the world pipeline's camera (moving-camera test sets)
                t["frame"] = z["frame"]
                t["pts"] = world_points(t, cams, Path(f).stem)
            if t["logits"].shape[1] == 2 * t["c"].shape[1]:  # split head: static then any contact (height only)
                h = z["h"].astype(np.float64) if "h" in z.files else np.full(t["c"].shape, np.nan)
                thr = np.array([0.04] * 4 + [0.05] * 4)  # fab/contact_regions.H_THR
                ca, va = (h < thr) & np.isfinite(h), t["valid"] * np.isfinite(h)
                if "valid_any" in z.files and z["valid_any"].any():  # the dataset's own any contact (RICH HSC)
                    ok = z["valid_any"] > 0
                    ca, va = np.where(ok, z["c_any"] > 0, ca), np.maximum(va, ok)
                t["c"] = np.concatenate([t["c"], ca], 1).astype(np.float32)  # float: soft labels (0 / 0.5 / 1) stay
                t["valid"] = np.concatenate([t["valid"], va], 1).astype(np.uint8)
            t["ids"] = z["ids"].astype(str)
            t["src"] = Path(d).name
            out.append(t)
        out += out[n0:] * (rep - 1)
    return out


VEL_WIN = 0.0  # > 0: also the backward-difference velocities over this span (s), less noisy than one frame


def cam_noise(pts, rng, step_t=0.004, step_r=0.003):
    """An estimated camera's error on world-frame keypoints (T, P, 3): a per-frame random walk of translation (m) and
    small rotations (rad) about the keypoints' centre, common to all points of a frame (training only)."""
    T = len(pts)
    tr = np.cumsum(np.array([[rng.gauss(0, step_t) for _ in range(3)] for _ in range(T)]), 0)
    ang = np.cumsum(np.array([[rng.gauss(0, step_r) for _ in range(3)] for _ in range(T)]), 0)
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(ang).as_matrix()
    ctr = pts.mean((0, 1))
    return np.einsum("tij,tkj->tki", R, pts - ctr) + ctr + tr[:, None]


def features(t, idx, cam=None, noise_rng=None):
    """(T,) frame indices of track t -> (T, D) inputs, (T, 8) base logits. cam = (v (3,) m/s, w rad/s): a simulated
    moving camera (translation v, yaw rate w about the camera's y axis) applied to the keypoints (training only)."""
    tm = t["time"][idx]
    pts = t["pts"][idx].astype(np.float64)
    if noise_rng is not None:
        pts = cam_noise(pts, noise_rng)
    if cam is not None:
        v, w = cam
        dt0 = tm - tm[0]
        c, s_ = np.cos(w * dt0), np.sin(w * dt0)
        R = np.zeros((len(idx), 3, 3))
        R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s_, 1, -s_, c
        ctr = pts.mean((0, 1))
        pts = np.einsum("tij,tkj->tki", R, pts - ctr) + ctr + dt0[:, None, None] * v
    anc = 0.5 * (pts[:, 0] + pts[:, 1])  # ankle midpoint (MHR 13 / 14)
    dt = np.diff(tm, prepend=tm[0])
    dts = np.where(dt > 0, dt, 1.0)[:, None, None]
    vel = np.diff(pts, axis=0, prepend=pts[:1]) / dts
    vel[dt <= 0] = 0
    vel = np.clip(vel, -10, 10)
    va = np.clip(np.diff(anc, axis=0, prepend=anc[:1]) / dts[:, 0], -10, 10)
    va[dt <= 0] = 0
    extra = []
    if VEL_WIN > 0:  # causal: frame i against the latest frame at least VEL_WIN s earlier (the window's first frame)
        j = np.clip(np.searchsorted(tm, tm - VEL_WIN, side="right") - 1, 0, None)
        span = tm - tm[j]
        ok = span > 0
        vw = np.zeros_like(pts)
        vw[ok] = (pts[ok] - pts[j[ok]]) / span[ok, None, None]
        extra.append(np.clip(vw, -10, 10).reshape(len(idx), -1))
    x = np.concatenate([t["feat"][idx].astype(np.float32) / 4.0, t["logits"][idx] / 4.0,
                        ((pts - anc[:, None]) * 5).reshape(len(idx), -1), vel.reshape(len(idx), -1), va,
                        np.clip(dt, 0, 1)[:, None] * 10] + extra, 1)
    return x.astype(np.float32), t["logits"][idx].astype(np.float32)


class ContactGRU(nn.Module):
    def __init__(self, d_in, hidden=256, layers=2, n_out=8):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, base, h=None):
        y, h = self.gru(self.inp(x), h)
        return base + self.out(y), h


def windows(tracks, T, strides, rng, p_cam=0.0, p_noise=0.0):
    t = tracks[rng.randrange(len(tracks))]
    n = len(t["time"])
    s = rng.choice(strides)
    span = (T - 1) * s + 1
    start = rng.randrange(max(n - span, 0) + 1)
    idx = np.arange(start, min(start + span, n), s)
    cam = None
    if rng.random() < p_cam:  # a handheld / following camera: the planted foot moves in the camera frame
        cam = (np.array([rng.gauss(0, 0.8), rng.gauss(0, 0.15), rng.gauss(0, 0.5)]), rng.gauss(0, 0.4))
    x, base = features(t, idx, cam, rng if rng.random() < p_noise else None)
    pad = T - len(idx)
    y, v = t["c"][idx].astype(np.float32), t["valid"][idx].astype(np.float32)
    if pad:
        x, base = np.pad(x, ((0, pad), (0, 0))), np.pad(base, ((0, pad), (0, 0)))
        y, v = np.pad(y, ((0, pad), (0, 0))), np.pad(v, ((0, pad), (0, 0)))
    return x, base, y, v


@torch.no_grad()
def predict(model, tracks, dev):
    """Causal pass over each whole track -> {id: prob (8,)} (and the per-frame base prob)."""
    out, base_out = {}, {}
    model.eval()
    for t in tracks:
        idx = np.arange(len(t["time"]))
        x, base = features(t, idx)
        y, _ = model(torch.from_numpy(x)[None].to(dev), torch.from_numpy(base)[None].to(dev))
        p = torch.sigmoid(y[0]).cpu().numpy()
        pb = 1 / (1 + np.exp(-base))
        for i, k in enumerate(t["ids"]):
            out[k], base_out[k] = p[i], pb[i]
    model.train()
    return out, base_out


def f1(tracks, probs):
    P, G = [], []
    for t in tracks:
        m = t["valid"][:, 0] > 0
        P.append(np.stack([probs[k] for k in t["ids"]])[m, :4] >= 0.5)
        G.append(t["c"][m, :4] > 0)
    P, G = np.concatenate(P), np.concatenate(G)
    tp, fp, fn = (P & G).sum(), (P & ~G).sum(), (~P & G).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return float(2 * pr * rc / max(pr + rc, 1e-9)), float(pr), float(rc)


def save_preds(tracks, probs, path):
    ids = np.array([k for t in tracks for k in t["ids"]])
    np.savez(path, ids=ids, prob=np.stack([probs[k] for k in ids]).astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", default="")
    p.add_argument("--val", default="")
    p.add_argument("--test", default="", help="comma-separated track dirs: predictions saved per dir")
    p.add_argument("--out", default="")
    p.add_argument("--T", type=int, default=48)
    p.add_argument("--strides", default="1,2,3")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--p_cam", type=float, default=0.0, help="share of training windows with a simulated moving camera")
    p.add_argument("--p_noise", type=float, default=0.0,
                   help="share of training windows with a simulated camera-estimate error (world-frame models)")
    p.add_argument("--vel_win", type=float, default=0.0, help="extra backward-difference velocities over this span (s)")
    p.add_argument("--w_smooth", type=float, default=0.0, help="weight of the frame-to-frame probability change penalty")
    p.add_argument("--train_cams", default="", help="camera dumps for moving-camera training tracks (e.g. 3DPW train with "
                   "its GT cam_poses): their keypoints in that first-camera world, as the test cameras do")
    p.add_argument("--eval_ckpt", default="")
    p.add_argument("--pred_out", default="")
    p.add_argument("--test_cams", default="", help="with --eval_ckpt: dir of world-pipeline dumps (T_world_cam per "
                                                   "track): keypoints in that world frame")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global VEL_WIN
    torch.backends.cudnn.enabled = False  # the GRU without cuDNN (as the released model was trained and evaluated)
    if a.eval_ckpt:
        st = torch.load(a.eval_ckpt, map_location="cpu", weights_only=False)
        VEL_WIN = float(st.get("args", {}).get("vel_win", 0.0))
        model = ContactGRU(st["d_in"], n_out=st.get("n_out", 8)).to(dev)
        model.load_state_dict(st["model"])
        for d in a.test.split(","):
            tr = load_tracks(d, a.test_cams)
            probs, base = predict(model, tr, dev)
            Path(a.pred_out).mkdir(parents=True, exist_ok=True)
            save_preds(tr, probs, f"{a.pred_out}/{Path(d).name}.npz")
            save_preds(tr, base, f"{a.pred_out}/{Path(d).name}_image.npz")
            print(f"{Path(d).name}: video F1/P/R {f1(tr, probs)}, image {f1(tr, base)}", flush=True)
        return
    VEL_WIN = a.vel_win
    train = load_tracks(a.tracks, a.train_cams)
    val = load_tracks(a.val)
    print(f"{len(train)} train tracks ({sum(len(t['time']) for t in train)} frames), {len(val)} val tracks", flush=True)
    rng = random.Random(0)
    strides = [int(s) for s in a.strides.split(",")]
    d_in = features(train[0], np.arange(2))[0].shape[1]
    n_out = train[0]["logits"].shape[1]
    model = ContactGRU(d_in, n_out=n_out).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, t0 = -1.0, time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 200) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
        xs, bs, ys, vs = zip(*[windows(train, a.T, strides, rng, a.p_cam, a.p_noise) for _ in range(a.batch)])
        x, b, y, v = (torch.from_numpy(np.stack(z)).to(dev) for z in (xs, bs, ys, vs))
        logit, _ = model(x, b)
        loss = (nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * v).sum() / v.sum().clamp_min(1)
        if a.w_smooth > 0:  # chattering: penalise frame-to-frame probability changes inside valid stretches
            pr = torch.sigmoid(logit)
            m = v[:, 1:] * v[:, :-1]
            loss = loss + a.w_smooth * ((pr[:, 1:] - pr[:, :-1]) ** 2 * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
        if val and (step % a.eval_every == 0 or step == a.steps):
            probs, base = predict(model, val, dev)
            fv, fb = f1(val, probs), f1(val, base)
            print(f"step {step} val feet F1/P/R video {fv} image {fb}", flush=True)
            state = {"model": model.state_dict(), "d_in": d_in, "n_out": n_out, "step": step, "args": vars(a)}
            torch.save(state, out / "last.pt")
            if fv[0] > best:
                best = fv[0]
                torch.save(state, out / "model.pt")
    if not val:
        torch.save({"model": model.state_dict(), "d_in": d_in, "n_out": n_out, "step": a.steps, "args": vars(a)},
                   out / "model.pt")
    print("done", best, flush=True)


if __name__ == "__main__":
    main()
