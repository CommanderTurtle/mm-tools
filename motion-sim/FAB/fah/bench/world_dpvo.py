"""Camera trajectories with DPVO (Teed et al., NeurIPS 2023, real-time sparse patch VO) for world-frame hands.

Frames from the cache (bench/world_frames.py), resized to --size long side (multiple of 16), every --stride-th frame
fed; hands handled like HaWoR's masked DROID: hand pixels zeroed and patch centroids sampled outside the hand masks
(--mask_run verts3d). --config default|fast (DPVO's yaml), --final_ba = terminate()'s closing BA iterations (DPVO 12;
0 = the online sliding-window poses). Skipped frames get interpolated poses.
--causal: online output instead -- each fed frame's pose as estimated --delays k fed frames after it was processed (k = 0:
strict causal; no final BA; skipped frames hold the last pose; frames before initialisation are output then); metric scale from ONE Metric3D call, fixed once --scale_at fed frames
have been processed, on the keyframe --scale_lag frames back (scale_causal). --no_mask: hands not masked, so the camera
does not depend on the hand model (runs concurrently with it); --extra_updates n --extra_every k: a causal BA burst of n
extra window updates after every k-th fed frame (before that frame's output); --mask_box --mask_delay k: masks from the padded predicted
hand boxes of frame t - k (what the concurrent pipeline in scripts/benchmark_world_e2e.py can use). --opts: DPVO config
overrides, e.g. LOOP_CLOSURE True (DPV-SLAM's proximity loop closure; causal poses are read out in the first map units).
--reinit_rot deg / --scale_pw: causal tracking-failure re-initialisation / piecewise metric scale (causal_reinit;
T_world_cam_d0 is then metric, scale_causal 1; --save_calls also stores each call's keyframe patches and the unit-scale
poses). --depth_model moge2: the scale calls with MoGe-2 instead of Metric3D (bench/world_depth.py, --depth_py env).
sig_cov (fed): the static coverage of each fed frame, the share of DPVO's 1/4-resolution patch grid left unmasked by the
hand masks (bench/world_fuse.py --t_sigma_cov; bench/world_coverage.py recomputes it for runs stored without it).
Metric scale (DPVO is up to scale), stored as npz scalars (apply with bench/world.py "<dir>@<key>"):
  m3d_kf / scale_m3d_<n>  Metric3D ViT-L on n evenly spaced retained keyframes: median over their patches (outside hands,
                          Metric3D depth 0.2-3 m) of Metric3D depth / DPVO depth
  scale_hand_bg           our wrist depth / median DPVO depth of the patches in a ring around the hand box, median
-> <out>/<clip>.npz: index, T_world_cam (DPVO units), scale_*, timings (s): t_track, t_ba, t_m3d_kf
"""
import argparse
import glob
import os
import pickle
import sys
import time
import zlib

import cv2
import numpy as np
import torch


def rasterize(pred, K, shape, faces):
    m = np.zeros(shape, np.uint8)
    for p in pred:
        v = p.get("verts3d")
        if v is None or not np.isfinite(v).all():
            continue
        uv = v[:, :2] / np.clip(v[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
        tri = faces[(v[faces, 2] > 0.01).all(1)]
        cv2.fillPoly(m, list(np.round(uv[tri]).astype(np.int32)), 1)
    return m


def box_mask(pred, scale, shape, pad=0.1):
    """Union of the predicted hand boxes (padded), at the SLAM resolution -- what a concurrent pipeline can pass on."""
    m = np.zeros(shape, np.uint8)
    for p in pred:
        x0, y0, x1, y1 = np.asarray(p["bbox"], np.float64) * scale
        d = pad * max(x1 - x0, y1 - y0)
        m[max(int(y0 - d), 0):max(int(y1 + d) + 1, 0), max(int(x0 - d), 0):max(int(x1 + d) + 1, 0)] = 1
    return m


# masked_patchify adapts DPVO's Patchifier.forward (https://github.com/princeton-vl/DPVO; MIT License, Copyright (c) 2022
# Princeton Vision & Learning Lab; THIRD_PARTY_NOTICES.md)
def masked_patchify(patchifier, altcorr, coords_grid_with_index, DIM):
    """Patchifier.forward with RANDOM centroids drawn outside patchifier.hand_mask ((h, w) bool at 1/4 res, cuda)."""
    def forward(images, patches_per_image=80, disps=None, centroid_sel_strat="RANDOM", return_color=False):
        fmap = patchifier.fnet(images) / 4.0
        imap = patchifier.inet(images) / 4.0
        b, n, c, h, w = fmap.shape
        P = patchifier.patch_size
        x = torch.randint(1, w - 1, size=[n, 4 * patches_per_image], device="cuda")
        y = torch.randint(1, h - 1, size=[n, 4 * patches_per_image], device="cuda")
        m = getattr(patchifier, "hand_mask", None)
        if m is not None:
            bad = m[y, x].float()  # free candidates first (stable sort keeps the random order)
            ix = torch.sort(bad, dim=1, stable=True)[1][:, :patches_per_image]
            x, y = torch.gather(x, 1, ix), torch.gather(y, 1, ix)
        else:
            x, y = x[:, :patches_per_image], y[:, :patches_per_image]
        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P // 2).view(b, -1, 128, P, P)
        clr = altcorr.patchify(images[0], 4 * (coords + 0.5), 0).view(b, -1, 3) if return_color else None
        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")
        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P // 2).view(b, -1, 3, P, P)
        index = torch.arange(n, device="cuda").view(n, 1).repeat(1, patches_per_image).reshape(-1)
        return (fmap, gmap, imap, patches, index, clr) if return_color else (fmap, gmap, imap, patches, index)

    return forward


def m3d_scale(metric, file, K0, pat, size, amp):
    """One Metric3D call: median over a keyframe's patches (Metric3D depth 0.2-3 m) of Metric3D depth x DPVO inverse depth."""
    W1, H1 = size
    pat = pat.float().cpu().numpy()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        dep = metric(file, np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]))
    dep = cv2.resize(np.asarray(dep, np.float32), (W1, H1))
    cx, cy, invd = pat[:, 0, 1, 1] * 4 + 1.5, pat[:, 1, 1, 1] * 4 + 1.5, pat[:, 2, 1, 1]
    md = dep[np.clip(np.round(cy).astype(int), 0, H1 - 1), np.clip(np.round(cx).astype(int), 0, W1 - 1)]
    ok = (invd > 0) & (md > 0.2) & (md < 3.0)
    return float(np.median(md[ok] * invd[ok])) if ok.sum() >= 5 else None


def step_residuals(sl, SE3):
    """Translation evidence of the newest fed frame: confidence-weighted reprojection residual (px at DPVO's 1/4 resolution)
    of the edges into it under DPVO's full pose vs a pure rotation (homography R_j R_i^T: no translation, depth-free). Equal
    residuals = the patch tracks do not see the translation (rotation-only / static), where monocular VO drifts."""
    pg, n, c = sl.pg, sl.n, sl.P // 2
    e = pg.jj == n - 1
    if not e.any():
        return 0.0, 0.0
    ii, kk = pg.ii[e], pg.kk[e]
    tgt, w = pg.target[0, e].float(), pg.weight[0, e].float().mean(-1)
    full = sl.reproject((ii, pg.jj[e], kk))[0, :, :, c, c].float()
    M = SE3(pg.poses_[:n]).matrix().float()
    R = M[n - 1, :3, :3] @ M[ii, :3, :3].transpose(-1, -2)
    uv = pg.patches_.view(-1, 3, sl.P, sl.P)[kk, :2, c, c].float()
    fx, fy, cx, cy = pg.intrinsics_[0].float()
    x = torch.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, torch.ones_like(uv[:, 0])], -1)
    y = (R @ x[..., None])[..., 0]
    rot = torch.stack([fx * y[:, 0] / y[:, 2] + cx, fy * y[:, 1] / y[:, 2] + cy], -1)
    ws = w.sum().clamp(min=1e-6)
    res = lambda p: float(((p - tgt).norm(dim=-1).clamp(max=20) * w).sum() / ws)
    return res(full), res(rot)


def step_residuals_t(sl, SE3):
    """step_residuals without host syncs (no boolean indexing / float()): all edges, the weights zeroed outside the edges
    into the newest frame -> (full, rot) as GPU scalars, read after the frame's synchronise."""
    pg, n, c = sl.pg, sl.n, sl.P // 2
    ii, kk = pg.ii, pg.kk
    tgt, w = pg.target[0].float(), pg.weight[0].float().mean(-1) * (pg.jj == n - 1)
    full = sl.reproject()[0, :, :, c, c].float()
    M = SE3(pg.poses_[:n]).matrix().float()
    R = M[n - 1, :3, :3] @ M[ii, :3, :3].transpose(-1, -2)
    uv = pg.patches_.view(-1, 3, sl.P, sl.P)[kk, :2, c, c].float()
    fx, fy, cx, cy = pg.intrinsics_[0].float()
    x = torch.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, torch.ones_like(uv[:, 0])], -1)
    y = (R @ x[..., None])[..., 0]
    rot = torch.stack([fx * y[:, 0] / y[:, 2] + cx, fy * y[:, 1] / y[:, 2] + cy], -1)
    ws = w.sum().clamp(min=1e-6)
    res = lambda p: ((p - tgt).norm(dim=-1).clamp(max=20) * w).sum() / ws
    return res(full), res(rot)


def no_inactive_store(sl, cfg):
    """DPVO stores every removed edge for the global BA of loop closure (a torch.cat of the whole history per fed frame,
    ~3.5 ms per fed frame on 30 s clips); without loop closure no long-range edge exists, the global BA never runs and the
    store is never read: skip it (same trajectory up to DPVO's run-to-run CUDA noise)."""
    if not (cfg.LOOP_CLOSURE or cfg.CLASSIC_LOOP_CLOSURE):
        rf = sl.remove_factors
        sl.remove_factors = lambda m, store: rf(m, False)
    return sl


def causal_reinit(args, make_slam, fed, imgs, masks, intr, scale_fn):
    """--reinit_rot: causal DPVO with tracking-failure re-initialisation. DPVO's long-clip failure is a sudden jump (one fed
    frame rotates 10-30 deg while the head turns <= 8 deg), after which its map stays inconsistent. When the newest output
    step rotates more than --reinit_rot deg (--reinit_sig rot) or differs from the previous step by more than that (acc)
    (and the instance has tracked --reinit_min fed frames since initialising), that
    frame's pose is held and a fresh DPVO instance (same network) starts from the buffered last --reinit_back fed frames,
    anchored on the output pose of its first frame; until it initialises the pose is held. Every instance gets its own
    metric scale (one Metric3D call --scale_at fed frames into it; the previous instance's scale if it ends first or the
    call has too few valid patches). --scale_pw (also without --reinit_rot): --scale_calls calls per instance every
    --scale_every fed frames, and each translation increment is scaled by the latest call available at that step (result
    --scale_delay fed frames after the call; the first call before that; --scale_rmed: the running median of the
    instance's calls available at that step) -- follows DPVO's scale drift on long clips.
    --reinit_scale r: also re-initialise when a call's result (available --scale_delay fed frames after it) differs from the
    instance's previous one by more than x r (the map's scale collapsed or exploded); the calls then run inline (their time
    is excluded from the tracking time, as in the background-call pipeline). --trans_gate g (with --signals): a step whose
    patch tracks a pure rotation explains about as well as the full pose (step_residuals ratio < g) keeps the camera
    centre (no translation increment): monocular VO invents translation where none is observable.
    Returns metric c2w per fed frame (len(fed), 4, 4) and per-step diagnostics."""
    from dpvo.lietorch import SE3
    from scipy.spatial.transform import Rotation

    def w2c(sl, c):  # instance-local frame c in the live graph; frames no longer in it via their delta chain
        ts = sl.pg.tstamps_[:sl.n]
        k = np.searchsorted(ts, c)
        if k < len(ts) and ts[k] == c:
            return SE3(sl.pg.poses_[k][None])
        if c not in sl.pg.delta:
            return SE3(sl.pg.poses_[max(k - 1, 0)][None])
        t0, dP = sl.pg.delta[c]
        return SE3(dP.data.reshape(1, 7)) * w2c(sl, int(t0))

    def c2w(sl, c):
        return w2c(sl, c).inv().data.reshape(-1, 7)[0].cpu().numpy().astype(np.float64)

    segs, nf = [], len(fed)
    loc = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (nf, 1))  # c2w in the units of each step's instance
    inst, held, rot, acc, wnew = np.zeros(nf, int), np.zeros(nf, bool), np.zeros(nf), np.zeros(nf), np.zeros(nf)
    res = np.zeros((nf, 2))  # --signals: step_residuals (full pose, pure rotation)
    t_m3d, why = 0.0, []  # inline Metric3D time (--reinit_scale); trigger of each re-init

    def start(j0):
        segs.append(dict(j0=j0, slam=make_slam(), init=None, calls=[]))

    def feed(sg, j):
        sg["slam"].network.patchify.hand_mask = masks[j]
        sg["slam"](fed[j], imgs[j], intr)

    start(0)
    for j in range(nf):
        sg = segs[-1]
        sl = sg["slam"]
        feed(sg, j)
        inst[j] = len(segs) - 1
        if sl.is_initialized and sg["init"] is None:
            sg["init"] = j
            for f in range(sg["j0"], j) if len(segs) == 1 else []:  # first instance: earlier frames are output now
                loc[f] = c2w(sl, f)
        if not sl.is_initialized:
            held[j] = len(segs) > 1
        else:
            loc[j] = c2w(sl, j - sg["j0"])
            if j > sg["init"]:
                rot[j] = np.degrees(2 * np.arccos(min(1.0, abs(float(loc[j, 3:] @ loc[j - 1, 3:])))))
            if j > sg["init"] + 1:  # change of the step: angle(d_{j-1}^-1 d_j), d_j = R_{j-1}^-1 R_j
                Rq = Rotation.from_quat(loc[j - 2:j + 1, 3:])
                acc[j] = np.degrees(((Rq[0].inv() * Rq[1]).inv() * Rq[1].inv() * Rq[2]).magnitude())
            e = sl.pg.jj == sl.n - 1
            wnew[j] = sl.pg.weight[0, e].float().mean().item() if e.any() else 0.0
            if args.signals:
                res[j] = step_residuals(sl, SE3)
            due = args.scale_at + len(sg["calls"]) * args.scale_every
            if len(sg["calls"]) < args.scale_calls and j - sg["j0"] >= due and sl.n > args.scale_lag:
                k = sl.n - 1 - args.scale_lag
                call = [j, int(sl.tlist[int(sl.pg.tstamps_[k])]), sl.pg.patches_[k].clone(), None]
                if args.reinit_scale > 0:
                    ta = time.perf_counter()
                    call[3] = scale_fn(call[1], call[2])
                    t_m3d += time.perf_counter() - ta
                sg["calls"].append(call)
            sig = (rot if args.reinit_sig == "rot" else acc)[j]
            trig = "rot" if args.reinit_rot > 0 and sig > args.reinit_rot else ""
            if args.reinit_scale > 0 and any(c[0] + args.scale_delay == j and c[3] is not None for c in sg["calls"]):
                av = [c[3] for c in sg["calls"] if c[3] is not None and c[0] + args.scale_delay <= j]
                if len(av) >= 2 and max(av[-1] / av[-2], av[-2] / av[-1]) > args.reinit_scale:
                    trig = trig or "scale"
            if trig and j - sg["init"] >= args.reinit_min and j + 1 < nf:
                why.append(trig)
                held[j] = True  # this step is not output; the new instance takes over
                del sg["slam"]
                start(max(j - args.reinit_back, 0))
                inst[j] = len(segs) - 1
                for f in range(segs[-1]["j0"], j + 1):  # catch up on the buffered frames
                    feed(segs[-1], f)
    prev = 1.0  # bench/world.py's fallback
    for sg in segs:  # (step, scale) of the valid calls
        cs = [(c, sc) for c, f, pat, s0 in sg["calls"] for sc in [s0 if s0 is not None else scale_fn(f, pat)] if sc is not None]
        sg["cs"] = cs or [(-1, prev)]
        prev = sg["cs"][-1][1]

    def scale(sg, j):
        av = [sc for c, sc in sg["cs"] if c + args.scale_delay <= j] if args.scale_pw else []
        if av and args.scale_rmed:  # running median (log) of the instance's calls so far: robust to single bad calls
            return float(np.exp(np.median(np.log(av))))
        return av[-1] if av else sg["cs"][0][1]

    T = np.tile(np.eye(4), (nf, 1, 1))
    G, pm, last = np.eye(4), np.zeros(3), None
    for j in range(nf):
        k = inst[j]
        if j and k != inst[j - 1]:  # new instance: its first frame (identity, fixed in DPVO) sits on that frame's output
            G, last = T[segs[k]["j0"]].copy(), None
        if held[j]:
            T[j] = T[j - 1]
            continue
        s = scale(segs[k], j)  # metric translation: increments at the scale of their step
        if args.trans_gate > 0 and res[j, 0] > 0 and res[j, 1] < args.trans_gate * res[j, 0]:
            s = 0.0
        pm = s * loc[j, :3] if last is None else pm + s * (loc[j, :3] - loc[last, :3])
        last = j
        L = np.eye(4)
        L[:3, :3] = Rotation.from_quat(loc[j, 3:]).as_matrix()
        L[:3, 3] = pm
        T[j] = G @ L
    diag = dict(seg_start=np.array([sg["j0"] for sg in segs]), seg_scale=np.array([sg["cs"][0][1] for sg in segs]),
                scale_calls=np.array([(c, sc) for sg in segs for c, sc in sg["cs"]]), n_reinit=len(segs) - 1,
                sig_rot=rot, sig_acc=acc, sig_wnew=wnew, sig_res=res, held=held, inst=inst, t_m3d_inline=t_m3d,
                reinit_why=np.array(why, dtype=str))
    if args.save_calls:  # each call's keyframe patches (x, y at 1/4 res, inverse depth) + the unit-scale poses: offline re-scaling
        cl = [(k, c, f, pat[:, :, 1, 1].float().cpu().numpy()) for k, sg in enumerate(segs) for c, f, pat, s0 in sg["calls"]]
        diag.update(call_seg=np.array([c[0] for c in cl], int), call_step=np.array([c[1] for c in cl], int),
                    call_frame=np.array([c[2] for c in cl], int), call_pat=np.array([c[3] for c in cl], np.float32),
                    loc=loc, fed=np.array(fed))
    return T, diag


def main(args):
    sys.path.insert(0, os.getcwd())  # DPVO checkout
    from dpvo import altcorr
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from dpvo.net import DIM
    from dpvo.utils import coords_grid_with_index
    from dpvo.lietorch import SE3
    from scipy.spatial.transform import Rotation, Slerp

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    cfg.merge_from_file(f"config/{args.config}.yaml")
    cfg.merge_from_list(args.opts)  # e.g. LOOP_CLOSURE True (DPV-SLAM proximity loop closure)
    metric = None
    if args.depth_model != "m3d":  # bench/world_depth.py: a depth model of another env behind a subprocess, same calls
        from bench.world_depth import Client

        metric = Client(args.depth_model, args.depth_py)
    elif args.m3d:
        H = os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "HaWoR")
        sys.path.insert(0, os.path.join(H, "thirdparty/Metric3D"))
        from metric import Metric3D

        metric = Metric3D(os.path.join(H, f"thirdparty/Metric3D/weights/metric_depth_vit_{args.m3d}_800k.pth"),
                          model_name={"large": "v2-L", "small": "v2-S"}[args.m3d])
    faces = pickle.load(open(os.path.join(args.mano_dir, "MANO_RIGHT.pkl"), "rb"))["f"].astype(np.int64)
    os.makedirs(args.out, exist_ok=True)
    clips = sorted(d for d in os.listdir(args.frames) if os.path.exists(os.path.join(args.frames, d, "done")))
    for name in clips[args.shard :: args.num_shards][:: -1 if args.reverse else 1]:
        out_path = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(out_path):
            continue
        files = sorted(glob.glob(os.path.join(args.frames, name, "*.jpg")))
        if args.seed >= 0:  # DPVO samples its patches randomly: per-clip seed for repeatable runs
            torch.manual_seed(args.seed * 100003 + zlib.crc32(name.encode()) % 100003)
        K0 = np.load(os.path.join(args.frames, name, "K.npy"))
        n = len(files)
        H0, W0 = cv2.imread(files[0]).shape[:2]
        s = args.size / max(H0, W0)
        H1, W1 = int(round(H0 * s)) // 16 * 16, int(round(W0 * s)) // 16 * 16
        K = K0.copy()
        K[0] *= W1 / W0
        K[1] *= H1 / H0
        frames = {f["index"]: f for f in pickle.load(open(os.path.join(args.mask_run, f"{name}.pkl"), "rb"))["frames"]}
        fed = list(range(0, n, args.stride))
        imgs, masks, cov = [], [], []
        for i in fed:  # decode / resize / masks shared with the hand pipeline: not timed
            im = cv2.resize(cv2.imread(files[i]), (W1, H1), interpolation=cv2.INTER_AREA)
            src = i - args.mask_delay  # hand predictions of an earlier frame (concurrent pipeline: the hands run alongside)
            if args.no_mask or src not in frames:
                m = np.zeros((H1, W1), np.uint8)
            elif args.mask_box:
                m = box_mask(frames[src]["pred"], W1 / frames[src]["image_size"][1], (H1, W1))
            else:
                m = rasterize(frames[src]["pred"], K, (H1, W1), faces)
            im[m > 0] = 0
            imgs.append(torch.from_numpy(im).permute(2, 0, 1).cuda())
            m4 = cv2.resize(m.astype(np.float32), (W1 // 4, H1 // 4), interpolation=cv2.INTER_AREA) > 0
            masks.append(None if args.no_mask else torch.from_numpy(m4).cuda())
            cov.append(1.0 - float(m4.mean()))
        cov = np.array(cov)
        intr = torch.as_tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32, device="cuda")

        if args.reinit_rot > 0 or args.scale_pw or args.signals or args.reinit_scale > 0:
            net = []

            def make_slam():  # instances share one network (patchifier patched once)
                sl = DPVO(cfg, net[0] if net else "dpvo.pth", ht=H1, wd=W1, viz=False)
                if not net:
                    net.append(sl.network)
                    sl.network.patchify.forward = masked_patchify(sl.network.patchify, altcorr, coords_grid_with_index, DIM)
                return sl

            def scale_fn(kf_t, pat):
                return m3d_scale(metric, files[kf_t], K0, pat, (W1, H1), args.m3d_amp) if metric is not None else None

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                Tf, diag = causal_reinit(args, make_slam, fed, imgs, masks, intr, scale_fn)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            t1 -= diag["t_m3d_inline"]
            T = Tf[np.searchsorted(np.array(fed), np.arange(n), side="right") - 1]  # skipped frames hold the last pose
            torch.cuda.empty_cache()
            out = dict(index=np.arange(n), T_world_cam=T, T_world_cam_d0=T, n_fed=len(fed), t_track=t1 - t0, scale_causal=1.0,
                       slam_size=np.array([H1, W1]), K=K, sig_cov=cov, **diag)
            tmp = f"{out_path}.{os.getpid()}.npz"
            np.savez(tmp, **out)
            os.replace(tmp, out_path)
            print(f"[dpvo] {name}: {n} frames, {len(fed)} fed, causal + re-init, track {t1 - t0:.2f}s, {diag['n_reinit']} "
                  f"re-inits at {diag['seg_start'][1:].tolist()} ({diag['reinit_why'].tolist()}), scales "
                  f"{np.round(diag['seg_scale'], 3).tolist()}", flush=True)
            continue
        slam = DPVO(cfg, "dpvo.pth", ht=H1, wd=W1, viz=False)
        pf = slam.network.patchify
        pf.forward = masked_patchify(pf, altcorr, coords_grid_with_index, DIM)
        gauge = [1.0]  # loop closure: its global BA renormalises the map scale (pg.normalize); poses are read out in the first units
        if cfg.LOOP_CLOSURE:
            def normalize(norm=slam.pg.normalize, pg=slam.pg):
                gauge[0] *= pg.patches_[:pg.n, :, 2].mean().item()
                norm()

            slam.pg.normalize = normalize
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        online, scale_kfs = [], []
        with torch.no_grad():
            for j, (t, im, m4) in enumerate(zip(fed, imgs, masks)):
                pf.hand_mask = m4
                slam(t, im, intr)
                if args.extra_updates and slam.is_initialized and j % args.extra_every == 0:
                    for _ in range(args.extra_updates):  # causal BA burst: more window updates before this frame's output
                        slam.update()
                if args.causal:  # pose-graph snapshot after this frame: (counter index of every frame, w2c poses)
                    online.append((slam.pg.tstamps_[:slam.n].copy(), slam.pg.poses_[:slam.n].clone(), slam.is_initialized, gauge[0]))
                    due = args.scale_at + len(scale_kfs) * args.scale_every
                    if len(scale_kfs) < args.scale_calls and j >= due and slam.is_initialized and slam.n > args.scale_lag:
                        k = slam.n - 1 - args.scale_lag  # a frame whose depth has had scale_lag more updates
                        scale_kfs.append((j, int(slam.tlist[int(slam.pg.tstamps_[k])]), slam.pg.patches_[k].clone(), gauge[0]))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        if args.causal:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            delta = slam.pg.delta  # removed / skipped frame t -> (t0, dP): w2c_t = dP * w2c_t0 (entries never change)
            init = next((j for j, sn in enumerate(online) if sn[2]), len(online) - 1)
            q = np.searchsorted(np.array(fed), np.arange(n), side="right") - 1  # skipped frames hold the last pose

            def w2c(f, ts, P, g=gauge[0]):  # frame f's pose in a snapshot; frames no longer in the graph via their delta chain
                k = np.searchsorted(ts, f)
                if k < len(ts) and ts[k] == f:
                    return SE3(P[k][None])
                if f not in delta:
                    return SE3(P[max(k - 1, 0)][None])
                t0, dP = delta[f]
                dP = SE3(dP.data.reshape(1, 7))  # deltas are in the final units
                return (dP if g == gauge[0] else dP.scale(torch.tensor(g / gauge[0], device="cuda"))) * w2c(int(t0), ts, P, g)

            Ts = {}
            for dl in [int(x) for x in args.delays.split(",")]:
                poses = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (len(fed), 1))
                for f in range(len(fed)):  # fed frame f as estimated at step max(f + delay, initialisation)
                    ts, P, _, g = online[min(max(f + dl, init), len(online) - 1)]
                    if len(ts):
                        poses[f] = w2c(f, ts, P, g).inv().data.reshape(-1, 7)[0].cpu().numpy()
                        poses[f, :3] /= g
                T = np.tile(np.eye(4), (n, 1, 1))
                T[:, :3, :3] = Rotation.from_quat(poses[q, 3:7]).as_matrix()
                T[:, :3, 3] = poses[q, :3]
                Ts[f"T_world_cam_d{dl}"] = T
            T = Ts[f"T_world_cam_d{args.delays.split(',')[0]}"]
            calls = []  # (step, frame, scale) per Metric3D call
            for j, kf_t, pat, g in scale_kfs if metric is not None else []:
                sc = m3d_scale(metric, files[kf_t], K0, pat, (W1, H1), args.m3d_amp)
                if sc is not None:
                    calls.append((j, kf_t, sc * g))
            s_c, kf_t = (calls[0][2], calls[0][1]) if calls else (float("nan"), -1)
            if calls:  # running median of the calls made so far (the first one before it is available: ~1 s warm-up)
                for dl in [int(x) for x in args.delays.split(",")]:
                    step = np.minimum(np.maximum(np.arange(len(fed)) + dl, init), len(fed) - 1)[q]
                    rs = np.array([np.median([c[2] for c in calls if c[0] <= st] or [calls[0][2]]) for st in step])
                    T = Ts[f"T_world_cam_d{dl}"].copy()
                    T[:, :3, 3] *= rs[:, None]
                    Ts[f"T_world_cam_d{dl}_rs"] = T
                T = Ts[f"T_world_cam_d{args.delays.split(',')[0]}"]
            n_gba = int(slam.ran_global_ba.sum())
            del slam
            torch.cuda.empty_cache()
            out = dict(index=np.arange(n), T_world_cam=T, n_fed=len(fed), t_track=t1 - t0, t_ba=0.0, scale_causal=s_c,
                       scale_frame=kf_t, scale_calls=np.array(calls).reshape(-1, 3), init_step=init,
                       slam_size=np.array([H1, W1]), K=K, sig_cov=cov, n_global_ba=n_gba, gauge=gauge[0], **Ts)
            tmp = f"{out_path}.{os.getpid()}.npz"
            np.savez(tmp, **out)
            os.replace(tmp, out_path)
            print(f"[dpvo] {name}: {n} frames, {len(fed)} fed, causal, track {t1 - t0:.2f}s, scale {s_c:.3f} "
                  f"from frame {kf_t}", flush=True)
            continue
        with torch.no_grad():
            if args.final_ba != 12:  # terminate() runs 12 closing updates; emulate fewer
                slam_update = slam.update
                cnt = [0]

                def limited():
                    cnt[0] += 1
                    if cnt[0] <= args.final_ba:
                        slam_update()

                slam.update = limited
            poses, tst = slam.terminate()  # (fed, 7) c2w xyz + quat xyzw, timestamps = fed frame indices
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        nk = slam.n
        kf_t = np.asarray(slam.pg.tstamps_[:nk]).astype(int)  # counter index of retained keyframes
        pat = slam.pg.patches_[:nk].float().cpu().numpy()  # (nk, M, 3, P, P): x, y (1/4 res), inverse depth
        kf_frames = np.array(slam.tlist)[kf_t]
        del slam
        torch.cuda.empty_cache()

        tst = np.asarray(tst, np.int64)
        T = np.tile(np.eye(4), (n, 1, 1))
        q = np.clip(np.arange(n), tst[0], tst[-1])
        T[:, :3, :3] = Slerp(tst, Rotation.from_quat(poses[:, 3:7]))(q).as_matrix()
        for d in range(3):
            T[:, d, 3] = np.interp(q, tst, poses[:, d])

        cx, cy, invd = pat[:, :, 0, 1, 1] * 4 + 1.5, pat[:, :, 1, 1, 1] * 4 + 1.5, pat[:, :, 2, 1, 1]  # full-res centres
        ring_r = []
        for k, t in enumerate(kf_frames):
            for p in frames.get(int(t), {"pred": []})["pred"]:
                j = p.get("joints3d")
                if j is None or not np.isfinite(j).all() or p["score"] < args.hand_score or j[0, 2] < 0.05:
                    continue
                uv = j[:, :2] / j[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
                (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
                pad = 0.5 * max(x1 - x0, y1 - y0)
                inn = (cx[k] >= x0) & (cx[k] <= x1) & (cy[k] >= y0) & (cy[k] <= y1)
                out = (cx[k] >= x0 - pad) & (cx[k] <= x1 + pad) & (cy[k] >= y0 - pad) & (cy[k] <= y1 + pad) & ~inn
                d = invd[k][out & (invd[k] > 0)]
                if len(d) >= 3:
                    ring_r.append(j[0, 2] * np.median(d))
        m3d_kf, t_m3d = np.full(nk, np.nan), np.nan
        if metric is not None:
            calib = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]])
            ts = []
            for k, t in enumerate(kf_frames):
                torch.cuda.synchronize()
                ta = time.perf_counter()
                dep = cv2.resize(metric(files[t], calib.copy()), (W1, H1))
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - ta)
                ui = np.clip(np.round(cx[k]).astype(int), 0, W1 - 1)
                vi = np.clip(np.round(cy[k]).astype(int), 0, H1 - 1)
                md = dep[vi, ui]
                ok = (invd[k] > 0) & (md > 0.2) & (md < 3.0)
                if ok.sum() >= 5:
                    m3d_kf[k] = np.median(md[ok] * invd[k][ok])
            t_m3d = float(np.median(ts[1:] if len(ts) > 1 else ts))
        out = dict(index=np.arange(n), T_world_cam=T, n_fed=len(fed), n_kf=nk, kf_frames=kf_frames, m3d_kf=m3d_kf,
                   t_track=t1 - t0, t_ba=t2 - t1, t_m3d_kf=t_m3d, slam_size=np.array([H1, W1]), K=K, sig_cov=cov,
                   scale_hand_bg=float(np.median(ring_r)) if ring_r else float("nan"))
        fin = np.isfinite(m3d_kf)
        out["scale_m3d_all"] = float(np.median(m3d_kf[fin])) if fin.any() else float("nan")
        for c in (1, 2, 3, 5):
            sel = np.unique(np.round(np.linspace(0, nk - 1, c + 2)[1:-1]).astype(int)) if nk else []
            v = m3d_kf[sel][np.isfinite(m3d_kf[sel])] if len(sel) else []
            out[f"scale_m3d_{c}"] = float(np.median(v)) if len(v) else out["scale_m3d_all"]
        tmp = f"{out_path}.{os.getpid()}.npz"
        np.savez(tmp, **out)
        os.replace(tmp, out_path)
        print(f"[dpvo] {name}: {n} frames, {len(fed)} fed, {nk} kf, track {t1 - t0:.2f}s ba {t2 - t1:.2f}s "
              f"m3d/kf {t_m3d:.2f}s scale m3d {out['scale_m3d_all']:.3f} hand_bg {out['scale_hand_bg']:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True)
    p.add_argument("--mask_run", required=True)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--config", default="default", choices=["default", "fast"])
    p.add_argument("--opts", nargs="*", default=[], help="DPVO config overrides, e.g. LOOP_CLOSURE True")
    p.add_argument("--final_ba", type=int, default=12)
    p.add_argument("--m3d", default="large", choices=["", "large", "small"])
    p.add_argument("--m3d_amp", action="store_true", help="Metric3D under fp16 autocast")
    p.add_argument("--extra_updates", type=int, default=0, help="extra DPVO updates (BA iterations) after a fed frame")
    p.add_argument("--extra_every", type=int, default=1, help="... after every k-th fed frame")
    p.add_argument("--hand_score", type=float, default=0.8)
    p.add_argument("--no_mask", action="store_true", help="hands not masked (camera independent of the hand model)")
    p.add_argument("--mask_box", action="store_true", help="mask the predicted hand boxes instead of the meshes")
    p.add_argument("--mask_delay", type=int, default=0, help="masks from the hand predictions of frame t - k")
    p.add_argument("--causal", action="store_true", help="online poses (each frame's estimate when it arrives), no final BA")
    p.add_argument("--scale_at", type=int, default=15, help="causal: fed frame after which the metric scale is fixed")
    p.add_argument("--scale_lag", type=int, default=4, help="causal: scale from the frame this many keyframes back")
    p.add_argument("--scale_calls", type=int, default=1, help="causal: Metric3D calls (running median -> *_rs poses)")
    p.add_argument("--scale_every", type=int, default=15, help="causal: fed frames between the Metric3D calls")
    p.add_argument("--trans_gate", type=float, default=0, help="causal_reinit: no translation where rotation-only / full "
                   "residual < this (needs --signals)")
    p.add_argument("--signals", action="store_true", help="causal_reinit: log step_residuals per fed frame (sig_res)")
    p.add_argument("--scale_pw", action="store_true", help="causal: piecewise metric scale from periodic calls (causal_reinit)")
    p.add_argument("--scale_rmed", action="store_true", help="--scale_pw: the running median of the instance's calls so far "
                   "instead of the latest call")
    p.add_argument("--save_calls", action="store_true", help="causal_reinit: also save the calls' keyframe patches + unit poses")
    p.add_argument("--depth_model", default="m3d", choices=["m3d", "moge2"], help="scale calls: Metric3D or bench/world_depth.py")
    p.add_argument("--depth_py", default=sys.executable, help="--depth_model's python env")
    p.add_argument("--scale_delay", type=int, default=5, help="... a call's result is used this many fed frames later")
    p.add_argument("--reinit_rot", type=float, default=0, help="causal: re-initialise when an output step rotates > deg")
    p.add_argument("--reinit_scale", type=float, default=0, help="causal: re-init when consecutive scale calls differ > x")
    p.add_argument("--reinit_sig", default="rot", choices=["rot", "acc"], help="... rotation step or its change")
    p.add_argument("--reinit_min", type=int, default=20, help="... at most once per this many fed frames of an instance")
    p.add_argument("--reinit_back", type=int, default=1, help="... the new instance starts this many fed frames back")
    p.add_argument("--delays", default="0,2,4,8", help="causal: output latencies in fed frames (T_world_cam_d<k>; first = "
                   "T_world_cam); frames before DPVO's initialisation are output at initialisation")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=-1, help=">= 0: seed DPVO's patch sampling per clip")
    p.add_argument("--reverse", action="store_true", help="process the shard's clips last-first (a helper process)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
