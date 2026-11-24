"""Causal fusion of an up-to-scale DPVO camera (bench/world_dpvo.py --causal) with the learned camera-motion head
(bench/world_camhead.py, same --stride): per fed step, DPVO's relative pose D_j = inv(T_{j-1}) T_j is kept for its local
precision and its translation is put at the head's metric scale s_j = sum |t_head| / sum |t_dpvo| over the last --win fed
steps (path lengths: the head is metric, DPVO's scale drifts); when D_j's rotation disagrees with the head's step by more
than --thr deg (or --k_sigma x the head's predicted rotation error), the head's step replaces it (DPVO tracking failure,
or DPVO not initialised yet). The output trajectory composes the chosen steps; skipped frames hold. With a metric DPVO input
(e.g. --scale_pw) --w_head w < 1 blends: the head's path-length correction enters as s_j^w (w = 0.5: geometric mean of the
two metric scales). --t_sigma k: a step whose (scaled) DPVO translation is further than k x the head's predicted
translation error (Laplace scale, x 3 axes) from the head's keeps DPVO's rotation but takes the head's translation
(spurious VO drift / lost translation); --t_sigma_low k' --t_sigma_cov c: k' instead of k on steps whose static coverage
(the smaller of its two fed frames' share of DPVO's patch grid left unmasked: bench/world_dpvo.py sig_cov, or --cov <dir>
of <clip>.npz with cov per fed frame, bench/world_coverage.py) is below c: where the hands fill the frame DPVO's few
patches sit on the hand edges and moving objects. --blend_sigma s: otherwise the step rotation is the inverse-variance
blend of DPVO's (error scale s deg, e.g. its HOT3D-train mean per-step error) and the head's (its predicted error scale).
--bridge: where DPVO is not tracking (before it initialises: no residuals yet in sig_res; re-initialisation warm-up: held)
the head's step.
--ev_lo / --ev_hi: the head's scale exponent fades from --w_head to 0 as DPVO's translation evidence (median over --win of
the rotation-only minus full-pose residual, sig_res) rises from ev_lo to ev_hi px (parallax: metric VO translation trusted).
--head_every k: the head ran on every k-th fed frame (bench/world_camhead.py --stride k x DPVO's): the decisions (scale,
replacement) are made at the head's steps on DPVO's k-step relative pose, the fed frames in between follow DPVO's own
increments at the current scale (causal; what scripts/benchmark_world_e2e.py --head_every times).
    python -m bench.world_fuse --dpvo <dir>[#key][@scale key] --head <dir> --out <dir> --win 30 --thr 5
-> <out>/<clip>.npz: T_world_cam = T_world_cam_d0 (metric, scale_causal 1), fused_scale (fed), fallback (fed bool)
"""
import argparse
import glob
import os

import numpy as np


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def ev_weight(ev, j, win, lo, hi):
    """Head-weight factor from DPVO's translation evidence (rotation-only minus full-pose residual, px): its median over the
    last win fed steps with residuals; 1 at <= lo px (no parallax: monocular VO translation unreliable), 0 at >= hi."""
    e = ev[max(j - win + 1, 0):j + 1]
    e = e[np.isfinite(e)]
    return 1.0 if not len(e) else float(np.clip((hi - np.median(e)) / (hi - lo), 0, 1))


def fuse(P, rel, lv, win, thr, k_sigma, w_head=1.0, t_sigma=0.0, blend_sigma=0.0, off=None, ev=None, ev_lo=0.0, ev_hi=0.0,
         ts_step=None):
    from scipy.spatial.transform import Rotation

    """P (nf, 4, 4) DPVO c2w of the fed frames (its units), rel (nf, 4, 4) head steps (rel[0] = I), lv (nf, 2); ts_step (nf,)
    the --t_sigma of each step (None: t_sigma everywhere)."""
    nf = len(P)
    out, s, hd, hh = [np.eye(4)], 1.0, [], []
    scales, fb = np.ones(nf), np.zeros(nf, bool)
    for j in range(1, nf):
        D, H = np.linalg.inv(P[j - 1]) @ P[j], rel[j]
        hd.append(np.linalg.norm(D[:3, 3]))
        hh.append(np.linalg.norm(H[:3, 3]))
        if sum(hd[-win:]) > 1e-9:
            s = sum(hh[-win:]) / sum(hd[-win:])
        a = rot_deg(D[:3, :3].T @ H[:3, :3])
        bad = (thr > 0 and a > thr) or (k_sigma > 0 and a > k_sigma * np.exp(lv[j, 0])) or (off is not None and off[j])
        if bad:
            step = H
        else:
            step = D.copy()
            step[:3, 3] *= s ** (w_head * (ev_weight(ev, j, win, ev_lo, ev_hi) if ev is not None and ev_hi > 0 else 1.0))
            if blend_sigma > 0:  # rotation: geodesic inverse-variance blend
                w = blend_sigma ** 2 / (blend_sigma ** 2 + np.exp(lv[j, 0]) ** 2)
                d = Rotation.from_matrix(D[:3, :3].T @ H[:3, :3]).as_rotvec()
                step[:3, :3] = D[:3, :3] @ Rotation.from_rotvec(w * d).as_matrix()
            ts = t_sigma if ts_step is None else ts_step[j]
            if ts > 0 and np.abs(step[:3, 3] - H[:3, 3]).sum() * 100 > ts * 3 * np.exp(lv[j, 1]):
                step[:3, 3] = H[:3, 3]
        scales[j], fb[j] = s, bad
        out.append(out[-1] @ step)
    return np.stack(out), scales, fb


def fuse_coarse(P, rel, lv, k, win, thr, k_sigma, w_head=1.0, off=None):
    """fuse() with the head on every k-th fed frame: rel[m] = the head's step from fed frame (m - 1) k to m k."""
    nf = len(P)
    out, s, hd, hh = [np.eye(4)], 1.0, [], []
    scales, fb = np.ones(nf), np.zeros(nf, bool)
    for j in range(1, nf):
        if j % k:  # between head steps: DPVO's increment at the current scale (held while DPVO is not tracking)
            step = np.linalg.inv(P[j - 1]) @ P[j]
            step[:3, 3] *= s ** w_head
            out.append(out[-1] @ (np.eye(4) if off is not None and off[j] else step))
            scales[j] = s
            continue
        m = j // k
        D, H = np.linalg.inv(P[j - k]) @ P[j], rel[m]
        hd.append(np.linalg.norm(D[:3, 3]))
        hh.append(np.linalg.norm(H[:3, 3]))
        if sum(hd[-max(win // k, 1):]) > 1e-9:
            s = sum(hh[-max(win // k, 1):]) / sum(hd[-max(win // k, 1):])
        a = rot_deg(D[:3, :3].T @ H[:3, :3])
        bad = (thr > 0 and a > thr) or (k_sigma > 0 and a > k_sigma * np.exp(lv[m, 0])) or (off is not None and off[j])
        if bad:
            step = H
        else:
            step = D.copy()
            step[:3, 3] *= s ** w_head
        scales[j], fb[j] = s, bad
        out.append(out[j - k] @ step)
    return np.stack(out), scales, fb


def main(args):
    os.makedirs(args.out, exist_ok=True)
    spec, sk = (args.dpvo.split("@") + [""])[:2]
    src, key = (spec.split("#") + ["T_world_cam_d0"])[:2]
    nfb, n_clips = [], 0
    for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
        name = os.path.basename(f)
        hf = os.path.join(args.head, name)
        if ".npz." in name or not os.path.exists(hf):  # a stage's temporary output (bench.world_camera.partial)
            continue
        d, h = np.load(f), np.load(hf)
        T = (d[key] if key in d else d["T_world_cam"]).copy()
        if sk:  # e.g. @scale_cal: the input's metric scale (bench/world.py convention)
            T[:, :3, 3] *= float(d[sk])
        n, nf = len(T), int(h["n_fed"]) * args.head_every
        st = int(np.ceil(n / nf))
        fed = np.arange(0, n, st)
        nf = len(fed)
        assert args.head_every > 1 or nf == len(h["rel"]), (name, n, nf)
        off = None
        if args.bridge:  # DPVO not tracking: before its first residuals, and held re-init steps
            r = d["sig_res"][:, 0]
            off = np.arange(nf) < (np.argmax(r > 0) if (r > 0).any() else nf)
            off |= d["held"].astype(bool) if "held" in d else False
        if args.head_every > 1:
            Tf, sc, fb = fuse_coarse(T[fed], h["rel"], h["logvar"], args.head_every, args.win, args.thr, args.k_sigma,
                                     args.w_head, off)
        else:
            ev = None
            if args.ev_hi > 0:  # translation evidence per fed step (needs --signals runs); nan where DPVO has no residuals
                r = d["sig_res"]
                ev = np.where(r[:, 0] > 0, r[:, 1] - r[:, 0], np.nan)
            ts_step = None
            if args.t_sigma_cov > 0:  # static coverage per fed frame: --cov dir, else the DPVO run's sig_cov
                cov = np.asarray(np.load(os.path.join(args.cov, name))["cov"] if args.cov else d["sig_cov"], np.float64)
                assert len(cov) == nf, (name, len(cov), nf)
                low = np.r_[False, np.minimum(cov[:-1], cov[1:]) < args.t_sigma_cov]  # step j: fed frames j - 1, j
                ts_step = np.where(low, args.t_sigma_low, args.t_sigma)
            Tf, sc, fb = fuse(T[fed], h["rel"], h["logvar"], args.win, args.thr, args.k_sigma, args.w_head, args.t_sigma,
                              args.blend_sigma, off, ev, args.ev_lo, args.ev_hi, ts_step)
        Tout = Tf[np.searchsorted(fed, np.arange(n), side="right") - 1]
        np.savez(os.path.join(args.out, name), index=np.arange(n), T_world_cam=Tout, T_world_cam_d0=Tout, scale_causal=1.0,
                 fused_scale=sc, fallback=fb)
        nfb.append(fb.mean())
        n_clips += 1
    print(f"{args.out}: {n_clips} clips, head-step share mean {np.mean(nfb):.3f}, clips with any {np.mean(np.array(nfb) > 0):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dpvo", required=True, help="causal DPVO npz dir[#<pose key>][@<scale key>] (DPVO units or metric)")
    p.add_argument("--head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--win", type=int, default=30, help="fed steps of the path-length scale ratio")
    p.add_argument("--thr", type=float, default=5.0, help="deg; DPVO step replaced by the head's above this (0: never)")
    p.add_argument("--k_sigma", type=float, default=0.0, help="... or above k x the head's predicted rotation error")
    p.add_argument("--head_every", type=int, default=1, help="the head ran on every k-th fed frame")
    p.add_argument("--bridge", action="store_true", help="head steps where DPVO is not tracking (needs --signals runs)")
    p.add_argument("--blend_sigma", type=float, default=0.0, help="DPVO per-step rotation error scale (deg) for blending")
    p.add_argument("--t_sigma", type=float, default=0.0, help="head translation replaces DPVO's beyond k x its error")
    p.add_argument("--t_sigma_low", type=float, default=0.0, help="--t_sigma on low static-coverage steps")
    p.add_argument("--t_sigma_cov", type=float, default=0.0, help="coverage below which --t_sigma_low applies (0: off)")
    p.add_argument("--cov", default="", help="dir of <clip>.npz with cov (fed) instead of the DPVO run's sig_cov")
    p.add_argument("--w_head", type=float, default=1.0, help="exponent of the head's scale correction (metric DPVO input)")
    p.add_argument("--ev_lo", type=float, default=0.0, help="px; full --w_head at or below this translation evidence")
    p.add_argument("--ev_hi", type=float, default=0.0, help="px; no head scale correction at or above (0: off)")
    main(p.parse_args())
