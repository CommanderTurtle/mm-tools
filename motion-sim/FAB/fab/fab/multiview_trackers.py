"""fab-multiview --trackers: world-space trackers (e.g. VIVE trackers on the hands, the wrists or the pelvis) fused with
the views by one batch smoother over the clip, before the release's SMPL fit (docs/multiview.md, "Optional: tracker
input"). Off unless --trackers is given: fab-multiview without it is unchanged.

    fab-multiview take/ --out output/mv --trackers left_wrist=left.npz,right_wrist=right.csv

The model (each constant once, below):
  state     every target point of the fit (the 468 dense points and the 14 joints) at every frame, with a
            constant-velocity prior (white acceleration, ACC m/s^2); the points are solved one by one.
  views     each view's points as the fusion places them (fab/multiview.py step 4: the view's similarity onto the
            triangulated joints, plus its nearest joint's residual; a triangulated joint is its triangulated position),
            with an anisotropic covariance: ALONG along the camera's ray (monocular depth), ACROSS across it, divided by
            the view's weight (the fusion's P(visible), at least 0.05, times Geman-McClure (multiview.SIGMA) on the view's
            distance from the fused point). The per-frame means get the release's floor rule (multiview.floor_contacts).
  trackers  a tracker carries the target points of its body segment (BODY: the dense points on those SMPL parts, and
            those joints): x(t) = p(t) + R(t) o, with one static offset o per point in the tracker's frame, estimated with
            the trajectories; covariance TRACKER. Samples flagged invalid, non-finite, or bitwise equal to the sample
            before (frozen) are dropped; a frame's pose is interpolated between the valid samples around it when they are
            at most GAP s apart. Each frame's sample is weighted by Tukey's biweight (GATE) on its distance from the
            views' own estimate (the smoother without trackers): a sample farther than GATE is not used, so lost, jumping
            or drifting samples give way to the views. The offsets start from a robust mean (the same weight) over the
            tracker's own frames; gate and offsets run ROUNDS times. The offsets come from the views, so a bias of the
            views that is constant in the tracker's frame stays in them; the tracker removes the rest.
            Without orientation, R(t) is the segment's rotation in the views' estimate (relative to its first frame).
            In the SMPL fit a carried point weighs (ACROSS / TRACKER)^2 times its sample's weight (the tracker's
            information over a view's), so that the fit follows the tracked segment.
  solve     per point, a banded least squares over the clip (the solution a Rauch-Tung-Striebel smoother gives), the
            offsets by the Schur complement.
"""

import csv
from pathlib import Path

import numpy as np

from fab import multiview as mv

ACC = 10.0  # m/s^2: the motion prior (white acceleration)
ALONG, ACROSS = 0.08, 0.02  # m: a view's point along / across its camera's ray, at weight 1
TRACKER = 0.003  # m: a tracker sample (its pose and the static offset to the body)
GAP = 0.1  # s: a frame's tracker pose is interpolated across gaps of at most this
GATE = 0.15  # m: a tracker sample this far from the views' estimate has no weight (Tukey's biweight)
ROUNDS = 2  # gate and offsets
VEL = 10.0  # m per frame: a vanishing velocity prior (a clip with one measured frame still solves)
# body point -> (the SMPL parts whose dense points the tracker carries, the joints it carries; joints in the 14: r_ankle,
# r_knee, r_hip, l_hip, l_knee, l_ankle, r_wrist, r_elbow, r_shoulder, l_shoulder, l_elbow, l_wrist, neck, head)
BODY = {"pelvis": ((0,), (2, 3)), "chest": ((9,), ()), "head": ((15,), (13,)),
        "left_wrist": ((20, 22), (11,)), "right_wrist": ((21, 23), (6,)),
        "left_foot": ((7, 10), (5,)), "right_foot": ((8, 11), (0,))}
ALIAS = {"left_hand": "left_wrist", "right_hand": "right_wrist", "left_ankle": "left_foot", "right_ankle": "right_foot"}


# ----------------------------------------------------------------------------------------------------------- input


def parse(spec):
    """'left_wrist=a.npz,right_wrist=b.csv' -> [(body point, path)]."""
    out = []
    for item in spec.split(","):
        body, _, path = item.partition("=")
        body = ALIAS.get(body.strip(), body.strip())
        if body not in BODY or not path.strip():
            raise ValueError(f"--trackers {item!r}: want <body point>=<file>, the body point one of {sorted(BODY) + sorted(ALIAS)}")
        out.append((body, Path(path.strip())))
    if len({b for b, _ in out}) < len(out):
        raise ValueError(f"--trackers {spec!r}: one tracker per body point")
    return out


def load(path):
    """A tracker file -> {t (N,) s, xyz (N, 3) m, quat (N, 4) wxyz (tracker -> world) or None, valid (N,) bool}, sorted
    by time: an npz with t, xyz and optional quat, valid; or a csv with the header t,x,y,z and optional qw,qx,qy,qz and
    valid (0 / 1)."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        d = {"t": z["t"], "xyz": z["xyz"], "quat": z["quat"] if "quat" in z else None, "valid": z["valid"] if "valid" in z else None}
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        keys = set(rows[0]) if rows else set()
        col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
        d = {"t": col("t"), "xyz": np.stack([col(k) for k in "xyz"], 1) if rows else np.zeros((0, 3)),
             "quat": np.stack([col(k) for k in ("qw", "qx", "qy", "qz")], 1) if "qw" in keys else None,
             "valid": col("valid") if "valid" in keys else None}
    t = np.asarray(d["t"], np.float64).ravel()
    n = len(t)
    out = {"t": t, "xyz": np.asarray(d["xyz"], np.float64).reshape(n, 3),
           "quat": None if d["quat"] is None else np.asarray(d["quat"], np.float64).reshape(n, 4),
           "valid": np.ones(n, bool) if d["valid"] is None else np.asarray(d["valid"]).ravel().astype(bool)}
    o = np.argsort(t, kind="stable")
    return {k: (v[o] if v is not None else None) for k, v in out.items()}


def good_samples(tr):
    """The samples a pose is read from: flagged valid, finite, a unit quaternion (within 0.1), and not bitwise equal to
    the sample before (a frozen tracker repeats its last pose)."""
    x = tr["xyz"] if tr["quat"] is None else np.concatenate([tr["xyz"], tr["quat"]], 1)
    ok = tr["valid"] & np.isfinite(x).all(1) & np.isfinite(tr["t"])
    if tr["quat"] is not None:
        ok &= np.abs(np.linalg.norm(np.nan_to_num(tr["quat"]), axis=1) - 1) < 0.1
    ok[1:] &= ~(x[1:] == x[:-1]).all(1)
    return ok


def at_frames(tr, ok, times, A=np.eye(3), gap=GAP):
    """The tracker's pose at the frame times: linear (position) and slerp (rotation) between the valid samples around
    each time, when they are at most gap s apart; turned by A (the rig's up turn). -> p (T, 3), R (T, 3, 3) (None without
    quaternions), have (T,)."""
    from scipy.spatial.transform import Rotation

    T = len(times)
    i = np.flatnonzero(ok)
    t = tr["t"][i]
    have = np.zeros(T, bool)
    p, R = np.zeros((T, 3)), (None if tr["quat"] is None else np.tile(np.eye(3), (T, 1, 1)))
    if len(i) == 0:
        return p, R, have
    k1 = np.searchsorted(t, times, side="left")  # the first valid sample at or after the frame time
    k0 = np.where((k1 < len(t)) & (t[np.minimum(k1, len(t) - 1)] == times), k1, k1 - 1)
    have = (k0 >= 0) & (k1 < len(t))
    k0, k1 = np.clip(k0, 0, len(t) - 1), np.clip(k1, 0, len(t) - 1)
    dt = t[k1] - t[k0]
    have &= dt <= gap
    a = np.where(dt > 0, (times - t[k0]) / np.where(dt > 0, dt, 1.0), 0.0)
    p = (1 - a)[:, None] * tr["xyz"][i[k0]] + a[:, None] * tr["xyz"][i[k1]]
    p = p @ np.asarray(A).T
    if R is not None:
        q0, q1 = tr["quat"][i[k0]], tr["quat"][i[k1]]
        r0, r1 = Rotation.from_quat(q0[:, [1, 2, 3, 0]]), Rotation.from_quat(q1[:, [1, 2, 3, 0]])
        R = np.asarray(A) @ (r0 * Rotation.from_rotvec((r0.inv() * r1).as_rotvec() * a[:, None])).as_matrix()
    p[~have] = 0.0
    return p, R, have


# ----------------------------------------------------------------------------------------------------------- views


def view_points(views, obs, f, vis_idx):
    """One frame's per-view target points as the fusion places them (multiview.fuse_frame): obs[v] the subject's slim dict
    in view v or None, f the frame's fusion record. -> Q (V, M, 3) world, the dense points then the 14 joints (NaN: no
    subject in the view), w (V, M) the fusion's weights (P(visible) of the point's nearest joint, at least W_FLOOR)."""
    V, S = len(views), len(vis_idx)
    Q, w = np.full((V, S + 14, 3), np.nan), np.zeros((V, S + 14))
    tj = f["tri"][mv.J14]
    for v, o in enumerate(obs):
        if o is None:
            continue
        T = views[v].T_world_cam
        s, R, t = f["align"][v] if f["align"][v] is not None else (1.0, np.eye(3), np.zeros(3))
        kw = mv.to_world(T, o["kp3d"].astype(np.float64))
        sw = mv.to_world(T, o["surf"].astype(np.float64))
        Pk, Ps = s * kw @ R.T + t, s * sw @ R.T + t
        res = np.where(tj[:, None], f["kp"][mv.J14] - Pk[mv.J14], 0.0)
        Q[v, :S] = Ps + res[vis_idx - 70]
        Q[v, S:] = np.where(tj[:, None], f["kp"][mv.J14], Pk[mv.J14])
        wk = np.maximum(o["vis"].astype(np.float64), mv.W_FLOOR)
        w[v, :S], w[v, S:] = wk[vis_idx], wk[mv.J14]
    return Q, w


def vision_info(Q, w, centres, fused):
    """The views' information on each point: Q (V, M, 3), w (V, M), the camera centres (V, 3), the fused points (M, 3)
    -> Lam (M, 3, 3), eta (M, 3) (the mean is Lam^-1 eta)."""
    d = Q - centres[:, None]
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    r2 = ((Q - fused) ** 2).sum(-1)
    g = np.where(np.isfinite(r2), w * (mv.SIGMA ** 2 / (mv.SIGMA ** 2 + np.nan_to_num(r2))) ** 2, 0.0)
    dd = np.nan_to_num(d[..., :, None] * d[..., None, :])
    info = g[..., None, None] * (dd / ALONG ** 2 + (np.eye(3) - dd) / ACROSS ** 2)
    return info.sum(0), (info @ np.nan_to_num(Q)[..., None])[..., 0].sum(0)


# ----------------------------------------------------------------------------------------------------------- solve


def _prior(T, fps):
    """The motion prior's normal matrix (one coordinate): its diagonals (main, first, second)."""
    d = [np.zeros(T), np.zeros(max(T - 1, 0)), np.zeros(max(T - 2, 0))]
    for coef, q in (((1.0, -1.0), 1.0 / VEL ** 2), ((1.0, -2.0, 1.0), 1.0 / (ACC / fps ** 2) ** 2)):
        n = T - len(coef) + 1
        if n <= 0:
            continue
        for a, ca in enumerate(coef):
            for b, cb in enumerate(coef[a:], a):
                d[b - a][a: a + n] += q * ca * cb
    return d


def solve_point(Lam, eta, fps, trk=None, prior=None):
    """One point's trajectory: Lam (T, 3, 3), eta (T, 3) its per-frame information, the motion prior, and optionally a
    tracker trk = (p (T, 3), R (T, 3, 3), wt (T,) information per frame) with a static offset o in the tracker's frame:
    minimises sum (x - mu)' Lam (x - mu) + prior + sum wt |x - R o - p|^2. -> x (T, 3), o (3,) or None."""
    from scipy.linalg import solveh_banded

    T = len(Lam)
    n = 3 * T
    main, off1, off2 = prior if prior is not None else _prior(T, fps)
    ab = np.zeros((7, n))  # lower band form: ab[i - j, j] = A[i, j]
    for c in range(3):
        ab[0, c::3] += main
        ab[3, c: n - 3: 3] += off1
        ab[6, c: n - 6: 3] += off2
    for a in range(3):
        for b in range(a + 1):
            ab[a - b, b::3] += Lam[:, a, b]
    rhs = eta.reshape(n).copy()
    if trk is None or not np.any(trk[2] > 0):
        return solveh_banded(ab, rhs, lower=True).reshape(T, 3), None
    p, R, wt = trk
    for c in range(3):
        ab[0, c::3] += wt
    rhs += (wt[:, None] * p).reshape(n)
    B = (-wt[:, None, None] * R).reshape(n, 3)
    Z = solveh_banded(ab, np.concatenate([rhs[:, None], B], 1), lower=True)
    S = wt.sum() * np.eye(3) - B.T @ Z[:, 1:] + 1e-9 * np.eye(3)
    o = np.linalg.solve(S, -np.einsum("t,tji,tj->i", wt, R, p) - B.T @ Z[:, 0])
    return (Z[:, 0] - Z[:, 1:] @ o).reshape(T, 3), o


def segment_rotation(xs, use):
    """The rotation of a point set xs (T, P, 3) relative to its first used frame (Kabsch per frame)."""
    R = np.tile(np.eye(3), (len(xs), 1, 1))
    if xs.shape[1] < 3 or not use.any():
        return R
    k0 = np.flatnonzero(use)[0]
    ref = xs[k0] - xs[k0].mean(0)
    for t in range(len(xs)):
        U, _, Vt = np.linalg.svd((xs[t] - xs[t].mean(0)).T @ ref)
        R[t] = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R


def _tukey(r):
    return np.clip(1 - (r / GATE) ** 2, 0, None) ** 2


def robust_offsets(xs, p, R, use, iters=5):
    """The static offsets (P, 3) in the tracker's frame of the points xs (T, P, 3) (the views' estimate) from the tracker
    poses p, R on the frames use: from the median, Tukey's biweight (GATE) on each frame's median distance."""
    y = np.einsum("tji,tpj->tpi", R[use], xs[use] - p[use, None])  # R' (x - p)
    o = np.median(y, 0)
    for _ in range(iters):
        g = _tukey(np.median(np.linalg.norm(y - o, axis=-1), 1))
        o = (g[:, None, None] * y).sum(0) / max(g.sum(), 1e-12)
    return o


def smooth(spec, views, info, times, people, sel, frames, valid, surf_w, j14_w, vis_idx, surface_idx, j14_reg, body,
           robust=True):
    """fab-multiview's targets with trackers: spec the --trackers string (or [(body point, tracker dict)]); the rest as
    in multiview.run. -> surf (T, S, 3), surf_w (T, S), j14 (T, 14, 3), j14_w (T, 14) (NaN / 0 where not valid), notes.
    robust=False keeps every valid sample at full weight (an ablation)."""
    trackers = [(b, load(p)) for b, p in parse(spec)] if isinstance(spec, str) else list(spec)
    T, V, S = len(frames), len(views), len(vis_idx)
    M, fps = S + 14, float(info["fps"])
    centres = np.stack([v.T_world_cam[:3, 3] for v in views])
    Lam, eta, mu = np.zeros((T, M, 3, 3)), np.zeros((T, M, 3)), np.full((T, M, 3), np.nan)
    for t, f in enumerate(frames):
        if not valid[t]:
            continue
        obs = [people[v][t][sel[t, v]] if sel[t, v] >= 0 else None for v in range(V)]
        Q, w = view_points(views, obs, f, vis_idx)
        Lam[t], eta[t] = vision_info(Q, w, centres, np.concatenate([f["surf"], f["j14"]]))
        seen = np.trace(Lam[t], axis1=1, axis2=2) > 0
        mu[t, seen] = np.linalg.solve(Lam[t, seen], eta[t, seen][..., None])[..., 0]
    n_floor = 0
    if info["floor_z"] is not None:  # the release's floor rule on the views' per-frame means
        ms, mj, n_floor = mv.floor_contacts(frames, valid, mu[:, :S], mu[:, S:], vis_idx, mv.contact_parts(j14_reg, body),
                                            float(info["floor_z"]))
        mu = np.concatenate([ms, mj], 1)
        eta = (Lam @ np.nan_to_num(mu)[..., None])[..., 0]
    prior = _prior(T, fps)
    xv = np.stack([solve_point(Lam[:, m], eta[:, m], fps, prior=prior)[0] for m in range(M)], 1)  # the views alone
    x = xv.copy()
    part = body.weights.cpu().numpy()[np.asarray(surface_idx)].argmax(1)
    A = mv.up_rotation(info.get("up", [0.0, 0.0, 1.0]))
    w_out = np.concatenate([surf_w, j14_w], 1).astype(np.float64)
    notes = {"constants": {"acc_m_s2": ACC, "along_m": ALONG, "across_m": ACROSS, "tracker_m": TRACKER, "gap_s": GAP,
                           "gate_m": GATE, "rounds": ROUNDS}, "robust": bool(robust), "floor_contacts": n_floor}
    for name, tr in trackers:
        parts, joints = BODY[name]
        pts = np.r_[np.flatnonzero(np.isin(part, parts)), S + np.asarray(joints, int)].astype(int)
        ok = good_samples(tr)
        p, R, have = at_frames(tr, ok, np.asarray(times, np.float64), A)
        oriented = R is not None
        if not oriented:
            R = segment_rotation(xv[:, pts], valid)
        use = have & valid
        rec = {"points": int(len(pts)), "samples": int(len(ok)), "dropped": int((~ok).sum()), "oriented": oriented,
               "frames_with_pose": float(have.mean())}
        if not use.any():
            notes[name] = {**rec, "used": False}
            print(f"[multiview] tracker {name}: no valid pose on a frame with the subject, not used", flush=True)
            continue
        o = robust_offsets(xv[:, pts], p, R, use)
        for _ in range(ROUNDS):
            e = np.median(np.linalg.norm(p[:, None] + np.einsum("tij,pj->tpi", R, o) - xv[:, pts], axis=-1), 1)
            g = np.where(valid, _tukey(e) if robust else 1.0, 1.0) * have
            for k, m in enumerate(pts):
                x[:, m], om = solve_point(Lam[:, m], eta[:, m], fps, (p, R, g / TRACKER ** 2), prior)
                o[k] = o[k] if om is None else om
        w_out[:, pts] = np.maximum(w_out[:, pts], g[:, None] * (ACROSS / TRACKER) ** 2)
        jo = o[-len(joints)] if len(joints) else np.median(o, 0)
        notes[name] = {**rec, "used": True, "accepted": float((g[use] >= 0.5).mean()), "offset_m": np.round(jo, 4).tolist(),
                       "views_m_median": float(np.median(e[use]))}
        print(f"[multiview] tracker {name}: {len(pts)} points, a pose on {have.mean() * 100:.0f} % of frames, "
              f"{notes[name]['accepted'] * 100:.0f} % accepted, offset {np.linalg.norm(jo) * 100:.1f} cm", flush=True)
    x[~valid], w_out[~valid] = np.nan, 0.0
    return x[:, :S], w_out[:, :S], x[:, S:], w_out[:, S:], notes
