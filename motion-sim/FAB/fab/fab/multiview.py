"""fab-multiview: one person's motion from N synchronised, calibrated cameras, by fusing the released FAB model's per-view
outputs with the model's own visibility predictions (docs/multiview.md).

Steps (one rule each):
  1. rig     per camera: frames (a folder or a video), intrinsics K (+ distortion), T_world_cam (camera -> world) and
             timestamps or an offset; rig.json, or a take folder (world.json + one frame folder per camera). Frames are
             matched to the first camera's timeline (the nearest frame within half a frame period).
  2. views   the released FABPredictor (its defaults) on each camera's frames with the known K (undistorted first).
  3. subject the first frame: the cross-view match (one person per view) whose triangulated keypoints reproject within
             5 % of the person's height, the most prominent (summed 2D height) of those; then per view the person nearest
             the projection of the subject's previous 3D keypoints (within 25 % of its height); lost in every view: the
             first rule again. --subject_boxes overrides.
  4. fuse    per frame: the 70 MHR keypoints + 14 joints triangulated from their 2D (weighted DLT over the views that
             see the person, weight = the visibility head's P(visible), at least 0.05; a view whose ray passes > 5 cm
             from the point is dropped, a point with two views that disagree is not triangulated); each view's dense points
             + joints placed on the triangulated joints by a robust similarity, and each dense point moved by the
             residual of its nearest joint (on the SMPL template); then each of the 468 dense points and 14 joints is the
             robust (Geman-McClure, 5 cm) weighted mean over the views, weight = the P(visible) of the point's nearest
             joint (at least 0.05); a triangulated joint keeps its triangulated position.
  5. floor   with the rig's floor_z (else off): a knee or hand the image contact head puts on the floor (static contact
             >= 0.5 in the view that sees it best) has its fused points moved vertically onto the floor; a foot below
             the floor is raised onto it, never pulled down (shoes); both within 20 cm.
  6. fit     one SMPL fit per frame to the fused points: the released one-pass fitter, two Gauss-Newton steps (the
             release's FastGN, Geman-McClure 50 mm, per-point weights = the fused confidence), then the clip's median
             betas and two more steps with the shape fixed; the release's world smoothing (rotations sigma 2, pelvis
             sigma 3 frames at 30 fps: 67 / 100 ms at any rate).
  7. export  fab-world's layout in the rig's world (z up; a flat floor at floor_z), so fab-retarget runs on it unchanged.
With one camera (--views of one), the output is the release's own single-view result: its per-frame SMPL fit mapped by
T_world_cam, the clip's median betas, the release's world smoothing (single_view_body + world_body).

    fab-multiview rig.json --out output/mv            # or a take folder holding world.json
    fab-multiview rig.json --out output/mv --views cam0,cam1 --subject_boxes boxes/   # <camera name>.npz box (T, 4)
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

IMAGES = (".jpg", ".jpeg", ".png", ".bmp")
N_KP = 84  # 70 MHR keypoints + the 14 evaluation joints: the visibility head's points
J14 = slice(70, 84)
VIS_MIN = 0.5  # subject following: the person's own keypoints with P(visible) >= this
GAP_MAX = 0.05  # m: a view's ray may pass this far from its triangulated point
SIGMA = 0.05  # m: Geman-McClure scale of the alignment and the fusion
W_FLOOR = 0.05  # a point's weight in a view is at least this (every view keeps a say)
INIT_ERR = 0.05  # init: median reprojection error / person height of a consistent cross-view match
FOLLOW_ERR = 0.25  # follow: mean distance to the projected previous keypoints / person height
MIN_TRI = 6  # joints a view's alignment needs
CONTACT_CAP = 0.20  # m: a predicted floor contact is placed on the floor when it is at most this far from it
FIT_BATCH = 64  # frames per Gauss-Newton batch (GPU memory)
# contact channels (l_heel, l_toe, r_heel, r_toe, l_hand, r_hand, l_knee, r_knee) -> the MHR keypoint whose visibility weighs them
CONTACT_KP = [17, 15, 20, 18, 62, 41, 11, 12]


# ----------------------------------------------------------------------------------------------------------- rig


class View:
    """One camera: name, K (3, 3), dist (k,) or None, T_world_cam (4, 4), frames (image paths) or video, times (n,) s."""

    def __init__(self, name, K, T_world_cam, frames=None, video=None, dist=None, times=None):
        self.name, self.K = name, np.asarray(K, np.float64).reshape(3, 3)
        self.T_world_cam = np.asarray(T_world_cam, np.float64).reshape(4, 4)
        self.frames, self.video = frames, video
        self.dist = None if dist is None or not np.any(dist) else np.asarray(dist, np.float64).ravel()
        self.times = None if times is None else np.asarray(times, np.float64).ravel()

    def __len__(self):
        if self.frames is not None:
            return len(self.frames)
        import cv2

        cap = cv2.VideoCapture(str(self.video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def images(self, ids):
        """RGB frames ids (increasing), undistorted to the pinhole K."""
        import cv2

        want = set(int(i) for i in ids)
        if self.frames is not None:
            src = ((i, cv2.imread(str(self.frames[i]))) for i in sorted(want))
        else:
            def read():
                cap, i = cv2.VideoCapture(str(self.video)), 0
                while True:
                    ok, f = cap.read()
                    if not ok or i > max(want):
                        break
                    if i in want:
                        yield i, f
                    i += 1
                cap.release()
            src = read()
        for i, bgr in src:
            if bgr is None:
                raise FileNotFoundError(f"{self.name}: frame {i} could not be read")
            if self.dist is not None:
                bgr = cv2.undistort(bgr, self.K, self.dist, None, self.K)
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _K(k):
    if isinstance(k, dict):
        return [[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]
    k = np.asarray(k, np.float64).ravel()
    return k.reshape(3, 3) if k.size == 9 else [[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]


def _frames(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMAGES)


def _times(path):
    a = np.load(path) if str(path).endswith(".npy") else np.loadtxt(path)
    return a[:, 0] if a.ndim == 2 else a


def up_rotation(up):
    """The rotation taking the unit vector up to +z (the smallest one)."""
    from scipy.spatial.transform import Rotation

    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    v, c = np.cross(up, [0.0, 0, 1]), up[2]
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def load_rig(path):
    """rig.json, or a take folder with world.json -> (views, info {fps, floor_z, up, source}). The world is turned so
    that `up` (default +z) is +z; floor_z is along it."""
    path = Path(path)
    if path.is_dir():
        W = json.loads((path / "world.json").read_text())
        views = []
        for d, c in W["cameras"].items():
            if c.get("T_world_cam") is None or not (path / d).is_dir():
                continue
            ti = path / d / "frame_index.npy"
            views.append(View(d, _K(c["K"]), c["T_world_cam"], frames=_frames(path / d), dist=c.get("dist"),
                              times=_times(ti) if ti.exists() else None))
        info = {"fps": float(W.get("fps", 30.0)), "floor_z": W.get("floor_z"), "up": W.get("up", [0.0, 0.0, 1.0])}
    else:
        R = json.loads(path.read_text())
        root = path.parent
        views = []
        for c in R["cameras"]:
            src = root / c.get("frames", c.get("video", ""))
            times = _times(root / c["timestamps"]) if c.get("timestamps") else None
            if times is not None and c.get("offset"):
                times = times + float(c["offset"])
            v = View(c["name"], _K(c["K"]), c["T_world_cam"], frames=_frames(src) if src.is_dir() else None,
                     video=None if src.is_dir() else src, dist=c.get("dist"), times=times)
            v.offset = float(c.get("offset", 0.0))
            views.append(v)
        info = {"fps": float(R.get("fps", 30.0)), "floor_z": R.get("floor_z"), "up": R.get("up", [0.0, 0.0, 1.0])}
    if not views:
        raise ValueError(f"{path}: no camera with T_world_cam and frames")
    A = up_rotation(info["up"])
    for v in views:
        v.T_world_cam = np.block([[A @ v.T_world_cam[:3, :3], (A @ v.T_world_cam[:3, 3])[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    info["source"] = str(path)
    return views, info


def sync(views, fps):
    """-> ids (T, N) frame index of each view on the first view's timeline (-1: no frame within half a period), t (T,)."""
    n0 = len(views[0])
    if all(v.times is not None for v in views):
        t = views[0].times[:n0]
    else:
        t = np.arange(n0) / fps + getattr(views[0], "offset", 0.0)
    ids = np.full((len(t), len(views)), -1, int)
    for j, v in enumerate(views):
        tv = v.times[: len(v)] if v.times is not None else np.arange(len(v)) / fps + getattr(v, "offset", 0.0)
        k = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1)
        k = np.where(np.abs(tv[k - 1] - t) <= np.abs(tv[k] - t), k - 1, k)
        ids[:, j] = np.where(np.abs(tv[k] - t) <= 0.5 / fps + 1e-6, k, -1)
    return ids, t


# ----------------------------------------------------------------------------------------------------------- geometry


def to_world(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def project(T_world_cam, K, X):
    """World points X (..., 3) -> pixels (..., 2), depth (...)."""
    Tc = np.linalg.inv(T_world_cam)
    Xc = X @ Tc[:3, :3].T + Tc[:3, 3]
    z = Xc[..., 2]
    q = Xc @ K.T
    return q[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None], z


def rays(T_world_cam, K, uv):
    """Pixels (..., 2) -> unit world ray directions (..., 3) and the camera centre (3,)."""
    d = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], -1) @ np.linalg.inv(K).T @ T_world_cam[:3, :3].T
    return d / np.linalg.norm(d, axis=-1, keepdims=True), T_world_cam[:3, 3]


def triangulate(Ts, Ks, uv, w, vis_min=0.0, gap_max=GAP_MAX):
    """Weighted DLT of P points seen in V views: Ts (V, 4, 4) T_world_cam, Ks (V, 3, 3), uv (V, P, 2) px, w (V, P)
    weights (P(visible); weights <= 0, below vis_min, and NaN pixels are left out). A view whose ray passes more than gap_max (m)
    from the point is dropped while more than two remain; a point with fewer than two views, or two that disagree, or
    behind a camera, is not triangulated. -> X (P, 3) (NaN where not), ok (P,), used (V, P)."""
    V, P = w.shape
    use = (w > 0) & (w >= vis_min) & np.isfinite(uv).all(-1)
    X = np.full((P, 3), np.nan)
    rows = []
    for v in range(V):
        Tc = np.linalg.inv(Ts[v])
        M = Tc[:3]  # world -> camera (3, 4)
        xn = np.concatenate([np.nan_to_num(uv[v]), np.ones((P, 1))], 1) @ np.linalg.inv(Ks[v]).T  # normalised (P, 3)
        rows.append(np.stack([xn[:, :1] * M[2] - M[0], xn[:, 1:2] * M[2] - M[1]], 1))  # (P, 2, 4)
    rows = np.stack(rows, 1)  # (P, V, 2, 4)
    dirs = [rays(Ts[v], Ks[v], np.nan_to_num(uv[v])) for v in range(V)]
    for _ in range(max(V - 1, 1)):
        n = use.sum(0)
        ok = n >= 2
        A = (rows * (np.where(use, w, 0.0).T)[:, :, None, None]).reshape(P, 2 * V, 4)
        X = np.full((P, 3), np.nan)
        if ok.any():
            _, _, Vt = np.linalg.svd(A[ok])
            h = Vt[:, -1]
            X[ok] = h[:, :3] / np.where(np.abs(h[:, 3:]) < 1e-12, 1e-12, h[:, 3:])
        gap = np.full((V, P), np.inf)
        for v in range(V):
            d, c = dirs[v]
            r = X - c
            gap[v] = np.linalg.norm(r - (r * d).sum(-1, keepdims=True) * d, axis=-1)
            gap[v][(r * d).sum(-1) <= 0] = np.inf  # behind the camera
        gap = np.where(use, np.nan_to_num(gap, nan=np.inf), -1.0)
        worst = gap.argmax(0)
        drop = ok & (gap.max(0) > gap_max) & (n > 2)
        if not drop.any():
            break
        use[worst[drop], np.flatnonzero(drop)] = False
    ok = (use.sum(0) >= 2) & (np.where(use, gap, -1.0).max(0) <= gap_max)
    X[~ok] = np.nan
    return X, ok, use & ok[None]


def similarity(X, Y, w, sigma=SIGMA, iters=3):
    """Robust weighted similarity (s, R, t) minimising sum w rho(|s R X + t - Y|) (Umeyama; Geman-McClure IRLS)."""
    wi = np.asarray(w, np.float64)
    for it in range(iters + 1):
        sw = wi.sum()
        mx, my = (wi[:, None] * X).sum(0) / sw, (wi[:, None] * Y).sum(0) / sw
        xs, ys = X - mx, Y - my
        U, S, Vt = np.linalg.svd((wi[:, None] * ys).T @ xs / sw)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        s = float((S * np.diag(D)).sum() / ((wi * (xs ** 2).sum(1)).sum() / sw))
        t = my - s * R @ mx
        if it < iters:
            r2 = ((s * X @ R.T + t - Y) ** 2).sum(1)
            wi = w * (sigma ** 2 / (sigma ** 2 + r2)) ** 2
    return s, R, t


def fuse(P, w, sigma=SIGMA, iters=3):
    """Robust weighted mean over views: P (V, M, 3), w (V, M) (0 / NaN P: absent) -> Y (M, 3), weight sum (M,)."""
    w = np.where(np.isfinite(P).all(-1), w, 0.0)
    P = np.nan_to_num(P)
    sw = w.sum(0)
    Y = (w[..., None] * P).sum(0) / np.maximum(sw, 1e-12)[:, None]
    for _ in range(iters):
        r2 = ((P - Y) ** 2).sum(-1)
        wr = w * (sigma ** 2 / (sigma ** 2 + r2)) ** 2
        Y = (wr[..., None] * P).sum(0) / np.maximum(wr.sum(0), 1e-12)[:, None]
    Y[sw <= 0] = np.nan
    return Y, sw


def nearest_joint(surface_idx, j14_reg, v_template):
    """Each dense point's nearest of the 14 joints on the SMPL template (its visibility index: 70 + joint)."""
    J = j14_reg @ v_template
    return 70 + np.linalg.norm(v_template[surface_idx][:, None] - J[None], axis=-1).argmin(1)


# ----------------------------------------------------------------------------------------------------------- views


def slim(pp, j14_reg):
    """The per-person outputs the fusion reads (camera frame): kp2d / kp3d (84, .) = the 70 MHR keypoints + the 14
    joints in the dense points' frame, vis (84,) P(visible), surf (468, 3), contact (16,), and the release's own
    single-view SMPL fit (pose, betas, fit_c = the fitted mesh's centroid)."""
    hip = (j14_reg @ pp["mesh_smpl"].astype(np.float64))[[2, 3]].mean(0)
    j14 = pp["joints_h36m14"].astype(np.float64) + hip
    kv = pp.get("keypoint_visibility")
    return {"box": np.asarray(pp["box"], np.float32),
            "kp2d": np.concatenate([pp["keypoints2d"], pp["joints2d_h36m14"]]).astype(np.float32),
            "kp3d": np.concatenate([pp["keypoints3d"], j14]).astype(np.float32),
            "vis": (kv[:, 0] if kv is not None else np.ones(N_KP)).astype(np.float32),
            "surf": pp["dense_surface_cam"].astype(np.float32), "surf2d": pp["dense_surface2d"].astype(np.float32),
            "contact": np.asarray(pp.get("contact", np.full(16, np.nan)), np.float32),
            "smpl_pose": pp["smpl_pose"].astype(np.float32), "smpl_betas": pp["smpl_betas"].astype(np.float32),
            "fit_c": pp["mesh_smpl_fit"].mean(0).astype(np.float32)}


def run_views(fab, views, ids, cache=None, key=""):
    """The released predictor's video mode on each view's matched frames -> people[v][t] (lists of slim dicts). A cache
    file (<cache>/<view>.pkl) is reused when it was written for the same frames and key (the model and path)."""
    out = []
    for j, v in enumerate(views):
        f = Path(cache) / f"{v.name}.pkl" if cache else None
        if f is not None and f.exists():
            c = pickle.loads(f.read_bytes())
            if isinstance(c, dict) and c.get("key") == key and np.array_equal(c.get("ids"), ids[:, j]):
                out.append(c["people"])
                continue
        want = np.unique(ids[ids[:, j] >= 0, j])
        res = {}
        for i, people in zip(want, fab.predict_video(v.images(want), v.K)):
            res[int(i)] = [slim(pp, fab.j14) for pp in people]
        pv = [res.get(int(i), []) if i >= 0 else [] for i in ids[:, j]]
        print(f"[multiview] {v.name}: {len(want)} frames, {sum(map(len, pv))} people", flush=True)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps({"key": key, "ids": ids[:, j], "people": pv}))
        out.append(pv)
    return out


# ----------------------------------------------------------------------------------------------------------- fusion


def _height(o):
    return float(np.ptp(o["kp2d"][:70, 1])) + 1e-6


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def init_subject(views, cands):
    """cands[v]: people of one frame per view -> the chosen index per view (-1: none). The cross-view pair (two views,
    one person each) whose triangulated keypoints reproject within INIT_ERR of the people's heights, the most prominent
    of those; with one view (or no consistent pair) the most prominent person of the first view that has people."""
    Ts, Ks = np.stack([v.T_world_cam for v in views]), np.stack([v.K for v in views])
    best, sel = -1.0, [-1] * len(views)
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            for p, oa in enumerate(cands[a]):
                for q, ob in enumerate(cands[b]):
                    X, ok, _ = triangulate(Ts[[a, b]], Ks[[a, b]], np.stack([oa["kp2d"], ob["kp2d"]]),
                                           np.maximum(np.stack([oa["vis"], ob["vis"]]), W_FLOOR), gap_max=np.inf)
                    if ok.sum() < MIN_TRI:
                        continue
                    e = [np.median(np.linalg.norm(project(Ts[v], Ks[v], X[ok])[0] - o["kp2d"][ok], axis=-1)) / _height(o)
                         for v, o in ((a, oa), (b, ob))]
                    prom = _height(oa) + _height(ob)
                    if max(e) < INIT_ERR and prom > best:
                        best, sel = prom, [-1] * len(views)
                        sel[a], sel[b] = p, q
    if best < 0:
        for v, c in enumerate(cands):
            if c:
                sel[v] = int(np.argmax([_height(o) for o in c]))
                break
    return sel


def follow(view, cand, X):
    """The person of cand nearest the projection of the subject's previous keypoints X (84, 3) world (mean distance over
    its own visible keypoints, within FOLLOW_ERR of its height), or -1."""
    ok = np.isfinite(X).all(1)
    if not cand or ok.sum() < MIN_TRI:
        return -1
    uv, z = project(view.T_world_cam, view.K, np.nan_to_num(X))
    ok &= z > 0
    best, bi = FOLLOW_ERR, -1
    for i, o in enumerate(cand):
        m = ok & (o["vis"] >= VIS_MIN)
        if m.sum() < MIN_TRI:
            m = ok
        e = np.linalg.norm(uv[m] - o["kp2d"][m], axis=-1).mean() / _height(o)
        if e < best:
            best, bi = e, i
    return bi


def fuse_frame(views, obs, align, vis_idx):
    """One frame. obs[v]: the subject's slim dict in view v or None; align[v]: the view's similarity (s, R, t) to use when
    this frame cannot place it (or None). -> dict: kp (84, 3) world keypoints, tri (84,) triangulated, surf (468, 3),
    surf_w (468,), j14 (14, 3), j14_w (14,), contact (16,), align (per view (s, R, t) or None), own (per view: placed
    this frame), n_views."""
    Ts, Ks = np.stack([v.T_world_cam for v in views]), np.stack([v.K for v in views])
    have = [o is not None for o in obs]
    V = len(views)
    uv = np.stack([o["kp2d"] if o is not None else np.full((N_KP, 2), np.nan) for o in obs])
    vis = np.stack([o["vis"] if o is not None else np.zeros(N_KP) for o in obs]).astype(np.float64)
    wk = np.maximum(vis, W_FLOOR) * np.array(have)[:, None]  # every view that sees the person, weighted by P(visible)
    Y, tri, _ = triangulate(Ts, Ks, uv, wk) if sum(have) >= 2 else (np.full((N_KP, 3), np.nan), np.zeros(N_KP, bool), None)
    al, own = list(align), [False] * V
    Pk = np.full((V, N_KP, 3), np.nan)
    Ps = np.full((V, len(vis_idx), 3), np.nan)
    for v, o in enumerate(obs):
        if o is None:
            continue
        kw = to_world(Ts[v], o["kp3d"].astype(np.float64))
        sw = to_world(Ts[v], o["surf"].astype(np.float64))
        m = tri[J14]
        if m.sum() >= MIN_TRI:  # the view's dense frame (its 14 joints) onto the triangulated joints
            al[v], own[v] = similarity(kw[J14][m], Y[J14][m], wk[v, J14][m]), True
        s, R, t = al[v] if al[v] is not None else (1.0, np.eye(3), np.zeros(3))
        Pk[v], Ps[v] = s * kw @ R.T + t, s * sw @ R.T + t
        res = np.where(tri[J14][:, None], Y[J14] - Pk[v][J14], 0.0)  # each point moves with its nearest joint's
        Ps[v] = Ps[v] + res[vis_idx - 70]  # triangulation residual (where that joint is triangulated)
    ws = wk[:, vis_idx]
    kp, _ = fuse(Pk, wk)
    kp[tri] = Y[tri]
    surf, _ = fuse(Ps, ws)
    j14_w = np.where(tri[J14], 1.0, wk[:, J14].max(0))
    con = np.full(16, np.nan)
    con_best = np.full(16, np.nan)
    if any(have):
        C = np.stack([o["contact"] if o is not None else np.zeros(16) for o in obs]).astype(np.float64)
        cw = np.tile(wk[:, CONTACT_KP], 2) * np.isfinite(C)
        con = (cw * np.nan_to_num(C)).sum(0) / np.maximum(cw.sum(0), 1e-12)
        con_best = np.nan_to_num(C)[cw.argmax(0), np.arange(16)]  # the channel in the view that sees its keypoint best
    return {"kp": kp, "tri": tri, "surf": surf, "surf_w": ws.max(0), "j14": kp[J14], "j14_w": j14_w, "contact": con,
            "contact_best": con_best,
            "align": al, "own": own, "n_views": int(sum(have))}


def track_and_fuse(views, people, vis_idx, boxes=None):
    """Subject selection (init_subject, then follow; boxes[v] (T, 4) override) and per-frame fusion, in time order;
    then a second pass where each view's frames without their own placement take the view's nearest-in-time placement.
    -> sel (T, V), frames (list of fuse_frame dicts or None)."""
    T, V = len(people[0]), len(views)
    sel = np.full((T, V), -1, int)
    last, align = None, [None] * V
    for t in range(T):
        cands = [people[v][t] for v in range(V)]
        if last is None:
            s = init_subject(views, cands)
        else:
            s = [follow(views[v], cands[v], last) for v in range(V)]
        for v in range(V):
            if boxes is not None and boxes[v] is not None and t < len(boxes[v]) and np.isfinite(boxes[v][t]).all() and cands[v]:
                ious = [_iou(o["box"], boxes[v][t]) for o in cands[v]]
                s[v] = int(np.argmax(ious)) if max(ious) > 0.3 else -1
        sel[t] = s
        if any(i >= 0 for i in s):
            f = fuse_frame(views, [cands[v][s[v]] if s[v] >= 0 else None for v in range(V)], align, vis_idx)
            align = f["align"]
            last = f["kp"]
        else:  # lost in every view: the first rule again
            last = None
    own = {v: {} for v in range(V)}  # second pass: the placements each view made itself
    frames = []
    for t in range(T):
        obs = [people[v][t][sel[t, v]] if sel[t, v] >= 0 else None for v in range(V)]
        f = fuse_frame(views, obs, [None] * V, vis_idx) if any(o is not None for o in obs) else None
        frames.append(f)
        if f is not None:
            for v in range(V):
                if f["own"][v]:
                    own[v][t] = f["align"][v]
    for t, f in enumerate(frames):
        if f is None:
            continue
        miss = [v for v in range(V) if sel[t, v] >= 0 and not f["own"][v] and own[v]]
        if miss:
            al = [None] * V
            for v in miss:
                k = min(own[v], key=lambda u: abs(u - t))
                al[v] = own[v][k]
            obs = [people[v][t][sel[t, v]] if sel[t, v] >= 0 else None for v in range(V)]
            frames[t] = fuse_frame(views, obs, al, vis_idx)
    return sel, frames


# ----------------------------------------------------------------------------------------------------------- fit


def contact_parts(j14_reg, body):
    """[(contact channels, joint index in the 14, rule)] for the support parts: knees and hands are held at the floor when
    in contact ("hold"), feet are only kept from going below it ("lift": shoes add their soles). The joints are the ones
    nearest the SMPL knees, wrists and ankles on the template; channels l_knee, r_knee, l_hand, r_hand, (feet: none)."""
    tmpl = body.v_template.cpu().numpy().astype(np.float64)
    J = body.J_regressor.cpu().numpy().astype(np.float64) @ tmpl
    J14t = j14_reg @ tmpl
    jidx = {k: int(np.linalg.norm(J14t - J[k], axis=1).argmin()) for k in (4, 5, 7, 8, 20, 21)}
    return [((6,), jidx[4], "hold"), ((7,), jidx[5], "hold"), ((4,), jidx[20], "hold"), ((5,), jidx[21], "hold"),
            ((), jidx[7], "lift"), ((), jidx[8], "lift")]


def floor_contacts(frames, valid, surf, j14, vis_idx, parts, floor_z, cap=CONTACT_CAP):
    """The rig's floor on the fused points, per support part (contact_parts): a part's fused dense points are those nearest
    its joint on the template. "hold" (knees, hands): when the part's static contact is >= 0.5 in the view that sees its
    keypoint best, its points and joint move vertically so that their lowest point is on the floor (when within cap, m).
    "lift" (feet): only a lowest point below the floor is raised onto it (within cap); a foot is never pulled down. The
    fit then treats these points as any other (a soft constraint). -> surf, j14, the number of moves."""
    surf, j14 = surf.copy(), j14.copy()
    n = 0
    for t, f in enumerate(frames):
        if f is None or not valid[t]:
            continue
        for ch, j, rule in parts:
            if rule == "hold" and max(f["contact_best"][list(ch)]) < 0.5:
                continue
            m = vis_idx - 70 == j
            dz = floor_z - surf[t, m, 2].min()
            if abs(dz) <= cap and (rule == "hold" or dz > 0):
                surf[t, m, 2] += dz
                j14[t, j, 2] += dz
                n += 1
    return surf, j14, n


def smpl_model(device):
    from bench.body_models import SMPL
    from fab import paths

    return SMPL(paths.smpl_dir() / "SMPL_NEUTRAL.pkl").to(device).eval()


def _gn(gn, R, betas, X, conf, R0, b0, iters=2, robust=50.0, anchor=1e-3):
    """The release's Gauss-Newton schedule (FABPredictor._gn_refine): Geman-McClure IRLS weights (robust mm) from the
    previous iterate times the per-point confidence, a prior toward (R0, b0)."""
    import torch

    w = conf.clone()
    s2 = robust ** 2
    t3 = None
    for it in range(iters):
        R, betas, t3, _ = gn(R, betas, X, w, R0=R0, b0=b0, w_anchor=anchor, w_anchor_b=anchor)
        if it + 1 < iters:
            vs, _ = gn.smpl(R, betas, full=False)
            P = torch.cat([vs[:, gn.vpos], torch.einsum("ns,bsc->bnc", gn.reg, vs)], 1)
            w = (s2 / (s2 + ((P + t3[:, None] - X) * 1000).pow(2).sum(-1))) ** 2 * conf
    return R, betas, t3


def fit_sequence(surf, surf_w, j14, j14_w, valid, R_fit, fitter_path, surface_idx, j14_reg, device="cuda", body=None):
    """One SMPL per valid frame from the fused points (world): surf (T, 468, 3), j14 (T, 14, 3) and their confidences.
    The fit runs in the rotated frame R_fit (world -> fit, e.g. a camera's orientation, the frame the learned fitter
    knows). -> R (T, 24, 3, 3) joint rotations (global: world), betas (10,), verts (T, 6890, 3) world (NaN-free rows
    only where valid)."""
    import torch

    from fab.gn_fit import FastGN
    from fab.smpl_fitter import load_fitter

    body = body if body is not None else smpl_model(device)
    fnet = load_fitter(fitter_path, device)
    sidx = torch.as_tensor(np.asarray(surface_idx), device=device)
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    Rf = tt(R_fit)
    idx = np.flatnonzero(valid)
    X = tt(np.concatenate([surf[idx], j14[idx]], 1)) @ Rf.T
    conf = tt(np.concatenate([surf_w[idx], j14_w[idx]], 1))
    gn1 = FastGN(body, sidx, j14_reg, iters=1, lam=1e-3)
    gn2 = FastGN(body, sidx, j14_reg, iters=1, lam=1e-3, w_beta_step=1e8)  # the clip's shape held
    Rs, bs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), FIT_BATCH):
            Xi, ci = X[i: i + FIT_BATCH], conf[i: i + FIT_BATCH]
            R0, b0 = fnet(Xi[:, : len(surface_idx)], body, sidx)
            R1, b1, _ = _gn(gn1, R0.float(), b0.float(), Xi, ci, R0.float(), b0.float())
            Rs.append(R1)
            bs.append(b1)
        R1, b1 = torch.cat(Rs), torch.cat(bs)
        betas = b1.median(0).values
        out_R, out_v = [], []
        for i in range(0, len(idx), FIT_BATCH):
            Xi, ci, Ri = X[i: i + FIT_BATCH], conf[i: i + FIT_BATCH], R1[i: i + FIT_BATCH]
            bi = betas[None].expand(len(Ri), -1).contiguous()
            R2, b2, t3 = _gn(gn2, Ri, bi, Xi, ci, Ri, bi)
            v, _ = body(torch.zeros(len(R2), 3, device=device), torch.zeros(len(R2), 69, device=device), b2, None, rotmats=R2)
            out_R.append(R2)
            out_v.append(v + t3[:, None])
    Rall = torch.cat(out_R).cpu().numpy().astype(np.float64)
    Vall = torch.cat(out_v).cpu().numpy().astype(np.float64)
    T = len(valid)
    R = np.tile(np.eye(3), (T, 24, 1, 1))
    V = np.zeros((T, 6890, 3))
    Rfn = np.asarray(R_fit, np.float64)
    R[idx] = Rall
    R[idx, 0] = Rfn.T @ Rall[:, 0]
    V[idx] = Vall @ Rfn
    return R, betas.cpu().numpy().astype(np.float32), V


def world_body(R, verts, betas, valid, fps, body, sigma_rot=2.0, sigma_pos=3.0):
    """The release's world smoothing (bench/videomimic/recon.py stage_world): frames without a body interpolated,
    joint rotations (global and body apart) and the pelvis Gaussian-smoothed (zero phase), the body re-posed with the
    clip's betas. The sigmas are the release's frames at 30 fps, i.e. sigma x fps / 30 frames at another rate (the same
    time span: 67 / 100 ms). -> pose (T, 72) axis-angle, trans (T, 3), joints (T, 24, 3), verts (T, 6890, 3)."""
    import torch
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    from bench.videomimic.recon import fill_invalid, smooth_quats

    T = len(R)
    dev = body.v_template.device
    Jreg = body.J_regressor.cpu().numpy().astype(np.float64)
    pel = np.einsum("v,nvc->nc", Jreg[0], verts)
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(T, 24, 4)
    q = fill_invalid(smooth_quats(q, 0), valid)
    if fps != 30.0:  # the same time span at another frame rate
        sigma_rot, sigma_pos = sigma_rot * fps / 30.0, sigma_pos * fps / 30.0
    q = np.concatenate([smooth_quats(q[:, :1], sigma_rot), smooth_quats(q[:, 1:], sigma_rot)], 1)
    pel = gaussian_filter1d(fill_invalid(pel, valid), sigma_pos, axis=0, mode="nearest")
    Rs = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(T, 24, 3, 3)
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    V, J = [], []
    with torch.no_grad():
        for i in range(0, T, 256):
            n = len(Rs[i: i + 256])
            v, j = body(torch.zeros(n, 3, device=dev), torch.zeros(n, 69, device=dev), tt(betas)[None].expand(n, -1), None,
                        rotmats=tt(Rs[i: i + 256]))
            V.append(v.cpu().numpy())
            J.append(j.cpu().numpy())
    V, J = np.concatenate(V).astype(np.float64), np.concatenate(J).astype(np.float64)
    V = V - J[:, :1] + pel[:, None]
    Jw = np.einsum("jv,nvc->njc", Jreg, V)
    with torch.no_grad():
        _, j0 = body(torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev), tt(betas)[None], None)
    trans = Jw[:, 0] - j0[0, 0].cpu().numpy()
    pose = Rotation.from_matrix(Rs.reshape(-1, 3, 3)).as_rotvec().reshape(T, 72)
    return pose, trans, Jw, V


def single_view_body(people_v, sel_v, view, body, device="cuda"):
    """The release's own single-view result in the rig world (the baseline): each frame's FAB SMPL fit (camera frame,
    placed at its fitted mesh's centroid) mapped by the view's T_world_cam. -> R (T, 24, 3, 3) world, betas (median,
    10), verts (T, 6890, 3), valid (T,)."""
    import torch
    from scipy.spatial.transform import Rotation

    T = len(sel_v)
    valid = sel_v >= 0
    obs = [people_v[t][sel_v[t]] for t in range(T) if valid[t]]
    pose = np.stack([o["smpl_pose"] for o in obs]).astype(np.float64)
    betas_f = np.stack([o["smpl_betas"] for o in obs])
    fit_c = np.stack([o["fit_c"] for o in obs]).astype(np.float64)
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    with torch.no_grad():
        v = torch.cat([body(tt(pose[i: i + 256, :3]), tt(pose[i: i + 256, 3:]), tt(betas_f[i: i + 256]), None)[0]
                       for i in range(0, len(pose), 256)]).cpu().numpy().astype(np.float64)
    v = v - v.mean(1, keepdims=True) + fit_c[:, None]
    Rc = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    Rc[:, 0] = view.T_world_cam[:3, :3] @ Rc[:, 0]
    R = np.tile(np.eye(3), (T, 24, 1, 1))
    V = np.zeros((T, 6890, 3))
    R[valid], V[valid] = Rc, to_world(view.T_world_cam, v)
    return R, np.median(betas_f, 0).astype(np.float32), V, valid


# ----------------------------------------------------------------------------------------------------------- export


def flat_floor(xy, floor_z, res=0.04, margin=2.0):
    lo, hi = xy.min(0) - margin, xy.max(0) + margin
    nx, ny = int(np.ceil((hi[0] - lo[0]) / res)) + 1, int(np.ceil((hi[1] - lo[1]) / res)) + 1
    return {"x0": float(lo[0]), "y0": float(lo[1]), "res": float(res), "height": np.full((ny, nx), floor_z, np.float32),
            "valid": np.ones((ny, nx), bool)}


def export(out, name, views, fps, t, body, pose, betas, trans, joints, verts, valid, floor_z, contact, surf, surface_idx,
           notes):
    """fab-world's layout: <out>/<name>.npz (the export), <out>/work/joint/recon/<name>.npz (+ verts) with _hfield.npz /
    _scene.ply (a flat floor at floor_z), <out>/work/ours/fab/<name>_raw.npz (the fused dense points, fab-retarget's points
    mode) and <out>/work/contacts/<name>__0.contact.npz (the fused image contact probabilities, --ground_aware)."""
    import trimesh

    from fab.retarget.contact import human_contact_labels

    out = Path(out)
    rd, fd, cd = out / "work" / "joint" / "recon", out / "work" / "ours" / "fab", out / "work" / "contacts"
    for d in (rd, fd, cd):
        d.mkdir(parents=True, exist_ok=True)
    T = len(pose)
    frames = np.arange(T)
    part = body.weights.argmax(1).cpu().numpy()
    Vrel = verts.copy()
    Vrel[..., 2] -= floor_z
    labels = human_contact_labels(Vrel, part, fps, 0.04, 0.5)
    contacts = json.dumps({g: v.astype(int).tolist() for g, v in labels.items()})
    T0 = np.repeat(views[0].T_world_cam[None], T, 0)
    common = dict(frames=frames, fps=float(fps), valid=valid, pose=pose.astype(np.float32), betas=betas.astype(np.float32),
                  trans=trans.astype(np.float32), joints=joints.astype(np.float32), contacts=contacts, T_world_cam=T0,
                  K=views[0].K, scale_fix=1.0, joint=json.dumps({"skipped": "multi-view: metric rig"}),
                  T_world_cams=np.stack([v.T_world_cam for v in views]), K_cams=np.stack([v.K for v in views]),
                  camera_names=np.array([v.name for v in views]), timestamps=t, floor_z=float(floor_z))
    hf = flat_floor(joints[:, 0, :2], floor_z)
    np.savez(rd / f"{name}_hfield.npz", **hf)
    x1, y1 = hf["x0"] + hf["res"] * (hf["height"].shape[1] - 1), hf["y0"] + hf["res"] * (hf["height"].shape[0] - 1)
    trimesh.Trimesh([[hf["x0"], hf["y0"], floor_z], [x1, hf["y0"], floor_z], [x1, y1, floor_z], [hf["x0"], y1, floor_z]],
                    [[0, 1, 2], [0, 2, 3]], process=False).export(rd / f"{name}_scene.ply")
    np.savez(rd / f"{name}.npz", **common, verts=verts.astype(np.float16), up_world=np.array([0.0, 0, 1]), R_align=np.eye(3),
             origin=np.zeros(3), z0=0.0, depth_factor=np.zeros(T), depth_scale=1.0)
    np.savez(fd / f"{name}_raw.npz", frames=frames, valid=valid, pts=surf.astype(np.float32), surface_idx=np.asarray(surface_idx),
             fps=float(fps))
    c = np.nan_to_num(contact).astype(np.float32)  # the image head's channels in fab-world's contact-file layout
    np.savez(cd / f"{name}__0.contact.npz", frame=frames, time=frames / float(fps), fps=np.float32(fps), thr=np.float32(0.5),
             channels=np.array(["l_heel", "l_toe", "r_heel", "r_toe"]), hk_channels=np.array(["l_hand", "r_hand", "l_knee", "r_knee"]),
             image_static_prob=c[:, 0:4], image_any_prob=c[:, 8:12], image_hk_static_prob=c[:, 4:8], image_hk_any_prob=c[:, 12:16],
             image_static=c[:, 0:4] >= 0.5, image_any=c[:, 8:12] >= 0.5,
             source="fab-multiview: the image contact head, fused over the views")
    np.savez(out / f"{name}.npz", **common, gravity_world=np.array([0.0, 0.0, -1.0]), notes=json.dumps(notes))
    return out / f"{name}.npz"


# ----------------------------------------------------------------------------------------------------------- main


def run(rig, out, name=None, views_only="", subject_boxes="", model="fab-l", fast=False, device="cuda", trackers=""):
    from fab.predictor import FABPredictor

    views, info = load_rig(rig)
    if views_only:
        keep = views_only.split(",")
        views = [v for v in views if v.name in keep]
        assert views, f"--views {views_only}: none of these cameras in {rig}"
    name = name or (Path(rig).name if Path(rig).is_dir() else Path(rig).stem)
    fps = info["fps"]
    ids, t = sync(views, fps)
    out = Path(out)
    work = out / "work"
    fab = FABPredictor(model, fast=fast, device=device)
    people = run_views(fab, views, ids, cache=work / "views", key=f"{model} fast={fast}")
    boxes = None
    if subject_boxes:
        boxes = [np.load(Path(subject_boxes) / f"{v.name}.npz")["box"].astype(np.float64)
                 if (Path(subject_boxes) / f"{v.name}.npz").exists() else None for v in views]
    body = fab.body if fab.body is not None else smpl_model(device)
    sidx = fab.model.dense_head.surface_idx.cpu().numpy()
    vis_idx = nearest_joint(sidx, fab.j14, body.v_template.cpu().numpy().astype(np.float64))
    sel, frames = track_and_fuse(views, people, vis_idx, boxes)
    T = len(frames)
    valid = np.array([f is not None and np.isfinite(f["surf"]).all() for f in frames])
    if not valid.any():
        raise SystemExit("[multiview] the subject was not found in any frame")
    nan = lambda *s: np.full(s, np.nan)  # noqa: E731
    surf = np.stack([f["surf"] if v else nan(len(sidx), 3) for f, v in zip(frames, valid)])
    surf_w = np.stack([f["surf_w"] if v else np.zeros(len(sidx)) for f, v in zip(frames, valid)])
    j14 = np.stack([np.nan_to_num(f["j14"]) if v else nan(14, 3) for f, v in zip(frames, valid)])
    j14_w = np.stack([np.where(np.isfinite(f["j14"]).all(1), f["j14_w"], 0.0) if v else np.zeros(14) for f, v in zip(frames, valid)])
    contact = np.stack([f["contact"] if f is not None else nan(16) for f in frames])
    n_floor = 0
    tnotes = None
    assert not (trackers and len(views) == 1), "--trackers needs two or more cameras (the fused path)"
    if len(views) == 1:  # one camera: the release's own single-view result (its SMPL fit, mapped by T_world_cam)
        R, betas, V, valid = single_view_body(people[0], sel[:, 0], views[0], body, device)
        method = "one camera: the release's own per-frame SMPL fit mapped by T_world_cam, the release's world smoothing"
    else:
        if info["floor_z"] is not None:  # the rig's floor: knees / hands in contact held on it, feet kept above it
            surf, j14, n_floor = floor_contacts(frames, valid, surf, j14, vis_idx, contact_parts(fab.j14, body),
                                                float(info["floor_z"]))
        if trackers:  # optional: world-space trackers fused with the views by a smoother (fab/multiview_trackers.py)
            from fab import multiview_trackers

            surf, surf_w, j14, j14_w, tnotes = multiview_trackers.smooth(trackers, views, info, t, people, sel, frames,
                                                                         valid, surf_w, j14_w, vis_idx, sidx, fab.j14, body)
        R_fit = np.linalg.inv(views[0].T_world_cam)[:3, :3]
        R, betas, V = fit_sequence(surf, surf_w, j14, j14_w, valid, R_fit, fab.smpl_fitter_path, sidx, fab.j14, device, body)
        method = ("the released FAB per view, keypoints triangulated, dense points fused with the visibility head's "
                  "P(visible), one SMPL fit per frame, the release's world smoothing")
    pose, trans, joints, verts = world_body(R, V, betas, valid, fps, body)
    floor_z = info["floor_z"]
    if floor_z is None:  # the 10th percentile of the body's lowest point
        floor_z = float(np.percentile(verts[valid][:, :, 2].min(1), 10))
    from bench.videomimic.recon import fill_invalid

    surf_f = fill_invalid(np.nan_to_num(surf), valid)
    stats = {"frames": T, "valid": float(valid.mean()), "views_per_frame": float(np.mean([f["n_views"] if f else 0 for f in frames])),
             "triangulated_joints": float(np.mean([f["tri"][J14].mean() for f in frames if f is not None])),
             "subject_per_view": {v.name: float((sel[:, j] >= 0).mean()) for j, v in enumerate(views)},
             "floor_contacts": n_floor}
    notes = {"pipeline": f"fab-multiview (docs/multiview.md): {method}",
             "rig": info["source"], "cameras": [v.name for v in views], "model": model, "fast": fast,
             "fitter": fab.smpl_fitter_path, "floor_z": floor_z, "floor_source": "rig" if info["floor_z"] is not None else "body",
             "world": "the rig's world turned so that up is +z; T_world_cam: the first camera (all: T_world_cams)", **stats,
             **({"trackers": tnotes} if tnotes is not None else {})}
    f = export(out, name, views, fps, t, body, pose, betas, trans, joints, verts, valid, floor_z, contact, surf_f, sidx, notes)
    np.savez(work / f"{name}_fusion.npz", sel=sel, kp=np.stack([f_["kp"] if f_ else nan(N_KP, 3) for f_ in frames]),
             tri=np.stack([f_["tri"] if f_ else np.zeros(N_KP, bool) for f_ in frames]), surf=surf, surf_w=surf_w,
             own=np.stack([f_["own"] if f_ else [False] * len(views) for f_ in frames]),
             scale=np.stack([[a[0] if a is not None else np.nan for a in f_["align"]] if f_ else [np.nan] * len(views)
                             for f_ in frames]), ids=ids)
    print(f"[multiview] {name}: {T} frames ({stats['valid'] * 100:.0f} % with the subject), {stats['views_per_frame']:.2f} views "
          f"per frame, {stats['triangulated_joints'] * 100:.0f} % of joints triangulated -> {f}", flush=True)
    return f


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("rig", help="rig.json, or a take folder with world.json (docs/multiview.md)")
    p.add_argument("--out", default="output/fab_multiview")
    p.add_argument("--name", default="", help="clip name (default: the rig file's or folder's name)")
    p.add_argument("--views", default="", help="comma-separated camera names to use (default: all)")
    p.add_argument("--subject_boxes", default="", help="a folder of <camera name>.npz with box (T, 4) xyxy per synced frame "
                   "(NaN: unknown): the subject in that view (identity only)")
    p.add_argument("--model", default="fab-l", help="fab-l (default) or fab-s (the dense head is needed)")
    p.add_argument("--fast", action="store_true", help="the compiled fast path (fast=True); default eager")
    p.add_argument("--weights", default="", help="weights folder (default $FAB_WEIGHTS or $FAB_HOME/weights)")
    p.add_argument("--trackers", default="", help="optional world-space trackers, <body point>=<file>[,...] (e.g. "
                   "left_wrist=l.npz,right_wrist=r.csv), fused with the views (docs/multiview.md, Optional: tracker input)")
    a = p.parse_args(argv)
    if a.weights:
        os.environ["FAB_WEIGHTS"] = a.weights
    run(a.rig, a.out, a.name or None, a.views, a.subject_boxes, a.model, a.fast, trackers=a.trackers)


if __name__ == "__main__":
    main()
