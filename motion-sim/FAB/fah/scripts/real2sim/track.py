"""Object 6D track + metric scale for the reconstructed mesh, and the FAH hand in the same metric camera frame
(real-to-sim clip after depth.py / hands.py / segment.py / reconstruct.py) -> <clip>/track.npz + recon/mesh_metric.glb.

Model-based, render-free: the mesh's surface points are fitted to each frame's observation with three terms: observed
object points (object mask from segment.py minus the hand mask, back-projected with MoGe-2 depth) -> nearest model point
(3D, truncated; init only by default, --w3d), projected model points -> distance transform of the complement of (object
mask U dilated hand mask) (silhouette; the hand may occlude), object-mask pixels -> nearest projected model point
(coverage). Init on the reconstruction frame: batched search over --n_rot rotations (scale and translation free), then
frame to frame from the previous pose (constant-velocity prior; static camera: the object stays put until a hand mask
touches it), a multi-frame per-axis scale refinement, and the object's metric scale from the hand (hand_metric_scale).
--grasp_hold 1: a second tracking pass through hand occlusion: while a hand holds the object (Carry), the prediction is the
pose carried by the hand's palm motion, and the refit on the masks replaces it only when the masks support it (re-acquired).
The manipulating hand (FAH, the side that comes closest to the object) keeps FAH's own metric camera-frame placement
(--hand_depth moge: moved onto the MoGe-2 depth inside its detector mask; worse on DexYCB, MoGe-2 puts the hand ~5 cm too
close). Support plane (gravity) = RANSAC plane of the depth around the object on the reconstruction frame.

    python scripts/real2sim/track.py --clip $R2S/mustard --recon recon_s3do --init layout --grasp_hold 1
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bench.seg.rle import decode  # noqa: E402

FAH_HOME = os.environ.get("FAH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "fah"))

dev = "cuda"


def rodrigues(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    z = torch.zeros_like(k[..., 0])
    Kx = torch.stack([z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z], -1).reshape(*k.shape[:-1], 3, 3)
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def rotation_grid(n_views, n_inplane):
    i = np.arange(n_views) + 0.5
    phi, th = np.arccos(1 - 2 * i / n_views), np.pi * (1 + 5 ** 0.5) * i
    out = []
    for z in np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1):
        x = np.cross(z, [0, 0, 1] if abs(z[2]) < 0.9 else [1, 0, 0])
        x /= np.linalg.norm(x)
        R0 = np.stack([x, np.cross(z, x), z], 1)
        for a in np.arange(n_inplane) * 2 * np.pi / n_inplane:
            c, s = np.cos(a), np.sin(a)
            out.append(R0 @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    return np.stack(out)


def backproject(K, uv, z):
    return np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)


class Frame:
    """Per-frame observation tensors: observed object points, coverage pixels, silhouette distance transform."""

    def __init__(self, K, depth, obj, hand, n_obs=1500, n_cov=800, rng=None):
        H, W = obj.shape
        rng = rng or np.random.default_rng(0)
        hand_d = cv2.dilate(hand.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        allowed = obj | hand_d
        self.dt = torch.from_numpy(cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 5)).to(dev)
        core = cv2.erode(obj.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~hand_d & (depth > 0)
        ys, xs = np.nonzero(core)
        self.ok = len(xs) >= 30
        if self.ok:
            z = depth[ys, xs]
            med, mad = np.median(z), np.median(np.abs(z - np.median(z))) + 1e-3
            keep = np.abs(z - med) < 4 * mad + 0.02
            ys, xs, z = ys[keep], xs[keep], z[keep]
            sel = rng.choice(len(xs), min(n_obs, len(xs)), replace=False)
            self.Q = torch.from_numpy(backproject(K, np.stack([xs[sel], ys[sel]], 1).astype(np.float64), z[sel].astype(np.float64))).float().to(dev)
        ys, xs = np.nonzero(obj & ~hand_d)
        self.ok &= len(xs) >= 30
        if self.ok:
            sel = rng.choice(len(xs), min(n_cov, len(xs)), replace=False)
            self.U = torch.from_numpy(np.stack([xs[sel], ys[sel]], 1)).float().to(dev)
        self.K = torch.tensor(K, dtype=torch.float32, device=dev)
        self.HW = (H, W)
        self.area = int(obj.sum())


def project(K, X):
    z = X[..., 2:].clamp_min(1e-3)
    return torch.cat([X[..., :1] / z * K[0, 0] + K[0, 2], X[..., 1:2] / z * K[1, 1] + K[1, 2]], -1)


def sample_dt(fr, uv):
    H, W = fr.HW
    g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1)
    v = torch.nn.functional.grid_sample(fr.dt[None, None].expand(len(uv), 1, H, W), g[:, :, None], align_corners=True,
                                        padding_mode="border")[:, 0, :, 0]
    out = (uv[..., 0] < 0) | (uv[..., 0] > W - 1) | (uv[..., 1] < 0) | (uv[..., 1] > H - 1)
    return torch.where(out, torch.zeros_like(v), v)  # out of view: unobserved, no penalty


W3D = 1.0


def losses(fr, P, R, t, s, trunc=0.03):
    X = s[:, None, None] * P[None] @ R.transpose(1, 2) + t[:, None]  # (B, N, 3)
    uv = project(fr.K, X)
    l3 = torch.cdist(fr.Q[None].expand(len(X), -1, -1), X).min(2).values.clamp(max=trunc).mean(1)
    ls = sample_dt(fr, uv).mean(1)
    lc = torch.cdist(fr.U[None].expand(len(X), -1, -1), uv).min(2).values.clamp(max=50).mean(1)
    behind = torch.relu(0.05 - X[..., 2]).mean(1) * 1e3
    return W3D * l3 / 0.01 + ls / 5 + lc / 5 + behind, (l3, ls, lc)


def fit(fr, P, R0, t0, s0, steps, lr_r=0.02, lr_t=0.005, lr_s=0.01, free_scale=False, prior=None):
    B = len(R0)
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    dt = torch.zeros(B, 3, device=dev, requires_grad=True)
    ls_ = torch.zeros(B, device=dev, requires_grad=True)
    groups = [dict(params=[w], lr=lr_r), dict(params=[dt], lr=lr_t)] + ([dict(params=[ls_], lr=lr_s)] if free_scale else [])
    opt = torch.optim.Adam(groups)
    for _ in range(steps):
        R, t, s = rodrigues(w) @ R0, t0 + dt, s0 * torch.exp(ls_)
        L, _ = losses(fr, P, R, t, s)
        if prior is not None:
            L = L + prior * ((dt / 0.01) ** 2).sum(1) + prior * ((w / 0.1) ** 2).sum(1)
        opt.zero_grad()
        L.sum().backward()
        opt.step()
    with torch.no_grad():
        R, t, s = rodrigues(w) @ R0, t0 + dt, s0 * torch.exp(ls_)
        L, parts = losses(fr, P, R, t, s)
    return R.detach(), t.detach(), s.detach(), L, parts


def silhouette_iou(K, X, obj, hand):
    uv = np.round(project(torch.tensor(K, dtype=torch.float32), torch.from_numpy(X).float()).numpy()).astype(int)
    H, W = obj.shape
    m = np.zeros((H, W), np.uint8)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    m[uv[ok, 1], uv[ok, 0]] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    vis = ~hand
    return float((m & obj & vis).sum() / max(((m | obj) & vis).sum(), 1))


def ransac_plane(X, iters=500, thr=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    best, bn = None, -1
    for _ in range(iters):
        a, b, c = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = (np.abs((X - a) @ n) < thr).sum()
        if k > bn:
            best, bn = (n, -n @ a), k
    n, d = best
    inl = np.abs(X @ n + d) < thr
    c = X[inl].mean(0)
    n = np.linalg.svd(X[inl] - c)[2][-1]
    return (n, -n @ c) if n @ c < 0 else (-n, n @ c), inl.mean()  # normal towards the camera (up for a table)


def align_depth(D, obj, hands, static, f0, ring_px=61):
    """Static camera: per-frame scale of the MoGe-2 depth to frame f0, measured on the support surface around the object's
    start position (a ring around its f0 mask, minus the dilated object / hand masks of both frames). MoGe-2 warps the
    depth locally when the arm enters (DexYCB: table next to the bottle +6 cm while the global scale stays put), so the
    reference is the static neighbourhood of the object rather than the whole background."""
    if not static:
        return D.astype(np.float32), np.ones(len(D), np.float32)
    D = D.astype(np.float32)
    k = lambda m, r: cv2.dilate(m.astype(np.uint8), np.ones((r, r), np.uint8)).astype(bool)
    ring = k(obj[f0], ring_px) & ~k(obj[f0] | hands[f0], 9) & (D[f0] > 0)
    s = np.ones(len(D), np.float32)
    for i in range(len(D)):
        ok = ring & ~k(obj[i] | hands[i], 15) & (D[i] > 0)
        if ok.sum() > 200:
            r = D[f0][ok] / D[i][ok]
            s[i] = np.median(r[np.abs(r - np.median(r)) < 0.05])
    s = gaussian_filter1d(median_filter(s, 5, mode="nearest"), 1.0, mode="nearest")
    return D * s[:, None, None], s


def first_touch(obj, hands, px=7, min_px=20):
    """First frame where a (dilated) FAH hand mask touches the object mask in the image (depth-free)."""
    for i, (o, h) in enumerate(zip(obj, hands)):
        if (cv2.dilate(h.astype(np.uint8), np.ones((px, px), np.uint8)).astype(bool) & o).sum() >= min_px:
            return i
    return len(obj)


def hand_metric_scale(T, mesh, V, valid, labels, ks=np.linspace(0.7, 1.45, 151), near=0.03, lead=6, T_move=None):
    """Monocular depth fixes the object only up to a scale about the camera centre (size and distance together; the
    silhouettes do not change). FAH's hand is metric (DexYCB root depth within ~1-2 cm), so pick the factor k at which
    the hand, while it manipulates the object, touches it: per MANO part the closest signed distance (point-to-plane
    to dense surface samples), cost 3 |sd| inside, min(sd, near) outside, mean over parts and frames."""
    from scipy.spatial import cKDTree

    pts, fi = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    nrm = mesh.face_normals[fi]
    tree = cKDTree(pts)
    Tm = T if T_move is None else T_move  # moving camera: the object's motion in the world
    move = np.linalg.norm(Tm[:, :3, 3] - Tm[0, :3, 3], axis=1) > 0.03 * np.linalg.norm(T[0, :3, 3])
    if not move.any():
        return 1.0, None
    win = np.zeros(len(T), bool)
    win[max(int(np.argmax(move)) - lead, 0):] = True
    win &= valid
    cost = []
    for k in ks:
        c = []
        for i in np.nonzero(win)[0]:
            q = (V[i] / k - T[i, :3, 3]) @ T[i, :3, :3]  # base object frame
            _, j = tree.query(q)
            sd = k * np.einsum("ij,ij->i", q - pts[j], nrm[j])
            part = np.array([sd[labels == p].min() for p in range(16)])
            c.append(np.where(part < 0, -3 * part, np.minimum(part, near)).mean())
        cost.append(np.mean(c))
    cost = np.array(cost)
    return float(ks[int(np.argmin(cost))]), dict(k=ks.tolist(), cost=cost.tolist(), frames=int(win.sum()))


def hand_depth_axis(T, mesh, V, valid, labels, f0, scales=np.linspace(0.8, 2.0, 25), shifts=np.linspace(-0.03, 0.03, 13),
                    near=0.03, lead=6):
    """Single-view reconstructions guess the extent along the init view's depth: fit the object-frame axis closest to
    the f0 viewing direction (scale about the centre + shift along it) to the manipulating hand's grasp, same contact cost
    as hand_metric_scale. Returns (axis, scale, shift)."""
    from scipy.spatial import cKDTree

    a = int(np.argmax(np.abs(T[f0][:3, :3].T @ np.array([0, 0, 1.0]))))
    pts, fi = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    nrm = mesh.face_normals[fi]
    tree = cKDTree(pts)
    move = np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1) > 0.03 * np.linalg.norm(T[0, :3, 3])
    if not move.any():
        return a, 1.0, 0.0
    win = np.zeros(len(T), bool)
    win[max(int(np.argmax(move)) - lead, 0):] = True
    win &= valid
    Q = [(V[i] - T[i, :3, 3]) @ T[i, :3, :3] for i in np.nonzero(win)[0]]  # hand in the object frame
    best = (np.inf, 1.0, 0.0)
    for sc in scales:
        d = np.ones(3)
        d[a] = sc
        for u in shifts:
            c = []
            for q in Q:
                q2 = q.copy()
                q2[:, a] -= u
                _, j = tree.query(q2 / d)
                n2 = nrm[j] / d
                n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
                sd = np.einsum("ij,ij->i", q2 - pts[j] * d, n2)
                part = np.array([sd[labels == p].min() for p in range(16)])
                c.append(np.where(part < 0, -3 * part, np.minimum(part, near)).mean())
            if np.mean(c) < best[0]:
                best = (np.mean(c), float(sc), float(u))
    return a, best[1], best[2]


def hand_masks(frames_h, shape):
    return np.stack([np.any([decode(h["mask_rle"]) for h in f if "mask_rle" in h], 0) if any("mask_rle" in h for h in f)
                     else np.zeros(shape, bool) for f in frames_h])


def pick_side(frames_h, T, Pm, touch=0.05):
    """The manipulating hand's side: the one that is within `touch` of the object in the most frames (contact frames; a few
    spurious detections of the other hand near the object do not win), ties by the 10th-percentile distance over the clip."""
    dist = {True: [], False: []}
    for i, f in enumerate(frames_h):
        X = Pm[::8] @ T[i][:3, :3].T + T[i][:3, 3]
        best = {}
        for h in f:
            if h.get("verts3d") is not None:
                d = np.min(np.linalg.norm(np.asarray(h["verts3d"])[::8, None] - X[None], axis=-1))
                best[bool(h["is_right"])] = min(best.get(bool(h["is_right"]), 9.0), d)
        for k, d in best.items():
            dist[k].append(d)
    near = {k: sum(d < touch for d in v) for k, v in dist.items()}
    return max((k for k in dist if dist[k]), key=lambda k: (near[k], -np.percentile(dist[k], 10)))


def pick_side_2d(frames_h, obj, px=7, min_px=20):
    """Depth-free choice of the manipulating hand: the side whose (dilated) detector mask touches the object mask in the most
    frames; None when no hand ever touches it (then pick_side, in 3D)."""
    count = {True: 0, False: 0}
    for f, o in zip(frames_h, obj):
        for side in (True, False):
            ms = [decode(h["mask_rle"]) for h in f if bool(h["is_right"]) == side and "mask_rle" in h]
            if ms and (cv2.dilate(np.any(ms, 0).astype(np.uint8), np.ones((px, px), np.uint8)).astype(bool) & o).sum() >= min_px:
                count[side] += 1
    return max(count, key=count.get) if max(count.values()) else None


def place_hands(frames_h, D, K, faces, side, obj_c, shift=False, max_obj_dist=0.6):
    """Hand of the given side per frame (highest score); shift: moved along its root ray onto the MoGe-2 depth."""
    n = len(frames_h)
    V, J = np.zeros((n, 778, 3), np.float32), np.zeros((n, 21, 3), np.float32)
    valid, dz = np.zeros(n, bool), np.full(n, np.nan)
    for i, f in enumerate(frames_h):
        c = [h for h in f if bool(h["is_right"]) == side and h.get("verts3d") is not None]
        if not c:
            continue
        h = max(c, key=lambda h: h["score"])
        v, j = np.asarray(h["verts3d"], np.float64), np.asarray(h["joints3d"], np.float64)
        V[i], J[i], valid[i] = v, j, True
        if "mask_rle" not in h:
            continue
        m = decode(h["mask_rle"])
        fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        front = (vn * v).sum(1) < 0  # facing the camera
        uv = np.round(project(torch.tensor(K, dtype=torch.float32), torch.from_numpy(v).float()).numpy()).astype(int)
        H, W = m.shape
        ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        ok[ok] &= m[uv[ok, 1], uv[ok, 0]] & (D[i][uv[ok, 1], uv[ok, 0]] > 0)
        if ok.sum() >= 20:
            dz[i] = np.median(D[i][uv[ok, 1], uv[ok, 0]] - v[ok, 2])
    # implausible detections: far from the object, or off the hand's own (median-filtered) wrist path
    wr = J[:, 0].astype(np.float64)
    far = np.linalg.norm(wr - obj_c, axis=1) > max_obj_dist
    vi = np.nonzero(valid & ~far)[0]
    if len(vi) >= 5:
        sm = median_filter(wr[vi], size=(7, 1), mode="nearest")
        jump = np.zeros(n, bool)
        jump[vi] = np.linalg.norm(wr[vi] - sm, axis=1) > 0.12
        far |= jump
    valid &= ~far
    dz[~valid] = np.nan
    good = np.isfinite(dz)
    raw = dz.copy()
    if good.sum() and shift:
        dz = np.interp(np.arange(n), np.nonzero(good)[0], np.clip(dz[good], -0.15, 0.15))
        dz = gaussian_filter1d(median_filter(dz, 7, mode="nearest"), 2.0, mode="nearest")
    else:
        dz = np.zeros(n)
    for i in np.nonzero(valid)[0]:
        r = J[i, 0] / J[i, 0, 2]
        V[i] += (dz[i] * r).astype(np.float32)
        J[i] += (dz[i] * r).astype(np.float32)
    return dict(hand_verts=V, hand_joints=J, hand_valid=valid, hand_dz=dz, hand_dz_raw=raw, hand_side=side, hand_dropped=far)


def obs_quality(K, X, obj, hand):
    """Silhouette IoU on the unoccluded pixels and the visible share of the projected model (object-mask pixels outside the
    dilated hand mask over the model's projected area)."""
    uv = np.round(project(torch.tensor(K, dtype=torch.float32), torch.from_numpy(X).float()).numpy()).astype(int)
    H, W = obj.shape
    m = np.zeros((H, W), np.uint8)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    m[uv[ok, 1], uv[ok, 0]] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    vis = ~cv2.dilate(hand.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return float((m & obj & vis).sum() / max(((m | obj) & vis).sum(), 1)), float((m & obj & vis).sum() / max(m.sum(), 1))


class Carry:
    """Grasp hold: the hands (vertices in the tracking frame, i.e. metric / k_hand) and the rule that decides when the object
    moves with a hand. Grasped at a frame = >= 2 MANO parts of a hand within `dist` of the object's surface at its pose
    there; the hand with most such parts carries it, by its palm's rigid motion (Kabsch on the part-0 vertices)."""

    def __init__(self, hands, labels, dist, P, scale, K, obj, hm, min_iou, min_vis):
        self.hands, self.labels, self.dist = hands, labels, dist
        self.P = P[::max(len(P) // 600, 1)].cpu().numpy().astype(np.float64)
        self.s = float(scale[0])
        self.K, self.obj, self.hm, self.min_iou, self.min_vis = K, obj, hm, min_iou, min_vis

    def grasp(self, j, R, t):
        X = self.s * self.P @ R.T + t
        best, who = 0, None
        for h, (V, valid, lab) in enumerate(self.hands):
            if not valid[max(j - 2, 0):j + 3].all():  # a steady hand (not one re-appearing after a gap)
                continue
            dm = np.linalg.norm(V[j][:, None] - X[None], axis=-1).min(1)
            k = sum(dm[lab == p].min() < self.dist for p in range(16))
            if k > best:
                best, who = k, h
        return who if best >= 2 else None

    def carry(self, h, j, i, R, t, max_step=0.05):
        """The pose moved with the hand's palm from frame j to i; None if the hand is missing or jumps (> max_step)."""
        V, valid, lab = self.hands[h]
        if not valid[i] or np.linalg.norm(V[i][lab == 0].mean(0) - V[j][lab == 0].mean(0)) > max_step:
            return None
        Rh, th = kabsch_np(V[j][lab == 0], V[i][lab == 0])
        return Rh @ R, Rh @ t + th


def kabsch_np(A, B):
    """R, t with R A + t ~ B (rows = points)."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, mb - R @ ma


def track_all(frames, P, f0, pose0, l0, scale, args, still_until=0, carry=None):
    """Frame-to-frame from f0 (forward, then backward), constant-velocity prediction; frames f0 < i < still_until
    (untouched: no hand near the object yet) keep the initial pose. With `carry` (Carry): while a hand holds the object at
    the previous frame, the prediction is the pose carried by that hand; the fit from it on the frame's masks is kept when
    the masks support it (re-acquired: silhouette IoU >= carry.min_iou and visible share >= carry.min_vis), otherwise the
    carried pose is (the hand occludes the object). Returns poses, losses and the carried frames."""
    n = len(frames)
    poses, Ls, held = {f0: pose0}, {f0: l0}, np.zeros(n, bool)
    to_np = lambda R, t: (R[0].cpu().numpy().astype(np.float64), t[0].cpu().numpy().astype(np.float64))
    to_t = lambda R, t: (torch.tensor(R, dtype=torch.float32, device=dev)[None], torch.tensor(t, dtype=torch.float32, device=dev)[None])
    for rng_, step in ((range(f0 + 1, n), -1), (range(f0 - 1, -1, -1), 1)):
        for i in rng_:
            if step == -1 and i < still_until:
                poses[i], Ls[i] = pose0, [np.nan] * 3
                continue
            Rp, tp = poses[i + step]
            Rpp, tpp = poses.get(i + 2 * step, (Rp, tp))
            tpred = tp + 0.5 * (tp - tpp)
            h = None if carry is None else carry.grasp(i + step, *to_np(Rp, tp))
            c = None if h is None else carry.carry(h, i + step, i, *to_np(Rp, tp))
            if c is not None:  # held: the fit from the carried prediction, else from the constant-velocity one, if the masks support it
                Rc, tc = to_t(*c)
                if frames[i].ok:
                    done = False
                    for R0_, t0_ in ((Rc, tc), (Rp, tpred)):
                        R1, t1, _, _, pr = fit(frames[i], P, R0_, t0_, scale, args.track_steps, lr_r=0.01, lr_t=0.002, prior=args.prior)
                        Rn, tn = to_np(R1, t1)
                        iou, vis = obs_quality(carry.K, carry.s * carry.P @ Rn.T + tn, carry.obj[i], carry.hm[i])
                        if iou >= carry.min_iou and vis >= carry.min_vis:
                            poses[i], Ls[i], done = (R1, t1), [float(x[0]) for x in pr], True
                            break
                    if done:
                        continue
                poses[i], Ls[i], held[i] = (Rc, tc), [np.nan] * 3, True
                continue
            if not frames[i].ok:
                poses[i], Ls[i] = (Rp, tpred), [np.nan] * 3
                continue
            R1, t1, _, _, pr = fit(frames[i], P, Rp, tpred, scale, args.track_steps, lr_r=0.01, lr_t=0.002, prior=args.prior)
            poses[i], Ls[i] = (R1, t1), [float(x[0]) for x in pr]
    return (poses, Ls, held) if carry is not None else (poses, Ls)


def main(args):
    t_start = time.time()
    clip = json.load(open(os.path.join(args.clip, "clip.json")))
    dep = np.load(os.path.join(args.clip, "depth.npz"))
    K = np.asarray(dep["K"], np.float64)
    obj = np.load(os.path.join(args.clip, "masks.npz"))["obj"]
    hp = pickle.load(open(os.path.join(args.clip, args.hands), "rb"))
    hm = hand_masks(hp["frames"], obj.shape[1:])
    rec = json.load(open(os.path.join(args.clip, args.recon, "recon.json")))
    f0 = rec["frame"] if args.frame < 0 else args.frame
    D, dscale = align_depth(dep["depth"], obj, hm, clip.get("static_camera", False) and args.depth_align == "ring", f0)
    mesh = trimesh.load(os.path.join(args.clip, args.recon, "mesh.glb"), force="mesh")
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(1.0 / (hi - lo).max())
    lay = None
    if args.init == "layout":  # reconstruct_s3do.py layout (X = s R v + t) in the normalised mesh frame
        L_ = json.load(open(os.path.join(args.clip, args.recon, "layout.json")))
        Rl, sl, tl = np.asarray(L_["R"]), float(L_["s"]), np.asarray(L_["t"])
        e, c0 = float((hi - lo).max()), (lo + hi) / 2
        lay = (Rl, sl * e, tl + sl * Rl @ c0)
    torch.manual_seed(0)
    P = torch.from_numpy(np.asarray(mesh.sample(args.n_pts), np.float64)).float().to(dev)
    P_init = P[torch.randperm(len(P))[:1500]]
    n = len(obj)
    rng = np.random.default_rng(0)
    frames = [Frame(K, D[i], obj[i], hm[i], rng=rng) for i in range(n)]

    # init on the reconstruction frame
    fr = frames[f0]
    Q = fr.Q.cpu().numpy()
    global W3D
    ys, xs = np.nonzero(obj[f0])
    zc = float(np.median(Q[:, 2]))
    s0 = max(xs.max() - xs.min(), ys.max() - ys.min()) * zc / K[0, 0]
    c = Q.mean(0)
    t0 = c + c / np.linalg.norm(c) * s0 * 0.25
    if lay is not None:  # one hypothesis: the reconstruction model's own layout, refined like the search's top k
        B = 1
        R, t, s = (torch.tensor(lay[0], dtype=torch.float32, device=dev)[None], torch.tensor(lay[2], dtype=torch.float32, device=dev)[None],
                   torch.tensor([lay[1]], dtype=torch.float32, device=dev))
        s0 = lay[1]
        order = torch.arange(1)
    else:
        Rg = torch.from_numpy(rotation_grid(args.n_rot // 8, 8)).float().to(dev)
        B = len(Rg)
        fi = Frame(K, D[f0], obj[f0], hm[f0], n_obs=800, n_cov=500, rng=rng)
        R, t, s, L, _ = fit(fi, P_init, Rg, torch.tensor(t0, dtype=torch.float32, device=dev).expand(B, 3).contiguous(),
                            torch.full((B,), s0, device=dev), args.init_steps, free_scale=True, lr_s=0.02)
        order = torch.argsort(L)[:args.top_k]
    R, t, s, L, parts = fit(fr, P, R[order], t[order], s[order], args.refine_steps, free_scale=True, lr_r=0.01, lr_t=0.002)
    k = int(torch.argmin(L))
    print("init frame %d: s0 %.3f -> scale %.3f m, loss %.3f (3D %.1f mm, sil %.2f px, cov %.2f px), %d hypotheses" %
          (f0, s0, float(s[k]), float(L[k]), float(parts[0][k]) * 1e3, float(parts[1][k]), float(parts[2][k]), B))
    scale = s[k:k + 1]
    W3D = args.w3d  # init uses the full depth term (unoccluded frame); tracking: MoGe-2 warps near the arm
    still = first_touch(obj, hm) if args.touch and clip.get("static_camera", False) else 0
    print("object untouched until frame %d (static there)" % still)
    init = ((R[k:k + 1], t[k:k + 1]), [float(x[k]) for x in parts])
    poses, Ls = track_all(frames, P, f0, *init, scale, args, still)
    axis_scale = np.ones(3)
    if args.shape_frames:  # multi-frame anisotropic scale (single-view reconstructions guess the unseen depth)
        sel = [i for i in np.linspace(0, n - 1, 4 * args.shape_frames).astype(int) if frames[i].ok]
        sel = sorted(set(sel[::max(len(sel) // args.shape_frames, 1)]) | {f0})
        a = torch.zeros(3, device=dev, requires_grad=True)
        W_ = torch.zeros(len(sel), 3, device=dev, requires_grad=True)
        dT = torch.zeros(len(sel), 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([dict(params=[a], lr=0.01), dict(params=[W_], lr=0.01), dict(params=[dT], lr=0.002)])
        R0 = torch.cat([poses[i][0] for i in sel])
        t0_ = torch.cat([poses[i][1] for i in sel])
        for _ in range(args.shape_steps):
            Ps = P * torch.exp(a)
            Rr, tt = rodrigues(W_) @ R0, t0_ + dT
            L = sum(losses(frames[i], Ps, Rr[j:j + 1], tt[j:j + 1], scale)[0][0] for j, i in enumerate(sel)) / len(sel)
            L = L + ((a / 0.3) ** 2).sum() * 0.01
            opt.zero_grad()
            L.backward()
            opt.step()
        axis_scale = torch.exp(a).detach().cpu().numpy().astype(np.float64)
        print("shape refinement on %d frames: axis scale %s" % (len(sel), np.round(axis_scale, 3).tolist()))
        P = P * torch.exp(a).detach()
        mesh.apply_transform(np.diag(np.r_[axis_scale, 1]))
        j0 = sel.index(f0)
        R0f, t0f, _, _, pr0 = fit(frames[f0], P, (rodrigues(W_) @ R0)[j0:j0 + 1].detach(), (t0_ + dT)[j0:j0 + 1].detach(), scale,
                                  args.refine_steps, lr_r=0.01, lr_t=0.002)
        init = ((R0f, t0f), [float(x[0]) for x in pr0])
        poses, Ls = track_all(frames, P, f0, *init, scale, args, still)
    S = float(scale[0])
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        T[i, :3, :3], T[i, :3, 3] = poses[i][0][0].cpu().numpy(), poses[i][1][0].cpu().numpy()
    if args.smooth > 0:  # light temporal smoothing of the translation (rotations kept)
        T[:, :3, 3] = gaussian_filter1d(T[:, :3, 3], args.smooth, axis=0, mode="nearest")
    mesh.apply_scale(S)  # metric object frame: T_cam_obj maps it to the camera
    mesh.export(os.path.join(args.clip, args.recon, "mesh_metric.glb" if args.out == "track.npz" else args.out[:-4] + "_mesh.glb"))
    Pm = np.asarray(mesh.sample(4000), np.float64)
    iou = np.array([silhouette_iou(K, Pm @ T[i][:3, :3].T + T[i][:3, 3], obj[i], hm[i]) for i in range(n)])
    # support plane around the object on the reconstruction frame
    ring = cv2.dilate(obj[f0].astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool) & \
        ~cv2.dilate((obj[f0] | hm[f0]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (D[f0] > 0)
    yy, xx = np.nonzero(ring)
    (pn, pd), frac = ransac_plane(backproject(K, np.stack([xx, yy], 1).astype(np.float64), D[f0][yy, xx].astype(np.float64)), rng=rng)
    from bench.common import ManoLayer

    side = args.side == "right" if args.side != "auto" else pick_side_2d(hp["frames"], obj)
    side = pick_side(hp["frames"], T, Pm) if side is None else side
    hands = place_hands(hp["frames"], D, K, ManoLayer(os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano"))).faces[side], side,
                        T[:, :3, 3], shift=args.hand_depth == "moge")
    if args.two_hands:  # the other hand too (two-hand / handover clips): hand2_*
        h2 = place_hands(hp["frames"], D, K, ManoLayer(os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano"))).faces[not side], not side,
                         T[:, :3, 3], shift=args.hand_depth == "moge")
        hands.update({"hand2_" + k[5:]: v for k, v in h2.items()})
    kh, contact_cost = 1.0, np.nan
    labels = ManoLayer(os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano"))).layers[side].lbs_weights.detach().cpu().numpy().argmax(1)
    if args.hand_scale:  # metric scale of the object from the manipulating hand
        cams = os.path.join(args.clip, "cams.npz")
        T_move = None
        if os.path.exists(cams):  # camera t -> camera 0, so that the camera's own motion is not taken for the object's
            Twc = np.load(cams)["T_world_cam"]
            T_move = np.linalg.inv(Twc[0]) @ Twc @ T
        kh, kc = hand_metric_scale(T, mesh, hands["hand_verts"].astype(np.float64), hands["hand_valid"], labels, T_move=T_move)
        contact_cost = min(kc["cost"]) if kc else np.nan
    held = np.zeros(n, bool)
    if args.grasp_hold:  # second pass: the object moves with the hand that holds it through occluded frames, re-acquired on the masks
        hs_ = [(hands["hand_verts"].astype(np.float64) / kh, hands["hand_valid"], labels)]
        if args.two_hands:
            hs_.append((hands["hand2_verts"].astype(np.float64) / kh, hands["hand2_valid"],
                        ManoLayer(os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano"))).layers[not side].lbs_weights.detach().cpu().numpy().argmax(1)))
        carry = Carry(hs_, labels, args.grasp_dist / kh, P, scale, K, obj, hm, args.hold_iou, args.hold_vis)
        poses, Ls, held = track_all(frames, P, f0, *init, scale, args, still, carry=carry)
        for i in range(n):
            T[i, :3, :3], T[i, :3, 3] = poses[i][0][0].cpu().numpy(), poses[i][1][0].cpu().numpy()
        if args.smooth > 0:
            T[:, :3, 3] = gaussian_filter1d(T[:, :3, 3], args.smooth, axis=0, mode="nearest")
        iou = np.array([silhouette_iou(K, Pm @ T[i][:3, :3].T + T[i][:3, 3], obj[i], hm[i]) for i in range(n)])
        print("grasp hold: %d frames carried by the hand" % held.sum())
    if args.hand_scale:
        T[:, :3, 3] *= kh
        S *= kh
        pd *= kh
        mesh.apply_scale(kh)
        mesh.export(os.path.join(args.clip, args.recon, "mesh_metric.glb" if args.out == "track.npz" else args.out[:-4] + "_mesh.glb"))
        print("hand metric scale k = %.3f (object size and distance x k)" % kh)
    depth_axis = (0, 1.0, 0.0)
    if args.hand_depth_axis:
        depth_axis = hand_depth_axis(T, mesh, hands["hand_verts"].astype(np.float64), hands["hand_valid"], labels, f0)
        a, sc, u = depth_axis
        d = np.ones(3)
        d[a] = sc
        mesh.apply_transform(np.diag(np.r_[d, 1]))
        off = np.zeros(3)
        off[a] = u
        T[:, :3, 3] += np.einsum("nij,j->ni", T[:, :3, :3], off)
        mesh.export(os.path.join(args.clip, args.recon, "mesh_metric.glb" if args.out == "track.npz" else args.out[:-4] + "_mesh.glb"))
        print("hand depth axis: object axis %d x %.2f, shift %.1f mm" % (a, sc, u * 1e3))
    L3 = np.array([Ls[i][0] for i in range(n)])
    np.savez(os.path.join(args.clip, args.out), T_cam_obj=T, scale=S, K=K, depth_scale=dscale, loss3d=L3,
             loss_sil=np.array([Ls[i][1] for i in range(n)]), loss_cov=np.array([Ls[i][2] for i in range(n)]), iou=iou,
             plane=np.r_[pn, pd], plane_inliers=frac, frame0=f0, axis_scale=axis_scale, still_until=still, hand_scale=kh,
             hand_depth_axis=np.array(depth_axis), contact_cost=contact_cost, recon=args.recon,
             init_scale=float(s0), fit_scale_f0=float(scale[0]), held=held, **hands)
    print("%s: scale %.3f m (extent %s mm), 3D residual median %.1f mm, silhouette IoU median %.3f / min %.3f, plane inliers %.2f,"
          " hand %s valid %d / %d, dz median %.1f mm, contact cost %.2f mm, %.0f s" %
          (args.clip, S, np.round((mesh.bounds[1] - mesh.bounds[0]) * 1e3).astype(int).tolist(), np.nanmedian(L3) * 1e3,
           np.median(iou), iou.min(), frac, "right" if hands["hand_side"] else "left", hands["hand_valid"].sum(), n,
           np.median(hands["hand_dz"]) * 1e3, contact_cost * 1e3, time.time() - t_start))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--hands", default="hands.pkl")
    p.add_argument("--recon", default="recon", help="reconstruction dir in the clip")
    p.add_argument("--two_hands", type=int, default=0, help="also place the other hand (saved as hand2_*)")
    p.add_argument("--init", default="search", choices=["search", "layout"],
                   help="frame-0 pose: rotation search, or layout.json of reconstruct_s3do.py (SAM 3D Objects)")
    p.add_argument("--frame", type=int, default=-1, help="init frame (default: the reconstruction frame)")
    p.add_argument("--n_pts", type=int, default=4000)
    p.add_argument("--n_rot", type=int, default=320)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--init_steps", type=int, default=150)
    p.add_argument("--refine_steps", type=int, default=300)
    p.add_argument("--track_steps", type=int, default=80)
    p.add_argument("--prior", type=float, default=0.02)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--shape_frames", type=int, default=12, help="frames for the anisotropic scale refinement (0: off)")
    p.add_argument("--shape_steps", type=int, default=300)
    p.add_argument("--touch", type=int, default=1, help="the object is static until a hand mask touches its mask (0: off)")
    p.add_argument("--hand_scale", type=int, default=1, help="object metric scale (size + distance) from the hand contact")
    p.add_argument("--hand_depth_axis", type=int, default=0, help="object extent along the init view's depth from the grasp (DexYCB: unreliable, off)")
    p.add_argument("--w3d", type=float, default=0.0, help="weight of the MoGe-2 depth term while tracking (silhouette terms = 1; DexYCB: 0 = no drift)")
    p.add_argument("--depth_align", default="none", choices=["none", "ring"])
    p.add_argument("--grasp_hold", type=int, default=0, help="second pass: while a hand holds the object (>= 2 parts within "
                   "--grasp_dist), the object is carried by the hand's palm motion; refits on the masks are kept only when they "
                   "support them (--hold_iou, --hold_vis)")
    p.add_argument("--grasp_dist", type=float, default=0.01, help="m (metric, the hand's scale)")
    p.add_argument("--hold_iou", type=float, default=0.5)
    p.add_argument("--hold_vis", type=float, default=0.3)
    p.add_argument("--out", default="track.npz")
    p.add_argument("--side", default="auto", choices=["auto", "right", "left"])
    p.add_argument("--hand_depth", default="fah", choices=["fah", "moge"],
                   help="fah: FAH's own metric placement (DexYCB: root depth +11 mm vs MoGe-2 shift -37 mm); moge: shift onto MoGe-2")
    main(p.parse_args())
