"""Our human + scene reconstruction of a monocular clip (VideoMimic's real2sim setting, GT-free): FAB body, ViPE camera
and metric depth, GeoCalib gravity and a TSDF scene from ViPE's depth; the human-scene step that follows is
bench/scene_recon/joint.py. Clips are listed as in bench/world_vipe.py --list ({name: {images, fps}}).

Stages (each skips clips whose output exists), as scripts/fab_world.sh runs them:
  fab    FAB on the frames with ViPE's intrinsics (bench/sonic_motion.run_fab: the most prominent tracked person,
         the released one-pass SMPL fit) -> <out>/fab/<name>_raw.npz (frames, valid, pose_c, betas, trans_c, box, K; pts /
         surface_idx: the dense surface points, camera frame, for fab-retarget's points mode)
  world  ViPE UniDepth camera (bench/world_vipe.py --depth) at ViPE's own metric scale (--f_own 1.0), applied to the
         whole world (camera, depth, person); gravity-aligned z-up world from GeoCalib (x = the first camera's
         horizontal view direction, origin under the first pelvis); SMPL smoothed (rotations sigma_rot, pelvis
         sigma_pos frames); scene = TSDF of ViPE's metric depth with ViPE's dynamic instances, the segmenter's people
         + sky (--seg_dir) and our projected body masked (open3d), no scene in front of the visible person
         (--vis_carve 1); heightfield = per cell the highest scene surface below the person's local pelvis height
         + 0.3 m (no ceilings / canopies).
    -> <out>/recon/<name>.npz: fps, frames, pose (T, 72) world axis-angle, betas (10,), trans (T, 3), joints (T, 24, 3),
       verts (T, 6890, 3) f16, contacts json (per group, human_contact_labels on the terrain), T_world_cam (T, 4, 4),
       K, scale_fix, scale_est, up_world (pre-alignment), R_align, origin
       <out>/recon/<name>_scene.ply (aligned world), <name>_hfield.npz (x0, y0, res, height (ny, nx), valid)

    python -m bench.videomimic.recon fab --list clips.json --out <dir> --vipe <vipe out>
        # defaults: the released FAB-L model, SMPL fitter and person detector ($FAB_WEIGHTS)
    python -m bench.videomimic.recon world --list clips.json --out <dir> --vipe <vipe out> --geocalib <geocalib out>
        --refine 0 --f_own 1.0 --vis_carve 1 --seg_dir <seg out>
    (the whole chain, these stages included, in one command: scripts/fab_world.sh)
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from fab import paths


def clips(args):
    with open(args.list) as f:
        c = json.load(f)
    return [(k, v["images"], float(v.get("fps", 30))) for k, v in sorted(c.items())
            if not args.only or any(o in k for o in args.only.split(","))][args.shard::args.num_shards]


def vipe_K(vipe, name):
    d = np.load(Path(vipe) / "intrinsics" / f"{name}.npz")["data"]
    fx, fy, cx, cy = np.median(d[:, :4], 0)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def stage_fab(args):
    from bench.sonic_motion import one_pass_fit, run_fab
    from bench.body_models import SMPL

    os.makedirs(Path(args.out) / "fab", exist_ok=True)
    body = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    fa = SimpleNamespace(fit_iters=0, fab_ckpt=args.fab_ckpt, detector=args.detector, bm_dir=args.bm_dir,
                         fitter=args.fitter, fast=False, predictor_kw=args.predictor_kw)
    for name, images, fps in clips(args):
        out = Path(args.out) / "fab" / f"{name}_raw.npz"
        if out.exists() or not (Path(args.vipe) / "intrinsics" / f"{name}.npz").exists():
            continue
        K = vipe_K(args.vipe, name)
        boxes = np.full((len(images), 4), np.nan)
        sb = Path(args.subject_boxes) / f"{name}.npz" if args.subject_boxes else None
        if sb is not None and sb.exists():  # the annotated target person (identity only: picks among the detections)
            boxes = np.load(sb)["box"].astype(np.float64)
        fab_out, surface_idx, joints = run_fab(fa, images, K, boxes)
        fit = one_pass_fit(body, fab_out)
        np.savez(out, frames=np.arange(len(images)), valid=fab_out["valid"], pose_c=fit["pose"], betas=fit["betas"],
                 trans_c=fit["trans"], g1_q=fab_out["g1_q"], g1_root_c=fab_out["g1_root"], box=fab_out["box"], K=K,
                 contact=fab_out["contact"], pts=fab_out["pts"].astype(np.float32), surface_idx=np.asarray(surface_idx),
                 joints=np.array(joints), fab_ckpt=args.fab_ckpt, fps=fps)
        print(f"[fab] {name}: {len(images)} frames, valid {fab_out['valid'].mean():.2f}", flush=True)


def fill_invalid(x, valid):
    if valid.all() or not valid.any():
        return x
    t, good = np.arange(len(x)), np.flatnonzero(valid)
    flat = x.reshape(len(x), -1)
    return np.stack([np.interp(t, good, flat[good, j]) for j in range(flat.shape[1])], 1).reshape(x.shape)


def smooth_quats(q, sigma):
    q = q.copy()
    for t in range(1, len(q)):
        q[t][(q[t] * q[t - 1]).sum(-1) < 0] *= -1
    if sigma > 0:
        q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rot_between(a, b):
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * np.array([1.0, 0, 0])).as_matrix()
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def scale_fix(args, name, T):
    """The GT-free two-prior scale: gmean(own x f_own, bridge(Metric3D ViPE run) x f_bridge); the first
    item alone when they disagree by more than 1.3x or the camera barely moves (no bridge windows)."""
    from bench.world_utils import bridge_scale

    est = {}
    mp = Path(args.vipe_m3d) / "cams" / f"{name}.npz" if args.vipe_m3d else None
    if mp is not None and mp.exists():
        Tm = np.load(mp)["T_world_cam"].astype(np.float64)
        est["bridge"] = bridge_scale(T[:, :3, 3], [Tm[: len(T), :3, 3]])
    v = [args.f_own] + ([est["bridge"] * args.f_bridge] if np.isfinite(est.get("bridge", np.nan)) else [])
    est["agree"] = bool(max(v) / min(v) <= 1.3)
    fix = float(np.exp(np.log(v).mean())) if est["agree"] else float(v[0])
    return fix, est


def human_ruler_raw(body, raw, dense, n, valid, stride=5):
    """The person as the world's metric ruler: per sampled frame, FAB's camera-frame body (z-buffered vertices) over the
    unscaled scene depth on the person's pixels (ViPE's dynamic mask); median over pixels, then frames. World x this =
    the scene / camera at the scale of FAB's metric body."""
    h, w = dense["depth"].shape[1:]
    fr, rat = dense["frames"], []
    for k in range(0, len(fr), stride):
        f = int(fr[k])
        if f >= n or not valid[f]:
            continue
        tt = lambda x: torch.as_tensor(np.asarray(x[f: f + 1]), dtype=torch.float32, device="cuda")  # noqa: E731
        with torch.no_grad():
            v, _ = body(tt(raw["pose_c"][:, :3]), tt(raw["pose_c"][:, 3:]), tt(raw["betas"]), tt(raw["trans_c"]))
        X = v[0].cpu().numpy().astype(np.float64)
        K = dense["K"][k].astype(np.float64)
        ok = X[:, 2] > 0.1
        u = np.round(X[ok, 0] / X[ok, 2] * K[0, 0] + K[0, 2]).astype(int)
        vv = np.round(X[ok, 1] / X[ok, 2] * K[1, 1] + K[1, 2]).astype(int)
        inb = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        zb = np.full(h * w, np.inf)
        np.minimum.at(zb, vv[inb] * w + u[inb], X[ok, 2][inb])
        d = dense["depth"][k].astype(np.float64).ravel()
        m = np.isfinite(zb) & dense["dyn"][k].ravel() & np.isfinite(d) & (d > 0.1)
        if m.sum() >= 30:
            rat.append(np.median(zb[m] / d[m]))
    return (float(np.median(rat)) if rat else float("nan")), len(rat)


def tsdf_scene(dense, T_wc, fix, person_masks, voxel, trunc, max_depth, stride=1, min_weight=3.0):
    """open3d TSDF fusion (tensor VoxelBlockGrid, CUDA) of the masked metric depth (x fix) with camera -> world poses
    T_wc (already x fix); every stride-th frame -> legacy TriangleMesh."""
    import open3d as o3d
    import open3d.core as o3c

    dev = o3c.Device("CUDA:0" if o3c.cuda.is_available() and os.environ.get("RECON_TSDF_DEVICE", "cuda") != "cpu" else "CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight"), attr_dtypes=(o3c.float32, o3c.float32),
                                        attr_channels=((1), (1)), voxel_size=voxel, block_resolution=16,
                                        block_count=100000, device=dev)
    depth, dyn, K, fr = dense["depth"], dense["dyn"], dense["K"], dense["frames"]
    for k in range(0, len(fr), stride):
        f = int(fr[k])
        if f >= len(T_wc):
            continue
        d = depth[k].astype(np.float32) * fix
        bad = dyn[k] | ~np.isfinite(d) | (d <= 0.05)
        if person_masks is not None:
            bad |= person_masks[f]
        d[bad] = 0
        if ((d > 0) & (d < max_depth)).sum() < 100:  # fully masked / out of range: nothing to integrate
            continue
        img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d))).to(dev)
        Kt, E = o3c.Tensor(K[k], o3c.float64), o3c.Tensor(np.linalg.inv(T_wc[f]), o3c.float64)
        try:
            blocks = vbg.compute_unique_block_coordinates(img, Kt, E, 1.0, max_depth)
            vbg.integrate(blocks, img, Kt, E, 1.0, max_depth, trunc / voxel)
        except RuntimeError:  # "No block is touched" (e.g. every valid pixel behind the truncation)
            continue
    try:
        mesh = vbg.extract_triangle_mesh(weight_threshold=min_weight).to_legacy()
    except RuntimeError:  # nothing integrated (every frame masked / beyond max_depth)
        return o3d.geometry.TriangleMesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def tsdf_scene_quality(dense, T_wc, fix, person_masks, voxel, trunc, max_depth, stride=1, min_weight=3.0, fallback_w=0.1,
                       z_min=0.3, z_pow=2.0):
    """tsdf_scene's projective TSDF (the same touched blocks, voxels, truncation, open3d's floor
    projection and extraction) with each observation weighted by q / max(z, z_min)^z_pow instead of 1: q = 1 on the
    MVS-verified pixels (dense["mvs_mask"], bench/scene_recon/mvs_depth.py), fallback_w on the others (all pixels
    alike without a mask). Far grazing views and ViPE-fallback frames do not outvote near MVS views on stair treads
    (stair treads tilt otherwise). The extraction threshold stays the observation count (min_weight)."""
    import open3d as o3d
    import open3d.core as o3c
    import torch.utils.dlpack

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = 16
    depth, dyn, K, fr = dense["depth"], dense["dyn"], dense["K"], dense["frames"]
    mm = dense["mvs_mask"] if "mvs_mask" in dense else None
    loc = torch.stack(torch.meshgrid(*[torch.arange(R, device=dev)] * 3, indexing="ij"), -1).reshape(-1, 3)  # [z, y, x]
    loc = loc.flip(-1)  # -> (x, y, z), in open3d's attribute order (block, z, y, x)
    band, steps = 8.0 * voxel, torch.linspace(0.0, 1.0, 4, device=dev)  # compute_unique_block_coordinates' default band
    slot, keys = {}, []
    S = Wt = N = torch.zeros(0, R ** 3, device=dev)
    for k in range(0, len(fr), stride):
        f = int(fr[k])
        if f >= len(T_wc):
            continue
        d = depth[k].astype(np.float32) * fix
        bad = dyn[k] | ~np.isfinite(d) | (d <= 0.05)
        if person_masks is not None:
            bad |= person_masks[f]
        d[bad] = 0
        d[d >= max_depth] = 0
        if (d > 0).sum() < 100:
            continue
        if not (d[::4, ::4] > 0).any():
            continue
        q = np.where(mm[k], 1.0, fallback_w).astype(np.float32) if mm is not None else np.ones_like(d)
        D, Q = torch.as_tensor(d, device=dev), torch.as_tensor(q, device=dev)
        Kk = torch.as_tensor(K[k], device=dev, dtype=torch.float32)
        Tw = torch.as_tensor(T_wc[f], device=dev, dtype=torch.float32)
        Ec = torch.linalg.inv(Tw)
        vv, uu = torch.nonzero(D > 0, as_tuple=True)
        s4 = (uu % 4 == 0) & (vv % 4 == 0)  # open3d's DepthTouch samples every 4th pixel
        vv, uu = vv[s4], uu[s4]
        z = D[vv, uu]
        ray = torch.stack([(uu - Kk[0, 2]) / Kk[0, 0], (vv - Kk[1, 2]) / Kk[1, 1], torch.ones_like(z)], 1)
        t0, t1 = torch.clamp(z - band, min=0.0), torch.clamp(z + band, max=max_depth)
        ts = t0[:, None] + (t1 - t0)[:, None] * steps[None]
        P = (ray[:, None] * ts[..., None]).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
        bk = torch.unique(torch.floor(P / (voxel * R)).long(), dim=0)  # blocks touched (open3d's DepthTouch)
        idx = []
        for b in map(tuple, bk.cpu().numpy().tolist()):
            s = slot.get(b)
            if s is None:
                s = slot[b] = len(keys)
                keys.append(b)
            idx.append(s)
        if len(keys) > len(S):  # grow the block store (x2)
            n = max(len(keys), 2 * len(S))
            S, Wt, N = (torch.cat([A_, torch.zeros(n - len(A_), R ** 3, device=dev)]) for A_ in (S, Wt, N))
        idx = torch.as_tensor(idx, device=dev)
        X = ((bk[:, None] * R + loc[None]).float() * voxel).reshape(-1, 3)  # voxel positions (open3d: index x voxel)
        Xc = X @ Ec[:3, :3].T + Ec[:3, 3]
        zc = Xc[:, 2]
        h, w = d.shape
        zs = torch.where(zc > 0, zc, torch.ones_like(zc))
        u = torch.floor(Kk[0, 0] * Xc[:, 0] / zs + Kk[0, 2]).long()
        v = torch.floor(Kk[1, 1] * Xc[:, 1] / zs + Kk[1, 2]).long()
        inb = (zc > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dp = torch.zeros_like(zc)
        dp[inb] = D[v[inb], u[inb]]
        sdf = dp - zc
        ok = inb & (dp > 0) & (sdf >= -trunc)
        t = torch.clamp(sdf, max=trunc) / trunc
        wq = torch.zeros_like(zc)
        wq[ok] = Q[v[ok], u[ok]] / torch.clamp(dp[ok], min=z_min) ** z_pow
        S.index_add_(0, idx, (wq * torch.where(ok, t, torch.zeros_like(t))).reshape(len(idx), -1))
        Wt.index_add_(0, idx, wq.reshape(len(idx), -1))
        N.index_add_(0, idx, ok.float().reshape(len(idx), -1))
    if not keys:
        return o3d.geometry.TriangleMesh()
    S, Wt, N = S[: len(keys)], Wt[: len(keys)], N[: len(keys)]
    tsdf = torch.where(Wt > 0, S / torch.clamp(Wt, min=1e-30), torch.ones_like(S))
    odev = o3c.Device("CUDA:0" if dev.type == "cuda" and o3c.cuda.is_available() else "CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight"), attr_dtypes=(o3c.float32, o3c.float32),
                                        attr_channels=((1), (1)), voxel_size=voxel, block_resolution=R,
                                        block_count=int(len(keys) * 1.2) + 16, device=odev)
    hm = vbg.hashmap()
    kt = o3c.Tensor(np.asarray(keys, dtype=np.int32), device=odev)
    hm.activate(kt)
    buf, found = hm.find(kt)
    bi = torch.utils.dlpack.from_dlpack(buf.to_dlpack()).long().to(dev)
    ta = torch.utils.dlpack.from_dlpack(vbg.attribute("tsdf").to_dlpack())
    wa = torch.utils.dlpack.from_dlpack(vbg.attribute("weight").to_dlpack())
    ta[bi] = tsdf.reshape(-1, R, R, R, 1).to(ta.device)
    wa[bi] = N.reshape(-1, R, R, R, 1).to(wa.device)
    try:
        mesh = vbg.extract_triangle_mesh(weight_threshold=min_weight).to_legacy()
    except RuntimeError:
        return o3d.geometry.TriangleMesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def person_masks_from_mesh(verts_c, faces, K, hw, dilate=6):
    """(T, h, w) bool: our camera-frame body mesh rasterised (filled triangles of the projected vertices), dilated."""
    import cv2

    h, w = hw
    out = np.zeros((len(verts_c), h, w), bool)
    ker = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    for t, v in enumerate(verts_c):
        z = v[:, 2]
        if (z < 0.05).all():
            continue
        uv = (v[:, :2] / np.clip(z[:, None], 1e-3, None)) @ K[t][:2, :2].T + K[t][:2, 2]
        m = np.zeros((h, w), np.uint8)
        tri = uv[faces].round().astype(np.int32)
        cv2.fillPoly(m, list(tri[(z[faces] > 0.05).all(1)]), 1)
        out[t] = cv2.dilate(m, ker) > 0
    return out


def heightfield(mesh, ref_xy, ref_z, res, margin, above, median=3):
    """Per cell of a grid over the person's path (+ margin) the highest scene surface below the local reference height
    (the nearest path sample's pelvis height + above: seats count, rails / table tops above the pelvis / ceilings do
    not): one downward ray per cell from that height (open3d raycasting). Cells without a hit take the nearest hit
    cell's height; then a median filter (median x median cells)."""
    import open3d as o3d
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree

    lo, hi = ref_xy.min(0) - margin, ref_xy.max(0) + margin
    nx, ny = int(np.ceil((hi[0] - lo[0]) / res)) + 1, int(np.ceil((hi[1] - lo[1]) / res)) + 1
    gx, gy = lo[0] + res * np.arange(nx), lo[1] + res * np.arange(ny)
    X, Y = np.meshgrid(gx, gy)
    _, k = cKDTree(ref_xy).query(np.stack([X.ravel(), Y.ravel()], 1))
    z0 = ref_z[k] + above
    if len(mesh.triangles) == 0:  # no scene: a flat floor under the path
        H = np.full((ny, nx), float(np.min(ref_z) - 0.9), np.float32)
        return {"x0": float(lo[0]), "y0": float(lo[1]), "res": float(res), "height": H, "valid": np.zeros((ny, nx), bool)}
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.zeros((len(z0), 6), np.float32)
    rays[:, 0], rays[:, 1], rays[:, 2], rays[:, 5] = X.ravel(), Y.ravel(), z0, -1.0
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    H = (z0 - t_hit).reshape(ny, nx)
    valid = np.isfinite(H)
    if not valid.any():  # nothing under the path: a flat floor
        H = np.full((ny, nx), float(np.min(ref_z) - 0.9))
    elif not valid.all():
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        H = H[iy, ix]
    if median > 1:  # thin structures / fusion speckle: a median filter of the observed heights
        from scipy.ndimage import median_filter

        H = median_filter(H, size=median, mode="nearest")
    return {"x0": float(lo[0]), "y0": float(lo[1]), "res": float(res), "height": H.astype(np.float32),
            "valid": valid}


def clean_heightfield(hf, path_xy, near=1.0, snap=0.04, flat_slope=0.08, min_share=0.03):
    """Level snapping of a reconstructed heightfield (TSDF speckle on floors / stair treads / seats): the height levels
    of the cells within `near` m of the path (peaks of the 1 cm height histogram holding >= min_share of those cells);
    every cell whose smoothed local slope is below flat_slope (~4.5 deg: ramps / hills are left alone) and whose height
    is within `snap` m of a level takes that level; then a 3x3 median. -> a new dict (+ levels)."""
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.signal import find_peaks
    from scipy.spatial import cKDTree

    H = np.asarray(hf["height"], np.float64)
    ny, nx = H.shape
    r = float(hf["res"])
    X, Y = np.meshgrid(float(hf["x0"]) + r * np.arange(nx), float(hf["y0"]) + r * np.arange(ny))
    d = cKDTree(np.asarray(path_xy)).query(np.stack([X.ravel(), Y.ravel()], 1))[0].reshape(ny, nx)
    hs = gaussian_filter(median_filter(H, size=5, mode="nearest"), 1.0, mode="nearest")
    gy, gx = np.gradient(hs, r)
    flat = np.hypot(gx, gy) < flat_slope
    v = H[(d < near) & flat]
    out = dict(hf)
    if len(v) < 20:
        out["levels"] = np.zeros(0)
        return out
    lo, hi = np.floor(v.min() * 100) / 100, np.ceil(v.max() * 100) / 100 + 0.01
    cnt, edges = np.histogram(v, bins=max(int(round((hi - lo) / 0.01)), 1), range=(lo, hi))
    cs = np.convolve(cnt, np.ones(3) / 3, mode="same")
    pk, _ = find_peaks(cs, height=min_share * len(v), distance=6)
    levels = (edges[pk] + edges[pk + 1]) / 2
    Hn = H.copy()
    if len(levels):
        k = np.abs(H[..., None] - levels).argmin(-1)
        L = levels[k]
        m = flat & (np.abs(H - L) < snap)
        Hn[m] = L[m]
    out["height"] = median_filter(Hn, size=3, mode="nearest").astype(np.float32)
    out["levels"] = levels
    return out


class Terrain:
    """Bilinear heightfield lookup (torch, differentiable in xy) of a heightfield dict."""

    def __init__(self, hf, device="cuda"):
        self.x0, self.y0, self.res = hf["x0"], hf["y0"], hf["res"]
        self.H = torch.as_tensor(hf["height"], dtype=torch.float32, device=device)

    def __call__(self, xy):
        ny, nx = self.H.shape
        xy = torch.nan_to_num(xy)
        u = ((xy[..., 0] - self.x0) / self.res).clamp(0, nx - 1.001)
        v = ((xy[..., 1] - self.y0) / self.res).clamp(0, ny - 1.001)
        i0, j0 = v.floor().long(), u.floor().long()
        a, b = v - i0, u - j0
        H = self.H
        return ((1 - a) * (1 - b) * H[i0, j0] + (1 - a) * b * H[i0, j0 + 1] + a * (1 - b) * H[i0 + 1, j0]
                + a * b * H[i0 + 1, j0 + 1])


def refine_contacts(Vc, R, c, terr, contact, foot_idx, low_idx, fps, iters=300, w_c=1.0, w_pen=10.0, w_s=0.02,
                    w_p=0.05, max_e=0.15):
    """Per-frame depth factors e_t of the person along the camera rays (world p = c_t + (1 + e_t) R_t p_cam) and one
    vertical offset dz: the contact foot vertices touch the terrain, no sampled body vertex penetrates it, e smooth
    and small. Vc (T, V, 3) camera-frame vertices, R (T, 3, 3), c (T, 3) aligned world cameras; contact (T, 2) bool
    (left / right foot); foot_idx [left idx, right idx]; low_idx vertex sample for the penetration term."""
    dev = "cuda"
    Vc_, R_, c_ = (torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (Vc, R, c))
    con = torch.as_tensor(contact, device=dev)
    e = torch.zeros(len(Vc), device=dev, requires_grad=True)
    dz = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.Adam([e, dz], lr=0.005)
    li = torch.as_tensor(low_idx, device=dev)

    def world(idx):
        p = torch.einsum("tij,tvj->tvi", R_, Vc_[:, idx]) * (1 + max_e * torch.tanh(e / max_e))[:, None, None]
        p = p + c_[:, None]
        return p + torch.stack([torch.zeros_like(dz), torch.zeros_like(dz), dz])

    def losses():
        L = {}
        cl = 0.0
        for k, idx in enumerate(foot_idx):
            p = world(idx)
            gap = (p[..., 2] - terr(p[..., :2])).min(1).values  # the sole's lowest point above the terrain
            if con[:, k].any():
                cl = cl + (gap[con[:, k]] ** 2).mean()
        L["contact"] = cl
        p = world(li)
        L["pen"] = (torch.relu(terr(p[..., :2]) - p[..., 2]) ** 2).mean()
        ee = max_e * torch.tanh(e / max_e)
        L["smooth"] = ((ee[1:] - ee[:-1]) * fps).pow(2).mean()
        L["prior"] = (ee ** 2).mean()
        return L

    with torch.no_grad():
        before = {k: float(v) for k, v in losses().items()}
    for _ in range(iters):
        L = losses()
        loss = w_c * L["contact"] + w_pen * L["pen"] + w_s * L["smooth"] + w_p * L["prior"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = {k: float(v) for k, v in losses().items()}
        ee = (max_e * torch.tanh(e / max_e)).cpu().numpy()
    return ee, float(dz.detach()), {"before": before, "after": after, "e_abs_mean": float(np.abs(ee).mean()),
                                    "dz": float(dz.detach())}


def stage_world(args):
    import trimesh

    from bench.body_models import SMPL
    from bench.world_utils import geocalib_up
    from fab.retarget.contact import human_contact_labels

    od = Path(args.out) / "recon"
    os.makedirs(od, exist_ok=True)
    body = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    faces = body.faces.cpu().numpy()
    part = body.weights.argmax(1).cpu().numpy()
    foot_idx = [np.flatnonzero(np.isin(part, js)) for js in ((7, 10), (8, 11))]
    low_idx = np.arange(0, 6890, 7)
    for name, images, fps in clips(args):
        out = od / f"{name}.npz"
        rp = Path(args.out) / "fab" / f"{name}_raw.npz"
        cp = Path(args.vipe) / "cams" / f"{name}.npz"
        dp = Path(args.vipe) / "dense" / f"{name}.npz"
        if out.exists() or not (rp.exists() and cp.exists() and dp.exists()):
            continue
        raw, dense = np.load(rp), dict(np.load(dp))
        if args.seg_dir:  # the segmenter's people + sky (clean_render.py seg) on top of ViPE's dynamic mask
            from bench.scene_recon.fuse_render import seg_union

            dense["dyn"] = seg_union(dense["dyn"], args.seg_dir, name)
        T = np.load(cp)["T_world_cam"].astype(np.float64)[: len(raw["frames"])].copy()
        n = len(T)
        valid = raw["valid"].astype(bool)[:n]
        fix, est = scale_fix(args, name, T)
        bs = fix  # the body's scale (default: the whole world, body included, x fix)
        if args.scale == "ruler":  # the world (camera + scene) to FAB's metric body; the body as estimated
            rho, n_rho = human_ruler_raw(body, raw, dense, n, valid)
            est.update(ruler=rho, ruler_frames=n_rho)
            if np.isfinite(rho):
                fix, bs = rho, 1.0
        T[:, :3, 3] *= fix
        # camera-frame SMPL with the clip's median shape, each frame's pelvis kept; whole-world scale on the person
        betas = np.median(raw["betas"][valid], 0).astype(np.float32)
        q = Rotation.from_rotvec(raw["pose_c"][:n].reshape(-1, 3)).as_quat().reshape(n, 24, 4)
        q = smooth_quats(q, 0)
        q = fill_invalid(q, valid)
        tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device="cuda")  # noqa: E731
        with torch.no_grad():
            _, jc = body(tt(raw["pose_c"][:n, :3]), tt(raw["pose_c"][:n, 3:]), tt(raw["betas"][:n]), tt(raw["trans_c"][:n]))
            _, j0 = body(torch.zeros(1, 3, device="cuda"), torch.zeros(1, 69, device="cuda"), tt(betas)[None], None)
        pelvis_c = fill_invalid(jc[:, 0].cpu().numpy().astype(np.float64), valid) * bs
        J0 = j0[0, 0].cpu().numpy().astype(np.float64)
        # gravity-aligned world: GeoCalib up -> +z, the first camera's horizontal view direction -> +x
        up = None
        gp = Path(args.geocalib) / f"{name}.npz" if args.geocalib else None
        if gp is not None and gp.exists():
            up = geocalib_up(T, np.load(gp), np.arange(n))[0]
        Rg = Rotation.from_quat(q[:, 0]).as_matrix()
        if up is None:  # fall back to the mean body up (pelvis -> spine) direction in the camera world
            up = (T[:, :3, :3] @ Rg @ np.array([0, 1.0, 0])).mean(0)
            up /= np.linalg.norm(up)
        A = rot_between(up, np.array([0.0, 0, 1]))
        fwd = A @ T[0, :3, 2]
        A = Rotation.from_euler("z", -np.arctan2(fwd[1], fwd[0])).as_matrix() @ A
        Tw = T.copy()
        Tw[:, :3, :3] = A @ T[:, :3, :3]
        Tw[:, :3, 3] = T[:, :3, 3] @ A.T
        pel_w = np.einsum("nij,nj->ni", Tw[:, :3, :3], pelvis_c) + Tw[:, :3, 3]
        origin = np.array([pel_w[0, 0], pel_w[0, 1], 0.0])
        Tw[:, :3, 3] -= origin
        # smoothing in the world (offline, zero phase): rotations, pelvis
        q[:, 0] = Rotation.from_matrix(Tw[:, :3, :3] @ Rotation.from_quat(q[:, 0]).as_matrix()).as_quat()
        q = np.concatenate([smooth_quats(q[:, :1], args.sigma_rot), smooth_quats(q[:, 1:], args.sigma_rot)], 1)
        R_loc = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(n, 24, 3, 3)
        pel_w_s = gaussian_filter1d(pel_w - origin, args.sigma_pos, axis=0, mode="nearest")  # world, smoothed
        pel_c_s = np.einsum("nji,nj->ni", Tw[:, :3, :3], pel_w_s - Tw[:, :3, 3])  # back to each camera frame
        # camera-frame vertices of the smoothed motion (for the masks and the ray-wise refinement)
        Rc = R_loc.copy()
        Rc[:, 0] = np.einsum("nji,njk->nik", Tw[:, :3, :3], R_loc[:, 0])  # world -> camera global orientation
        with torch.no_grad():
            vc, jcs = [], []
            for i in range(0, n, 128):
                v, j = body(torch.zeros(len(Rc[i:i + 128]), 3, device="cuda"), torch.zeros(len(Rc[i:i + 128]), 69, device="cuda"),
                            tt(betas)[None].expand(len(Rc[i:i + 128]), -1), None, rotmats=tt(Rc[i:i + 128]))
                vc.append(v.cpu().numpy())
                jcs.append(j.cpu().numpy())
        vc = np.concatenate(vc).astype(np.float64)
        jcs = np.concatenate(jcs).astype(np.float64)
        Vc = (vc - jcs[:, :1]) * bs + pel_c_s[:, None]  # camera frame; whole-world scale: body size and depth x fix
        # scene: TSDF of ViPE's depth (x fix) with dynamics and our body masked
        hw = dense["depth"].shape[1:]
        pm = person_masks_from_mesh(Vc[: n], faces, dense["K"][np.clip(np.arange(n), 0, len(dense["K"]) - 1)], hw)
        dk = float(json.loads(Path(args.depth_scale_json).read_text()).get(name, 1.0)) if args.depth_scale_json else 1.0
        uniform = args.tsdf_zpow == 0 and not ("mvs_mask" in dense and dense["mvs_mask"].any())
        if args.tsdf_weight == "quality" and not uniform:  # MVS-verified first, then near (tsdf_scene_quality)
            mesh = tsdf_scene_quality(dense, Tw, fix * dk, pm, args.voxel, args.trunc, args.max_depth, args.tsdf_stride,
                                      fallback_w=args.fallback_weight, z_pow=args.tsdf_zpow)
        elif args.tsdf_weight == "quality":  # every weight equal (no MVS pixel, no range term): open3d's TSDF is the same sum
            mesh = tsdf_scene(dense, Tw, fix * dk, pm, args.voxel, args.trunc, args.max_depth, args.tsdf_stride)
        else:
            mesh = tsdf_scene(dense, Tw, fix * dk, pm, args.voxel, args.trunc, args.max_depth, args.tsdf_stride)  # depth x dk only
        tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
        Vw = np.einsum("nij,nvj->nvi", Tw[:, :3, :3], Vc) + Tw[:, None, :3, 3]
        if args.vis_carve:  # no scene in front of the visible person (fuse_render.human_visibility_carve), before the hfield
            from bench.scene_recon.fuse_render import foot_face_mask, human_visibility_carve

            keep = human_visibility_carve(np.asarray(mesh.vertices), Vw, Tw, dense, faces, args.tsdf_stride * 2,
                                          foot_faces=foot_face_mask(body))
            mesh.remove_vertices_by_mask(~keep)
            tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
            est["vis_carved_vertices"] = int((~keep).sum())
        pel_now = np.einsum("jv,nvc->nc", body.J_regressor[:1].cpu().numpy().astype(np.float64), Vw)
        hf = heightfield(mesh, pel_now[:, :2], pel_now[:, 2], args.hf_res, args.hf_margin, args.hf_above)
        terr = Terrain(hf)
        # foot contacts from velocity alone (the terrain is not trusted yet): lowest sole vertex slow
        sole = [Vw[:, idx] for idx in foot_idx]
        speed = [np.r_[0, np.linalg.norm(np.diff(s[np.arange(n), s[..., 2].argmin(1)], axis=0), axis=-1) * fps] for s in sole]
        contact0 = np.stack([sp < args.v_contact for sp in speed], 1)
        if args.refine:
            e, dz, rep = refine_contacts(Vc, Tw[:, :3, :3], Tw[:, :3, 3], terr, contact0, foot_idx, low_idx, fps)
        else:  # --refine 0: the image-consistent body as estimated (no ray-wise depth factors, no vertical offset)
            e, dz, rep = np.zeros(n), 0.0, {"e_abs_mean": 0.0, "before": {"contact": float("nan")},
                                            "after": {"contact": float("nan")}}
        Vc_r = Vc * (1 + e)[:, None, None]
        Vw = np.einsum("nij,nvj->nvi", Tw[:, :3, :3], Vc_r) + Tw[:, None, :3, 3] + np.array([0, 0, dz])
        # floor at the start: z = 0 under the first frame's feet (shift world + scene + terrain)
        fi = np.concatenate(foot_idx)
        z0 = float(np.median(terr(torch.as_tensor(Vw[0, fi, :2], dtype=torch.float32, device="cuda")).cpu().numpy()))
        Vw[..., 2] -= z0
        Tw[:, 2, 3] -= z0
        tm.vertices[:, 2] -= z0
        hf["height"] = hf["height"] - z0
        Jw = np.einsum("jv,nvc->njc", body.J_regressor.cpu().numpy().astype(np.float64), Vw)
        trans = Jw[:, 0] - J0
        pose = Rotation.from_matrix(R_loc.reshape(-1, 3, 3)).as_rotvec().reshape(n, 72)
        terr = Terrain(hf)
        Vt = torch.as_tensor(Vw, dtype=torch.float32, device="cuda")
        Vrel = Vw.copy()
        Vrel[..., 2] -= terr(Vt[..., :2]).cpu().numpy()  # heights above the terrain -> human_contact_labels
        labels = human_contact_labels(Vrel, part, fps, args.h_contact, args.v_contact)
        rep["penetration_mm_p95"] = float(np.percentile(np.clip(-Vrel[:, low_idx, 2].min(1), 0, None), 95) * 1000)
        rep["foot_gap_mm_contact"] = {g: float(np.abs(Vrel[:, foot_idx[k], 2].min(1)[labels[g]]).mean() * 1000)
                                      if labels[g].any() else float("nan") for k, g in enumerate(("lfoot", "rfoot"))}
        tm.export(od / f"{name}_scene.ply")
        np.savez(od / f"{name}_hfield.npz", **hf)
        np.savez(out, fps=fps, frames=raw["frames"][:n], valid=valid, pose=pose.astype(np.float32), betas=betas,
                 trans=trans.astype(np.float32), joints=Jw.astype(np.float32), verts=Vw.astype(np.float16),
                 contacts=json.dumps({g: v.astype(int).tolist() for g, v in labels.items()}),
                 T_world_cam=Tw, K=raw["K"], scale_fix=fix, scale_est=json.dumps(est), up_world=up, R_align=A,
                 origin=origin, z0=z0, refine=json.dumps(rep), depth_factor=e, depth_scale=dk)
        print(f"[world] {name}: {n} frames, scale_fix {fix:.3f} {est}, scene {len(tm.faces)} faces, hfield "
              f"{hf['height'].shape} (valid {hf['valid'].mean():.2f}), refine |e| {rep['e_abs_mean']:.3f} dz "
              f"{dz:.3f}, contact {rep['before']['contact']:.4f} -> {rep['after']['contact']:.4f}, pen p95 "
              f"{rep['penetration_mm_p95']:.0f} mm", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["fab", "world"])
    p.add_argument("--list", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--vipe", required=True, help="bench/world_vipe.py --depth output (UniDepth prior)")
    p.add_argument("--vipe_m3d", default="", help="the Metric3D ViPE run (pose only) for the bridge scale")
    p.add_argument("--geocalib", default="")
    p.add_argument("--f_own", type=float, default=1.0, help="a factor on ViPE's own scale (default 1: none; no calibration "
                   "factors fitted on data)")
    p.add_argument("--f_bridge", type=float, default=1.0, help="with --vipe_m3d only")
    p.add_argument("--fab_ckpt", default=str(paths.model_file("fab-l")), help="default: the released FAB-L model")
    p.add_argument("--fitter", default=str(paths.shared("smpl_fitter.pt")), help="default: the released SMPL fitter")
    p.add_argument("--detector", default="deployed", help="default: the released person detector (FABPredictor's)")
    p.add_argument("--predictor_kw", default="", help="fab: FABPredictor kwargs (JSON), e.g. the released SIMPLE_PROFILE's")
    p.add_argument("--subject_boxes", default="",
                   help="fab: dir of <clip>.npz box (N, 4) xyxy of the annotated target person (NaN: unknown); the subject is "
                        "the detection overlapping it (identity only). Empty: the most prominent tracked person")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--sigma_rot", type=float, default=2.0)
    p.add_argument("--sigma_pos", type=float, default=3.0)
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--trunc", type=float, default=0.12)
    p.add_argument("--max_depth", type=float, default=8.0)
    p.add_argument("--tsdf_stride", type=int, default=1)
    p.add_argument("--tsdf_weight", default="count", choices=("count", "quality"),
                   help="world: count = open3d's TSDF (every observation 1); quality = q / z^2 (tsdf_scene_quality)")
    p.add_argument("--fallback_weight", type=float, default=0.1,
                   help="world, --tsdf_weight quality: q of the pixels without MVS (dense mvs_mask False)")
    p.add_argument("--tsdf_zpow", type=float, default=2.0,
                   help="world, --tsdf_weight quality: the range term 1 / max(z, 0.3)^zpow (0 = MVS preference only)")
    p.add_argument("--hf_res", type=float, default=0.04)
    p.add_argument("--hf_margin", type=float, default=2.0)
    p.add_argument("--hf_above", type=float, default=0.1,
                   help="heightfield: the highest surface below the nearest path sample's pelvis + this (m)")
    p.add_argument("--h_contact", type=float, default=0.04)
    p.add_argument("--v_contact", type=float, default=0.5)
    p.add_argument("--vis_carve", type=int, default=0, help="world: 1 = human-visibility carving of the TSDF mesh")
    p.add_argument("--seg_dir", default="", help="world: segmenter people + sky masks (clean_render.py seg) added to ViPE's dyn")
    p.add_argument("--depth_scale_json", default="",
                   help="world: {clip: k} the scene depth x k about each camera, the trajectory unchanged; "
                        "saved as depth_scale")
    p.add_argument("--scale", default="two_prior", choices=("two_prior", "ruler"),
                   help="world: two_prior (scale_fix, body x it too) or ruler (world x human_ruler_raw, body x 1)")
    p.add_argument("--refine", type=int, default=0, help="world: 1 = the simple contact refinement (default 0: the "
                   "human-scene step is bench.scene_recon.joint)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()
    {"fab": stage_fab, "world": stage_world}[args.stage](args)


if __name__ == "__main__":
    main()
