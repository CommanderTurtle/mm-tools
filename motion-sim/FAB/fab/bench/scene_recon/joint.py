"""Joint human-scene optimisation (JHSO), one constant algorithm for every
video, on top of bench/videomimic/recon.py world (--refine 0) and its scene (TSDF mesh, optionally visibility-carved):

  variables  a = the log of the body / scene relative scale (the scene and the camera trajectory are scaled by s = exp(-a)
             around the world origin; FAB's metric body keeps its size), b_t = a per-frame log depth factor of the body's
             pelvis along its camera ray (the body keeps its size; smooth, small)
  terms      contact      on contact frames (probability >= --p_contact) the channel's vertices (fab/contact_regions
                          smpl_regions: heels / toes, hands, knees) at the scene: unsigned distance to the scene mesh
                          (a grid around the path) and height above the terrain (Geman-McClure, sigma --sigma)
             penetration  every sampled body vertex above the terrain (recon.heightfield: the highest surface below the
                          local pelvis + 0.3 m), squared hinge
             ruler        the body's depth vs the scene depth map (ViPE dense) on the visible person (dynamic mask x body
                          silhouette): the per-frame robust log ratio (the human as the metric ruler, a data term)
             temporal     root acceleration, b smoothness, b prior, foot skate on consecutive static contacts
  output     the recon npz with the world rescaled (cameras, body placed by exp(b_t) along the rays), the scene mesh x s,
             a new heightfield and contacts -> <out>/recon/<clip>.npz (+ _scene.ply, _hfield.npz, _joint.json)
  options (--ground_q 1 --pen_mode near --depth_stage 1 --foot_lock 1 --snap 1 --snap_max 0.1
             --snap_clamp 1): contacts vote only on well-observed level ground and a frame's penetration counts like a
             contact's gap (the scale); then b_t from the supported frames only; then the heel / toe lock with a leg refit;
             then the bounded local height offsets of the walked surface (snap, capped)

    python -m bench.scene_recon.joint --recon <world out>/recon --vipe <vipe out> --list clips.json \
        --out <dir> [--contacts <contacts dir>] [--only <clip>]      (scripts/fab_world.sh: --snap 0)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fab import paths

CH_FOOT = (0, 1, 2, 3)  # l_heel, l_toe, r_heel, r_toe (fab/contact_regions.CONTACT_NAMES)


def contact_probs(clip, n, r, a):
    """(T, 8) "any" (touching a surface) and (T, 8) "static" (planted) contact probabilities per CONTACT_NAMES channel:
    the contact files (--contacts) > the FAB raw's contact head (8 static, or 8 static + 8 any) > the recon's velocity labels."""
    cp = Path(a.contacts) / f"{clip}__0.contact.npz" if a.contacts else None
    if cp is not None and cp.exists():
        z = np.load(cp)
        Pa, Ps = np.zeros((n, 8), np.float32), np.zeros((n, 8), np.float32)
        fr = z["frame"].astype(int)
        ok = fr < n
        head = next(h for h in ("video_", "image_", "") if f"{h}any_prob" in z.files)  # the video model first
        Pa[fr[ok], :4], Ps[fr[ok], :4] = z[f"{head}any_prob"][ok], z[f"{head}static_prob"][ok]
        for dst, k in ((Pa, f"{head}hk_any_prob"), (Ps, f"{head}hk_static_prob"), (Pa, "hands_knees_any_prob"),
                       (Ps, "hands_knees_static_prob")):
            if k in z.files:
                dst[fr[ok], 4:] = z[k][ok]
        return Pa, Ps, f"contacts:{head or 'any'}:{cp.name}"
    fab = Path(a.recon).parent / "fab" / f"{clip}_raw.npz"
    if fab.exists():
        z = np.load(fab)
        if "contact" in z.files and np.isfinite(z["contact"]).all():
            c = z["contact"][:n].astype(np.float32)
            return (c[:, 8:16] if c.shape[1] >= 16 else c[:, :8]), c[:, :8], "fab_raw"
    lab = json.loads(str(r["contacts"]))
    P = np.zeros((n, 8), np.float32)
    for k, g in ((0, "lfoot"), (1, "lfoot"), (2, "rfoot"), (3, "rfoot")):
        if g in lab:
            P[:, k] = np.asarray(lab[g], np.float32)[:n]
    return P, P, "recon_velocity"


def udf_grid(mesh, lo, hi, vox):
    """Unsigned distance to the scene mesh on a grid (open3d raycasting) -> (grid tensor (nz, ny, nx), lo, vox)."""
    import open3d as o3d

    n = np.ceil((hi - lo) / vox).astype(int) + 1
    gx, gy, gz = (lo[i] + vox * np.arange(n[i]) for i in range(3))
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    d = np.concatenate([sc.compute_distance(o3d.core.Tensor(P[i: i + 4_000_000])).numpy() for i in range(0, len(P), 4_000_000)])
    return torch.as_tensor(d.reshape(n[2], n[1], n[0]), device="cuda"), lo, vox


def grid_lookup(G, lo, vox, X):
    """Trilinear lookup of G (nz, ny, nx) at X (..., 3) (clamped to the grid)."""
    nz, ny, nx = G.shape
    q = (X - torch.as_tensor(lo, device=X.device, dtype=X.dtype)) / vox
    sz = torch.tensor([nx - 1.001, ny - 1.001, nz - 1.001], device=X.device, dtype=X.dtype)
    q = torch.minimum(q.clamp(min=0), sz)
    i0 = q.floor().long()
    f = q - i0
    out = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
                     * (f[..., 2] if dz else 1 - f[..., 2]))
                out = out + w * G[i0[..., 2] + dz, i0[..., 1] + dy, i0[..., 0] + dx]
    return out


def ruler_ratios(Vc, faces, dense, n, stride):
    """Per frame: median log(body depth / scene depth) on the visible person (dense dynamic mask x body silhouette), nan
    where too few pixels. Vc (T, V, 3) camera frame."""
    from bench.scene_recon.fuse_render import body_zbuffer

    out = np.full(n, np.nan)
    h, w = dense["depth"].shape[1:]
    for k in range(0, len(dense["frames"]), stride):
        f = int(dense["frames"][k])
        if f >= n:
            continue
        zb = body_zbuffer(Vc[f], faces, np.eye(4), dense["K"][k].astype(np.float64), h, w)
        d = dense["depth"][k].astype(np.float64)
        m = np.isfinite(zb) & dense["dyn"][k] & np.isfinite(d) & (d > 0.1)
        if m.sum() >= 30:
            out[f] = float(np.median(np.log(zb[m] / d[m])))
    return out


def footprints(V, P, regions, p_contact, min_run=3):
    """Contact footprints of the (final, metric) body V (T, 6890, 3): per foot channel and run of >= min_run consecutive
    contact frames, the median over the run of the channel's lowest vertex -> (k, 3)."""
    out = []
    for ch in CH_FOOT:
        on = P[:, ch] >= p_contact
        low = V[:, regions[ch]]
        low = low[np.arange(len(V)), low[..., 2].argmin(1)]
        t = 0
        while t < len(on):
            if not on[t]:
                t += 1
                continue
            e = t
            while e < len(on) and on[e]:
                e += 1
            if e - t >= min_run:
                out.append(np.median(low[t:e], 0))
            t = e
    return np.array(out).reshape(-1, 3)


def snap_scene(X, fp, surf_h, sigma, max_d, band):
    """Contact snapping of the scene: points X (n, 3) near a footprint's top surface move vertically by the footprint's
    residual (footprint z - the surface height under it, |.| <= max_d), blended with a horizontal Gaussian (sigma); only
    points within +-band of the surface height there (the walked-on surface, not walls / rails). surf_h: xy -> height."""
    from scipy.spatial import cKDTree

    if not len(fp):
        return X, 0
    res = fp[:, 2] - surf_h(fp[:, :2])
    ok = np.isfinite(res) & (np.abs(res) <= max_d)
    fp, res = fp[ok], res[ok]
    if not len(fp):
        return X, 0
    tree = cKDTree(fp[:, :2])
    idx = tree.query_ball_point(X[:, :2], 3 * sigma)
    X = X.copy()
    hx = surf_h(X[:, :2])
    moved = 0
    for i, js in enumerate(idx):
        if not js or not np.isfinite(hx[i]) or abs(X[i, 2] - hx[i]) > band:
            continue
        d2 = ((X[i, :2] - fp[js, :2]) ** 2).sum(1)
        w = np.exp(-d2 / (2 * sigma ** 2))
        X[i, 2] += float((w * res[js]).sum() / w.sum()) * float(w.max())
        moved += 1
    return X, moved


LEG = (1, 2, 4, 5, 7, 8, 10, 11)  # SMPL hips, knees, ankles, feet


def smpl_sub(body, Rm, betas, idx):
    """SMPL vertices idx (T, k, 3) of local joint rotations Rm (T, 24, 3, 3) (global orientation first) and one shape,
    without translation (the pelvis at its rest position)."""
    T = len(Rm)
    vs = body.v_template + torch.einsum("l,vcl->vc", betas, body.shapedirs)
    J = body.J_regressor @ vs
    pf = (Rm[:, 1:] - torch.eye(3, device=Rm.device)).reshape(T, -1)
    vp = vs[idx] + (pf @ body.posedirs.reshape(-1, 6890, 3)[:, idx].reshape(pf.shape[1], -1)).view(T, len(idx), 3)
    rel = J.clone()
    rel[1:] -= J[body.parents[1:]]
    tf = [body._rt(Rm[:, 0], rel[0].expand(T, 3))]
    for i in range(1, 24):
        tf.append(tf[body.parents[i]] @ body._rt(Rm[:, i], rel[i].expand(T, 3)))
    tf = torch.stack(tf, 1)
    Jh = torch.cat([J, torch.zeros_like(J[:, :1])], -1)[None, :, :, None]
    tf = torch.cat([tf[..., :3], tf[..., 3:] - tf @ Jh], -1)
    sk = torch.einsum("vj,bjrc->bvrc", body.weights[idx], tf)
    return (sk[..., :3, :3] @ vp[..., None])[..., 0] + sk[..., :3, 3]


def lock_offsets(F, on, blend, min_len=3, max_len=0, max_off=0.0, fade="lin"):
    """F (T, G, ...) per foot group its centre (..., 3) or its vertices (..., k, 3); on (T, G) lock frames -> offsets O like F:
    over each run of >= min_len lock frames (cut into pieces of <= max_len frames, and where the foot has moved more than
    2 max_off from the piece's start, when > 0) the group is pinned at its run median; the offset fades in before / out after
    the run over `blend` frames (linear or cosine)."""
    T, ng = on.shape
    O, wt = np.zeros_like(F), np.zeros((T, ng))
    runs = 0
    for g in range(ng):
        t = 0
        while t < T:
            if not on[t, g]:
                t += 1
                continue
            s0 = t
            while t < T and on[t, g]:
                t += 1
            if t - s0 < min_len:
                continue
            step = max_len if max_len > 0 else t - s0
            cuts = []
            for s in range(s0, t, step):
                e = min(s + step, t)
                while max_off > 0 and s < e:  # drift: a new pin once the foot is 2 max_off from the piece's start
                    c = F[s:e, g].reshape(e - s, -1, 3).mean(1)
                    far = np.flatnonzero(np.linalg.norm(c - c[0], axis=1) > 2 * max_off)
                    if not len(far):
                        break
                    cuts.append((s, s + far[0]))
                    s += far[0]
                cuts.append((s, e))
            for s, e in cuts:
                if e <= s:
                    continue
                runs += 1
                O[s:e, g] = np.median(F[s:e, g], 0) - F[s:e, g]
                wt[s:e, g] = 1
            for k in range(1, blend + 1):
                f = 1 - k / (blend + 1) if fade == "lin" else 0.5 * (1 + np.cos(np.pi * k / (blend + 1)))
                for tt, src in ((t - 1 + k, t - 1), (s0 - k, s0)):
                    if 0 <= tt < T and wt[tt, g] < f:
                        O[tt, g], wt[tt, g] = O[src, g] * f, f
    return O, runs


def foot_lock_refit(body, pose, betas, pel, scl, regions, Pst, fps, args):
    """The world pipeline's per-vertex foot lock (the world evaluation's foot lock: heel and toe groups pinned over their static
    contact runs, faded over --lock_blend frames) as targets, then a refit of the SMPL legs: per frame increments of the hip /
    knee / ankle / foot rotations and a small smooth pelvis shift (one shape) that put the heel / toe group centres on their
    targets (the unlocked groups stay where they were). pose (T, 72) world, pel (T, 3) world pelvis, scl the body scale.
    -> pose (T, 72), pelvis shift (T, 3), report."""
    from scipy.spatial.transform import Rotation

    from fab.gn_fit import so3_exp

    dev = "cuda"
    T = len(pose)
    groups = [np.asarray(regions[ch]) for ch in CH_FOOT]
    idx = np.concatenate(groups)
    seg = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
    M = torch.zeros(len(groups), len(idx), device=dev)
    M[torch.as_tensor(seg, device=dev), torch.arange(len(idx), device=dev)] = 1
    M = M / M.sum(1, keepdim=True)
    Rm0 = torch.as_tensor(Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3), dtype=torch.float32,
                          device=dev)
    bt = torch.as_tensor(betas, dtype=torch.float32, device=dev)
    it = torch.as_tensor(idx, device=dev)
    j0 = (body.J_regressor[0] @ (body.v_template + torch.einsum("l,vcl->vc", bt, body.shapedirs)))
    pl = torch.as_tensor(pel, dtype=torch.float32, device=dev)

    def verts(Rm, dp):
        return (smpl_sub(body, Rm, bt, it) - j0) * scl + pl[:, None] + dp[:, None]

    def centres(Rm, dp):
        return torch.einsum("gk,tkc->tgc", M, verts(Rm, dp))

    with torch.no_grad():
        X0 = verts(Rm0, torch.zeros(T, 3, device=dev)).cpu().numpy().astype(np.float64)
    F = np.einsum("gk,tkc->tgc", M.cpu().numpy(), X0)
    sp = np.r_[np.zeros((1, len(groups))), np.linalg.norm(np.diff(F, axis=0), axis=-1)] * fps  # m / s
    on = (Pst[:T, list(CH_FOOT)] >= args.p_contact) & (sp < args.lock_gate)
    ml = int(round(args.lock_max_s * fps))
    if args.lock_rigid:  # every vertex of the group pinned (its orientation too), not only the centre
        kk = np.cumsum([0] + [len(g) for g in groups])
        Og, runs = zip(*[lock_offsets(X0[:, kk[g]:kk[g + 1]][:, None], on[:, g:g + 1], args.lock_blend, max_len=ml,
                                      max_off=args.lock_max_off, fade=args.lock_fade) for g in range(len(groups))])
        runs = sum(runs)
        Ov = np.concatenate([o[:, 0] for o in Og], 1)
        tgt = torch.as_tensor(X0 + Ov, dtype=torch.float32, device=dev)
        O = np.einsum("gk,tkc->tgc", M.cpu().numpy(), Ov)
    else:
        O, runs = lock_offsets(F, on, args.lock_blend, max_len=ml, max_off=args.lock_max_off, fade=args.lock_fade)
        tgt = torch.as_tensor(F + O, dtype=torch.float32, device=dev)
    fit = verts if args.lock_rigid else centres
    li = list(LEG)
    d = torch.zeros(T, len(li), 3, device=dev, requires_grad=True)
    dp = torch.zeros(T, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [d], "lr": 0.01}, {"params": [dp], "lr": 0.002}])

    def rot(d):
        Rm = Rm0.clone()
        Rm[:, li] = Rm0[:, li] @ so3_exp(d)
        return Rm

    for _ in range(args.lock_iters):
        C = fit(rot(d), dp)
        L = (((C - tgt) ** 2).sum(-1).mean() / args.lock_sigma ** 2 + 0.01 * (d ** 2).sum(-1).mean() / 0.3 ** 2
             + ((d[1:] - d[:-1]) ** 2).sum(-1).mean() / 0.05 ** 2 + (dp ** 2).sum(-1).mean() / args.lock_sigma_root ** 2
             + ((dp[1:] - dp[:-1]) ** 2).sum(-1).mean() / 0.005 ** 2
             + args.lock_acc * (((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).sum(-1).mean() / 0.005 ** 2
                                + ((dp[2:] - 2 * dp[1:-1] + dp[:-2]) ** 2).sum(-1).mean() / 0.001 ** 2))
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
    with torch.no_grad():
        C = fit(rot(d), dp)
        res = (C - tgt).norm(dim=-1).cpu().numpy()
        Rm = rot(d).cpu().numpy()
    lk = on.any(1)
    st_all = Pst[:T, list(CH_FOOT)] >= args.p_contact
    pose2 = Rotation.from_matrix(Rm.reshape(-1, 3, 3)).as_rotvec().reshape(T, 72).astype(np.float32)
    off = np.linalg.norm(O, axis=-1)
    rep = {"lock_runs": runs, "lock_frame_share": float(lk.mean()), "static_locked_share": float(on.sum() / max(st_all.sum(), 1)), "offset_mm_p50": float(np.median(off[off > 0]) * 1000)
           if (off > 0).any() else 0.0, "offset_mm_max": float(off.max() * 1000), "refit_residual_mm_mean": float(res.mean() * 1000),
           "refit_residual_mm_p95": float(np.percentile(res, 95) * 1000),
           "pelvis_shift_mm_mean": float(dp.detach().norm(dim=-1).mean() * 1000),
           "leg_rot_deg_mean": float(np.degrees(d.detach().norm(dim=-1).mean().item()))}
    return pose2, dp.detach().cpu().numpy().astype(np.float64), rep


def gm(x, s):
    return x ** 2 / (x ** 2 + s ** 2)


def ground_quality(hf, slope_max, rough_max, win=5, min_valid=0.8):
    """Per heightfield cell 1 where the scene is well-observed level ground: >= min_valid of the win x win cells around it
    hit by the downward rays (observed, not filled in), local slope <= slope_max (rise / run) and height spread <= rough_max
    (scene units) over the window (stair edges, smeared / blobby stairs and ramps fail; flat floors and stair treads pass)."""
    from scipy.ndimage import uniform_filter

    H = hf["height"].astype(np.float64)
    vf = uniform_filter(hf["valid"].astype(np.float64), win, mode="nearest")
    gy, gx = np.gradient(uniform_filter(H, 3, mode="nearest"), hf["res"])
    m, m2 = uniform_filter(H, win, mode="nearest"), uniform_filter(H * H, win, mode="nearest")
    rough = np.sqrt(np.clip(m2 - m * m, 0, None))
    return ((vf >= min_valid) & (np.hypot(gx, gy) <= slope_max) & (rough <= rough_max)).astype(np.float32)


def optimise(Vc_sub, Pc, Vch, R, c, terr, udf, P, Pst, rul, a_init, args, sub_inview=None, sigma_a=1.0, qterr=None,
             stage="scale", b_frames=None, skate_P=None):
    """Vc_sub (T, Ns, 3), Pc (T, 3) pelvis, Vch list of (T, k, 3) channel vertices, all camera frame; R (T, 3, 3), c (T, 3)
    unit-scale world cameras; P (T, 8) contact probabilities; rul (T,) log ratios. -> a, b (T,), report.
    qterr: ground-quality lookup (ground_quality as a Terrain): contacts count only on well-observed level ground.
    stage "depth": the scale a stays a_init and only b_t moves, with data (contact / penetration / skate) on b_frames only
    (the supported frames) and smoothness / prior elsewhere; no ruler. skate_P: the static contacts of the skate term
    (default Pst; then not limited to b_frames)."""
    dev = "cuda"
    T = len(Vc_sub)
    depth = stage == "depth"
    a = torch.tensor(float(a_init), device=dev, requires_grad=not depth)
    b = torch.zeros(T, device=dev, requires_grad=bool(args.free_b) or depth)
    opt = torch.optim.Adam([b] if depth else ([a, b] if args.free_b else [a]), lr=args.lr)
    bfun = lambda: b  # noqa: E731
    if depth and args.b_knot > 0:  # a low-frequency b: linear between knots every --b_knot frames
        tk = torch.arange(T, device=dev, dtype=torch.float32) / args.b_knot
        i0 = tk.floor().long()
        fk = tk - i0
        bk = torch.zeros(int(i0.max()) + 2, device=dev, requires_grad=True)
        bfun = lambda: bk[i0] * (1 - fk) + bk[i0 + 1] * fk  # noqa: E731
        opt = torch.optim.Adam([bk], lr=args.lr)
    on = torch.as_tensor(P >= args.p_contact, device=dev)
    on_st = torch.as_tensor(Pst >= args.p_contact, device=dev)
    rul_t = torch.as_tensor(np.nan_to_num(rul), device=dev, dtype=torch.float32)
    rul_ok = torch.as_tensor(np.isfinite(rul), device=dev)
    Rt, ct = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (R, c))
    Vs, Pp = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (Vc_sub, Pc))
    Vk = [torch.as_tensor(x, device=dev, dtype=torch.float32) for x in Vch]
    static = on_st[1:] & on_st[:-1]
    if skate_P is not None:  # (a planted foot supports the body whether or not the scene under it is observed)
        sk_on = torch.as_tensor(skate_P >= args.p_contact, device=dev)
        static = sk_on[1:] & sk_on[:-1]
    sig_sk = args.sigma_skate if args.sigma_skate > 0 else args.sigma
    sv = (torch.as_tensor(sub_inview, device=dev).float() if sub_inview is not None
          else torch.ones(Vs.shape[:2], device=dev))  # penetration only for the body vertices in view
    fm = torch.as_tensor(b_frames, device=dev).float() if b_frames is not None else torch.ones(T, device=dev)

    def world(X, G, e):  # camera frame (T, k, 3): size x G, pelvis depth x G e -> unit world
        Y = (X - Pp[:, None]) * G + Pp[:, None] * (G * e)[:, None, None]
        return torch.einsum("tij,tkj->tki", Rt, Y) + ct[:, None]

    def qw(W):  # 1 where the channel stands on well-observed level ground (no gradient through the mask)
        return 1.0 if qterr is None else (qterr(W[..., :2].mean(1).detach()) >= 0.5).float()

    def pen_term(B, l1=False):
        h = terr(B[..., :2])
        d = torch.relu(h - B[..., 2])
        d = d.clamp(max=1.0) if l1 else d ** 2 / args.sigma ** 2
        if args.pen_q and qterr is not None:  # penetration counts only over well-observed level ground too
            d = d * (qterr(B[..., :2].detach()) >= 0.5).float()
        if args.pen_mode == "near":  # per frame over the in-view vertices near / below the terrain: a frame's sink counts
            nw = sv * ((B[..., 2] - h).detach() < args.pen_near).float()  # as much as a contact frame's gap
            has = (nw.sum(1) > 0).float() * fm
            return (has * (nw * d).sum(1) / nw.sum(1).clamp(min=1)).sum() / has.sum().clamp(min=1)
        w_ = sv * fm[:, None]
        return (w_ * d).sum() / w_.sum().clamp(min=1)

    def losses():
        b = bfun()
        G, e = torch.exp(a), torch.exp(b)
        g = G * e
        L = {}
        B = world(Vs, G, e)
        L["pen"] = pen_term(B)
        con, skate = 0.0, 0.0
        for ch, X in enumerate(Vk):
            if ch not in args.channels:
                continue
            W = world(X, G, e)
            hgt = (W[..., 2] - terr(W[..., :2])).min(1)[0]  # the channel's lowest vertex over the terrain
            dist = grid_lookup(*udf, W).min(1)[0]
            m = on[:, ch].float() * qw(W) * fm
            con = con + (m * (gm(dist, args.sigma) + gm(hgt, args.sigma))).sum() / m.sum().clamp(min=1)
            st = static[:, ch].float() * (1.0 if skate_P is not None else fm[1:] * fm[:-1])
            skate = skate + (st * ((W[1:] - W[:-1]) ** 2).sum(-1).mean(1)).sum() / st.sum().clamp(min=1) / sig_sk ** 2
        L["contact"], L["skate"] = con, skate
        x = a + b + rul_t
        rl = gm(x, args.sigma_ruler) if args.ruler_loss == "gm" else (torch.sqrt(x ** 2 + 0.05 ** 2) - 0.05) / args.sigma_ruler
        L["ruler"] = 0.0 * a if depth else (rul_ok.float() * rl).sum() / rul_ok.float().sum().clamp(min=1)
        root = torch.einsum("tij,tj->ti", Rt, Pp * g[:, None]) + ct
        L["acc"] = ((root[2:] - 2 * root[1:-1] + root[:-2]) ** 2).sum(-1).mean() / args.sigma_acc ** 2
        L["bsmooth"] = ((b[1:] - b[:-1]) ** 2).mean() / args.sigma_b_step ** 2
        L["bprior"] = (b ** 2).mean() / args.sigma_b ** 2
        L["aprior"] = ((a - a_init) / sigma_a) ** 2  # the scale moves only as far as the in-view contact evidence allows
        return L

    wts = {"pen": args.w_pen, "contact": args.w_contact, "skate": args.w_skate, "ruler": args.w_ruler, "acc": args.w_acc,
           "bsmooth": args.w_bsmooth, "bprior": args.w_bprior, "aprior": 1.0}

    with torch.no_grad():
        before = {k: float(v) for k, v in losses().items()}
    grid = {}
    if args.grid and not depth:  # coarse 1-D search of the global log-scale with truncated L1 terms (the robust terms saturate far away)
        with torch.no_grad():
            e1 = torch.ones(T, device=dev)
            for ag in np.linspace(a_init - args.grid_range, a_init + args.grid_range, args.grid):
                G = torch.tensor(float(np.exp(ag)), device=dev)
                B = world(Vs, G, e1)
                pen = pen_term(B, l1=True)
                con, nc = 0.0, 0
                for ch, X in enumerate(Vk):
                    if ch not in args.channels:
                        continue
                    W = world(X, G, e1)
                    hgt = (W[..., 2] - terr(W[..., :2])).min(1)[0].abs().clamp(max=1.0)
                    m = on[:, ch].float() * qw(W)
                    con, nc = con + (m * hgt).sum(), nc + m.sum()
                con = con / nc.clamp(min=1)
                rul = (rul_ok.float() * (ag + rul_t).abs().clamp(max=1.0)).sum() / rul_ok.float().sum().clamp(min=1)
                grid[float(ag)] = float(args.w_pen * pen + args.w_contact * con + args.w_ruler * rul
                                        + ((ag - a_init) / sigma_a) ** 2)
        best = min(grid, key=grid.get)
        a.data.fill_(best)
    for it in range(args.iters):
        L = losses()
        loss = sum(wts[k] * v for k, v in L.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = {k: float(v) for k, v in losses().items()}
    return float(a.detach()), bfun().detach().cpu().numpy(), {"before": before, "after": after,
                                                          "grid_best": (min(grid, key=grid.get) if grid else None)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", required=True, help="recon.py world --refine 0 output dir (recon/)")
    p.add_argument("--vipe", required=True, help="ViPE dir with dense/<clip>.npz")
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--contacts", default="", help="<clip>__0.contact.npz dir (scripts/contact_clips.py; else the FAB raw / recon labels)")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--p_contact", type=float, default=0.5)
    p.add_argument("--contact_inview", type=int, default=1, help="1: contacts only where the contact vertices are in view")
    p.add_argument("--channels", default="0,1,2,3", help="contact channels used (CONTACT_NAMES indices)")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--grid", type=int, default=81, help="coarse global-scale search points (0: off)")
    p.add_argument("--sigma_a_min", type=float, default=0.02, help="log-scale prior width without in-view contact evidence")
    p.add_argument("--sigma_a_max", type=float, default=0.5, help="... with >= --ev_frames frames of it")
    p.add_argument("--ev_frames", type=int, default=150)
    p.add_argument("--grid_range", type=float, default=1.2, help="+- log-scale range of the coarse search")
    p.add_argument("--free_b", type=int, default=0, help="1: per-frame pelvis depth factors b_t too (0: the global scale only)")
    p.add_argument("--body_trust", type=float, default=1.0,
                   help="the absolute scale: 1 = FAB's metric body as is (the scene moves), 0 = the scene's (the body moves)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.05, help="contact / penetration scale (scene units ~ m)")
    p.add_argument("--sigma_ruler", type=float, default=0.15)
    p.add_argument("--ruler_loss", default="l1", choices=("l1", "gm"),
                   help="l1: Charbonnier (robust, keeps pulling far away); gm: Geman-McClure (saturates)")
    p.add_argument("--sigma_acc", type=float, default=0.01)
    p.add_argument("--sigma_b", type=float, default=0.1)
    p.add_argument("--sigma_b_step", type=float, default=0.01)
    p.add_argument("--w_pen", type=float, default=1.0)
    p.add_argument("--w_contact", type=float, default=1.0)
    p.add_argument("--w_skate", type=float, default=0.1)
    p.add_argument("--w_ruler", type=float, default=0.5)
    p.add_argument("--w_acc", type=float, default=0.1)
    p.add_argument("--w_bsmooth", type=float, default=0.1)
    p.add_argument("--w_bprior", type=float, default=0.1)
    p.add_argument("--ruler_stride", type=int, default=5)
    p.add_argument("--udf_vox", type=float, default=0.04)
    p.add_argument("--hf_above", type=float, default=0.3)
    p.add_argument("--snap", type=int, default=0, help="1: contact snapping of the scene mesh to the footprints (snap_scene)")
    p.add_argument("--snap_sigma", type=float, default=0.08)
    p.add_argument("--snap_max", type=float, default=0.3)
    p.add_argument("--snap_band", type=float, default=0.25)
    p.add_argument("--fix_scale_json", default="", help="{clip: s}: the scene / camera scale s fixed (no optimisation)")
    p.add_argument("--snap_clamp", type=int, default=0,
                   help="1: footprints farther than --snap_max from the surface still snap, by --snap_max (counted as capped)")
    p.add_argument("--ground_q", type=int, default=0,
                   help="1: contacts vote (scale, depth) only on well-observed level ground (ground_quality)")
    p.add_argument("--q_slope", type=float, default=0.18, help="ground_quality: max slope (rise / run; 0.18 ~ 10 deg)")
    p.add_argument("--q_rough", type=float, default=0.02, help="ground_quality: max height spread over the window")
    p.add_argument("--q_win", type=int, default=5, help="ground_quality: window (heightfield cells)")
    p.add_argument("--pen_mode", default="all", choices=("all", "near"),
                   help="near: per-frame penetration over the in-view vertices within --pen_near of the terrain")
    p.add_argument("--pen_near", type=float, default=0.1)
    p.add_argument("--pen_q", type=int, default=0, help="1: with --ground_q, penetration only over well-observed level ground")
    p.add_argument("--sigma_skate", type=float, default=0.0, help="skate term scale (m / frame; 0: --sigma)")
    p.add_argument("--w_skate_depth", type=float, default=-1, help="the depth stage's skate weight (< 0: --w_skate)")
    p.add_argument("--b_knot", type=int, default=0, help="depth stage: b_t linear between knots every n frames (0: per frame)")
    p.add_argument("--depth_stage", type=int, default=0,
                   help="1: after the scale, per-frame body depth b_t with data on the supported frames only")
    p.add_argument("--foot_lock", type=int, default=0, help="1: heel / toe lock over static contacts + a leg refit")
    p.add_argument("--lock_gate", type=float, default=1.0, help="foot lock: no lock while the group moves faster (m/s)")
    p.add_argument("--lock_blend", type=int, default=6)
    p.add_argument("--lock_iters", type=int, default=300)
    p.add_argument("--lock_sigma_root", type=float, default=0.05, help="foot lock refit: pelvis shift prior (m)")
    p.add_argument("--lock_sigma", type=float, default=0.005, help="foot lock refit: target scale (m)")
    p.add_argument("--lock_rigid", type=int, default=0, help="foot lock: pin every vertex of the heel / toe group (0: its centre)")
    p.add_argument("--lock_max_s", type=float, default=0.0, help="foot lock: re-pin long runs every n s (0: one pin per run)")
    p.add_argument("--lock_max_off", type=float, default=0.0, help="foot lock: re-pin once a run drifts 2x this far (m; 0: off)")
    p.add_argument("--lock_fade", default="lin", choices=("lin", "cos"))
    p.add_argument("--lock_acc", type=float, default=0.0, help="foot lock refit: weight of the rotation / shift acceleration terms")
    a = p.parse_args()
    a.channels = [int(x) for x in a.channels.split(",")]
    import open3d as o3d
    import trimesh

    from bench.body_models import SMPL
    from bench.scene_recon.fuse_render import smooth_verts
    from bench.videomimic.recon import Terrain, heightfield
    from fab.retarget.contact import human_contact_labels
    from fab.contact_regions import smpl_regions

    body = SMPL(Path(a.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    faces = body.faces.cpu().numpy()
    part = body.weights.argmax(1).cpu().numpy()
    Jreg = body.J_regressor.cpu().numpy().astype(np.float64)
    regions = smpl_regions(a.bm_dir)
    sub = np.arange(0, 6890, 7)
    od = Path(a.out) / "recon"
    od.mkdir(parents=True, exist_ok=True)
    clips = json.loads(Path(a.list).read_text())
    for clip in sorted(clips):
        if a.only and not any(o in clip for o in a.only.split(",")):
            continue
        rp, mp, dp = Path(a.recon) / f"{clip}.npz", Path(a.recon) / f"{clip}_scene.ply", Path(a.vipe) / "dense" / f"{clip}.npz"
        if (od / f"{clip}.npz").exists() or not (rp.exists() and mp.exists() and dp.exists()):
            continue
        r = dict(np.load(rp))
        T = r["T_world_cam"].astype(np.float64)
        Vw = smooth_verts(r, body).astype(np.float64)
        n = min(len(T), len(Vw))
        T, Vw = T[:n], Vw[:n]
        Rw, cw = T[:, :3, :3], T[:, :3, 3]
        Vc = np.einsum("tji,tvj->tvi", Rw, Vw - cw[:, None])  # camera frame (metric FAB body x the recon's scale fix)
        fix = float(r["scale_fix"])
        est = json.loads(str(r["scale_est"])) if "scale_est" in r else {}
        bs = 1.0 if np.isfinite(est.get("ruler", np.nan)) else fix  # the recon's body scale (--scale ruler: 1)
        Vc = Vc / bs  # FAB's own metric body, camera frame
        mesh = o3d.io.read_triangle_mesh(str(mp))
        pel = np.einsum("v,tvc->tc", Jreg[0], Vw)
        hf = heightfield(mesh, pel[:, :2], pel[:, 2], 0.04, 2.0, a.hf_above)
        terr = Terrain(hf)
        lo = np.r_[Vw[..., :2].reshape(-1, 2).min(0) - 1.0, Vw[..., 2].min() - 1.0]
        hi = np.r_[Vw[..., :2].reshape(-1, 2).max(0) + 1.0, Vw[..., 2].max() + 0.5]
        empty = len(mesh.triangles) == 0  # no scene (e.g. the depth failed): the recon's world is kept as is
        udf = None if empty else udf_grid(mesh, lo, hi, a.udf_vox)
        P, Pst, src = contact_probs(clip, n, r, a)
        dense = dict(np.load(dp))
        # contacts count only where the contact vertices are in view (the scene there is observed, not extrapolated)
        h, w = dense["depth"].shape[1:]
        Kd = dense["K"][np.clip(np.searchsorted(dense["frames"], np.arange(n)), 0, len(dense["K"]) - 1)].astype(np.float64)
        inview = np.zeros((n, 8), bool)
        for ch, ix in enumerate(regions):
            X = Vc[:, ix]
            z = np.clip(X[..., 2], 1e-3, None)
            u = X[..., 0] / z * Kd[:, None, 0, 0] + Kd[:, None, 0, 2]
            v = X[..., 1] / z * Kd[:, None, 1, 1] + Kd[:, None, 1, 2]
            inview[:, ch] = ((u >= 0) & (u < w) & (v >= 0) & (v < h) & (X[..., 2] > 0.1)).mean(1) > 0.5
        Xs = Vc[:, sub]
        zs = np.clip(Xs[..., 2], 1e-3, None)
        us = Xs[..., 0] / zs * Kd[:, None, 0, 0] + Kd[:, None, 0, 2]
        vs = Xs[..., 1] / zs * Kd[:, None, 1, 1] + Kd[:, None, 1, 2]
        sub_inview = (us >= 0) & (us < w) & (vs >= 0) & (vs < h) & (Xs[..., 2] > 0.1)
        Pst_all = Pst.copy()  # (the foot lock: every static contact, in view or not)
        if a.contact_inview:
            P, Pst = P * inview, Pst * inview
        else:
            sub_inview[:] = True
        rul = ruler_ratios(Vc, faces, dense, n, a.ruler_stride) - np.log(fix)  # vs the recon world's depth (raw x fix)
        Pc = np.einsum("v,tvc->tc", Jreg[0], Vc)
        a0 = float(np.log(bs))  # the start: the body as the recon placed it
        qterr, qch = None, np.ones((n, 8), bool)
        if a.ground_q and not empty:  # well-observed level ground; a channel's evidence counts only there (at the start)
            qterr = Terrain({**hf, "height": ground_quality(hf, a.q_slope, a.q_rough, a.q_win)})

            def on_q(G):
                out = np.zeros((n, 8), bool)
                for ch, ix in enumerate(regions):
                    Wc = np.einsum("tij,tj->ti", Rw, Vc[:, ix].mean(1) * G) + cw
                    out[:, ch] = qterr(torch.as_tensor(Wc[:, :2], dtype=torch.float32, device="cuda")).cpu().numpy() >= 0.5
                return out

            qch = on_q(np.exp(a0))
        n_ev = int(((P[:, :4] >= a.p_contact) & qch[:, :4]).any(1).sum())  # frames with an in-view foot contact: the scale's evidence
        sigma_a = a.sigma_a_min + (a.sigma_a_max - a.sigma_a_min) * min(1.0, n_ev / a.ev_frames)
        sfix = json.loads(Path(a.fix_scale_json).read_text()).get(clip) if a.fix_scale_json else None
        if sfix is not None:  # the trajectory scale s_t given (--fix_scale_json), no optimisation
            aa, bb, rep = -float(np.log(sfix)) / a.body_trust, np.zeros(n), {"before": {}, "after": {}, "fixed_scale": float(sfix)}
        elif empty:
            aa, bb, rep = a0, np.zeros(n), {"before": {}, "after": {}, "skipped": "empty scene mesh"}
        else:
            aa, bb, rep = optimise(Vc[:, sub], Pc, [Vc[:, ix] for ix in regions], Rw, cw, terr, udf, P, Pst, rul, a0, a,
                                   sub_inview, sigma_a, qterr)
        rep.update(evidence_frames=n_ev, sigma_a=sigma_a)
        if a.depth_stage and not empty:  # per-frame body depth on the supported frames: an in-view static foot contact
            qs = on_q(np.exp(aa)) if qterr is not None else np.ones((n, 8), bool)  # on well-observed ground, at the scale
            sup = ((Pst[:, :4] >= a.p_contact) & qs[:, :4]).any(1)
            ws = a.w_skate
            a.w_skate = a.w_skate_depth if a.w_skate_depth >= 0 else ws
            _, bb, rep2 = optimise(Vc[:, sub], Pc, [Vc[:, ix] for ix in regions], Rw, cw, terr, udf, P, Pst, rul, aa, a,
                                   sub_inview, sigma_a, qterr, stage="depth", b_frames=sup, skate_P=Pst_all)
            a.w_skate = ws
            rep["depth"] = {"supported_frames": int(sup.sum()), "supported_share": float(sup.mean()),
                            "b_p5": float(np.percentile(bb, 5)), "b_p50": float(np.median(bb)), "b_p95": float(np.percentile(bb, 95)),
                            "b_absmax": float(np.abs(bb).max()), "b_abs_mean_supported": float(np.abs(bb[sup]).mean()) if sup.any() else 0.0,
                            "before": rep2["before"], "after": rep2["after"]}
        # the optimisation fixes the body / scene relative scale exp(aa); the absolute one: the scene x exp(x), the body
        # (size and camera-frame depth) x exp(y), y - x = aa, weighted by --body_trust (1: FAB's metric body as is)
        x, y = -a.body_trust * aa, (1 - a.body_trust) * aa
        s = float(np.exp(x))
        g = np.exp(bb)
        T2 = T.copy()
        T2[:, :3, 3] *= s
        V2 = np.einsum("tij,tvj->tvi", Rw, ((Vc - Pc[:, None]) + Pc[:, None] * g[:, None, None]) * np.exp(y)) + T2[:, None, :3, 3]
        J2 = np.einsum("jv,tvc->tjc", Jreg, V2)
        with torch.no_grad():
            _, j0 = body(torch.zeros(1, 3, device="cuda"), torch.zeros(1, 69, device="cuda"),
                         torch.as_tensor(r["betas"], dtype=torch.float32, device="cuda")[None], None)
        trans = J2[:, 0] - j0[0, 0].cpu().numpy().astype(np.float64)
        pose = np.asarray(r["pose"], np.float32)
        if a.foot_lock:  # heel / toe lock over the static contacts, then a leg refit (the body stays SMPL)
            pose, dpel, rep["lock"] = foot_lock_refit(body, pose[:n].astype(np.float64), r["betas"], J2[:, 0], float(np.exp(y)),
                                                      regions, Pst_all, float(r["fps"]), a)
            Vl = []
            with torch.no_grad():
                bt = torch.as_tensor(r["betas"], dtype=torch.float32, device="cuda")
                for i in range(0, n, 256):
                    pi = torch.as_tensor(pose[i: i + 256], device="cuda")
                    v, _ = body(pi[:, :3], pi[:, 3:], bt[None].expand(len(pi), -1), None)
                    Vl.append(v.cpu().numpy().astype(np.float64))
            j0n = j0[0, 0].cpu().numpy().astype(np.float64)
            V2 = (np.concatenate(Vl) - j0n) * np.exp(y) + (J2[:, 0] + dpel)[:, None]
            J2 = np.einsum("jv,tvc->tjc", Jreg, V2)
            trans = J2[:, 0] - j0n
        mesh.scale(s, center=np.zeros(3))
        pel2 = J2[:, 0]
        snap = {"footprints": 0, "moved_vertices": 0}
        if a.snap and not empty:  # contact snapping of the scene to the footprints (the walked-on surface only)
            hf1 = heightfield(mesh, pel2[:, :2], pel2[:, 2], 0.04, 2.0, a.hf_above)
            t1 = Terrain(hf1, device="cpu")
            sh = lambda xy: t1(torch.as_tensor(xy, dtype=torch.float32)).numpy().astype(np.float64)  # noqa: E731
            fp = footprints(V2, Pst, regions, a.p_contact)
            res = fp[:, 2] - sh(fp[:, :2]) if len(fp) else np.zeros(0)
            n_cap = int((np.abs(res) > a.snap_max).sum())
            if a.snap_clamp and len(fp):  # capped footprints snap by the cap (stored so the cloud's re-snap matches)
                cap = a.snap_max * (1 - 1e-4)
                ok = np.isfinite(res)
                fp[ok, 2] = sh(fp[ok, :2]) + np.clip(res[ok], -cap, cap)
            X0 = np.asarray(mesh.vertices).copy()
            Xm, nm = snap_scene(X0, fp, sh, a.snap_sigma, a.snap_max, a.snap_band)
            dh = np.abs(Xm[:, 2] - X0[:, 2])
            dh = dh[dh > 0]
            mesh.vertices = o3d.utility.Vector3dVector(Xm)
            snap = {"footprints": int(len(fp)), "moved_vertices": int(nm), "sigma": a.snap_sigma, "max": a.snap_max,
                    "band": a.snap_band, "clamp": int(a.snap_clamp), "footprints_over_cap": n_cap,
                    "footprint_res_mm_p50": float(np.median(np.abs(res[np.isfinite(res)])) * 1000) if np.isfinite(res).any() else 0.0,
                    "dh_mm_p50": float(np.median(dh) * 1000) if len(dh) else 0.0,
                    "dh_mm_p95": float(np.percentile(dh, 95) * 1000) if len(dh) else 0.0,
                    "dh_mm_max": float(dh.max() * 1000) if len(dh) else 0.0}
            np.savez(od / f"{clip}_snap.npz", footprints=fp, **{f"hf_{k}": v for k, v in hf1.items()})
        hf2 = heightfield(mesh, pel2[:, :2], pel2[:, 2], 0.04, 2.0, 0.1)
        terr2 = Terrain(hf2)
        Vrel = V2.copy()
        Vrel[..., 2] -= terr2(torch.as_tensor(V2[..., :2], dtype=torch.float32, device="cuda")).cpu().numpy()
        labels = human_contact_labels(Vrel, part, float(r["fps"]), 0.05, 0.3)
        o3d.io.write_triangle_mesh(str(od / f"{clip}_scene.ply"), mesh)
        np.savez(od / f"{clip}_hfield.npz", **hf2)
        out = dict(r)
        out.update(T_world_cam=T2, verts=V2.astype(np.float16), joints=J2.astype(np.float32), trans=trans.astype(np.float32), pose=pose,
                   contacts=json.dumps({k: v.astype(int).tolist() for k, v in labels.items()}),
                   scale_fix=float(fix * s), depth_factor=(g * np.exp(y) - 1).astype(np.float32),
                   joint=json.dumps({"s": s, "body_scale": float(np.exp(y)), "a": aa, "b_abs_mean": float(np.abs(bb).mean()),
                                     "contact_src": src, "snap": snap, "contact_frames_inview": float(inview[:, :4].mean()),
                                     "ruler_frames": int(np.isfinite(rul).sum()), **rep}))
        np.savez(od / f"{clip}.npz", **out)
        (od / f"{clip}_joint.json").write_text(json.dumps({"s": s, "body_scale": float(np.exp(y)), "a": aa, "contact_src": src, "snap": snap,
                                                           **rep}, indent=1) + "\n")
        print(f"[joint] {clip}: scene x {s:.3f}, body x {np.exp(y):.3f} (recon fix {fix:.3f}), |b| {np.abs(bb).mean():.3f}, contacts {src}, "
              f"before {json.dumps({k: round(v, 3) for k, v in rep['before'].items()})} after "
              f"{json.dumps({k: round(v, 3) for k, v in rep['after'].items()})}", flush=True)


if __name__ == "__main__":
    main()
