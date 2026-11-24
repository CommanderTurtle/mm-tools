"""Human + scene from a monocular clip ("ORIGINAL VIDEO | HUMAN + SCENE" renders), on top of bench/videomimic/recon.py's
world stage (ViPE camera + metric depth, two-prior scale, GeoCalib
z-up world, FAB body, TSDF scene):

  fuse    the static scene as a coloured point cloud fused over the clip: every --stride-th frame's ViPE metric depth
          (x the recon's scale fix) and RGB at the dense depth's resolution, back-projected with the recon's camera
          (gravity-aligned world); ViPE's dynamic instances, our projected body (dilated) and depth edges (flying
          pixels) masked; voxel hashing (--voxel, colour / position averaged) keeping voxels seen in >= --min_count
          frames -> <out>/<name>_scene_points.ply (+ npz)
  render  per frame, the fused points (z-buffered square splats, size ~ voxel x f / z) + the recon's body (points
          sampled on its mesh, Lambert-shaded blue) from the clip's camera, next to the original frame:
          <out>/<name>_human_scene.mp4; and a third-person orbit (the camera circling the scene centre once over the
          clip): <out>/<name>_orbit.mp4 (H.264, yuv420p, faststart)

    python -m bench.scene_recon.fuse_render fuse --recon <world out>/recon --vipe <vipe out> --list clips.json --out <dir>
    python -m bench.scene_recon.fuse_render render --recon <world out>/recon --list clips.json --out <dir>
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from fab import paths

BG = (25, 18, 15)  # BGR of the reference's dark navy
MAX_SPLAT = 8  # largest splat half-size (px): near-field points stay closed
CYAN = (230, 200, 80)  # BGR


def clip_list(path, only):
    c = json.loads(Path(path).read_text())
    return [(k, v["images"], float(v.get("fps", 30))) for k, v in sorted(c.items()) if not only or any(o in k for o in only.split(","))]


def person_masks(verts_w, T_wc, K, hw, faces, dilate=4):
    """(T, h, w) bool masks of the recon body (world vertices) in each camera at the dense depth's resolution."""
    h, w = hw
    out = np.zeros((len(verts_w), h, w), bool)
    ker = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    for t in range(len(verts_w)):
        Tcw = np.linalg.inv(T_wc[t])
        v = verts_w[t].astype(np.float64) @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = v[:, 2]
        if (z < 0.05).all():
            continue
        uv = (v[:, :2] / np.clip(z[:, None], 1e-3, None)) @ K[t][:2, :2].T + K[t][:2, 2]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, list(uv[faces].round().astype(np.int32)[(z[faces] > 0.05).all(1)]), 1)
        out[t] = cv2.dilate(m, ker) > 0
    return out


class VoxelAcc:
    """Voxel hash on the GPU: per voxel the sum of positions / colours and the number of frames that saw it."""

    def __init__(self, voxel):
        self.voxel, self.keys, self.pos, self.col, self.cnt = voxel, None, None, None, None

    def add(self, X, C):
        X, C = torch.as_tensor(X, device="cuda", dtype=torch.float32), torch.as_tensor(C, device="cuda", dtype=torch.float32)
        q = torch.floor(X / self.voxel).long() + (1 << 20)
        key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        uk, inv = torch.unique(key, return_inverse=True)  # one count per voxel per frame
        pos = torch.zeros(len(uk), 3, device="cuda").index_add_(0, inv, X)
        col = torch.zeros(len(uk), 3, device="cuda").index_add_(0, inv, C)
        n = torch.zeros(len(uk), device="cuda").index_add_(0, inv, torch.ones(len(X), device="cuda"))
        pos, col, cnt = pos / n[:, None], col / n[:, None], torch.ones(len(uk), device="cuda")
        if self.keys is None:
            self.keys, self.pos, self.col, self.cnt = uk, pos, col, cnt
            return
        k2 = torch.cat([self.keys, uk])
        u2, inv2 = torch.unique(k2, return_inverse=True)
        w = torch.cat([self.cnt, cnt])
        P = torch.zeros(len(u2), 3, device="cuda").index_add_(0, inv2, torch.cat([self.pos, pos]) * w[:, None])
        Cc = torch.zeros(len(u2), 3, device="cuda").index_add_(0, inv2, torch.cat([self.col, col]) * w[:, None])
        W = torch.zeros(len(u2), device="cuda").index_add_(0, inv2, w)
        self.keys, self.pos, self.col, self.cnt = u2, P / W[:, None], Cc / W[:, None], W


def carve(X, T, depth, ok_masks, K, fr, fix, stride, tau):
    """Free-space carving of the fused points: per --carve_stride-th frame, a point that projects onto a valid static
    pixel at a depth below the observed depth x (1 - tau) is a free-space vote (the camera sees past it); within
    +-tau x depth a support vote. Returns the keep mask (free votes <= max(1, support votes))."""
    Xt = torch.as_tensor(X, device="cuda", dtype=torch.float32)
    free = torch.zeros(len(Xt), device="cuda")
    sup = torch.zeros(len(Xt), device="cuda")
    h, w = depth.shape[1:]
    for k in range(0, len(fr), stride):
        f = int(fr[k])
        if f >= len(T) or ok_masks[k] is None:
            continue
        Tcw = torch.as_tensor(np.linalg.inv(T[f]), device="cuda", dtype=torch.float32)
        Xc = Xt @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = Xc[:, 2]
        u = (Xc[:, 0] / z.clamp(min=1e-3) * K[k][0, 0] + K[k][0, 2]).round().long()
        v = (Xc[:, 1] / z.clamp(min=1e-3) * K[k][1, 1] + K[k][1, 2]).round().long()
        inb = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = torch.as_tensor(depth[k].astype(np.float32) * fix, device="cuda")
        m = torch.as_tensor(ok_masks[k], device="cuda")
        idx = inb.nonzero()[:, 0]
        dd, mm, zz = d[v[idx], u[idx]], m[v[idx], u[idx]], z[idx]
        free.index_add_(0, idx, ((zz < dd * (1 - tau)) & mm).float())
        sup.index_add_(0, idx, (((zz - dd).abs() <= dd * tau) & mm).float())
    return (free <= sup.clamp(min=1)).cpu().numpy(), free.cpu().numpy(), sup.cpu().numpy()


def body_zbuffer(V, faces, T_wc, K, h, w):
    """Ray-cast depth (camera z, inf off the body) of the SMPL mesh V (world) from camera T_wc (camera -> world), K."""
    import open3d as o3d

    Tcw = np.linalg.inv(T_wc)
    Vc = (V @ Tcw[:3, :3].T + Tcw[:3, 3]).astype(np.float32)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.core.Tensor(Vc), o3d.core.Tensor(np.asarray(faces).astype(np.uint32)))
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))], -1).reshape(-1, 3)
    nr = np.linalg.norm(dc, axis=1, keepdims=True)
    rays = np.concatenate([np.zeros_like(dc), dc / nr], 1).astype(np.float32)
    return (rc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() / nr[:, 0]).reshape(h, w)


FOOT_PARTS = (7, 8, 10, 11)  # SMPL ankles + feet: the body below the ankles


def foot_face_mask(body):
    """(F,) bool: SMPL faces whose vertices all belong to the parts below the ankles (FOOT_PARTS)."""
    part = body.weights.argmax(1).cpu().numpy()
    return np.isin(part, FOOT_PARTS)[body.faces.cpu().numpy()].all(1)


def human_visibility_carve(X, verts_w, T_wc, dense, faces, stride=2, tol=0.05, tol_rel=0.02, skip_feet=0.15, foot_faces=None):
    """Human-visibility carving: the person is seen in the input wherever ViPE's dynamic mask and the body's silhouette
    overlap, so no static scene can lie between the camera and the body there. Per stride-th dense frame, the points
    (world X) that project into those pixels at a depth below the body's (ray-cast SMPL) - (tol + tol_rel x depth) are
    dropped. The feet are left out of the test (foot_faces: the faces below the ankles; without it, the body's lowest
    skip_feet m): a foot in the ground is a contact error, resolved by the joint optimisation's snapping, not by holes in
    the ground; shins and seated legs stay in. Returns the keep mask."""
    Xt = torch.as_tensor(np.asarray(X), device="cuda", dtype=torch.float32)
    bad = torch.zeros(len(Xt), dtype=torch.bool, device="cuda")
    fr = dense["frames"]
    h, w = dense["depth"].shape[1:]
    for k in range(0, len(fr), stride):
        f = int(fr[k])
        if f >= len(verts_w) or f >= len(T_wc):
            continue
        K = dense["K"][k].astype(np.float64)
        Vf = verts_w[f].astype(np.float64)
        if foot_faces is not None:
            fk = faces[~foot_faces]
        else:
            fk = faces[Vf[faces, 2].min(1) > Vf[:, 2].min() + skip_feet] if skip_feet > 0 else faces
        zb = body_zbuffer(Vf, fk, T_wc[f], K, h, w)
        vis = np.isfinite(zb) & dense["dyn"][k]
        if vis.sum() < 20:
            continue
        lim = torch.as_tensor(np.where(vis, zb - (tol + tol_rel * zb), -np.inf), device="cuda", dtype=torch.float32)
        Tcw = torch.as_tensor(np.linalg.inv(T_wc[f]), device="cuda", dtype=torch.float32)
        Xc = Xt @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = Xc[:, 2]
        u = (Xc[:, 0] / z.clamp(min=1e-3) * K[0, 0] + K[0, 2]).round().long()
        v = (Xc[:, 1] / z.clamp(min=1e-3) * K[1, 1] + K[1, 2]).round().long()
        idx = ((z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).nonzero()[:, 0]
        bad[idx] |= z[idx] < lim[v[idx], u[idx]]
    return (~bad).cpu().numpy()


def seg_union(dyn, seg_dir, name, dilate=3):
    """ViPE's dynamic mask OR the segmenter's people + sky (clean_render.py stage seg: <seg_dir>/<name>_seg.npz on every
    stride-th dense frame; a frame between takes the nearest segmented one), people dilated -> (T, h, w) bool."""
    sp = Path(seg_dir) / f"{name}_seg.npz" if seg_dir else None
    if sp is None or not sp.exists():
        return dyn
    z = np.load(sp)
    used, person, sky = z["used"].astype(int), z["person"], z["sky"]  # loaded once (NpzFile re-reads per access)
    ker = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    per = np.stack([cv2.dilate(m.astype(np.uint8), ker) > 0 for m in person]) | sky
    out = np.array(dyn, copy=True)
    out |= per[np.abs(used[None] - np.arange(len(out))[:, None]).argmin(1)]
    return out


def write_ply(path, X, C):
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(X)}\nproperty float x\nproperty float y\n"
                f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        rec = np.zeros(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = X[:, 0], X[:, 1], X[:, 2]
        rec["r"], rec["g"], rec["b"] = C[:, 0], C[:, 1], C[:, 2]
        f.write(rec.tobytes())


def stage_fuse(a):
    from bench.body_models import SMPL

    body_cuda = SMPL(Path(a.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    faces = body_cuda.faces.cpu().numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, images, fps in clip_list(a.list, a.only):
        dst = out / f"{name}_scene_points.npz"
        rp, dp = Path(a.recon) / f"{name}.npz", Path(a.vipe) / "dense" / f"{name}.npz"
        if dst.exists() or not (rp.exists() and dp.exists()):
            continue
        r, dense = np.load(rp), np.load(dp)
        T, fix, verts = r["T_world_cam"].astype(np.float64), float(r["scale_fix"]), r["verts"].astype(np.float32)
        if "depth_scale" in r.files:  # the scene depth's own scale about each camera (recon.py --depth_scale_json)
            fix *= float(r["depth_scale"])
        depth, dyn, K, fr = dense["depth"], dense["dyn"], dense["K"].astype(np.float64), dense["frames"]
        dyn = seg_union(dyn, a.seg_dir, name)
        h, w = depth.shape[1:]
        pm = person_masks(verts, T, K[np.clip(np.arange(len(T)), 0, len(K) - 1)], (h, w), faces, a.dilate)
        acc = VoxelAcc(a.voxel)
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        oks = [None] * len(fr)
        for k in range(len(fr)):
            f = int(fr[k])
            if f >= len(T) or f >= len(images) or (k % a.stride and k % a.carve_stride):
                continue
            d = depth[k].astype(np.float32) * fix
            gy, gx = np.gradient(d)
            edge = np.hypot(gx, gy) > a.edge * np.clip(d, 1e-3, None)
            ok = (~dyn[k]) & (~pm[f]) & (~edge) & np.isfinite(d) & (d > 0.1) & (d < a.max_depth)
            if a.carve and k % a.carve_stride == 0:
                oks[k] = ok
            if k % a.stride or ok.sum() < 100:
                continue
            img = cv2.resize(cv2.imread(images[f]), (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            z = d[ok]
            x = (uu[ok] - K[k][0, 2]) / K[k][0, 0] * z
            y = (vv[ok] - K[k][1, 2]) / K[k][1, 1] * z
            Xc = np.stack([x, y, z], 1)
            Xw = Xc @ T[f][:3, :3].T + T[f][:3, 3]
            acc.add(Xw, img[ok].astype(np.float32))
        keep = acc.cnt >= a.min_count
        X, C = acc.pos[keep].cpu().numpy(), acc.col[keep].clamp(0, 255).cpu().numpy().astype(np.uint8)
        cnt = acc.cnt[keep].cpu().numpy().astype(np.int32)
        n0 = len(X)
        if a.carve:
            kc, fv, sv = carve(X, T, depth, oks, K, fr, fix, a.carve_stride, a.carve_tau)
            X, C, cnt = X[kc], C[kc], cnt[kc]
        sp = Path(a.recon) / f"{name}_snap.npz"
        if sp.exists():  # the joint optimisation's contact snapping (bench/scene_recon/joint.py --snap 1), same rule
            from bench.scene_recon.joint import snap_scene
            from bench.videomimic.recon import Terrain

            z = np.load(sp)
            t1 = Terrain({k[3:]: z[k] for k in z.files if k.startswith("hf_")}, device="cpu")
            sh = lambda xy: t1(torch.as_tensor(xy, dtype=torch.float32)).numpy().astype(np.float64)  # noqa: E731
            jj = json.loads(str(r["joint"])).get("snap", {}) if "joint" in r.files else {}
            X, _ = snap_scene(X.astype(np.float64), z["footprints"], sh, jj.get("sigma", 0.08), jj.get("max", 0.3),
                              jj.get("band", 0.25))
            X = X.astype(np.float32)
        n_vis = 0
        if a.vis_carve:
            kv = human_visibility_carve(X, smooth_verts(r, body_cuda), T, {"frames": fr, "depth": depth, "dyn": dyn, "K": K},
                                        faces, a.carve_stride, foot_faces=foot_face_mask(body_cuda))
            n_vis = int((~kv).sum())
            X, C, cnt = X[kv], C[kv], cnt[kv]
        np.savez(dst, xyz=X.astype(np.float32), rgb=C, count=cnt, voxel=np.float32(a.voxel), stride=a.stride,
                 min_count=a.min_count, carve=a.carve, carve_tau=a.carve_tau, vis_carve=a.vis_carve)
        write_ply(out / f"{name}_scene_points.ply", X, C)
        print(f"[fuse] {name}: {len(X)} points ({int(keep.sum())} / {len(keep)} voxels seen >= {a.min_count}x, "
              f"{n0 - len(X) - n_vis} carved as free space, {n_vis} in front of the visible person), voxel "
              f"{a.voxel * 100:.1f} cm", flush=True)


def sample_surface(faces, n, seed=0):
    rng = np.random.default_rng(seed)
    fi = rng.integers(0, len(faces), n)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1).astype(np.float32)
    return fi, bary


def splat(Xc, col, K, H, W, rad_px, zbuf=None, cbuf=None):
    """z-buffered square splats on the GPU: Xc (N, 3) camera frame, col (N, 3) uint8-ish, per-point radius (N,) px."""
    z = Xc[:, 2]
    ok = z > 0.05
    Xc, col, rad_px, z = Xc[ok], col[ok], rad_px[ok], z[ok]
    u = (Xc[:, 0] / z * K[0, 0] + K[0, 2]).round().long()
    v = (Xc[:, 1] / z * K[1, 1] + K[1, 2]).round().long()
    if zbuf is None:
        zbuf = torch.full((H * W,), 1 << 62, dtype=torch.long, device="cuda")
    zq = (z * 10000).long().clamp(max=(1 << 30) - 1)
    ids = torch.arange(len(z), device="cuda")
    R = int(rad_px.max().item()) if len(rad_px) else 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = rad_px >= max(abs(dx), abs(dy))
            uu, vv = u[m] + dx, v[m] + dy
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            pix = vv[inb] * W + uu[inb]
            key = (zq[m][inb] << 32) | ids[m][inb]
            zbuf.scatter_reduce_(0, pix, key, reduce="amin")
    hit = zbuf < (1 << 62)
    img = torch.zeros(H * W, 3, device="cuda")
    img[hit] = col[(zbuf[hit] & ((1 << 32) - 1))].float()
    return img.view(H, W, 3), hit.view(H, W)


def render_view(Xs, Cs, rs, Xb, Cb, rb, R_cw, t_cw, K, H, W, max_splat=MAX_SPLAT, near=0.0):
    """Scene points + body points in one z-buffer (camera R_cw / t_cw: world -> camera); scene points closer than near (m)
    to the camera are dropped (the camera holder / the ground under the lens); splat half-size <= max_splat px."""
    Xsc = Xs @ R_cw.T + t_cw
    keep = Xsc[:, 2] > near
    X = torch.cat([Xsc[keep], Xb @ R_cw.T + t_cw])
    C = torch.cat([Cs[keep], Cb])
    zz = X[:, 2].clamp(min=0.05)
    rad = torch.cat([rs[keep], rb]) / zz
    rad = rad.clamp(0, max_splat).round()
    img, hit = splat(X, C, K, H, W, rad)
    out = torch.tensor(BG[::-1], device="cuda", dtype=torch.float32).expand(H, W, 3).clone()
    out[hit] = img[hit]
    return out.clamp(0, 255).byte().cpu().numpy()


def header(width, text, color):
    band = np.zeros((56, width, 3), np.uint8)
    band[:] = BG
    cv2.putText(band, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return band


def encode(raw, final, crf=27):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(crf), "-movflags", "+faststart", str(final)], check=True)
    raw.unlink()


def smooth_verts(r, body):
    """The recon body at float32: SMPL(pose, betas, trans) moved / scaled per frame onto the stored float16 vertices'
    centroid / spread (keeps the contact refinement's ray-wise depth factor; f16 world coordinates step ~1.5 cm at
    16-32 m)."""
    dev = next(body.buffers()).device  # the body's device (CPU too)
    pose, betas, trans = (torch.tensor(np.asarray(r[k], np.float32), device=dev) for k in ("pose", "betas", "trans"))
    verts16 = torch.tensor(r["verts"].astype(np.float32), device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(pose), 256):
            V, _ = body(pose[i: i + 256, :3], pose[i: i + 256, 3:], betas[None].expand(len(pose[i: i + 256]), -1),
                        trans[i: i + 256])
            ref = verts16[i: i + 256]
            cV, cR = V.mean(1, keepdim=True), ref.mean(1, keepdim=True)
            sc = (ref - cR).norm(dim=-1).mean(1, keepdim=True)[..., None] / (V - cV).norm(dim=-1).mean(1, keepdim=True)[..., None]
            out.append((cR + sc * (V - cV)).cpu().numpy())
    return np.concatenate(out)


def stage_render(a):
    from bench.body_models import SMPL

    body = SMPL(Path(a.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    faces = body.faces.cpu().numpy()
    out = Path(a.out)
    for name, images, fps in clip_list(a.list, a.only):
        sp, rp = out / f"{name}_scene_points.npz", Path(a.recon) / f"{name}.npz"
        if not (sp.exists() and rp.exists()):
            continue
        s, r = np.load(sp), np.load(rp)
        T = r["T_world_cam"].astype(np.float64)
        verts = smooth_verts(r, body)
        Kf = np.asarray(r["K"], np.float64)  # full-resolution intrinsics of the clip (ViPE's)
        img0 = cv2.imread(images[0])
        H0, W0 = img0.shape[:2]
        H = a.height
        sc = H / H0
        W = int(W0 * sc) // 2 * 2
        K = torch.tensor(np.diag([sc, sc, 1.0]) @ Kf, device="cuda", dtype=torch.float32)
        Xs = torch.tensor(s["xyz"], device="cuda")
        Cs = torch.tensor(s["rgb"], device="cuda", dtype=torch.float32)
        rs = torch.full((len(Xs),), float(s["voxel"]) * float(K[0, 0]) * 0.6, device="cuda")  # px x depth (m)
        fi, bary = sample_surface(faces, a.body_points)
        fi_t, bary_t = torch.tensor(fi, device="cuda"), torch.tensor(bary, device="cuda")
        F = torch.tensor(faces.astype(np.int64), device="cuda")
        light = torch.tensor([0.3, -0.5, -0.8], device="cuda")
        light = light / light.norm()
        base = torch.tensor([30.0, 150.0, 214.0], device="cuda")  # RGB of the reference's blue body
        n = min(len(T), len(images), len(verts))
        raw = out / f"{name}_human_scene_raw.mp4"
        wr = None
        orbit_frames = []
        from scipy.ndimage import gaussian_filter1d

        ctrs = gaussian_filter1d(verts.mean(1).astype(np.float64), 15, axis=0, mode="nearest")  # the body, smoothed
        for t in range(n):
            V = torch.tensor(verts[t], device="cuda")
            tri = V[F[fi_t]]
            Xb = (tri * bary_t[..., None]).sum(1)
            nrm = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
            nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
            Tcw = np.linalg.inv(T[t])
            R_cw, t_cw = torch.tensor(Tcw[:3, :3], device="cuda", dtype=torch.float32), torch.tensor(Tcw[:3, 3], device="cuda", dtype=torch.float32)
            shade = (nrm @ R_cw.T @ light).abs().clamp(0.25, 1.0)[:, None]
            Cb = base[None] * shade
            rb = torch.full((len(Xb),), 0.006 * float(K[0, 0]), device="cuda")
            view = render_view(Xs, Cs, rs, Xb, Cb, rb, R_cw, t_cw, K, H, W, a.max_splat, a.near)[:, :, ::-1]
            orig = cv2.resize(cv2.imread(images[t]), (W, H), interpolation=cv2.INTER_AREA)
            frame = np.concatenate([np.concatenate([header(W, "ORIGINAL VIDEO", (235, 235, 235)), orig], 0),
                                    np.concatenate([header(W, f"HUMAN + SCENE | {t / fps:05.2f}s", CYAN), view], 0)], 1)
            if wr is None:
                wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            wr.write(np.ascontiguousarray(frame))
            if t % a.orbit_every == 0:  # third person: following the body from the camera's side, swinging +-45 deg
                ctr = ctrs[t]
                cd = T[t][:3, 3] - ctr
                cd[2] = 0
                cd /= max(np.linalg.norm(cd), 1e-6)
                ang = np.deg2rad(45) * np.sin(2 * np.pi * t / max(n - 1, 1))
                Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
                eye = ctr + a.orbit_radius * (Rz @ cd) + np.array([0, 0, a.orbit_height])
                fwd = ctr + np.array([0, 0, 0.1]) - eye
                fwd /= np.linalg.norm(fwd)
                right = np.cross(fwd, [0, 0, 1.0])
                right /= np.linalg.norm(right)
                down = np.cross(fwd, right)
                Rcw = np.stack([right, down, fwd])
                tcw = -Rcw @ eye
                Ro = torch.tensor(Rcw, device="cuda", dtype=torch.float32)
                to = torch.tensor(tcw, device="cuda", dtype=torch.float32)
                Ko = torch.tensor([[1.1 * a.height, 0, a.height * 0.8], [0, 1.1 * a.height, a.height / 2], [0, 0, 1]],
                                  device="cuda", dtype=torch.float32)
                shade_o = (nrm @ Ro.T @ light).abs().clamp(0.25, 1.0)[:, None]
                ov = render_view(Xs, Cs, torch.full_like(rs, float(s["voxel"]) * float(Ko[0, 0]) * 0.6), Xb,
                                 base[None] * shade_o, torch.full((len(Xb),), 0.006 * float(Ko[0, 0]), device="cuda"),
                                 Ro, to, Ko, a.height, int(a.height * 1.6) // 2 * 2, a.max_splat)[:, :, ::-1]
                orbit_frames.append(np.concatenate([header(ov.shape[1], f"HUMAN + SCENE, THIRD PERSON | {t / fps:05.2f}s", CYAN), ov], 0))
        wr.release()
        encode(raw, out / f"{name}_human_scene.mp4")
        raw = out / f"{name}_orbit_raw.mp4"
        wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / a.orbit_every,
                             (orbit_frames[0].shape[1], orbit_frames[0].shape[0]))
        for f in orbit_frames:
            wr.write(np.ascontiguousarray(f))
        wr.release()
        encode(raw, out / f"{name}_orbit.mp4")
        print(f"[render] {name}: {n} frames, {len(Xs)} scene points", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("fuse", "render"))
    p.add_argument("--recon", required=True)
    p.add_argument("--vipe", default="")
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()), help="body models dir (smpl/SMPL_NEUTRAL.pkl)")
    p.add_argument("--voxel", type=float, default=0.015)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--max_depth", type=float, default=12.0)
    p.add_argument("--edge", type=float, default=0.06, help="depth-gradient / depth above this = a depth edge (masked)")
    p.add_argument("--dilate", type=int, default=4)
    p.add_argument("--carve", type=int, default=0, help="1: free-space carving of the fused points (floaters)")
    p.add_argument("--carve_tau", type=float, default=0.08)
    p.add_argument("--carve_stride", type=int, default=2)
    p.add_argument("--vis_carve", type=int, default=0, help="1: human-visibility carving (human_visibility_carve)")
    p.add_argument("--seg_dir", default="", help="segmenter masks (clean_render.py seg: people + sky) added to ViPE's dynamic mask")
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--body_points", type=int, default=120000)
    p.add_argument("--orbit_every", type=int, default=2)
    p.add_argument("--max_splat", type=int, default=3, help="render: the largest scene / body splat half-size (px)")
    p.add_argument("--near", type=float, default=0.3, help="render: scene points closer than this (m) to the camera are dropped")
    p.add_argument("--orbit_radius", type=float, default=3.5)
    p.add_argument("--orbit_height", type=float, default=1.3)
    a = p.parse_args()
    {"fuse": stage_fuse, "render": stage_render}[a.stage](a)


if __name__ == "__main__":
    main()
