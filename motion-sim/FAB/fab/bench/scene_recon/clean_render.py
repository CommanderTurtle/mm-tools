"""Clean human + scene reconstructions and renders (a
uniform ~3 cm cloud drawn as small round sprites, no floaters, no sky, no other people, the body drawn unoccluded, a raised
chase view), on any recon.py world output (recon dir + its ViPE dense depth). One pipeline and one set of constants for every
clip:

  seg     per used dense frame (every --stride-th), sky and person masks from Mask2Former (Swin-L, ADE20K semantic:
          facebook/mask2former-swin-large-ade-semantic, classes 'sky' / 'person') -> <out>/<clip>_seg.npz
  fuse    the static scene cloud:
          1. per used frame, the metric depth x the recon's scale fix, masked: ViPE's dynamic instances, the segmenter's
             people (dilated) and sky, the recon body's silhouette (dilated), depth edges, depth outside (0.1, --max_depth) m
          2. multi-view consistency: each pixel's world point is re-projected into the --nbrs neighbouring used frames; it
             agrees with one where that frame's depth there is within --rel x depth (masked pixels do not vote); a
             consistent observation has >= --min_votes agreements
          3. voxels of --voxel m (position / colour averaged over the consistent observations, else all), then onto the
             row's TSDF (the sim asset; per-frame depth is 2-4 % noisy, the TSDF averages it): a voxel seen by >=
             --min_count frames within --snap_tsdf of the TSDF moves to its closest TSDF point; one beyond --far (structure
             the TSDF lacks: far trees, buildings) is kept as is when >= --min_cons frames saw it consistently; the rest is
             dropped; re-voxelised (--snap_tsdf 0: keep >= --min_cons consistent, or within --band of the TSDF)
          4. free-space carving (fuse_render.carve, --carve_tau: a voxel the used frames see through more often than on is
             dropped; removes moving people's ghosts and structure that later views contradict)
          5. statistical outlier removal (--sor_k neighbours, --sor_std)
          6. the recon's contact snapping when it has one and the TSDF step is off (<recon>/<clip>_snap.npz)
          7. human-visibility carving (fuse_render.human_visibility_carve: no scene between the camera and the visible body)
          -> <out>/<clip>_clean_points.npz (xyz, rgb, voxel; qa_human_scene.py points:) / .ply, <clip>_fuse.json
  render  round z-buffered sprites (radius --sprite x the voxel's projected size, --min_sprite..--max_sprite) and the body mesh
          (ray-cast, smooth-shaded, drawn over the scene) from the clip's camera and from a chase view (--chase_back m behind
          and --chase_up m above the smoothed camera, looking at the smoothed pelvis), next to the input frame ->
          <out>/<clip>_clean.mp4 (input | camera | chase) and <clip>_clean_views.mp4 (camera | chase)

    python -m bench.scene_recon.clean_render all --recon <world out>/recon --dense <vipe out>/dense --list clips.json \
        --only <clip> --out <dir>      (scripts/fab_world.sh runs the seg stage: person / sky masks for the MVS step)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fab import paths

from bench.scene_recon.fuse_render import BG, CYAN, carve, clip_list, encode, foot_face_mask, header, human_visibility_carve, \
    person_masks, smooth_verts, write_ply

M2F = "facebook/mask2former-swin-large-ade-semantic"
BODY_RGB = (30.0, 150.0, 214.0)  # the reference's blue


def load(a, name):
    r = np.load(Path(a.recon) / f"{name}.npz", allow_pickle=True)
    d = np.load(Path(a.dense) / f"{name}.npz")
    dense = {k: d[k] for k in ("depth", "dyn", "K", "frames")}
    return r, dense


def used_frames(dense, stride, n):
    return np.array([k for k in range(0, len(dense["frames"]), stride) if int(dense["frames"][k]) < n])


def stage_seg(a, name, images):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    dst = Path(a.out) / f"{name}_seg.npz"
    if dst.exists():
        return
    r, dense = load(a, name)
    h, w = dense["depth"].shape[1:]
    ks = used_frames(dense, a.stride, min(len(images), len(r["T_world_cam"])))
    proc = AutoImageProcessor.from_pretrained(M2F)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F).cuda().eval()
    lab = {i: s.split(";")[0].strip() for i, s in model.config.id2label.items()}
    sky_id = [i for i, s in lab.items() if s == "sky"]
    per_id = [i for i, s in lab.items() if s == "person"]
    sky, per = np.zeros((len(ks), h, w), bool), np.zeros((len(ks), h, w), bool)
    for b in range(0, len(ks), 8):
        ims = []
        for k in ks[b: b + 8]:
            im = cv2.imread(images[int(dense["frames"][k])])[:, :, ::-1]
            s = 640 / max(im.shape[:2])
            ims.append(cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA))
        x = proc(images=ims, return_tensors="pt")
        with torch.no_grad():
            o = model(pixel_values=x["pixel_values"].cuda(), pixel_mask=x["pixel_mask"].cuda())
        seg = proc.post_process_semantic_segmentation(o, target_sizes=[(h, w)] * len(ims))
        for i, sm in enumerate(seg):
            sm = sm.cpu().numpy()
            sky[b + i], per[b + i] = np.isin(sm, sky_id), np.isin(sm, per_id)
    np.savez_compressed(dst, used=ks, sky=sky, person=per, model=M2F)
    print(f"[seg] {name}: {len(ks)} frames, sky {sky.mean():.3f}, person {per.mean():.3f} of the pixels", flush=True)


def masks_and_points(a, r, dense, seg, images, faces, verts):
    """Per used frame on the GPU: depth (x fix), the bad-pixel mask, RGB, camera (world -> camera) and intrinsics."""
    T, fix = r["T_world_cam"].astype(np.float64), float(r["scale_fix"])
    ks = seg["used"]
    fr = dense["frames"][ks].astype(int)
    h, w = dense["depth"].shape[1:]
    K = dense["K"][ks].astype(np.float64)
    pm = person_masks(verts[fr], T[fr], K, (h, w), faces, a.dilate)
    ker = np.ones((2 * a.dilate + 1, 2 * a.dilate + 1), np.uint8)
    D = np.zeros((len(ks), h, w), np.float32)
    bad = np.zeros((len(ks), h, w), bool)
    rgb = np.zeros((len(ks), h, w, 3), np.uint8)
    for i, k in enumerate(ks):
        d = dense["depth"][k].astype(np.float32) * fix
        gy, gx = np.gradient(d)
        edge = cv2.dilate((np.hypot(gx, gy) > a.edge * np.clip(d, 1e-3, None)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        people = cv2.dilate((seg["person"][i] | dense["dyn"][k]).astype(np.uint8), ker) > 0
        bad[i] = people | seg["sky"][i] | pm[i] | edge | ~np.isfinite(d) | (d < 0.1) | (d > a.max_depth)
        D[i] = np.where(np.isfinite(d), d, 0)
        rgb[i] = cv2.resize(cv2.imread(images[fr[i]]), (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    Tcw = np.linalg.inv(T[fr])
    g = lambda x, dt=torch.float32: torch.as_tensor(x, device="cuda", dtype=dt)  # noqa: E731
    return g(D), g(bad, torch.bool), g(rgb), g(Tcw), g(K), g(T[fr]), fr


def stage_fuse(a, name, images):
    import open3d as o3d

    from bench.body_models import SMPL

    dst = Path(a.out) / f"{name}_clean_points.npz"
    if dst.exists():
        return
    r, dense = load(a, name)
    seg = np.load(Path(a.out) / f"{name}_seg.npz")
    body = SMPL(Path(a.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    faces = body.faces.cpu().numpy()
    verts = smooth_verts(r, body)
    D, bad, rgb, Tcw, K, Twc, fr = masks_and_points(a, r, dense, seg, images, faces, verts)
    U, h, w = D.shape
    vv, uu = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
    nbrs = [int(o) for o in a.nbrs.split(",")]
    rows = []
    stats = {"pixels_valid": 0, "pixels_consistent": 0}
    for i in range(U):
        ok = ~bad[i]
        if ok.sum() < 100:
            continue
        z = D[i][ok]
        Xc = torch.stack([(uu[ok] - K[i, 0, 2]) / K[i, 0, 0] * z, (vv[ok] - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)
        Xw = Xc @ Twc[i, :3, :3].T + Twc[i, :3, 3]
        votes = torch.zeros(len(z), device="cuda")
        for o in nbrs:
            for j in (i - o, i + o):
                if not 0 <= j < U:
                    continue
                Xj = Xw @ Tcw[j, :3, :3].T + Tcw[j, :3, 3]
                zj = Xj[:, 2]
                u = (Xj[:, 0] / zj.clamp(min=1e-3) * K[j, 0, 0] + K[j, 0, 2]).round().long()
                v = (Xj[:, 1] / zj.clamp(min=1e-3) * K[j, 1, 1] + K[j, 1, 2]).round().long()
                inb = (zj > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                uc, vc = u.clamp(0, w - 1), v.clamp(0, h - 1)
                dj, bj = D[j][vc, uc], bad[j][vc, uc]
                votes += (inb & ~bj & ((zj - dj).abs() <= a.rel * dj)).float()
        cons = votes >= a.min_votes
        C = rgb[i][ok].float()
        q = torch.floor(Xw / a.voxel).long() + (1 << 20)
        key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        uk, inv = torch.unique(key, return_inverse=True)
        cf = cons.float()[:, None]
        src = torch.cat([Xw, C, torch.ones_like(cf), Xw * cf, C * cf, cf], 1)  # all: pos, rgb, n | consistent: pos, rgb, n
        rows.append((uk, torch.zeros(len(uk), 14, device="cuda").index_add_(0, inv, src)))
        stats["pixels_valid"] += int(len(z))
        stats["pixels_consistent"] += int(cons.sum())
    keys = torch.cat([k for k, _ in rows])
    agg = torch.cat([g for _, g in rows])
    del rows
    uk, inv = torch.unique(keys, return_inverse=True)
    tot = torch.zeros(len(uk), 14, device="cuda").index_add_(0, inv, agg)
    n_frames = torch.zeros(len(uk), device="cuda").index_add_(0, inv, torch.ones(len(inv), device="cuda"))
    n_cons = torch.zeros(len(uk), device="cuda").index_add_(0, inv, (agg[:, 13] > 0).float())
    has_c = tot[:, 13] > 0
    X = torch.where(has_c[:, None], tot[:, 7:10] / tot[:, 13:14].clamp(min=1), tot[:, 0:3] / tot[:, 6:7])
    Cc = torch.where(has_c[:, None], tot[:, 10:13] / tot[:, 13:14].clamp(min=1), tot[:, 3:6] / tot[:, 6:7])
    X, Cc, n_frames, n_cons = X.cpu().numpy(), Cc.clamp(0, 255).cpu().numpy().astype(np.uint8), n_frames.cpu().numpy(), n_cons.cpu().numpy()
    stats["voxels_seen"] = int(len(X))
    tsdf = Path(a.recon) / f"{name}_scene.ply"
    snapped = False
    if a.snap_tsdf > 0 and tsdf.exists():  # onto the row's TSDF surface (the sim asset); far consistent structure kept as is
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(tsdf))))
        cp = sc.compute_closest_points(o3d.core.Tensor(X.astype(np.float32)))["points"].numpy()
        dist = np.linalg.norm(cp - X, axis=1)
        near = (dist < a.snap_tsdf) & (n_frames >= a.min_count)
        keep = near | ((dist > a.far) & (n_cons >= a.min_cons))
        X = np.where(near[:, None], cp, X)[keep]
        Cc = Cc[keep]
        q = np.floor(X / a.voxel).astype(np.int64)  # one point per voxel again
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        cnt = np.bincount(inv).astype(np.float64)[:, None]
        X = np.stack([np.bincount(inv, X[:, c]) for c in range(3)], 1) / cnt
        Cc = (np.stack([np.bincount(inv, Cc[:, c].astype(np.float64)) for c in range(3)], 1) / cnt).astype(np.uint8)
        stats["tsdf_snapped"] = int(near[keep].sum())
        stats["kept_far_consistent"] = int((keep & ~near).sum())
        snapped = True
    else:
        keep = n_cons >= a.min_cons
        if a.band > 0 and tsdf.exists():
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(tsdf))))
            dist = sc.compute_distance(o3d.core.Tensor(X.astype(np.float32))).numpy()
            keep |= (dist < a.band) & (n_frames >= a.min_count)
        X, Cc = X[keep], Cc[keep]
    stats["voxels_kept"] = int(len(X))
    oks = [None] * len(dense["frames"])
    bad_np = bad.cpu().numpy()
    for i, k in enumerate(seg["used"]):
        oks[k] = ~bad_np[i]
    kc, _, _ = carve(X, r["T_world_cam"].astype(np.float64), dense["depth"], oks, dense["K"].astype(np.float64),
                     dense["frames"], float(r["scale_fix"]), a.stride, a.carve_tau)
    stats["free_space_carved"] = int((~kc).sum())
    X, Cc = X[kc], Cc[kc]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X.astype(np.float64)))
    _, ind = pc.remove_statistical_outlier(nb_neighbors=a.sor_k, std_ratio=a.sor_std)
    X, Cc = X[ind], Cc[ind]
    stats["after_sor"] = int(len(X))
    T = r["T_world_cam"].astype(np.float64)
    sp = Path(a.recon) / f"{name}_snap.npz"
    if sp.exists() and not snapped:  # the recon's contact snapping (fuse_render.py fuse); a snapped TSDF already has it
        from bench.scene_recon.joint import snap_scene
        from bench.videomimic.recon import Terrain

        z = np.load(sp)
        t1 = Terrain({k[3:]: z[k] for k in z.files if k.startswith("hf_")}, device="cpu")
        sh = lambda xy: t1(torch.as_tensor(xy, dtype=torch.float32)).numpy().astype(np.float64)  # noqa: E731
        jj = json.loads(str(r["joint"])).get("snap", {}) if "joint" in r.files else {}
        X, _ = snap_scene(X.astype(np.float64), z["footprints"], sh, jj.get("sigma", 0.08), jj.get("max", 0.3),
                          jj.get("band", 0.25))
        X = X.astype(np.float32)
        stats["snapped"] = True
    kv = human_visibility_carve(X, verts, T, dense, faces, a.stride, foot_faces=foot_face_mask(body))
    stats["visibility_carved"] = int((~kv).sum())
    X, Cc = X[kv], Cc[kv]
    stats["points"] = int(len(X))
    np.savez(dst, xyz=X.astype(np.float32), rgb=Cc, voxel=np.float32(a.voxel))
    write_ply(Path(a.out) / f"{name}_clean_points.ply", X, Cc)
    stats.update({k: getattr(a, k) for k in ("voxel", "stride", "nbrs", "rel", "min_votes", "min_cons", "band", "snap_tsdf", "far",
                                             "min_count", "carve_tau", "sor_k", "sor_std", "max_depth", "edge", "dilate")})
    (Path(a.out) / f"{name}_fuse.json").write_text(json.dumps(stats, indent=1))
    print(f"[fuse] {name}: {json.dumps(stats)}", flush=True)


def splat_round(Xc, col, K, H, W, rad):
    """z-buffered round sprites on the GPU: Xc (N, 3) camera frame, col (N, 3), integer radius (N,) px."""
    z = Xc[:, 2]
    u = (Xc[:, 0] / z * K[0, 0] + K[0, 2]).round().long()
    v = (Xc[:, 1] / z * K[1, 1] + K[1, 2]).round().long()
    zbuf = torch.full((H * W,), 1 << 62, dtype=torch.long, device="cuda")
    zq = (z * 10000).long().clamp(max=(1 << 30) - 1)
    ids = torch.arange(len(z), device="cuda")
    R = int(rad.max().item()) if len(rad) else 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = rad * rad + rad >= dx * dx + dy * dy  # a disc (r + 0.5)^2 > dx^2 + dy^2
            uu, vv = u[m] + dx, v[m] + dy
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            zbuf.scatter_reduce_(0, vv[inb] * W + uu[inb], (zq[m][inb] << 32) | ids[m][inb], reduce="amin")
    hit = zbuf < (1 << 62)
    img = torch.tensor(BG[::-1], device="cuda", dtype=torch.float32).expand(H * W, 3).clone()
    img[hit] = col[zbuf[hit] & ((1 << 32) - 1)].float()
    return img.view(H, W, 3)


def body_layer(V, faces, vn, Tcw, K, H, W, light):
    """The body mesh ray-cast from a camera (world -> camera Tcw): smooth-shaded RGB and the hit mask (numpy)."""
    import open3d as o3d

    Vc = (V @ Tcw[:3, :3].T + Tcw[:3, 3]).astype(np.float32)
    nc = vn @ Tcw[:3, :3].T
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.core.Tensor(Vc), o3d.core.Tensor(faces.astype(np.uint32)))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3)
    dc /= np.linalg.norm(dc, axis=1, keepdims=True)
    o = rc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dc), dc], 1).astype(np.float32)))
    hit = np.isfinite(o["t_hit"].numpy())
    pid = o["primitive_ids"].numpy()[hit]
    b = o["primitive_uvs"].numpy()[hit]
    tri = faces[pid]
    n = nc[tri[:, 0]] * (1 - b[:, :1] - b[:, 1:]) + nc[tri[:, 1]] * b[:, :1] + nc[tri[:, 2]] * b[:, 1:]
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    sh = 0.35 + 0.65 * np.abs(n @ light)
    img = np.zeros((H * W, 3), np.float32)
    img[hit] = np.asarray(BODY_RGB)[None] * sh[:, None]
    return img.reshape(H, W, 3), hit.reshape(H, W)


def vertex_normals(V, faces):
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-9)


def look_at(eye, target):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ eye
    return T


def stage_render(a, name, images, fps):
    from scipy.ndimage import gaussian_filter1d

    from bench.body_models import SMPL

    s = np.load(Path(a.out) / f"{name}_clean_points.npz")
    r, _ = load(a, name)
    body = SMPL(Path(a.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    faces = body.faces.cpu().numpy().astype(np.int64)
    verts = smooth_verts(r, body).astype(np.float64)
    T = r["T_world_cam"].astype(np.float64)
    Kf = np.asarray(r["K"], np.float64)
    H0, W0 = cv2.imread(images[0]).shape[:2]
    sc = min(a.height / H0, a.max_width / W0)
    H, W = int(round(H0 * sc)) // 2 * 2, int(round(W0 * sc)) // 2 * 2
    Kc = np.diag([sc, sc, 1.0]) @ Kf
    Kh = Kc.copy()
    Kh[0, 2], Kh[1, 2] = W / 2, H / 2
    n = min(len(T), len(images), len(verts))
    n = min(n, a.start + a.max_frames) if a.max_frames else n
    cam = gaussian_filter1d(T[:n, :3, 3], a.smooth, axis=0, mode="nearest")
    fwd = T[:n, :3, 2].copy()
    fwd[:, 2] = 0
    fwd = gaussian_filter1d(fwd / np.linalg.norm(fwd, axis=1, keepdims=True).clip(1e-9), a.smooth, axis=0, mode="nearest")
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True).clip(1e-9)
    pel = gaussian_filter1d(verts[:n].mean(1), a.smooth, axis=0, mode="nearest")
    Xs = torch.tensor(s["xyz"], device="cuda", dtype=torch.float32)
    Cs = torch.tensor(s["rgb"], device="cuda", dtype=torch.float32)
    vox = float(s["voxel"])
    light = np.array([0.3, -0.5, -0.8])
    light /= np.linalg.norm(light)

    def view(Tcw, K, near):
        R_, t_ = torch.tensor(Tcw[:3, :3], device="cuda", dtype=torch.float32), torch.tensor(Tcw[:3, 3], device="cuda", dtype=torch.float32)
        Xc = Xs @ R_.T + t_
        m = Xc[:, 2] > near
        rad = (a.sprite * vox * float(K[0, 0]) / Xc[m, 2]).round().clamp(a.min_sprite, a.max_sprite).long() - 1
        img = splat_round(Xc[m], Cs[m], torch.tensor(K, device="cuda", dtype=torch.float32), H, W, rad).cpu().numpy()
        bl, hit = body_layer(verts[t], faces, vn, Tcw, K, H, W, light)
        img[hit] = bl[hit]
        return np.ascontiguousarray(img[:, :, ::-1].clip(0, 255).astype(np.uint8))

    out = Path(a.out)
    raws = [out / f"{name}_clean_raw.mp4", out / f"{name}_clean_views_raw.mp4"]
    wr = [None, None]
    for t in range(min(a.start, n), n):
        vn = vertex_normals(verts[t], faces)
        v_cam = view(np.linalg.inv(T[t]), Kc, a.near)
        eye = cam[t] - a.chase_back * fwd[t] + np.array([0, 0, a.chase_up])
        v_chase = view(look_at(eye, pel[t]), Kh, a.near)
        orig = cv2.resize(cv2.imread(images[t]), (W, H), interpolation=cv2.INTER_AREA)
        ts = f"{t / fps:05.2f}s"
        panels = [np.concatenate([header(W, "ORIGINAL VIDEO", (235, 235, 235)), orig], 0),
                  np.concatenate([header(W, f"HUMAN + SCENE | {ts}", CYAN), v_cam], 0),
                  np.concatenate([header(W, f"CHASE VIEW | {ts}", CYAN), v_chase], 0)]
        for i, fr in enumerate((np.concatenate(panels, 1), np.concatenate(panels[1:], 1))):
            if wr[i] is None:
                wr[i] = cv2.VideoWriter(str(raws[i]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fr.shape[1], fr.shape[0]))
            wr[i].write(fr)
    for w_, raw, fin in zip(wr, raws, (out / f"{name}_clean.mp4", out / f"{name}_clean_views.mp4")):
        w_.release()
        encode(raw, fin, crf=23)
    print(f"[render] {name}: {n} frames, {len(Xs)} points, panels {W}x{H}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("seg", "fuse", "render", "all"))
    p.add_argument("--recon", required=True)
    p.add_argument("--dense", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()), help="body models dir (smpl/SMPL_NEUTRAL.pkl)")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--nbrs", default="1,2,4,8,12", help="neighbour offsets in used frames (both sides)")
    p.add_argument("--rel", type=float, default=0.03)
    p.add_argument("--min_votes", type=int, default=3)
    p.add_argument("--min_cons", type=int, default=2)
    p.add_argument("--band", type=float, default=0.06)
    p.add_argument("--snap_tsdf", type=float, default=0.15, help="move voxels within this of the TSDF onto it (0: off)")
    p.add_argument("--far", type=float, default=0.5, help="with --snap_tsdf: consistent voxels beyond this from the TSDF kept")
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--carve_tau", type=float, default=0.08)
    p.add_argument("--sor_k", type=int, default=16)
    p.add_argument("--sor_std", type=float, default=2.0)
    p.add_argument("--max_depth", type=float, default=12.0)
    p.add_argument("--edge", type=float, default=0.06)
    p.add_argument("--dilate", type=int, default=4)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max_width", type=int, default=960)
    p.add_argument("--sprite", type=float, default=0.3)
    p.add_argument("--max_sprite", type=int, default=3)
    p.add_argument("--min_sprite", type=int, default=2, help="sprite diameter class: 1 = 1 px, 2 = 3 px, 3 = 5 px")
    p.add_argument("--near", type=float, default=0.4)
    p.add_argument("--smooth", type=float, default=10.0)
    p.add_argument("--chase_back", type=float, default=1.2)
    p.add_argument("--chase_up", type=float, default=0.9)
    p.add_argument("--max_frames", type=int, default=0, help="render: only n frames from --start (0 = all)")
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name, images, fps in clip_list(a.list, a.only):
        if not (Path(a.recon) / f"{name}.npz").exists():
            continue
        for st, fn in (("seg", stage_seg), ("fuse", stage_fuse)):
            if a.stage in (st, "all"):
                fn(a, name, images)
        if a.stage in ("render", "all"):
            stage_render(a, name, images, fps)


if __name__ == "__main__":
    main()
