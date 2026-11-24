"""The pipeline tiles of the paper's Fig. 2, on one HOT3D frame (clip-001910, frame 110), with the release model (FAH-L,
image mode, default configuration; mesh both, contact, detector masks, part maps):

  p2_detector.jpg     frame with the detector's boxes and instance masks (side colours)
  p3_crop.jpg         the right hand's square crop (context 1.25)
  p4_points2d_wide.jpg  the point head on the right hand: 2D joints + visible dense points (finger colours), in the wide window
  p4_points3d.png     the root-relative dense points and joints in 3D (inset)
  p5_mhr_wide.jpg     the fitted MHR-Hand meshes in the wide window
  p5_meshes3d.png     MHR-Hand (shaded) with the fitted MANO mesh outline, novel view (inset)
  p6_parts_wide.jpg   the right hand's part map inside its detector mask, the rest faded, in the wide window
  p7_contact_wide.jpg / p7_contact_wide_turned.png  per-vertex contact on the right hand's MANO mesh (grey -> violet,
                      linear 0-1) and the same mesh from its contact side
  strip_<frame>.jpg   frames of the causal video run with its 2D joints and the last frames' thumb / index tip trails (the video strip)
  p9_world.jpg        world space: camera path, frusta and both wrists from the video run and its camera trajectory

Inputs: the HOT3D clip (or an .npz with image / K), the release weights, and for the strip and world tiles a bench.run
prediction pickle of the clip in video mode (--vid_run) and its camera trajectory (--cams: T_world_cam (N, 4, 4), index).

    python scripts/paper_fig2.py --hot3d_clip clip-001910.tar --frame 110 --weights <weights> --sam3d <sam3d> \
        --vid_run hot3d/<video run>/clip-001910.pkl --cams <cameras>/clip-001910.npz --out fig2/
"""
import argparse
import io
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FINGER_COLORS = [(40, 200, 230), (60, 60, 255), (0, 170, 255), (60, 220, 60), (255, 160, 0), (220, 60, 220)]  # palm, thumb..pinky (BGR)
WIDE_ASPECT = 1.06 / 0.63  # the paper's tile aspect
TIPS_MANO = (744, 320, 443, 554, 671)  # thumb .. pinky fingertip vertices of the MANO template
PTS3D_LAYOUT = (473, 420), (0, 16, 458, 408)  # the 3D points inset: canvas size, box the trimmed render is scaled into
TRAIL_COLORS = {True: (188, 121, 66), False: (72, 132, 222)}  # video strip tip trails, right / left (BGR)


def crop_box(boxes, W, H, aspect, margin):
    """Union of the boxes, grown by margin, widened to aspect, kept inside the image."""
    b = np.asarray(boxes, np.float64)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    bw, bh = max(bw, bh * aspect), max(bh, bw / aspect)
    s = min(1.0, W / bw, H / bh)
    bw, bh = bw * s, bh * s
    cx, cy = float(np.clip(cx, bw / 2, W - bw / 2)), float(np.clip(cy, bh / 2, H - bh / 2))
    return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)


def cut(img, box, out_h):
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (int(round(out_h * c.shape[1] / c.shape[0])), out_h), interpolation=cv2.INTER_AREA)


def pad_black(img, pad):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def hand_crop(p, scale=1.25):
    """Square box around a hand box (context scale) and the padding that keeps it inside a padded frame."""
    x0, y0, x1, y1 = np.asarray(p["bbox"], np.float64)
    s = max(x1 - x0, y1 - y0) * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)), int(s)


def shift(box, pad):
    return tuple(v + pad for v in box)


def full_label_map(p, H, W):
    px0, py0, roi = p["parts_roi"]
    roi = np.asarray(roi)
    lab = np.zeros((H, W), np.uint8)
    h_, w_ = min(roi.shape[0], H - py0), min(roi.shape[1], W - px0)
    lab[py0:py0 + h_, px0:px0 + w_] = roi[:h_, :w_]
    return lab


def contact_colors_linear(p, rgb=(148, 0, 211)):
    """Per-vertex BGR 0-1, linear in the contact probability over 0-1: mesh grey -> violet."""
    t = np.clip(np.asarray(p, np.float64), 0, 1)[:, None]
    return (1 - t) * np.array([0.72, 0.74, 0.78]) + t * np.array(rgb[::-1]) / 255.0


def palm_view(v, c, prob, faces, size=540):
    """The whole hand from the view (yaw / pitch about its centroid, searched) that shows its contact side: front-facing
    contact vertices x the smallest 2D gap between the five fingertips, rendered on white, size x size."""
    from fah.demo_render import render_contact_smooth

    x = v - v.mean(0)
    fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    w = np.clip(np.asarray(prob, np.float64), 0, 1)
    best = None
    for yaw in np.radians(np.arange(0, 360, 15)):
        for pitch in np.radians(np.arange(-60, 61, 15)):
            cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
            R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            xr, nr = x @ R.T, vn @ R.T
            t2 = xr[list(TIPS_MANO), :2]
            gap = min(np.linalg.norm(t2[i] - t2[j]) for i in range(5) for j in range(i + 1, 5))
            score = ((nr[:, 2] < 0).astype(np.float64) * w).sum() / max(w.sum(), 1e-9) * gap
            if best is None or score > best[0]:
                best = (score, R)
    xr = x @ best[1].T
    span = np.ptp(xr[:, :2], 0).max()
    z0 = 0.6
    f = 0.85 * size * z0 / span
    vv = xr + np.array([0, 0, z0])
    uv = vv[:, :2] / vv[:, 2:] * f
    K = np.array([[f, 0, size / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2], [0, f, size / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2], [0, 0, 1]])
    return render_contact_smooth(np.full((size, size, 3), 255, np.uint8), [(vv, c)], faces, K, alpha=1.0, s=3)


def raster(v, f, K, size, col):
    """Painter's algorithm with Lambert shading on white; returns the image and the coverage mask."""
    uv = (K @ v.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    c = tri.mean(1)
    view = -c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    shade = 0.35 + 0.65 * np.abs((n * view).sum(1))
    W, H = size
    out = np.full((H, W, 3), 255, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    for i in np.argsort(-c[:, 2]):
        pts = np.round(uv[f[i]] * 4).astype(np.int32)  # 2 bits subpixel
        cv2.fillConvexPoly(out, pts, tuple(float(x) for x in (shade[i] * np.asarray(col) * 255)), cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(mask, pts, 255, shift=2)
    return out, mask


def load_frame(a):
    if a.frame_npz:
        z = np.load(a.frame_npz)
        return np.ascontiguousarray(z["image"]), z["K"].astype(np.float64)
    from bench.datasets.hot3d import Hot3DClip

    fr = Hot3DClip(a.hot3d_clip, a.mano_dir).frame(a.frame)
    return np.ascontiguousarray(fr["image"]), fr["K"].astype(np.float64)


def tiles_frame(a, img, K, preds, faces_mhr, ml):
    """p2-p7: the single-frame tiles from the image-mode predictions (right hand first)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bench.seg.rle import decode
    from fah.demo_render import render_contact_smooth
    from fah.vis import HAND_SKELETON, LEFT_COLOR, RIGHT_COLOR, draw_hand_skeleton, render_meshes
    from train.part_masks import COLORS as PART_COLORS

    H, W = img.shape[:2]
    OH = a.h
    side_bgr = lambda p: tuple(int(255 * c) for c in (RIGHT_COLOR if p["is_right"] else LEFT_COLOR))
    side_dark = lambda p: (200, 120, 30) if p["is_right"] else (30, 120, 230)
    meta = {}

    # detector: frame + boxes + instance masks, side colours
    fbox = crop_box([p["bbox"] for p in preds], W, H, 1.6, 0.45)
    det = img.copy()
    for p in preds:
        mk = decode(p["mask_det"]) > 0
        det[mk] = (0.45 * det[mk] + 0.55 * np.array(side_bgr(p))).astype(np.uint8)
        cnt, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(det, cnt, -1, side_dark(p), 3, cv2.LINE_AA)
        x0, y0, x1, y1 = np.asarray(p["bbox"]).astype(int)
        cv2.rectangle(det, (x0, y0), (x1, y1), side_dark(p), 4, cv2.LINE_AA)
        cv2.putText(det, "R" if p["is_right"] else "L", (x0 + 6, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, side_dark(p), 4, cv2.LINE_AA)
    write(a, "p2_detector", cut(det, fbox, OH))

    # the right hand's crop
    pr = preds[0]
    pl = [p for p in preds[1:]]
    cb, pad = hand_crop(pr)
    crop = cut(pad_black(img, pad), shift(cb, pad), OH)
    write(a, "p3_crop", crop)
    sc = OH / (cb[3] - cb[1])
    to_crop = lambda uv: (np.asarray(uv, np.float64) - np.array(cb[:2])) * sc

    # point head: 2D joints + visible dense points on the crop (finger colours), and the 3D points
    j0, v0 = ml(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
    nj = np.linalg.norm(v0[:, None] - j0[None], axis=-1).argmin(1)
    part = np.where(np.isin(nj, [0, 1, 5, 9, 13, 17]), 0, (nj - 1) // 4 + 1)
    vidx = a.dense_vidx
    vpart = part[vidx] if vidx is not None else part
    s2 = to_crop(pr["surface2d_head"])
    vis = np.asarray(pr.get("surface_visible", np.ones(len(s2), bool)))
    nb = len(s2) - len(vpart)
    pts = crop.copy()
    for k, ((u, v), sv) in enumerate(zip(s2.round().astype(int), vis)):
        if k >= nb and sv:
            cv2.circle(pts, (u, v), 3, FINGER_COLORS[vpart[k - nb]], -1, cv2.LINE_AA)
    pts = draw_hand_skeleton(pts, to_crop(pr["joints2d_head"]), thickness=3, radius=5)
    P3 = np.asarray(pr["surface3d_head"])[nb:] - np.asarray(pr["joints3d_head"])[0]
    J3 = np.asarray(pr["joints3d_head"]) - np.asarray(pr["joints3d_head"])[0]
    fig = plt.figure(figsize=(3, 3), dpi=OH / 3)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P3[:, 0], -P3[:, 1], -P3[:, 2], c=np.array(FINGER_COLORS)[vpart][:, ::-1] / 255.0, s=4, depthshade=True, linewidths=0)
    for e, (i0, i1) in enumerate(HAND_SKELETON):
        c = np.array(FINGER_COLORS[e // 4 + 1 if e // 4 < 5 else 0])[::-1] / 255.0
        ax.plot([J3[i0, 0], J3[i1, 0]], [-J3[i0, 1], -J3[i1, 1]], [-J3[i0, 2], -J3[i1, 2]], color=c, lw=1.5)
    r = np.percentile(np.abs(np.concatenate([P3, J3])), 99)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.view_init(elev=20, azim=-70)
    ax.set_box_aspect((1, 1, 1), zoom=2.6)
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    rgba = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    nz = np.argwhere(rgba[..., 3] > 0)
    (r0, q0), (r1, q1) = nz.min(0), nz.max(0)
    (cw, ch), (bx0, by0, bx1, by1) = PTS3D_LAYOUT
    inset = np.zeros((ch, cw, 4), np.uint8)
    inset[by0:by1, bx0:bx1] = cv2.resize(rgba[r0:r1 + 1, q0:q1 + 1], (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(a.out, "p4_points3d.png"), inset)

    # the wide window around the right hand (its mask and joints), the paper's tile aspect
    mr = decode(pr["mask_det"])
    ys, xs = np.where(mr)
    j2 = np.asarray(pr["joints2d"], np.float64)
    x0, x1 = min(xs.min(), j2[:, 0].min()), max(xs.max(), j2[:, 0].max())
    y0, y1 = min(ys.min(), j2[:, 1].min()), max(ys.max(), j2[:, 1].max())
    hgt = (y1 - y0) * 1.12
    wid = max(hgt * WIDE_ASPECT, (x1 - x0) * 1.12)
    hgt = wid / WIDE_ASPECT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.03 * hgt
    cx, cy = float(np.clip(cx, wid / 2, W - wid / 2)), float(np.clip(cy, hgt / 2, H - hgt / 2))
    box = (int(round(cx - wid / 2)), int(round(cy - hgt / 2)), int(round(cx + wid / 2)), int(round(cy + hgt / 2)))
    cutw = lambda im: cv2.resize(im[box[1]:box[3], box[0]:box[2]], (a.wide_w, int(round(a.wide_w / WIDE_ASPECT))), interpolation=cv2.INTER_AREA)
    big = img.copy()
    big[cb[1]:cb[3], cb[0]:cb[2]] = cv2.resize(pts, (cb[2] - cb[0], cb[3] - cb[1]), interpolation=cv2.INTER_CUBIC)
    write(a, "p4_points2d_wide", cutw(big))
    mm = [(np.asarray(p["verts3d_mhr"], np.float64), faces_mhr if p["is_right"] else faces_mhr[:, ::-1], RIGHT_COLOR if p["is_right"] else LEFT_COLOR)
          for p in preds if p.get("verts3d_mhr") is not None]
    write(a, "p5_mhr_wide", cutw(render_meshes(img, mm, K)))
    lab = full_label_map(pr, H, W)
    lab[~mr] = 0
    out = img.astype(np.float64)
    focus = cv2.GaussianBlur(cv2.dilate(mr.astype(np.uint8) * 255, np.ones((25, 25), np.uint8)).astype(np.float64) / 255, (0, 0), 12)
    fade = 0.35 * (1 - focus)
    for q in pl:
        mq = cv2.GaussianBlur(cv2.dilate(decode(q["mask_det"]).astype(np.uint8) * 255, np.ones((21, 21), np.uint8)).astype(np.float64) / 255, (0, 0), 8)
        fade = np.maximum(fade, 0.6 * mq * (1 - focus))
    out = out * (1 - fade[..., None]) + 255 * fade[..., None]
    mk = lab > 0
    out[mk] = 0.3 * out[mk] + 0.7 * PART_COLORS[np.minimum(lab[mk], 16)]
    out = out.clip(0, 255).astype(np.uint8)
    cnt, _ = cv2.findContours(mr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, (200, 120, 30), 2, cv2.LINE_AA)
    write(a, "p6_parts_wide", cutw(out))

    # MHR-Hand + fitted MANO outline from a novel view (yaw -60 deg about the vertical axis)
    vm, vh = np.asarray(pr["verts3d"], np.float64), np.asarray(pr["verts3d_mhr"], np.float64)
    c0 = vm.mean(0)
    th = np.deg2rad(-60.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    tm, th_ = (vm - c0) @ R.T, (vh - c0) @ R.T
    allv = np.concatenate([tm, th_])
    S, dist = 600, 0.8
    ext = np.abs(allv[:, :2]).max(0) * 1.06
    Wc, f_ = int(round(S * ext[0] / ext[1])), (S / 2) * dist / ext[1]
    Kv = np.array([[f_, 0, Wc / 2], [0, f_, S / 2], [0, 0, 1.0]])
    img3, _ = raster(th_ + [0, 0, dist], faces_mhr, Kv, (Wc, S), RIGHT_COLOR)
    _, m2 = raster(tm + [0, 0, dist], np.asarray(ml.faces[True]), Kv, (Wc, S), (0.5, 0.5, 0.5))
    cnt, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img3, cnt, -1, (38, 25, 191), 9, cv2.LINE_AA)
    nz = np.argwhere((img3 < 250).any(2))
    (r0, q0), (r1, q1) = nz.min(0), nz.max(0)
    mg = int(0.04 * S)
    img3 = cv2.copyMakeBorder(img3, mg, mg, mg, mg, cv2.BORDER_CONSTANT, value=(255, 255, 255))[r0:r1 + 2 * mg, q0:q1 + 2 * mg]
    cv2.imwrite(os.path.join(a.out, "p5_meshes3d.png"), img3)

    # contact: per-vertex probability on the right hand's fitted MANO mesh (grey -> violet, linear 0-1), and its contact side
    if pr.get("contact") is not None:
        v = np.asarray(pr["verts3d"], np.float64)
        faces = np.asarray(ml.faces[True])
        c = contact_colors_linear(pr["contact"])
        rend = render_contact_smooth(img, [(v, c)], faces, K, s=3)
        x0_, y0_, x1_, y1_ = a.contact_box
        tw, th2 = a.contact_size
        write(a, "p7_contact_wide", cv2.resize(rend[y0_:y1_, x0_:x1_], (tw, th2), interpolation=cv2.INTER_AREA), quality=95)
        cv2.imwrite(os.path.join(a.out, "p7_contact_wide_turned.png"), palm_view(v, c, pr["contact"], faces, a.contact_inset))
        meta["contact"] = dict(score=float(pr["contact_score"]), vertices_ge_05=int((np.asarray(pr["contact"]) >= 0.5).sum()))
    meta.update(crop_frame=fbox, crop_hand=cb, wide_window=box)
    return meta


def tiles_video(a, vid):
    """The video strip: frames of the clip with the video run's 2D joints under the thumb / index tip trails of the last
    --trail frames, one crop window for all of them."""
    from bench.datasets.hot3d import Hot3DClip
    from fah.vis import draw_hand_skeleton

    run = pickle.load(open(vid, "rb"))["frames"]
    clip = Hot3DClip(a.hot3d_clip, a.mano_dir)
    idx = list(range(a.strip_start, a.frame + 1, a.strip_step))
    imgs = [np.ascontiguousarray(clip.frame(i)["image"]) for i in idx]
    H, W = imgs[0].shape[:2]
    box = crop_box([h["bbox"] for i in idx for h in run[i]["pred"]], W, H, a.strip_aspect, a.strip_margin)
    for i, im in zip(idx, imgs):
        im = im.copy()
        for right in (False, True):
            for j in (4, 8):
                tr = [np.asarray(h["joints2d"], np.float64)[j] for f in range(max(0, i - a.trail), i + 1) for h in run[f]["pred"]
                      if bool(h["is_right"]) == right]
                if len(tr) > 1:
                    cv2.polylines(im, [np.round(np.array(tr)).astype(np.int32)], False, TRAIL_COLORS[right], 7, cv2.LINE_AA)
        for h in run[i]["pred"]:
            im = draw_hand_skeleton(im, np.asarray(h["joints2d"], np.float64), thickness=3, radius=4)
        x0, y0, x1, y1 = box
        c = im[y0:y1, x0:x1]
        write(a, f"strip_{i:03d}", cv2.resize(c, (a.strip_w, int(round(a.strip_w * c.shape[0] / c.shape[1]))), interpolation=cv2.INTER_AREA),
              quality=a.strip_quality)
    return dict(strip_frames=idx, strip_window=box)


def tile_world(a):
    """World space: camera path + frusta + both wrists (the video run placed with its camera trajectory), 10 cm bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    RIGHT, LEFT = np.array([0.98, 0.70, 0.45]), np.array([0.55, 0.80, 0.98])  # BGR, as fah.vis
    ref = pickle.load(open(a.vid_run, "rb"))["frames"]
    z = np.load(a.cams, allow_pickle=True)
    T, idx = z["T_world_cam"], list(z["index"])
    h = a.world_h
    fig = plt.figure(figsize=(1.4 * 3, 3), dpi=h / 3)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    C = T[:, :3, 3]
    ax.plot(C[:, 0], C[:, 1], C[:, 2], color=(0.45, 0.45, 0.45), lw=3.2)
    fz = 0.05
    for i in np.linspace(0, len(T) - 1, 6).round().astype(int):
        Rw, c = T[i, :3, :3], T[i, :3, 3]
        corners = [Rw @ np.array([sx * fz * 0.8, sy * fz * 0.6, fz]) + c for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for q in corners:
            ax.plot(*zip(c, q), color=(0.7, 0.7, 0.7), lw=1.4)
        ax.plot(*zip(*(corners + corners[:1])), color=(0.7, 0.7, 0.7), lw=1.4)
    HP = []
    for right, col in ((True, RIGHT), (False, LEFT)):
        P = []
        for fi, f in enumerate(ref):
            hs = [h_ for h_ in f["pred"] if bool(h_["is_right"]) == right]
            k = idx.index(fi) if fi in idx else None
            if hs and k is not None:
                P.append(T[k, :3, :3] @ np.asarray(hs[0]["joints3d"])[0] + T[k, :3, 3])
        if P:
            P = np.array(P)
            P = P[np.linalg.norm(P - np.median(P, 0), axis=1) < 0.35]
            HP.append(P)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col[::-1] * 0.8, lw=5.0)
    allp = np.concatenate([C] + HP)
    mid = (allp.min(0) + allp.max(0)) / 2
    r = np.abs(allp - mid).max() + 0.02
    lim = lambda: (ax.set_xlim(mid[0] - r, mid[0] + r), ax.set_ylim(mid[1] - r, mid[1] + r), ax.set_zlim(mid[2] - r, mid[2] + r))
    lim()
    welev, wazim = 25.0, 150.0
    ax.view_init(elev=welev, azim=wazim)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()

    def scr(p):
        x2, y2, _ = proj3d.proj_transform(p[..., 0], p[..., 1], p[..., 2], ax.get_proj())
        return ax.transData.transform(np.stack([np.ravel(x2), np.ravel(y2)], 1))

    S = scr(allp)
    az, el = np.radians(wazim), np.radians(welev)
    cands = [np.array([-np.sin(az), np.cos(az), 0.0]), np.array([np.sin(az), -np.cos(az), 0.0])]
    up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
    k = int(np.argmin(S[:, 1] - 0.3 * (S[:, 0] - S[:, 0].min())))  # low and left on screen
    best = None
    for right in cands:  # a 10 cm bar below the lowest point, horizontal and pointing right on screen
        for su in (up, -up):
            b0 = allp[k] + 0.05 * su
            e = scr(np.stack([b0, b0 + 0.1 * right]))
            d = e[1] - e[0]
            sc = (d[0] > 0) * 10 - abs(d[1]) / (abs(d[0]) + 1e-9) - (e[0][1] > S[k][1])
            if best is None or sc > best[0]:
                best = (sc, b0, b0 + 0.1 * right)
    b0, b1 = best[1], best[2]
    ax.plot([b0[0], b1[0]], [b0[1], b1[1]], [b0[2], b1[2]], color="black", lw=4.5)
    lim()
    fig.canvas.draw()
    px = []
    for p in (b0, b1):
        x2, y2, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
        X, Y = ax.transData.transform((x2, y2))
        px.append([float(X), float(fig.bbox.height - Y)])  # image coordinates (origin top-left)
    fig.canvas.draw()
    full = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    plt.close(fig)
    x0, y0, x1, y1 = a.world_crop
    out = full[y0:y1, x0:x1]
    write(a, "p9_world", cv2.resize(out, (int(round(out.shape[1] * a.world_scale)), int(round(out.shape[0] * a.world_scale))), interpolation=cv2.INTER_AREA))
    return dict(world_bar_px=px, world_full_size=list(full.shape[:2][::-1]))


def write(a, name, im, quality=None):
    cv2.imwrite(os.path.join(a.out, name + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, quality or a.jpeg_quality])


def main(a):
    import fah
    from bench.common import ManoLayer

    os.makedirs(a.out, exist_ok=True)
    img, K = load_frame(a)
    m = fah.FAH("fah", "image", weights=a.weights, sam3d=a.sam3d, trt_dir=a.trt_dir, mano_dir=a.mano_dir, mesh="both", contact=True,
                det_masks=True, parts=True)
    preds = sorted(m(img, K), key=lambda p: not p["is_right"])  # right hand first
    vidx = getattr(m.method.est.model.keypoint_head, "dense_vidx", None)
    a.dense_vidx = vidx.cpu().numpy() if vidx is not None else None
    ml = ManoLayer(a.mano_dir)
    meta = tiles_frame(a, img, K, preds, np.asarray(m.faces_mhr), ml)
    if a.vid_run:
        meta.update(tiles_video(a, a.vid_run))
        if a.cams:
            meta.update(tile_world(a))
    meta["hands"] = [dict(is_right=bool(p["is_right"]), score=float(p["score"])) for p in preds]
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hot3d_clip", default="", help="the HOT3D clip .tar (frame source; needed for the video strip)")
    p.add_argument("--frame_npz", default="", help="alternatively an .npz with image (H, W, 3) BGR and K (3, 3)")
    p.add_argument("--frame", type=int, default=110)
    p.add_argument("--weights", required=True)
    p.add_argument("--sam3d", default="")
    p.add_argument("--trt_dir", default="", help="TensorRT engines (default: eager PyTorch, as the paper's tiles)")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--vid_run", default="", help="bench.run video-mode predictions of the clip (strip, world)")
    p.add_argument("--cams", default="", help="camera trajectory of the clip (npz: T_world_cam, index)")
    p.add_argument("--h", type=int, default=420, help="tile height of the detector / crop tiles")
    p.add_argument("--wide_w", type=int, default=640, help="width of the wide tiles")
    p.add_argument("--contact_box", type=int, nargs=4, default=[525, 906, 1207, 1312], help="frame window of the contact tile (x0 y0 x1 y1)")
    p.add_argument("--contact_size", type=int, nargs=2, default=[640, 380])
    p.add_argument("--contact_inset", type=int, default=540)
    p.add_argument("--strip_start", type=int, default=86)
    p.add_argument("--strip_step", type=int, default=4)
    p.add_argument("--strip_aspect", type=float, default=1.3)
    p.add_argument("--strip_margin", type=float, default=0.10)
    p.add_argument("--strip_w", type=int, default=640)
    p.add_argument("--strip_quality", type=int, default=90)
    p.add_argument("--trail", type=int, default=24, help="frames of the video strip's tip trails")
    p.add_argument("--world_h", type=int, default=840)
    p.add_argument("--world_crop", type=int, nargs=4, default=[266, 164, 983, 794], help="window of the rendered world panel (x0 y0 x1 y1)")
    p.add_argument("--world_scale", type=float, default=1.0)
    p.add_argument("--jpeg_quality", type=int, default=92)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse())
