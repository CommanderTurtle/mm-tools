"""Drawing for fah-video (fah.video_demo): whole-hand meshes with per-vertex contact colours (z-buffered, smooth shading),
the palm-side contact view, the per-hand contact timeline, the world tile (hands + camera path) and the panel headers."""
import cv2
import numpy as np

VIOLET = np.array([211, 0, 148]) / 255.0  # BGR
GREY = np.array([0.72, 0.74, 0.78])
TIPS_MANO = (744, 320, 443, 554, 671)  # thumb .. pinky fingertip vertices of the MANO template
R_COL, L_COL, CAM_COL, FR_COL = (205, 145, 91), (110, 160, 200), (110, 110, 110), (175, 175, 175)  # BGR (world tile)


def render_contact_smooth(img, hands, faces, K, alpha=0.95, s=2):
    """Z-buffered rendering of whole (verts (778, 3) camera frame, per-vertex colours) hands with per-vertex normals and
    Gouraud-interpolated Lambert shading; nothing of the hand is clipped (only its own hidden surface is resolved)."""
    h, w = img.shape[:2]
    Ks = K.astype(np.float64).copy()
    Ks[:2] *= s
    H, W = h * s, w * s
    zbuf, cbuf = np.full((H, W), np.inf), np.zeros((H, W, 3))
    for v, c in hands:
        if (v[:, 2] <= 1e-3).any():
            continue
        uv = v[:, :2] / v[:, 2:] * Ks[[0, 1], [0, 1]] + Ks[[0, 1], [2, 2]]
        tri = v[faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        view = -v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        vcol = (0.35 + 0.65 * np.abs((vn * view).sum(1)))[:, None] * c
        for f in faces:
            p2, z, col = uv[f], v[f, 2], vcol[f]
            x0, y0 = np.floor(p2.min(0)).astype(int)
            x1, y1 = np.ceil(p2.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, ay), (bx, by), (cx, cy) = p2
            d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(d) < 1e-12:
                continue
            l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
            l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
            l2 = 1 - l0 - l1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
            sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zz < sub_z)
            sub_z[upd] = zz[upd]
            cc = l0[..., None] * col[0] + l1[..., None] * col[1] + l2[..., None] * col[2]
            cbuf[y0:y1 + 1, x0:x1 + 1][upd] = cc[upd]
    m = np.isfinite(zbuf).astype(np.float64)
    canvas = cv2.resize((cbuf * 255).clip(0, 255).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    mm = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)[..., None]
    canvas = canvas / np.maximum(mm, 1e-6)  # supersampled coverage: un-premultiply the edge pixels
    return (img * (1 - alpha * mm) + canvas * (alpha * mm)).clip(0, 255).astype(np.uint8)


def contact_colors_ramp(p, lo=0.3, hi=0.7):
    """Per-vertex BGR 0-1: mesh grey -> saturated violet, ramp lo-hi (half violet at 0.5)."""
    t = np.clip((np.asarray(p, np.float64) - lo) / (hi - lo), 0, 1)[:, None]
    return (1 - t) * GREY + t * VIOLET


def contact_colors_linear(p):
    """Per-vertex BGR 0-1, linear in the contact probability over 0-1: mesh grey -> violet (the paper's contact figures)."""
    t = np.clip(np.asarray(p, np.float64), 0, 1)[:, None]
    return (1 - t) * GREY + t * VIOLET


def kabsch(src, dst):
    """Rotation R with R (src - mean) ~ dst - mean (no scale)."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def best_view(x, prob, faces):
    """Yaw / pitch rotation (about the centroid of x) that shows the most contact on front-facing vertices and separates the
    fingertips."""
    x = x - x.mean(0)
    fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    w = np.clip(np.asarray(prob, np.float64), 0, 1) + 1e-3
    best = None
    for yaw in np.radians(np.arange(0, 360, 15)):
        for pitch in np.radians(np.arange(-60, 61, 15)):
            cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
            R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            xr, nr = x @ R.T, vn @ R.T
            t2 = xr[list(TIPS_MANO), :2]
            gap = min(np.linalg.norm(t2[i] - t2[j]) for i in range(5) for j in range(i + 1, 5))
            score = ((nr[:, 2] < 0) * w).sum() / w.sum() * gap
            if best is None or score > best[0]:
                best = (score, R)
    return best[1]


def render_view(x, c, faces, R, span, size, bg=255):
    """The mesh x (centred) rotated by R, rendered on a plain background, scaled so that `span` (m) fills 85 % of size."""
    xr = (x - x.mean(0)) @ R.T
    z0 = 0.6
    f = 0.85 * size * z0 / span
    vv = xr + np.array([0, 0, z0])
    uv = vv[:, :2] / vv[:, 2:] * f
    K = np.array([[f, 0, size / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2], [0, f, size / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2], [0, 0, 1.0]])
    return render_contact_smooth(np.full((size, size, 3), bg, np.uint8), [(vv, c)], faces, K, alpha=1.0, s=2)


def timeline(pred, flag, gt, W, fps, H=300, flag_img=None, names=True):
    """Static strip: per side a lane (R above, L below) with the predicted contact score (line), the predicted flag (bar), the
    image-only flag (thin light bar, if given: video vs image stability) and the GT flag (faint bar, if any); time axis in
    seconds. Returns the strip and the frame -> x mapping (for a cursor). names=False: no model name in the legend."""
    img = np.full((H, W, 3), 255, np.uint8)
    n, x0, x1 = len(pred), 70, W - 20
    X = lambda t: int(x0 + (x1 - x0) * t / max(n - 1, 1))
    lanes = [(10, "R"), (120, "L")]
    for k, (y0, name) in enumerate(lanes):
        cv2.putText(img, name, (14, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y0 + 70), (238, 238, 238), -1)
        cv2.line(img, (x0, y0 + 35), (x1, y0 + 35), (205, 205, 205), 1)  # score 0.5
        for t in range(n):
            if flag[t, k] == 1:
                cv2.rectangle(img, (X(t), y0 + 74), (X(t + 1), y0 + 84), tuple(int(v * 255) for v in VIOLET), -1)
            if flag_img is not None and flag_img[t, k] == 1:
                cv2.rectangle(img, (X(t), y0 + 87), (X(t + 1), y0 + 91), (230, 160, 215), -1)
            if gt[t, k] == 1:
                cv2.rectangle(img, (X(t), y0 + 95), (X(t + 1), y0 + 100), (150, 150, 150), -1)
        pts = [(X(t), int(y0 + 70 - 70 * pred[t, k])) for t in range(n) if np.isfinite(pred[t, k])]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            if p1[0] - p0[0] <= 2 * max((x1 - x0) / max(n - 1, 1), 1):
                cv2.line(img, p0, p1, tuple(int(v * 255) for v in VIOLET), 2, cv2.LINE_AA)
    yA = 232
    cv2.line(img, (x0, yA), (x1, yA), (0, 0, 0), 1)
    for sec in range(int((n - 1) / fps) + 1):
        cv2.line(img, (X(sec * fps), yA - 4), (X(sec * fps), yA + 4), (0, 0, 0), 1)
        cv2.putText(img, f"{sec} s", (X(sec * fps) - 10, yA + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    legend = (f"line: contact score (0-1)   violet bar: {'FAH-Video' if names else 'video'} contact   thin pink bar: image-only contact"
              if flag_img is not None else "line: contact score (0-1)   violet bar: predicted contact")
    if np.isfinite(gt).any():
        legend += "   grey bar: ground truth"
    cv2.putText(img, legend, (x0, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "colour = per-vertex contact probability, grey <= 0.3 -> violet >= 0.7; header = hand-level contact flag", (x0, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    return img, X


def bar(img, text, h=30, s=0.62):
    """A panel title on a dark translucent bar at the top left."""
    out = img.copy()
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    sub = out[:h, :tw + 20]
    sub[:] = (0.35 * sub + 0.65 * np.array([40, 40, 40])).astype(np.uint8)
    cv2.putText(out, text, (10, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def joints_and_parts(img, hands, tile_w):
    """2D joints, and the part maps + instance masks, drawn at the input resolution (line widths scaled to the tile)."""
    from bench.seg.rle import decode
    from train.part_masks import COLORS

    from .vis import FINGER_COLORS, HAND_SKELETON

    k = max(1.0, img.shape[1] / tile_w)
    j2d, pm = img.copy(), img.copy()
    for h in hands:
        p = np.round(np.asarray(h["joints2d"])).astype(int)
        for e, (a, b) in enumerate(HAND_SKELETON):
            cv2.line(j2d, tuple(p[a]), tuple(p[b]), FINGER_COLORS[e // 4], int(round(2.5 * k)), cv2.LINE_AA)
        for j, q in enumerate(p):
            cv2.circle(j2d, tuple(q), int(round(3.5 * k)), (255, 255, 255) if j == 0 else FINGER_COLORS[(j - 1) // 4], -1, cv2.LINE_AA)
        if h.get("mask_det") is not None:
            m = decode(h["mask_det"])
            col = np.array([255, 128, 0] if h["is_right"] else [0, 128, 255], np.float32)
            pm[m] = (pm[m] * 0.6 + col * 0.4).astype(np.uint8)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pm, cnt, -1, tuple(int(c) for c in col), int(round(1.5 * k)), cv2.LINE_AA)
    H, W = img.shape[:2]
    for h in hands:
        if h.get("parts_roi") is None:
            continue
        x0, y0, png = h["parts_roi"]
        roi = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero((roi > 0) & (roi < len(COLORS)))
        ys, xs, lab = ys + y0, xs + x0, roi[ys, xs]
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        ys, xs, lab = ys[ok], xs[ok], lab[ok]
        pm[ys, xs] = (pm[ys, xs] * 0.4 + COLORS[lab] * 0.6).astype(np.uint8)
    return j2d, pm


def to_world(T, x):
    return np.einsum("nij,nkj->nki", T[:, :3, :3], x) + T[:, None, :3, 3]


def wrists(frames, min_score=0.3):
    """(N, 2, 3) camera-frame wrist per side (R, L) of the best shown hand; nan if none."""
    out = np.full((len(frames), 2, 3), np.nan)
    for t, fr in enumerate(frames):
        for k, side in enumerate((True, False)):
            hs = [h for h in fr["pred"] if bool(h["is_right"]) == side and h["score"] >= min_score]
            if hs:
                out[t, k] = np.asarray(max(hs, key=lambda h: h["score"])["joints3d"], np.float64)[0]
    return out


class WorldTile:
    """World tile for one camera trajectory T (N, 4, 4) T_world_cam: wrist trails (R blue / L tan), the camera path and
    frusta, a 10 cm bar; fixed orthographic view from above / behind the mean camera (tilt deg)."""

    def __init__(self, T, wc, size, name, tilt=35.0, every=30, scale_bar=0.1, trail=90):
        n = min(len(T), len(wc))
        self.T, self.ww, self.name, self.size, self.every, self.bar = T[:n], to_world(T[:n], wc[:n]), name, size, every, scale_bar
        self.trail = trail  # the last `trail` frames in full colour, the earlier path light
        R = T[:n, :3, :3]
        fwd, right = R[:, :, 2].mean(0), R[:, :, 0].mean(0)
        fwd /= np.linalg.norm(fwd)
        right -= fwd * (right @ fwd)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        a = np.radians(tilt)  # look more from above: rotate the view direction toward -down about the right axis
        d = np.cos(a) * fwd + np.sin(a) * down
        self.u, self.v = right, np.cross(d, right)
        pts = np.concatenate([T[:n, :3, 3], self.ww.reshape(-1, 3)])
        pts = pts[np.isfinite(pts[:, 0])]
        p2 = np.stack([pts @ self.u, pts @ self.v], 1)
        lo, hi = np.percentile(p2, 0.5, 0), np.percentile(p2, 99.5, 0)
        ctr, span = (lo + hi) / 2, (hi - lo).max() * 1.15 + 1e-6
        self.ctr, self.s = ctr, min(size) / span * 0.92 if isinstance(size, tuple) else size / span

    def px(self, x):
        p = np.stack([x @ self.u, x @ self.v], -1)
        W, H = self.size
        return np.stack([W / 2 + (p[..., 0] - self.ctr[0]) * self.s, H / 2 + (p[..., 1] - self.ctr[1]) * self.s], -1)

    def frustum(self, img, T, col, th, depth=0.05, f=0.6):
        c = T[:3, 3]
        corners = np.array([[-f, -f * 0.55, 1], [f, -f * 0.55, 1], [f, f * 0.55, 1], [-f, f * 0.55, 1]]) * depth
        q = self.px(np.concatenate([c[None], corners @ T[:3, :3].T + c]))
        q = np.round(q * 4).astype(np.int32)
        for i in range(4):
            cv2.line(img, tuple(q[0]), tuple(q[1 + i]), col, th, cv2.LINE_AA, 2)
            cv2.line(img, tuple(q[1 + i]), tuple(q[1 + (i + 1) % 4]), col, th, cv2.LINE_AA, 2)

    def draw(self, t):
        W, H = self.size
        img = np.full((H, W, 3), 255, np.uint8)
        t = min(t, len(self.T) - 1)
        lw = max(2, int(round(min(W, H) / 70)))
        light = lambda c: tuple(int(0.35 * x + 0.65 * 255) for x in c)
        t0 = max(0, t - self.trail)

        def path(P, col, w):  # break the polyline where the hand was not shown
            seg = []
            for p in list(P) + [np.array([np.nan, np.nan])]:
                if np.isfinite(p[0]):
                    seg.append(p)
                elif seg:
                    cv2.polylines(img, [np.round(np.array(seg) * 4).astype(np.int32)], False, col, w, cv2.LINE_AA, 2)
                    seg = []

        C = self.px(self.T[:t + 1, :3, 3])
        path(C[:t0 + 1], light(CAM_COL), lw)
        for k in list(range(0, t, self.every)):
            self.frustum(img, self.T[k], FR_COL, 1)
        for k, col in ((1, L_COL), (0, R_COL)):
            path(self.px(self.ww[:t0 + 1, k]), light(col), lw)
        path(C[t0:], CAM_COL, lw)
        self.frustum(img, self.T[t], (60, 60, 60), 2)
        for k, col in ((1, L_COL), (0, R_COL)):
            P = self.px(self.ww[t0:t + 1, k])
            path(P, col, lw)
            if np.isfinite(P[-1, 0]):
                cv2.circle(img, tuple(np.round(P[-1]).astype(int)), lw + 4, col, -1, cv2.LINE_AA)
        L = int(round(self.bar * self.s))
        cv2.rectangle(img, (W - 20 - L, H - 26), (W - 20, H - 18), (0, 0, 0), -1)
        cv2.putText(img, f"{int(self.bar * 100)} cm", (W - 20 - L, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, self.name, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
        return img
