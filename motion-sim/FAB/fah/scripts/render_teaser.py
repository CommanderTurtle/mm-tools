"""The paper's Fig. 1: one FAH forward pass per image, five outputs per scene, cropped around all detected hands.
Panels: 2D keypoints (heatmap joints) | dense surface points at their 2D positions (finger colours; visible points only) |
fitted MANO mesh | part labels inside the detector's instance mask (outlined in the hand's side colour) | contact: each
hand's mesh turned to its contact side (the automatic yaw / pitch search of palm_view; grey 0 -> violet 1, linear in the
per-vertex probability), hands side by side in image x order, labelled R / L. One cell per image (<out>/<name>.png / .jpg,
5 tiles of --tile px plus 2 px white borders), optionally assembled into a grid (--grid).

Paper configuration (Fig. 1): FAH-H with its part head and contact head (identity NMS by wrist distance, the MANO stage
relative to the target wrist), mask-guided fingertips on, focal = the longer image side, margin 0.04, tile 300, two
columns:

    python scripts/render_teaser.py --model fah-h --margin 0.04 --tile 300 \
        --images keyboard.jpg,knitting.jpg,pottery.jpg,<bench2d v2>/images/hot3d/clip-001910_000110.jpg,lego.jpg,rubik.jpg,\
<bench2d v2>/images/hot3d/clip-001892_000090.jpg,<bench2d v2>/images/hot3d/clip-001903_000140.jpg,\
<bench2d v2>/images/hint_epick/EK_0107_P21_01_frame_0000010525.jpg,<bench2d v2>/images/hint_epick/EK_0003_P01_107_frame_0000002926.jpg \
        --names 01_keyboard,02_knitting,03_pottery,04_hot3d_milk,05_lego,06_rubik,07_hot3d_glasses,08_hot3d_ring,09_epic_cup,10_epic_box \
        --out fig1 --grid fig1/fig1.jpg

Image credits (Fig. 1): keyboard: Wikimedia Commons "Keyboard with fingers", Cangjie6, CC BY-SA 4.0; knitting: "Oma strickt
Strümpfe", Sadarama, CC BY-SA 4.0; pottery: "Earthen Potter Hands", Earthen Potter, CC BY-SA 4.0; Lego: "Playing with Lego
bricks", Nenad Stojkovic, CC BY 2.0; Rubik's cube: "Rubik cube (345056969)", theilr, CC BY-SA 2.0; HOT3D frames: HOT3D test
split, CC BY-SA 4.0 (hand annotations CC BY-NC-SA 4.0); EPIC frames: EPIC-KITCHENS-100 via the HInt test split, CC BY-NC 4.0.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fah.demo_render import GREY, TIPS_MANO, render_contact_smooth  # noqa: E402
from fah.vis import LEFT_COLOR, RIGHT_COLOR, draw_hand_skeleton, render_meshes  # noqa: E402

FINGER_COLORS = [(40, 200, 230), (60, 60, 255), (0, 170, 255), (60, 220, 60), (255, 160, 0), (220, 60, 220)]  # palm (yellow), thumb..pinky
MASK_COLORS = {True: (255, 160, 60), False: (60, 170, 255)}
LABELS = ["2D keypoints", "dense surface points", "fitted MANO mesh", "part segmentation", "contact"]


def contact_colors_linear(p, rgb=(148, 0, 211)):
    """Per-vertex BGR 0-1, linear in the contact probability over 0-1: mesh grey -> violet."""
    t = np.clip(np.asarray(p, np.float64), 0, 1)[:, None]
    return (1 - t) * GREY + t * np.array(rgb[::-1]) / 255.0


def palm_view(v, c, prob, faces, size=540):
    """The whole hand from the view (yaw / pitch about its centroid, searched) that shows its contact side: front-facing
    contact vertices x the smallest 2D gap between the five fingertips (every finger visible), rendered on white, size^2."""
    ctr = v.mean(0)
    x = v - ctr
    fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    w = np.clip(np.asarray(prob, np.float64), 0, 1)  # contact probability per vertex
    best = None
    for yaw in np.radians(np.arange(0, 360, 15)):
        for pitch in np.radians(np.arange(-60, 61, 15)):
            cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
            R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            xr, nr = x @ R.T, vn @ R.T
            front = (nr[:, 2] < 0).astype(np.float64)  # camera looks along +z
            t2 = xr[list(TIPS_MANO), :2]
            gap = min(np.linalg.norm(t2[i] - t2[j]) for i in range(5) for j in range(i + 1, 5))
            score = (front * w).sum() / max(w.sum(), 1e-9) * gap
            if best is None or score > best[0]:
                best = (score, R)
    xr = x @ best[1].T
    span = np.ptp(xr[:, :2], 0).max()
    z0 = 0.6
    f = 0.85 * size * z0 / span
    vv = xr + np.array([0, 0, z0])
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    uv = vv[:, :2] / vv[:, 2:] * f
    K[0, 2], K[1, 2] = size / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2, size / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2
    return render_contact_smooth(np.full((size, size, 3), 255, np.uint8), [(vv, c)], faces, K, alpha=1.0, s=3)


def panels(img, hands, K, faces, vpart):
    """The four image panels of one image: 2D keypoints | dense points | fitted MANO mesh | parts in the mask."""
    from bench.seg.rle import decode
    from train.part_masks import COLORS as PART_COLORS

    kp, dense, mesh, mask = img.copy(), img.copy(), img.copy(), img.copy()
    meshes = []
    for p in hands:
        kp = draw_hand_skeleton(kp, np.asarray(p.get("joints2d_head", p["joints2d"])), thickness=3, radius=4)
        s2 = p.get("surface2d_head")
        if s2 is not None:
            s2 = np.asarray(s2)
            vis = np.asarray(p.get("surface_visible", np.ones(len(s2), bool)))
            nb = len(s2) - len(vpart)
            r = max(2, int(round(2 * max(img.shape[:2]) / 1000)))
            for k, ((u, v), sv) in enumerate(zip(s2.round().astype(int), vis)):
                if k >= nb and sv:  # visible surface points only (hidden ones take their 2D from the 3D)
                    cv2.circle(dense, (u, v), r, FINGER_COLORS[vpart[k - nb]], -1, cv2.LINE_AA)
        if p.get("verts3d") is not None:
            meshes.append((np.asarray(p["verts3d"], np.float64), faces[bool(p["is_right"])], RIGHT_COLOR if p["is_right"] else LEFT_COLOR))
        if p.get("parts_roi") is not None:  # part segmentation inside the instance mask, outlined in the hand's side colour
            x0, y0, roi = p["parts_roi"]
            lab = np.zeros(img.shape[:2], np.uint8)
            h_, w_ = min(roi.shape[0], lab.shape[0] - y0), min(roi.shape[1], lab.shape[1] - x0)
            lab[y0:y0 + h_, x0:x0 + w_] = roi[:h_, :w_]
            m = lab > 0
            mask[m] = (0.35 * mask[m] + 0.65 * PART_COLORS[np.minimum(lab[m], 16)]).astype(np.uint8)
            c = np.array(MASK_COLORS[bool(p["is_right"])], np.float64)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, cnt, -1, tuple(int(x) for x in c), 2)
        elif p.get("mask_rle") is not None:
            m = decode(p["mask_rle"]) > 0
            c = np.array(MASK_COLORS[bool(p["is_right"])], np.float64)
            mask[m] = (0.45 * mask[m] + 0.55 * c).astype(np.uint8)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, cnt, -1, tuple(int(x) for x in c), 2)
    mesh = render_meshes(mesh, meshes, K) if meshes else mesh
    return [kp, dense, mesh, mask]


def crop_window(boxes, w, h, margin, aspect):
    """Window around all hand boxes (xyxy), enlarged by margin per side, at the tile aspect (w / h), inside the image."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    bw, bh = max(bw, bh * aspect), max(bh, bw / aspect)
    s = min(1.0, w / bw, h / bh)
    bw, bh = bw * s, bh * s
    cx = float(np.clip((x0 + x1) / 2, bw / 2, w - bw / 2))
    cy = float(np.clip((y0 + y1) / 2, bh / 2, h - bh / 2))
    return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)


def contact_tile(hands, faces, tw, th, ss=2, fill=0.85, label_px=28):
    """Each hand's mesh at its contact side, side by side in image x order on white (th x tw), labelled R / L in the top
    band (outer corner of each slot)."""
    hands = sorted([h for h in hands if h.get("verts3d") is not None and h.get("contact") is not None],
                   key=lambda h: (h["bbox"][0] + h["bbox"][2]) / 2)
    H, W = th * ss, tw * ss
    canvas = np.full((H, W, 3), 255, np.uint8)
    n = len(hands)
    if not n:
        return cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_AREA)
    cw = W // n
    for i, hnd in enumerate(hands):
        prob = np.asarray(hnd["contact"], np.float64)
        pv = palm_view(np.asarray(hnd["verts3d"], np.float64), contact_colors_linear(prob), prob, faces[bool(hnd["is_right"])], size=4 * max(cw, H))
        ys, xs = np.nonzero((pv < 250).any(2))
        pv = pv[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # the hand's own extent, then scaled to fill the slot
        sc = min(fill * H / pv.shape[0], 0.96 * cw / pv.shape[1])
        pv = cv2.resize(pv, (max(1, int(round(pv.shape[1] * sc))), max(1, int(round(pv.shape[0] * sc)))), interpolation=cv2.INTER_AREA)
        band = int(round((label_px + 6) * ss))  # top band for the labels
        y0 = min(H - pv.shape[0] - int(0.01 * H), max((H - pv.shape[0]) // 2, band))  # centred, but below the label band
        x0 = i * cw + (cw - pv.shape[1]) // 2
        canvas[y0:y0 + pv.shape[0], x0:x0 + pv.shape[1]] = pv
    for i, hnd in enumerate(hands):
        lab = "R" if hnd["is_right"] else "L"
        fs = label_px * ss / 21.5  # Hershey simplex cap height ~21.5 px at scale 1
        t = max(2, int(round(0.11 * label_px * ss)))
        (lw, lh), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        m = 3 * ss
        org = (i * cw + m, m + lh) if (n == 1 or i < n - 1) else (i * cw + cw - m - lw, m + lh)
        cv2.putText(canvas, lab, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t + 3 * ss, cv2.LINE_AA)
        cv2.putText(canvas, lab, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (50, 50, 50), t, cv2.LINE_AA)
    return cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_AREA)


def cell(img, hands, K, faces, vpart, a):
    """One scene's 5-panel cell."""
    h, w = img.shape[:2]
    tiles = panels(img, hands, K, faces, vpart)
    if hands:  # crop around all hands (fixed tile aspect, with margin)
        x0, y0, x1, y1 = crop_window([p["bbox"] for p in hands], w, h, a.margin, a.aspect)
        tiles = [t[y0:y1, x0:x1] for t in tiles]
    tiles = [cv2.resize(t, (a.tile, int(round(a.tile * t.shape[0] / t.shape[1]))), interpolation=cv2.INTER_AREA) for t in tiles]
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    if a.contact:
        tiles.append(contact_tile(hands, faces, a.tile, H))
    return np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles], 1)


def grid(cells, cols, tile, labels=LABELS, gap_x=8, gap_y=6, head=34):
    """Cells in rows of `cols` (white gaps: gap_x between cells, gap_y under each row) under a header naming the panels."""
    W = cols * cells[0].shape[1] + (cols - 1) * gap_x
    rows = []
    for r in range(0, len(cells), cols):
        row = cells[r:r + cols]
        H = max(c.shape[0] for c in row)
        out = np.full((H + gap_y, W, 3), 255, np.uint8)
        for i, c in enumerate(row):
            x = i * (cells[0].shape[1] + gap_x)
            out[:c.shape[0], x:x + c.shape[1]] = c
        rows.append(out)
    top = np.full((head, W, 3), 255, np.uint8)
    for c in range(cols):
        for k, lab in enumerate(labels[:cells[0].shape[1] // (tile + 4)]):
            cv2.putText(top, lab, (c * (cells[0].shape[1] + gap_x) + k * (tile + 4) + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([top] + rows, 0)


def vertex_parts(mano_dir, keypoint_head):
    """MANO vertex -> palm (0) / finger 1-5 (nearest rest-pose joint), on the network's dense points."""
    from bench.common import ManoLayer

    ml = ManoLayer(mano_dir)
    j0, v0 = ml(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
    nj = np.linalg.norm(v0[:, None] - j0[None], axis=-1).argmin(1)
    part = np.where(np.isin(nj, [0, 1, 5, 9, 13, 17]), 0, (nj - 1) // 4 + 1)
    vidx = getattr(keypoint_head, "dense_vidx", None)
    return ml, part[vidx.cpu().numpy()] if vidx is not None else part


def main(a):
    from fah import FAH

    extra = dict(mask_split=bool(a.mask_split))
    fah = FAH(a.model, "image", weights=a.weights, sam3d=a.sam3d, trt_dir=a.trt_dir, mano_dir=a.mano_dir, mesh="mano",
              parts=True, det_masks=True, contact=bool(a.contact), **extra)
    ml, vpart = vertex_parts(a.mano_dir or os.environ.get("MANO_DIR", "data/mano"), fah.method.est.model.keypoint_head)
    images = a.images.split(",")
    names = a.names.split(",") if a.names else [os.path.splitext(os.path.basename(p))[0] for p in images]
    os.makedirs(a.out, exist_ok=True)
    cells = []
    for path, name in zip(images, names):
        img = cv2.imread(path)
        s = a.max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        f = a.focal_ratio * max(h, w)
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], np.float64)
        fah.reset()
        hands = fah(img, K)
        c = cell(img, hands, K, ml.faces, vpart, a)
        cv2.imwrite(os.path.join(a.out, f"{name}.png"), c)
        cv2.imwrite(os.path.join(a.out, f"{name}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cells.append(c)
        print(name, len(hands), "hands; cell", c.shape[1], "x", c.shape[0], flush=True)
    if a.grid:
        cv2.imwrite(a.grid, grid(cells, a.cols, a.tile), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("->", a.grid)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="comma-separated image paths")
    p.add_argument("--names", default="", help="comma-separated output names (default: the image file names)")
    p.add_argument("--model", default="fah-h")
    p.add_argument("--weights", default=os.environ.get("FAH_WEIGHTS", ""), help="FAH weights directory (default: FAH_WEIGHTS / the API default)")
    p.add_argument("--sam3d", default=os.environ.get("FAH_SAM3D", ""), help="SAM 3D Body config directory (default: FAH_SAM3D / the API default)")
    p.add_argument("--trt_dir", default="", help="TensorRT engines (fah-build-trt --out); PyTorch when empty")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", ""))
    p.add_argument("--contact", type=int, default=1, help="1: the contact panel (the model's contact head)")
    p.add_argument("--mask_split", type=int, default=1, help="split shared detector-box masks (the release default)")
    p.add_argument("--focal_ratio", type=float, default=1.0, help="focal = ratio x the longer image side")
    p.add_argument("--max_side", type=int, default=1600)
    p.add_argument("--margin", type=float, default=0.04)
    p.add_argument("--aspect", type=float, default=4 / 3, help="tile width / height")
    p.add_argument("--tile", type=int, default=300)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--grid", default="", help="also write the assembled figure here (rows of --cols cells)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not a.weights:
        from fah.api import DEFAULT_WEIGHTS

        a.weights = DEFAULT_WEIGHTS
    if not a.sam3d:
        from fah.api import DEFAULT_SAM3D

        a.sam3d = a.sam3d or DEFAULT_SAM3D
    main(a)
