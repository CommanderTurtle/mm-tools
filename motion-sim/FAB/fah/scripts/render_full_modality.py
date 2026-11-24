"""Every output of one FAH forward pass side by side, on test images (bench2d-format sets) and videos: input | 2D joints
(fingertips marked) | dense surface points (coloured by MANO part) | the MANO mesh fitted to the dense points | part
segmentation head | hand instance mask. Images -> one sheet per set; videos (mp4 or a cached .npy frame stack) -> mp4.

    python scripts/render_full_modality.py --method_args '<json>' --images $DATA/bench2d_v2:8,$DATA/hard_pose_v1:8 \
        --videos clip.mp4 --out output/full_modality
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np

from train.part_masks import COLORS, MANO_J_TO_PART

SKEL = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_COL = [(0, 0, 255), (0, 165, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255)]  # thumb..pinky (BGR)
TIPS = (4, 8, 12, 16, 20)


def label(img, text):
    s = max(img.shape[1] / 640.0, 0.6)  # readable after the sheet is scaled to the tile width
    h = int(34 * s)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (int(8 * s), int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, (255, 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
    return img


def blend(img, col, m, a=0.55):
    out = img.copy()
    out[m] = (a * col[m] + (1 - a) * img[m]).astype(np.uint8)
    return out


class Renderer:
    def __init__(self, method, mano_dir):
        mm = pickle.load(open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
        self.vpart = np.array([MANO_J_TO_PART[j] for j in np.asarray(mm["weights"]).argmax(1)])
        self.faces = np.asarray(mm["f"], np.int64)
        self.fpart = np.array([np.bincount(self.vpart[f], minlength=17).argmax() for f in self.faces])
        kh = method.est.model.keypoint_head
        dv = getattr(kh, "dense_vidx", None)
        self.dense_part = self.vpart[np.asarray(dv.cpu())] if dv is not None else None
        self.m = method

    def panels(self, img, K, zoom=None):
        """-> (2 x 3 sheet, hands); zoom: callable(hands, (H, W)) -> crop box x0, y0, x1, y1 applied to every panel."""
        from bench.seg.rle import decode

        hands = self.m(img, K)
        H, W = img.shape[:2]
        p_j, p_d, p_m, p_p, p_k = (img.copy() for _ in range(5))
        sc = max(H, W) / 640.0
        r, t = max(1, int(round(2 * sc))), max(1, int(round(2 * sc)))
        mesh_rgb, mesh_on = np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.uint8)
        part_lab = np.zeros((H, W), np.uint8)
        mask_any, mask_col = np.zeros((H, W), bool), np.zeros((H, W, 3), np.uint8)
        for h in hands:
            j2 = np.asarray(h["joints2d"], np.float64)
            for a, b in SKEL:
                cv2.line(p_j, tuple(j2[a].astype(int)), tuple(j2[b].astype(int)), FINGER_COL[(b - 1) // 4], t, cv2.LINE_AA)
            for k in range(21):
                cv2.circle(p_j, tuple(j2[k].astype(int)), r + (r if k in TIPS else 0), (255, 255, 255) if k in TIPS else (40, 40, 40), -1)
            s2 = h.get("surface2d_head")
            if s2 is not None and self.dense_part is not None:
                # surface2d_head starts with 40 bone points; vertex part IDs
                # correspond only to the trailing dense surface points.
                for q, pid in zip(np.asarray(s2)[-len(self.dense_part):], self.dense_part):
                    cv2.circle(p_d, tuple(np.asarray(q).astype(int)), r, tuple(int(c) for c in COLORS[pid]), -1)
            v = h.get("verts3d")
            if v is not None and len(v) == 778:  # the fitted MANO mesh, z-buffered per face (painter's order)
                v = np.asarray(v, np.float64)
                uv = (v @ K.T.astype(np.float64))[:, :2] / np.clip(v[:, 2:], 1e-4, None)
                z = v[self.faces, 2].mean(1)
                fn = np.cross(v[self.faces[:, 1]] - v[self.faces[:, 0]], v[self.faces[:, 2]] - v[self.faces[:, 0]])
                fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
                cen = v[self.faces].mean(1)
                lam = np.abs((fn * cen).sum(1)) / (np.linalg.norm(cen, axis=1) + 1e-12)  # facing the camera -> bright
                for fi in np.argsort(-z):
                    tri = uv[self.faces[fi]].round().astype(np.int32)
                    col = COLORS[int(self.fpart[fi])].astype(np.float64) * (0.35 + 0.65 * lam[fi])
                    cv2.fillConvexPoly(mesh_rgb, tri, tuple(int(c) for c in col))
                    cv2.fillConvexPoly(mesh_on, tri, 1)
            pr = h.get("parts_roi")
            if pr is not None:
                x0, y0, roi = pr
                sub = part_lab[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]]
                sub[roi > 0] = roi[roi > 0]
            md = h.get("mask_det", h.get("mask_rle"))
            if md is not None:
                mk = decode(md).astype(bool)
                mask_any |= mk
                mask_col[mk] = (60, 200, 60) if h["is_right"] else (200, 120, 40)
        p_m = blend(p_m, mesh_rgb, mesh_on > 0, 0.85)
        p_p = blend(p_p, COLORS[np.minimum(part_lab, 16)], part_lab > 0, 0.65)
        p_k = blend(p_k, mask_col, mask_any, 0.55)
        un = (part_lab > 0).astype(np.uint8)  # the union of the predicted parts, outlined on the part and mask panels
        cs, _ = cv2.findContours(un, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for q in (p_p, p_k):
            cv2.drawContours(q, cs, -1, (255, 255, 255), max(1, int(round(1.5 * sc))), cv2.LINE_AA)
        if zoom is not None:
            x0, y0, x1, y1 = zoom(hands, (H, W))
            img, p_j, p_d, p_m, p_p, p_k = (q[y0:y1, x0:x1] for q in (img, p_j, p_d, p_m, p_p, p_k))
        part_title = "parts inside detector mask" if any("parts_roi_head" in h for h in hands) else "part segmentation head"
        tiles = [label(img.copy(), "input"), label(p_j, "2D joints (fingertips white)"), label(p_d, "dense surface points (by part)"),
                 label(p_m, "MANO mesh fitted to the dense points"), label(p_p, part_title + " (white: union of parts)"),
                 label(p_k, "detector masks (green R / blue L), white: part union")]
        return np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0), len(hands)


def hands_box(hands, hw, margin=0.35, min_frac=0.45):
    """Square box around every detected hand (+ margin), at least min_frac of the frame's short side, inside the frame."""
    H, W = hw
    if not hands:
        return 0, 0, W, H
    b = np.array([h["bbox"] for h in hands], np.float64)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x1 - x0, y1 - y0) * (1 + 2 * margin), min(H, W))
    side = max(side, min_frac * min(H, W))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = np.clip(cx - side / 2, 0, W - side), np.clip(cy - side / 2, 0, H - side)
    return int(x0), int(y0), int(x0 + side), int(y0 + side)


class SmoothBox:  # video zoom: the hands' box smoothed over time (exponential), so the view does not jitter
    def __init__(self, a=0.08):
        self.a, self.b = a, None

    def __call__(self, hands, hw):
        nb = np.array(hands_box(hands, hw), np.float64)
        self.b = nb if self.b is None else (1 - self.a) * self.b + self.a * nb
        x0, y0, x1, y1 = self.b
        side = max(x1 - x0, y1 - y0)
        return int(x0), int(y0), int(x0 + side), int(y0 + side)


def fit_width(img, w):
    s = w / img.shape[1]
    return cv2.resize(img, (w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def main(args):
    from bench.run import load_method

    m = load_method("fah_pipeline", json.loads(os.path.expandvars(args.method_args)))
    R = Renderer(m, args.mano_dir)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    for spec in [s for s in args.images.split(",") if s]:
        root, n = spec.rsplit(":", 1)
        ann = json.load(open(os.path.join(root, "annotations.json")))["images"]
        by_sub = {}
        for im in ann:
            by_sub.setdefault(im.get("subset", ""), []).append(im)
        rows = []
        for sub, ims in sorted(by_sub.items()):
            for im in [ims[i] for i in rng.choice(len(ims), min(int(n), len(ims)), replace=False)]:
                img = cv2.imread(os.path.join(root, im["file"]))
                if img is None:
                    continue
                h, w = img.shape[:2]
                K = np.asarray(im["K"], np.float32) if im.get("K") else np.array([[max(h, w), 0, w / 2], [0, max(h, w), h / 2], [0, 0, 1]], np.float32)
                sheet, _ = R.panels(img, K, zoom=hands_box if args.zoom else None)
                rows.append(label(fit_width(sheet, 3 * args.tile), f"{os.path.basename(root)} / {sub} / {os.path.basename(im['file'])}"))
        name = os.path.basename(root.rstrip("/"))
        for k in range(0, len(rows), args.per_sheet):
            cv2.imwrite(os.path.join(args.out, f"{name}_{k // args.per_sheet:02d}.jpg"), np.concatenate(rows[k:k + args.per_sheet], 0),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("images", name, len(rows), flush=True)
    for v in [s for s in args.videos.split(",") if s]:
        if v.endswith(".npy"):  # cached frames (scripts/benchmark_latency.py cache) + <stem>_K.npy
            frames = np.load(v, mmap_mode="r")
            K = np.load(v[:-4] + "_K.npy").astype(np.float32)
            src = (np.ascontiguousarray(frames[i]) for i in range(min(len(frames), args.max_frames)))
            fps = 30.0
        else:
            cap = cv2.VideoCapture(v)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ok, f0 = cap.read()
            h, w = f0.shape[:2]
            K = np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1]], np.float32)

            def gen(f0=f0, cap=cap):
                yield f0
                for _ in range(args.max_frames - 1):
                    ok, f = cap.read()
                    if not ok:
                        return
                    yield f
            src = gen()
        if hasattr(m, "reset"):
            m.reset()
        out, wr = os.path.join(args.out, os.path.splitext(os.path.basename(v))[0] + "_full_modality.mp4"), None
        zb = SmoothBox() if args.zoom and v.endswith(".npy") else None  # cached benchmark clips: small hands in a wide view
        for i, f in enumerate(src):
            sheet, _ = R.panels(f, K, zoom=zb)
            sheet = fit_width(sheet, 3 * args.tile)
            sheet = sheet[: sheet.shape[0] // 2 * 2, : sheet.shape[1] // 2 * 2]
            if wr is not None and sheet.shape[:2] != size:  # zoomed frames: one output size
                sheet = cv2.resize(sheet, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            if wr is None:
                size = sheet.shape[:2]
                tmp = os.path.join(args.tmp, os.path.basename(out) + ".mp4v.mp4")
                wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sheet.shape[1], sheet.shape[0]))
            wr.write(sheet)
        wr.release()
        os.system(f"ffmpeg -loglevel error -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 23 {out}")  # browser-playable H.264
        print("video", out, i + 1, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method_args", required=True)
    p.add_argument("--images", default="", help="bench2d-format roots with images per subset: root:n,...")
    p.add_argument("--videos", default="", help="mp4 files or cached .npy frame stacks")
    p.add_argument("--focal", type=float, default=930.0)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--tile", type=int, default=480)
    p.add_argument("--per_sheet", type=int, default=6)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--out", default="output/full_modality")
    p.add_argument("--tmp", default="/tmp", help="scratch for the raw mp4v stream")
    p.add_argument("--zoom", action="store_true", help="crop every panel to the hands (images; cached clips smoothed)")
    main(p.parse_args())
