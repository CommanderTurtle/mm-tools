"""Hand part segmentation labels for a crop pool (train.build_crops layout): per crop a uint8 PNG at crop resolution with
0 = background, 1 = palm (MANO wrist joint), 2-16 = the 15 finger segments (thumb / index / middle / ring / pinky x proximal /
middle / distal; MANO skinning joints), 255 = ignore. Two routes:

  3D pools (records with verts_file: HOT3D, DexYCB, FreiHAND, InterHand, OakInk2, HO3D, HanCo): the GT MANO mesh rendered with
     per-face part ids (z-buffered). With a SAM hand mask (<pool>/masks/<file>.png) the mask decides the silhouette: mask pixels
     the mesh does not cover (fingertips beyond MANO's tip vertex, flesh the fit misses) take the nearest rendered part,
     rendered pixels outside the mask become ignore (occluder or fit overshoot).
  2D pools (labelled 2D joints + a SAM mask: COCO-WB, CMU, HInt, VISOR, EgoDex 2D): mask pixels inside the palm polygon (wrist +
     MCPs + thumb CMC) are palm, the others take the nearest finger segment of the labelled skeleton (bones with both joints
     visible); a mask pixel far from every bone / the palm is ignore.

    python -m train.part_masks --pool $DATA_ROOT/crops/hot3d_lab_mesh --shard 0 --num_shards 4 [--vis /tmp/parts_vis.jpg]
-> <pool>/masks_parts/<file>.png + <pool>/masks_parts_index.<shard>.jsonl (file, route, parts present)
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np

# MANO skinning joints -> part ids: 0 wrist (palm) -> 1; thumb 13-15 -> 2-4; index 1-3 -> 5-7; middle 4-6 -> 8-10;
# ring 10-12 -> 11-13; pinky 7-9 -> 14-16
MANO_J_TO_PART = {0: 1, 13: 2, 14: 3, 15: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 10: 11, 11: 12, 12: 13, 7: 14, 8: 15, 9: 16}
# OpenPose 2D bones -> part ids (thumb CMC->MCP->IP->tip, fingers MCP->PIP->DIP->tip)
BONES = [((1, 2), 2), ((2, 3), 3), ((3, 4), 4), ((5, 6), 5), ((6, 7), 6), ((7, 8), 7), ((9, 10), 8), ((10, 11), 9), ((11, 12), 10),
         ((13, 14), 11), ((14, 15), 12), ((15, 16), 13), ((17, 18), 14), ((18, 19), 15), ((19, 20), 16)]
PALM = [0, 1, 5, 9, 13, 17]
COLORS = np.array([[0, 0, 0], [200, 200, 200]] + [[0, 0, 255 - 60 * k] for k in range(3)] + [[0, 200 - 50 * k, 255] for k in range(3)]
                  + [[0, 255 - 60 * k, 0] for k in range(3)] + [[255 - 60 * k, 128, 0] for k in range(3)]
                  + [[255 - 50 * k, 0, 255 - 50 * k] for k in range(3)], np.uint8)


def mano_parts(mano_dir, with_vpart=False):
    m = pickle.load(open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
    vpart = np.array([MANO_J_TO_PART[j] for j in np.asarray(m["weights"]).argmax(1)], np.uint8)
    faces = np.asarray(m["f"], np.int64)
    fpart = np.array([np.bincount(vpart[f], minlength=17).argmax() for f in faces], np.uint8)  # majority of its 3 vertices
    return (faces, fpart, vpart) if with_vpart else (faces, fpart)


def render_parts(v_crop, z, faces, fpart, size):
    """Painter's algorithm (far to near) of the mesh's faces with their part ids; v_crop (V, 2) crop px, z (V,) depth."""
    out = np.zeros((size, size), np.uint8)
    order = np.argsort(-z[faces].mean(1))
    for fi in order:
        tri = v_crop[faces[fi]].round().astype(np.int32)
        cv2.fillConvexPoly(out, tri, int(fpart[fi]))
    return out


def label_transfer(parts, mask, max_fill=None):
    """Silhouette from the mask: uncovered mask pixels take the nearest rendered part (ignore beyond max_fill px of the
    rendered hand: e.g. the forearm inside a hand + arm mask), rendered pixels outside it -> ignore."""
    out = parts.copy()
    covered = parts > 0
    if covered.any():
        dist, lab = cv2.distanceTransformWithLabels((~covered).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(covered)
        idx_to_part = np.zeros(lab.max() + 1, np.uint8)
        idx_to_part[lab[ys, xs]] = parts[ys, xs]
        fill = mask & ~covered
        out[fill] = idx_to_part[lab[fill]]
        if max_fill is not None:
            out[fill & (dist > max_fill)] = 255
    out[covered & ~mask] = 255
    out[~mask & ~covered] = 0
    return out


def skeleton_parts(kp, mask, max_dist_frac=0.25):
    """2D route: palm polygon + nearest visible bone segment for every mask pixel."""
    H, W = mask.shape
    out = np.zeros((H, W), np.uint8)
    vis = kp[:, 2] > 0
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    P = np.stack([xs, ys], 1).astype(np.float64)
    best, lab = np.full(len(P), np.inf), np.zeros(len(P), np.uint8)
    size = max(np.ptp(kp[vis, 0]), np.ptp(kp[vis, 1]), 1.0) if vis.sum() >= 2 else 1.0
    for (a, b), part in BONES:
        if not (vis[a] and vis[b]):
            continue
        A, B = kp[a, :2], kp[b, :2]
        d = B - A
        t = np.clip(((P - A) @ d) / max(d @ d, 1e-6), 0, 1)
        dist = np.linalg.norm(P - (A + t[:, None] * d), axis=1)
        better = dist < best
        best[better], lab[better] = dist[better], part
    if vis[PALM].sum() >= 4:  # palm polygon from the visible palm joints (convex hull)
        hull = cv2.convexHull(kp[PALM][vis[PALM], :2].astype(np.float32)).reshape(-1, 2)
        inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in P[:: 1]])
        lab[inside] = 1
        best[inside] = 0
    lab[best > max_dist_frac * size] = 255
    out[ys, xs] = lab
    return out


def fit_ortho(src, dst, w):
    """Scale s and offset t with dst ~ s * src + t (weights w): WiLoR's weak-perspective camera keeps the crop's axes."""
    mu_s, mu_d = (w[:, None] * src).sum(0) / w.sum(), (w[:, None] * dst).sum(0) / w.sum()
    a, b = src - mu_s, dst - mu_d
    sc = (w * (a * b).sum(1)).sum() / max((w * (a * a).sum(1)).sum(), 1e-9)
    return sc, mu_d - sc * mu_s


FINGER_J = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]  # OpenPose joints per finger
FINGER_P = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]  # their part ids


def ext_parts(j3, v3, kp, mask, faces, fpart, size, vpart, max_err=0.15):
    """2D route with an external MANO hand (WiLoR, train.wilor_kd: root-relative, the crop's camera axes): scaled +
    shifted onto the labelled 2D joints, then each part's vertices moved by the 2D similarity that takes its fitted bone
    onto the labelled bone (palm: least squares over the visible palm joints), so the regions follow the labelled
    skeleton with the mesh's shapes; rendered per part, silhouette from the mask (label transfer). A part whose bone is
    unlabelled keeps the global fit; a bone stretched beyond 2x / 0.5x is ignore. None if the global fit is poor."""
    vis = kp[:, 2] > 0
    if vis.sum() < 6:
        return None
    sc, t = fit_ortho(j3[:, :2], kp[:, :2], vis.astype(np.float64))
    if sc <= 0:
        return None
    uvj = j3[:, :2] * sc + t
    size_h = max(np.ptp(kp[vis, 0]), np.ptp(kp[vis, 1]), 1.0)
    err = np.linalg.norm(uvj - kp[:, :2], axis=1) / size_h
    if np.median(err[vis]) > max_err:
        return None
    uv = v3[:, :2] * sc + t
    z = uv[:, 0] + 1j * uv[:, 1]
    zj, zk = uvj[:, 0] + 1j * uvj[:, 1], kp[:, 0] + 1j * kp[:, 1]
    out, bad = z.copy(), []
    pv = [j for j in PALM if vis[j]]
    if len(pv) >= 3:  # palm: complex least squares q = a p + b over the visible palm joints
        A = np.stack([zj[pv], np.ones(len(pv))], 1)
        ab = np.linalg.lstsq(A, zk[pv], rcond=None)[0]
        out[vpart == 1] = ab[0] * z[vpart == 1] + ab[1]
    for (a, b), p in BONES:
        if not (vis[a] and vis[b]) or abs(zj[b] - zj[a]) < 1e-6:
            continue
        m = (zk[b] - zk[a]) / (zj[b] - zj[a])
        if not 0.5 < abs(m) < 2.0:
            bad.append(p)
        sel = vpart == p
        out[sel] = zk[a] + m * (z[sel] - zj[a])
    lab = render_parts(np.stack([out.real, out.imag], 1), v3[:, 2], faces, fpart, size)
    lab = label_transfer(lab, mask)
    if bad:
        lab[np.isin(lab, bad)] = 255
    return lab


def main(args):
    faces, fpart, vpart = mano_parts(args.mano_dir, with_vpart=True)
    recs = [json.loads(l) for l in open(os.path.join(args.pool, "index.jsonl"))]
    ext = None
    if args.ext_kd and os.path.exists(os.path.join(args.ext_kd, "ext_kd.npz")):  # external MANO hands (train.wilor_kd)
        z = np.load(os.path.join(args.ext_kd, "ext_kd.npz"))
        vf = os.path.join(args.ext_kd, "ext_kd_v3d.npy")
        ext = ({f: n for n, f in enumerate(z["files"].tolist())}, z["j3d"], np.load(vf, mmap_mode="r") if os.path.exists(vf) else z["v3d"],
               z["err2d"])
    recs = recs[args.shard::args.num_shards]
    if args.limit:
        recs = recs[:args.limit]
    os.makedirs(os.path.join(args.pool, args.out_dir), exist_ok=True)
    idx_f = open(os.path.join(args.pool, f"{args.out_dir}_index.{args.shard}.jsonl"), "w")
    vis_tiles = []
    for n, r in enumerate(recs):
        img_path = os.path.join(args.pool, r["file"])
        mpath = os.path.join(args.pool, args.mask_dir, os.path.splitext(r["file"])[0] + ".png")
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) if os.path.exists(mpath) else None
        size = int(mask.shape[0]) if mask is not None else 640
        lab, route, extra = None, None, {}
        if "verts_file" in r:
            v = np.load(os.path.join(os.path.dirname(img_path), os.path.basename(r["verts_file"]))).astype(np.float64)
            K, M = np.asarray(r["K"], np.float64), np.asarray(r["M"], np.float64)
            uv = (v @ K.T)[:, :2] / np.clip(v[:, 2:], 1e-4, None)  # stored verts / K already in the (mirrored) crop's camera, as train.finetune
            uvc = uv @ M[:, :2].T + M[:, 2]
            lab = render_parts(uvc, v[:, 2], faces, fpart, size)  # MANO left / right share the vertex / part layout
            route, extra = "mesh", {}
            if mask is not None:
                sil, m = lab > 0, mask > 127
                iou = float((sil & m).sum() / max((sil | m).sum(), 1))  # the mask vs the GT mesh silhouette
                extra["mask_iou"] = round(iou, 3)
                if iou >= args.min_mask_iou:  # a mask that agrees with the GT hand: silhouette (visible fingertip ends, occluders)
                    ys, xs = np.nonzero(sil)
                    hs = max(np.ptp(xs), np.ptp(ys), 1) if len(xs) else size
                    lab, route = label_transfer(lab, m, max_fill=args.max_fill * hs), "mesh+mask"
        elif mask is not None and r.get("kp2d") is not None and not args.mesh_only:
            extra = {}
            kp = np.asarray(r["kp2d"], np.float64)
            n_ext = ext[0].get(r["file"]) if ext is not None else None
            if n_ext is not None and np.isfinite(ext[3][n_ext]) and ext[3][n_ext] < args.max_ext_err:
                lab = ext_parts(ext[1][n_ext].astype(np.float64), np.asarray(ext[2][n_ext], np.float64), kp, mask > 127, faces, fpart, size, vpart)
                route = "ext+mask"
            if lab is None:
                lab, route = skeleton_parts(kp, mask > 127), "skeleton+mask"
        if lab is None:
            continue
        out = os.path.join(args.pool, args.out_dir, os.path.splitext(r["file"])[0] + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, lab)
        idx_f.write(json.dumps(dict(file=r["file"], route=route, parts=np.unique(lab[(lab > 0) & (lab < 255)]).tolist(), **extra)) + "\n")
        if args.vis and len(vis_tiles) < 8 and n % max(len(recs) // 8, 1) == 0:
            img = cv2.imread(img_path)
            col = COLORS[np.minimum(lab, 16)].copy()
            col[lab == 255] = (255, 255, 255)
            blend = np.where((lab > 0)[..., None], (0.45 * img + 0.55 * col).astype(np.uint8), img)
            vis_tiles.append(cv2.resize(np.concatenate([img, blend], 1), (512, 256)))
    idx_f.close()
    if args.vis and vis_tiles:
        while len(vis_tiles) % 2:
            vis_tiles.append(np.zeros_like(vis_tiles[0]))
        cv2.imwrite(args.vis, np.concatenate([np.concatenate(vis_tiles[i:i + 2], 1) for i in range(0, len(vis_tiles), 2)], 0))
    print("done", args.pool, args.shard, len(recs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--vis", default="")
    p.add_argument("--ext_kd", default="", help="2D pools: external MANO hands (train.wilor_kd <out>/<pool>) for the ext+mask route")
    p.add_argument("--max_ext_err", type=float, default=0.1, help="ext+mask route only where WiLoR's own 2D error (err2d) is below this")
    p.add_argument("--out_dir", default="masks_parts", help="label directory inside the pool")
    p.add_argument("--mask_dir", default="masks", help="hand masks inside the pool")
    p.add_argument("--min_mask_iou", type=float, default=0.0, help="3D pools: use the mask only if its IoU with the GT mesh silhouette reaches this")
    p.add_argument("--max_fill", type=float, default=1e9, help="label transfer: at most this x the hand size beyond the rendered mesh")
    p.add_argument("--mesh_only", action="store_true", help="no 2D routes (pools with GT meshes only)")
    main(p.parse_args())
