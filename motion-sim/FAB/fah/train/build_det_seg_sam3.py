"""Build YOLO hand-instance labels directly from SAM 3 masks.

The interactive head uses labelled boxes/keypoints; the text prompt supplies optional candidates for
unlabelled hands. Each frame is encoded once. Minimal labelled-data workflow (SAM 3 environment):

    python -m train.build_det_seg_sam3 sam3 --src $DATA_ROOT/det_boxes --sam_dir $DATA_ROOT/det_sam3
    python -m train.build_det_seg_sam3 write --src $DATA_ROOT/det_boxes --sam_dir $DATA_ROOT/det_sam3 \
        --dst $DATA_ROOT/det_seg --recover 0 --split_pairs 0.5

Rejected masks fall back to a keypoint-expanded box ellipse. Optional --sam2_dir / --old inputs
supply alternative fallback labels. To recover unlabelled hands, run the `side` command with a FAH
pose model before `write`; --recover_overlap also considers overlapping candidates. Inspect the
written labels and qc_summary.json before training the detector. See docs/training.md.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from train.build_det_seg import HAND_PARENTS, bad_mask, ellipse, join_keypoints, mask_to_polygon, prompts, qc, read_labels

GRAY = 114  # train.build_det_hard ignore fill


def load_recs(d, pattern="sam.*.jsonl"):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, pattern))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # torn last line after a preemption
                continue
            recs[r["img"]] = r
    return recs


def sam3_stage(args):
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from torch.utils.data import DataLoader, Dataset

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    items = sorted(f"{s}/{f}" for s in ("train", "val") for f in os.listdir(os.path.join(args.src, "images", s)))
    items = items[args.shard :: args.num_shards]
    if args.limit:
        items = items[:args.limit]
    os.makedirs(args.sam_dir, exist_ok=True)
    out_path = os.path.join(args.sam_dir, f"sam.{args.shard}.jsonl" if args.num_sub == 1 else f"sam.{args.shard}.{args.sub}.jsonl")
    done = set(load_recs(args.sam_dir))  # all shards: a rerun may shard differently
    items = [i for i in items if i not in done][args.sub :: args.num_sub]  # --num_sub: split a shard's remaining images
    print(f"shard {args.shard}: {len(items)} images to label ({len(done)} done)", flush=True)

    class Images(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            split, f = items[i].split("/")
            img = cv2.imread(os.path.join(args.src, "images", split, f))
            H, W = img.shape[:2]
            hands = read_labels(os.path.join(args.src, "labels", split, os.path.splitext(f)[0] + ".txt"), W, H)
            return items[i], img[:, :, ::-1].copy(), hands

    model = build_sam3_image_model(checkpoint_path=args.ckpt or None, load_from_HF=not args.ckpt,
                                   enable_inst_interactivity=True)
    proc = Sam3Processor(model, confidence_threshold=args.text_min_score)
    loader = DataLoader(Images(), batch_size=1, num_workers=args.workers, collate_fn=list)
    fout = open(out_path, "a")
    for bi, ((name, img, hands),) in enumerate(loader):
        H, W = img.shape[:2]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(torch.from_numpy(img).permute(2, 0, 1))
            rec = []
            if hands:
                box, xy, lab = prompts(hands)
                ms, sc, _ = model.predict_inst(state, point_coords=xy, point_labels=lab, box=box, multimask_output=False)
                ms = np.asarray(ms).reshape(len(hands), -1, H, W)[:, 0] > 0
                sc = np.asarray(sc).reshape(len(hands), -1)[:, 0]
                for (_, b, kp), m, s in zip(hands, ms, sc):
                    x1, y1, x2, y2 = b
                    e = 0.03 * max(x2 - x1, y2 - y1) + 1
                    c = np.zeros_like(m)
                    c[max(int(y1 - e), 0):int(np.ceil(y2 + e)), max(int(x1 - e), 0):int(np.ceil(x2 + e))] = True
                    out_frac = float((m & ~c).sum() / max(m.sum(), 1))
                    m &= c
                    poly = mask_to_polygon(m)
                    rec.append(dict(poly=None if poly is None else poly.tolist(), score=float(s), out_frac=out_frac,
                                    **qc(m, b, kp)))
            st = proc.set_text_prompt(prompt=args.text, state=state)
            tm = st["masks"][:, 0].cpu().numpy()
            tb = st["boxes"].float().cpu().numpy()
            ts = st["scores"].float().cpu().numpy()
        text = []
        for m, b, s in zip(tm, tb, ts):
            poly = mask_to_polygon(m)
            if poly is not None:
                text.append(dict(poly=poly.tolist(), box=[round(float(v), 1) for v in b], score=float(s), area=int(m.sum())))
        fout.write(json.dumps(dict(img=name, hw=[H, W], hands=rec, text=text)) + "\n")
        if bi % 200 == 0:
            fout.flush()
            print(f"{bi + 1}/{len(items)}", flush=True)
    fout.close()


def poly_mask(poly, hw):
    m = np.zeros(hw, np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    return m > 0


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def box_iou(a, b):
    w, h = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h, 1e-9)


def side_boxes(sam_dir):
    """{image: {box: side record}} merged over every side*.jsonl (side.jsonl + later passes, e.g. side_ov.jsonl)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(sam_dir, "side*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(r["img"], {}).update({tuple(c["box"]): c for c in r["cand"]})
    return out


def candidates(r, hands, img, args):
    """Text instances that are not one of the labelled hands: score, size, mask IoU / box overlap with every
    labelled hand (GT box +10 %, its visible keypoints), gray ignore patches, duplicates (mask NMS)."""
    H, W = r["hw"]
    out, kept = [], []
    lab_boxes = []
    for _, b, kp in hands:
        s = 0.1 * max(b[2] - b[0], b[3] - b[1])
        lab_boxes.append((np.array([b[0] - s, b[1] - s, b[2] + s, b[3] + s]), kp[kp[:, 2] > 0, :2]))
    for t in sorted(r.get("text", []), key=lambda t: -t["score"]):
        x1, y1, x2, y2 = t["box"]
        why = None
        if t["score"] < args.text_score:
            why = "score"
        elif min(x2 - x1, y2 - y1) < args.min_side:
            why = "small"
        m = None
        if why is None:
            m = poly_mask(t["poly"], (H, W))
            if not m.any():
                why = "empty"
        over = False
        if why is None:
            ys, xs = np.nonzero(m)
            hit, kin_any = [], False
            for (lb, kv), (_, b, _) in zip(lab_boxes, hands):
                inside = ((xs >= lb[0]) & (xs <= lb[2]) & (ys >= lb[1]) & (ys <= lb[3])).mean()
                kin_any |= bool(len(kv) and m[np.clip(kv[:, 1].round().astype(int), 0, H - 1), np.clip(kv[:, 0].round().astype(int), 0, W - 1)].any())
                if inside > args.max_overlap:
                    hit.append((b, len(kv)))
            if kin_any:  # a labelled hand's own joints inside: that hand
                why = "labelled"
            elif hit:
                # --recover_overlap: another hand over / under a labelled one: its box barely overlaps the labelled box and
                # none of the labelled joints is inside it (only next to labelled hands with joints, where that test means something)
                over = bool(getattr(args, "recover_overlap", 0)) and all(nv >= 3 for _, nv in hit) and \
                    max(box_iou(t["box"], b) for b, _ in hit) < args.overlap_box_iou
                why = None if over else "labelled"
        if why is None and img is not None:
            ys, xs = np.nonzero(m)
            if (img[ys, xs] == GRAY).all(1).mean() > 0.2:
                why = "ignore_patch"
        if why is None and any(iou(m, k) > 0.5 for k in kept):
            why = "duplicate"
        if why is None:
            kept.append(m)
        out.append(dict(t, why=why, over=over))
    return out


def side_stage(args):
    """FAH pose model on each candidate box (as right and as left) -> P(right), presence."""
    import torch
    from bench.methods.fah import FAHMethod

    recs = load_recs(args.sam_dir)
    out_path = os.path.join(args.sam_dir, args.side_out)
    done = set(load_recs(args.sam_dir, args.side_out))
    have = side_boxes(args.sam_dir) if args.side_out != "side.jsonl" else {}  # extra pass: only boxes without a side yet
    m = FAHMethod(**json.loads(os.path.expandvars(args.method_args)))
    fout = open(out_path, "a")
    n = 0
    for name in sorted(recs):
        if name in done or not recs[name].get("text"):
            continue
        r = recs[name]
        split, f = name.split("/")
        img = cv2.imread(os.path.join(args.src, "images", split, f))
        hands = read_labels(os.path.join(args.src, "labels", split, os.path.splitext(f)[0] + ".txt"), *r["hw"][::-1])
        cand = [c for c in candidates(r, hands, img if f.startswith("hard_") else None, args) if c["why"] is None
                and tuple(c["box"]) not in have.get(name, {})]
        if not cand and have:
            continue
        res = []
        if cand:
            boxes = np.array([c["box"] for c in cand], np.float32)
            with torch.inference_mode():
                hr = m.est.process_image(img, boxes=boxes, is_right=np.ones(len(boxes), bool))
                hl = m.est.process_image(img, boxes=boxes, is_right=np.zeros(len(boxes), bool))
            for c, a, b in zip(cand, hr, hl):
                p_right = 0.5 * (a["side_score"] + 1 - b["side_score"])
                res.append(dict(box=c["box"], p_right=float(p_right), presence=float(max(a["presence"], b["presence"])),
                                presence_r=float(a["presence"]), presence_l=float(b["presence"])))
        fout.write(json.dumps(dict(img=name, cand=res)) + "\n")
        n += 1
        if n % 200 == 0:
            fout.flush()
            print(n, flush=True)
    fout.close()


def line_poly(line, W, H):
    return np.asarray(line.split()[1:], np.float64).reshape(-1, 2) * [W, H]


def skeleton_dist(kp, hw, width):
    """Distance (px) of every pixel to a hand's visible bones / joints."""
    vis = kp[:, 2] > 0
    m = np.ones(hw, np.uint8)
    q = np.round(kp[:, :2]).astype(int)
    for c, pa in enumerate(HAND_PARENTS):
        if pa >= 0 and vis[c] and vis[pa]:
            cv2.line(m, tuple(q[pa]), tuple(q[c]), 0, width)
    for c in np.nonzero(vis)[0]:
        cv2.circle(m, tuple(q[c]), max(1, width // 2), 0, -1)
    return cv2.distanceTransform(m, cv2.DIST_L2, 3)


def split_collapsed(lines, hands, old_lines, W, H, args, st, rows, name):
    """Labelled hands whose label masks collapsed onto one region (pairwise mask IoU > --split_pairs, e.g. both crossed hands
    in each mask): the group's union is split by the nearest own skeleton (visible bones / joints), each part inside its own
    label box, then the keypoint join and the SAM 2.1 label QC; where that fails (a hand without >= 3 visible joints, QC,
    still collapsed) the group falls back to its SAM 2.1 labels. Edits lines in place."""
    n = len(hands)
    polys = [line_poly(lines[i], W, H) for i in range(n)]
    masks = [poly_mask(p, (H, W)) for p in polys]
    par = list(range(n))

    def root(i):
        while par[i] != i:
            i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if box_iou(np.r_[polys[i].min(0), polys[i].max(0)], np.r_[polys[j].min(0), polys[j].max(0)]) > 0 and \
                    iou(masks[i], masks[j]) > args.split_pairs:
                st["collapsed_pairs"] += 1
                (_, bi, ki), (_, bj, kj) = hands[i], hands[j]
                both = (ki[:, 2] > 0) & (kj[:, 2] > 0)
                side = max(bi[2] - bi[0], bi[3] - bi[1], bj[2] - bj[0], bj[3] - bj[1])
                if both.sum() >= 3 and np.linalg.norm(ki[both, :2] - kj[both, :2], axis=1).mean() < args.dup_joint_dist * side:
                    st["duplicate_pairs"] = st.get("duplicate_pairs", 0) + 1  # the same hand labelled twice: left as is
                    continue
                par[root(j)] = root(i)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    for G in [g for g in groups.values() if len(g) > 1]:
        st["groups"] += 1
        why, new = None, {}
        if any((hands[i][2][:, 2] > 0).sum() < 3 for i in G):
            why = "no_kps"
        else:
            U = np.any([masks[i] for i in G], 0)
            side = max(max(hands[i][1][2] - hands[i][1][0], hands[i][1][3] - hands[i][1][1]) for i in G)
            D = np.stack([skeleton_dist(hands[i][2], (H, W), max(2, int(round(args.kp_join * side)))) for i in G])
            own = D.argmin(0)
            for k, i in enumerate(G):
                x1, y1, x2, y2 = np.r_[polys[i].min(0), polys[i].max(0)]
                clip = np.zeros((H, W), bool)
                clip[max(int(y1), 0):int(np.ceil(y2)) + 1, max(int(x1), 0):int(np.ceil(x2)) + 1] = True
                m = U & (own == k) & clip
                q = qc(m, hands[i][1], hands[i][2])
                p = mask_to_polygon(m)
                if p is None or (q["kp_in"] is not None and q["kp_in"] < args.min_kp_in) or q["fill"] < args.min_fill or q["box_iou"] < args.min_box_iou:
                    why = "qc"
                    break
                if args.kp_join > 0:
                    p = join_keypoints(p, hands[i][2], hands[i][1], args.kp_join)
                new[i] = p
            if why is None:
                nm = {i: poly_mask(new[i], (H, W)) for i in G}
                if any(iou(nm[i], nm[j]) > args.split_pairs for a, i in enumerate(G) for j in G[a + 1:]):
                    why = "still_collapsed"
        if why is None:
            for i in G:
                lines[i] = " ".join([lines[i].split()[0]] + [f"{v:.5f}" for v in np.clip(new[i] / [W, H], 0, 1).ravel()])
            st["split"] += 1
        else:
            st["fail"][why] = st["fail"].get(why, 0) + 1
            om = {i: poly_mask(line_poly(old_lines[i], W, H), (H, W)) for i in G}
            if any(iou(om[i], om[j]) > args.split_pairs for a, i in enumerate(G) for j in G[a + 1:]):
                st["fallback_still_collapsed"] += 1
            for i in G:
                lines[i] = old_lines[i]
            st["fallback_sam2"] += 1
        rows.append(dict(img=name, hands=G, path="split" if why is None else "sam2", why=why))


def write_stage(args):
    recs = load_recs(args.sam_dir)
    old = load_recs(args.sam2_dir) if args.sam2_dir else {}  # none: frames without SAM 2.1 labels (train.build_det_cross)
    sides = side_boxes(args.sam_dir)
    stats = dict(images=0, hands=0, sam3=0, fallback_sam2=0, fallback_ellipse=0, missing_images=0, recovered=0,
                 recovered_images=0, cand_reject={}, recovered_overlap=0,
                 pairs=dict(collapsed_pairs=0, duplicate_pairs=0, groups=0, split=0, fallback_sam2=0, fallback_still_collapsed=0, fail={}))
    pair_rows = []
    rows, rec_rows = [], []
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, kind, split), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(args.src, "images", split))):
            stem = os.path.splitext(f)[0]
            r = recs.get(f"{split}/{f}")
            if r is None:
                stats["missing_images"] += 1
                continue
            link = os.path.join(args.dst, "images", split, f)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(args.src, "images", split, f)), link)
            H, W = r["hw"]
            hands = read_labels(os.path.join(args.src, "labels", split, stem + ".txt"), W, H)
            o = old.get(f"{split}/{f}")
            assert len(hands) == len(r["hands"]) and (o is None or len(o["hands"]) == len(hands)), f
            if args.old:
                old_lines = open(os.path.join(args.old, "labels", split, stem + ".txt")).read().split("\n")
            else:  # no SAM 2.1 label: the fallback is the GT-box ellipse (+ the keypoint join), as the SAM 2.1 labels' own
                old_lines = []
                for cls, box, kp in hands:
                    q = join_keypoints(ellipse(box), kp, box, args.kp_join) if args.kp_join > 0 else ellipse(box)
                    old_lines.append(" ".join([str(cls)] + [f"{v:.5f}" for v in np.clip(q / [W, H], 0, 1).ravel()]))
            h2s = o["hands"] if o is not None else [None] * len(hands)
            lines = []
            for i, ((cls, box, kp), h, h2) in enumerate(zip(hands, r["hands"], h2s)):
                bad3, bad2 = bad_mask(h, args), (bad_mask(h2, args) if h2 is not None else True)
                if not bad3:
                    src, poly = "sam3", np.asarray(h["poly"], np.float64)
                    if args.kp_join > 0:
                        poly = join_keypoints(poly, kp, box, args.kp_join)
                    line = " ".join([str(cls)] + [f"{v:.5f}" for v in np.clip(poly / [W, H], 0, 1).ravel()])
                else:  # the SAM 2.1 label (SAM 2.1 mask or the ellipse, + join), unchanged
                    src, line = ("sam2" if not bad2 else "ellipse"), old_lines[i]
                stats[{"sam3": "sam3", "sam2": "fallback_sam2", "ellipse": "fallback_ellipse"}[src]] += 1
                lines.append(line)
                new_poly = np.asarray(line.split()[1:], np.float64).reshape(-1, 2) * [W, H]
                old_poly = np.asarray(old_lines[i].split()[1:], np.float64).reshape(-1, 2) * [W, H]
                mn, mo = poly_mask(new_poly, (H, W)), poly_mask(old_poly, (H, W))
                raw = iou(poly_mask(h["poly"], (H, W)), poly_mask(h2["poly"], (H, W))) if h2 is not None and h["poly"] and h2["poly"] else None
                vis = kp[:, 2] > 0
                lo, hi = new_poly.min(0) - 1, new_poly.max(0) + 1
                kp_box = float(((kp[vis, :2] >= lo) & (kp[vis, :2] <= hi)).all(1).mean()) if vis.sum() >= 3 else None
                rows.append(dict(img=f"{split}/{f}", hand=i, src=f.split("_")[0], label=src, bad3=bool(bad3), bad2=bool(bad2),
                                 label_iou=iou(mn, mo), raw_iou=raw, area_new=int(mn.sum()), area_old=int(mo.sum()),
                                 kp_box=kp_box, **{f"{k}3": h[k] for k in ("score", "kp_in", "fill", "box_iou")},
                                 **{f"{k}2": (h2[k] if h2 is not None else None) for k in ("score", "kp_in", "fill", "box_iou")}))
            if args.split_pairs > 0 and len(hands) > 1:
                split_collapsed(lines, hands, old_lines, W, H, args, stats["pairs"], pair_rows, f"{split}/{f}")
            lab_masks = None
            if args.recover and r.get("text"):
                img = cv2.imread(os.path.join(args.src, "images", split, f)) if f.startswith("hard_") else None
                side = sides.get(f"{split}/{f}", {})
                n_rec = 0
                for c in candidates(r, hands, img, args):
                    why = c["why"]
                    s = side.get(tuple(c["box"])) if why is None else None
                    if why is None and s is None:
                        why = "no_side"
                    elif why is None and s["presence"] < args.min_presence:
                        why = "presence"
                    elif why is None and abs(s["p_right"] - 0.5) < args.side_margin:
                        why = "side_unsure"
                    if why is not None:
                        stats["cand_reject"][why] = stats["cand_reject"].get(why, 0) + 1
                        continue
                    poly = np.asarray(c["poly"], np.float64)
                    if args.split_pairs > 0 and len(hands):  # a recovered hand must not duplicate a labelled one's region
                        if lab_masks is None:
                            lab_masks = [poly_mask(line_poly(x, W, H), (H, W)) for x in lines[:len(hands)]]
                        mc = poly_mask(poly, (H, W))
                        if any(iou(mc, lm) > args.split_pairs for lm in lab_masks):
                            stats["cand_reject"]["collapsed"] = stats["cand_reject"].get("collapsed", 0) + 1
                            continue
                    stats["recovered_overlap"] += bool(c.get("over"))
                    lines.append(" ".join([str(int(s["p_right"] > 0.5))] + [f"{v:.5f}" for v in np.clip(poly / [W, H], 0, 1).ravel()]))
                    rec_rows.append(dict(img=f"{split}/{f}", hand=len(lines) - 1, src=f.split("_")[0], score=c["score"], box=c["box"],
                                         p_right=s["p_right"], presence=s["presence"], area=c["area"], over=bool(c.get("over"))))
                    n_rec += 1
                stats["recovered"] += n_rec
                stats["recovered_images"] += n_rec > 0
            stats["images"] += 1
            stats["hands"] += len(lines)
            open(os.path.join(args.dst, "labels", split, stem + ".txt"), "w").write("\n".join(lines) + "\n")
    names = "names:\n  0: left\n  1: right\n"
    open(os.path.join(args.dst, "hands.yaml"), "w").write(f"path: {args.dst}\ntrain: images/train\nval: images/val\n{names}")
    summ = dict(stats, thresholds={k: getattr(args, k) for k in ("min_fill", "min_kp_in", "min_box_iou", "kp_join", "text_score",
                                                                 "min_side", "max_overlap", "min_presence", "side_margin",
                                                                 "recover_overlap", "overlap_box_iou", "split_pairs")})
    for src in sorted({r["src"] for r in rows}) + ["all"]:
        sel = [r for r in rows if src in ("all", r["src"])]
        d = dict(n=len(sel), **{k: float(np.mean([r["label"] == k for r in sel])) for k in ("sam3", "sam2", "ellipse")},
                 bad3=float(np.mean([r["bad3"] for r in sel])), bad2=float(np.mean([r["bad2"] for r in sel])),
                 changed=float(np.mean([r["label_iou"] < args.changed_iou for r in sel])),
                 recovered=sum(x["src"] == src or src == "all" for x in rec_rows))
        for k in ("label_iou", "raw_iou", "score3", "score2", "kp_in3", "kp_in2", "box_iou3", "box_iou2", "kp_box"):
            v = np.array([r[k] for r in sel if r[k] is not None])
            if len(v):
                d[k] = dict(mean=float(v.mean()), median=float(np.median(v)), p10=float(np.percentile(v, 10)))
        d["area_ratio_median"] = float(np.median([r["area_new"] / max(r["area_old"], 1) for r in sel]))
        summ[src] = d
    json.dump(summ, open(os.path.join(args.dst, "qc_summary.json"), "w"), indent=1)
    with open(os.path.join(args.dst, "changes.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    with open(os.path.join(args.dst, "recovered.jsonl"), "w") as fo:
        for r in rec_rows:
            fo.write(json.dumps(r) + "\n")
    with open(os.path.join(args.dst, "pairs.jsonl"), "w") as fo:
        for r in pair_rows:
            fo.write(json.dumps(r) + "\n")
    print(json.dumps(summ, indent=1))
    if args.sheets:
        change_sheet(args, rows, rec_rows)


def change_sheet(args, rows, rec_rows, n=30):
    """Tiles of the n largest label changes (lowest old-vs-new label IoU) and n random recovered hands, public
    det_boxes sources only (no data-engine hard frames, no dark copies): red = SAM 2.1 label, green = SAM 3 label."""
    os.makedirs(args.sheets, exist_ok=True)
    public = ("coco", "hot3d", "hint", "visor", "cmu")
    big = sorted([r for r in rows if r["src"] in public and r["label"] == "sam3"], key=lambda r: r["label_iou"])[:n]
    rng = np.random.default_rng(0)
    rec = [r for r in rec_rows if r["src"] in public]
    rec = [rec[i] for i in rng.choice(len(rec), min(n, len(rec)), replace=False)] if rec else []

    def tile(img_name, box, polys, text):
        split, f = img_name.split("/")
        img = cv2.imread(os.path.join(args.src, "images", split, f))
        H, W = img.shape[:2]
        for poly, col in polys:
            cv2.polylines(img, [np.round(poly).astype(np.int32)], True, col, max(1, int(round(max(H, W) / 600))))
        x1, y1, x2, y2 = box
        s = int(max(x2 - x1, y2 - y1) * 1.6 + 16)
        c0, r0 = int((x1 + x2 - s) / 2), int((y1 + y2 - s) / 2)
        img = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT)
        t = cv2.resize(img[r0 + s:r0 + 2 * s, c0 + s:c0 + 2 * s], (256, 256))
        cv2.putText(t, text, (4, 16), 0, 0.45, (0, 255, 255), 1)
        return t

    def lab(img_name, i):
        split, f = img_name.split("/")
        return open(os.path.join(args.dst if i >= 0 else args.old, "labels", split, os.path.splitext(f)[0] + ".txt")).read().split("\n")

    for name, sel in (("largest_changes", big), ("recovered", rec)):
        tiles = []
        for r in sel:
            split, f = r["img"].split("/")
            H, W = cv2.imread(os.path.join(args.src, "images", split, f)).shape[:2]
            P = lambda line: np.asarray(line.split()[1:], np.float64).reshape(-1, 2) * [W, H]
            if name == "largest_changes":
                po, pn = P(lab(r["img"], -1)[r["hand"]]), P(lab(r["img"], 0)[r["hand"]])
                pts = np.concatenate([po, pn])
                box = [*pts.min(0), *pts.max(0)]
                tiles.append(tile(r["img"], box, [(po, (0, 0, 255)), (pn, (0, 255, 0))],
                                  f"{r['src']} IoU {r['label_iou']:.2f} s3 {r['score3']:.2f}"))
            else:
                news = [P(x) for x in lab(r["img"], 0) if x.strip()]
                tiles.append(tile(r["img"], r["box"], [(p, (0, 255, 0)) for p in news],
                                  f"{r['src']} s {r['score']:.2f} R {r['p_right']:.2f} pr {r['presence']:.2f}"))
        if not tiles:
            continue
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
        cv2.imwrite(os.path.join(args.sheets, f"masks_sam3_{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("sheets ->", args.sheets)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["sam3", "side", "write"])
    p.add_argument("--src", required=True, help="YOLO-pose dataset (images/, labels/)")
    p.add_argument("--sam_dir", required=True, help="SAM 3 results (sam.<shard>.jsonl, side.jsonl)")
    p.add_argument("--sam2_dir", default="", help="write: the SAM 2.1 results (train.build_det_seg sam --sam_dir)")
    p.add_argument("--old", default="", help="write: the SAM 2.1 YOLO-seg labels of --src (train.build_det_seg write; fallback, comparison)")
    p.add_argument("--dst", default="", help="write: YOLO-seg dataset")
    p.add_argument("--ckpt", default="", help="sam3.pt (default: facebook/sam3 from the HF hub)")
    p.add_argument("--text", default="hand")
    p.add_argument("--text_min_score", type=float, default=0.3, help="sam3: text instances stored above this score")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--sub", type=int, default=0)
    p.add_argument("--num_sub", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--method_args", default="", help="side: bench.methods.fah args (FAH pose model)")
    p.add_argument("--min_fill", type=float, default=0.08)
    p.add_argument("--min_kp_in", type=float, default=0.6, help="QC of the SAM 2.1 labels (train.build_det_seg)")
    p.add_argument("--min_box_iou", type=float, default=0.2)
    p.add_argument("--kp_join", type=float, default=0.04, help="keypoint join (as train.build_det_seg)")
    p.add_argument("--recover", type=int, default=1, help="write: add recovered unlabelled hands")
    p.add_argument("--text_score", type=float, default=0.7, help="recovered hands: min SAM 3 text score")
    p.add_argument("--min_side", type=float, default=12, help="recovered hands: min box side (px, stored image)")
    p.add_argument("--max_overlap", type=float, default=0.2,
                   help="reject a candidate with more than this fraction of its mask inside a labelled hand's box (+10 %%)")
    p.add_argument("--min_presence", type=float, default=0.5, help="recovered hands: min FAH presence (better side run)")
    p.add_argument("--side_margin", type=float, default=0.2, help="recovered hands: |P(right) - 0.5| needed (else side unknown, dropped)")
    p.add_argument("--changed_iou", type=float, default=0.9, help="stats: a label counts as changed below this IoU")
    p.add_argument("--recover_overlap", type=int, default=0,
                   help="also recover hands over / under a labelled hand (box IoU < --overlap_box_iou with it, none of its joints inside)")
    p.add_argument("--overlap_box_iou", type=float, default=0.3)
    p.add_argument("--split_pairs", type=float, default=0.0,
                   help="write: split labelled hands whose masks overlap above this IoU by the nearest own skeleton (0 = off)")
    p.add_argument("--dup_joint_dist", type=float, default=0.15,
                   help="split: a collapsed pair whose corresponding joints are closer than this x hand size on average is one hand "
                        "labelled twice (left unchanged)")
    p.add_argument("--side_out", default="side.jsonl", help="side: output file (another name = a pass over the boxes without a side)")
    p.add_argument("--sheets", default="", help="write: QC sheet directory")
    a = p.parse_args()
    {"sam3": sam3_stage, "side": side_stage, "write": write_stage}[a.stage](a)
