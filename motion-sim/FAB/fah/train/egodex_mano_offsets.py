"""EgoDex (ARKit visionOS skeleton) vs MANO joint convention, measured against MANO-convention models.

predict: run a model (bench.methods.fah args) on QC-passed EgoDex crops with the crop's own box and camera
  (640 crop = similarity of the mirrored frame: K_crop = A K) and store per crop the ARKit label, the model's
  21 joints and its MANO-mesh joints (J_regressor + HOT3D tips on its 778 MANO vertices, the strict MANO skeleton).
fit: per-joint offset of the labels in the label's own hand frame (origin wrist, x = pinky MCP -> index MCP,
  y = wrist -> middle MCP orthogonalised, z = x X y; unit = |wrist -> middle MCP|), after a similarity alignment of
  the reference onto the label on the anchor joints (finger PIPs + DIPs by default); correction label' = label +
  size * R_label @ delta_j (per-joint median, optionally + ridge on the label's local coordinates), fit on 80 % /
  validated on 20 % of the episodes (crc32 split). --bias_runs: the reference's own per-joint bias vs MANO GT,
  measured the same way on EgoStandard bench pickles (GT = MANO fits), is removed from delta.

    python -m train.egodex_mano_offsets predict --root $DATA_ROOT/crops --pools egodex_test,egodex_part1 --num 20000 \
        --method_args '<fah pipeline args>' --tag ref --out $CKPT_ROOT/egodex_mano
    python -m train.egodex_mano_offsets fit --out $CKPT_ROOT/egodex_mano --tags ref --ref ref:mesh
"""
import argparse
import glob
import json
import os
import zlib

import cv2
import numpy as np

WORK = os.environ.get("FAH_WORK", "output")  # QC files / outputs
QC_MAX = 0.12  # = train.finetune.LABEL_QC
ANCHOR = [6, 7, 10, 11, 14, 15, 18, 19]  # finger PIPs / DIPs (index..pinky)
NAMES = ["wrist", "T_cmc", "T_mcp", "T_ip", "T_tip", "I_mcp", "I_pip", "I_dip", "I_tip", "M_mcp", "M_pip", "M_dip", "M_tip",
         "R_mcp", "R_pip", "R_dip", "R_tip", "L_mcp", "L_pip", "L_dip", "L_tip"]


def qc_keep(pool):
    qc = lambda d: glob.glob(os.path.join(WORK, "qc", d, "qc.jsonl")) + glob.glob(os.path.join(WORK, "qc", d, "qc.[0-9]*.jsonl"))
    files = qc(pool) or qc(pool.split("_")[0])
    return {q["file"] for f in files for q in map(json.loads, open(f)) if q["err2d"] < QC_MAX}


class _Boxes:  # stands in for the detector schedule: the crop's own hand box, right hand (crops are mirrored)
    box = None

    def __call__(self, image, prev):
        return np.asarray([self.box], np.float32), np.ones(1, np.float32), np.ones(1, bool)

    def reset(self):
        pass


class _YoloBoxes:  # stands in for WiLoR's YOLO detector: the crop's own box as a right hand
    box = None

    class _B:
        def __init__(self, box):
            import torch

            self.xyxy, self.cls, self.conf = torch.tensor([box], dtype=torch.float32), torch.ones(1), torch.ones(1)

        def __len__(self):
            return 1

    def __call__(self, image, **_):
        return [type("R", (), {"boxes": self._B(self.box)})()]


def predict(args):
    import torch

    from bench.common import VertexJoints, box_iou

    recs = []
    for pool in args.pools.split(","):
        keep = qc_keep(pool) if args.qc else None  # label QC: EgoDex (3D-only labels); 2D pools: every labelled crop
        rs = [dict(r, pool=pool) for r in map(json.loads, open(os.path.join(args.root, pool, "index.jsonl")))
              if (keep is None or r["file"] in keep) and (r.get("joints3d") is not None or r.get("kp2d") is not None)]
        rs.sort(key=lambda r: zlib.crc32(r["file"].encode()))
        recs += rs[: args.num // len(args.pools.split(","))]
        print(f"{pool}: {len(rs)} candidates, using {min(len(rs), args.num // len(args.pools.split(',')))}", flush=True)
    if args.method == "wilor":  # MANO model that never saw EgoDex (reference for the ARKit -> MANO fit)
        from bench.methods.wilor import WiLoRMethod

        m = WiLoRMethod(**json.loads(args.method_args))
        m.detector = boxes = _YoloBoxes()
        m.reset = lambda: None
    else:
        from bench.methods.fah import FAHMethod

        m = FAHMethod(**dict(json.loads(args.method_args), verify=False))
        m.schedule = boxes = _Boxes()
    vj = VertexJoints(args.mano_dir)
    out = dict(label=[], kp2d=[], K=[], joints=[], joints2d=[], mesh=[], crop=[], pool=[], episode=[], size_px=[])
    for k, r in enumerate(recs):
        img = cv2.imread(os.path.join(args.root, r["pool"], r["file"]))
        H, W = r["hw"]
        M = np.asarray(r["M"], np.float64)
        if r.get("K") is not None:
            K = np.asarray(r["K"], np.float64)
        else:  # as LabeledCrops
            from train.finetune import default_K

            K = default_K(H, W, r["is_right"])
        b = np.asarray(r["box"], np.float64)
        if not r["is_right"]:
            b = np.array([W - b[2] - 1, b[1], W - b[0] - 1, b[3]])
        box = np.concatenate([M[:, :2] @ b[:2] + M[:, 2], M[:, :2] @ b[2:] + M[:, 2]])
        Kc = np.concatenate([M, [[0, 0, 1]]]) @ K
        boxes.box = box
        m.reset()
        preds = m(img, Kc)
        if not preds:
            continue
        p = max(preds, key=lambda h: box_iou(np.asarray(h["bbox"]), box))
        if p.get("verts3d") is None or not np.isfinite(p["joints3d"]).all():
            continue
        out["label"].append(np.asarray(r["joints3d"] if r.get("joints3d") is not None else np.full((21, 3), np.nan), np.float32))
        out["kp2d"].append(np.asarray(r["kp2d"] if r.get("kp2d") is not None else np.full((21, 3), np.nan), np.float32))
        out["K"].append(Kc.astype(np.float32))
        out["joints"].append(np.asarray(p["joints3d"], np.float32))
        out["joints2d"].append(np.asarray(p["joints2d"], np.float32))
        out["mesh"].append(vj(np.asarray(p["verts3d"], np.float64)).astype(np.float32))
        out["crop"].append(r["file"])
        out["pool"].append(r["pool"])
        out["episode"].append(r.get("episode", r["file"].split("/")[0]))
        out["size_px"].append(float(max(box[2] - box[0], box[3] - box[1])))
        if k % 2000 == 0:
            print(f"{k}/{len(recs)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f"preds_{args.tag}.npz"), **{k: np.asarray(v) for k, v in out.items()})
    print(f"{args.tag}: {len(out['label'])} crops -> {args.out}")


def frame(X):
    """Hand frame of (N, 21, 3) joints: R (N, 3, 3) with columns x, y, z; size (N,)."""
    x = X[:, 5] - X[:, 17]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = X[:, 9] - X[:, 0]
    y -= (y * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1), np.linalg.norm(X[:, 9] - X[:, 0], axis=-1)


def sim_align(P, G, idx):
    """Similarity-align each P[n] (21, 3) onto G[n] using joints idx; returns aligned P (N, 21, 3)."""
    out = np.empty_like(P)
    for n in range(len(P)):
        p, g = P[n, idx], G[n, idx]
        mp, mg = p.mean(0), g.mean(0)
        U, S, Vt = np.linalg.svd((p - mp).T @ (g - mg))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = (U @ D @ Vt).T
        s = (S * np.diag(D)).sum() / max(((p - mp) ** 2).sum(), 1e-12)
        out[n] = s * (P[n] - mp) @ R.T + mg
    return out


def local_offsets(lab, ref, anchor):
    """(N, 21, 3) offsets of the aligned reference w.r.t. the label, in the label's hand frame / hand size."""
    R, size = frame(lab)
    d = sim_align(ref, lab, anchor) - lab
    return np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), d) / size[:, None, None], size


def apply(lab, delta, ridge=None):
    """Corrected labels: lab + size * R @ (delta [+ ridge(local coords)])."""
    R, size = frame(lab)
    d = np.broadcast_to(delta, lab.shape).copy()
    if ridge is not None:
        d += (local_coords(lab) @ ridge["W"] + ridge["b"]).reshape(-1, 21, 3)
    return lab + np.einsum("nij,nkj->nki", R, d) * size[:, None, None]


def local_coords(X):
    R, size = frame(X)
    return (np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), X - X[:, :1]) / size[:, None, None]).reshape(len(X), -1)


def pa_err(P, G):
    """Per-joint error (N, J) mm after the all-joint similarity alignment of P onto G."""
    return np.linalg.norm(sim_align(P, G, list(range(P.shape[1]))) - G, axis=-1) * 1000


def bias_from_runs(runs, kind, anchor):
    """Reference model's own mean local offset vs MANO GT (EgoStandard bench pickles: gt = MANO fits)."""
    import pickle

    from bench.common import VertexJoints, match_hands

    vj = VertexJoints(os.environ.get("MANO_DIR", "data/mano")) if kind == "mesh" else None
    G, P = [], []
    for path in sorted(glob.glob(os.path.join(runs, "*.pkl")))[::4]:
        for fr in pickle.load(open(path, "rb"))["frames"][::5]:
            for gi, pi in match_hands(fr["pred"], fr["gt"], 0.3)[0]:
                if pi is None:
                    continue
                h, g = fr["pred"][pi], fr["gt"][gi]
                p = vj(np.asarray(h["verts3d"], np.float64)) if vj is not None else np.asarray(h["joints3d"], np.float64)
                gj = np.asarray(g["joints3d"], np.float64).copy()
                if not g["is_right"]:  # mirror into right hands like the EgoDex crops
                    p, gj = p * [-1, 1, 1], gj * [-1, 1, 1]
                G.append(gj)
                P.append(p)
    d, _ = local_offsets(np.stack(G), np.stack(P), anchor)  # label := GT, reference := model
    return np.median(d, 0), len(G)


def fit(args):
    anchor = list(range(21)) if args.anchor == "all" else [int(i) for i in args.anchor.split(",")] if args.anchor else ANCHOR
    tags = args.tags.split(",")
    data = {t: dict(np.load(os.path.join(args.out, f"preds_{t}.npz"))) for t in tags}
    for d in data.values():
        d["key"] = np.char.add(np.char.add(d["pool"].astype(str), "/"), d["crop"].astype(str))
    keys = set.intersection(*(set(d["key"]) for d in data.values()))
    for t, d in data.items():  # common crops, same order
        o = np.argsort(d["key"])
        d.update({k: v[o] for k, v in d.items()})
        m = np.isin(d["key"], list(keys))
        d.update({k: v[m] for k, v in d.items()})
    rt, rk = args.ref.split(":")
    ref = data[rt]["joints" if rk == "joints" else "mesh"].astype(np.float64)
    if args.ref_kin:  # reference mesh-J -> kinematic MANO joints (HOT3D-fitted, train.egodex_mano_offsets gtpairs / fit)
        ref = apply(ref, np.asarray(json.load(open(args.ref_kin))["delta"]))
    if args.src == "label":
        lab = data[rt]["label"].astype(np.float64)
    else:  # another skeleton convention as the "label", e.g. the reference model's direct (MHR -> MANO regressor) joints
        st, sk = args.src.split(":")
        lab = data[st]["joints" if sk == "joints" else "mesh"].astype(np.float64)
    ep = data[rt]["episode"]
    val = np.array([zlib.crc32(e.encode()) % 5 == 0 for e in ep])
    off, size = local_offsets(lab, ref, anchor)
    hs = np.median(size) * 1000
    lines = [f"# EgoDex {'ARKit labels' if args.src == 'label' else args.src} vs {args.ref} on {len(lab)} QC-passed crops (anchor joints {anchor}; "
             f"offsets = reference - label in the label's hand frame, x: pinky->index MCP, y: wrist->middle MCP, z: palm "
             f"normal; mm at the median hand size {hs:.1f} mm)", "",
             "| joint | median dx | dy | dz | |median| mm | spread (median |dev|) mm |", "|---|---|---|---|---|---|"]
    med = np.median(off[~val], 0)
    for j in range(21):
        dev = np.linalg.norm(off[:, j] - med[j], axis=-1)
        lines.append(f"| {NAMES[j]} | {med[j, 0] * hs:+.1f} | {med[j, 1] * hs:+.1f} | {med[j, 2] * hs:+.1f} | "
                     f"{np.linalg.norm(med[j]) * hs:.1f} | {np.median(dev) * hs:.1f} |")
    delta, delta_unb = med.copy(), None
    if args.bias_runs:
        b, nb = bias_from_runs(args.bias_runs, rk, anchor)
        lines += ["", f"Reference bias vs MANO GT on EgoStandard ({nb} hands, same frame, |median| mm): " +
                  ", ".join(f"{NAMES[j]} {np.linalg.norm(b[j]) * hs:.1f}" for j in range(21)), ""]
        delta_unb = med - b  # label -> MANO = label -> reference - (MANO -> reference)
        if args.use_bias:
            delta = delta_unb
    ridge = None
    if args.ridge > 0:
        Xc = local_coords(lab[~val])
        mu = Xc.mean(0)
        Y = (off[~val] - delta).reshape(len(Xc), -1)
        A = Xc - mu
        W = np.linalg.solve(A.T @ A + args.ridge * len(A) * np.eye(A.shape[1]), A.T @ (Y - Y.mean(0)))
        ridge = dict(W=W, b=Y.mean(0) - mu @ W)
    lines += ["", "Validation (held-out 20 % of episodes): label vs each model's joints, PA-MPJPE per joint group (mm)", "",
              "| reference | labels | all | wrist | CMC/MCP | PIP | DIP | tip |", "|---|---|---|---|---|---|---|---|"]
    groups = dict(wrist=[0], mcp=[1, 5, 9, 13, 17], pip=[2, 6, 10, 14, 18], dip=[3, 7, 11, 15, 19], tip=[4, 8, 12, 16, 20])
    variants = dict(ARKit=lab[val], fixed=apply(lab[val], med))
    if delta_unb is not None:
        variants["fixed, ref bias removed"] = apply(lab[val], delta_unb)
    if ridge:
        variants["fixed+ridge"] = apply(lab[val], delta, ridge)
    for t in tags:
        for kind in ("joints", "mesh"):
            R = data[t][kind][val].astype(np.float64)
            for name, L in variants.items():
                e = pa_err(L, R)
                lines.append(f"| {t}:{kind} | {name} | {e.mean():.2f} | " + " | ".join(f"{e[:, g].mean():.2f}" for g in groups.values()) + " |")
    chains = dict(T=[0, 1, 2, 3, 4], I=[0, 5, 6, 7, 8], M=[0, 9, 10, 11, 12], R=[0, 13, 14, 15, 16], L=[0, 17, 18, 19, 20])
    lines += ["", "Median bone length / |wrist -> middle MCP| (held-out crops; 0: wrist / metacarpal, 1-3: phalanges)", "",
              "| joints | " + " | ".join(chains) + " |", "|---" * (len(chains) + 1) + "|"]
    for name, L in dict(variants, **{args.ref: ref[val]}).items():
        n = np.linalg.norm(L[:, 9] - L[:, 0], axis=-1)
        cells = [" / ".join(f"{np.median(np.linalg.norm(L[:, c[i + 1]] - L[:, c[i]], axis=-1) / n):.2f}" for i in range(4)) for c in chains.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    pops = {}
    for pd in [x for x in args.pop_pools.split(",") if x]:  # GT populations (crop pools with 3D labels)
        rs = [json.loads(l) for _, l in zip(range(200000), open(os.path.join(pd, "index.jsonl")))]
        J = np.asarray([r["joints3d"] for r in rs[:: max(len(rs) // 20000, 1)] if r.get("joints3d") is not None], np.float64)
        pops[f"GT {os.path.basename(pd.rstrip('/'))}"] = J
    lines += ["", "Median bone length / middle proximal phalanx (M_mcp -> M_pip); 0: wrist -> MCP (thumb: -> CMC), 1-3: phalanges", "",
              "| joints | " + " | ".join(chains) + " |", "|---" * (len(chains) + 1) + "|"]
    for name, L in dict(variants, **{args.ref + (" (kin)" if args.ref_kin else ""): ref[val]}, **pops).items():
        n = np.linalg.norm(L[:, 10] - L[:, 9], axis=-1)
        cells = [" / ".join(f"{np.median(np.linalg.norm(L[:, c[i + 1]] - L[:, c[i]], axis=-1) / n):.2f}" for i in range(4)) for c in chains.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    corr = dict(ref=args.ref, anchor=anchor, frame="origin wrist; x = unit(I_mcp - L_mcp); y = unit(M_mcp - wrist) orthogonalised; z = x cross y",
                unit="|M_mcp - wrist| of the ARKit label", delta=delta.round(5).tolist(), delta_raw=med.round(5).tolist(),
                delta_bias_removed=None if delta_unb is None else delta_unb.round(5).tolist(), bias_runs=args.bias_runs or None,
                n_fit=int((~val).sum()), n_val=int(val.sum()),
                ridge=None if ridge is None else dict(W=ridge["W"].round(6).tolist(), b=ridge["b"].round(6).tolist(), lam=args.ridge))
    tag = ("" if args.src == "label" else args.src.replace(":", "_") + "_to_") + args.ref.replace(":", "_") + ("_kin" if args.ref_kin else "") \
        + ("_pa" if args.anchor == "all" else "")
    json.dump(corr, open(os.path.join(args.out, f"{'arkit_to_mano' if args.src == 'label' else 'corr'}_{tag}.json"), "w"))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, f"report_{tag}.md"), "w").write(txt + "\n")


def gtpairs(args):
    """GT MANO crops (pool with verts_file, e.g. hot3d_lab_mesh = HOT3D train sequences): kinematic joints (the pool's
    joints3d: smplx joints + tip vertices) vs mesh-J joints (J_regressor on the posed 778 vertices), stored like
    predict's output (label = joints = kinematic, mesh = mesh-J) for fit --src <tag>:mesh --ref <tag>:joints."""
    from bench.common import VertexJoints

    vj = VertexJoints(args.mano_dir)
    rs = [r for r in map(json.loads, open(os.path.join(args.root, args.pools, "index.jsonl"))) if r.get("verts_file")]
    rs = sorted(rs, key=lambda r: zlib.crc32(r["file"].encode()))[: args.num]
    out = dict(label=[], joints=[], mesh=[], crop=[], pool=[], episode=[])
    for r in rs:
        j = np.asarray(r["joints3d"], np.float64)
        v = np.load(os.path.join(args.root, args.pools, r["verts_file"])).astype(np.float64)
        out["label"].append(j)
        out["joints"].append(j)
        out["mesh"].append(vj(v))
        out["crop"].append(r["file"])
        out["pool"].append(args.pools)
        out["episode"].append(r.get("sequence_id", r["file"].split("/")[0]))
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f"preds_{args.tag}.npz"), **{k: np.asarray(v) for k, v in out.items()})
    print(f"{args.tag}: {len(rs)} GT hands -> {args.out}")


def frame2d(L):
    """2D hand frame of (N, 21, 2) points: origin wrist, y = unit(middle MCP - wrist), x = unit(index MCP - pinky MCP)
    orthogonalised; returns R (N, 2, 2) with columns x, y and the hand length |middle MCP - wrist|."""
    y = L[:, 9] - L[:, 0]
    size = np.linalg.norm(y, axis=-1)
    y = y / np.clip(size, 1e-6, None)[:, None]
    x = L[:, 5] - L[:, 17]
    x = x - (x * y).sum(-1, keepdims=True) * y
    x = x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6, None)
    return np.stack([x, y], -1), size


def sim2d(P, G, w):
    """Similarity-align each P[n] (21, 2) onto G[n] with joint weights w[n] (visibility)."""
    out = np.empty_like(P)
    for n in range(len(P)):
        m = w[n] > 0
        p, g = P[n, m], G[n, m]
        mp, mg = p.mean(0), g.mean(0)
        U, S, Vt = np.linalg.svd((p - mp).T @ (g - mg))
        D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
        R = (U @ D @ Vt).T
        sc = (S * np.diag(D)).sum() / max(((p - mp) ** 2).sum(), 1e-12)
        out[n] = sc * (P[n] - mp) @ R.T + mg
    return out


def fit2d(args):
    """2D label convention per source: human / projected labels vs the reference model's kinematic-convention joints (mesh-J ->
    kinematic correction --corr) projected with the crop camera, after a per-crop 2D similarity alignment on the
    visible joints; offsets (label - reference) in the label's 2D hand frame, % of its hand length. Sources whose
    labels are projected kinematic MANO GT (--controls) measure the reference's own bias, which is subtracted."""
    delta_k = np.asarray(json.load(open(args.corr))["delta"])
    res = {}
    for t in args.tags.split(","):
        d = dict(np.load(os.path.join(args.out, f"preds_{t}.npz")))
        kp, K = d["kp2d"].astype(np.float64), d["K"].astype(np.float64)
        ref3 = apply(d["mesh"].astype(np.float64), delta_k) if args.ref == "kin" else d["joints"].astype(np.float64)
        ref = ref3[..., :2] / np.clip(ref3[..., 2:], 1e-6, None) * K[:, None, [0, 1], [0, 1]] + K[:, None, [0, 1], 2]
        vis = kp[..., 2] > 0
        ok = vis[:, [0, 5, 9, 17]].all(1) & (vis.sum(1) >= 12)
        lab, ref, vis = kp[ok, :, :2], ref[ok], vis[ok]
        al = sim2d(ref, lab, vis)
        R, size = frame2d(lab)
        off = np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), lab - al) / size[:, None, None]
        off[~vis] = np.nan
        val = np.array([zlib.crc32(c.encode()) % 5 == 0 for c in d["crop"][ok]])
        res[t] = dict(off=off, val=val, lab=lab, al=al, vis=vis, R=R, size=size, n=int(ok.sum()))
    ctl = [c for c in args.controls.split(",") if c in res]
    bias = np.nanmedian(np.concatenate([res[c]["off"][~res[c]["val"]] for c in ctl]), 0) if ctl else np.zeros((21, 2))
    lines = [f"# 2D label offsets vs the reference model {'kinematic (mesh-J + ' + os.path.basename(args.corr) + ')' if args.ref == 'kin' else 'direct'} "
             f"joints, % of the 2D hand length (wrist -> middle MCP); x: towards the index side, y: along the hand", "",
             f"Reference bias (median over controls {ctl}) removed below; columns = median dx / dy, then |median| (spread).", ""]
    names = NAMES
    hdr = "| joint | " + " | ".join(f"{t} (n={res[t]['n']})" for t in res) + " |"
    lines += [hdr, "|---" * (len(res) + 1) + "|"]
    corr = {}
    for t, r in res.items():
        med = np.nanmedian(r["off"][~r["val"]], 0) - (bias if t not in ctl else 0)
        dev = np.nanmedian(np.linalg.norm(r["off"] - np.nanmedian(r["off"], 0), axis=-1), 0)
        r["med"], r["dev"] = med, dev
        corr[t] = (np.nanmedian(r["off"][~r["val"]], 0) - bias)  # label convention vs kinematic, label frame units
    for j in range(21):
        lines.append(f"| {names[j]} | " + " | ".join(f"{r['med'][j, 0] * 100:+.1f} / {r['med'][j, 1] * 100:+.1f} "
                                                    f"({r['dev'][j] * 100:.1f})" for r in res.values()) + " |")
    lines += ["", "Validation (held-out 20 %): mean 2D distance label vs aligned reference, % hand length, before -> after "
              "removing the per-source offset (raw for controls)", "", "| source | all | wrist | MCP | PIP | DIP | tip |", "|---|---|---|---|---|---|---|"]
    groups = dict(wrist=[0], mcp=[1, 5, 9, 13, 17], pip=[2, 6, 10, 14, 18], dip=[3, 7, 11, 15, 19], tip=[4, 8, 12, 16, 20])
    for t, r in res.items():
        v = r["val"]
        lab_c = r["lab"][v] - np.einsum("nij,kj->nki", r["R"][v], np.nan_to_num(corr[t])) * r["size"][v][:, None, None]
        al_c = sim2d(r["al"][v], lab_c, r["vis"][v])  # re-align after the correction
        e0 = np.where(r["vis"][v], np.linalg.norm(r["lab"][v] - r["al"][v], axis=-1) / r["size"][v][:, None], np.nan)
        e1 = np.where(r["vis"][v], np.linalg.norm(lab_c - al_c, axis=-1) / r["size"][v][:, None], np.nan)
        cell = lambda g: f"{np.nanmean(e0[:, g]) * 100:.1f} -> {np.nanmean(e1[:, g]) * 100:.1f}"
        lines.append(f"| {t} | {cell(list(range(21)))} | " + " | ".join(cell(g) for g in groups.values()) + " |")
    json.dump(dict(ref=args.ref, corr=args.corr, controls=ctl, frame="2D: origin label wrist, y = unit(M_mcp - wrist), x = unit(I_mcp - L_mcp) orth.",
                   unit="label 2D |M_mcp - wrist|", bias=bias.round(5).tolist(),
                   delta2d={t: np.nan_to_num(c).round(5).tolist() for t, c in corr.items()}), open(os.path.join(args.out, "label2d_offsets.json"), "w"))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, "report_label2d.md"), "w").write(txt + "\n")


def posthoc(args):
    """Apply alpha x the ARKit -> MANO correction to a bench run's predicted joints (left hands mirrored into the
    right-hand frame and back) and score PA vs the GT of its own matched hands: did the model learn ARKit palms?"""
    import pickle

    from bench.common import match_hands

    corr = json.load(open(args.corr))
    delta = np.asarray(corr["delta"])
    P, G = [], []
    for path in sorted(glob.glob(os.path.join(args.runs, "*.pkl")))[:: args.clip_stride]:
        for fr in pickle.load(open(path, "rb"))["frames"][:: args.frame_stride]:
            for gi, pi in match_hands(fr["pred"], fr["gt"], 0.3)[0]:
                if pi is None or fr["pred"][pi].get("joints3d") is None:
                    continue
                m = np.array([1.0, 1, 1]) if fr["gt"][gi]["is_right"] else np.array([-1.0, 1, 1])
                P.append(np.asarray(fr["pred"][pi]["joints3d"], np.float64) * m)
                G.append(np.asarray(fr["gt"][gi]["joints3d"], np.float64) * m)
    P, G = np.stack(P), np.stack(G)
    groups = dict(wrist=[0], mcp=[1, 5, 9, 13, 17], pip=[2, 6, 10, 14, 18], dip=[3, 7, 11, 15, 19], tip=[4, 8, 12, 16, 20])
    lines = [f"# {os.path.basename(os.path.normpath(args.runs))}: alpha x ({os.path.basename(args.corr)}) on {len(P)} matched hands, "
             "PA-MPJPE (mm)", "", "| alpha | PA-21 | PA-16 (no tips) | " + " | ".join(groups) + " |", "|---" * (len(groups) + 3) + "|"]
    for alpha in [float(a) for a in args.alphas.split(",")]:
        Q = apply(P, alpha * delta) if alpha else P
        e = pa_err(Q, G)
        nt = [j for j in range(21) if j not in groups["tip"]]
        e16 = pa_err(Q[:, nt], G[:, nt])
        lines.append(f"| {alpha:g} | {e.mean():.2f} | {e16.mean():.2f} | " + " | ".join(f"{e[:, g].mean():.2f}" for g in groups.values()) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["predict", "fit", "posthoc", "gtpairs", "fit2d"])
    p.add_argument("--root", default="$DATA_ROOT/crops")
    p.add_argument("--pools", default="egodex_test,egodex_part1")
    p.add_argument("--num", type=int, default=20000)
    p.add_argument("--qc", type=int, default=1, help="predict: only label-QC-passed crops (EgoDex); 0 for 2D pools")
    p.add_argument("--method_args", default="{}")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--tag", default="ref")
    p.add_argument("--tags", default="ref")
    p.add_argument("--ref", default="ref:mesh", help="<tag>:joints|mesh = the MANO-convention reference")
    p.add_argument("--src", default="label", help="skeleton to correct: label (ARKit) or <tag>:joints|mesh")
    p.add_argument("--ref_kin", default="", help="fit: mesh-J -> kinematic correction json applied to the reference")
    p.add_argument("--pop_pools", default="", help="fit: comma-separated pool dirs whose GT bone proportions are listed")
    p.add_argument("--method", default="fah", choices=["fah", "wilor"], help="predict: model")
    p.add_argument("--anchor", default="", help="joint ids (comma-separated) for the label <-> reference alignment, or all (PA)")
    p.add_argument("--ridge", type=float, default=0.0, help="> 0: also fit a ridge correction on local coordinates")
    p.add_argument("--bias_runs", default="", help="EgoStandard bench run dir of the reference model (measures its MANO bias)")
    p.add_argument("--use_bias", action="store_true", help="correction = label->reference offset minus the reference's MANO bias")
    p.add_argument("--out", default=os.path.join(WORK, "train", "egodex_mano"))
    p.add_argument("--runs", default="", help="posthoc: bench run dir (pickles)")
    p.add_argument("--corr", default="", help="posthoc: arkit_to_mano_*.json")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--controls", default="hot3d_lab_mesh,dexycb_lab", help="fit2d: pools with projected kinematic MANO GT")
    p.add_argument("--clip_stride", type=int, default=1)
    p.add_argument("--frame_stride", type=int, default=3)
    a = p.parse_args()
    dict(predict=predict, fit=fit, posthoc=posthoc, gtpairs=gtpairs, fit2d=fit2d)[a.cmd](a)
