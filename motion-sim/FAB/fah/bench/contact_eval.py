"""Hand-object contact evaluation (the paper's contact tables) on held-out contact crop sets (train.contact_labels --test:
hot3d_bench = training-disjoint HOT3D, dexycb_s0test = DexYCB s0 test; HOI4D / ARCTIC / MOW test sets from their builders)
and on bench.run pickles of the full pipeline.

    predict     FAH contact head on the GT-box views of a set (the training framing, no augmentation) -> <test_root>/<set>/
                pred_<tag>.npz: files, prob (N, 778) fp16 (dense points -> 778 by the estimator's basis map), prob_nn (nearest
                kept point), hand (N,) hand-level score [, prob_k (N, 778, K) at each tolerance]. --set poolval: the pools'
                validation crops (threshold selection, + their labels).
    score       metrics of prediction files vs the labels, both label rules (signed < 10 mm = ours; |d| < 10 mm = HACO's): pooled
                vertex P / R / F1 at --thr, AP (pooled, threshold-free, tied scores grouped), hand-flag F1 (GT: any vertex in
                contact; prediction: hand score > 0.5), HACO's per-hand protocol (per-hand P / R / F1 at thr, mean over hands with GT
                contact), per-finger flags, visible / occluded vertices and the temporal categories (onset / release / stationary /
                hover / free / occluded contact, from the set's per-frame GT); clip-bootstrap CIs. HACO's stored predictions
                (pred_haco_<crop>.npz, bench/methods/haco.py) are scored the same way.
    video       per-frame (s1, s1c = the calibrated head's own map) vs video (s2 alone, s4 = the shipped blend + FAH-Video flag) contact
                on bench.run pickles of a held-out set (full pipeline, matched GT hands by box IoU), GT = train.contact_labels
                --video_gt: the metrics above + flag switches per 100 hand-frames, onset / release latency; rows for bench.contact_ci.
    geometric   contact from predicted geometry + the GT object meshes / poses + the 10 mm rule (no learned contact).
    pick_thr    the vertex threshold maximising the pooled vertex F1 on poolval (pools weighted equally); calibrate writes
                contact_bias = logit(threshold) into a head (0.5 = that threshold) -> <name>_cal.pt.
    haco        HACO (bench/methods/haco.py, its own environment) on a set's original frames with the GT boxes -> pred_haco_*.npz.
    mow_score   MOW test (HACO's 92 images, 35 mm labels; predictions: train.build_mow_contact preds): HACO's per-image protocol.

Temporal metrics follow real frame adjacency: switches, their latency and the onset / release / stationary categories are
computed inside contiguous runs of a hand's frames. The video flag is one online hysteresis state per side over every clip frame
(the side's highest-score prediction with contact, as fah.contact_video.VideoContact), kept across frames without a hand and
reset off after >= --reset_t (32) such frames; a GT hand with no matched prediction gets no flag.

    python -m bench.contact_eval predict --set hot3d_bench --model $WEIGHTS/fah/model.pt --ckpt $WEIGHTS/fah/contact.pt --tag fah
    python -m bench.contact_eval score --set hot3d_bench --preds fah --thr 0.5 --name fah
    python -m bench.contact_eval video --set hot3d_bench --dataset hot3d --pred_dir $TRACKS/hot3d/test-fah --video_npz <apply npz> \
        --video_ckpt $CKPT_ROOT/contact_video/fah/model.pt --vidx $WEIGHTS/fah/model.pt --thr 0.68 --debias 0.7538 --out video_hot3d.json
(paths: --test_root $DATA_ROOT/contact/test, --video_gt $DATA_ROOT/contact/video_gt; docs/contact.md)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
CONTACT_ROOT = os.environ.get("CONTACT_ROOT", os.path.join(DATA_ROOT, "contact"))  # the contact data root (docs/contact_data.md)
TEST = os.path.join(CONTACT_ROOT, "test")  # --test_root
VIDEO_GT = os.path.join(CONTACT_ROOT, "video_gt")  # --video_gt
POOLS = ["hot3d_lab_mesh", "dexycb_lab", "ho3d_lab", "oakink2_ego_lab", "oakink2_lab"]
RESET_T = 32  # --reset_t
MANO = os.environ.get("MANO_DIR", "data/mano")


def runs(ts):
    """Slices of the contiguous runs (consecutive frame indices) of a sorted frame list."""
    ts = np.asarray(ts)
    edges = np.r_[0, np.nonzero(np.diff(ts) != 1)[0] + 1, len(ts)]
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def predict(a):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from bench.methods.fah import FAHMethod
    from fah.models.heads.spectral import template_basis
    from train.contact_head import ContactCrops
    from train.finetune import Trainable, make_batch

    m = FAHMethod(checkpoint=a.checkpoint, device="cuda", finetune_ckpt=a.model, contact_ckpt=a.ckpt, verify=True, min_crop=1.1,
                  detector_weights=a.detector_weights)
    model, enc = m.est.model, m.est.student.float()
    assert getattr(model.keypoint_head, "contact_on", False), "no contact head in the checkpoint"
    cargs = torch.load(a.ckpt, map_location="cpu", weights_only=False).get("contact_args", {})
    dev, S, S_enc, size = torch.device("cuda"), 16 * enc.grid, enc.input_size, cargs.get("size", 384)  # the head's training view size
    tr = Trainable(model, enc).to(dev).eval()
    cd = getattr(model.keypoint_head, "contact_dists", None)  # a multi-tolerance head: labels at each of its tolerances
    dk = dict(dists=cd.tolist()) if cd is not None else {}
    if a.set == "poolval":  # the pools' held-out val sequences (threshold selection; not a test set)
        ds = ContactCrops(a.crops, a.pools.split(","), a.labels, min_pen=-1e9, size=size, aug=False, split="val", min_crop=1.1, **dk)
    else:
        ds = ContactCrops(TEST, [a.set], TEST, min_pen=-1e9, size=size, aug=False, split="all", min_crop=1.1, **dk)
    vi = model.keypoint_head.dense_vidx.cpu().numpy()
    U, Mw = template_basis(MANO, min(256, len(vi) // 2))  # = HandEstimator.sub_R
    U, mw = U.astype(np.float64), Mw.astype(np.float64)[vi]
    UW = U[vi].T * mw[None]
    R = U @ np.linalg.solve(UW @ U[vi] + 1e-3 * np.eye(U.shape[1]), UW)
    from bench.common import ManoLayer

    _, rest = ManoLayer(MANO)(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
    nn_idx = np.linalg.norm(rest[:, None] - rest[vi][None], axis=-1).argmin(1)  # nearest kept point per template vertex
    probs, probs_nn, hand, labs, probs_k, labs_k = [], [], [], [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, num_workers=8):
            labs.append(b["contact"].numpy().astype(bool))
            labs_k.append(b["contact_k"].numpy().astype(bool))
            b = {k: v.to(dev) for k, v in b.items()}
            x = b["img"].float() / 255.0
            bs = dict(b, A=b["A"] * (S / size))
            bs["img_f"] = F.interpolate(x, size=(S, S), mode="bilinear", antialias=True)
            out = tr(make_batch(bs, S), F.interpolate(x, size=(S_enc, S_enc), mode="bilinear", antialias=True))
            p = torch.sigmoid(out["contact_logits"].float()).cpu().numpy().astype(np.float64)
            probs.append(np.clip(p @ R.T, 0, 1).astype(np.float16))
            kh_ = model.keypoint_head
            if getattr(kh_, "contact_dists", None) is not None:  # every tolerance, monotone (N, 778, K)
                pk = torch.sigmoid(kh_.last_contact_k.float()).cpu().numpy().astype(np.float64)
                probs_k.append(np.maximum.accumulate(np.clip(np.einsum("vn,bnk->bvk", R, pk), 0, 1), axis=2).astype(np.float16))
            probs_nn.append(p[:, nn_idx].astype(np.float16))
            hand.append(torch.sigmoid(out["contact_hand"].float()).cpu().numpy())
    os.makedirs(os.path.join(TEST, a.set), exist_ok=True)
    extra = dict(lab=np.concatenate(labs), src=np.asarray([a.pools.split(",")[k] for k in ds.src])) if a.set == "poolval" else {}
    if probs_k:
        extra.update(prob_k=np.concatenate(probs_k), dists=model.keypoint_head.contact_dists.cpu().numpy())
        if a.set == "poolval":
            extra["lab_k"] = np.concatenate(labs_k)
    np.savez(os.path.join(TEST, a.set, f"pred_{a.tag}.npz"), files=np.asarray([r["file"] for r in ds.items]), prob=np.concatenate(probs),
             prob_nn=np.concatenate(probs_nn), hand=np.concatenate(hand), **extra)
    print(a.set, a.tag, len(ds), "crops", flush=True)


def ap_score(p, y):
    """Average precision (non-interpolated, = sklearn average_precision_score) of scores p for binary labels y: one
    precision / recall point per distinct score (ties grouped), AP = sum over them of (R_k - R_k-1) P_k; 0 without positives."""
    p, y = np.asarray(p), np.asarray(y)
    o = np.argsort(-p, kind="stable")
    ps, y = p[o], y[o].astype(np.float64)
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1] if len(ps) else np.zeros(0, int)  # last row of each tie group
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    if not len(tp) or tp[-1] == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def scores(prob, hand, lab, thr, mask=None):
    """Pooled vertex P / R / F1 + hand-flag F1 (the paper's protocol) + HACO's per-hand protocol (mean over hands with GT contact) at thr; mask (N, 778)
    restricts the vertex terms (visible / occluded vertices)."""
    pb = prob >= thr
    m = np.ones_like(lab) if mask is None else mask
    tp, fp, fn = (pb & lab & m).sum(), (pb & ~lab & m).sum(), (~pb & lab & m).sum()
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    gt_f = lab.any(1)
    out = dict(vP=P, vR=R, vF1=2 * P * R / max(P + R, 1e-9), n=int(len(lab)), pos=int((lab & m).sum()))
    for k, pf in (("frameF1", hand > 0.5), ("frameF1_anyv", pb.any(1))):
        out[k] = 2 * (pf & gt_f).sum() / max(pf.sum() + gt_f.sum(), 1)
    out["frameAcc"] = float(((hand > 0.5) == gt_f).mean()) if len(lab) else float("nan")
    h = gt_f & (lab & m).any(1)
    tph, fph, fnh = (pb & lab & m).sum(1)[h], (pb & ~lab & m).sum(1)[h], (~pb & lab & m).sum(1)[h]
    Ph, Rh = tph / np.maximum(tph + fph, 1), tph / np.maximum(tph + fnh, 1)
    out.update(hP=Ph.mean() if h.any() else float("nan"), hR=Rh.mean() if h.any() else float("nan"),
               hF1=np.mean(2 * Ph * Rh / np.maximum(Ph + Rh, 1e-9)) if h.any() else float("nan"))
    out["vFPrate"] = float((pb & ~lab & m).sum() / max((~lab & m).sum(), 1))  # hover / free frames: contact claimed
    return out


def finger_groups(mano_dir):
    """(6, 778) bool: palm, thumb, index, middle, ring, pinky (MANO skinning argmax, = train.part_masks' coarse parts)."""
    import pickle

    w = np.asarray(pickle.load(open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")["weights"]).argmax(1)
    g = {0: 0, 13: 1, 14: 1, 15: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 10: 4, 11: 4, 12: 4, 7: 5, 8: 5, 9: 5}
    gi = np.array([g[j] for j in w])
    return np.stack([gi == k for k in range(6)])


def categories(set_name, files):
    """Per test crop: (vis (778,) bool, {category: bool}) from the per-frame GT of its clip / hand (train.contact_labels --frames).
    contact c(t) = any vertex < 10 mm; onset / release = within 3 frames of an off -> on / on -> off switch with >= 5 frames on
    each side; stationary touch = on over [t-7, t+7], contact-set IoU(t-7, t+7) >= 0.5, not onset / release; hover = off with
    10 <= min distance < 30 mm; free = min distance >= 30 mm; occluded contact = on with < 50 % of its contact vertices visible."""
    fdir = os.path.join(TEST, set_name + "_frames")
    seqs = {}
    out_vis, out_cat = [], []
    for f in files:
        clip, base = f.rsplit("/", 1)  # crop dir = the clip (ARCTIC: <subject>/<sequence>, frames file <subject>_<sequence>.npz)
        if clip not in seqs:
            z = np.load(os.path.join(fdir, clip.replace("/", "_") + ".npz"))
            seqs.clear()
            seqs[clip] = {}
            for side in (True, False):
                sel = np.nonzero(z["is_right"] == side)[0]
                sel = sel[np.argsort(z["key"][sel])]
                keys = np.asarray([int(k) for k in z["key"][sel]])
                seqs[clip][side] = ({int(k): n for n, k in enumerate(keys)}, z["sdf"][sel].astype(np.float32), z["vis"][sel], keys)
        side = base.split("_")[-1].startswith("r")
        fnum = int(base.split("_")[0])
        kmap, sd_all, vs, keys = seqs[clip][side]
        t_all = kmap[fnum]  # crop names carry the frame number (HOT3D: clip index = key)
        r = next(r for r in runs(keys) if r.start <= t_all < r.stop)  # onset / release / stationary inside the frame's contiguous run
        sd, t = sd_all[r], t_all - r.start
        c = (sd < 10).any(1)
        mn = sd.min(1)
        T = len(c)
        cat = {}
        on = lambda a, b: c[max(a, 0):min(b, T)]
        sw_on = [i for i in range(1, T) if c[i] and not c[i - 1] and (~on(i - 5, i)).all() and on(i, i + 5).all()]
        sw_off = [i for i in range(1, T) if not c[i] and c[i - 1] and on(i - 5, i).all() and (~on(i, i + 5)).all()]
        cat["onset"] = any(abs(t - i) <= 3 for i in sw_on)
        cat["release"] = any(abs(t - i) <= 3 for i in sw_off)
        a0, b0 = t - 7, t + 7
        st = a0 >= 0 and b0 < T and c[a0:b0 + 1].all()
        if st:
            A, B = sd[a0] < 10, sd[b0] < 10
            st = (A & B).sum() / max((A | B).sum(), 1) >= 0.5
        cat["stationary"] = bool(st) and not (cat["onset"] or cat["release"])
        cat["hover"] = (not c[t]) and 10 <= mn[t] < 30
        cat["free"] = mn[t] >= 30
        cv = sd[t] < 10
        cat["occluded_contact"] = bool(c[t]) and vs[t_all][cv].mean() < 0.5
        cat["hidden_contact"] = bool(c[t]) and vs[t_all][cv].mean() < 0.1
        cat["contact"] = bool(c[t])
        out_vis.append(vs[t_all])
        out_cat.append(cat)
    return np.stack(out_vis), out_cat


def posthoc_cal(prob_nn, thr, vidx_ckpt):
    """What the calibrated head (contact_bias = logit(thr), bench.contact_eval calibrate) outputs, from an uncalibrated head's stored
    predictions: its dense-point probabilities (prob_nn at the kept vertices) -> logit - logit(thr) -> sigmoid -> the 400 -> 778
    basis map. Scored at 0.5 this is the shipped computation (the bias acts before the map, so the 778-vertex AP differs from
    thresholding the stored map at thr)."""
    from train.contact_video import dense_vidx

    vi = dense_vidx(vidx_ckpt)
    R = basis_map(vi, MANO)
    p = np.clip(np.asarray(prob_nn)[:, vi].astype(np.float64), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)) - np.log(thr / (1 - thr))
    return np.clip(1 / (1 + np.exp(-lg)) @ R.T, 0, 1).astype(np.float32)


def score(a):
    root = os.path.join(TEST, a.set)
    meta = [json.loads(l) for l in open(os.path.join(root, "contact.jsonl"))]
    sdf = np.load(os.path.join(root, "sdf.npy")).astype(np.float32)
    files = [m["file"] for m in meta]
    row = {f: i for i, f in enumerate(files)}
    clips = np.asarray([f.rsplit("/", 1)[0] for f in files])  # bootstrap unit = the crop's clip / sequence dir
    has_frames = os.path.isdir(os.path.join(TEST, a.set + "_frames"))
    vis, cats = categories(a.set, files) if has_frames else (None, None)
    G = finger_groups(MANO)
    rng = np.random.default_rng(0)
    uc = np.unique(clips)
    by = {c: np.nonzero(clips == c)[0] for c in uc}
    boots = [np.concatenate([by[c] for c in rng.choice(uc, len(uc))]) for _ in range(a.boot)]  # shared: paired across preds
    res = {}
    for tag in a.preds.split(","):
        z = np.load(os.path.join(root, f"pred_{tag}.npz"))
        assert [row[f] for f in z["files"].tolist()] == list(range(len(files))), "prediction rows must follow contact.jsonl"
        for key in ("prob", "prob_nn"):
            if key not in z.files or (key == "prob_nn" and not a.nn):
                continue
            prob, hand = z[key].astype(np.float32), z["hand"].astype(np.float32)
            if a.cal_thr and key == "prob":  # the calibrated head's output from an uncalibrated head's predictions (the paper's FAH rows)
                prob = posthoc_cal(z["prob_nn"], a.cal_thr, a.vidx)
            for rule, lab in (("signed", sdf < 10), ("unsigned", np.abs(sdf) < 10)):
                s = scores(prob, hand, lab, a.thr)
                s["AP"] = ap_score(prob.ravel(), lab.ravel())
                fp_, fl_ = np.stack([(prob[:, g] >= a.thr).any(1) for g in G], 1), np.stack([lab[:, g].any(1) for g in G], 1)
                s["fingerF1"] = 2 * (fp_ & fl_).sum() / max(fp_.sum() + fl_.sum(), 1)
                if a.boot:
                    bs = np.array([[scores(prob[b], hand[b], lab[b], a.thr)[k] for k in ("vF1", "frameF1", "hF1")] for b in boots])
                    s["boot"] = bs.tolist()  # for paired differences
                    s["vF1_ci"] = np.percentile(bs[:, 0], [2.5, 97.5]).tolist()
                    s["frameF1_ci"] = np.percentile(bs[:, 1], [2.5, 97.5]).tolist()
                if vis is not None:
                    s["visible"] = {k: v for k, v in scores(prob, hand, lab, a.thr, vis).items() if k in ("vP", "vR", "vF1", "pos")}
                    s["occluded"] = {k: v for k, v in scores(prob, hand, lab, a.thr, ~vis).items() if k in ("vP", "vR", "vF1", "pos")}
                    for cname in ("contact", "stationary", "hover", "free", "occluded_contact", "hidden_contact", "onset", "release"):
                        sel = np.asarray([c[cname] for c in cats])
                        if sel.any():
                            cs = scores(prob[sel], hand[sel], lab[sel], a.thr)
                            s[f"cat_{cname}"] = {k: cs[k] for k in ("n", "vF1", "hF1", "frameAcc", "vFPrate")}
                fmt = lambda v: round(float(v), 4) if np.isscalar(v) else v
                res[f"{tag}:{key}:{rule}"] = {k: ({kk: fmt(vv) for kk, vv in v.items()} if isinstance(v, dict) else (fmt(v) if k != "boot" else v))
                                              for k, v in s.items()}
                print(a.set, f"{tag}:{key}:{rule}", json.dumps({k: v for k, v in res[f"{tag}:{key}:{rule}"].items() if k != "boot"}), flush=True)
    json.dump(res, open(os.path.join(root, f"scores_{a.name or 'thr' + str(a.thr)}.json"), "w"))


def basis_map(vi, mano_dir):
    """(778, n) map of the estimator (HandEstimator.sub_R): dense-point values -> template vertices."""
    from fah.models.heads.spectral import template_basis

    U, Mw = template_basis(mano_dir, min(256, len(vi) // 2))
    U, mw = U.astype(np.float64), Mw.astype(np.float64)[vi]
    UW = U[vi].T * mw[None]
    return U @ np.linalg.solve(UW @ U[vi] + 1e-3 * np.eye(U.shape[1]), UW)


def hysteresis(p, on, off):
    out, st = np.zeros(len(p), bool), False
    for i, x in enumerate(p):
        st = x > on if not st else x >= off
        out[i] = st
    return out


def switches(c, kind, w=5):
    """Frames t of off -> on (kind 'on') or on -> off switches with >= w frames on each side."""
    T, out = len(c), []
    for i in range(1, T):
        if kind == "on" and c[i] and not c[i - 1] and (~c[max(i - w, 0):i]).all() and c[i:i + w].all() and i >= w and i + w <= T:
            out.append(i)
        if kind == "off" and not c[i] and c[i - 1] and c[max(i - w, 0):i].all() and (~c[i:i + w]).all() and i >= w and i + w <= T:
            out.append(i)
    return out


def video(a):
    """Per-frame (stage 1) vs video (stage 2) contact on bench.run pickles of a held-out set, matched GT hands (box IoU, as
    bench.evaluate), GT = train.contact_labels --video_gt. Vertex P / R / F1 at thr, AP, flag F1 (stage 1: hand score > 0.5; stage 2:
    hysteresis h_on / h_off on its flag), per GT hand track: flag switches per 100 frames, onset / release latency (first
    predicted switch within +-15 frames of a GT switch; misses counted), categories as `score` (visibility from the
    <set>_frames GT)."""
    import glob
    import pickle

    from bench.common import match_hands

    mano = MANO
    st2 = np.load(a.video_npz) if a.video_npz else None
    if a.video_ckpt:  # the stage-2 thresholds / hysteresis picked on its val clips (train.contact_video pick)
        import torch

        ck = torch.load(a.video_ckpt, map_location="cpu", weights_only=False)
        a.thr2, a.h_on, a.h_off = ck.get("thr_video", a.thr2), ck.get("h_on", a.h_on), ck.get("h_off", a.h_off)
        a.blend, a.thr_blend = ck.get("blend_alpha", 0.5), ck.get("thr_blend", a.thr)
    if a.debias:  # calibrated-head pickles: per-frame maps back to the uncalibrated scale (dense logits + bias, then the basis map)
        from train.contact_video import dense_vidx as _dv

        _R = basis_map(_dv(a.vidx), mano)
    s2 = {} if st2 is None else {(str(c), int(f), int(k)): (p, fl) for c, f, k, p, fl in zip(st2["clip"], st2["frame"], st2["pred"], st2["prob"], st2["flag"])}
    R = None
    gt_root = os.path.join(VIDEO_GT, a.dataset)
    fdir = os.path.join(TEST, a.set + "_frames") if a.set else ""
    rows = dict(s1=[], s2=[], s3=[], s4=[], s1c=[], lab=[], vis=[], s1f=[], s2f=[], s3f=[], s4f=[], s1cf=[], cat=[], clip=[], flip1=[], flip2=[])
    a.blend, a.thr_blend = getattr(a, "blend", 0.5), getattr(a, "thr_blend", a.thr)
    temporal = dict(s1=dict(flips=0, on=[], off=[]), s2=dict(flips=0, on=[], off=[]), gt=dict(flips=0), frames=0)
    rehyst = None
    if a.rehyst:  # the shipped video model's own hysteresis, re-run on the stored flag scores
        import torch

        ck_ = torch.load(a.rehyst, map_location="cpu", weights_only=False)
        rehyst = (float(ck_.get("h_on", 0.6)), float(ck_.get("h_off", 0.4)))
    for f in sorted(glob.glob(os.path.join(a.pred_dir, "*.pkl"))):
        name = os.path.basename(f)[:-4]
        clip = pickle.load(open(f, "rb"))
        g = np.load(os.path.join(gt_root, name + ".npz"))
        grow = {(int(t), int(k), int(h)): n for n, (t, k, h) in enumerate(zip(g["frame"], g["kind"], g["hand"]))}
        vz = np.load(os.path.join(fdir, name + ".npz")) if fdir and os.path.exists(os.path.join(fdir, name + ".npz")) else None
        vrow = {} if vz is None else {(str(k), bool(r)): n for n, (k, r) in enumerate(zip(vz["key"], vz["is_right"]))}
        state = {}  # stage-2 flag per side over every frame (online, as VideoContact: the side's highest-score prediction)
        for side in (True, False):
            st_, gap, out_ = False, 0, np.zeros(len(clip["frames"]), bool)
            for t, fr in enumerate(clip["frames"]):
                cand = [(p.get("score", 0), k) for k, p in enumerate(fr["pred"]) if bool(p["is_right"]) == side and "contact" in p]
                if not cand:  # no hand: the state is kept, and reset after RESET_T consecutive frames without one
                    gap += 1
                    if gap >= RESET_T:
                        st_ = False
                    continue
                gap = 0
                k = max(cand)[1]
                p = fr["pred"][k]
                q = s2.get((name, t, k))
                on_, off_ = a.h_on, a.h_off
                if a.release_keys and rehyst is not None and "contact_score_video" in p:
                    x, (on_, off_) = float(p["contact_score_video"]), rehyst
                elif a.release_keys and "contact_flag_video" in p:
                    x = float(p["contact_flag_video"])
                elif q is not None:
                    x = float(q[1])
                else:
                    x = float(p["contact_score"])
                st_ = x > on_ if not st_ else x >= off_
                out_[t] = st_
            state[side] = out_
        per_side = {True: [], False: []}
        for t, fr in enumerate(clip["frames"]):
            pairs, _ = match_hands(fr["pred"], fr["gt"])
            for gi, pi in pairs:
                r = grow.get((t, 0, gi))
                if r is None or not g["valid"][r]:
                    continue
                side = bool(fr["gt"][gi]["is_right"])
                sd = g["sdf"][r].astype(np.float32)
                p = fr["pred"][pi] if pi is not None else None
                if p is not None and "contact" in p and a.debias:
                    _pd = np.clip(p["contact_dense"].astype(np.float64), 1e-4, 1 - 1e-4)
                    p1 = np.clip(_R @ (1 / (1 + np.exp(-(np.log(_pd / (1 - _pd)) + a.debias)))), 0, 1).astype(np.float32)
                    f1 = float(p["contact_score"])
                elif p is not None and "contact" in p:
                    p1, f1 = p["contact"].astype(np.float32), float(p["contact_score"])
                else:
                    p1, f1 = np.zeros(778, np.float32), 0.0
                # s1c: the pickle's own per-frame map as stored (calibrated head = the shipped FAH-Image output), scored at --thr_cal
                p1c = p["contact"].astype(np.float32) if p is not None and "contact" in p else np.zeros(778, np.float32)
                q = s2.get((name, t, pi)) if pi is not None else None
                if a.release_keys and p is not None and "contact_video" in p:  # the shipped FAH-Video path (fah.contact_video)
                    p2, f2 = p["contact_video"].astype(np.float32), float(p["contact_flag_video"])
                elif q is not None:
                    if R is None:
                        from train.contact_video import dense_vidx

                        R = basis_map(dense_vidx(a.vidx), mano)
                    p2, f2 = np.clip(R @ q[0].astype(np.float64), 0, 1).astype(np.float32), float(q[1])
                else:
                    p2, f2 = p1, f1
                vv = vz["vis"][vrow[(fr["key"], side)]] if vz is not None and (fr["key"], side) in vrow else np.ones(778, bool)
                per_side[side].append((t, sd, p1, p2, f1, f2, vv, bool(state[side][t]) if p is not None else False, p1c))
        for side, seq in per_side.items():
            if not seq:
                continue
            ts = np.array([x[0] for x in seq])
            sd = np.stack([x[1] for x in seq])
            c = (sd < 10).any(1)
            f1s = np.array([x[4] for x in seq])
            pf1, pf2 = f1s > 0.5, np.array([x[7] for x in seq])
            cont = np.r_[True, np.diff(ts) == 1]
            RS = runs(ts)
            temporal["frames"] += len(seq)
            for k_, pf in (("s1", pf1), ("s2", pf2)):
                temporal[k_]["flips"] += int(((pf[1:] != pf[:-1]) & cont[1:]).sum())
                for r_ in RS:  # switches and their latency inside contiguous runs only
                    for kind in ("on", "off"):
                        for i in switches(c[r_], kind):
                            pred_sw = [j for j in switches(pf[r_], kind, 1) if abs(j - i) <= 15]
                            temporal[k_][kind].append(min(pred_sw, key=lambda j: abs(j - i)) - i if pred_sw else None)
            temporal["gt"]["flips"] += int(((c[1:] != c[:-1]) & cont[1:]).sum())
            fl1, fl2 = (np.r_[False, (pf[1:] != pf[:-1]) & cont[1:]] for pf in (pf1, pf2))  # per hand-frame flag switches (bootstrap)
            mn = sd.min(1)
            onset, release, stat = (np.zeros(len(seq), bool) for _ in range(3))
            for r_ in RS:  # onset / release / stationary inside contiguous runs only
                cr, sr = c[r_], sd[r_]
                on_sw, off_sw = switches(cr, "on"), switches(cr, "off")
                for j in range(len(cr)):
                    n_ = r_.start + j
                    onset[n_], release[n_] = any(abs(j - i) <= 3 for i in on_sw), any(abs(j - i) <= 3 for i in off_sw)
                    st = j - 7 >= 0 and j + 7 < len(cr) and cr[j - 7:j + 8].all()
                    if st:
                        A_, B_ = sr[j - 7] < 10, sr[j + 7] < 10
                        st = (A_ & B_).sum() / max((A_ | B_).sum(), 1) >= 0.5
                    stat[n_] = bool(st) and not (onset[n_] or release[n_])
            for n_, (t, sdv, p1, p2, f1, f2, vv, _, p1c) in enumerate(seq):
                cat = dict(contact=bool(c[n_]), onset=bool(onset[n_]), release=bool(release[n_]), hover=(not c[n_]) and 10 <= mn[n_] < 30,
                           free=mn[n_] >= 30, stationary=bool(stat[n_]))
                cat["occluded_contact"] = bool(c[n_]) and vv[sdv < 10].mean() < 0.5
                rows["s1"].append(p1.astype(np.float16)), rows["s2"].append(p2.astype(np.float16)), rows["lab"].append(sdv < 10)
                rows["s1c"].append(p1c.astype(np.float16)), rows["s1cf"].append(f1)
                rows["s3"].append((p1 * float(pf2[n_])).astype(np.float16)), rows["s3f"].append(float(pf2[n_]))  # s3: per-frame map, video flag gate
                rows["s4"].append(((1 - a.blend) * p1 + a.blend * p2).astype(np.float16)), rows["s4f"].append(float(pf2[n_]))  # s4: val-picked blend
                rows["vis"].append(vv), rows["s1f"].append(f1), rows["s2f"].append(float(pf2[n_])), rows["cat"].append(cat), rows["clip"].append(name)
                rows["flip1"].append(bool(fl1[n_])), rows["flip2"].append(bool(fl2[n_]))
    lab, vis = np.stack(rows["lab"]), np.stack(rows["vis"])
    res = {}
    temporal["s3"] = temporal["s4"] = temporal["s2"]
    temporal["s1c"] = temporal["s1"]  # same hand flag (contact_score) as s1
    thr_of = lambda k_: (a.thr2 or a.thr) if k_ == "s2" else (a.thr_blend if k_ == "s4" else (a.thr_cal if k_ == "s1c" else a.thr))
    for k_ in ("s1", "s2", "s3", "s4", "s1c"):
        prob, hand = np.stack(rows[k_]).astype(np.float32), np.asarray(rows[k_ + "f"], np.float32)
        thr = thr_of(k_)
        s_ = scores(prob, hand, lab, thr)
        s_["AP"] = ap_score(prob.ravel(), lab.ravel())
        s_["visible"] = {k: v for k, v in scores(prob, hand, lab, thr, vis).items() if k in ("vF1", "pos")}
        s_["occluded"] = {k: v for k, v in scores(prob, hand, lab, thr, ~vis).items() if k in ("vF1", "pos")}
        for cname in ("contact", "stationary", "hover", "free", "occluded_contact", "onset", "release"):
            sel = np.asarray([c[cname] for c in rows["cat"]])
            if sel.any():
                cs = scores(prob[sel], hand[sel], lab[sel], thr)
                s_[f"cat_{cname}"] = {k: round(float(cs[k]), 4) for k in ("n", "vF1", "frameAcc", "vFPrate")}
        tm = temporal[k_]
        s_["flips_per100"] = 100 * tm["flips"] / max(temporal["frames"], 1)
        for kind in ("on", "off"):
            lat = [x for x in tm[kind] if x is not None]
            s_[f"{kind}_miss"] = 1 - len(lat) / max(len(tm[kind]), 1)
            s_[f"{kind}_lat_mean"] = float(np.mean(lat)) if lat else float("nan")
            s_[f"{kind}_lat_absmed"] = float(np.median(np.abs(lat))) if lat else float("nan")
        res[k_] = {k: (round(float(v), 4) if np.isscalar(v) else v) for k, v in s_.items()}
        print(a.pred_dir, k_, json.dumps(res[k_]), flush=True)
    cnts = {}  # per hand-frame counts at each variant's threshold (paired bootstrap over clips / sequences)
    for k_ in ("s1", "s2", "s3", "s4", "s1c"):
        thr = thr_of(k_)
        pb = np.stack(rows[k_]).astype(np.float32) >= thr
        cnts[k_] = np.stack([(pb & lab).sum(1), (pb & ~lab).sum(1), (~pb & lab).sum(1), np.asarray(rows[k_ + "f"]) > 0.5], 1)
    cats = {f"cat_{c}": np.asarray([r[c] for r in rows["cat"]]) for c in ("contact", "stationary", "hover", "free", "occluded_contact", "onset", "release")}
    np.savez(a.out.replace(".json", "_rows.npz"), clip=np.asarray(rows["clip"]), gt_any=lab.any(1), flip1=np.asarray(rows["flip1"]),
             flip2=np.asarray(rows["flip2"]), lab=lab, prob_s1=np.stack(rows["s1"]), prob_s2=np.stack(rows["s2"]),
             prob_s4=np.stack(rows["s4"]), prob_s1c=np.stack(rows["s1c"]), **cats, **cnts)
    res["gt_flips_per100"] = 100 * temporal["gt"]["flips"] / max(temporal["frames"], 1)
    res["hand_frames"] = int(len(lab))
    json.dump(res, open(a.out, "w"), indent=1)


def _geo_work(arg):
    f, dataset, keys, VIDEO_GT = arg
    import pickle

    from bench.common import match_hands
    from train import contact_labels as C

    name = os.path.basename(f)[:-4]
    d = pickle.load(open(f, "rb"))
    g = np.load(os.path.join(VIDEO_GT, dataset, name + ".npz"))
    grow = {(int(t), int(k), int(h)): n for n, (t, k, h) in enumerate(zip(g["frame"], g["kind"], g["hand"]))}
    objs = C.OBJECTS[dataset](name, [fr["key"] for fr in d["frames"]])
    out = {k: [] for k in keys}
    lab = []
    for t, fr in enumerate(d["frames"]):
        ok, ob = objs[fr["key"]]
        if not ok or not ob:
            continue
        pairs, _ = match_hands(fr["pred"], fr["gt"])
        for gi, pi in pairs:
            r = grow.get((t, 0, gi))
            if pi is None or r is None:
                continue
            p, gh = fr["pred"][pi], fr["gt"][gi]
            if any(p.get(k) is None for k in keys):
                continue
            lab.append(g["sdf"][r].astype(np.float32) < 10)
            shift = np.asarray(gh["joints3d"], np.float64)[0] - np.asarray(p["joints3d"], np.float64)[0]  # wrist onto the GT wrist
            for k in keys:
                v = np.asarray(p[k], np.float64)
                out[k].append(np.stack([C.label(v, ob)[0] < 10, C.label(v + shift, ob)[0] < 10]))
    return name, np.asarray(lab, bool).reshape(-1, 778), {k: np.asarray(v, bool).reshape(-1, 2, 778) for k, v in out.items()}


def geometric(a):
    """Contact from predicted hand geometry + the GT object meshes / poses + the 10 mm rule (no learned contact): per
    prediction key (e.g. verts3d_dense = FAH dense points, verts3d = its fitted MANO mesh; WiLoR's verts3d), the predicted
    mesh as placed and with its wrist moved onto the GT wrist (articulation + shape only), matched GT hands (box IoU)."""
    import glob
    from multiprocessing import Pool

    keys = a.keys.split(",")
    files = sorted(glob.glob(os.path.join(a.pred_dir, "*.pkl")))
    with Pool(a.workers) as pool:
        parts = pool.map(_geo_work, [(f, a.dataset, keys, VIDEO_GT) for f in files])
    lab = np.concatenate([p[1] for p in parts])
    clips = np.concatenate([[p[0]] * len(p[1]) for p in parts])
    res = dict(hands=int(len(lab)))
    for k in keys:
        pr = np.concatenate([p[2][k] for p in parts])
        for j, place in enumerate(("placed", "wrist_aligned")):
            s_ = scores(pr[:, j].astype(np.float32), pr[:, j].any(1).astype(np.float32), lab, 0.5)
            res[f"{k}:{place}"] = {m: round(float(s_[m]), 4) for m in ("vP", "vR", "vF1", "frameF1", "hF1")}
            print(a.pred_dir, k, place, res[f"{k}:{place}"], flush=True)
    np.savez(a.out.replace(".json", ".npz"), lab=lab, clips=clips, **{k: np.concatenate([p[2][k] for p in parts]) for k in keys})
    json.dump(res, open(a.out, "w"), indent=1)


def _mow(root):
    ids = json.load(open(os.path.join(root, "splits", "test.json")))
    lab = np.stack([np.load(os.path.join(root, "preprocessed_data", "test", "contact_data", f"{i}.npy")) > 0.5 for i in ids])
    return root, ids, lab


def poolval_index(a):
    """<test_root>/poolval/index.jsonl: the rows of a `predict --set poolval` file (--ref) with their pool records (box, side, GT
    vertices), for haco --set poolval."""
    z = np.load(os.path.join(TEST, "poolval", f"pred_{a.ref}.npz"))
    want = list(zip(z["src"].tolist(), z["files"].tolist()))
    recs = {}
    for pool in sorted(set(z["src"].tolist())):
        for l in open(os.path.join(a.crops, pool, "index.jsonl")):
            r = json.loads(l)
            recs[(pool, r["file"])] = r
    with open(os.path.join(TEST, "poolval", "index.jsonl"), "w") as f:
        for pool, fl in want:
            r = dict(recs[(pool, fl)])
            r.update(pool=pool, file=f"{pool}/{fl}", verts_file=os.path.join(a.crops, pool, r["verts_file"]))
            f.write(json.dumps(r) + "\n")


def _frame_source(name, r, cache):
    """(image BGR, K, [(name, mesh path, T object -> camera)]) of a test / pool record's original frame (None if the raw
    frame is not available). name: the test set or, for poolval, the record's pool. Raw data under --data_root
    (train.contact_labels.configure)."""
    import cv2

    from train import contact_labels as C

    ds = next(k for k in ("hot3d", "dexycb", "ho3d", "oakink2") if name.startswith(k))
    g, k = C.group_of(ds, r)
    if g not in cache:
        cache.clear()
        if ds == "hot3d":
            from bench.datasets.hot3d import Hot3DClip

            tar = next((os.path.join(d, g + ".tar") for d in C.HOT3D_TARS if os.path.exists(os.path.join(d, g + ".tar"))), None)
            clip = Hot3DClip(tar, None, mano=object())
            cache[g] = (clip, {q: i for i, q in enumerate(clip.frame_keys)}, C.hot3d_objects(g, clip.frame_keys))
        elif ds == "dexycb":
            from bench.datasets.dexycb import SUBJECTS, DexYCBClip

            subj = {s_[:8]: s_ for s_ in SUBJECTS}[g[:8]]
            clip = DexYCBClip(os.path.join(C.D, "dexycb", "clips", subj, g + ".tar"), mano=object())
            cache[g] = (clip, {q: i for i, q in enumerate(clip.frame_keys)}, C.dexycb_objects(g, clip.frame_keys))
        elif ds == "ho3d":
            import zipfile

            z = zipfile.ZipFile(os.path.join(C.D, "ho3d_raw", "HO3D_v3.zip"))
            cache[g] = (z, "" if "train.txt" in z.namelist() else z.namelist()[0].split("/")[0] + "/", {})
        else:
            from train.build_oakink2_crops import load_anno

            anno = load_anno(os.path.join(C.D, "oakink2_raw", "anno_preview", f"{g}.pkl"))
            cache[g] = (anno, {}, {})
    if ds == "hot3d":
        from hand_tracking_toolkit import camera as htt_camera

        clip, idx, objs = cache[g]
        key = clip.frame_keys[idx[k]]
        if clip._maps is None:
            clip._setup_rectification(htt_camera.from_json(clip._json(key, "cameras")[clip.stream]))
        raw = cv2.imdecode(np.frombuffer(clip.tar.extractfile(f"{key}.image_{clip.stream}.jpg").read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.remap(raw, clip._maps[0], clip._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if clip.rotate else img
        return img, clip.K.astype(np.float64), objs[k][1]
    if ds == "dexycb":
        clip, idx, objs = cache[g]
        img = cv2.imdecode(np.frombuffer(clip.tar.extractfile(f"color_{k}.jpg").read(), np.uint8), cv2.IMREAD_COLOR)
        return img, clip.K.astype(np.float64), objs[k][1]
    if ds == "ho3d":
        from train.build_ho3d_crops import load_meta

        z, root, _ = cache[g]
        img = cv2.imdecode(np.frombuffer(z.read(f"{root}train/{g}/rgb/{k}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        return img, np.asarray(load_meta(z, root, g, k)["camMat"], np.float64), C.ho3d_objects(g, [k])[k][1]
    import tarfile

    from train.build_oakink2_crops import seq_tar

    anno, tars, _ = cache[g]
    view, fid = k
    if view not in tars:
        t = seq_tar(os.path.join(C.D, "oakink2_raw"), g, [view])
        tars[view] = None if t is None else (tarfile.open(t), None)
        if tars[view] is not None:
            tars[view] = (tars[view][0], {n.split("/", 1)[1]: n for n in tars[view][0].getnames() if n.endswith(".png")})
    if tars[view] is None:
        return None
    serial = {v: s_ for s_, v in anno["cam_def"].items()}[view]
    name_ = tars[view][1].get(f"{serial}/{fid:06d}.png")
    if name_ is None:
        return None
    img = cv2.imdecode(np.frombuffer(tars[view][0].extractfile(name_).read(), np.uint8), cv2.IMREAD_COLOR)
    return img, np.asarray(anno["cam_intr"][view][fid], np.float64), C.oakink2_objects(g, [k])[k][1]


def haco(a):
    """HACO (bench/methods/haco.py, run in HACO's own environment) on a set's original frames with the rows' GT hand boxes,
    crop --haco_crop (hand2x = HACO's demo crop, the contact table; JPEG round trip at quality 95 as the evaluation's crops)
    -> <test_root>/<set>/pred_haco_haco_<crop>.npz: files, prob (N, 778) fp16, hand = the max vertex probability. Rows whose raw
    frame is missing (OakInk2 allocentric views without their tar) get zeros and are listed in haco_missing.json (pick_thr
    --same_rows drops them for every model). --rows a:b runs a row range (chunks, merged by --rows '')."""
    from bench.methods.haco import HACOMethod

    root = os.path.join(TEST, a.set)
    if a.set == "poolval" and not os.path.exists(os.path.join(root, "index.jsonl")):
        poolval_index(a)
    recs = [json.loads(l) for l in open(os.path.join(root, "index.jsonl"))]
    r_lo, r_hi = (int(x) for x in a.rows.split(":")) if a.rows else (0, len(recs))
    m = HACOMethod(repo=a.haco_repo or None, crop=a.haco_crop)
    cache, missing, probs = {}, [], []
    for n in range(r_lo, r_hi):
        r = recs[n]
        src = _frame_source(r.get("pool", a.set), dict(r, file=r["file"].split("/", 1)[1]) if a.set == "poolval" else r, cache)
        if src is None:
            missing.append(n)
            probs.append(np.zeros(778, np.float32))
            continue
        probs.append(m.predict(src[0], [r["box"]], [bool(r["is_right"])])[0]["contact"])
        if n % 500 == 0:
            print(a.set, n, "/", len(recs), flush=True)
    p = np.stack(probs)
    sfx = "_%d_%d" % (r_lo, r_hi) if a.rows else ""
    np.savez(os.path.join(root, f"pred_haco_haco_{a.haco_crop}{sfx}.npz"), files=np.asarray([r["file"] for r in recs[r_lo:r_hi]]),
             prob=p.astype(np.float16), hand=p.max(1))
    json.dump(dict(missing=missing), open(os.path.join(root, f"haco_missing{sfx}.json"), "w"))
    print(a.set, "haco", a.haco_crop, p.shape, len(missing), "frames missing", flush=True)


def mow_score(a):
    """HACO's MOW protocol: per-image P / R / F1 at thr (undefined -> 0, every test image has contact), mean; + pooled F1, AP."""
    _, ids, lab = _mow(a.mow_root)
    out = {}
    for tag_thr in a.preds.split(","):
        tag, thr = tag_thr.split("@") if "@" in tag_thr else (tag_thr, a.thr)
        z = np.load(os.path.join(TEST, "mow", f"pred_{tag}.npz"))
        prob = z["prob"].astype(np.float32)
        pb = prob >= float(thr)
        tp, fp, fn = (pb & lab).sum(1), (pb & ~lab).sum(1), (~pb & lab).sum(1)
        P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        R = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        F1 = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-9), 0.0)
        out[tag_thr] = dict(P=round(float(P.mean()), 4), R=round(float(R.mean()), 4), F1=round(float(F1.mean()), 4),
                            pooledF1=round(float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)), 4),
                            AP=round(ap_score(prob.ravel(), lab.ravel()), 4), n=len(ids))
        print("mow", tag_thr, out[tag_thr], flush=True)
    json.dump(out, open(os.path.join(TEST, "mow", "scores.json"), "w"), indent=1)


def calibrate(a):
    """Store the contact logit bias in a head checkpoint (contact.pt or a full train checkpoint): bias_k = logit(t_k) for the
    vertex thresholds picked on pool val (--thrs, one per tolerance), so the head's F1-optimal vertex threshold becomes 0.5 ->
    <ckpt>_cal.pt (the released contact.pt is this file)."""
    import torch

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    t = np.asarray([float(x) for x in a.thrs.split(",")], np.float64)
    nk = ck["trainable"]["model.keypoint_head.contact_out.1.weight"].shape[0]
    assert len(t) == nk, (len(t), nk)
    ck["trainable"]["model.keypoint_head.contact_bias"] = torch.as_tensor(np.log(t / (1 - t)), dtype=torch.float32)
    ck["contact_calibration"] = dict(thresholds=t.tolist(), source="pool val (train.contact_head val split), pooled vertex F1")
    out = os.path.splitext(a.ckpt)[0] + "_cal.pt"
    torch.save(ck, out + ".tmp")
    os.replace(out + ".tmp", out)
    print(out, "bias", np.log(t / (1 - t)).round(4).tolist(), flush=True)


def pick_thr(a):
    """Vertex threshold maximising the pooled vertex F1 on the pools' val sequences (pred_<tag>.npz of `predict --set poolval`),
    the pools weighted equally (per-pool pooled counts averaged)."""
    out = {}
    for tag in a.preds.split(","):
        z = np.load(os.path.join(TEST, "poolval", f"pred_{tag}.npz"))
        ref = z if "lab" in z.files else np.load(os.path.join(TEST, "poolval", f"pred_{a.ref}.npz"))  # labels / pools, same row order
        prob, lab, src = z["prob"].astype(np.float32), ref["lab"], ref["src"]
        if a.chan >= 0:  # one tolerance of a multi-tolerance head (its own labels)
            prob, lab = z["prob_k"][..., a.chan].astype(np.float32), z["lab_k"][..., a.chan]
        keep = np.ones(len(prob), bool)
        import glob

        for mf in glob.glob(os.path.join(TEST, "poolval", "haco_missing*.json")) if a.same_rows else []:  # rows HACO could run on, for every model
            keep[json.load(open(mf))["missing"]] = False
        prob, lab, src = prob[keep], lab[keep], src[keep]
        best = None
        for t in np.arange(0.3, 0.96, 0.02):
            f1 = []
            for sname in np.unique(src):
                sel = src == sname
                pb, lb = prob[sel] >= t, lab[sel]
                tp, fp, fn = (pb & lb).sum(), (pb & ~lb).sum(), (~pb & lb).sum()
                f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            if best is None or np.mean(f1) > best[1]:
                best = (round(float(t), 2), float(np.mean(f1)))
        out[tag] = best
        print("poolval", tag, "best thr", best, flush=True)
    json.dump(out, open(os.path.join(TEST, "poolval", f"thr{'_same' if a.same_rows else ''}{'_c%d' % a.chan if a.chan >= 0 else ''}.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["predict", "score", "video", "pick_thr", "geometric", "haco", "mow_score", "calibrate"])
    p.add_argument("--test_root", default=TEST, help="held-out contact sets (train.contact_labels --test / the other builders)")
    p.add_argument("--video_gt", default=VIDEO_GT, help="train.contact_labels --video_gt output")
    p.add_argument("--labels", default=CONTACT_ROOT, help="predict --set poolval: the pools' label root")
    p.add_argument("--pools", default=",".join(POOLS), help="predict --set poolval: the pools whose val crops select thresholds")
    p.add_argument("--crops", default=os.path.join(DATA_ROOT, "crops"), help="predict / haco --set poolval: the crop pools")
    p.add_argument("--mow_root", default=os.path.join(DATA_ROOT, "mow"), help="mow_score: MOW as train.build_mow_contact expects it")
    p.add_argument("--thrs", default="", help="calibrate: pool-val vertex thresholds, one per tolerance")
    p.add_argument("--keys", default="verts3d_dense,verts3d", help="geometric: prediction mesh keys")
    p.add_argument("--same_rows", type=int, default=1, help="pick_thr: only the pool-val rows HACO has crops for (poolval/haco_missing*.json)")
    p.add_argument("--chan", type=int, default=-1, help="pick_thr: tolerance channel of a multi-tolerance head")
    p.add_argument("--ref", default="", help="pick_thr / haco --set poolval: the poolval prediction file whose lab / src rows the others follow")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--haco_crop", default="hand2x", help="haco: crop rule of bench.methods.haco (hand2x | box13)")
    p.add_argument("--haco_repo", default="", help="haco: the HACO checkout (default $THIRDPARTY/HACO)")
    p.add_argument("--rows", default="", help="haco: a row range a:b")
    p.add_argument("--set", default="", help="contact test set (hot3d_bench | dexycb_s0test | hoi4d_test | arctic_val | poolval)")
    p.add_argument("--pred_dir", default="", help="video / geometric: bench.run pickles of a contact model on the set's clips")
    p.add_argument("--dataset", default="", help="video / geometric: hot3d | dexycb | oakink2 | hoi4d | arctic (video_gt dir)")
    p.add_argument("--video_npz", default="", help="video: train.contact_video apply output")
    p.add_argument("--vidx", default="", help="video: a checkpoint with the image head's dense point indices (model.pt)")
    p.add_argument("--thr2", type=float, default=0.0, help="video: video-model vertex threshold (default --thr)")
    p.add_argument("--release_keys", type=int, default=0, help="video: s2 = the pickles' contact_video / contact_flag_video "
                   "(shipped path; use --thr 0.5 --thr2 0.5 --h_on 0.5 --h_off 0.5)")
    p.add_argument("--video_ckpt", default="", help="video: video model checkpoint (thresholds / hysteresis from `pick`)")
    p.add_argument("--h_on", type=float, default=0.6)
    p.add_argument("--h_off", type=float, default=0.4)
    p.add_argument("--out", default="")
    p.add_argument("--model", default="", help="predict: the network (<weights>/<model>/model.pt)")
    p.add_argument("--ckpt", default="", help="predict / calibrate: the contact head (contact.pt)")
    p.add_argument("--tag", default="")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--detector_weights", default="")
    p.add_argument("--preds", default="")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--cal_thr", type=float, default=0.0, help="score: post-hoc calibration of uncalibrated predictions at this val "
                   "threshold (scored at --thr 0.5; needs --vidx): the calibrated head's output")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--nn", action="store_true", help="also score the nearest-point 400 -> 778 mapping")
    p.add_argument("--name", default="", help="scores_<name>.json")
    p.add_argument("--thr_cal", type=float, default=0.5, help="video: threshold of s1c (the pickles' own map; 0.5 for a calibrated head)")
    p.add_argument("--debias", type=float, default=0.0, help="video: the image head's contact_bias when the pickles come from a calibrated head")
    p.add_argument("--reset_t", type=int, default=32, help="video: flag state reset after this many frames without a hand")
    p.add_argument("--rehyst", default="", help="video + --release_keys: video checkpoint whose h_on / h_off re-run on contact_score_video")
    p.add_argument("--data_root", default=DATA_ROOT, help="geometric / haco: the raw datasets (frames, object meshes / poses; train.contact_labels)")
    a = p.parse_args()
    TEST, VIDEO_GT, RESET_T = a.test_root, a.video_gt, a.reset_t
    if a.cmd in ("geometric", "haco"):
        from train import contact_labels

        contact_labels.configure(a.data_root)
    dict(predict=predict, score=score, video=video, pick_thr=pick_thr, geometric=geometric, haco=haco, mow_score=mow_score,
         calibrate=calibrate)[a.cmd](a)
