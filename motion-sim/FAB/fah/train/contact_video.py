"""FAH-Video contact training: a small causal transformer over a per-hand track of per-frame FAH predictions (bench.run
pickles of a model with the contact head: contact_dense, contact_score, joints3d, score, token) that refines the per-dense-point
contact and a per-frame contact flag online (each frame sees only the last T frames); the inference side is fah.contact_video.

Per frame: present, score, hand contact logit, the dense contact logits, side-normalised root-relative joints (dm) and root
(x/z, y/z, log z); the model adds their causal velocity / acceleration and the per-hand pose token (--use_tok). Outputs are
residuals on the per-frame logits (zero-initialised, so training starts at the image result). Tracks as fah.contact_video
(per side, the highest-score prediction); targets from the GT hand the prediction matches (box IoU >= 0.3, as bench.evaluate),
labels from train.contact_labels --video_gt (10 mm rule). Validation: --val_rule recording = the clips whose recording has
crc32 % 10 == 0 (every camera view / segment of a capture on one side: DexYCB = clip name without the camera serial, OakInk2
= without the start frame, HOT3D = the clip tar's sequence_id, HOI4D / ARCTIC = the sequence), from the stored partition
(--split_file, written by `split`; clips missing from it fall back to the same rule). `pick` sets the vertex thresholds, the
per-frame / video blend and the flag hysteresis (on > h_on, off < h_off; frames without a hand never update the state, a
hand back after >= T such frames starts off) on those val clips.

Training tracks come from 2-fold stacking (`folds`): fold head k (train.contact_head --fold_out k) is run with bench.run on the
training clips of fold k, so no track is predicted by a head that saw its sequence. Released recipe (FAH; FAH-S adds the
HOI4D / ARCTIC video tracks, docs/contact.md):

    python -m train.contact_video folds --sets hot3d:hot3d_train.json,dexycb:dexycb_train.json --out $DATA_ROOT/contact/splits
    python -m train.contact_video split --train hot3d:$TRACKS/hot3d/train-fold,dexycb:$TRACKS/dexycb/train-fold \
        --split_file $DATA_ROOT/contact/splits/video_val_by_recording.json
    python -m train.contact_video train --batch "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --train hot3d:$TRACKS/hot3d/train-fold,dexycb:$TRACKS/dexycb/train-fold \
        --vidx $WEIGHTS/fah/model.pt --use_tok --noise 0.05 --steps 20000 --out $CKPT_ROOT/contact_video/fah
    python -m train.contact_video pick --ckpt $CKPT_ROOT/contact_video/fah/model.pt
    python -m train.contact_video apply --ckpt $CKPT_ROOT/contact_video/fah/model.pt --pred_dir <bench.run dir> --bias <head bias> --out <npz>
"""
import argparse
import glob
import json
import os
import pickle
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from bench.common import box_iou
from fah.contact_video import ContactVideo, frame_feats, logit  # noqa: F401 (the model / features live with the inference code)

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
CONTACT_ROOT = os.environ.get("CONTACT_ROOT", os.path.join(DATA_ROOT, "contact"))
GT_ROOT = os.path.join(CONTACT_ROOT, "video_gt")  # --video_gt
SPLIT_FILE = os.path.join(CONTACT_ROOT, "splits", "video_val_by_recording.json")  # --split_file
_PART = {}


def recording(ds, name):
    """The capture a clip comes from (the train / val unit): every camera view / segment of a recording shares it."""
    if ds in ("dexycb", "oakink2"):  # <date>_<time>_<camera serial> / <sequence>_<start frame>
        return name.rsplit("_", 1)[0]
    if ds == "hot3d":  # HOT3D-Clips: the sequence the clip was cut from
        import tarfile

        from train.contact_labels import HOT3D_TARS

        tar = next((os.path.join(d, name + ".tar") for d in HOT3D_TARS if os.path.exists(os.path.join(d, name + ".tar"))), None)
        if tar is None:
            return name
        t = tarfile.open(tar)
        first = sorted(n for n in t.getnames() if n[:6].isdigit() and n.endswith(".info.json"))[0]
        return json.load(t.extractfile(first))["sequence_id"]
    return name  # HOI4D / ARCTIC: one egocentric view per sequence


def build_tracks(clip, gt, n_vert, vidx, bias=0.0):
    """clip = bench.run pickle; gt = video GT npz (or None) -> per side dict(x (T, D), tok, pred_idx, lab (T, n_vert), flag (T,),
    has (T,)). bias: the image head's contact_bias when the pickles come from a calibrated head (its dense logits back to the
    uncalibrated scale, as fah.contact_video.VideoContact)."""
    T, D = len(clip["frames"]), 3 + n_vert + 66
    rows = {} if gt is None else {(int(f), int(k), int(h)): n for n, (f, k, h) in enumerate(zip(gt["frame"], gt["kind"], gt["hand"]))}
    out = {}
    for side in (True, False):
        x, pidx = np.zeros((T, D), np.float32), -np.ones(T, np.int64)
        lab, flag, has, tok = np.zeros((T, n_vert), np.float32), np.zeros(T, np.float32), np.zeros(T, np.float32), None
        for t, fr in enumerate(clip["frames"]):
            cand = [(p.get("score", 0), k) for k, p in enumerate(fr["pred"]) if bool(p["is_right"]) == side and "contact_dense" in p]
            if not cand:
                continue
            k = max(cand)[1]
            p = fr["pred"][k]
            x[t], pidx[t] = frame_feats(p, n_vert, bias), k
            if p.get("token") is not None:
                tok = np.zeros((T, len(p["token"])), np.float16) if tok is None else tok
                tok[t] = p["token"]
            if gt is None:
                continue
            ious = [(box_iou(p["bbox"], h["bbox"]), gi) for gi, h in enumerate(fr["gt"])]
            ious = [a for a in ious if a[0] >= 0.3]
            if ious:
                r = rows.get((t, 0, max(ious)[1]))
                if r is not None and gt["valid"][r]:
                    sd = gt["sdf"][r].astype(np.float32)
                    lab[t], flag[t], has[t] = sd[vidx] < 10, float((sd < 10).any()), 1.0
        out[side] = dict(x=x, pred_idx=pidx, lab=lab, flag=flag, has=has, **({"tok": tok} if tok is not None else {}))
    return out


def windows(tracks, T, strides=(1,)):
    out = []
    for tr in tracks:
        n = len(tr["x"])
        for k in strides:
            if n < T * k:
                continue
            for s in range(0, n - T * k + 1, T * k // 2):
                sl = slice(s, s + T * k, k)
                if tr["has"][sl].sum() >= 4:
                    out.append({kk: vv[sl] for kk, vv in tr.items() if kk != "pred_idx"})
    return out


def is_val(ds, name, rule="recording"):
    """recording: crc32(recording) % 10 == 0, from the stored partition (SPLIT_FILE); pool: the clips of the contact pools' val
    split (train.finetune VAL_MOD 50 on the pool's sequence key: HOT3D clip, DexYCB date_time), i.e. clips the image head
    never trained on; clip10: crc32(clip) % 10 == 0 (camera views of one recording can fall on both sides)."""
    if rule == "recording":
        if not _PART and os.path.exists(SPLIT_FILE):
            _PART.update(json.load(open(SPLIT_FILE))["clips"])
        e = _PART.get(ds, {}).get(name)
        return (e["split"] == "val") if e else zlib.crc32(recording(ds, name).encode()) % 10 == 0
    if rule == "pool":
        key = "_".join(name.split("_")[:2]) if ds == "dexycb" else name
        return zlib.crc32(key.encode()) % 50 == 0
    return zlib.crc32(name.encode()) % 10 == 0


def kv(spec, cast=str):
    """'a=x,b=y' -> {a: cast(x), b: cast(y)} (cast=list: the values of a repeated key as a list)."""
    out = {}
    for it in (spec or "").split(","):
        if it:
            k, v = it.split("=", 1)
            out.setdefault(k, []).append(v) if cast is list else out.__setitem__(k, cast(v))
    return out


def load(spec, n_vert, vidx, split, rule="recording", debias="", exclude=""):
    """spec 'ds:dir,...' -> tracks of the train or val clips (is_val rule); debias 'ds=bias,...' (calibrated-head pickles),
    exclude 'ds=substr,...' (clips whose name contains substr are skipped, e.g. hoi4d=_C20_ = HOI4D Chair)."""
    tracks = []
    db, ex = kv(debias, float), kv(exclude, list)
    for item in spec.split(","):
        ds, d = item.split(":", 1)
        for f in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            name = os.path.basename(f)[:-4]
            if is_val(ds, name, rule) != (split == "val"):
                continue
            if any(x in name for x in ex.get(ds, [])):
                continue
            g = os.path.join(GT_ROOT, ds, name + ".npz")
            if not os.path.exists(g):
                continue
            tracks += [t for t in build_tracks(pickle.load(open(f, "rb")), np.load(g), n_vert, vidx, db.get(ds, 0.0)).values() if t["has"].sum() >= 4]
    return tracks


def dense_vidx(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return ck["trainable"]["model.keypoint_head.dense_vidx"].numpy()


def switch_weight(fl, has, w, win):
    """(B, T) loss weight 1 + w on frames within +-win frames of a GT flag switch (onset / release: flag(t) != flag(t-1), both
    frames with labels) inside the window, else 1."""
    sw = torch.zeros_like(fl)
    sw[:, 1:] = ((fl[:, 1:] > 0) != (fl[:, :-1] > 0)).float() * (has[:, 1:] > 0) * (has[:, :-1] > 0)
    near = F.max_pool1d(sw[:, None], 2 * win + 1, 1, win)[:, 0] if win > 0 else sw
    return 1 + w * near


def bce_balanced(lg, lab, has):
    npos, nneg = (lab * has).sum().clamp(min=1), ((1 - lab) * has).sum().clamp(min=1)
    w = has * (lab * 0.5 / npos + (1 - lab) * 0.5 / nneg)
    return (F.binary_cross_entropy_with_logits(lg, lab, reduction="none") * w).sum()


def train(args):
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vidx = dense_vidx(args.vidx)
    nv = len(vidx)
    tr = load(args.train, nv, vidx, "train", args.val_rule, args.debias, args.exclude)
    va = load(args.train, nv, vidx, "val", args.val_rule, args.debias, args.exclude)
    win = windows(tr, args.T, tuple(int(s) for s in args.strides.split(",")))
    vwin = windows(va, args.T, (1,))
    use_tok = args.use_tok and all("tok" in w for w in win)
    print(f"{len(tr)} train tracks -> {len(win)} windows, {len(va)} val tracks -> {len(vwin)} windows; tok {use_tok}", flush=True)
    stack = lambda ws, k: torch.as_tensor(np.stack([w[k] for w in ws]).astype(np.float32))
    X, L, Fl, H = (stack(win, k) for k in ("x", "lab", "flag", "has"))
    TK = stack(win, "tok") if use_tok else None
    model = ContactVideo(nv, args.dim, args.layers, tok_dim=TK.shape[-1] if use_tok else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)

    def evaluate():
        if not vwin:
            return {}
        model.eval()
        with torch.no_grad():
            x, lab, fl, has = (stack(vwin, k).to(dev) for k in ("x", "lab", "flag", "has"))
            tok = stack(vwin, "tok").to(dev) if use_tok else None
            lg, fg = model(x, tok)
            res = {}
            for name, l_, f_ in (("frame", x[..., 3:3 + nv], x[..., 2]), ("video", lg, fg)):
                pb, gt = (torch.sigmoid(l_) >= 0.52) & (has[..., None] > 0), (lab > 0) & (has[..., None] > 0)
                tp, fp, fn = (pb & gt).sum().item(), (pb & ~gt).sum().item(), (~pb & gt).sum().item()
                pf, gf = (f_ > 0) & (has > 0), (fl > 0) & (has > 0)
                res[f"{name}_vF1"] = 2 * tp / max(2 * tp + fp + fn, 1)
                res[f"{name}_flagF1"] = 2 * (pf & gf).sum().item() / max(pf.sum().item() + gf.sum().item(), 1)
                flips = ((pf[:, 1:] != pf[:, :-1]) & (has[:, 1:] > 0) & (has[:, :-1] > 0)).sum().item()
                res[f"{name}_flips_per100"] = 100 * flips / max(has.sum().item(), 1)
            gfl = (((fl[:, 1:] > 0) != (fl[:, :-1] > 0)) & (has[:, 1:] > 0) & (has[:, :-1] > 0)).sum().item()
            res["gt_flips_per100"] = 100 * gfl / max(has.sum().item(), 1)
        model.train()
        return res

    print("[eval 0]", evaluate(), flush=True)
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / 500) * 0.5 * (1 + np.cos(np.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.as_tensor(rng.integers(0, len(X), args.batch))
        x, lab, fl, has = X[idx].to(dev), L[idx].to(dev), Fl[idx].to(dev), H[idx].to(dev)
        tok = TK[idx].to(dev) if use_tok else None
        drop = (torch.rand(x.shape[:2], device=dev) < args.p_drop) & (x[..., 0] > 0)  # detector misses
        x = x.clone()
        x[drop] = 0
        has = has * (~drop)
        x[..., 2:3 + nv] += torch.randn_like(x[..., 2:3 + nv]) * args.logit_noise * (x[..., :1] > 0)
        x[..., 3 + nv:] += torch.randn_like(x[..., 3 + nv:]) * args.noise * (x[..., :1] > 0)
        lg, fg = model(x, tok)
        hw = has * switch_weight(fl, has, args.w_switch, args.switch_win) if args.w_switch > 0 else has  # onset / release weighting
        loss = bce_balanced(lg, lab, hw[..., None].expand_as(lab)) + args.w_flag * (
            F.binary_cross_entropy_with_logits(fg, fl, reduction="none") * hw).sum() / hw.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            print(f"[eval {step}]", evaluate(), flush=True)
            torch.save(dict(model=model.state_dict(), args=vars(args), n_vert=nv, vidx=vidx, use_tok=use_tok,
                            tok_dim=TK.shape[-1] if use_tok else 0), os.path.join(args.out, "model.pt"))


@torch.no_grad()
def apply(args):
    """Online: each frame from the last T frames of its track -> <out>.npz: clip, frame, pred index, refined dense probs, flag."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a, nv, vidx = ck["args"], ck["n_vert"], ck["vidx"]
    model = ContactVideo(nv, a["dim"], a["layers"], tok_dim=ck["tok_dim"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    T = a["T"]
    res = dict(clip=[], frame=[], pred=[], prob=[], flag=[])
    for f in sorted(glob.glob(os.path.join(args.pred_dir, "*.pkl"))):
        clip = pickle.load(open(f, "rb"))
        for side, tr in build_tracks(clip, None, nv, vidx, args.bias).items():
            n = len(tr["x"])
            pad = np.concatenate([np.zeros((T - 1, tr["x"].shape[1]), np.float32), tr["x"]])
            xs = torch.as_tensor(np.stack([pad[t:t + T] for t in range(n)])).to(dev)
            tok = None
            if ck["use_tok"] and "tok" in tr:
                tp = np.concatenate([np.zeros((T - 1, tr["tok"].shape[1]), np.float16), tr["tok"]])
                tok = torch.as_tensor(np.stack([tp[t:t + T] for t in range(n)]).astype(np.float32)).to(dev)
            lg, fg = model(xs, tok)
            for t in np.nonzero(tr["pred_idx"] >= 0)[0]:
                res["clip"].append(os.path.basename(f)[:-4])
                res["frame"].append(t)
                res["pred"].append(tr["pred_idx"][t])
                res["prob"].append(torch.sigmoid(lg[t, -1]).cpu().numpy().astype(np.float16))
                res["flag"].append(float(torch.sigmoid(fg[t, -1])))
    np.savez(args.out, **{k: np.asarray(v) for k, v in res.items()})
    print(args.pred_dir, len(res["frame"]), "hand-frames ->", args.out, flush=True)


def hysteresis_track(f, present, on, off, T):
    """Online flag of one side's track (the release rule, fah.contact_video): frames without a predicted hand never update
    the state and get no flag; a hand back after >= T such frames starts off."""
    st, miss, out = False, 0, np.zeros(len(f), bool)
    for t, (x, pr) in enumerate(zip(f, present)):
        if not pr:
            miss += 1
            continue
        if miss >= T:
            st = False
        miss = 0
        st = x > on if not st else x >= off
        out[t] = st
    return out


def split_file(args):
    """Partition of every clip of the given track dirs into train / val by recording -> SPLIT_FILE (or --out)."""
    clips, recs = {}, {}
    for item in args.train.split(","):
        ds, d = item.split(":", 1)
        for f in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            name = os.path.basename(f)[:-4]
            r = recording(ds, name)
            clips.setdefault(ds, {})[name] = dict(recording=r, split="val" if zlib.crc32(r.encode()) % 10 == 0 else "train")
            recs.setdefault(ds, set()).add(r)
    old_cross = {ds: sum(len({zlib.crc32(n.encode()) % 10 == 0 for n, e in c.items() if e["recording"] == r}) > 1 for r in recs[ds])
                 for ds, c in clips.items()}
    stats = {ds: dict(clips=len(c), recordings=len(recs[ds]), val_clips=sum(e["split"] == "val" for e in c.values()),
                      val_recordings=sum(zlib.crc32(r.encode()) % 10 == 0 for r in recs[ds]), old_rule_crossing_recordings=old_cross[ds])
             for ds, c in clips.items()}
    out = args.out or SPLIT_FILE
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(rule="val iff crc32(recording) % 10 == 0; recording = train.contact_video.recording(ds, clip)", stats=stats,
                   clips=clips), open(out, "w"), indent=1)
    print(json.dumps(stats), "->", out, flush=True)


def folds(args):
    """2-fold stacking splits by recording: fold k = the clips whose recording key has crc32 % 2 == k (the key
    train.contact_head --fold_key sequence uses). Each input is a bench.run --split file ({"clips": [...]} or
    {"clips": {group: [...]}}); the outputs keep its format."""
    meta = kv(args.clip_meta)
    key = lambda ds, c: (json.load(open(os.path.join(meta[ds], c, "meta.json")))["sequence"] if ds in meta
                         else recording(ds, os.path.splitext(os.path.basename(c))[0]))
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    for item in args.sets.split(","):
        ds, f = item.split(":", 1)
        with open(f) as f_:
            sp = json.load(f_)
        for k in (0, 1):
            pick_ = lambda cs: [c for c in cs if zlib.crc32(key(ds, c).encode()) % 2 == k]
            clips = {g: pick_(v) for g, v in sp["clips"].items()} if isinstance(sp["clips"], dict) else pick_(sp["clips"])
            with open(os.path.join(args.out, f"{ds}_train_f{k}.json"), "w") as f_:
                json.dump(dict(sp, clips=clips, note=f"fold {k} of 2 by recording"), f_, indent=1)
            stats[f"{ds}_f{k}"] = sum(map(len, clips.values())) if isinstance(clips, dict) else len(clips)
    with open(os.path.join(args.out, "folds.json"), "w") as f_:
        json.dump(stats, f_, indent=1)
    print(json.dumps(stats), flush=True)


@torch.no_grad()
def pick(args):
    """Vertex thresholds (pooled F1 on the val clips' windows) of the per-frame input and of the video model, and the flag
    hysteresis (h_on, h_off) maximising flag F1 - 0.1 x excess switches per 100 frames -> written into the checkpoint."""
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a, nv, vidx = ck["args"], ck["n_vert"], ck["vidx"]
    model = ContactVideo(nv, a["dim"], a["layers"], tok_dim=ck["tok_dim"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    va = load(a["train"], nv, vidx, "val", args.val_rule or a.get("val_rule", "clip10"), a.get("debias", ""), a.get("exclude", ""))  # a checkpoint's own split (old ones: clip10)
    out = {}
    lab_all, p1_all, p2_all, f2_tr, fl_tr = [], [], [], [], []
    for tr in va:
        n, T = len(tr["x"]), a["T"]
        pad = np.concatenate([np.zeros((T - 1, tr["x"].shape[1]), np.float32), tr["x"]])
        xs = torch.as_tensor(np.stack([pad[t:t + T] for t in range(n)])).to(dev)
        tok = None
        if ck["use_tok"] and "tok" in tr:
            tp = np.concatenate([np.zeros((T - 1, tr["tok"].shape[1]), np.float16), tr["tok"]])
            tok = torch.as_tensor(np.stack([tp[t:t + T] for t in range(n)]).astype(np.float32)).to(dev)
        lg, fg = model(xs, tok)
        h = tr["has"] > 0
        lab_all.append(tr["lab"][h] > 0)
        p1_all.append(1 / (1 + np.exp(-tr["x"][h, 3:3 + nv])))
        p2_all.append(torch.sigmoid(lg[:, -1]).cpu().numpy()[h])
        f2_tr.append((torch.sigmoid(fg[:, -1]).cpu().numpy(), tr["flag"], h, tr["x"][:, 0] > 0))
    lab, p1, p2 = np.concatenate(lab_all), np.concatenate(p1_all), np.concatenate(p2_all)
    for name, p in (("thr_frame", p1), ("thr_video", p2)):
        best = max(((2 * ((p >= t) & lab).sum() / max((p >= t).sum() + lab.sum(), 1), t) for t in np.arange(0.3, 0.96, 0.02)))
        out[name], out[name + "_f1"] = round(float(best[1]), 2), round(float(best[0]), 4)
    best = None  # blend of the per-frame and video maps: (1 - alpha) p_frame + alpha p_video
    for al in (0.25, 0.5, 0.75):
        p = (1 - al) * p1 + al * p2
        for t in np.arange(0.3, 0.96, 0.02):
            f1 = 2 * ((p >= t) & lab).sum() / max((p >= t).sum() + lab.sum(), 1)
            if best is None or f1 > best[0]:
                best = (f1, al, t)
    out["blend_alpha"], out["thr_blend"], out["thr_blend_f1"] = best[1], round(float(best[2]), 2), round(float(best[0]), 4)
    best = None
    for on in np.arange(0.4, 0.91, 0.05):
        for off in np.arange(0.1, on + 1e-6, 0.05):
            tp = fp = fn = sw = gsw = nfr = 0
            for f, g, h, pres in f2_tr:
                pf = hysteresis_track(f, pres, on, off, a["T"])
                adj = h[1:] & h[:-1]  # switches between adjacent valid frames only
                g = g > 0
                tp, fp, fn = tp + (pf & g)[h].sum(), fp + (pf & ~g)[h].sum(), fn + (~pf & g)[h].sum()
                sw, gsw, nfr = sw + ((pf[1:] != pf[:-1]) & adj).sum(), gsw + ((g[1:] != g[:-1]) & adj).sum(), nfr + h.sum()
            sc = 2 * tp / max(2 * tp + fp + fn, 1) - 0.1 * max(sw - gsw, 0) * 100 / max(nfr, 1)
            if best is None or sc > best[0]:
                best = (sc, round(float(on), 2), round(float(off), 2))
    out["h_on"], out["h_off"] = best[1], best[2]
    ck.update(out)
    torch.save(ck, args.ckpt)
    print(args.ckpt, out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True, help="ds:pred_dir,... (bench.run pickles of a contact model)")
    t.add_argument("--vidx", required=True, help="stage-1 checkpoint (its dense point indices)")
    t.add_argument("--out", required=True)
    t.add_argument("--T", type=int, default=32)
    t.add_argument("--dim", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, required=True, help="training batch size; choose for your setup")
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--w_flag", type=float, default=0.5)
    t.add_argument("--p_drop", type=float, default=0.1)
    t.add_argument("--noise", type=float, default=0.02, help="kinematic input noise (dm / root units)")
    t.add_argument("--logit_noise", type=float, default=0.3)
    t.add_argument("--strides", default="1,2,3")
    t.add_argument("--use_tok", action="store_true")
    t.add_argument("--eval_every", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--debias", default="", help="ds=bias,...: calibrated-head pickles of ds (their dense logits + bias)")
    t.add_argument("--exclude", default="", help="ds=substr,...: skip clips of ds whose name contains substr")
    t.add_argument("--w_switch", type=float, default=0.0, help="onset / release weighting: loss x (1 + w) within +-switch_win frames of a GT flag switch")
    t.add_argument("--switch_win", type=int, default=3)
    t.add_argument("--val_rule", default="recording", help="recording (stored partition) | pool (val = clips the image head never trained on) | clip10")
    t4 = sub.add_parser("split")
    t4.add_argument("--train", required=True, help="ds:pred_dir,... (every clip of these dirs)")
    t4.add_argument("--out", default="", help="default --split_file")
    t5 = sub.add_parser("folds")
    t5.add_argument("--sets", required=True, help="ds:split.json,... (bench.run --split files of the training clips)")
    t5.add_argument("--clip_meta", default="", help="ds=dir,...: <dir>/<clip>/meta.json 'sequence' is the recording (HOI4D / ARCTIC clips)")
    t5.add_argument("--out", required=True, help="writes <ds>_train_f0.json / _f1.json + folds.json")
    t3 = sub.add_parser("pick")
    t3.add_argument("--ckpt", required=True)
    t3.add_argument("--val_rule", default="", help="override the checkpoint's own val rule")
    t2 = sub.add_parser("apply")
    t2.add_argument("--ckpt", required=True)
    t2.add_argument("--pred_dir", required=True)
    t2.add_argument("--out", required=True)
    t2.add_argument("--bias", type=float, default=0.0, help="calibrated-head pickles: their contact_bias (main tolerance)")
    for q in (t, t4, t3):
        q.add_argument("--video_gt", default=GT_ROOT, help="train.contact_labels --video_gt output (<dataset>/<clip>.npz)")
        q.add_argument("--split_file", default=SPLIT_FILE, help="the stored train / val partition (`split`)")
    args = p.parse_args()
    GT_ROOT, SPLIT_FILE = getattr(args, "video_gt", GT_ROOT), getattr(args, "split_file", SPLIT_FILE)
    dict(train=train, apply=apply, pick=pick, split=split_file, folds=folds)[args.cmd](args)
