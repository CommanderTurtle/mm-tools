"""Hand-object contact head on a frozen FAH network: HandKeypointHead(contact=L) logits per dense surface point + one per hand,
trained on the contact labels of train.contact_labels (GT MANO vertex within 10 mm of, or inside, a GT object mesh; with
--dists also at 5 / 20 / 35 mm). Only the contact_* parameters train (their inputs are detached in the head), so joints,
points, mesh and parts are bit-identical to the init model; the output is <out>/contact.pt (the keypoint_head.contact_*
tensors + the arguments, the format of <weights>/<model>/contact.pt), read by bench.methods.fah (contact_ckpt) next to
the init model.pt. Calibrate it afterwards (bench.contact_eval pick_thr + calibrate) so that 0.5 is the validation threshold.

Views as train.finetune (LabeledCrops: random square window of the 640 crop, photometric, box jitter; student-only low light /
ego degradation), no teacher. Vertex loss: BCE with per-batch class-balanced weights (positives and negatives of the valid
crops each weigh 1/2) on the kept dense points (labels at their template vertices), per tolerance with --dists; hand loss:
BCE on "any vertex in contact". --zero_sources pools carry no objects (every vertex labelled 0, e.g. InterHand2.6M).
--branch k: a trainable copy of the encoder's last k blocks feeds the head (its own learning rate --branch_lr).

Released recipe (FAH-S / FAH-B / FAH-L / FAH-H: same arguments, --size 448 for FAH-H; 24k steps):

    python -m train.contact_head --crops $DATA_ROOT/crops --contact $DATA_ROOT/contact/root \
        --init_ckpt $WEIGHTS/fah/model.pt --detector_weights $WEIGHTS/shared/detector.pt \
        --sources hot3d_lab_mesh,dexycb_lab,ho3d_lab,oakink2_ego_lab,oakink2_lab,interhand_lab,mow_train_c35,arctic_ego,hoi4d \
        --weights 1,0.6,0.3,0.8,0.5,0.3,0.05,0.5,0.5 --zero_sources interhand_lab --layers 4 --contact_dim 256 --branch 4 \
        --branch_lr 1e-4 --dists 5,10,20,35 --steps 24000 --batch_size "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --lr 3e-4 --eval_every 2000 --workers 8 \
        --out $CKPT_ROOT/contact/fah

Video model training tracks (2-fold stacking): two fold heads with the same arguments, --steps 8000 --fold_out 0 / 1
--fold_pools hot3d_lab_mesh,dexycb_lab,arctic_ego,hoi4d --fold_key sequence; fold head k is run on the training clips
of fold k (train.contact_video folds) and those tracks train the video model (docs/contact.md).
"""
import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.distill import seed_worker
from train.finetune import LabeledCrops, Trainable, ego_degrade, low_light, make_batch

THRESH_MM = 10.0
FOLD_POOLS = ("hot3d_lab_mesh", "dexycb_lab")  # pools whose sequences the video model's training tracks come from


def fold_key(r, key="src"):
    """2-fold stacking key of a pool record: src = src before ':' (HOT3D clip, DexYCB date_time); sequence = the record's recording
    (sequence_id: HOT3D sequence, DexYCB date_time, HOI4D / ARCTIC sequence), so every clip / view of a capture is in one fold."""
    if key == "sequence" and r.get("sequence_id"):
        return str(r["sequence_id"])
    return r.get("src", r["file"]).split(":")[0]


class ContactCrops(LabeledCrops):
    """LabeledCrops + contact (778,) {0, 1} at the main tolerance (thresh), has_contact (valid label), contact_flag (any vertex
    in contact); with dists also contact_k (778, K) / contact_flag_k (K,) at each tolerance and has_k (K,): the tolerances a
    pool labels (<contact_root>/<pool>/channels.json {"dists": [...]}, e.g. MOW = 35 mm only; default all)."""

    def __init__(self, root, sources, contact_root, zero_sources=(), thresh=THRESH_MM, min_pen=-25.0, dists=None, fold_out=-1,
                 fold_pools=FOLD_POOLS, fold_key_by="src", sdf_out=False, **kw):
        super().__init__(root, sources, **kw)
        self.sdf_out = sdf_out  # also the signed distances (778,) mm (clipped +-500; zero sources 1e3 -> 500), for --bnd_w
        if fold_out >= 0:  # 2-fold stacking: drop one half of the fold pools' sequences (the clips the video model trains on)
            drop = [k for k, s in enumerate(sources) if s in fold_pools]
            keep = [i for i, r in enumerate(self.items) if int(self.src[i]) not in drop
                    or zlib.crc32(fold_key(r, fold_key_by).encode()) % 2 != fold_out]
            self.items, self.src = [self.items[i] for i in keep], self.src[keep]
        self.thresh, self.zero = thresh, {k for k, s in enumerate(sources) if s in zero_sources}
        self.dists = np.asarray(dists if dists is not None else [thresh], np.float32)
        self.lab, self.has_k = {}, {}  # source -> (file -> row, sdf memmap); source -> (K,) labelled tolerances
        for k, s in enumerate(sources):
            cf = os.path.join(contact_root, s, "channels.json")
            dk = json.load(open(cf))["dists"] if os.path.exists(cf) else list(self.dists)
            self.has_k[k] = np.asarray([float(d in dk) for d in self.dists], np.float32)
            f = os.path.join(contact_root, s, "contact.jsonl")
            if k in self.zero or not os.path.exists(f):
                continue
            m = [json.loads(l) for l in open(f)]
            # implausible GT (deep object penetration, a fit error) is ignored rather than learnt
            ok = {q["file"]: n for n, q in enumerate(m) if q["valid"] and q["min_mm"] >= min_pen}
            self.lab[k] = (ok, np.load(os.path.join(contact_root, s, "sdf.npy"), mmap_mode="r"))
        keep = [i for i, r in enumerate(self.items) if int(self.src[i]) in self.zero or r["file"] in self.lab.get(int(self.src[i]), ({},))[0]]
        self.items, self.src = [self.items[i] for i in keep], self.src[keep]  # labelled crops only

    def __getitem__(self, i):
        d = super().__getitem__(i)
        k = int(self.src[i])
        if k in self.zero:
            sd = np.full(778, 1e3, np.float32)
        else:
            ok, sdf = self.lab[k]
            sd = np.asarray(sdf[ok[self.items[i]["file"]]], np.float32)
        c, ck = (sd < self.thresh).astype(np.float32), (sd[:, None] < self.dists[None]).astype(np.float32)
        hk = self.has_k[k]
        has = float(d["present"] * d["side_ok"])
        main = float(hk[list(self.dists).index(self.thresh)]) if self.thresh in self.dists else 1.0
        d.update(contact=torch.from_numpy(c), has_contact=torch.tensor(has * main), contact_flag=torch.tensor(float(c.any())),
                 contact_k=torch.from_numpy(ck), has_k=torch.from_numpy(hk * has), contact_flag_k=torch.from_numpy(ck.any(0).astype(np.float32)))
        if self.sdf_out:
            d["sdf"] = torch.from_numpy(np.clip(sd, -500, 500).astype(np.float32))
        return d


def spectral_map(vi, dev):
    from fah.models.heads.spectral import template_basis

    U, Mw = template_basis(os.environ.get("MANO_DIR", "data/mano"), min(256, len(vi) // 2))
    U, mw = U.astype(np.float64), Mw.astype(np.float64)[vi]
    UW = U[vi].T * mw[None]
    return torch.as_tensor(U @ np.linalg.solve(UW @ U[vi] + 1e-3 * np.eye(U.shape[1]), UW), dtype=torch.float32, device=dev)


def build(args, dev):
    """The init FAH network (bench.methods.fah loader) with a contact head added to its keypoint head."""
    from bench.methods.fah import FAHMethod
    from fah.models.heads.keypoint_head import HandKeypointHead

    m = FAHMethod(checkpoint=args.checkpoint, device="cuda", finetune_ckpt=args.init_ckpt, verify=True, min_crop=1.1,
                  detector_weights=args.detector_weights)
    model, enc = m.est.model, m.est.student
    sd = {k[6:]: v for k, v in torch.load(args.init_ckpt, map_location="cpu", weights_only=False)["trainable"].items() if k.startswith("model.")}
    old = model.keypoint_head
    new = HandKeypointHead(img_dim=sd["keypoint_head.img_proj.0.weight"].shape[1], surface_idx=sd["keypoint_head.surface_idx"].tolist(),
                           pool=True, dense=sd["keypoint_head.dense"].tolist(), dim=sd["keypoint_head.point"].shape[1],
                           stride=4 if "keypoint_head.up.3.weight" in sd else 8, depth="keypoint_head.depth.1.weight" in sd,
                           vis="keypoint_head.vis.1.weight" in sd, dense_vidx=sd["keypoint_head.dense_vidx"].tolist() if "keypoint_head.dense_vidx" in sd else None,
                           parts=sd["keypoint_head.part_emb"].shape[0] if "keypoint_head.part_emb" in sd else 0,
                           part_hr="keypoint_head.part_refine.0.weight" in sd,
                           part_depth=sum(1 for k in sd if k.startswith("keypoint_head.part_conv.") and k.endswith(".weight") and sd[k].dim() == 4) or 2,
                           contact=args.layers, contact_dists=[float(x) for x in args.dists.split(",")], contact_dim=args.contact_dim,
                           contact_tokmem=args.tokmem)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("contact_") for k in missing), (missing[:4], unexpected[:4])
    for k, v in vars(old).items():  # plain attributes set by the estimator (e.g. local_r)
        if not k.startswith("_") and k not in vars(new):
            setattr(new, k, v)
    model.keypoint_head = new.to(dev)
    if args.branch:  # trainable contact branch: copies of the encoder's last blocks on its hidden state
        from fah.models.heads.keypoint_head import attach_contact_branch

        attach_contact_branch(enc, model.keypoint_head, args.branch)
        model.keypoint_head.contact_branch_ckpt = args.branch_ckpt
    model.requires_grad_(False)
    enc.requires_grad_(False)
    for n, p in model.keypoint_head.named_parameters():
        p.requires_grad_(n.startswith("contact_"))
    return model, enc.float()


def metrics(logits, lab, has, thr=0.5):
    """Vertex P / R / F1 per hand (HACO-style: per-sample, then mean over hands with labels) at threshold thr."""
    p = (torch.sigmoid(logits) > thr).float()
    tp, fp, fn = (p * lab).sum(1), (p * (1 - lab)).sum(1), ((1 - p) * lab).sum(1)
    P, R = tp / (tp + fp).clamp(min=1), tp / (tp + fn).clamp(min=1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    both0 = (tp + fp + fn) == 0  # no contact and none predicted: perfect
    f1 = torch.where(both0, torch.ones_like(f1), f1)
    sel = has > 0
    return P[sel], R[sel], f1[sel]


def main(args):
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, enc = build(args, dev)
    S, S_enc = 16 * enc.grid, enc.input_size
    vidx = model.keypoint_head.dense_vidx.to(dev) if hasattr(model.keypoint_head, "dense_vidx") else torch.arange(778, device=dev)
    trainable = Trainable(model, enc).to(dev)
    params = [p for p in trainable.parameters() if p.requires_grad]
    br = {id(p) for n, p in model.keypoint_head.named_parameters() if n.startswith(("contact_branch.blocks.", "contact_branch.norm."))}
    groups = [dict(params=[p for p in params if id(p) not in br], lr_mult=1.0)]
    if br:  # the copied (pretrained) blocks at a lower learning rate
        groups.append(dict(params=[p for p in params if id(p) in br], lr_mult=args.branch_lr / args.lr))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    sources = args.sources.split(",")
    zero = tuple(args.zero_sources.split(",")) if args.zero_sources else ()
    kw = dict(size=args.size, box_jitter=args.box_jitter, min_crop=args.min_crop)
    dists = [float(x) for x in args.dists.split(",")]
    if args.bnd_w > 0:
        kw.update(sdf_out=True)
    ds = ContactCrops(args.crops, sources, args.contact, zero, args.thresh, args.min_pen, dists=dists, fold_out=args.fold_out,
                      fold_pools=tuple(args.fold_pools.split(",")), fold_key_by=args.fold_key, **kw)
    val = ContactCrops(args.crops, sources, args.contact, zero, args.thresh, args.min_pen, aug=False, split="val", dists=dists, **kw)
    weights = np.asarray([float(w) for w in args.weights.split(",")] if args.weights else [1.0] * len(sources))
    counts = np.bincount(ds.src, minlength=len(sources))
    w = weights[ds.src] / np.maximum(counts[ds.src], 1)
    print(f"train {len(ds)} / val {len(val)} labelled crops; per source {dict(zip(sources, counts.tolist()))}; S={S} S_enc={S_enc}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "FAH"), entity=os.environ.get("WANDB_ENTITY"),
                         name=args.run_name or os.path.basename(args.out), config=vars(args), dir=args.out)
    ratio = S / args.size

    def forward(b, aug):
        x = b["img"].float() / 255.0
        if aug:
            x = ego_degrade(low_light(x, args.p_dark), args.p_ego)
        bs = dict(b, A=b["A"] * ratio)
        bs["img_f"] = F.interpolate(x, size=(S, S), mode="bilinear", antialias=True) if S != args.size else x
        x_enc = F.interpolate(x, size=(S_enc, S_enc), mode="bilinear", antialias=True) if S_enc != args.size else x
        out = trainable(make_batch(bs, S), x_enc)
        return out["contact_logits"], out["contact_hand"]

    @torch.no_grad()
    def evaluate(step):
        idx = np.linspace(0, len(val) - 1, min(len(val), args.val_num)).astype(int)
        res = {}
        for b in DataLoader(torch.utils.data.Subset(val, idx), batch_size=args.batch_size, num_workers=4):
            b = {k: v.to(dev) for k, v in b.items()}
            lg_, hl_ = forward(b, False)
            P, R, f1 = metrics(lg_.float(), b["contact"][:, vidx], b["has_contact"])
            hp = (hl_ > 0).float()
            for s in b["src"][b["has_contact"] > 0].unique().tolist():
                sel = b["src"][b["has_contact"] > 0] == s
                r = res.setdefault(sources[s], [[], [], [], [], []])
                r[0] += P[sel].tolist()
                r[1] += R[sel].tolist()
                r[2] += f1[sel].tolist()
                hs = (b["src"] == s) & (b["has_contact"] > 0)
                r[3] += hp[hs].tolist()
                r[4] += b["contact_flag"][hs].tolist()
        m = {}
        for s, (P, R, f1, hp, hf) in res.items():
            hp, hf = np.asarray(hp), np.asarray(hf)
            m.update({f"P/{s}": float(np.mean(P)), f"R/{s}": float(np.mean(R)), f"F1/{s}": float(np.mean(f1)),
                      f"hand_F1/{s}": float(2 * (hp * hf).sum() / max(hp.sum() + hf.sum(), 1))})
        print(f"[eval {step}] " + " ".join(f"{k}={v:.3f}" for k, v in sorted(m.items())), flush=True)
        if run:
            run.log({f"eval/{k}": v for k, v in m.items()}, step=step)
        return m

    step, epoch, t_last, seen = 0, 0, time.time(), 0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last):  # resume (preemption): contact head + optimiser
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.keypoint_head.load_state_dict(ck["head"], strict=False)
        opt.load_state_dict(ck["opt"])
        step, epoch = ck["step"], ck["epoch"]
        print(f"resumed at step {step}", flush=True)
    else:
        evaluate(0)
    while step < args.steps:
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        idx = torch.multinomial(torch.as_tensor(w), args.batch_size * 500, replacement=True, generator=g)
        for b in DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, drop_last=True, prefetch_factor=4, worker_init_fn=seed_worker):
            if step >= args.steps:
                break
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            for gr in opt.param_groups:
                gr["lr"] = lr * gr.get("lr_mult", 1.0)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            lg, hl = forward(b, True)
            kh = model.keypoint_head
            if len(dists) > 1:  # every tolerance: class-balanced BCE per tolerance (its own balance), hand BCE per tolerance
                LG, HL = kh.last_contact_k.float(), kh.last_contact_hand_k.float()
                lab_k, has_k = b["contact_k"][:, vidx], b["has_k"]  # (B, n, K), (B, K)
                lv, lh = torch.zeros((), device=dev), torch.zeros((), device=dev)
                for j in range(len(dists)):
                    hj = has_k[:, j:j + 1]
                    if args.bnd_w > 0:  # boundary weights (signed distance near this tolerance); the classes still weigh 1/2 each
                        m_ = 1 + args.bnd_w * torch.exp(-((b["sdf"][:, vidx] - dists[j]) / args.bnd_sigma) ** 2)
                        pos, neg = lab_k[..., j] * hj * m_, (1 - lab_k[..., j]) * hj * m_
                        w_ = pos * 0.5 / pos.sum().clamp(min=1e-6) + neg * 0.5 / neg.sum().clamp(min=1e-6)
                    else:
                        npos, nneg = (lab_k[..., j] * hj).sum().clamp(min=1), ((1 - lab_k[..., j]) * hj).sum().clamp(min=1)
                        w_ = hj * (lab_k[..., j] * 0.5 / npos + (1 - lab_k[..., j]) * 0.5 / nneg)
                    lv = lv + (F.binary_cross_entropy_with_logits(LG[..., j], lab_k[..., j], reduction="none") * w_).sum()
                    lh = lh + (F.binary_cross_entropy_with_logits(HL[:, j], b["contact_flag_k"][:, j], reduction="none") * hj[:, 0]).sum() \
                        / hj.sum().clamp(min=1)
                lv, lh = lv / len(dists), lh / len(dists)
                lab, has = b["contact"][:, vidx], b["has_contact"][:, None]
                npos, nneg = (lab * has).sum().clamp(min=1), ((1 - lab) * has).sum().clamp(min=1)
            else:
                lg, lab, has = lg.float(), b["contact"][:, vidx], b["has_contact"][:, None]
                npos, nneg = (lab * has).sum().clamp(min=1), ((1 - lab) * has).sum().clamp(min=1)
                wv = has * (lab * 0.5 / npos + (1 - lab) * 0.5 / nneg)  # class-balanced: each class weighs 1/2 of the batch
                lv = (F.binary_cross_entropy_with_logits(lg, lab, reduction="none") * wv).sum()
                if args.w_f1 > 0:  # soft F1 per hand with labels
                    pr = torch.sigmoid(lg)
                    tp, sp, sl = (pr * lab).sum(1), pr.sum(1), lab.sum(1)
                    lv = lv + args.w_f1 * ((1 - (2 * tp + 1) / (sp + sl + 1)) * has[:, 0]).sum() / has.sum().clamp(min=1)
                lh = (F.binary_cross_entropy_with_logits(hl.float(), b["contact_flag"], reduction="none") * has[:, 0]).sum() \
                    / has.sum().clamp(min=1)
            loss = lv + args.w_hand * lh
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            seen += b["img"].shape[0]
            if step % 50 == 0:
                log = dict(loss=loss.item(), lv=lv.item(), lh=lh.item(), pos_frac=(npos / (npos + nneg)).item(), grad_norm=gn.item(), lr=lr,
                           img_per_s=seen / (time.time() - t_last))
                print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in log.items()), flush=True)
                if run:
                    run.log({f"train/{k}": v for k, v in log.items()}, step=step)
                t_last, seen = time.time(), 0
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
                ck = dict(trainable={f"model.keypoint_head.{k}": v.cpu() for k, v in model.keypoint_head.state_dict().items() if k.startswith("contact_")},
                          contact_args=vars(args), step_contact=step, base=os.path.basename(args.init_ckpt))  # the contact.pt format
                torch.save(ck, os.path.join(args.out, "contact.pt.tmp"))
                os.replace(os.path.join(args.out, "contact.pt.tmp"), os.path.join(args.out, "contact.pt"))
                torch.save(dict(head={k: v for k, v in model.keypoint_head.state_dict().items() if k.startswith("contact_")},
                                opt=opt.state_dict(), step=step, epoch=epoch + 1), last + ".tmp")
                os.replace(last + ".tmp", last)
        epoch += 1
    if run:
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", required=True)
    p.add_argument("--contact", required=True, help="train.contact_labels output root (<pool>/contact.jsonl + sdf.npy [+ channels.json])")
    p.add_argument("--init_ckpt", required=True, help="the frozen FAH network (a released <weights>/<model>/model.pt or a train.finetune checkpoint)")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--detector_weights", default="", help="only to build the estimator (not used)")
    p.add_argument("--sources", required=True)
    p.add_argument("--weights", default="")
    p.add_argument("--zero_sources", default="", help="pools without objects: every vertex labelled 0")
    p.add_argument("--thresh", type=float, default=THRESH_MM, help="contact = signed distance < thresh (mm)")
    p.add_argument("--min_pen", type=float, default=-25.0, help="crops with a deeper GT object penetration (mm) are ignored")
    p.add_argument("--layers", type=int, default=2, help="contact head attention layers")
    p.add_argument("--contact_dim", type=int, default=128, help="contact head width")
    p.add_argument("--tokmem", action="store_true", help="the encoder tokens join the head's cross-attention memory")
    p.add_argument("--fold_out", type=int, default=-1, help="0 / 1: train without that half of the fold pools' sequences (stacking)")
    p.add_argument("--fold_pools", default=",".join(FOLD_POOLS), help="pools the fold split applies to")
    p.add_argument("--fold_key", default="src", help="src (src before ':') | sequence (the record's sequence_id: by recording)")
    p.add_argument("--branch", type=int, default=0, help="trainable contact branch: copies of the encoder's last k blocks")
    p.add_argument("--branch_lr", type=float, default=3e-5, help="learning rate of the branch's copied blocks")
    p.add_argument("--branch_ckpt", action="store_true", help="activation checkpointing in the branch blocks (memory only, e.g. --branch 24)")
    p.add_argument("--dists", default="10", help="contact tolerances (mm), e.g. 5,10,20,35; 10 = the main output")
    p.add_argument("--bnd_w", type=float, default=0.0, help="> 0: BCE weight x (1 + bnd_w exp(-((d - tol) / bnd_sigma)^2)) near each tolerance")
    p.add_argument("--bnd_sigma", type=float, default=5.0, help="boundary weight width (mm)")
    p.add_argument("--w_hand", type=float, default=0.5)
    p.add_argument("--w_f1", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, required=True, help="training samples per process; choose for your setup")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--box_jitter", type=float, default=1.6)
    p.add_argument("--min_crop", type=float, default=1.1)
    p.add_argument("--p_dark", type=float, default=0.3)
    p.add_argument("--p_ego", type=float, default=0.25)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--val_num", type=int, default=3000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--run_name", default="")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (opt-in; WANDB_PROJECT / WANDB_ENTITY from the environment)")
    main(p.parse_args())
