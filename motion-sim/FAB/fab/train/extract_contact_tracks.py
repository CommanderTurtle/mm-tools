"""Per-track contact-head outputs of a FAB checkpoint with a contact head (the released FAB-L / FAB-S models have one), for
the video contact model (train/train_contact_video.py).

Writes <out>/<name>/<track>.npz: per frame logits (8), feat (512, fp16; the contact head's last hidden layer), pts
(12, 3; the head's MHR keypoints in the camera frame = anchored dense 3D + camera translation), cam_t (3), frame, time
(s), ids, and the world-GT labels c / valid (8) of fab/contact_regions.py.
Sources:
  bench:<ds>   bench/datasets.py sets via train/extract_tracks.tracks_from_bench (3dpw:<split>, emdb1, emdb2); labels from
               --labels (a bench label file, ids as the bench set's)
  pool:<glob>  pool tars (RICH / BEDLAM / MOYO keys); labels from each tar's <tar>.<--suffix>.npz sidecar
  clips:<json> image lists + per-frame boxes + K per track (scripts/contact_clips.py prep); no labels

    python -m train.extract_contact_tracks --source clips:<work>/clips_spec.json --name tracks --fab_ckpt <model.pt> --out <work>
    python -m train.extract_contact_tracks --source pool:"<pool dir>/*.tar" --name <name> --fab_ckpt <model.pt> --out <dir> \
        --shard 0 --num_shards 4
"""

from __future__ import annotations

import argparse
import json
import glob
import io
import re
import tarfile
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

GEO = ("bbox_center", "bbox_scale", "ori_size", "cam_int", "affine")
PATTERNS = [  # key -> (track, frame); BEDLAM tracks also split by body shape
    (re.compile(r"^rich_\w+?-(.+_cam_\d+)-(\d+)_(\d+)$"), lambda m: (f"{m[1]}__{m[3]}", int(m[2]), 30.0)),
    (re.compile(r"^bedlam-(.+)_p\d+of\d+-(seq_\d+)_seq_\d+_(\d+)_p\d+$"), lambda m: (f"{m[1]}__{m[2]}", int(m[3]), 30.0)),
    (re.compile(r"^moyo-(.+)-(\d+)_cam(\d+)$"), lambda m: (f"{m[1]}__cam{m[3]}", int(m[2]), 30.0)),
    (re.compile(r"^egobody\w*?-(recording_.+_(?:master|sub_\d))-(\d+)_(interactee|camera_wearer)$"),
     lambda m: (f"{m[1]}__{m[3]}", int(m[2]), 30.0)),
    (re.compile(r"^3dpw_smpl-(?:train|validation)-(.+)-(\d+)_p(\d+)$"), lambda m: (f"{m[1]}__p{m[3]}", int(m[2]), 30.0)),
]


def first_side(path, names):
    """The first existing <tar>.<name>.npz of a comma-separated list, or None."""
    for n in [x for x in names.split(",") if x]:
        side = Path(f"{path[:-4]}.{n}.npz")
        if side.exists():
            return side
    return None


def pool_tracks(pattern, suffix, any_suffix="", knee_suffix=""):
    """suffix / any_suffix: comma-separated sidecar names, the first present per tar (physc,contact). Physics sidecars
    (with stable): frames with stable != 0 are masked in both. knee_suffix: the knee channels (6, 7) from that sidecar."""
    tracks = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        anyl = {}  # the dataset's own any-surface contact (RICH human-scene contact sidecars) where present
        lab, bad = {}, set()
        side = first_side(path, suffix)
        if side is not None:
            with np.load(side) as z:
                h = z["h"] if "h" in z.files else np.full(z["c"].shape, np.nan, np.float16)
                keys = z["keys"].astype(str)
                valid = z["valid"]
                if "stable" in z.files:
                    valid = valid * (z["stable"] == 0)[:, None].astype(valid.dtype)
                    bad = set(keys[z["stable"] != 0].tolist())
                lab = dict(zip(keys, zip(z["c"], valid, h)))
        side = first_side(path, any_suffix) if any_suffix else None
        if side is not None:
            with np.load(side) as z:
                keys = z["keys"].astype(str)
                va = z["valid"]
                if side.name.split(".")[-2].startswith("phys"):
                    va = va * np.array([k not in bad for k in keys], va.dtype)[:, None]
                anyl = dict(zip(keys, zip(z["c"], va)))
        side = first_side(path, knee_suffix) if knee_suffix else None
        if side is not None and lab:
            with np.load(side) as z:
                hk = z["h"] if "h" in z.files else np.full(z["c"].shape, np.nan, np.float16)
                for k, ck, vk, hh in zip(z["keys"].astype(str), z["c"], z["valid"], hk):
                    if k in lab:
                        c_, v_, h_ = (np.array(x) for x in lab[k])
                        c_[6:8], v_[6:8], h_[6:8] = ck[6:8], vk[6:8], hh[6:8]
                        lab[k] = (c_, v_, h_)
                        if k in anyl:
                            ca_, va_ = (np.array(x) for x in anyl[k])
                            va_[6:8] = 0
                            anyl[k] = (ca_, va_)
        members = {}
        with tarfile.open(path) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit(".", 1)
                if ext == "npz":
                    members.setdefault(key, {})["npz"] = tf.extractfile(m).read()
                elif ext == "jpg":
                    members.setdefault(key, {})["jpg"] = (path, m.offset_data, m.size)
        for key, parts in members.items():
            if "npz" not in parts or "jpg" not in parts or key not in lab:
                continue
            for pat, f in PATTERNS:
                m = pat.match(key)
                if m:
                    break
            else:
                continue
            z = np.load(io.BytesIO(parts["npz"]))
            track, frame, fps = f(m)
            if key.startswith("bedlam-"):
                track += f"__{zlib.crc32(np.round(z['gt_smpl_betas'][:10], 4).tobytes()):08x}"
            rec = {k: z[k] for k in GEO}
            rec.update(jpg_ref=parts["jpg"], id=f"{Path(path).stem}/{key}", c=lab[key][0], valid=lab[key][1], h=lab[key][2],
                       fps=fps)
            if key in anyl:
                rec["c_any"], rec["valid_any"] = anyl[key]
            tracks[track].append((frame, rec))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in tracks.items()}


def bench_tracks(source, root, labels):
    raise NotImplementedError("the bench:<dataset> track source is not part of this release; use clips:<spec json>")

    z = np.load(labels)
    lab = dict(zip(z["ids"].astype(str), z["c"]))
    hgt = dict(zip(z["ids"].astype(str), z["h"]))
    tracks = tracks_from_bench(source, root)
    out = {}
    for k, v in tracks.items():
        out[k] = [(fr, dict(s, c=lab.get(s["id"], np.zeros(8, np.uint8)), h=hgt.get(s["id"], np.full(8, np.nan)),
                            valid=np.full(8, int(s["id"] in lab), np.uint8), fps=30.0)) for fr, s in v]
    return out


def clip_tracks(spec):
    """clips:<spec json> (world mode's contact step, scripts/contact_clips.py prep): {track:
    {images, box (T, 4), valid (T,), K (3, 3), frames (T,), fps}} -> one track per entry, the valid frames only."""
    d = json.load(open(spec))
    out = {}
    for name, c in d.items():
        K = np.asarray(c["K"], np.float32)
        rows = []
        for img, box, ok, fr in zip(c["images"], c["box"], c["valid"], c["frames"]):
            if not ok or not np.isfinite(box).all():
                continue
            rows.append((int(fr), dict(image=img, bbox=np.asarray(box, np.float32), K=K, id=f"{name}/{int(fr)}",
                                       c=np.zeros(8, np.uint8), valid=np.zeros(8, np.uint8), h=np.full(8, np.nan),
                                       fps=float(c.get("fps", 30.0)))))
        out[name] = rows
    return out


def load_model(ckpt, teacher_dir, vis_from=""):
    from fab.dense_head import dense_hr, dense_part_depth, dense_xyz_layers
    from fab.decoder_model import build_fab_decoder

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    ta = state.get("args", {})
    model, tb = build_fab_decoder(Path(teacher_dir), ta.get("backbone", "dinov3_vitb16"), ta.get("size", 512),
                                  pretrained=False, dec_layers=ta.get("dec_layers", 0))
    del tb
    w = dict(state["model"])
    w.update(state.get("ema") or {})
    from fab.dense_head import dense_head_kwargs

    model.add_dense_head(w["dense_head.surface_idx"].tolist(), residual=bool(int(w.get("dense_head.residual", torch.tensor(0)))),
                         parts="dense_head.part_emb" in w, xyz_layers=dense_xyz_layers(w), part_depth=dense_part_depth(w),
                         hr=dense_hr(w), **dense_head_kwargs(w)).cuda()
    from fab.contact_head import contact_head_args

    model.add_contact_head(**contact_head_args(w)).cuda()
    if vis_from:  # a visibility head trained on the same frozen model: out["vis_logits"]
        from fab.vis_head import has_vis_head, vis_head_args

        vs = torch.load(vis_from, map_location="cpu", weights_only=False, mmap=True)
        w.update({k: v for k, v in {**vs["model"], **(vs.get("ema") or {})}.items() if k.startswith("vis_head.")})
        if has_vis_head(w):
            model.add_vis_head(**vis_head_args(w)).cuda()
    model.load_state_dict(w, strict=False)
    model = model.eval()
    model.body.head_pose.enable_fast()
    from fab.precision import fast_inference

    fast_inference(model)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--root", default="")
    p.add_argument("--labels", default="")
    p.add_argument("--suffix", default="contact")
    p.add_argument("--any_suffix", default="", help="pool: any-contact sidecars (e.g. contact_hsc) stored as c_any")
    p.add_argument("--knee_suffix", default="", help="pool: the knee channels from this sidecar (e.g. contact)")
    p.add_argument("--fab_ckpt", required=True)
    p.add_argument("--teacher_dir", default="", help="SAM 3D Body config + MHR model dir (default: fab.paths.sam3d_dir())")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--min_len", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--vis_from", default="", help="also store the visibility head's P(visible / occluded / out) at the "
                   "contact points (vis, (12, 3)) from this checkpoint's vis_head.* (same frozen model)")
    a = p.parse_args()
    if not a.teacher_dir:
        from fab import paths

        a.teacher_dir = str(paths.sam3d_dir())

    pool = a.source.startswith("pool:")
    if a.source.startswith("clips:"):  # world-mode clips (contact_clips.py prep): images + boxes + K per track
        tracks = clip_tracks(a.source[6:])
    else:
        tracks = pool_tracks(a.source[5:], a.suffix, a.any_suffix, a.knee_suffix) if pool else bench_tracks(a.source[6:], a.root, a.labels)
    names = sorted(k for k, v in tracks.items() if len(v) >= a.min_len)[a.shard :: a.num_shards]
    out_dir = Path(a.out) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.source}: {len(tracks)} tracks, {len(names)} in this shard", flush=True)
    model = load_model(a.fab_ckpt, a.teacher_dir, a.vis_from)
    size = model.size
    if not pool:
        from torchvision.transforms import ToTensor

        from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
        from fab.data.utils.prepare_batch import prepare_batch

        transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                             VisionTransformWrapper(ToTensor())])
    files = {}
    for n, name in enumerate(names):
        path = out_dir / f"{name.replace('/', '_')}.npz"
        if path.exists():
            continue
        crops, geo, meta = [], [], []
        for frame, s in tracks[name]:
            if pool:
                tp, off, nb = s["jpg_ref"]
                fh = files.get(tp) or files.setdefault(tp, open(tp, "rb"))
                fh.seek(off)
                img = cv2.cvtColor(cv2.imdecode(np.frombuffer(fh.read(nb), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                crop = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                aff = s["affine"].astype(np.float32)
                if crop.shape[-1] != size:
                    aff = aff * (size / crop.shape[-1])
                    crop = torch.nn.functional.interpolate(crop[None], size=(size, size), mode="bilinear",
                                                           align_corners=False)[0]
                geo.append((s["bbox_center"], s["bbox_scale"], s["ori_size"], s["cam_int"], aff))
            else:
                image = cv2.imread(s["image"])
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                batch = prepare_batch(image, transform, s["bbox"][None].astype(np.float32), None, None,
                                      cam_int=torch.from_numpy(s["K"]).float()[None])
                crop = batch["img"][0, 0]
                if crop.shape[-1] != size:
                    crop = torch.nn.functional.interpolate(crop[None], size=(size, size), mode="area")[0]
                geo.append((batch["bbox_center"][0, 0].numpy(), batch["bbox_scale"][0, 0].numpy(),
                            batch["ori_img_size"][0, 0].numpy(), s["K"],
                            batch["affine_trans"][0, 0].numpy() * (size / batch["img"].shape[-1])))
            crops.append(crop)
            meta.append((frame, s["id"], s["c"], s["valid"], s["fps"], s["h"],
                         s.get("c_any", np.zeros(8, np.uint8)), s.get("valid_any", np.zeros(8, np.uint8))))
        if len(crops) < a.min_len:
            continue
        res = defaultdict(list)
        for i in range(0, len(crops), a.batch):
            sl = slice(i, i + a.batch)
            g = [torch.from_numpy(np.stack([x[j] for x in geo[sl]]).astype(np.float32)).cuda() for j in range(5)]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(torch.stack(crops[sl]).cuda(), g[0], g[1], g[2], g[3], affine=g[4], return_vertices=False)
            cam_t = out["pred_cam_t"].float()
            res["logits"].append(out["contact_logits"].float().cpu().numpy())
            res["feat"].append(out["contact_feat"].half().cpu().numpy())
            res["pts"].append((out["contact_pts"].float() + cam_t[:, None]).cpu().numpy())
            res["cam_t"].append(cam_t.cpu().numpy())
            if "vis_logits" in out:  # (B, 84, 3) -> the contact points' (fab/contact_head.CONTACT_POINTS)
                from fab.contact_head import CONTACT_POINTS

                res["vis"].append(torch.softmax(out["vis_logits"].float(), -1)[:, CONTACT_POINTS].half().cpu().numpy())
            if "dense_uv" in out:  # the contact points' image px: crop px (dense_uv is in crop units, x size as the
                from fab.contact_head import CONTACT_POINTS  # predictor's keypoints2d) through the inverse affine

                A3 = torch.cat([g[4], torch.tensor([0.0, 0.0, 1.0], device=g[4].device).expand(len(g[4]), 1, 3)], 1)
                uvc = out["dense_uv"][:, CONTACT_POINTS].float() * size
                q = torch.einsum("bij,bnj->bni", torch.linalg.inv(A3), torch.cat([uvc, torch.ones_like(uvc[..., :1])], -1))
                res["uv"].append(q[..., :2].cpu().numpy())
        fr = np.array([m[0] for m in meta], np.int64)
        fps = meta[0][4]
        np.savez(path, **{k: np.concatenate(v) for k, v in res.items()}, frame=fr, time=fr / fps, fps=np.float32(fps),
                 ids=np.array([m[1] for m in meta]),
                 c=(lambda c_: c_.astype(np.uint8) if np.isin(c_, (0, 1)).all() else c_.astype(np.float16))(
                     np.stack([m[2] for m in meta])),  # soft labels (0 / 0.5 / 1) stay fractional
                 valid=np.stack([m[3] for m in meta]).astype(np.uint8), h=np.stack([m[5] for m in meta]).astype(np.float16),
                 c_any=np.stack([m[6] for m in meta]).astype(np.uint8), valid_any=np.stack([m[7] for m in meta]).astype(np.uint8))
        if n % 50 == 0:
            print(f"[{n + 1}/{len(names)}] {name}: {len(crops)} frames", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
