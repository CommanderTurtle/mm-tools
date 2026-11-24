"""Offline WiLoR 3D targets (external KD) for a crop pool: WiLoR's MANO hand on every crop as root-relative shape in
the crop's camera frame -> <out>/<pool>/ext_kd.npz (files (N,), j3d / j3d_mesh (N, 21, 3) f32, v3d (N, 778, 3) f16,
err2d (N,) f32; --v3d_npy: v3d as ext_kd_v3d.npy, same row order), read by train.finetune.LabeledCrops.

Crops are right-hand oriented (left hands mirrored): WiLoR runs with is_right = 1, no detector. Its box is the record's
label box mapped into the crop by M (else the teacher region, the central CROP / CONTEXT square) x rescale_factor, as
WiLoR's demo does for detector boxes. Its orientation is used as is: rotating it onto the box-centre ray of the true crop
camera (K_crop = M o K) raised the root-relative error on 200 HOT3D crops from 9.9 to 60 mm (WiLoR is trained on
camera-frame labels of off-centre crops), so the camera is not needed.
Joints, OpenPose order: j3d is the kinematic convention of the HOT3D / EgoStandard labels (smplx FK joints of WiLoR's
MANO pose + betas, J_transformed, + fingertip vertices 744/320/443/554/671 = WiLoR's own pred_keypoints_3d); j3d_mesh is
MANO's 16-joint regressor on the posed vertices + the same tips (bench.common.VertexJoints). j3d and v3d are relative to
the kinematic wrist, j3d_mesh to its own wrist. err2d: WiLoR's own 2D vs the record's labelled 2D (mean px / hand
size; projected 3D labels where a pool has no 2D labels), for QC. The parts also keep WiLoR's MANO rotations + betas.

    python -m train.wilor_kd --root $DATA_ROOT/crops --pool coco_wb --out $DATA_ROOT/kd/wilor
Resumable: chunks go to <out>/<pool>/parts_kin/ (skipped if present), merged into ext_kd.npz at the end.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train.build_crops import CONTEXT, SIZE as CROP  # (train.distill imports fah, not in the WiLoR env)


def label_box(r):
    """Record box (original image, unmirrored) -> crop pixels (xyxy)."""
    x1, y1, x2, y2 = r["box"]
    if not r["is_right"]:
        w = r["hw"][1]
        x1, x2 = w - x2 - 1, w - x1 - 1
    M = np.asarray(r["M"], np.float64)
    p = np.array([[x1, y1], [x2, y2]]) @ M[:, :2].T + M[:, 2]
    return np.concatenate([p.min(0), p.max(0)])


class Crops(Dataset):
    def __init__(self, root, pool, recs, cfg, vitdet, box, rescale):
        self.root, self.pool, self.recs, self.cfg, self.vitdet = root, pool, recs, cfg, vitdet
        self.box, self.rescale = box, rescale

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = cv2.imread(os.path.join(self.root, self.pool, r["file"]))
        ok = img is not None
        if not ok:
            img = np.zeros((CROP, CROP, 3), np.uint8)
        side = CROP / CONTEXT
        b = label_box(r) if self.box == "label" and "box" in r else \
            np.array([(CROP - side) / 2, (CROP - side) / 2, (CROP + side) / 2, (CROP + side) / 2])
        item = self.vitdet(self.cfg, img, b[None], np.ones(1), rescale_factor=self.rescale)[0]
        kp = np.zeros((21, 3))
        if "kp2d" in r:
            kp = np.asarray(r["kp2d"], np.float64)
        elif "joints3d" in r and "K" in r:  # 3D labels only (EgoDex): their projection into the crop
            M, uvw = np.asarray(r["M"], np.float64), np.asarray(r["joints3d"], np.float64) @ np.asarray(r["K"], np.float64).T
            kp[:, :2] = (uvw[:, :2] / uvw[:, 2:]) @ M[:, :2].T + M[:, 2]
            kp[:, 2] = (uvw[:, 2] > 0) & (kp[:, :2] >= 0).all(1) & (kp[:, :2] < CROP).all(1)
        lab = kp[:, 2] > 0
        size = np.ptp(kp[lab, :2], 0).max() if lab.sum() >= 4 else 0.0  # hand size for err2d (>= 4 labelled joints)
        return dict(img=torch.as_tensor(item["img"]), box_center=torch.as_tensor(item["box_center"], dtype=torch.float32),
                    box_size=torch.tensor(float(item["box_size"])), right=torch.tensor(1.0),
                    kp2d=torch.as_tensor(kp, dtype=torch.float32), size=torch.tensor(float(size)), ok=torch.tensor(ok),
                    idx=torch.tensor(i))


@torch.no_grad()
def main(args):
    sys.path.insert(0, args.wilor_repo)
    cwd = os.getcwd()
    os.chdir(args.wilor_repo)  # WiLoR resolves ./mano_data relative to CWD
    try:
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor

        model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                cfg_path="./pretrained_models/model_config.yaml")
    finally:
        os.chdir(cwd)
    from bench.common import MANO21_TO_OPENPOSE, MANO_TIP_VERTS, VertexJoints

    dev = torch.device("cuda")
    model = model.to(dev).eval()
    J = torch.as_tensor(VertexJoints(args.mano_dir).J, dtype=torch.float32, device=dev)  # (16, 778)

    recs = [r for r in map(json.loads, open(os.path.join(args.root, args.pool, "index.jsonl"))) if not r.get("negative")]
    if args.limit:
        recs = recs[: args.limit]
    out = os.path.join(args.out, args.name or args.pool)
    os.makedirs(os.path.join(out, "parts_kin"), exist_ok=True)
    dyn = args.shard.startswith("dyn:")  # processes claim chunks first come first served (uneven GPUs)
    si, sn = (os.getpid(), 1) if dyn else map(int, args.shard.split("/"))
    for c0 in range(0, len(recs), args.chunk):
        pf = os.path.join(out, "parts_kin", f"{c0 // args.chunk:05d}.npz")
        claim = f"{pf}.claim.{args.shard[4:]}"
        if os.path.exists(pf) or (not dyn and (c0 // args.chunk) % sn != si):
            continue
        if dyn:
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        sub = recs[c0 : c0 + args.chunk]
        loader = DataLoader(Crops(args.root, args.pool, sub, cfg, ViTDetDataset, args.box, args.rescale),
                            batch_size=args.batch_size, num_workers=args.workers)
        n = len(sub)
        j3d, j3m, v3d = np.zeros((n, 21, 3), np.float32), np.zeros((n, 21, 3), np.float32), np.zeros((n, 778, 3), np.float16)
        rot, betas = np.zeros((n, 16, 3, 3), np.float32), np.zeros((n, 10), np.float32)
        err, ok = np.full(n, np.nan, np.float32), np.zeros(n, bool)
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            o = model(dict(img=b["img"].float(), right=b["right"], box_center=b["box_center"], box_size=b["box_size"]))
            v = o["pred_vertices"].float()  # (B, 778, 3), right hand, camera orientation
            jk = o["pred_keypoints_3d"].float()  # WiLoR's MANO wrapper: FK joints + smplx MANO tips, OpenPose order
            jm = torch.cat([torch.einsum("jv,bvc->bjc", J, v), v[:, MANO_TIP_VERTS]], 1)[:, MANO21_TO_OPENPOSE]
            mp = o["pred_mano_params"]
            uv = o["pred_keypoints_2d"].float() * b["box_size"][:, None, None] + b["box_center"][:, None]
            kp, lab = b["kp2d"], b["kp2d"][..., 2] > 0
            e = ((uv - kp[..., :2]).norm(dim=-1) * lab).sum(1) / lab.sum(1).clamp(min=1) / b["size"].clamp(min=1)
            e = torch.where(b["size"] > 0, e, torch.nan)
            i = b["idx"].cpu().numpy()
            j3d[i], j3m[i] = (jk - jk[:, :1]).cpu().numpy(), (jm - jm[:, :1]).cpu().numpy()
            v3d[i] = (v - jk[:, :1]).cpu().numpy().astype(np.float16)
            rot[i] = torch.cat([mp["global_orient"].reshape(-1, 1, 3, 3), mp["hand_pose"].reshape(-1, 15, 3, 3)], 1).float().cpu().numpy()
            betas[i] = mp["betas"].reshape(-1, 10).float().cpu().numpy()
            err[i], ok[i] = e.cpu().numpy(), b["ok"].cpu().numpy()
        np.savez(pf + f".tmp{si}.npz", files=np.array([r["file"] for r in sub]), j3d=j3d, j3d_mesh=j3m, v3d=v3d, err2d=err, ok=ok,
                 mano_rot=rot, betas=betas)
        os.replace(pf + f".tmp{si}.npz", pf)
        if dyn:
            os.remove(claim)
        print(f"[{args.pool}] {c0 + n}/{len(recs)}  ok {ok.mean():.3f}  err2d median {np.nanmedian(err):.3f}", flush=True)
    pfs = sorted(glob.glob(os.path.join(out, "parts_kin", "[0-9]" * 5 + ".npz")))  # not other processes' .tmp files
    if len(pfs) < -(-len(recs) // args.chunk):
        print(f"[{args.pool}] shard {args.shard} done, {len(pfs)} parts: merge after the other shards", flush=True)
        return
    parts = [np.load(f) for f in pfs]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("files", "j3d", "j3d_mesh", "err2d", "ok")}
    keep = cat["ok"] & np.isfinite(cat["j3d"]).all((1, 2)) & ~(cat["err2d"] > args.max_err2d)
    vf = os.path.join(out, "ext_kd_v3d.npy")
    keys = ["files", "j3d", "j3d_mesh", "err2d"]
    if args.v3d_npy:  # big pools: vertices as a separate array the loader memory-maps (same row order as files)
        tv = os.path.join(out, f"ext_kd_v3d.tmp{si}.npy")
        mm = np.lib.format.open_memmap(tv, mode="w+", dtype=np.float16, shape=(int(keep.sum()), 778, 3))
        n0, r0 = 0, 0
        for p in parts:
            k = keep[r0 : r0 + len(p["files"])]
            mm[n0 : n0 + k.sum()] = p["v3d"][k]
            n0, r0 = n0 + k.sum(), r0 + len(k)
        mm.flush()
        del mm
        os.replace(tv, vf)
    else:
        cat["v3d"] = np.concatenate([p["v3d"] for p in parts])
        keys.insert(3, "v3d")
        if os.path.exists(vf):  # a stale separate array would shadow v3d in the npz
            os.remove(vf)
    tmp = os.path.join(out, f"ext_kd.tmp{si}.npz")
    np.savez(tmp, **{k: cat[k][keep] for k in keys})
    os.replace(tmp, os.path.join(out, "ext_kd.npz"))
    e = cat["err2d"][np.isfinite(cat["err2d"])]
    print(json.dumps(dict(pool=args.name or args.pool, records=len(recs), written=int(keep.sum()), coverage=float(keep.sum() / max(len(recs), 1)),
                          unreadable=int((~cat["ok"]).sum()), dropped_err2d=int((cat["err2d"] > args.max_err2d).sum()),
                          err2d_labelled=int(len(e)), err2d_median=float(np.median(e)) if len(e) else None,
                          err2d_p90=float(np.percentile(e, 90)) if len(e) else None)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "crops"))
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="", help="output pool name (default --pool), e.g. a pool linking the same crops")
    p.add_argument("--wilor_repo", default=os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "WiLoR"))
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--box", default="label", choices=["label", "center"], help="label box mapped by M, or the teacher region")
    p.add_argument("--rescale", type=float, default=2.0, help="WiLoR crop side / box side (WiLoR's demo: 2.0)")
    p.add_argument("--max_err2d", type=float, default=float("inf"), help="drop crops whose WiLoR 2D disagrees with the label")
    p.add_argument("--v3d_npy", action="store_true", help="v3d as <pool>/ext_kd_v3d.npy (memory-mapped by the loader)")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--shard", default="0/1", help="i/n: this process does chunks i, i + n, ...; dyn:<tag>: claims free "
                   "chunks (a new tag per launch, so a preempted run's claims are ignored); the last one merges")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
