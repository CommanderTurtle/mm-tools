"""MANO fits of the EgoStandard tracked keypoints ("MANO-fit targets"): per clip and hand one
shape, per frame global orientation / articulation / translation (full 45-D pose), fitted in the world frame to the
21 keypoints (MANO definition: 16 joints + HOT3D tip vertices, OpenPose order). Writes <clip>/mano.npz with world
joints / vertices (NaN where the hand has no keypoints) and the per-frame fit residual (mm), which also measures
how far the tracker skeleton is from MANO's. EgoStandardClip(targets="mano") / EGOSTD_TARGETS=mano uses them
(joints, mesh, projected-mesh boxes and visibility, as for HOT3D).

    python -m bench.datasets.egostandard_fit --root $DATA_ROOT/egostandard --mano_dir $MANO_DIR [--device cuda]
Check on HOT3D GT MANO joints (clip-001881, CPU): joint residual 0.5 mm, mesh PVE 3.1-3.4 mm (21 joints leave
shape / twist ambiguous) -> use the fitted mesh for boxes / visibility, not as mesh GT.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from bench.common import MANO21_TO_OPENPOSE, MANO_TIP_VERTS

PALM = [0, 5, 9, 13, 17]  # wrist + finger MCPs (OpenPose order)


def kabsch(src, dst):
    """Rotation R (B, 3, 3) minimizing ||R src - dst|| for centred point sets (B, N, 3)."""
    U, _, Vt = torch.linalg.svd(src.transpose(1, 2) @ dst)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return Vt.transpose(1, 2) @ D @ U.transpose(1, 2)


def matrix_to_axis_angle(R):
    cos = ((R.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return axis / (2 * torch.sin(ang)[:, None]) * ang[:, None]


def mano21(layer, betas, orient, pose, transl):
    out = layer(betas=betas, global_orient=orient, hand_pose=pose, transl=transl)
    j = torch.cat([out.joints[:, :16], out.vertices[:, MANO_TIP_VERTS]], 1)[:, MANO21_TO_OPENPOSE]
    return j, out.vertices


def fit_hand(layer, target, steps=(150, 800), w_pose=1e-2, w_shape=1e-1, w_smooth=1e-1, device="cpu"):
    """target (F, 21, 3) metres -> fitted joints (F, 21, 3), vertices (F, 778, 3), residual (F,) mm, params."""
    T = torch.as_tensor(target, dtype=torch.float32, device=device)
    F = len(T)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    pose = torch.zeros(F, 45, device=device, requires_grad=True)
    with torch.no_grad():
        j0, _ = mano21(layer, betas.expand(F, -1), torch.zeros(F, 3, device=device), pose, torch.zeros(F, 3, device=device))
        R = kabsch(j0[:, PALM] - j0[:, :1], T[:, PALM] - T[:, :1])
        orient0 = matrix_to_axis_angle(R)
        transl0 = T[:, 0] - j0[:, 0]  # smplx rotates about the root joint, so the wrist only moves by transl
    orient = orient0.clone().requires_grad_(True)
    transl = transl0.clone().requires_grad_(True)
    for stage, n in enumerate(steps):
        params = [orient, transl] if stage == 0 else [orient, transl, pose, betas]
        opt = torch.optim.Adam(params, lr=0.02 if stage == 0 else 0.01)
        for _ in range(n):
            j, _ = mano21(layer, betas.expand(F, -1), orient, pose, transl)
            loss = (((j - T) * 1000) ** 2).sum(-1).mean()  # mm^2
            if stage:
                loss = loss + w_pose * (pose ** 2).mean() + w_shape * (betas ** 2).mean()
                if F > 2:
                    loss = loss + w_smooth * ((pose[2:] - 2 * pose[1:-1] + pose[:-2]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        j, v = mano21(layer, betas.expand(F, -1), orient, pose, transl)
        res = (j - T).norm(dim=-1).mean(-1) * 1000
    params = dict(betas=betas.detach().cpu().numpy()[0], global_orient=orient.detach().cpu().numpy(),
                  hand_pose=pose.detach().cpu().numpy(), transl=transl.detach().cpu().numpy())
    return j.cpu().numpy(), v.cpu().numpy(), res.cpu().numpy(), params


def fit_clip(clip_dir, layers, device="cpu"):
    gt = dict(np.load(os.path.join(clip_dir, "gt.npz")))
    out, stats = {}, {}
    for side, is_right in (("left", False), ("right", True)):
        J = gt[f"joints_{side}"]
        ok = np.isfinite(J).all((1, 2))
        n = len(J)
        jf, vf, res = np.full((n, 21, 3), np.nan, np.float32), np.full((n, 778, 3), np.nan, np.float32), np.full(n, np.nan, np.float32)
        if ok.sum():
            j, v, r, p = fit_hand(layers[is_right], J[ok], device=device)
            jf[ok], vf[ok], res[ok] = j, v, r
            out[f"betas_{side}"] = p["betas"]
            stats[side] = dict(frames=int(ok.sum()), res_mean=float(r.mean()), res_p95=float(np.percentile(r, 95)))
        out[f"joints_{side}"], out[f"verts_{side}"], out[f"res_{side}"] = jf, vf, res
    np.savez_compressed(os.path.join(clip_dir, "mano.npz"), **out)
    return stats


def main(args):
    import smplx

    layers = {r: smplx.create(f"{args.mano_dir}/MANO_{'RIGHT' if r else 'LEFT'}.pkl", "mano", is_rhand=r, use_pca=False,
                              flat_hand_mean=True).to(args.device) for r in (True, False)}
    l, r = layers[False], layers[True]
    if torch.sum(torch.abs(l.shapedirs[:, 0, :] - r.shapedirs[:, 0, :])) < 1:  # smplx left-hand shapedirs bug
        l.shapedirs[:, 0, :] *= -1
    clips = sorted(glob.glob(os.path.join(args.root, "clips", "*", "gt.npz")))[args.shard :: args.num_shards]
    allstats = {}
    for k, g in enumerate(clips):
        d = os.path.dirname(g)
        if os.path.exists(os.path.join(d, "mano.npz")) and not args.overwrite:
            continue
        allstats[os.path.basename(d)] = s = fit_clip(d, layers, args.device)
        print(f"[{k + 1}/{len(clips)}] {os.path.basename(d)}: " + ", ".join(f"{a} {b['frames']} fr res {b['res_mean']:.1f} mm "
              f"(p95 {b['res_p95']:.1f})" for a, b in s.items()), flush=True)
    if allstats:
        json.dump(allstats, open(os.path.join(args.root, f"mano_fit_stats_{os.getpid()}.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
