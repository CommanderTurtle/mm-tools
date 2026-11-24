"""SSP-3D (Sengupta et al. BMVC 2020; github.com/akashsengupta1997/SSP-3D, MIT): body shape of 311 sportspeople in tight
clothing. Metric PVE-T-SC (the dataset's metrics.py): per-vertex error of the T-posed bodies after scale correction (the
prediction centred, scaled to unit RMS radius, then mapped to the target's centre and RMS radius), mm.

  build  labels.npz + images -> <out>/index_joints.pkl ({id, image, bbox (square, from the label's centre / size), K
         (f = 5000 on the 512 px crops, as the labels' camera), gt_betas, gt_gender}) and <out>/index.pkl (the same people
         with the label SMPL, for bench/eval_hmr.py --dataset ssp3d).
  score  predictions (bench/eval_hmr.py --save_pred: SMPL-topology meshes) -> SMPL shape by the official MHR tool's fit (for every
         method, NLF included, so every shape comes from the same neutral SMPL fit) -> neutral T-pose vs the GT gendered
         T-pose -> PVE-T-SC; also the GT-neutral variant.

    python -m bench.eval_ssp3d build --src <SSP-3D dir with labels.npz, images/> --out $FAB_DATA/ssp3d
    python -m bench.eval_ssp3d score --index $FAB_DATA/ssp3d/index_joints.pkl --method fab-l:$FAB_HOME/bench/preds/fab-l --out <json>
"""

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from fab import paths


def build(a):
    z = np.load(Path(a.src) / "labels.npz")
    print({k: z[k].shape for k in z.files})
    index = []
    for i, fn in enumerate(z["fnames"]):
        c, s = np.asarray(z["bbox_centres"][i], np.float32), float(np.asarray(z["bbox_whs"][i]).max())
        index.append({"id": f"ssp3d/{Path(str(fn)).stem}", "image": str(Path(a.src) / "images" / str(fn)),
                      "bbox": np.array([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2], np.float32),
                      "K": np.array([[5000.0, 0, 256.0], [0, 5000.0, 256.0], [0, 0, 1]], np.float32),
                      "gt_betas": np.asarray(z["shapes"][i], np.float32)[:10], "gt_gender": str(z["genders"][i]),
                      "source": "ssp3d", "licence": "SSP-3D: MIT (GitHub repository)"})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "index_joints.pkl").write_bytes(pickle.dumps(index))
    # the same people with the label SMPL (pose, shape, camera translation of the f = 5000 crop camera) for
    # bench/eval_hmr.py (dataset "ssp3d"): its --save_fit gives the dense-point SMPL fit for the points route
    g = {"m": "male", "f": "female"}
    wds = [dict(s, gender=g.get(s["gt_gender"][:1].lower(), "neutral"),
                global_orient=np.asarray(z["poses"][i][:3], np.float32), body_pose=np.asarray(z["poses"][i][3:72], np.float32),
                betas=s["gt_betas"], transl=np.asarray(z["cam_trans"][i], np.float32), extrinsic=np.eye(4, dtype=np.float32))
           for i, s in enumerate(index)]
    (Path(a.out) / "index.pkl").write_bytes(pickle.dumps(wds))
    print(len(index), "people ->", a.out)


def fit_betas(verts, bm_dir, batch=256):
    """Neutral SMPL betas of SMPL-topology meshes by the official MHR -> SMPL fit (bench/eval_standard_protocol)."""
    import trimesh

    from bench.eval_fast3db_protocol import import_official_fitting, smplx_neutral
    from bench.eval_standard_protocol import MHR_TOOLS

    Fitting = import_official_fitting(MHR_TOOLS)
    body = smplx_neutral(bm_dir).cuda()
    for q in body.parameters():
        q.requires_grad_(False)
    edges = torch.from_numpy(trimesh.Trimesh(body.v_template.cpu().numpy(), body.faces, process=False)
                             .edges_unique.copy()).long().cuda()
    fitter = Fitting(smpl_model=body, smpl_edges=edges, smpl_model_type="smpl", hand_pose_dim=0, device="cuda",
                     batch_size=batch)
    out = []
    for s in range(0, len(verts), 16 * batch):
        prm = fitter.fit(target_vertices=torch.from_numpy(verts[s : s + 16 * batch]).float().cuda(), single_identity=False,
                         is_tracking=False)
        out.append(prm["betas"].detach().cpu().numpy()[:, :10])
    return np.concatenate(out)


def tpose(body, betas):
    t = torch.from_numpy(np.asarray(betas, np.float32))
    n = len(t)
    with torch.no_grad():
        v, _ = body(torch.zeros(n, 3), torch.zeros(n, 69), t, torch.zeros(n, 3))
    return v.numpy().astype(np.float64)


def scale_corrected(pred, gt):
    pc, gc = pred - pred.mean(1, keepdims=True), gt - gt.mean(1, keepdims=True)
    ps = np.sqrt((pc ** 2).sum(-1).mean(-1))[:, None, None]
    gs = np.sqrt((gc ** 2).sum(-1).mean(-1))[:, None, None]
    return pc / ps * gs + gt.mean(1, keepdims=True)


def score(a):
    from bench.body_models import SMPL

    gt = {s["id"]: s for s in pickle.loads(Path(a.index).read_bytes())}
    bm = Path(a.bm_dir)
    smpl = {g: SMPL(bm / "smpl" / f"SMPL_{g.upper()}.pkl").eval() for g in ("neutral", "male", "female")}
    res = {}
    for spec in a.method:
        name, d = spec.split(":", 1)
        src = "pred"
        if name.endswith("@fit"):  # eval_hmr --save_fit preds: the saved SMPL fit
            name, src = name[: -len("@fit")], "fit"
        parts = [np.load(f) for f in sorted(glob.glob(f"{d}/{a.name}_joints_shard*.npz"))] or \
            [np.load(f) for f in sorted(glob.glob(f"{d}/{a.name}_pred_shard*.npz"))]
        if not parts:
            continue
        ids = np.concatenate([q["ids"] for q in parts]).astype(str)
        v = np.concatenate([q[f"{src}_verts"] for q in parts]).astype(np.float32) + \
            np.concatenate([q[f"{src}_root"] for q in parts])[:, None]
        name = name if src == "pred" else f"{name}-fit"
        cache = Path(a.out).with_name(f"ssp3d_betas_{name}.npz")
        if cache.exists() and np.array_equal(np.load(cache)["ids"].astype(str), ids):
            betas = np.load(cache)["betas"]
        else:
            betas = fit_betas(v, a.bm_dir)
            np.savez(cache, ids=ids, betas=betas)
        pred_t = tpose(smpl["neutral"], betas)
        g_gender = np.array([{"m": "male", "f": "female"}.get(gt[i]["gt_gender"][:1].lower(), "neutral") for i in ids])
        g_betas = np.stack([gt[i]["gt_betas"] for i in ids])
        gt_t = np.zeros_like(pred_t)
        for g in ("male", "female", "neutral"):
            k = g_gender == g
            if k.any():
                gt_t[k] = tpose(smpl[g], g_betas[k])
        gt_n = tpose(smpl["neutral"], g_betas)
        e = np.linalg.norm(scale_corrected(pred_t, gt_t) - gt_t, axis=-1).mean(-1) * 1000
        en = np.linalg.norm(scale_corrected(pred_t, gt_n) - gt_n, axis=-1).mean(-1) * 1000
        res[name] = {"n": len(ids), "pve_t_sc": round(float(e.mean()), 2), "pve_t_sc_gt_neutral": round(float(en.mean()), 2)}
        print(name, res[name], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["build", "score"])
    p.add_argument("--src", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--index", default="")
    p.add_argument("--name", default="ssp3d")
    p.add_argument("--method", action="append", default=[],
                   help="score: name:dir (eval_hmr --save_pred files; name@fit = the saved SMPL fit)")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    a = p.parse_args()
    {"build": build, "score": score}[a.step](a)


if __name__ == "__main__":
    main()
