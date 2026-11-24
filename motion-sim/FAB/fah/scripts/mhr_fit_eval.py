"""MHR-Hand fit (fah/mhr_fit.py) vs the MANO fit (fah/fast_mano.py MeshStage) on stored bench.run predictions:
every fit to the same dense surface (verts3d_dense) + 3D joints (joints3d) of every GT-matched hand, then placed onto
the 2D joints (joints2d_head: the heatmap 2D before the fingertip step) by PnP as in bench/methods/fah.py. Metrics
(mm; PCK in [0, 1]):
  res_v / res_j     distance of the fitted MANO-topology mesh / joints to FAH's dense surface / joints (before placement)
  pa_mpvpe          MANO-topology mesh vs GT mesh (the MHR mesh through its MHR -> MANO-778 regressor)
  pa_mpjpe, mpjpe   the fit's own 21 joints vs GT joints, Procrustes-aligned / root-relative. MANO: the fitted MANO's
                    kinematic joints + tip vertices; MHR "rig": the rig's skeleton joints at SAM 3D Body's 21 right-hand
                    keypoints (MHRHead.keypoint_mapping[21:42], OpenPose order); MHR "reg": the MHR -> MANO21
                    kinematic regressor on the MHR mesh
  pa_mpjpe_v        MANO regressor joints of the MANO-topology mesh (bench.evaluate pa_mpjpe_v)
  pck05             2D reprojection of those joints after the PnP placement vs GT 2D, PCK@0.05 of the GT box's long side
The FAH row is the input itself (dense surface, its 3D joints, its 2D joints). --targets gt: every fit to the GT mesh /
joints instead (representation floor).

    python scripts/mhr_fit_eval.py --runs $R/hot3d/fah-image,$R/dexycb/fah-image --out eval.json   # the shipped fits
    python scripts/mhr_fit_eval.py --runs ... --targets gt --every_clip 2 --every_frame 3 \
        --mhr '[{"name": "mhr", "iters": 16, "fit_args": {"adaptive": true, "lam0": 1e-4}}]'           # representation floors
    python scripts/mhr_fit_eval.py --online $R/hot3d/fah-image-both   # bench.evaluate of a run's MANO outputs and of its MHR fit
Defaults: the shipped fits (MANO: --lmf <weights>/shared/mesh_fitter.pt and --mano; MHR-Hand: one step from
<weights>/shared/mhr_fitter.pt).
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.common import VertexJoints, match_hands, procrustes  # noqa: E402


def collect(run, every_clip, every_frame, vidx=None):
    """GT-matched predictions with a dense surface (camera frame). vidx: a dense-subset model's vertices, whose rows
    are replaced by its raw predicted points (surface3d_head after the bone points)."""
    keys = ("tV", "tJ", "uv", "right", "K", "gV", "gJ", "g2", "gbox")
    d = {k: [] for k in keys}
    for f in sorted(glob.glob(os.path.join(run, "*.pkl")))[::every_clip]:
        clip = pickle.load(open(f, "rb"))
        for fr in clip["frames"][::every_frame]:
            for gi, pi in match_hands(fr["pred"], fr["gt"])[0]:
                if pi is None:
                    continue
                p, g = fr["pred"][pi], fr["gt"][gi]
                if p.get("verts3d_dense") is None or g.get("verts3d") is None or not np.isfinite(p["verts3d_dense"]).all():
                    continue
                v = np.asarray(p["verts3d_dense"], np.float64).copy()
                if vidx is not None:
                    v[vidx] = np.asarray(p["surface3d_head"], np.float64)[-len(vidx):]
                for k, x in zip(keys, (v, p["joints3d"], p.get("joints2d_head", p["joints2d"]), bool(p["is_right"]), fr["K"], g["verts3d"],
                                       g["joints3d"], g["joints2d"], g["bbox"])):
                    d[k].append(np.asarray(x, np.float64) if k != "right" else x)
    return {k: np.stack(v) for k, v in d.items()}


def project(X, K):
    return (X[..., :2] / np.clip(X[..., 2:], 1e-6, None)) @ K[:, :2, :2].transpose(0, 2, 1) + K[:, None, :2, 2]


def scores(V, J, d, vj, Jreg=None, placed_J=None, placed_Jreg=None):
    """Means over hands (mm / PCK) of a fit: V (N, 778, 3) MANO-topology mesh, J (N, 21, 3) its own joints (camera frame,
    as fitted); Jreg: a second joint set (MHR: the regressor joints); placed_*: the joints after the 2D placement (PCK)."""
    pa = lambda A, G: np.array([np.linalg.norm(procrustes(a, g) - g, axis=1).mean() for a, g in zip(A, G)])
    rr = lambda A, G: np.linalg.norm((A - A[:, :1]) - (G - G[:, :1]), axis=-1).mean(1)
    S = np.maximum(np.maximum(d["gbox"][:, 2] - d["gbox"][:, 0], d["gbox"][:, 3] - d["gbox"][:, 1]), 1.0)
    pck = lambda A: float((np.linalg.norm(project(A, d["K"]) - d["g2"], axis=-1) / S[:, None] < 0.05).mean())
    gv = np.stack([vj(g) for g in d["gV"]])
    r = dict(res_v=np.linalg.norm(V - d["tV"], axis=-1).mean(1).mean() * 1000, res_j=np.linalg.norm(J - d["tJ"], axis=-1).mean(1).mean() * 1000,
             pa_mpvpe=pa(V, d["gV"]).mean() * 1000, pa_mpjpe=pa(J, d["gJ"]).mean() * 1000, mpjpe=rr(J, d["gJ"]).mean() * 1000,
             pa_mpjpe_v=pa(np.stack([vj(v) for v in V]), gv).mean() * 1000, pck05=pck(J if placed_J is None else placed_J))
    if Jreg is not None:
        r.update(res_j_reg=np.linalg.norm(Jreg - d["tJ"], axis=-1).mean(1).mean() * 1000, pa_mpjpe_reg=pa(Jreg, d["gJ"]).mean() * 1000,
                 mpjpe_reg=rr(Jreg, d["gJ"]).mean() * 1000, pck05_reg=pck(Jreg if placed_Jreg is None else placed_Jreg))
    return {k: round(float(v), 4) for k, v in r.items()}


def placed(d, Jfit, arrays):
    """PnP of each hand's fitted joints onto its 2D joints (fah.mhr_fit.place_2d); the same rigid motion on arrays."""
    from fah.mhr_fit import place_2d

    out = [a.copy() for a in arrays]
    for k in range(len(Jfit)):
        pl = place_2d(Jfit[k], d["uv"][k], d["K"][k])
        if pl is not None:
            R, t, c = pl
            for a in out:
                a[k] = (a[k] - c) @ R.T + t
    return out


def fit_mano(d, stage, vj):
    M = np.where(d["right"][:, None], 1.0, -1.0) * [1.0, 0, 0] + [0, 1.0, 1.0]
    P, J = d["tV"] * M[:, None], d["tJ"] * M[:, None]
    mv, mj = [], []
    for i in range(0, len(P), 2):
        mv.append(stage(P[i:i + 2], J[i:i + 2]))
        fs = stage.graphs[len(mv[-1])][0]["fs"]  # the fit's parameters -> its kinematic joints (what it fits to tJ)
        with torch.no_grad():
            _, j = stage.fitter.mano(fs["out_x"][:, :45], fs["out_x"][:, 45:55])
            c = J[i:i + 2, :1] if stage.centre else 0.0  # a centred fit's parameters are relative to the target wrist
            mj.append((j @ fs["out_R"].mT + fs["out_x"][:, None, 58:61]).cpu().numpy() + c)
    V, Jk = np.concatenate(mv) * M[:, None], np.concatenate(mj) * M[:, None]
    Jmesh = np.stack([vj(v) for v in V])  # the mesh stage places by its mesh's regressor joints + tips (_align_2d)
    return V, Jk, placed(d, Jmesh, [Jk])[0]


def fit_mhr(d, fitter):
    M = np.where(d["right"][:, None], 1.0, -1.0) * [1.0, 0, 0] + [0, 1.0, 1.0]
    o = fitter(d["tV"] * M[:, None], d["tJ"] * M[:, None])
    V, Jr, Jg = (o[k] * M[:, None] for k in ("verts_mano", "joints_rig", "joints"))
    return (V, Jr, Jg), placed(d, Jg, [Jr, Jg])  # placed by the regressor joints, as bench/methods/fah.py


def load_fitter(spec, checkpoint, device, chunk):
    """{"asset": cached build_asset file (built from the checkpoint with "vreg" when missing), "iters", "init", "fit_args"}."""
    from fah.mhr_fit import MHRHandFitter, build_asset

    if spec.get("asset") and os.path.exists(spec["asset"]):
        a = torch.load(spec["asset"], map_location="cpu", weights_only=False)
    else:
        if checkpoint:
            from fah import load_hand_model

            head = load_hand_model(checkpoint, device="cpu", verbose=False)[0].head_pose_hand
        else:  # the released FAH-L model's MHR head (SAM 3D Body's model config + the MHR model: fah.download_sam3d)
            from fah.api import FAH

            head = FAH("fah", "image", mesh="mhr", device="cpu").method.est.model.head_pose_hand
        a = build_asset(head, **({"vreg": spec["vreg"]} if spec.get("vreg") else {}))
        if spec.get("asset"):
            torch.save(a, spec["asset"])
    return MHRHandFitter(a, device, iters=spec.get("iters", 2), init_net=spec.get("init") or None, graph=False, compile=False, chunk=chunk,
                         **spec.get("fit_args", {}))


def online(run, mano_dir):
    """bench.evaluate metrics of a bench.run directory with mhr_fit outputs: as stored (MANO fit) and with
    verts3d / joints3d = verts3d_mhr_mano / joints3d_mhr_fit."""
    import tempfile

    from bench.evaluate import evaluate_dir

    res = {"mano": evaluate_dir(run, mano_dir=mano_dir)}
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted(glob.glob(os.path.join(run, "*.pkl"))):
            d = pickle.load(open(f, "rb"))
            for fr in d["frames"]:
                fr["pred"] = [dict(p, verts3d=p.get("verts3d_mhr_mano"), joints3d=p.get("joints3d_mhr_fit", p["joints3d"])) for p in fr["pred"]]
            pickle.dump(d, open(os.path.join(tmp, os.path.basename(f)), "wb"))
        res["mhr"] = evaluate_dir(tmp, mano_dir=mano_dir)
    keys = ("pa_mpjpe", "pa_mpjpe_v", "pa_mpvpe", "pa_mpvpe_p", "f5", "f15", "num_matched", "latency_ms_median")
    for k in res:
        print(run, k, {m: round(res[k][m], 3) for m in keys if res[k].get(m) is not None}, flush=True)
    return res


def main(args):
    if args.online:
        res = {r: online(r, args.mano_dir) for r in args.online.split(",")}
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1)
        return
    from fah.fast_mano import MeshStage

    torch.backends.cuda.matmul.allow_tf32 = False
    vj = VertexJoints(args.mano_dir)
    from fah.api import DEFAULT_WEIGHTS

    mano = dict(json.loads(args.mano), **({"iters": args.mano_iters} if args.mano_iters is not None else {}))
    lmf = os.path.join(DEFAULT_WEIGHTS, "shared", "mesh_fitter.pt") if args.lmf is None else args.lmf
    stage = MeshStage(args.mano_dir, args.device, lmf=lmf or None, **mano)
    specs = json.loads(args.mhr) if args.mhr else [{"name": "mhr", "iters": 1, "init": os.path.join(DEFAULT_WEIGHTS, "shared", "mhr_fitter.pt")}]
    fitters = {s["name"]: load_fitter(s, args.checkpoint, args.device, args.chunk) for s in specs}
    res = {}
    for run in args.runs.split(","):
        name = "/".join(run.rstrip("/").split("/")[-2:])
        d = collect(run, args.every_clip, args.every_frame, np.load(args.vidx) if args.vidx else None)
        if args.targets == "gt":  # representation floor: every fit to the GT mesh / joints
            d["tV"], d["tJ"] = d["gV"].copy(), d["gJ"].copy()
        out = {"n": int(len(d["tV"]))}
        print(f"{name}: {out['n']} hands", flush=True)
        out["fah"] = scores(d["tV"], d["tJ"], d, vj)
        out["fah"]["pck05"] = float((np.linalg.norm(d["uv"] - d["g2"], axis=-1) / np.maximum(np.maximum(d["gbox"][:, 2] - d["gbox"][:, 0],
                                                                                                  d["gbox"][:, 3] - d["gbox"][:, 1]), 1.0)[:, None] < 0.05).mean())
        t0 = time.time()
        V, Jk, Jp = fit_mano(d, stage, vj)
        out["mano"] = dict(scores(V, Jk, d, vj, placed_J=Jp), sec=round(time.time() - t0, 1))
        print(" fah", out["fah"], "\n mano", out["mano"], flush=True)
        for k, f in fitters.items():
            t0 = time.time()
            (V, Jr, Jg), (Jrp, Jgp) = fit_mhr(d, f)
            out[k] = dict(scores(V, Jr, d, vj, Jreg=Jg, placed_J=Jrp, placed_Jreg=Jgp), sec=round(time.time() - t0, 1))
            print(f" {k}", out[k], flush=True)
        res[name] = out
    if args.out:
        json.dump(dict(args=vars(args), results=res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--online", default="", help="bench.run dirs with mhr_fit outputs (comma-separated)")
    ap.add_argument("--mhr", default="", help="MHR fits to compare (json list, load_fitter; default: the shipped fit)")
    ap.add_argument("--checkpoint", default="", help="a SAM 3D Body / FAH checkpoint whose MHR head builds the fitter asset "
                    "(default: the released FAH-L model)")
    ap.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    ap.add_argument("--lmf", default=None, help="learned MANO fitter (default: <weights>/shared/mesh_fitter.pt; '' = none)")
    ap.add_argument("--mano", default='{"iters": 1, "wv": 0.3, "jac": "fd", "centre": true, "fd_eps": 0.01}',
                    help="MANO stage (fah.fast_mano.MeshStage keyword arguments; default: the shipped fit)")
    ap.add_argument("--mano_iters", type=int, default=None, help="overrides --mano's iters")
    ap.add_argument("--every_clip", type=int, default=1)
    ap.add_argument("--every_frame", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--targets", default="dense", choices=["dense", "gt"])
    ap.add_argument("--vidx", default="", help=".npy of a dense-subset model's vertices: their raw predicted points as targets")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    main(ap.parse_args())
