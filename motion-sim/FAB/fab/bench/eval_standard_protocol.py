"""The standard benchmark protocol for MHR-based and SMPL-based methods, on the samples every method shares
(SAM 3D Body paper / WHAM / TRAM convention; reproduces SAM 3D Body's published 3DPW PA 33.8 and EMDB-24 PA 38.2):
  * MHR outputs -> SMPL by the official MHR tool's iterative fit (bench/eval_fast3db_protocol.py: PyTorchSMPLFitting
    on the barycentric SMPL-topology mesh); SMPL outputs (NLF, our fitted SMPL) are scored as they are;
  * gendered SMPL ground truth (camera frame);
  * 3DPW: 14 H36M-regressor joints, hip-midpoint aligned (also 24 SMPL joints); EMDB: 24 SMPL joints, pelvis aligned
    (also 14); PVE after the hip-midpoint alignment.
Inputs: bench/eval_hmr.py outputs. Method spec name:kind:dir with kind
  mhr   <dir>/<dataset>_pred_shard*.npz from --save_pred (MHR mesh through the barycentric map), fitted here (cached)
  smpl  same files, the prediction already SMPL (no fit)
  rows  <dir>/<dataset>_shard*.npz metric rows of the fitted SMPL (*_fit keys; mean metrics only)
Writes <out_dir>/standard_<dataset>.json (means, per sequence, per joint) and standard.md.

    python -m bench.eval_standard_protocol --out_dir $FAB_HOME/bench/standard \
        --method fab-l:smpl:$FAB_HOME/bench/preds/fab-l --method sam3db:mhr:$FAB_HOME/bench/preds/sam3db --datasets 3dpw
"""

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bench.body_models import SMPL
from bench.eval_hmr import H36M_TO_J14, procrustes
from fab import paths

FOOT_V = {"l": (3387, 3216, 3226), "r": (6787, 6617, 6624)}  # SMPL heel, big toe, small toe vertices

from bench.eval_fast3db_protocol import MHR_TOOLS_DEFAULT as MHR_TOOLS  # noqa: E402
SMPL24 = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3", "l_foot",
          "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
          "r_wrist", "l_hand", "r_hand"]


def load_preds(d, ds):
    parts = [np.load(f) for f in sorted(glob.glob(f"{d}/{ds}_pred_shard*.npz"))]
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    v = lambda n: out[f"{n}_verts"].astype(np.float32) + out[f"{n}_root"][:, None]  # noqa: E731
    return out["ids"].astype(str), v("pred"), v("gt")


def official_fit(pred, bm_dir, batch=256):
    from bench.eval_fast3db_protocol import import_official_fitting, smplx_neutral
    import trimesh

    Fitting = import_official_fitting(MHR_TOOLS)
    body = smplx_neutral(bm_dir).cuda()
    for q in body.parameters():
        q.requires_grad_(False)
    edges = torch.from_numpy(trimesh.Trimesh(body.v_template.cpu().numpy(), body.faces, process=False)
                             .edges_unique.copy()).long().cuda()
    fitter = Fitting(smpl_model=body, smpl_edges=edges, smpl_model_type="smpl", hand_pose_dim=0, device="cuda",
                     batch_size=batch)
    out = []
    for s in range(0, len(pred), 16 * batch):
        prm = fitter.fit(target_vertices=torch.from_numpy(pred[s : s + 16 * batch]).float().cuda(),
                         single_identity=False, is_tracking=False)
        with torch.no_grad():
            out.append(body(betas=prm["betas"], body_pose=prm["body_pose"], global_orient=prm["global_orient"],
                            transl=prm["transl"]).vertices.cpu().numpy())
    return np.concatenate(out)


def fitted_cached(name, d, ds, ids, pred, out_dir, bm_dir, cache_dirs=()):
    """Official-fit vertices for an MHR method: <out_dir>/fit_<name>_<ds>.npz, else an earlier
    bench/eval_fast3db_protocol.py cache (protocol_<dirname>_<ds>.npz next to <dir> or in cache_dirs), else fit now."""
    cache = Path(out_dir) / f"fit_{name}_{ds}.npz"
    old = [Path(c) / f"protocol_{Path(d).name}_{ds}.npz" for c in (Path(d).parent, *cache_dirs)]
    probe = pred[:, ::500].astype(np.float32)  # a stale cache (same ids, new predictions) is refit
    for c in (cache, *old):
        if c.exists():
            z = np.load(c)
            if "fitted" in z.files and np.array_equal(z["ids"].astype(str), ids) and (
                    "probe" not in z.files or np.allclose(z["probe"], probe, atol=1e-3)):
                return z["fitted"].astype(np.float32) + z["root"]
    t0 = time.time()
    v = official_fit(pred, bm_dir)
    root = pred.mean(1, keepdims=True)
    np.savez(cache, ids=ids, root=root.astype(np.float32), fitted=(v - root).astype(np.float16), probe=probe)
    print(f"  fitted {name} {ds}: {len(v)} in {time.time() - t0:.0f} s", flush=True)
    return v


def per_sample(pred, gt, j14, j24):
    """(N,) metrics dict and (N, 24) per-joint errors (PA-aligned on 24 joints and pelvis-aligned)."""
    m = defaultdict(list)
    pj_pa, pj_mp = [], []
    for p, g in zip(pred.astype(np.float64), gt.astype(np.float64)):
        a14, b14 = j14 @ p, j14 @ g
        ha, hb = a14[[2, 3]].mean(0), b14[[2, 3]].mean(0)
        a14, b14 = a14 - ha, b14 - hb
        a24, b24 = j24 @ p, j24 @ g
        a24, b24 = a24 - a24[:1], b24 - b24[:1]
        e24_pa = np.linalg.norm(procrustes(a24, b24) - b24, axis=1) * 1000
        e24_mp = np.linalg.norm(a24 - b24, axis=1) * 1000
        m["pa14"].append(np.linalg.norm(procrustes(a14, b14) - b14, axis=1).mean() * 1000)
        m["mpjpe14"].append(np.linalg.norm(a14 - b14, axis=1).mean() * 1000)
        m["pa24"].append(e24_pa.mean())
        m["mpjpe24"].append(e24_mp.mean())
        m["pve"].append(np.linalg.norm((p - ha) - (g - hb), axis=1).mean() * 1000)
        # feet (the ground-truth feet are unverified on most sets): heel -> toe-midpoint direction
        # and sole normal (heel, big toe, small toe) angles after the body's 24-joint PA rotation, mean of both feet, deg
        u, _, vt = np.linalg.svd((a24 - a24.mean(0)).T @ (b24 - b24.mean(0)))
        z = np.eye(3)
        z[-1, -1] = np.sign(np.linalg.det(u @ vt))
        R = vt.T @ z @ u.T
        fd, fs = [], []
        for h, bt, st in FOOT_V.values():
            dp, dg = (p[bt] + p[st]) / 2 - p[h], (g[bt] + g[st]) / 2 - g[h]
            npv, ngv = np.cross(p[bt] - p[h], p[st] - p[h]), np.cross(g[bt] - g[h], g[st] - g[h])
            for x, y, acc in ((dp, dg, fd), (npv, ngv, fs)):
                c = (R @ x) @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)
                acc.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        m["foot_dir"].append(float(np.mean(fd)))
        m["foot_sole"].append(float(np.mean(fs)))
        pj_pa.append(e24_pa)
        pj_mp.append(e24_mp)
    return {k: np.array(v) for k, v in m.items()}, np.array(pj_pa), np.array(pj_mp)


def seq_of(i, ds):
    parts = i.split("/")
    return "/".join(parts[:2]) if ds.startswith("emdb") else parts[0]  # 3DPW / pool-unit / RICH / Harmony4D ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--method", action="append", default=[], help="name:kind:dir (kind mhr | smpl | rows)")
    p.add_argument("--datasets", nargs="+", default=["3dpw", "emdb1"])
    p.add_argument("--fit_cache", nargs="*", default=[], help="dirs with earlier eval_fast3db_protocol.py fits")
    p.add_argument("--per_sample", action="store_true", help="also write <out_dir>/samples_<method>_<dataset>.npz")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j14 = np.load(Path(args.bm_dir) / "smpl" / "J_regressor_h36m.npy")[H36M_TO_J14].astype(np.float64)
    j24 = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").J_regressor.numpy().astype(np.float64)
    specs = [m.split(":", 2) for m in args.method]
    md = ["| method | " + " | ".join(f"{ds} PA14 / MPJPE14 / PA24 / MPJPE24 / PVE" for ds in args.datasets) + " |",
          "|---|" + "---|" * len(args.datasets)]
    table = defaultdict(dict)
    for ds in args.datasets:
        data = {}
        for name, kind, d in specs:
            if kind == "rows":
                parts = [np.load(f) for f in sorted(glob.glob(f"{d}/{ds}_shard*.npz"))]
                if not parts:
                    continue
                ids = np.concatenate([q["ids"] for q in parts]).astype(str)
                key = {"pa14": "pa_mpjpe_fit", "mpjpe14": "mpjpe_fit", "pa24": "pa_mpjpe24_fit",
                       "mpjpe24": "mpjpe24_fit", "pve": "pve_fit"}
                if "pa_mpjpe_fit" not in parts[0].files:  # eval_hmr --predictor rows: the fitted SMPL under the plain keys
                    key = {k: v[: -len("_fit")] for k, v in key.items()}
                data[name] = (ids, {k: np.concatenate([q[v] for q in parts]) for k, v in key.items()}, None, None)
                continue
            if kind == "dense":  # the dense head's own 14 joints (eval_hmr *_dense rows; no 24 joints / PVE)
                parts = [np.load(f) for f in sorted(glob.glob(f"{d}/{ds}_shard*.npz"))]
                if not parts or "pa_mpjpe_dense" not in parts[0].files:
                    continue
                ids = np.concatenate([q["ids"] for q in parts]).astype(str)
                n = len(ids)
                m = {"pa14": np.concatenate([q["pa_mpjpe_dense"] for q in parts]),
                     "mpjpe14": np.concatenate([q["mpjpe_dense"] for q in parts])}
                m.update({k: np.full(n, np.nan) for k in ("pa24", "mpjpe24", "pve")})
                data[name] = (ids, m, None, None)
                continue
            got = load_preds(d, ds)
            if got is None:
                continue
            ids, pred, gt = got
            if kind == "mhr":
                pred = fitted_cached(name, d, ds, ids, pred, out_dir, args.bm_dir, args.fit_cache)
            m, pj_pa, pj_mp = per_sample(pred, gt, j14, j24)
            data[name] = (ids, m, pj_pa, pj_mp)
        common = sorted(set.intersection(*[set(v[0].tolist()) for v in data.values()]))
        res = {"n": len(common), "methods": {}}
        for name, (ids, m, pj_pa, pj_mp) in data.items():
            pos = {k: i for i, k in enumerate(ids.tolist())}
            ix = np.array([pos[k] for k in common])
            entry = {"mean": {k: round(float(np.nanmean(v[ix])), 2) for k, v in m.items()}}
            seqs = defaultdict(list)
            for k, i in zip(common, ix):
                seqs[seq_of(k, ds)].append(i)
            entry["per_sequence"] = {s: {"n": len(v), "pa14": round(float(np.nanmean(m["pa14"][v])), 2),
                                         "pa24": round(float(np.nanmean(m["pa24"][v])), 2),
                                         "mpjpe24": round(float(np.nanmean(m["mpjpe24"][v])), 2)}
                                     for s, v in sorted(seqs.items())}
            if pj_pa is not None:
                entry["per_joint_pa24"] = dict(zip(SMPL24, np.round(pj_pa[ix].mean(0), 2).tolist()))
                entry["per_joint_mpjpe24"] = dict(zip(SMPL24, np.round(pj_mp[ix].mean(0), 2).tolist()))
            res["methods"][name] = entry
            table[name][ds] = entry["mean"]
            if args.per_sample:  # per-sample metrics on the common ids (subset tables, e.g. FAB-Hard difficulty)
                np.savez(out_dir / f"samples_{name}_{ds}.npz", ids=np.array(common), **{k: v[ix] for k, v in m.items()})
        (out_dir / f"standard_{ds}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"{ds}: {len(common)} common samples", flush=True)
    for name, _, _ in specs:
        cells = []
        for ds in args.datasets:
            m = table[name].get(ds)
            cells.append(" / ".join(f"{m[k]:.1f}" for k in ("pa14", "mpjpe14", "pa24", "mpjpe24", "pve")) if m else "--")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    (out_dir / "standard.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
