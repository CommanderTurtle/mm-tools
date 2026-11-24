"""Fast SAM 3D Body's evaluation protocol on saved predictions (bench/eval_hmr.py --save_pred): the MHR mesh is
converted to SMPL by the official MHR tool's iterative fit (tools/mhr_smpl_conversion, PyTorchSMPLFitting: Adam on
vertex + edge losses, neutral SMPL, per frame) instead of being scored through the barycentric map alone, then scored
with the same joints as ours (14 H36M-regressor joints, hip-midpoint aligned; 24 SMPL joints, pelvis aligned; PVE).
Their paper's Table 1 converts with this tool ("the procedure described in [MHR]") and reports 3DPW on 14 joints and
EMDB on 24 joints. The barycentric target of the fit equals our saved pred mesh (same mhr2smpl_mapping.npz).

Also the helpers of the official MHR -> SMPL fit used by bench/eval_standard_protocol.py and bench/eval_ssp3d.py
(import_official_fitting, smplx_neutral). The MHR tool is the public MHR repository's tools/mhr_smpl_conversion
(docs/evaluation.md); Fast SAM 3D Body's converter (--fast3db_root) is its official code, installed by you.

    python -m bench.eval_fast3db_protocol --preds $FAB_HOME/bench/preds/<method> --dataset 3dpw \
        --mhr_tools $FAB_HOME/src/MHR/tools/mhr_smpl_conversion --fast3db_root <Fast SAM 3D Body checkout>
"""

import argparse
import glob
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from bench.body_models import SMPL, load_smpl_pkl
from bench.eval_hmr import H36M_TO_J14, errors
from fab import paths

# the public MHR repository's MHR -> SMPL conversion tool (github.com/facebookresearch/MHR, tools/mhr_smpl_conversion)
MHR_TOOLS_DEFAULT = Path(__import__("os").environ.get("FAB_MHR_TOOLS", paths.home() / "src" / "MHR" / "tools" / "mhr_smpl_conversion"))


def import_official_fitting(tools_dir, extra_path=""):
    """Import the official PyTorchSMPLFitting without pymomentum (the MHR -> SMPL path never calls it)."""
    class AnyModule(types.ModuleType):  # every attribute is a placeholder class (type annotations at import time)
        def __getattr__(self, k):
            if k.startswith("__"):
                raise AttributeError(k)
            return type(k, (), {})

    def stub(name, **attrs):
        import importlib.machinery

        m = sys.modules.get(name) or AnyModule(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)  # torch._dynamo's trace rules call find_spec
        m.__dict__.update(attrs)
        sys.modules[name] = m
        if "." in name:  # bind as an attribute of the parent ("import a.b as c" reads a.b)
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, m)
        return m

    for n in ("pymomentum", "pymomentum.solver", "pymomentum.geometry", "pymomentum.torch",
              "pymomentum.torch.character"):
        stub(n)
    stub("mhr")
    stub("mhr.mhr", MHR=type("MHR", (), {}))
    try:
        import sklearn.cluster  # noqa: F401
    except ImportError:
        stub("sklearn")
        stub("sklearn.cluster", KMeans=None)
    for p in (extra_path, str(tools_dir)):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from pytorch_fitting import PyTorchSMPLFitting

    return PyTorchSMPLFitting


def smplx_neutral(bm_dir):
    """The smplx library's SMPL (as the official tool uses) from our chumpy-free neutral model data."""
    import smplx
    from smplx.utils import Struct

    d = load_smpl_pkl(Path(bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl")
    data = Struct(v_template=d["v_template"], shapedirs=d["shapedirs"], posedirs=d["posedirs"],
                  J_regressor=d["J_regressor"], weights=d["weights"], kintree_table=d["kintree_table"], f=d["faces"])
    return smplx.SMPL(model_path=str(Path(bm_dir) / "smpl"), data_struct=data, gender="neutral")


def mlp_convert(pred, fast3db_root, bm_dir, batch=1024):
    """Fast SAM 3D Body's deployed MHR -> SMPL network (mhr2smpl/experiments/multiview_n30000_e500, single view, as
    run_publisher.py): barycentric SMPL-topology vertices in SAM 3D Body's y-up frame, 1500-vertex subsample,
    centred -> global orient (6D) + 21 body joints + betas -> neutral SMPL, returned in the camera frame."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mv_mhr2smpl", Path(fast3db_root) / "mocap/core/multiview_mhr2smpl.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    d = Path(fast3db_root) / "mhr2smpl/experiments/multiview_n30000_e500"
    net = mv.MinimalMultiViewNet(input_dim=1500 * 3).cuda().eval()
    net.load_state_dict(torch.load(d / "best_model.pth", map_location="cuda", weights_only=True))
    idx = torch.from_numpy(np.load(d / "sample_idx.npy").astype(np.int64)).cuda()
    body = SMPL(Path(bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda()
    flip = torch.tensor([1.0, -1.0, -1.0], device="cuda")
    out = []
    with torch.no_grad():
        for s in range(0, len(pred), batch):
            v = torch.from_numpy(pred[s : s + batch]).float().cuda() * flip  # camera -> SAM 3D Body's y-up frame
            sub = v[:, idx]
            x = (sub - sub.mean(1, keepdim=True)).flatten(1)
            go6d, bp, betas = mv.split_output(net.forward_single_view(x))
            bp69 = torch.cat([bp, torch.zeros(len(bp), 69 - bp.shape[1], device="cuda")], 1)
            vs, _ = body(mv.rot6d_to_aa(go6d), bp69, betas, None)
            vs = vs - vs[:, idx].mean(1, keepdim=True) + sub.mean(1, keepdim=True)
            out.append((vs * flip).cpu().numpy())
    return np.concatenate(out)


def load(path, dataset):
    parts = [np.load(f) for f in sorted(glob.glob(f"{path}/{dataset}_pred_shard*.npz"))]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    v = lambda n: out[f"{n}_verts"].astype(np.float32) + out[f"{n}_root"][:, None]  # noqa: E731
    return out["ids"].astype(str), v("pred"), v("gt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="dir with <dataset>_pred_shard*.npz")
    p.add_argument("--dataset", required=True)
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--mhr_tools", default=str(MHR_TOOLS_DEFAULT))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--fast3db_root", required=True, help="a checkout of Fast SAM 3D Body's official code")
    p.add_argument("--dataset_root", default="", help="also score against neutral-model GT (protocol variant)")
    args = p.parse_args()
    ids, pred, gt = load(args.preds, args.dataset)
    if args.max:
        ids, pred, gt = ids[: args.max], pred[: args.max], gt[: args.max]
    Fitting = import_official_fitting(args.mhr_tools)
    import trimesh

    body = smplx_neutral(args.bm_dir).cuda()
    for q in body.parameters():
        q.requires_grad_(False)
    edges = torch.from_numpy(trimesh.Trimesh(body.v_template.cpu().numpy(), body.faces, process=False)
                             .edges_unique.copy()).long().cuda()
    fitter = Fitting(smpl_model=body, smpl_edges=edges, smpl_model_type="smpl", hand_pose_dim=0, device="cuda",
                     batch_size=args.batch)
    t0 = time.time()
    fitted = []
    for s in range(0, len(pred), 16 * args.batch):  # chunks (the solver batches inside)
        tgt = torch.from_numpy(pred[s : s + 16 * args.batch]).float().cuda()
        prm = fitter.fit(target_vertices=tgt, single_identity=False, is_tracking=False)
        with torch.no_grad():
            v = body(betas=prm["betas"], body_pose=prm["body_pose"], global_orient=prm["global_orient"],
                     transl=prm["transl"]).vertices
        fitted.append(v.cpu().numpy())
        print(f"fitted {min(s + 16 * args.batch, len(pred))}/{len(pred)} ({time.time() - t0:.0f} s)", flush=True)
    fitted = np.concatenate(fitted)
    mlp = mlp_convert(pred, args.fast3db_root, args.bm_dir)
    j14 = np.load(Path(args.bm_dir) / "smpl" / "J_regressor_h36m.npy")[H36M_TO_J14].astype(np.float64)
    j24 = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").J_regressor.numpy().astype(np.float64)
    gts = {"gendered_gt": gt}
    if args.dataset_root:  # GT posed with the NEUTRAL model (protocol variant: some evaluators ignore the gender)
        from bench.datasets import DATASETS

        by_id = {s["id"]: s for s in DATASETS[args.dataset](args.dataset_root)}
        neutral = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl")
        gn = []
        for i in ids:
            s = by_id[i]
            t = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None]  # noqa: E731
            with torch.no_grad():
                v, _ = neutral(t(s["global_orient"]), t(s["body_pose"]), t(s["betas"]), t(s["transl"]))
            gn.append(v[0].numpy() @ s["extrinsic"][:3, :3].T + s["extrinsic"][:3, 3])
        gts["neutral_gt"] = np.stack(gn)
    rows = {}
    fit_err = []
    for gname, gv in gts.items():
        for mname, pv in (("barycentric", pred), ("official_fit", fitted), ("fast3db_mlp", mlp)):
            rows[f"{mname}/{gname}"] = [errors(pv[i].astype(np.float64), gv[i].astype(np.float64), j14, j24)
                                        for i in range(len(pred))]
    fit_err = [float(np.linalg.norm(fitted[i] - pred[i], axis=1).mean() * 1000) for i in range(len(pred))]
    summary = {k: {m: round(float(np.mean([r[m] for r in v])), 2) for m in v[0]} for k, v in rows.items()}
    summary["n"] = len(pred)
    summary["fit_vertex_error_mm"] = round(float(np.mean(fit_err)), 2)
    summary["fit_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
        root = pred.mean(1, keepdims=True)
        np.savez(Path(args.out).with_suffix(".npz"), ids=ids, root=root.astype(np.float32),
                 fitted=(fitted - root).astype(np.float16), mlp=(mlp - root).astype(np.float16),
                 **{f"{k.replace('/', '__')}__{m}": np.array([r[m] for r in v]) for k, v in rows.items() for m in v[0]})


if __name__ == "__main__":
    main()
