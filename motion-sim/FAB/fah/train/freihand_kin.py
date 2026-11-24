"""FreiHAND joint convention: training_xyz vs the MANO model's kinematic joints (smplx joints of training_mano + the
HOT3D tip vertices, the HOT3D / DexYCB / EgoStandard GT convention) and vs mesh-J (MANO J_regressor applied to the
posed 778 vertices + tips). Kinematic targets for scripts/fit_mano_joint_regressor.py and the converted pool
<pool>_kin (same layout as train/convert_egodex_labels.py pools).

    python -m train.freihand_kin check --freihand $DATA_ROOT/freihand_train
    python -m train.freihand_kin convert --root $DATA_ROOT/crops --pool freihand_lab_mesh --freihand $DATA_ROOT/freihand_train --pack
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

NUM_ANNOS = 32560
_CACHE = {}


def _json(path):
    if path not in _CACHE:
        _CACHE[path] = json.load(open(path))
    return _CACHE[path]


def kinematic(freihand, ids, mano_dir="data/mano", flat_hand_mean=False, with_verts=False):
    """(N, 21, 3) kinematic MANO joints (OpenPose order, HOT3D tips) for FreiHAND annotation ids, camera frame (m).
    The smplx mesh is placed on the FreiHAND mesh by its mean offset (FreiHAND's MANO translates after rotating about
    the origin, smplx about the root joint: a per-sample constant)."""
    import smplx
    import torch

    from bench.common import MANO21_TO_OPENPOSE, MANO_TIP_VERTS

    ids = np.asarray(ids) % NUM_ANNOS
    P = np.asarray([_json(os.path.join(freihand, "training_mano.json"))[i][0] for i in ids], np.float32)
    Vgt = np.asarray([_json(os.path.join(freihand, "training_verts.json"))[i] for i in ids], np.float32)
    m = smplx.create(f"{mano_dir}/MANO_RIGHT.pkl", "mano", is_rhand=True, use_pca=False, flat_hand_mean=flat_hand_mean)
    t = lambda x: torch.as_tensor(x)
    with torch.no_grad():
        out = m(global_orient=t(P[:, :3]), hand_pose=t(P[:, 3:48]), betas=t(P[:, 48:58]), transl=torch.zeros(len(P), 3))
    V, J = out.vertices.numpy(), out.joints[:, :16].numpy()
    off = (Vgt - V).mean(1, keepdims=True)
    V, J = V + off, J + off
    J21 = np.concatenate([J, V[:, MANO_TIP_VERTS]], 1)[:, MANO21_TO_OPENPOSE]
    return (J21, V, Vgt) if with_verts else J21


def check(args):
    from bench.common import MANO_TIP_VERTS, VertexJoints

    ids = np.arange(0, NUM_ANNOS, args.every)
    xyz = np.asarray([_json(os.path.join(args.freihand, "training_xyz.json"))[i] for i in ids], np.float64)
    best = None
    for flat in (False, True):
        J, V, Vgt = kinematic(args.freihand, ids, args.mano_dir, flat, with_verts=True)
        verr = np.linalg.norm(V - Vgt, axis=-1).mean() * 1000
        print(f"flat_hand_mean={flat}: smplx mesh vs training_verts {verr:.3f} mm", flush=True)
        if best is None or verr < best[0]:
            best = (verr, flat, J, Vgt)
    _, flat, J, Vgt = best
    vj = VertexJoints(args.mano_dir)
    MJ = np.stack([vj(v) for v in Vgt.astype(np.float64)])
    d_kin = np.linalg.norm(xyz - J, axis=-1) * 1000
    d_mj = np.linalg.norm(xyz - MJ, axis=-1) * 1000
    names = ["wrist", "T1", "T2", "T3", "T4", "I1", "I2", "I3", "I4", "M1", "M2", "M3", "M4", "R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4"]
    print(f"\nFreiHAND training_xyz ({len(ids)} annotations, flat_hand_mean={flat}) distance per joint (mm):")
    print("| joint | vs kinematic (smplx joints + tips) | vs mesh-J (J_regressor on the posed mesh + tips) |")
    print("|---|---|---|")
    for k in range(21):
        print(f"| {names[k]} | {d_kin[:, k].mean():.3f} | {d_mj[:, k].mean():.3f} |")
    print(f"| mean | {d_kin.mean():.3f} | {d_mj.mean():.3f} |")
    tips = [4, 8, 12, 16, 20]
    for k, tv in zip(tips, MANO_TIP_VERTS):
        near = Counter(np.linalg.norm(Vgt - xyz[:, k:k + 1], axis=-1).argmin(1).tolist()).most_common(2)
        print(f"tip {names[k]}: HOT3D vertex {tv}, nearest FreiHAND-mesh vertices {near}")


def convert(args):
    from train.convert_egodex_labels import reproject, write_pool

    recs = [json.loads(l) for l in open(os.path.join(args.root, args.pool, "index.jsonl"))]
    idx = [i for i, r in enumerate(recs) if r.get("joints3d") is not None]
    def fid(v):  # FreiHAND image id: an int, "<...>:<id>" or the rgb path ".../training/rgb/00001234.jpg"
        if not isinstance(v, str):
            return int(v)
        return int(os.path.splitext(os.path.basename(v))[0]) if "/" in v else int(v.split(":")[-1])

    ids = [fid(recs[i][args.id_field]) for i in idx]
    J = np.concatenate([kinematic(args.freihand, ids[k:k + 4096], args.mano_dir, args.flat_hand_mean) for k in range(0, len(ids), 4096)])
    shift = []
    for i, j in zip(idx, J.astype(np.float64)):
        r = recs[i]
        if not r.get("is_right", True):  # crops store right-hand-frame labels (FreiHAND is right hands only)
            j = j * [-1, 1, 1]
        shift.append(np.linalg.norm(np.asarray(r["joints3d"]) - j, axis=-1).mean() * 1000)
        r["joints3d"] = j.round(5).tolist()
        reproject(r, j)
        r["label_conv"] = "freihand xyz -> kinematic MANO joints (train/freihand_kin.py)"
    print(f"{args.pool}: {len(idx)} / {len(recs)} records, mean joint shift {np.mean(shift):.2f} mm", flush=True)
    write_pool(args.root, args.pool, args.pool + args.suffix, recs, args.pack)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "convert"])
    p.add_argument("--freihand", default=os.path.expandvars("$DATA_ROOT/freihand_train"))
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--every", type=int, default=20, help="check: every n-th annotation")
    p.add_argument("--root", default="$DATA_ROOT/crops")
    p.add_argument("--pool", default="freihand_lab_mesh")
    p.add_argument("--id_field", default="src", help="record field holding the FreiHAND image / annotation id")
    p.add_argument("--flat_hand_mean", action="store_true", help="as found by check")
    p.add_argument("--suffix", default="_kin")
    p.add_argument("--pack", action="store_true")
    main = dict(check=check, convert=convert)
    a = p.parse_args()
    main[a.cmd](a)
