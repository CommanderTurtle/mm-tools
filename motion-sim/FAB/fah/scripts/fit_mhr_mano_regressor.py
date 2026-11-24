"""Fit the MHR-hand-vertex -> MANO-778 vertex regressor of the MHR-Hand fit (fah/mhr_fit.py) on GT MANO meshes of
training splits only. Alternates (a) MHR fits (the inference objective: surface + kinematic joints) to the GT meshes /
joints through the current regressor and (b) per MANO vertex a convex combination (non-negative, sums to one) of the K
fitted MHR hand vertices nearest to it over all samples, so that the fitted MHR mesh maps onto the GT MANO surface.
Starts from fah/assets/mhr_to_mano778_regressor.npz (fitted on SAM 3D Body teacher meshes of FreiHAND train).

Data: crop-pool index files (/data/crops/<pool>/index.jsonl + the per-crop GT vertices *_v.npy; left hands are stored
in the right-hand frame):
FreiHAND train (freihand_lab_mesh_kin), DexYCB s0 train (dexycb_lab), HOT3D train clips (hot3d_lab_mesh), OakInk2
(oakink2_ego_lab). Sequences of the benchmarks (HOT3D v1 = HaWoR's validation sequences, DexYCB s0 test, the
OakInk2 dev split; EgoStandard and bench2d images are in none of these pools) are excluded and the overlap asserted to
be zero; 10 % of each pool's training sequences are held out for validation (the round with the lowest held-out residual
is kept).

    python scripts/fit_mhr_mano_regressor.py --pools hot3d_lab_mesh:6000,dexycb_lab:6000,freihand_lab_mesh_kin:6000,oakink2_ego_lab:6000 \
        --rounds 4 --out fah/assets/mhr_to_mano778.npz --report overlap.json
(held-out pool GT residual 2.72 -> 2.39 mm; round 3 is kept as fah/assets/mhr_to_mano778.npz)
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fah.mhr_fit import VREG_TEACHER, MHRHandFitter, build_asset  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def benchmark_sequences():
    """Sequences no training sample may come from."""
    sp = lambda f: json.load(open(os.path.join(REPO, "bench", "splits", f)))
    hot3d = set(sp("hot3d_bench.json")["hawor_val_sequences"])
    dexycb = set(sp("dexycb_s0_test.json")["test_sequences"])
    oak = {c.split("@")[0] for c in sp("oakink2_ego_dev.json")["clips"]}
    return dict(hot3d=hot3d, dexycb=dexycb, oakink2=oak)


def sequence_of(pool, r):
    if "sequence_id" in r:
        return r["sequence_id"]
    return "freihand:%d" % (int(os.path.basename(r["src"])[:8]) % 32560)  # FreiHAND train: one annotation (4 backgrounds) = one "sequence"


def load_pool(root, pool, n, rng, exclude, held_out=0.1):
    """n training + held-out GT hands of a pool (right-hand frame), their sequences; records of the excluded
    (benchmark) sequences dropped first."""
    recs = [json.loads(line) for line in open(os.path.join(root, pool, "index.jsonl"))]
    n_all = len(recs)
    recs = [r for r in recs if sequence_of(pool, r) not in exclude]
    seqs = sorted({sequence_of(pool, r) for r in recs})
    hold = {s for s in seqs if zlib.crc32(s.encode()) % 100 < held_out * 100}
    out = {}
    for name, keep in (("train", lambda s: s not in hold), ("val", lambda s: s in hold)):
        cand = [r for r in recs if keep(sequence_of(pool, r))]
        pick = [cand[i] for i in rng.choice(len(cand), min(n if name == "train" else n // 6, len(cand)), replace=False)]
        V, J, S = [], [], []
        for r in pick:  # crop pools store left hands mirrored already (flipped crops): vertices and joints in the right-hand frame
            vf = os.path.join(root, pool, r.get("verts_file") or r["file"].replace(".jpg", "_v.npy"))
            V.append(np.load(vf).astype(np.float64)), J.append(np.asarray(r["joints3d"], np.float64)), S.append(sequence_of(pool, r))
        out[name] = (np.stack(V), np.stack(J), S)
    return out, {sequence_of(pool, r) for r in recs}, n_all - len(recs)


def save(path, W, o_idx):
    j, v = np.nonzero(W)
    np.savez(path, joint=j.astype(np.int16), vertex=o_idx[v].astype(np.int32), weight=W[j, v].astype(np.float32), num_joints=np.int64(len(W)))


def main(args):
    from fah import load_hand_model

    rng = np.random.RandomState(0)
    bench = benchmark_sequences()
    data, report = {"train": [], "val": []}, {}
    for spec in args.pools.split(","):
        pool, n = spec.split(":")
        d, seqs, dropped = load_pool(args.crops, pool, int(n), rng, set().union(*bench.values()))
        used = set(d["train"][2]) | set(d["val"][2])
        overlap = {k: sorted(used & s) for k, s in bench.items()}
        report[pool] = dict(sequences=len(seqs), used_sequences=len(used), train=len(d["train"][0]), val=len(d["val"][0]),
                            benchmark_records_dropped=dropped, overlap={k: len(v) for k, v in overlap.items()})
        assert not any(overlap.values()), (pool, overlap)  # no benchmark sequence in the fitting data
        print(pool, report[pool], flush=True)
        for k in data:
            data[k].append(d[k])
    cat = lambda k: [np.concatenate([d[i] for d in data[k]]) for i in (0, 1)]
    (Vt, Jt), (Vv, Jv) = cat("train"), cat("val")
    report["benchmark_sequences_excluded"] = {k: len(v) for k, v in bench.items()}
    model, _ = load_hand_model(args.checkpoint, device="cpu", verbose=False)
    head = model.head_pose_hand
    vreg, best = VREG_TEACHER, (1e9, None)
    for rnd in range(args.rounds + 1):
        a = build_asset(head, vreg=vreg)
        fit = MHRHandFitter(a, args.device, iters=args.iters, graph=False, compile=False, chunk=args.chunk)
        o = fit(Vv, Jv)
        val = float(np.linalg.norm(o["verts_mano"] - Vv, axis=-1).mean() * 1000)
        print(f"round {rnd}: {os.path.basename(vreg)}: held-out GT residual {val:.3f} mm", flush=True)
        report[f"round{rnd}_val_mm"] = val
        if val < best[0]:
            best = (val, vreg)
        if rnd == args.rounds:
            break
        o = fit(Vt, Jt)
        M = o["verts"]  # (N, 2445, 3) fitted MHR hand meshes
        print(f"  train GT residual {np.linalg.norm(o['verts_mano'] - Vt, axis=-1).mean() * 1000:.3f} mm", flush=True)
        step = max(len(M) // 400, 1)
        d = torch.cdist(torch.as_tensor(Vt[::step], dtype=torch.float32), torch.as_tensor(M[::step], dtype=torch.float32)).mean(0).numpy()  # (778, 2445)
        W = np.zeros((778, M.shape[1]))
        for i in range(778):
            cand = np.argsort(d[i])[:args.k]
            A = np.concatenate([M[:, cand].transpose(0, 2, 1).reshape(-1, args.k), np.full((1, args.k), args.sum_w)])
            w, _ = nnls(A, np.concatenate([Vt[:, i].reshape(-1), [args.sum_w]]))
            W[i, cand] = w / max(w.sum(), 1e-12)
        vreg = os.path.splitext(args.out)[0] + f"_r{rnd}.npz"
        save(vreg, W, np.asarray(a["o_idx"]))
        print(f"  -> {vreg}: {int((W > 0).sum())} weights, <= {int((W > 0).sum(1).max())} per vertex", flush=True)
    report["kept"] = os.path.basename(best[1])
    if best[1] != VREG_TEACHER:
        os.replace(best[1], args.out)
    print("kept", report["kept"], "->", args.out if best[1] != VREG_TEACHER else "(original regressor)", flush=True)
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    ap.add_argument("--crops", default="/data/crops")
    ap.add_argument("--pools", default="hot3d_lab_mesh:6000,dexycb_lab:6000,freihand_lab_mesh_kin:6000,oakink2_ego_lab:6000")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--k", type=int, default=16, help="candidate MHR vertices per MANO vertex")
    ap.add_argument("--sum_w", type=float, default=10.0, help="weight of the sum-to-one row")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default="")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
