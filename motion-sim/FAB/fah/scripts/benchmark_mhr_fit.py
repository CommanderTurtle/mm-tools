"""Paired timing of the MHR-Hand fit (fah/mhr_fit.py) in the image pipeline: one FAHMethod, every frame run
with the fit on and off (order alternating per frame, the device synchronised around every call), the same for the MANO
mesh stage; HOT3D frames (bench.run's loader, mostly 2 hands). Reports median ms / fps per setting and the median
paired difference; --repeats R: every setting R times per frame, the per-frame minimum kept (robust to other jobs
time-sharing the GPU).

    python scripts/benchmark_mhr_fit.py --method_args '{..., "mesh_fit": 1, "mhr_fit": 1, "mhr_fit_init": ...}' --clips 4 --frames 60 \
        --out mhr_fit_timing.json
    python scripts/benchmark_mhr_fit.py --stage $R/hot3d/image --lmf $L --mhr '[{"name": ..., "iters": 1, "init": ...}]'
--stage: the mesh stages alone (MANO MeshStage vs MHR fit variants, CUDA graphs) on stored 2-hand frames of a run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stage(args):
    import glob
    import pickle

    from fah.fast_mano import MeshStage
    from fah.mhr_fit import MHRHandFitter

    torch.backends.cuda.matmul.allow_tf32 = False
    frames = []
    for f in sorted(glob.glob(os.path.join(args.stage, "*.pkl")))[:args.clips]:
        d = pickle.load(open(f, "rb"))
        frames += [fr for fr in d["frames"] if len(fr["pred"]) == 2 and all(p.get("verts3d_dense") is not None for p in fr["pred"])][:args.frames]
    m = lambda p: np.array([1.0 if p["is_right"] else -1.0, 1, 1])
    data = [(np.stack([p["verts3d_dense"] * m(p) for p in fr["pred"]]), np.stack([p["joints3d"] * m(p) for p in fr["pred"]])) for fr in frames]
    fns = {"mano": MeshStage(device="cuda", iters=1, wv=0.3, lmf=args.lmf or None, jac="fd")}
    for sp in json.loads(args.mhr):
        fns[sp["name"]] = MHRHandFitter(torch.load(sp["asset"], weights_only=False), "cuda", iters=sp.get("iters", 2), init_net=sp.get("init") or None)
    for f in fns.values():
        for _ in range(5):
            f(*data[0])
    best = {k: [1e9] * len(data) for k in fns}
    for _ in range(args.repeats):
        for k, f in fns.items():
            for i, (tV, tJ) in enumerate(data):
                torch.cuda.synchronize()
                t = time.perf_counter()
                f(tV, tJ)
                torch.cuda.synchronize()
                best[k][i] = min(best[k][i], (time.perf_counter() - t) * 1000)
    res = dict(frames=len(data), gpu=torch.cuda.get_device_name(), ms={k: float(np.median(v)) for k, v in best.items()})
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(dict(args=vars(args), **res), open(args.out, "w"), indent=1)


def main(args):
    if args.stage:
        return stage(args)
    from bench.methods.fah import FAHMethod
    from bench.run import iter_clips

    m = FAHMethod(**json.loads(os.path.expandvars(args.method_args)))
    on, mano = m.mhr_fit, m.mesh_fit
    settings = {"base": (mano, 0), "mhr": (mano, on), "nomano": (0, 0), "nomano_mhr": (0, on)}
    ns = argparse.Namespace(dataset="hot3d", split=args.split, data_root=args.data_root, mano_dir=args.mano_dir, shard=0, num_shards=1)
    lat = {k: [] for k in settings}
    hands = []
    for c, (name, make) in enumerate(iter_clips(ns)):
        if c >= args.clips:
            break
        clip = make()
        idx = np.linspace(0, len(clip) - 1, args.frames).astype(int)
        for n, i in enumerate(idx):
            fr = clip.frame(int(i))
            order = list(settings) if n % 2 == 0 else list(settings)[::-1]
            for k in order:
                m.mesh_fit, m.mhr_fit = settings[k]
                best = 1e9
                for _ in range(args.repeats):
                    m.reset()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    out = m(fr["image"], fr["K"])
                    torch.cuda.synchronize()
                    best = min(best, (time.perf_counter() - t) * 1000)
                if n >= args.warmup:
                    lat[k].append(best)
            if n >= args.warmup:
                hands.append(len(out))
        print(name, {k: round(float(np.median(v)), 2) for k, v in lat.items() if v}, flush=True)
    L = {k: np.asarray(v) for k, v in lat.items()}
    res = dict(frames=len(hands), hands_per_frame=float(np.mean(hands)), gpu=torch.cuda.get_device_name(),
               ms={k: float(np.median(v)) for k, v in L.items()}, fps={k: float(1000 / np.median(v)) for k, v in L.items()},
               mhr_extra_ms=float(np.median(L["mhr"] - L["base"])), mhr_extra_ms_nomano=float(np.median(L["nomano_mhr"] - L["nomano"])),
               mano_stage_ms=float(np.median(L["base"] - L["nomano"])))
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(dict(args=vars(args), **res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_args", default="{}")
    ap.add_argument("--stage", default="", help="bench.run dir with dense outputs: time the mesh stages alone")
    ap.add_argument("--lmf", default="", help="--stage: the MANO stage's learned fitter")
    ap.add_argument("--mhr", default="[]", help="--stage: MHR fits (json list of {name, asset (build_asset file), iters, init})")
    ap.add_argument("--split", default="bench/splits/hot3d_bench.json")
    ap.add_argument("--data_root", default=os.path.expandvars("$DATA_ROOT/hot3d"))
    ap.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="")
    main(ap.parse_args())
