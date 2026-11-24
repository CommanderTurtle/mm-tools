"""Apples-to-apples per-frame timing on identical pre-decoded frames: any bench.methods adapter (FAH or a baseline),
one process, one GPU, CUDA-synchronised around every frame (and around a video method's window inference in
finalize_clip, amortised over the window's frames as in the benchmark run). Headline latency comes from
an uninstrumented pass; the stage split from a second pass with per-stage synchronisation (its own e2e reported as
profiled_*_ms).

    # 1) decode the frames once (CPU): 4 EgoStandard clips (first 300 frames) + 4 HOT3D streams (2 clips of a sequence)
    python scripts/benchmark_latency.py cache --data_root $DATA_ROOT --out /tmp/latency_frames
    # 2) one system per process
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_latency.py run --cache /tmp/latency_frames --method fah_pipeline \
        --method_args "$(python -c 'import fah, json; print(json.dumps(fah.method_args("fah", "image")))')" \
        --tag fah --out $RESULTS/latency
Stages: FAH detector / pose (bench adapter -> HandEstimator.process_image, incl. crop warp + network + decoding
+ recrop, and the alternate-handedness pass) / refiner (video: fah.video.LearnedSmoother.update, split into its
network, the One Euro filter and the mesh following the filtered joints) / post (verification fusion, NMS, output
dicts), split into post.tips (the mask-guided fingertip step, fah.tips), post.fit (the MANO / MHR-Hand mesh fit) and
post.other; other adapters: end to end only.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EGOSTD_CLIPS = ["2e82e5f6-5bfe-4143-892f-36f0fd946ebc", "98a3fe8c-a694-4164-a0e1-94edce5ffab2",  # first 4 of sub24
                "0b67279e-c39c-415e-beba-0f75a320ca13", "05eb7b34-4ad6-44d1-ae6f-6317207ee8f9"]
HOT3D_PAIRS = [("clip-001881", "clip-001884"), ("clip-001984", "clip-001990"), ("clip-002196", "clip-002200"),
               ("clip-002471", "clip-002475")]


def cache(args):
    os.makedirs(args.out, exist_ok=True)
    E = args.data_root
    from bench.datasets.egostandard import EgoStandardClip
    from bench.datasets.hot3d import Hot3DClip

    for c in json.load(open(args.egostd_split))["clips"][:4] if args.egostd_split else EGOSTD_CLIPS:
        clip = EgoStandardClip(os.path.join(E, "egostandard/clips", c))
        frs = [clip.frame(i) for i in range(args.frames)]
        np.save(f"{args.out}/egostd_{c[:8]}.npy", np.stack([f["image"] for f in frs]))
        np.save(f"{args.out}/egostd_{c[:8]}_K.npy", frs[0]["K"])
        print("egostd", c, frs[0]["image"].shape, flush=True)
    for a, b in HOT3D_PAIRS:
        clips = [Hot3DClip(f"{E}/hot3d/train_aria/{n}.tar", os.environ.get("MANO_DIR", "data/mano")) for n in (a, b)]
        assert clips[0].sequence_id == clips[1].sequence_id, (a, b)
        frs = [c.frame(i) for c in clips for i in range(len(c))][:args.frames]
        np.save(f"{args.out}/hot3d_{a[5:]}.npy", np.stack([f["image"] for f in frs]))
        np.save(f"{args.out}/hot3d_{a[5:]}_K.npy", frs[0]["K"])
        print("hot3d", a, b, clips[0].sequence_id, len(frs), frs[0]["image"].shape, flush=True)


class Timer:
    def __init__(self):
        self.t = defaultdict(float)

    def wrap(self, name, fn):
        def timed(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            self.t[name] += time.perf_counter() - t
            return out

        return timed


class TimedCall:  # instance __call__ cannot be patched: proxy the detector object
    def __init__(self, obj, fn):
        self.obj, self.fn = obj, fn

    def __call__(self, *a, **k):
        return self.fn(*a, **k)

    def __getattr__(self, k):
        return getattr(self.obj, k)


def instrument(m, timer):
    if hasattr(m, "est"):  # fah
        d = m.detector
        m.detector = TimedCall(d, timer.wrap("detector", d.__call__))
        m.est.process_image = timer.wrap("pose", m.est.process_image)
        m._fit_meshes = timer.wrap("post.fit", m._fit_meshes)
        m._fit_mhr = timer.wrap("post.fit", m._fit_mhr)
        m._extend_tips = timer.wrap("post.tips", m._extend_tips)
        m.est._make_batch = timer.wrap("pose.crop", m.est._make_batch)
        if m.est.graphed is not None:
            m.est.graphed = timer.wrap("pose.network", m.est.graphed)
        else:
            m.est._forward = timer.wrap("pose.network", m.est._forward)
        if getattr(m, "refiner", None):  # the LearnedSmoother object persists across reset() (created at warm-up)
            import fah.video as video

            m.tracker.update = timer.wrap("refiner", m.tracker.update)
            m.tracker._forward = timer.wrap("refiner.network", m.tracker._forward)
            video.OneEuroFilter.__call__ = timer.wrap("refiner.euro", video.OneEuroFilter.__call__)
            video.follow_joints = timer.wrap("refiner.follow", video.follow_joints)


def stream(m, imgs, K, n=None):
    """Run one stream like bench.run (+ CUDA sync); returns per-frame e2e seconds, wall seconds, hands/frame."""
    if getattr(m, "reset", None):
        m.reset()
    imgs = np.array(imgs[:n or len(imgs)])  # whole stream in RAM before the clock starts (no decode / disk I/O)
    recs = []
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    for img in imgs:
        torch.cuda.synchronize()
        t = time.perf_counter()
        p = m(img, K)
        torch.cuda.synchronize()
        recs.append(dict(latency=time.perf_counter() - t, pred=p, subset=None))
    if getattr(m, "finalize_clip", None):
        m.finalize_clip(recs)
        torch.cuda.synchronize()
    wall = time.perf_counter() - w0
    return np.array([r["latency"] for r in recs]), wall, float(np.mean([len(r["pred"]) for r in recs]))


def run(args):
    import cv2

    from bench.run import load_method

    torch.set_num_threads(args.threads)  # same CPU budget for every system (OMP / MKL set to it by the caller too)
    cv2.setNumThreads(args.threads)
    margs = json.loads(os.path.expandvars(args.method_args))
    t = time.perf_counter()
    m = load_method(args.method, margs)
    m = getattr(m, "method", m)  # fah.FAH: time the pipeline object itself (bench.methods.fah.FAHMethod)
    load_s = time.perf_counter() - t
    files = sorted(f for f in glob.glob(f"{args.cache}/*.npy") if not f.endswith("_K.npy"))
    data = {os.path.basename(f)[:-4]: (np.load(f, mmap_mode="r"), np.load(f[:-4] + "_K.npy")) for f in files}
    for ds in ("egostd", "hot3d"):  # warm-up per resolution (cudnn autotune, CUDA graphs, allocator), not timed
        k = next(k for k in data if k.startswith(ds))
        stream(m, data[k][0], data[k][1], n=args.warmup)
    res = dict(method=args.method, tag=args.tag, method_args=margs, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               threads=args.threads, load_s=load_s, warmup_frames_per_resolution=args.warmup, streams={})
    per = defaultdict(list)
    walls = defaultdict(float)
    for k, (imgs, K) in data.items():  # headline latency: uninstrumented pass (only the per-frame synchronisation)
        lat, wall, hpf = stream(m, imgs, K)
        ds = k.split("_")[0]
        per[ds].append(lat)
        walls[ds] += wall
        res["streams"][k] = dict(frames=len(lat), shape=list(imgs.shape[1:]), wall_s=wall, hands_per_frame=hpf,
                                 median_ms=float(np.median(lat) * 1000), mean_ms=float(lat.mean() * 1000))
        print(k, res["streams"][k], flush=True)
    timer = Timer()
    instrument(m, timer)
    prof = defaultdict(list)
    stages = defaultdict(lambda: defaultdict(float))
    for k, (imgs, K) in data.items():  # stage split: a second, instrumented pass (its synchronisations add overhead)
        before = dict(timer.t)
        lat, _, _ = stream(m, imgs, K)
        ds = k.split("_")[0]
        prof[ds].append(lat)
        for s, v in timer.t.items():
            stages[ds][s] += v - before.get(s, 0.0)

    def summary(lats, wall, st, plats):
        lat = np.concatenate(lats)
        n = len(lat)
        out = dict(frames=n, median_ms=float(np.median(lat) * 1000), mean_ms=float(lat.mean() * 1000),
                   p90_ms=float(np.percentile(lat, 90) * 1000), wall_s=wall, wall_ms_per_frame=wall / n * 1000, wall_fps=n / wall)
        plat = np.concatenate(plats)  # stages are shares of the instrumented pass
        out["profiled_median_ms"], out["profiled_mean_ms"] = float(np.median(plat) * 1000), float(plat.mean() * 1000)
        tot = plat.sum()
        stage = {s: v / n * 1000 for s, v in st.items()}
        if "detector" in st:
            stage["post"] = (tot - st["detector"] - st["pose"] - st.get("refiner", 0.0)) / n * 1000
            stage["post.other"] = stage["post"] - stage.get("post.fit", 0.0) - stage.get("post.tips", 0.0)
            stage["pose.decode+recrop"] = (st["pose"] - st["pose.crop"] - st["pose.network"]) / n * 1000
            if "refiner" in st:  # features, blending, out-of-distribution guard, 2D projection
                stage["refiner.rest"] = (st["refiner"] - st["refiner.network"] - st.get("refiner.euro", 0.0)
                                         - st.get("refiner.follow", 0.0)) / n * 1000
        else:
            stage["preprocess"] = (tot - st["network"] - st["post"]) / n * 1000  # __call__ (+ finalize bookkeeping)
        out["stage_ms_per_frame"] = stage
        out["stage_share"] = {s: v / (tot / n * 1000) for s, v in stage.items() if "." not in s}
        return out

    allst = defaultdict(float)
    for ds in stages:
        for s, v in stages[ds].items():
            allst[s] += v
    res["summary"] = {ds: summary(per[ds], walls[ds], stages[ds], prof[ds]) for ds in per}
    res["summary"]["all"] = summary([x for v in per.values() for x in v], sum(walls.values()), allst, [x for v in prof.values() for x in v])
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps(res["summary"], indent=1))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(f"{args.out}/{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--out", required=True)
    c.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data"), help="dir with egostandard/ and hot3d/")
    c.add_argument("--egostd_split", default="", help="default: EGOSTD_CLIPS (= the first 4 of egostd_sub24.json)")
    c.add_argument("--frames", type=int, default=300)
    r = sub.add_parser("run")
    r.add_argument("--cache", required=True)
    r.add_argument("--method", required=True)
    r.add_argument("--method_args", default="{}")
    r.add_argument("--tag", required=True)
    r.add_argument("--warmup", type=int, default=48)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--out", required=True)
    a = p.parse_args()
    cache(a) if a.cmd == "cache" else run(a)
