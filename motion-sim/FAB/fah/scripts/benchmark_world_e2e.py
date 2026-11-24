"""End-to-end world-space throughput on ONE GPU: hands (bench.methods.fah, e.g. the final pipeline) and the causal DPVO
camera (bench/world_dpvo.py --causal --no_mask) running at the same time.

The clip is decoded / rectified once (bench.datasets.hot3d) into a raw frame file before timing (excluded, as in all our
latency numbers, e.g. scripts/benchmark_latency.py); --frames_dir clips are read from their JPEGs. Each stage streams its
frames (FrameSource: a decode-ahead ring of --ahead frames filled on a background thread before the start, as a camera's
capture thread; no stage holds the clip); --frame_cache: the earlier harness (the whole clip in one raw frame file that
every stage memory-maps, --frames_dir decoded into it first; a long clip can make the stages swap). Each stage runs in its own
process (own env, own CUDA context; CPU-side launch overhead overlaps, the GPU is time-shared) and processes every frame
as fast as it can:
  hands : FAHMethod(**--hand_args)(frame, K) per frame (upload, detector, crops, model, refiner), synchronised
  camera: upload + GPU resize to --size, causal DPVO on every --stride-th frame, --scale_calls Metric3D-L calls from
          --scale_at fed frames on, every --scale_every (the clip's metric scale), synchronised; hands not masked (independent stages) or, with
          --cam_mask, masked with the padded hand boxes of frame t - 1 that the hand stage publishes (the camera
          waits for them: a streaming dependency, still overlapped)
--bg_depth: the Metric3D calls run in a third process (own CUDA context) that serves the camera's requests, so DPVO keeps
tracking; --extra_updates n --extra_every k: causal BA burst in the camera; --reinit_rot: tracking-failure re-initialisation
(as bench/world_dpvo.py, a fresh scale call per instance); --head: the learned camera-motion head on every fed frame in
the camera process (bench/world_fuse.py fuses it). --no_store / --shared_upload / --fast_res: camera-stage speedups with
the same outputs; --mps: the stages as clients of a private CUDA MPS daemon (their kernels run concurrently instead of
time-slicing). All stages warm up (CUDA graphs, cudnn, Metric3D), reset, then start together at a file barrier.
    python scripts/benchmark_world_e2e.py --clip <tar> --mode both|hands|camera --gpu 2 --hand_args '<json>' --out x.json
    (--frames_dir <dir of *.jpg + K.npy> --frames 900: a long clip, e.g. an EgoStandard frame dir; use its own --work)
-> per stage: wall ms/frame (stage end - common start) / frames, per-frame call time median / p95 / max, steady-state
   ms/frame after the last background Metric3D result (what a long video costs once the scale is fixed);
   both: e2e ms/frame = (later stage end - start) / frames and fps
--dump: the stages also save their outputs (hands: side, score, box, joints3d per frame; camera: each fed frame's causal
pose) and an md5 per input frame to --work (parity checks; not for timing)
"""
import argparse
import atexit
import collections
import glob
import hashlib
import json
import os
import pickle
import queue
import subprocess
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.environ.get("THIRDPARTY", "thirdparty")  # DPVO, HaWoR (Metric3D) checkouts: docs/world.md


def now():
    return time.clock_gettime(time.CLOCK_MONOTONIC)  # comparable across processes


class FrameSource:
    """The frames of a stage without the clip in memory: <work>/frames.txt (JPEG paths, cv2.imread BGR) or the raw frame
    file <work>/frames.npy (memory-mapped rows); the same arrays as the frame cache. iter() starts a background thread that
    decodes ahead into a ring of `ahead` frames and returns once the ring is full, then yields the frames in order; the
    last `keep` yielded frames serve look-ups (the camera's scale calls), other indices are decoded on demand."""

    def __init__(self, work, ahead=32, keep=32):
        p = os.path.join(work, "frames.txt")
        self.files = open(p).read().splitlines() if os.path.exists(p) else None
        self.raw = None if self.files else np.load(os.path.join(work, "frames.npy"), mmap_mode="r")
        self.n = len(self.files) if self.files else len(self.raw)
        self.ahead, self.keep, self.recent = ahead, keep, collections.OrderedDict()

    def __len__(self):
        return self.n

    def load(self, i):
        if self.files:
            import cv2

            return cv2.imread(self.files[i])
        return np.array(self.raw[i])  # read here (the ring's thread), not at first use

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self.n))]
        i = range(self.n)[i]
        return self.recent[i] if i in self.recent else self.load(i)

    def __iter__(self):
        ring = queue.Queue(self.ahead)
        threading.Thread(target=lambda: [ring.put(self.load(i)) for i in range(self.n)], daemon=True).start()
        while ring.qsize() < min(self.ahead, self.n):
            time.sleep(0.001)
        return self._ordered(ring)

    def _ordered(self, ring):
        self.recent.clear()
        for i in range(self.n):
            f = ring.get()
            self.recent[i] = f
            if len(self.recent) > self.keep:
                self.recent.popitem(last=False)
            yield f


def digest(f):
    return hashlib.md5(np.ascontiguousarray(f).tobytes()).hexdigest()


def wait_go(work):
    open(os.path.join(work, f"ready_{os.getpid()}"), "w").close()
    while not os.path.exists(os.path.join(work, "go")):
        time.sleep(0.0005)


def stage_hands(args, frames, K):
    import torch

    from bench.methods.fah import FAHMethod

    m = FAHMethod(**json.loads(args.hand_args))
    for _ in range(2):  # warm-up: CUDA graphs, cudnn autotune, refiner
        m.reset()
        for f in frames[: args.warmup]:
            m(f, K)
    m.reset()
    torch.cuda.synchronize()
    boxes, done = shared_boxes(args.work, len(frames))
    it = iter(frames)  # a streaming source fills its decode-ahead ring here, before the start
    wait_go(args.work)
    ts, dump = [], []
    for i, f in enumerate(it):
        t = now()
        out = m(f, K)
        torch.cuda.synchronize()
        if args.cam_mask:  # hand boxes for the camera stage (its hand mask for the next frame)
            for k, h in enumerate(sorted(out, key=lambda h: -h["score"])[:4]):
                boxes[i, k] = [*h["bbox"], 1.0]
            done[i] = 1
        ts.append((t, now()))
        if args.dump:
            dump.append((digest(f), [(bool(h["is_right"]), float(h["score"]), np.asarray(h["bbox"]), np.asarray(h["joints3d"]))
                                     for h in out]))
    if args.dump:
        pickle.dump(dump, open(os.path.join(args.work, "hands_dump.pkl"), "wb"))
    return ts


def shared_boxes(work, n):
    """Per-frame hand boxes passed from the hand stage to the camera stage (memory-mapped, written then flagged)."""
    b = np.memmap(os.path.join(work, "boxes.dat"), np.float32, "r+", shape=(n, 4, 5))
    d = np.memmap(os.path.join(work, "boxes_done.dat"), np.int8, "r+", shape=(n,))
    return b, d


def stage_camera(args, frames, K0):
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, os.getcwd())  # DPVO checkout
    from dpvo import altcorr
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from dpvo.net import DIM
    from dpvo.utils import coords_grid_with_index

    from bench.world_dpvo import masked_patchify, no_inactive_store, step_residuals, step_residuals_t
    from dpvo.lietorch import SE3

    cfg.merge_from_file(f"config/{args.config}.yaml")
    cfg.merge_from_list(args.opts)  # e.g. LOOP_CLOSURE True (DPV-SLAM proximity loop closure)
    metric = None if args.bg_depth else load_metric(args)
    H0, W0 = frames[0].shape[:2]
    s = args.size / max(H0, W0)
    H1, W1 = int(round(H0 * s)) // 16 * 16, int(round(W0 * s)) // 16 * 16
    intr = torch.as_tensor([K0[0, 0] * W1 / W0, K0[1, 1] * H1 / H0, K0[0, 2] * W1 / W0, K0[1, 2] * H1 / H0],
                           dtype=torch.float32, device="cuda")
    calib = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]])

    boxes, done = shared_boxes(args.work, len(frames))
    req, res = shared_depth(args.work)
    head_step = make_head(args, frames, K0, boxes) if args.head and not args.head_bg else None

    def run(timed):
        slam = DPVO(cfg, "dpvo.pth", ht=H1, wd=W1, viz=False)
        if args.no_store:
            no_inactive_store(slam, cfg)
        hprev, gate = None, []
        pf = slam.network.patchify
        pf.forward = masked_patchify(pf, altcorr, coords_grid_with_index, DIM)
        ts, scales, fed, dump = [], [], 0, []
        nxt, buf, q_prev, since, n_re = args.scale_at, [], None, 0, 0  # next call due; re-init state (--reinit_rot)
        seen, s_prev, first_call = 0, None, 0  # background results seen; instance's last scale; its first call index
        it = iter(frames)  # a streaming source fills its decode-ahead ring here, before the start
        if timed:
            torch.cuda.synchronize()
            wait_go(args.work)
        with torch.no_grad():
            for i, f in enumerate(it):
                t = now()
                if i % args.stride == 0:
                    xu = torch.from_numpy(f).cuda()  # one upload for DPVO and the head (--shared_upload)
                    x = xu.permute(2, 0, 1)[None].float()
                    x = F.interpolate(x, size=(H1, W1), mode="area")[0]
                    pf.hand_mask = None
                    if timed and args.cam_mask and i > 0:  # streaming dependency: the hands of frame i - 1
                        while not done[i - 1]:
                            time.sleep(0.0002)
                        m = torch.zeros(H1, W1, dtype=torch.bool, device="cuda")
                        for x0, y0, x1, y1, ok in boxes[i - 1]:
                            if ok:
                                x0, y0, x1, y1 = (np.array([x0, y0, x1, y1]) * W1 / W0).tolist()
                                d = 0.1 * max(x1 - x0, y1 - y0)
                                m[max(int(y0 - d), 0):max(int(y1 + d) + 1, 0), max(int(x0 - d), 0):max(int(x1 + d) + 1, 0)] = True
                        x = x * ~m
                        pf.hand_mask = F.max_pool2d(m[None, None].float(), 4)[0, 0] > 0
                    slam(i, x, intr)
                    if args.extra_updates and slam.is_initialized and fed % args.extra_every == 0:
                        for _ in range(args.extra_updates):  # causal BA burst before this frame's output
                            slam.update()
                    _ = slam.pg.poses_[max(slam.n - 1, 0)].clone()  # this frame's causal pose
                    if timed and args.dump:
                        dump.append((i, digest(f), _.cpu().double().numpy()))
                    if head_step is not None and fed % args.head_every == 0:  # the head on every k-th fed frame
                        hprev = head_step(xu if args.shared_upload else f, i, hprev)
                    if args.trans_gate and slam.is_initialized:  # the translation-evidence residuals (bench/world_dpvo.py)
                        if args.fast_res:
                            gate.append(torch.stack(step_residuals_t(slam, SE3)))
                        else:
                            step_residuals(slam, SE3)
                    fed += 1
                    if args.reinit_rot > 0:  # bench/world_dpvo.py causal_reinit: rotation-step failure -> fresh instance
                        buf = (buf + [(i, x, pf.hand_mask)])[-(args.reinit_back + 1):]
                        q = _.cpu().double().numpy()[3:7]
                        since = since + 1 if slam.is_initialized else 0
                        a = 0.0 if q_prev is None else np.degrees(2 * np.arccos(min(1.0, abs(float(q @ q_prev)))))
                        q_prev = q if slam.is_initialized else None
                        jump = False
                        while timed and args.reinit_scale > 0 and seen < len(scales) and res[seen, 0] == 1:
                            sc = float(res[seen, 1])  # a background result landed: scale jump within this instance?
                            if seen >= first_call and s_prev is not None and np.isfinite(sc) and sc > 0:
                                jump |= max(sc / s_prev, s_prev / sc) > args.reinit_scale
                            s_prev = sc if seen >= first_call and np.isfinite(sc) and sc > 0 else s_prev
                            seen += 1
                        if (a > args.reinit_rot or jump) and since >= args.reinit_min:
                            slam = DPVO(cfg, slam.network, ht=H1, wd=W1, viz=False)
                            if args.no_store:
                                no_inactive_store(slam, cfg)
                            for ib, xb, mb in buf:  # catch up on the buffered frames
                                pf.hand_mask = mb
                                slam(ib, xb, intr)
                            q_prev, since, n_re = None, 0, n_re + 1
                            nxt = fed + args.scale_at  # a fresh scale call for the new instance
                            s_prev, first_call = None, len(scales)
                    due = nxt
                    if len(scales) < args.scale_calls and fed > due and slam.is_initialized and slam.n > 4:
                        nxt += args.scale_every
                        k = slam.n - 5
                        pat = slam.pg.patches_[k].float().cpu().numpy()
                        fi = int(slam.tlist[int(slam.pg.tstamps_[k])])
                        if metric is None:  # hand the request to the depth stage, keep tracking
                            if timed:
                                j = len(scales)
                                req[j, 1], req[j, 2:2 + 3 * len(pat)] = fi, np.stack(
                                    [pat[:, 0, 1, 1], pat[:, 1, 1, 1], pat[:, 2, 1, 1]], 1).ravel()
                                req[j, 0] = len(pat)
                            scales.append(None)
                        else:
                            scales.append(patch_scale(metric, frames[fi], calib, pat[:, 0, 1, 1], pat[:, 1, 1, 1],
                                                      pat[:, 2, 1, 1], H1, W1))
                torch.cuda.synchronize()
                if gate:
                    _ = gate.pop().tolist()  # this frame's (full, rot) residuals
                ts.append((t, now()))
        if timed and args.reinit_rot > 0:
            print(f"[camera] {n_re} re-initialisations", flush=True)
        if timed and args.dump:
            pickle.dump(dict(poses=dump, scales=[float(v) if v is not None else None for v in scales]),
                        open(os.path.join(args.work, "camera_dump.pkl"), "wb"))
        return ts

    run(False)  # warm-up (cudnn, Metric3D)
    return run(True)


AMP = False  # Metric3D under fp16 autocast (--m3d_amp)


def load_metric(args):
    sys.path.insert(0, os.path.join(TP, "HaWoR/thirdparty/Metric3D"))
    from metric import Metric3D

    name = {"large": "v2-L", "small": "v2-S"}[args.m3d]
    return Metric3D(os.path.join(TP, f"HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_{args.m3d}_800k.pth"),
                    model_name=name)


def patch_scale(metric, frame, calib, px, py, invd, H1, W1):
    """median over the DPVO patches (1/4-res centres, inverse depths) of Metric3D depth x inverse depth"""
    import cv2

    import torch

    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        dep = metric(frame[..., ::-1].copy(), calib.copy())
    dep = cv2.resize(np.asarray(dep, np.float32), (W1, H1))
    cx, cy = px * 4 + 1.5, py * 4 + 1.5
    md = dep[np.clip(np.round(cy).astype(int), 0, H1 - 1), np.clip(np.round(cx).astype(int), 0, W1 - 1)]
    ok = (invd > 0) & (md > 0.2) & (md < 3.0)
    return float(np.median(md[ok] * invd[ok])) if ok.sum() >= 5 else float("nan")


def shared_depth(work, n=64):
    """Metric3D requests (n patches, frame, 128 x (px, py, invd)) from the camera, results (done, scale, t_end)."""
    q = np.memmap(os.path.join(work, "depth_req.dat"), np.float32, "r+", shape=(n, 2 + 3 * 128))
    r = np.memmap(os.path.join(work, "depth_res.dat"), np.float64, "r+", shape=(n, 3))
    return q, r


def make_head(args, frames, K0, boxes):
    """Learned camera-motion head (train/camera_motion.py) on (previous fed frame, this frame) as one CUDA graph -> a step
    function step(frame, index, prev) -> state for the next step."""
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, REPO)  # our train/ package, not DPVO's train.py in the cwd
    sys.modules.pop("train", None)
    from bench.world_camhead import load_head
    from train.camera_motion import GraphedStep, box_tokens, crop_K, crop_params, graph_safe_raft

    H0, W0 = frames[0].shape[:2]
    head = load_head(args.head, torch.device("cuda"))
    graph_safe_raft(head.raft)
    hgraph = GraphedStep(head)
    hcrop = crop_params((H0, W0), head.size)
    hK = torch.as_tensor(crop_K(K0, hcrop, head.size), dtype=torch.float32, device="cuda")[None]
    hx0, hy0, hcw, hch = [int(round(v)) for v in hcrop]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hray = head.rays(hK)

    def head_step(f, i, prev):
        f = f if torch.is_tensor(f) else torch.from_numpy(f).cuda()
        xh = f[hy0:hy0 + hch, hx0:hx0 + hcw].flip(-1).permute(2, 0, 1)[None].float()  # BGR frames -> the head's RGB
        xh = F.interpolate(xh, size=head.size, mode="area")
        hb = np.asarray(boxes[i - 1]) if i > 0 else np.zeros((0, 5))
        hm = torch.from_numpy(box_tokens(hb, hcrop, head.size).reshape(1, -1)).cuda()
        if prev is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return head.encode(xh), xh, hm
        o = hgraph(dict(tok_a=prev[0], img_a=prev[1], img_b=xh, ha=prev[2], hb=hm, ray=hray, K=hK))  # CUDA graph
        _ = (o["R"].clone(), o["t"].clone())  # this step's relative pose
        return o["tok_b"].clone(), xh, hm

    return head_step


def stage_head(args, frames, K0):
    """--head_bg: the camera-motion head in its own process on every --stride-th frame (with the hand boxes of t - 1)."""
    import torch

    boxes, done = shared_boxes(args.work, len(frames))
    step = make_head(args, frames, K0, boxes)

    def run(timed):
        ts, prev = [], None
        it = iter(frames)  # a streaming source fills its decode-ahead ring here, before the start
        if timed:
            torch.cuda.synchronize()
            wait_go(args.work)
        with torch.no_grad():
            for i, f in enumerate(it):
                t = now()
                if i % args.stride == 0:
                    if timed and args.cam_mask and i > 0:
                        while not done[i - 1]:
                            time.sleep(0.0002)
                    prev = step(f, i, prev)
                torch.cuda.synchronize()
                ts.append((t, now()))
        return ts

    run(False)
    return run(True)


def stage_depth(args, frames, K0):
    """Background Metric3D (or --depth_model, bench/world_depth.py): serves the camera's scale requests on its own process /
    CUDA context (--depth_py: its python env)."""
    import torch

    if args.depth_model == "m3d":
        metric = load_metric(args)
    else:
        sys.path.insert(0, REPO)
        from bench.world_depth import load

        metric = load(args.depth_model)
    H0, W0 = frames[0].shape[:2]
    s = args.size / max(H0, W0)
    H1, W1 = int(round(H0 * s)) // 16 * 16, int(round(W0 * s)) // 16 * 16
    calib = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]])
    metric(frames[0][..., ::-1].copy(), calib.copy())  # warm-up
    torch.cuda.synchronize()
    req, res = shared_depth(args.work)
    wait_go(args.work)
    ts, j = [], 0
    while j < args.scale_calls and not os.path.exists(os.path.join(args.work, "stop")):
        if req[j, 0] == 0:
            time.sleep(0.0005)
            continue
        f = frames[int(req[j, 1])]  # (a streaming source decodes it here, before the call's timer)
        t = now()
        m = int(req[j, 0])
        p = np.asarray(req[j, 2:2 + 3 * m]).reshape(m, 3)
        sc = patch_scale(metric, f, calib, p[:, 0], p[:, 1], p[:, 2], H1, W1)
        torch.cuda.synchronize()
        res[j] = [1.0, sc, now()]
        ts.append((t, now()))
        j += 1
    return ts


def summarise(ts, n):
    d = np.array([b - a for a, b in ts]) * 1000
    return dict(per_frame_ms_median=float(np.median(d)), per_frame_ms_p95=float(np.percentile(d, 95)),
                per_frame_ms_max=float(d.max()), frames=n)


def main(args):
    global AMP
    AMP = args.m3d_amp
    if args.stage:  # child
        if args.frame_cache:  # the whole clip memory-mapped from the raw frame file
            frames = np.load(os.path.join(args.work, "frames.npy"), mmap_mode="r")
            frames = [np.ascontiguousarray(f) for f in frames]
        else:
            frames = FrameSource(args.work, args.ahead)
        K = np.load(os.path.join(args.work, "K.npy"))
        fn = {"hands": stage_hands, "camera": stage_camera, "depth": stage_depth, "head": stage_head}[args.stage]
        ts = fn(args, frames, K)
        json.dump(ts, open(os.path.join(args.work, f"{args.stage}.json"), "w"))
        return
    sys.path.insert(0, REPO)
    os.makedirs(args.work, exist_ok=True)
    for f in os.listdir(args.work):
        if f.startswith(("ready_", "go", "stop", "hands.json", "camera.json", "depth.json", "head.json", "frames.txt",
                         "hands_dump.pkl", "camera_dump.pkl")):
            os.remove(os.path.join(args.work, f))
    if args.frames_dir and not args.frame_cache:  # long clips, streamed: the stages read the JPEGs (FrameSource) + K.npy
        files = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))[:args.frames]
        open(os.path.join(args.work, "frames.txt"), "w").write("\n".join(files))
        np.save(os.path.join(args.work, "K.npy"), np.load(os.path.join(args.frames_dir, "K.npy")).astype(np.float64))
    elif not os.path.exists(os.path.join(args.work, "frames.npy")) and args.frames_dir:  # long clips: cached JPEGs + K.npy
        import cv2

        files = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))[:args.frames]
        np.save(os.path.join(args.work, "frames.npy"), np.stack([cv2.imread(f) for f in files]))  # BGR, as the HOT3D path
        np.save(os.path.join(args.work, "K.npy"), np.load(os.path.join(args.frames_dir, "K.npy")).astype(np.float64))
    if not os.path.exists(os.path.join(args.work, "frames.txt")) and not os.path.exists(os.path.join(args.work, "frames.npy")):
        from bench.datasets.hot3d import Hot3DClip  # --clip only (HOT3D loader: the eval extra)

        clip = Hot3DClip(args.clip, args.mano_dir)
        fr = [clip.frame(i) for i in range(min(len(clip), args.frames))]
        np.save(os.path.join(args.work, "frames.npy"), np.stack([f["image"] for f in fr]))
        np.save(os.path.join(args.work, "K.npy"), fr[0]["K"].astype(np.float64))
    n = len(FrameSource(args.work))
    np.memmap(os.path.join(args.work, "boxes.dat"), np.float32, "w+", shape=(n, 4, 5)).flush()
    np.memmap(os.path.join(args.work, "boxes_done.dat"), np.int8, "w+", shape=(n,)).flush()
    np.memmap(os.path.join(args.work, "depth_req.dat"), np.float32, "w+", shape=(64, 2 + 3 * 128)).flush()
    np.memmap(os.path.join(args.work, "depth_res.dat"), np.float64, "w+", shape=(64, 3)).flush()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    if args.mps:  # a private MPS daemon on this GPU: the stages' kernels share the GPU concurrently instead of time-slicing
        pipe = os.path.join(args.work, "mps")
        os.makedirs(pipe, exist_ok=True)
        menv = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=pipe)
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=menv, check=True)
        atexit.register(subprocess.run, ["nvidia-cuda-mps-control"], input=b"quit\n", env=menv)
        env.update(CUDA_VISIBLE_DEVICES="0", CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=pipe)  # index in the server's GPUs
    common = [os.path.abspath(__file__), "--work", args.work, "--clip", args.clip, "--ahead", str(args.ahead)]
    common += ["--frame_cache"] if args.frame_cache else []
    common += ["--dump"] if args.dump else []
    stages = ["hands", "camera"] if args.mode == "both" else [args.mode]
    if args.bg_depth and "camera" in stages:
        stages.append("depth")
    if args.head and args.head_bg and "camera" in stages:
        stages.insert(2, "head")
    procs = []
    for st in stages:
        if st == "hands":
            cmd = [args.hands_py] + common + ["--stage", "hands", "--hand_args", args.hand_args, "--warmup", str(args.warmup)]
            cmd += ["--cam_mask"] if args.cam_mask else []
            procs.append(subprocess.Popen(cmd, cwd=REPO, env=dict(env, PYTHONPATH=REPO)))
        else:
            dp = os.path.join(TP, "DPVO")
            py = args.depth_py if st == "depth" and args.depth_model != "m3d" else args.camera_py
            cmd = [py] + common + ["--stage", st, "--size", str(args.size), "--stride", str(args.stride),
                                               "--config", args.config, "--scale_at", str(args.scale_at),
                                               "--scale_calls", str(args.scale_calls), "--scale_every", str(args.scale_every)]
            cmd += ["--cam_mask"] if args.cam_mask else []
            cmd += ["--bg_depth"] if args.bg_depth else []
            cmd += ["--m3d", args.m3d, "--extra_updates", str(args.extra_updates), "--extra_every", str(args.extra_every)]
            cmd += ["--m3d_amp"] if args.m3d_amp else []
            cmd += ["--opts", *args.opts] if args.opts else []
            cmd += ["--reinit_rot", str(args.reinit_rot), "--reinit_min", str(args.reinit_min), "--reinit_back", str(args.reinit_back),
                    "--reinit_scale", str(args.reinit_scale)]
            cmd += ["--head", args.head] if args.head else []
            cmd += ["--head_bg"] if args.head_bg else []
            cmd += ["--head_every", str(args.head_every)]
            cmd += ["--trans_gate"] if args.trans_gate else []
            cmd += ["--no_store"] if args.no_store else []
            cmd += ["--shared_upload"] if args.shared_upload else []
            cmd += ["--fast_res"] if args.fast_res else []
            cmd += ["--depth_model", args.depth_model]
            procs.append(subprocess.Popen(cmd, cwd=dp, env=dict(env, PYTHONPATH=f"{REPO}/fah/_vendor:{REPO}:{dp}")))
    while len([f for f in os.listdir(args.work) if f.startswith("ready_")]) < len(stages):
        if any(p.poll() is not None for p in procs):
            for p in procs:  # do not leave the other stages waiting for the barrier
                p.kill()
            raise SystemExit("a stage died during warm-up")
        time.sleep(0.05)
    t0 = now()
    open(os.path.join(args.work, "go"), "w").close()
    for p in procs[:len([st for st in stages if st != "depth"])]:
        p.wait()
    open(os.path.join(args.work, "stop"), "w").close()  # the depth stage ends with the camera
    for p in procs:
        p.wait()
    res = dict(mode=args.mode, gpu=args.gpu, clip=os.path.basename(args.frames_dir or args.clip), frames=n, hand_args=json.loads(args.hand_args),
               frame_source="cache" if args.frame_cache else f"stream (ahead {args.ahead})",
               camera_cfg=dict(size=args.size, stride=args.stride, config=args.config, scale_at=args.scale_at,
                               scale_calls=args.scale_calls, scale_every=args.scale_every, cam_mask=args.cam_mask,
                               bg_depth=args.bg_depth, m3d=args.m3d, m3d_amp=args.m3d_amp, extra_updates=args.extra_updates,
                               extra_every=args.extra_every, opts=args.opts, reinit_rot=args.reinit_rot, head=args.head,
                               no_store=args.no_store, shared_upload=args.shared_upload, fast_res=args.fast_res,
                               depth_model=args.depth_model), mps=args.mps)
    end, tss = t0, {st: json.load(open(os.path.join(args.work, f"{st}.json"))) for st in stages}
    if "depth" in tss:  # scale availability: when each background Metric3D result landed (s after start)
        ts = tss.pop("depth")
        res["depth"] = dict(calls=len(ts), call_ms=[(b - a) * 1000 for a, b in ts], ready_s=[b - t0 for a, b in ts])
    t_fix = t0 + max(res.get("depth", {}).get("ready_s", [0]) or [0])  # metric scale fixed from here on
    steady = []
    for st, ts in tss.items():
        res[st] = dict(summarise(ts, n), wall_ms_per_frame=(ts[-1][1] - t0) * 1000 / n)
        after = [(a, b) for a, b in ts if a >= t_fix]
        if len(after) > 10:  # steady state once the scale calls are done (long videos)
            res[st]["steady_ms_per_frame"] = (after[-1][1] - after[0][0]) * 1000 / len(after)
            steady.append(res[st]["steady_ms_per_frame"])
        end = max(end, ts[-1][1])
    res["e2e_ms_per_frame"] = (end - t0) * 1000 / n
    res["e2e_steady_ms_per_frame"] = max(steady) if steady else None
    res["e2e_fps"] = 1000 / res["e2e_ms_per_frame"]
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "hot3d/train_aria/clip-002471.tar"))
    p.add_argument("--mode", default="both", choices=["both", "hands", "camera"])
    p.add_argument("--stage", default="", choices=["", "hands", "camera", "depth", "head"])
    p.add_argument("--gpu", default="0")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--frames_dir", default="", help="instead of --clip: a cached frame dir (*.jpg + K.npy), e.g. EgoStandard")
    p.add_argument("--frame_cache", action="store_true", help="the earlier harness: the clip in one raw frame file that every "
                   "stage memory-maps, instead of the streaming decode-ahead ring")
    p.add_argument("--ahead", type=int, default=32, help="streaming: frames decoded ahead per stage")
    p.add_argument("--dump", action="store_true", help="save the stages' outputs + input frame md5s to --work (parity)")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--hand_args", default="{}")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--config", default="fast")
    p.add_argument("--opts", nargs="*", default=[], help="DPVO config overrides, e.g. LOOP_CLOSURE True")
    p.add_argument("--scale_at", type=int, default=15)
    p.add_argument("--scale_calls", type=int, default=1)
    p.add_argument("--bg_depth", action="store_true", help="Metric3D in a third process (off the camera's critical path)")
    p.add_argument("--m3d", default="large", choices=["large", "small"])
    p.add_argument("--m3d_amp", action="store_true")
    p.add_argument("--extra_updates", type=int, default=0, help="camera: causal BA burst (extra DPVO updates)")
    p.add_argument("--extra_every", type=int, default=1)
    p.add_argument("--scale_every", type=int, default=15)
    p.add_argument("--reinit_rot", type=float, default=0, help="camera: re-init on a rotation step > deg (bench/world_dpvo.py)")
    p.add_argument("--trans_gate", action="store_true", help="camera: also compute the translation-gate residuals")
    p.add_argument("--head_every", type=int, default=1, help="camera: the head on every k-th fed frame (pairs k apart)")
    p.add_argument("--head_bg", action="store_true", help="the head in its own process (not on the camera's critical path)")
    p.add_argument("--head", default="", help="camera: also run the camera-motion head (checkpoint) on every fed frame")
    p.add_argument("--reinit_min", type=int, default=20)
    p.add_argument("--reinit_scale", type=float, default=0, help="camera: also re-init when consecutive background scales jump > x")
    p.add_argument("--reinit_back", type=int, default=8)
    p.add_argument("--cam_mask", action="store_true", help="both: camera masks the hand boxes of frame t - 1 (waits for them)")
    p.add_argument("--no_store", action="store_true", help="camera: skip DPVO's inactive-edge store (bench/world_dpvo.py "
                   "no_inactive_store; only read by loop closure)")
    p.add_argument("--shared_upload", action="store_true", help="camera: the head crops the frame DPVO uploaded")
    p.add_argument("--fast_res", action="store_true", help="camera: sync-free gate residuals (step_residuals_t)")
    p.add_argument("--mps", action="store_true", help="run the stages under a private CUDA MPS daemon on --gpu")
    p.add_argument("--depth_model", default="m3d", choices=["m3d", "moge2"], help="the background scale calls' depth model")
    p.add_argument("--depth_py", default=sys.executable, help="python env of a non-Metric3D --depth_model")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--hands_py", default=sys.executable)
    p.add_argument("--camera_py", default=sys.executable, help="python env with DPVO")
    p.add_argument("--work", default="/tmp/world_e2e")
    p.add_argument("--out", default="")
    a = p.parse_args()
    if a.depth_model != "m3d" and not a.bg_depth:  # the inline camera stage only loads Metric3D (load_metric)
        p.error(f"--depth_model {a.depth_model} needs --bg_depth")
    main(a)
