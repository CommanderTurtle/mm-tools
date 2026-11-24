#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Timing, CUDA-graph capture, distributed setup and JSON output shared by all profilers.

Every component except the Wan VAE (``decoders.py``) is timed with the same protocol:

1. ``warmup`` untimed calls (after compilation / graph capture);
2. ``iters`` single calls, each bracketed by CUDA events (``event_ms``);
3. a sustained window of ``seconds``: synchronized batches of back-to-back calls
   lasting ~``batch_ms`` each; every batch contributes its wall time per call
   (``sustained_ms``).

The reported latency is the median of the sustained batches (for multi-GPU runs,
the slowest rank's median). CUDA-event and sustained medians agree to about 1% for
the DiT and within a few percent for the other components.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- distributed
class Dist:
    """Rank/world of this process; ``world == 1`` for plain ``python`` launches."""

    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        self.group_name = None
        if self.world > 1:
            dist.init_process_group("nccl", device_id=self.device)
            self.group_name = dist.group.WORLD.group_name

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def gather(self, value):
        """All-gather a picklable value; returns a list indexed by rank."""
        if self.world == 1:
            return [value]
        out = [None] * self.world
        dist.all_gather_object(out, value)
        return out

    def broadcast(self, value):
        if self.world == 1:
            return value
        box = [value if self.rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def close(self):
        if self.world > 1:
            dist.barrier()
            dist.destroy_process_group()

    def print(self, *values):
        if self.is_main:
            print(*values, flush=True)


# --------------------------------------------------------------------------- graphs
def capture_graph(fn, warmup=3, before_capture=None):
    """Warm ``fn`` on a side stream, then capture it into a CUDA graph.

    Returns ``(graph, output)``; ``graph.replay()`` refreshes ``output`` in place.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if before_capture is not None:
        before_capture()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


# --------------------------------------------------------------------------- measurement
def _stats(samples):
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def measure(fn, ctx: Dist, *, warmup=16, iters=60, seconds=60.0, batch_ms=1000.0):
    """Time ``fn`` (one component call) with the protocol described in the module docstring."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ctx.barrier()

    events = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        events.append(start.elapsed_time(end))

    # Identical batch sizes and stop decisions on every rank keep collectives matched.
    repeats = max(1, round(batch_ms / max(statistics.median(events), 1e-3)))
    repeats = max(ctx.gather(repeats)) if ctx.world > 1 else repeats
    sustained = []
    deadline = time.perf_counter() + seconds
    while True:
        keep_going = ctx.broadcast(time.perf_counter() < deadline or not sustained)
        if not keep_going:
            break
        torch.cuda.synchronize()
        ctx.barrier()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        sustained.append((time.perf_counter() - t0) * 1000.0 / repeats)
        if seconds <= 0:
            break

    local = {
        "rank": ctx.rank,
        "event": _stats(events),
        "sustained": _stats(sustained),
        "sustained_samples_ms": sustained,
        "repeats_per_batch": repeats,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    ranks = ctx.gather(local)
    return {
        "latency_ms": max(r["sustained"]["median_ms"] for r in ranks),
        "event_median_ms": max(r["event"]["median_ms"] for r in ranks),
        "statistic": "slowest-rank median of sustained batch wall time per call",
        "protocol": {"warmup": warmup, "iters": iters, "seconds": seconds, "batch_ms": batch_ms},
        "ranks": ranks,
    }


# --------------------------------------------------------------------------- output
def environment(ctx: Dist):
    try:
        clocks = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # diagnostic only
        clocks = repr(error)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(ctx.device),
        "world_size": ctx.world,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": clocks,
    }


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


def relative_l2(got, ref):
    got, ref = got.float(), ref.float()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-12))


def compare(got, ref):
    """Max-abs / relative-L2 / cosine agreement of two tensors."""
    g, r = got.float().flatten(), ref.float().flatten()
    return {
        "max_abs_diff": float((g - r).abs().max()),
        "relative_l2": relative_l2(g, r),
        "cosine": float(torch.nn.functional.cosine_similarity(g, r, dim=0)),
        "bitwise_equal": bool(torch.equal(got, ref)),
        "finite": bool(torch.isfinite(g).all()),
    }
