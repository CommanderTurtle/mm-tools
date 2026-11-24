#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Integrated latency of FastFrames-LingBot (Fig. 5, FastFrames rows).

Times the complete recurrent step as it runs in the interactive application: camera rays
from the pose path, cache update, four DiT denoising steps, KV refinement and streaming
pixel decoding to uint8 RGB on the GPU (16 frames per chunk).

  python apps/lingbot/latency.py --model lingbot-1.3b --output outputs/fig5/ff_1.3b_w1_r1.json
  torchrun --nproc_per_node 8 apps/lingbot/latency.py --model lingbot-1.3b --output outputs/fig5/ff_1.3b_w8_r1.json

Protocol: an upstream example (default 00, 961 frames = 60 chunks) is generated start to
finish until no new code is compiled (at least two and at most three passes). Then whole
videos are generated again; every chunk starts after a barrier and a device
synchronization and ends with a device synchronization. Chunks 0-7 are not recorded (the
18-frame cache is full from chunk 4 on; chunks 4-7 are additional warm-up). Sampling stops
after ``--seconds`` of timed chunks and at least ``--min-samples`` chunks. The reported
latency is the median over chunks of the slowest rank's wall time; run three fresh
processes and take the median (``summarize.py``).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # repository root, not this directory
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from PIL import Image  # noqa: E402

from apps.lingbot.conditioning import Trajectory  # noqa: E402
from apps.lingbot.generate import add_model_arguments, init_distributed, load_pipeline  # noqa: E402
from apps.lingbot.upstream import import_upstream  # noqa: E402

FIRST_TIMED_CHUNK = 8


def compiler_counters():
    from torch._dynamo.utils import counters
    return {k: dict(v) for k, v in counters.items() if k in ("stats", "inductor", "aot_autograd")}


def run_video(pipe, trajectory, seed, on_chunk=None):
    pipe.restart(seed=seed)
    for chunk in range(trajectory.num_chunks):
        if on_chunk is None:
            pipe.step(trajectory.rays(chunk))
        elif not on_chunk(chunk):
            return False
    return True


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--example", type=Path, help="upstream example (default: <lingbot-root>/examples/00)")
    parser.add_argument("--frames", type=int, default=961)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)

    world, rank = init_distributed()
    wan = import_upstream(args.lingbot_root)
    example = args.example or Path(wan.__file__).resolve().parents[1] / "examples" / "00"
    trajectory = Trajectory(np.load(example / "poses.npy"), np.load(example / "intrinsics.npy"), args.frames)
    if trajectory.num_chunks <= FIRST_TIMED_CHUNK:
        raise ValueError("the camera path is too short for steady-state chunks")
    pipe = load_pipeline(args, world, rank)
    pipe.start(Image.open(example / "image.jpg").convert("RGB"), (example / "prompt.txt").read_text().strip(),
               seed=args.seed, latents=trajectory.latents)

    passes = 0
    while True:  # warm up: compile, autotune and capture every cache geometry
        before = compiler_counters()
        run_video(pipe, trajectory, args.seed)
        torch.cuda.synchronize()
        passes += 1
        stable = torch.tensor(int(compiler_counters() == before), device="cuda")
        if world > 1:
            dist.all_reduce(stable, op=dist.ReduceOp.MIN)
        if passes >= 2 and stable.item():
            break
        if passes == 3:
            raise RuntimeError("compilation did not settle within three warm-up passes")

    host = torch.empty((16, 464, 832, 3), dtype=torch.uint8, pin_memory=True) if rank == 0 else None
    samples, timed, clip = [], 0.0, 0
    before = compiler_counters()

    def timed_chunk(chunk):
        nonlocal timed
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        frames = pipe.step(trajectory.rays(chunk))
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - begin) * 1000
        if chunk < FIRST_TIMED_CHUNK:
            return True
        if rank == 0:
            host.copy_(frames, non_blocking=True)
            torch.cuda.synchronize()
        cpu_ms = (time.perf_counter() - begin) * 1000
        slowest = torch.tensor(gpu_ms, device="cuda", dtype=torch.float64)
        if world > 1:
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        samples.append(dict(clip=clip, chunk=chunk, ms=slowest.item(), rank_ms=gpu_ms, cpu_ready_ms=cpu_ms))
        timed += slowest.item() / 1000
        return timed < args.seconds or len(samples) < args.min_samples

    while run_video(pipe, trajectory, args.seed + clip, timed_chunk):
        clip += 1
    recompiled = torch.tensor(int(compiler_counters() != before), device="cuda")
    if world > 1:
        dist.all_reduce(recompiled, op=dist.ReduceOp.MAX)
    if recompiled.item():
        raise RuntimeError("code was compiled during timed chunks")

    if rank == 0:
        values = [s["ms"] for s in samples]
        record = dict(
            model=args.model, variant="fastframes", gpus=world, example=example.name, frames=args.frames,
            model_ms=statistics.median(values), mean_ms=statistics.mean(values), min_ms=min(values),
            max_ms=max(values), samples=len(values), timed_seconds=timed, warmup_passes=passes,
            cpu_ready_ms=statistics.median(s["cpu_ready_ms"] for s in samples),
            statistic="median over chunks >= 8 of the slowest rank's synchronized chunk wall time",
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, chunk_samples=samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "chunk_samples"}), flush=True)
    pipe.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
