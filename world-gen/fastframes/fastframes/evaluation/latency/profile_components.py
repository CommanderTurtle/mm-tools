#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Component latency profiler for FastFrames and its autoregressive baselines.

Measures DiT, KV refinement, pixel decoding or VAE decoding and writes one JSON
record. Defaults: steady state, batch 1, BF16, one latent frame per step and a
21-latent-frame cache. Use ``--chunk 3`` for three-latent units and DiT
``--timestep 0`` for a clean cache write. See docs/inference.md#profiling for commands.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from dit import FrameDiT, Geometry
from models import MODEL_DEFAULTS, load_dit, tuned_split_kv
from timing import Dist, environment, measure, write_json

# Timing defaults per component (warm-up calls, CUDA-event samples, sustained seconds, batch length).
TIMING = {
    "dit": dict(warmup=16, iters=60, seconds=60.0, batch_ms=1000.0),
    "kv-refine": dict(warmup=12, iters=40, seconds=60.0, batch_ms=500.0),
    "pixel-decoder": dict(warmup=100, iters=200, seconds=60.0, batch_ms=200.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("component", choices=("dit", "kv-refine", "pixel-decoder", "vae"))
    p.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="1.3b")
    p.add_argument("--resolution", choices=("480p", "720p"), default="480p")
    p.add_argument("--frames", type=int, default=21, help="latent frames held in the KV cache")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--seed", type=int, default=0)
    t = p.add_argument_group("timing (defaults depend on the component)")
    t.add_argument("--warmup", type=int)
    t.add_argument("--iters", type=int)
    t.add_argument("--seconds", type=float)
    t.add_argument("--batch-ms", type=float)
    d = p.add_argument_group("dit")
    d.add_argument("--dit", help="DiT weights (.safetensors or a .pt with generator_ema); random when omitted. "
                   "kv-refine also reads its patch embedding from here")
    d.add_argument("--mode", choices=("baseline", "optimized"), default="optimized")
    d.add_argument("--timestep", type=float, help="model timestep (default: the final steady-state step)")
    d.add_argument("--chunk", type=int, default=1, help="latent frames per generation unit (dit, vae)")
    d.add_argument("--split-kv", type=int, help="KV segments of the optimized attention")
    d.add_argument("--compile-mode", default=None,
                   help="torch.compile mode; default 'max-autotune-no-cudagraphs' (optimized) / 'default' "
                        "(baseline); 'none' disables compilation")
    d.add_argument("--no-cuda-graph", action="store_true")
    k = p.add_argument_group("kv-refine")
    k.add_argument("--refiner", help="KV refinement weights (.safetensors)")
    k.add_argument("--shard", choices=("tokens", "layers"), default="tokens",
                   help="multi-GPU partition of the refinement heads")
    k.add_argument("--streams", type=int, default=1, help="CUDA streams for the per-layer heads (1 GPU)")
    k.add_argument("--no-compile", action="store_true", help="do not compile the refinement heads")
    x = p.add_argument_group("decoders")
    x.add_argument("--decoder", help="pixel decoder weights (.safetensors)")
    v = p.add_argument_group("vae")
    v.add_argument("--vae", help="Wan2.1_VAE.pth")
    v.add_argument("--latents", help="optional latent tensor file [1, F, 16, h, w] (.pt or .safetensors)")
    v.add_argument("--streams-measured", type=int, default=6, help="VAE: measured decoding streams")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved configuration for WORLD_SIZE (default 1) processes and exit "
                        "without using a GPU, e.g. WORLD_SIZE=8 python profile_components.py dit ... --dry-run")
    return p.parse_args()


def plan(args, world):
    """Resolved configuration of a run (no GPU work); raises on partitions that do not divide evenly."""
    from fastframes.wan.model import WAN_CONFIGS
    defaults = MODEL_DEFAULTS[args.model]
    layers = WAN_CONFIGS[defaults["wan"]].num_layers
    geometry = Geometry.from_resolution(args.resolution, args.frames)
    out = dict(component=args.component, model=args.model, resolution=args.resolution, gpus=world,
               frame_tokens=geometry.tokens, kv_len=geometry.kv_len)

    def need(ok, message):
        if not ok:
            raise ValueError(message)
    if args.component == "dit":
        tuned = tuned_split_kv(args.model, args.resolution, world)
        split = args.split_kv or (tuned if args.mode == "optimized" else 1)
        tokens = geometry.tokens * args.chunk
        need(world == 1 or args.mode == "optimized", "context parallelism needs --mode optimized")
        need(tokens % world == 0, f"{tokens} query tokens are not divisible by {world} ranks")
        need(geometry.kv_len % split == 0, f"KV length {geometry.kv_len} is not divisible by split={split}")
        out.update(mode=args.mode, chunk=args.chunk, split_kv=split, query_tokens_per_rank=tokens // world,
                   timestep=defaults["timestep"] if args.timestep is None else args.timestep)
    elif args.component == "kv-refine":
        need(args.streams == 1 or (world == 1 and layers % args.streams == 0), "--streams needs 1 GPU, equal groups")
        if world > 1:
            total = geometry.tokens if args.shard == "tokens" else layers
            need(total % world == 0, f"{total} {args.shard} are not divisible by {world} ranks")
            out.update(shard=args.shard, per_rank=total // world)
        out.update(streams=args.streams, layers=layers)
    elif args.component == "pixel-decoder":
        need(geometry.tokens % world == 0, f"{geometry.tokens} tokens are not divisible by {world} ranks")
        out.update(patch_rows_per_rank=-(-geometry.grid[0] // world))
    else:
        need(world == 1, "the VAE decoder runs on one GPU")
        out.update(latents_per_call=args.chunk)
    return out


def timing_kwargs(args, component):
    kw = dict(TIMING[component])
    for key in kw:
        value = getattr(args, key)
        if value is not None:
            kw[key] = value
    return kw


def profile_dit(args, ctx, record):
    defaults = MODEL_DEFAULTS[args.model]
    timestep = defaults["timestep"] if args.timestep is None else args.timestep
    tuned = tuned_split_kv(args.model, args.resolution, ctx.world)
    split = args.split_kv or (tuned if args.mode == "optimized" else 1)
    compile_mode = args.compile_mode or ("max-autotune-no-cudagraphs" if args.mode == "optimized" else "default")
    compile_mode = None if compile_mode == "none" else compile_mode
    if args.mode == "optimized":
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True  # keep eager BF16 cast boundaries under fusion
    model = load_dit(args.model, args.dit, ctx.device)
    geometry = Geometry.from_resolution(args.resolution, args.frames)
    dit = FrameDiT(model, geometry, mode=args.mode, chunk=args.chunk, split_kv=split, rank=ctx.rank,
                   world=ctx.world, group_name=ctx.group_name, compile_mode=compile_mode,
                   cuda_graph=not args.no_cuda_graph)
    dit.fill_cache(seed=args.seed + 1)
    dit.set_inputs(timestep=timestep, seed=args.seed)
    started = time.perf_counter()
    dit.prepare(barrier=ctx.barrier)
    record["prepare_seconds"] = time.perf_counter() - started
    out = dit()
    torch.cuda.synchronize()
    record["settings"] = dict(mode=args.mode, timestep=timestep, chunk=args.chunk, split_kv=split,
                              split_kv_is_tuned_default=split == tuned,
                              compile_mode=compile_mode, cuda_graph=dit.graph is not None,
                              query_tokens=dit.tokens, query_tokens_per_rank=dit.local, kv_len=geometry.kv_len,
                              weights=args.dit or "random", output_finite=bool(torch.isfinite(out).all()))
    return dit


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(plan(args, int(os.environ.get("WORLD_SIZE", "1")))))
        return
    ctx = Dist()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    record = {"component": args.component, "model": args.model, "resolution": args.resolution,
              "gpus": ctx.world, "unit": "ms per call"}
    if args.component == "dit":
        fn = profile_dit(args, ctx, record)
    elif args.component == "kv-refine":
        from kv_refine import profile_kv_refine
        fn = profile_kv_refine(args, ctx, record)
    elif args.component == "pixel-decoder":
        from decoders import profile_decoder
        fn = profile_decoder(args, ctx, record)
    else:
        from decoders import profile_vae
        record.update(profile_vae(args, ctx))
        fn = None
    if fn is not None:
        record["timing"] = measure(fn, ctx, **timing_kwargs(args, args.component))
        record["latency_ms"] = record["timing"]["latency_ms"]
        torch.cuda.synchronize()
        if getattr(fn, "graph", None) is not None:
            fn.graph.reset()  # release graphs that captured NCCL work before the process group
        del fn
    record["environment"] = environment(ctx)
    if ctx.is_main:
        write_json(args.output, record)
        print(f"{args.component} {args.model} {args.resolution} gpus={ctx.world} "
              f"{record.get('settings', {}).get('mode', '')}: {record['latency_ms']:.3f} ms", flush=True)
    ctx.close()


if __name__ == "__main__":
    main()
