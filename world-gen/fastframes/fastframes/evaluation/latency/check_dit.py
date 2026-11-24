#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Numerical agreement of the profiled DiT paths with ``CausalWanModel.forward_frame``.

The KV history of frames ``0 .. F-1`` is produced by the released model itself
(``forward_frame`` on random latents), so the check runs on realistic cache
statistics. Frame ``F`` is then evaluated by

* ``reference``: ``forward_frame(fast=None)`` (Conv3d patch, complex RoPE);
* ``reference_fastops``: ``forward_frame(fast=FastOps)`` (the release's fused QKV / FP32 RoPE ops);
* the profiled ``optimized`` path: eager without split-KV, eager with split-KV, and the
  compiled + CUDA-graph forward that is actually timed (context parallel when launched
  with ``torchrun``);
* the profiled ``baseline`` path (compiled).

Reported per output: max |diff|, relative L2, cosine, bitwise equality, for the flow
prediction, the pre-head features and the live K/V written into the last layer's cache.

    python check_dit.py --model 1.3b --dit CKPT/causal-forcing++/framewise-2step.pt --output check.json
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from dit import FrameDiT, Geometry
from models import MODEL_DEFAULTS, load_dit, tuned_split_kv
from timing import Dist, compare, environment, write_json

from fastframes.wan.causal_model import FastOps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODEL_DEFAULTS), default="1.3b")
    p.add_argument("--dit", help="DiT weights (.safetensors or a .pt with generator_ema); random when omitted")
    p.add_argument("--resolution", choices=("480p", "720p"), default="480p")
    p.add_argument("--frames", type=int, default=21, help="latent frames in the cache (F + 1)")
    p.add_argument("--timestep", type=float)
    p.add_argument("--split-kv", type=int)
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--fp32-reference", action="store_true",
                   help="also evaluate the frame with an FP32 copy of the model (error floor of BF16 paths)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ctx = Dist()
    torch.set_grad_enabled(False)
    defaults = MODEL_DEFAULTS[args.model]
    timestep = defaults["timestep"] if args.timestep is None else args.timestep
    split = args.split_kv or tuned_split_kv(args.model, args.resolution, ctx.world)
    import torch._inductor.config as inductor_config
    inductor_config.emulate_precision_casts = True

    model = load_dit(args.model, args.dit, ctx.device)
    geometry = Geometry.from_resolution(args.resolution, args.frames)
    frame = geometry.frames - 1
    tokens = geometry.tokens
    live = slice(frame * tokens, (frame + 1) * tokens)

    gen = torch.Generator(device=ctx.device).manual_seed(args.seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=ctx.device, dtype=torch.bfloat16)

    prompt = randn(1, model.text_len, model.config.text_dim)
    latents = randn(1, geometry.frames, 16, geometry.height, geometry.width)
    text = model.text_cache(prompt)

    # Realistic history: frames < F written by the reference model at t = 0 (clean context).
    dit = FrameDiT(model, geometry, mode="optimized", split_kv=split, rank=ctx.rank, world=ctx.world,
                   group_name=ctx.group_name, compile_mode=args.compile_mode)
    cache = dit.kv_cache
    t0 = torch.zeros(1, 1, device=ctx.device)
    e0_clean, eh_clean = model.time_conditioning(t0, torch.bfloat16)
    for f in range(frame):
        model.forward_frame(latents[:, f:f + 1], e0_clean, eh_clean, text, cache, f)
    history = [(c["k"][:, :live.start].clone(), c["v"][:, :live.start].clone()) for c in cache]

    def restore():
        for c, (k, v) in zip(cache, history):
            c["k"][:, :live.start].copy_(k)
            c["v"][:, :live.start].copy_(v)
            c["k"][:, live].fill_(float("nan"))  # every path must write its own live slot
            c["v"][:, live].fill_(float("nan"))

    def live_kv():
        return cache[-1]["k"][:, live].clone(), cache[-1]["v"][:, live].clone()

    x_live = latents[:, frame:frame + 1]
    t = torch.full((1, 1), timestep, device=ctx.device)
    e0, eh = model.time_conditioning(t, torch.bfloat16)
    outputs = {}

    restore()
    flow, feats = model.forward_frame(x_live, e0, eh, text, cache, frame)
    outputs["reference"] = (flow.clone(), feats.clone(), *live_kv())
    restore()
    fast = FastOps(model, geometry.height, geometry.width, geometry.frames)
    flow, feats = model.forward_frame(x_live, e0, eh, text, cache, frame, fast=fast)
    outputs["reference_fastops"] = (flow.clone(), feats.clone(), *live_kv())

    dit.set_inputs(latent=x_live, timestep=timestep, prompt=prompt)
    for name, s in (("optimized_eager_split1", 1), (f"optimized_eager_split{split}", split)):
        restore()
        dit.split_kv = s
        flow = dit.eager()
        feats = _gather_features(dit, ctx)
        outputs[name] = (flow.clone(), feats, *live_kv())
    dit.split_kv = split
    restore()
    dit.prepare(barrier=ctx.barrier)
    restore()
    dit()
    torch.cuda.synchronize()
    outputs[f"optimized_graph_split{split}_world{ctx.world}"] = (dit.output.clone(), _gather_features(dit, ctx),
                                                                 *live_kv())
    # Changed input must change the captured output (the graph reads the static buffers).
    dit.latent.copy_(randn(*x_live.shape))
    dit()
    torch.cuda.synchronize()
    changed = compare(dit.output, outputs[f"optimized_graph_split{split}_world{ctx.world}"][0])
    del dit

    if not args.skip_baseline and ctx.world == 1:
        base = FrameDiT(model, geometry, mode="baseline")
        base.kv_cache = cache
        base.set_inputs(latent=x_live, timestep=timestep, prompt=prompt)
        restore()
        flow = base.eager()
        outputs["baseline_eager"] = (flow.clone(), base.features.clone(), *live_kv())
        restore()
        base.prepare()
        restore()
        flow = base()
        outputs["baseline_compiled"] = (flow.clone(), None, *live_kv())
        del base

    ref_names = ["reference", "reference_fastops"]
    if args.fp32_reference and ctx.world == 1:
        model32 = copy.deepcopy(model).float()
        cache32 = [{"k": c["k"].float(), "v": c["v"].float()} for c in cache]
        for c, (k, v) in zip(cache32, history):
            c["k"][:, :live.start].copy_(k)
            c["v"][:, :live.start].copy_(v)
        e0_32, eh_32 = model32.time_conditioning(t, torch.float32)
        with fp32_attention():
            text32 = model32.text_cache(prompt.float())
            flow, feats = model32.forward_frame(x_live.float(), e0_32, eh_32, text32, cache32, frame)
        outputs["reference_fp32"] = (flow, feats, cache32[-1]["k"][:, live].clone(), cache32[-1]["v"][:, live].clone())
        ref_names.append("reference_fp32")
        del model32, cache32

    names = ("flow", "features", "live_key_last_layer", "live_value_last_layer")
    results = {}
    for name, tensors in outputs.items():
        results[name] = {}
        for ref_name in ref_names:
            ref = outputs[ref_name]
            results[name][f"vs_{ref_name}"] = {
                n: compare(a, b) for n, a, b in zip(names, tensors, ref) if a is not None}
    record = {"model": args.model, "resolution": args.resolution, "frame": frame, "kv_len": geometry.kv_len,
              "timestep": timestep, "split_kv": split, "world": ctx.world, "weights": args.dit or "random",
              "history": "frames < F written by forward_frame at t=0 on random latents",
              "changed_input_vs_original_graph_output": changed,
              "results": results, "environment": environment(ctx)}
    if ctx.is_main:
        write_json(args.output, record)
        for name, r in results.items():
            v = r["vs_reference"]
            print(f"{name:34s} " + "  ".join(
                f"{k}: rel_l2={m['relative_l2']:.2e} max_abs={m['max_abs_diff']:.2e} cos={m['cosine']:.7f}"
                for k, m in v.items() if k in ("flow", "live_key_last_layer")), flush=True)
    ctx.close()


@contextmanager
def fp32_attention():
    """Run every ``fastframes`` attention call in FP32 (the default kernel casts to BF16)."""
    import fastframes.wan.causal_model as causal_model
    import fastframes.wan.model as wan_model

    def attention(q, k, v, causal=False, dropout_p=0.0):
        out = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                             v.transpose(1, 2).float(), is_causal=causal)
        return out.transpose(1, 2).contiguous().to(q.dtype)
    saved = causal_model.attention, wan_model.attention, wan_model.cross_attention
    causal_model.attention = wan_model.attention = wan_model.cross_attention = attention
    try:
        yield
    finally:
        causal_model.attention, wan_model.attention, wan_model.cross_attention = saved


def _gather_features(dit, ctx):
    """Full pre-head features ``[1, tokens, C]`` (all-gathered in context-parallel runs)."""
    x = dit.features.clone()
    if ctx.world > 1:
        import torch.distributed as dist
        full = x.new_empty(ctx.world, *x.shape[1:])
        dist.all_gather_into_tensor(full, x[0].contiguous())
        x = full.reshape(1, -1, x.shape[-1])
    return x


if __name__ == "__main__":
    main()
