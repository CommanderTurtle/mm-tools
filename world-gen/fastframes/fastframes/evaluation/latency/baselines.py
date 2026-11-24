#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Profile one upstream Self-Forcing or Causal Forcing++ DiT call with a full KV cache.

Uses cuDNN SDPA and torch.compile without CUDA graphs. ``--chunk`` selects the
generation unit: 1 latent frame for CF++, 3 for Self-Forcing. JSON records share
the component profiler's schema with ``settings.mode = "upstream"``.
The summarizer also uses Self-Forcing's chunk timing for CausVid.
See docs/inference.md#profiling for setup and commands.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from timing import Dist, compare, environment, measure, write_json
from torch.nn.attention import SDPBackend, sdpa_kernel

WAN_1P3B = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=1536, ffn_dim=8960,
                freq_dim=256, text_dim=4096, out_dim=16, num_heads=12, num_layers=30, qk_norm=True,
                cross_attn_norm=True, eps=1e-6)
FRAME_TOKENS = {"480p": (60, 104, 1560)}


def cudnn_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                    causal=False, window_size=(-1, -1), **kwargs):
    """Drop-in for the upstream ``attention`` ([B, L, H, D] layout) on cuDNN SDPA."""
    if q_scale is not None:
        q = q * q_scale
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().type_as(q)


def load_upstream(path):
    sys.path.insert(0, str(path))
    import wan.modules.causal_model as causal_model
    import wan.modules.model as wan_model
    causal_model.attention = cudnn_attention
    wan_model.flash_attention = cudnn_attention
    return causal_model.CausalWanModel


def build(args, device):
    CausalWanModel = load_upstream(args.upstream)
    torch.manual_seed(args.seed)
    model = CausalWanModel(**WAN_1P3B, local_attn_size=-1, sink_size=0)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        state = state.get("generator_ema", state.get("generator", state))
        state = {k.replace("model._fsdp_wrapped_module.", "", 1).removeprefix("model."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    else:  # Wan zero-initialises the output head; randomise it so the numerical check is informative
        for param in model.parameters():
            if param.dim() > 1 and not param.any():
                torch.nn.init.normal_(param, std=0.02)
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", required=True, help="Self-Forcing or Causal-Forcing checkout")
    p.add_argument("--checkpoint", help="generator checkpoint (.pt); random weights when omitted")
    p.add_argument("--chunk", type=int, default=1, help="latent frames per generation unit")
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--timestep", type=float, default=833.3333129882812)
    p.add_argument("--compile", choices=("default", "none"), default="default")
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--batch-ms", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ctx = Dist()
    if ctx.world != 1:
        raise ValueError("baselines run on one GPU")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._inductor.config.triton.cudagraphs = False
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
    device = ctx.device
    model = build(args, device)
    h, w, frame_tokens = FRAME_TOKENS["480p"]
    kv_len = args.frames * frame_tokens
    start = (args.frames - args.chunk) * frame_tokens
    heads, head_dim = WAN_1P3B["num_heads"], WAN_1P3B["dim"] // WAN_1P3B["num_heads"]
    g = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape):
        return torch.randn(shape, generator=g, device=device, dtype=torch.bfloat16)
    # History of ``frames - chunk`` latent frames already written; the call writes the last chunk.
    kv_cache = [{"k": randn(1, kv_len, heads, head_dim), "v": randn(1, kv_len, heads, head_dim),
                 "global_end_index": torch.tensor([start], dtype=torch.long, device=device),
                 "local_end_index": torch.tensor([start], dtype=torch.long, device=device)}
                for _ in range(WAN_1P3B["num_layers"])]
    cross_cache = [{"k": torch.zeros(1, 512, heads, head_dim, device=device, dtype=torch.bfloat16),
                    "v": torch.zeros(1, 512, heads, head_dim, device=device, dtype=torch.bfloat16),
                    "is_init": False} for _ in range(WAN_1P3B["num_layers"])]
    x = randn(1, 16, args.chunk, h, w)
    t = torch.full((1, args.chunk), args.timestep, device=device)
    context = randn(1, 512, WAN_1P3B["text_dim"])
    kwargs = dict(t=t, context=context, seq_len=kv_len, kv_cache=kv_cache, crossattn_cache=cross_cache,
                  current_start=start)

    eager = model.forward
    ref = eager(x, **kwargs).clone()  # also initialises the cross-attention cache
    # Repeated calls rewrite the same cache slot (global/local end index stay at the cache end).
    fn = torch.compile(eager, mode="default", dynamic=False) if args.compile == "default" else eager
    started = time.perf_counter()
    got = fn(x, **kwargs).clone()
    torch.cuda.synchronize()
    record = {"component": "dit", "model": "1.3b", "resolution": "480p", "gpus": 1, "unit": "ms per call",
              "prepare_seconds": time.perf_counter() - started,
              "settings": dict(mode="upstream", upstream=str(args.upstream), chunk=args.chunk,
                               timestep=args.timestep, compile=args.compile, cuda_graph=False,
                               query_tokens=args.chunk * frame_tokens, kv_len=kv_len,
                               weights=args.checkpoint or "random"),
              "check": {"compiled_vs_eager": compare(got, ref),
                        "repeat_eager": compare(eager(x, **kwargs), ref)}}
    record["timing"] = measure(lambda: fn(x, **kwargs), ctx, warmup=args.warmup, iters=args.iters,
                               seconds=args.seconds, batch_ms=args.batch_ms)
    record["latency_ms"] = record["timing"]["latency_ms"]
    record["environment"] = environment(ctx)
    write_json(args.output, record)
    print(f"upstream dit chunk={args.chunk}: {record['latency_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
