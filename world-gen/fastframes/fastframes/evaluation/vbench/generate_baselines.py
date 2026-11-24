#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generate VBench videos with Self-Forcing, Causal Forcing++ or CausVid.

Imports the baseline from ``--upstream`` and generates one video per call with
cuDNN SDPA. Self-Forcing and CF++ use FP32 weights under BF16 autocast and a BF16
VAE; CausVid uses BF16. Sampling follows ``evaluation/vbench/protocol.py``;
valid existing videos are skipped. See docs/inference.md#evaluation for setup and commands.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import random
import subprocess
import sys
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol  # noqa: E402


@dataclass(frozen=True)
class Baseline:
    url: str
    commit: str  # the upstream commit the generation was verified at
    package: str  # top-level package of the upstream code
    configs: tuple[str, ...]  # merged in order, relative to the clone
    payload: str  # checkpoint entry holding the generator weights
    protocol: str
    bf16: bool  # whole pipeline in BF16 (else FP32 weights under BF16 autocast, BF16 VAE)


BASELINES = {
    "self-forcing": Baseline("https://github.com/guandeh17/Self-Forcing", "33593df3e81fa3ec10239271dd2c100facac6de1",
                             "wan", ("configs/default_config.yaml", "configs/self_forcing_dmd.yaml"),
                             "generator_ema", protocol.EXTENDED_PROTOCOL, False),
    "cfpp": Baseline("https://github.com/thu-ml/Causal-Forcing", "730d305eea6eca5f194f42a715dd58eec606c8c4", "wan",
                     ("configs/default_config.yaml", "configs/causal_forcing_dmd_framewise_2step.yaml"),
                     "generator_ema", protocol.EXTENDED_PROTOCOL, False),
    "causvid": Baseline("https://github.com/tianweiy/CausVid", "adb6a5ecd07666b4d0290042915c8406e6d5ce22", "causvid",
                        ("configs/wan_causal_dmd.yaml",),
                        "generator", protocol.CAUSVID_PROTOCOL, True),
}
NOISE_SHAPE = (1, protocol.EXPECTED_LATENT_FRAMES, 16, 60, 104)


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                    causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """Wan's ``flash_attention`` ([B, L, H, D]) implemented with cuDNN SDPA.

    Sequence lengths are ignored; every call in these pipelines uses full-length sequences.
    """
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)
    q, k, v = half(q.flatten(0, 1)), half(k.flatten(0, 1)), half(v.flatten(0, 1))
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        x = F.scaled_dot_product_attention(
            q.unflatten(0, (b, lq)).transpose(1, 2), k.unflatten(0, (b, lk)).transpose(1, 2),
            v.unflatten(0, (b, lk)).transpose(1, 2), dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return x.transpose(1, 2).contiguous().type(out_dtype)


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, fa_version=None):
    """Wan's ``attention`` without flash-attn (its SDPA branch), on the cuDNN backend; returns ``dtype``."""
    q, k, v = (x.transpose(1, 2).to(dtype) for x in (q, k, v))
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous()


def use_cudnn_attention(package):
    """Point every imported reference to Wan's attention helpers at the versions above."""
    replaced = 0
    for name, module in list(sys.modules.items()):
        if module is None or not (name == package or name.startswith(package + ".")):
            continue
        for attr, shim in (("flash_attention", flash_attention), ("attention", attention)):
            fn = getattr(module, attr, None)
            if callable(fn) and getattr(fn, "__module__", "").endswith("modules.attention"):
                setattr(module, attr, shim)
                replaced += 1
    if not replaced:
        raise RuntimeError(f"no Wan attention helpers found in the imported '{package}' modules")


def stub_sdxl():
    """``causvid.models`` imports its SDXL wrapper, which fails with current transformers; Wan never uses it."""
    stub = types.ModuleType("causvid.models.sdxl.sdxl_wrapper")
    for name in ("SDXLWrapper", "SDXLTextEncoder", "SDXLVAE"):
        setattr(stub, name, type(name, (), {}))
    sys.modules[stub.__name__] = stub


def git_head(root):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def load_pipeline(spec, upstream, checkpoint, device):
    from omegaconf import OmegaConf

    head = git_head(upstream)
    if head != spec.commit:
        warnings.warn(f"{upstream} is at {head}; this baseline was verified at {spec.commit}")
    os.chdir(upstream)  # the upstream code opens wan_models/... relative to the clone
    sys.path.insert(0, str(upstream))
    config = OmegaConf.merge(*(OmegaConf.load(name) for name in spec.configs))
    if spec.package == "causvid":
        stub_sdxl()
        from causvid.models.wan.causal_inference import InferencePipeline
        pipe = InferencePipeline(config, device=device).to(device=device, dtype=torch.bfloat16)
    else:
        from pipeline import CausalInferencePipeline
        pipe = CausalInferencePipeline(config, device=device)
    use_cudnn_attention(spec.package)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)[spec.payload]
    for prefix in ("model._fsdp_wrapped_module.", "model._checkpoint_wrapped_module.", "model._orig_mod."):
        state = {"model." + k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    pipe.generator.load_state_dict(state, strict=True)
    del state
    if not spec.bf16:
        pipe.generator.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device=device, dtype=torch.bfloat16)
    return pipe.eval().requires_grad_(False)


def generate(pipe, spec, job, device):
    """uint8 video ``[T, H, W, 3]`` of one job."""
    if not spec.bf16:  # upstream re-allocates these caches when they are None
        pipe.kv_cache1 = pipe.crossattn_cache = None
        pipe.vae.model.clear_cache()
    random.seed(job.seed)
    np.random.seed(job.seed)
    torch.manual_seed(job.seed)
    noise = torch.randn(NOISE_SHAPE, device=device, dtype=torch.bfloat16)
    precision = (contextlib.nullcontext() if spec.bf16 else
                 torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False))
    with torch.inference_mode(), precision:
        video = pipe.inference(noise=noise, text_prompts=[job.conditioning_prompt])
    frames = (video[0].float().clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    expected = (protocol.EXPECTED_RGB_FRAMES, protocol.EXPECTED_HEIGHT, protocol.EXPECTED_WIDTH, 3)
    if tuple(frames.shape) != expected:
        raise RuntimeError(f"video shape {tuple(frames.shape)} != {expected}")
    return frames.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(BASELINES), required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="clone of the baseline's repository")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the baseline's released generator (.pt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="generate at most this many videos of the shard")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    spec = BASELINES[args.model]
    output_dir, checkpoint, upstream = (p.resolve() for p in (args.output_dir, args.checkpoint, args.upstream))

    protocol.require_encoder()
    jobs = protocol.build_sample_specs(protocol.load_suite(spec.protocol))[args.shard::args.num_shards]
    pending = [job for job in jobs[:args.limit] if not protocol.is_valid_video(output_dir / job.filename)]
    print(f"{args.model} shard {args.shard}/{args.num_shards}: {len(jobs)} jobs, {len(pending)} to generate",
          flush=True)
    if not pending:
        return
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    pipe = load_pipeline(spec, upstream, checkpoint, device)
    for done, job in enumerate(pending, 1):
        start = time.perf_counter()
        protocol.write_faststart_video(output_dir / job.filename, generate(pipe, spec, job, device))
        print(f"{done}/{len(pending)}  {job.filename}  seed={job.seed}  {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
