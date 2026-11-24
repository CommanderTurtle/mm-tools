#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Model construction for the latency profilers.

Loaders use random weights when ``path=None``. Supply checkpoints for profiling
with trained weights and for numerical comparisons.
"""
from __future__ import annotations

import timing  # noqa: F401  (puts the repository root on sys.path)
import torch

from fastframes.checkpoints import build_wan, load_state_dict
from fastframes.models.kv_refinement import KVRefinementNetwork
from fastframes.models.pixel_decoder import BridgedPixelDecoder, PixelDecoder
from fastframes.wan.causal_model import CausalWanModel
from fastframes.wan.model import WAN_CONFIGS

__all__ = ["MODEL_DEFAULTS", "tuned_split_kv", "load_dit", "load_refiner", "load_decoder"]

# Per-model profiling settings.
#   timestep : the final steady-state denoising timestep (actual model input)
#   split_kv : KV segments of the optimized attention, keyed by (resolution, GPUs); tuned on B200
#              720p 1.3B/8 GPUs and 14B/1 GPU use untuned defaults of 12 and 1.
#   refiner_autocast / contiguous_stem / decoder_dtype : numerics of the released recipe
#              (``fastframes.pipeline.FF_S`` / ``FF_L``)
MODEL_DEFAULTS = {
    "1.3b": {
        "wan": "Wan2.1-T2V-1.3B",
        "timestep": 833.3333129882812,
        "split_kv": {("480p", 1): 5, ("480p", 8): 12, ("720p", 1): 3, ("720p", 8): 12},
        "refiner_autocast": False,
        "contiguous_stem": True,
        "decoder_dtype": torch.float32,
    },
    "14b": {
        "wan": "Wan2.1-T2V-14B",
        "timestep": 833.0,
        "split_kv": {("480p", 1): 1, ("480p", 8): 10, ("720p", 1): 1, ("720p", 8): 5},
        "refiner_autocast": True,
        "contiguous_stem": False,
        "decoder_dtype": torch.bfloat16,
    },
}


def tuned_split_kv(model, resolution, gpus):
    """Configured split-KV count; unlisted GPU counts use the 1-GPU value."""
    table = MODEL_DEFAULTS[model]["split_kv"]
    return table.get((resolution, gpus), table[(resolution, 1)])


def load_dit(model, path=None, device="cuda"):
    """BF16 ``CausalWanModel`` (released weights when ``path`` is given)."""
    name = MODEL_DEFAULTS[model]["wan"]
    if path is not None:
        return build_wan(name, load_state_dict(path, "generator_ema"), device=device)
    torch.manual_seed(0)
    with torch.device(device):
        dit = CausalWanModel(WAN_CONFIGS[name])
    return dit.to(dtype=torch.bfloat16).eval().requires_grad_(False)


def load_refiner(model, path=None, device="cuda"):
    """BF16 ``KVRefinementNetwork`` of the given DiT size."""
    wan = WAN_CONFIGS[MODEL_DEFAULTS[model]["wan"]]
    state = load_state_dict(path) if path is not None else None
    d_a = state["stem_in.weight"].shape[0] if state is not None else 768
    torch.manual_seed(0)
    net = KVRefinementNetwork(dim=wan.dim, num_heads=wan.num_heads, head_dim=wan.dim // wan.num_heads,
                              num_blocks=wan.num_layers, d_a=d_a)
    if state is not None:
        net.load_state_dict(state, strict=True)
    else:  # zero-initialised residual branches would make the network trivially cheap to reason about
        for p in net.parameters():
            if p.dim() > 1 and p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.02)
    return net.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def load_decoder(model, path=None, device="cuda"):
    """The FastFrames pixel decoder (with the 5120->1536 bridge for 14B)."""
    defaults = MODEL_DEFAULTS[model]
    dim = WAN_CONFIGS[defaults["wan"]].dim
    torch.manual_seed(0)
    net = BridgedPixelDecoder(in_dim=dim) if model == "14b" else PixelDecoder(dim)
    dtype = defaults["decoder_dtype"]
    if path is not None:
        state = load_state_dict(path)
        if isinstance(net, PixelDecoder):
            state = {k.removeprefix("pixel_head."): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
    else:
        for p in net.parameters():
            if p.dim() > 1 and p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.02)
    return net.to(device=device, dtype=dtype).eval().requires_grad_(False)
