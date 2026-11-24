#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Checkpoint loading for Wan backbones, FastFrames generators and their add-on networks."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .wan.causal_model import CausalWanModel
from .wan.model import WAN_CONFIGS, WanModel

__all__ = ["load_state_dict", "build_wan", "load_wan_base"]

_PREFIXES = ("model._fsdp_wrapped_module.", "_fsdp_wrapped_module.", "model.")


def load_state_dict(path, key=None):
    """Tensor state from ``.safetensors`` (a file, or the ``.safetensors.index.json`` of a sharded
    file) or a torch pickle (optionally nested under ``key``)."""
    path = Path(path)
    if path.name.endswith(".safetensors.index.json"):
        from safetensors.torch import load_file
        state = {}
        for name in sorted(set(json.loads(path.read_text())["weight_map"].values())):
            state.update(load_file(str(path.parent / name)))
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if key is not None:
            state = state[key]
    return {_strip(k): v for k, v in state.items()}


def _strip(name):
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def build_wan(name, state_dict, *, causal=True, dtype=torch.bfloat16, device="cuda"):
    """Instantiate a (causal) Wan2.1 transformer without materialising random weights."""
    cls = CausalWanModel if causal else WanModel
    with torch.device("meta"):
        model = cls(WAN_CONFIGS[name])
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_wan_base(wan_dir, *, causal=False, dtype=torch.bfloat16, device="cuda"):
    """Load the released Wan2.1 weights (``diffusion_pytorch_model*.safetensors``)."""
    from safetensors.torch import load_file
    wan_dir = Path(wan_dir)
    index = wan_dir / "diffusion_pytorch_model.safetensors.index.json"
    files = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists()
             else ["diffusion_pytorch_model.safetensors"])
    state = {}
    for name in files:
        state.update(load_file(str(wan_dir / name)))
    config = json.loads((wan_dir / "config.json").read_text())
    name = "Wan2.1-T2V-14B" if config["dim"] == 5120 else "Wan2.1-T2V-1.3B"
    return build_wan(name, state, causal=causal, dtype=dtype, device=device)
