#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Loading the upstream LingBot-World model.

Imports a local clone at ``UPSTREAM_COMMIT``, selected by ``--lingbot-root`` or
``LINGBOT_WORLD_ROOT``. See docs/installation.md#lingbot for source and weight setup.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = ["UPSTREAM_URL", "UPSTREAM_COMMIT", "import_upstream", "Upstream", "load_upstream"]

UPSTREAM_URL = "https://github.com/robbyant/lingbot-world-v2"
UPSTREAM_COMMIT = "1895d300d8ac936401689b26389f51cbd36530eb"
ROOT_ENV = "LINGBOT_WORLD_ROOT"


def _git_head(root: Path):
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def import_upstream(root=None):
    """Import the upstream ``wan`` package from a LingBot-World clone."""
    root = Path(root or os.environ.get(ROOT_ENV, "")).expanduser()
    if not (root / "wan" / "modules" / "model_fast.py").is_file():
        raise FileNotFoundError(
            f"no LingBot-World clone at '{root}'; pass --lingbot-root or set {ROOT_ENV}")
    root = root.resolve()
    loaded = sys.modules.get("wan")
    if loaded is not None:
        if root not in Path(loaded.__file__).resolve().parents:
            raise RuntimeError(f"another 'wan' package is already imported from {loaded.__file__}")
        return loaded
    head = _git_head(root)
    if head is not None and head != UPSTREAM_COMMIT:
        warnings.warn(f"LingBot-World clone is at {head}; FastFrames-LingBot was verified at {UPSTREAM_COMMIT}")
    sys.path.insert(0, str(root))
    return importlib.import_module("wan")


@dataclass
class Upstream:
    """Official components: causal-fast DiT, umT5 text encoder, Wan2.1 VAE, flow scheduler."""

    config: object
    dit: torch.nn.Module
    text_encoder: object
    vae: object
    scheduler: object


def _weight_files(dit_dir: Path):
    for index in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        if (dit_dir / index).is_file():
            weight_map = json.loads((dit_dir / index).read_text())["weight_map"]
            return [dit_dir / name for name in sorted(set(weight_map.values()))]
    for name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        if (dit_dir / name).is_file():
            return [dit_dir / name]
    raise FileNotFoundError(f"no DiT safetensors in {dit_dir}")


def load_dit(config, ckpt_dir, *, fp32_time_weights: bool, device="cuda"):
    """Upstream ``WanModelFast`` with BF16 weights and an FP32 time MLP.

    ``fp32_time_weights`` keeps the time MLP weights of the FP32 checkpoint; otherwise they are
    rounded to BF16 like all other weights (the upstream loader) and evaluated in FP32.
    """
    from safetensors import safe_open
    from wan.image2video import _dit_kwargs_from_config, _resolve_dit_dir
    from wan.modules.model_fast import WanModelFast

    with torch.device("meta"):
        model = WanModelFast(**_dit_kwargs_from_config(config, dict(local_attn_size=18, sink_size=6)))
    state = {}
    for path in _weight_files(Path(_resolve_dit_dir(str(ckpt_dir), config.fast_checkpoint))):
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            for name in shard.keys():
                tensor = shard.get_tensor(name)
                if name.startswith(("time_embedding.", "time_projection.")):
                    state[name] = tensor.float() if fp32_time_weights else tensor.to(torch.bfloat16).float()
                else:
                    state[name] = tensor.to(torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device)


def load_upstream(config_name, ckpt_dir, assets_dir=None, *, fp32_time_weights, device="cuda"):
    """Build the official components; T5/VAE files are looked up in ``ckpt_dir`` then ``assets_dir``."""
    from wan.configs import WAN_CONFIGS
    from wan.image2video import _resolve_asset_path
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    config = WAN_CONFIGS[config_name]
    ckpt_dir, assets_dir = str(ckpt_dir), None if assets_dir is None else str(assets_dir)
    text_encoder = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=torch.device("cpu"),
        checkpoint_path=_resolve_asset_path(config.t5_checkpoint, ckpt_dir, assets_dir),
        tokenizer_path=_resolve_asset_path(config.t5_tokenizer, ckpt_dir, assets_dir))
    vae = Wan2_1_VAE(vae_pth=_resolve_asset_path(config.vae_checkpoint, ckpt_dir, assets_dir), device=device)
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1,
                                            use_dynamic_shifting=False)
    dit = load_dit(config, ckpt_dir, fp32_time_weights=fp32_time_weights, device=device)
    return Upstream(config, dit, text_encoder, vae, scheduler)
