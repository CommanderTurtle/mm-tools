#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FastFrames-L stage 0: ODE trajectories of the bidirectional Wan2.1 teacher.

Prompt ``i`` of ``data.prompts`` is integrated from Gaussian noise seeded with
``generation.seed + i`` by ``generation.steps`` Euler steps of the shifted flow
schedule under classifier-free guidance. The latents *before* the steps listed in
``generation.stored_steps`` are kept; a last entry equal to ``steps`` is the final
clean latent, which is what the later stages train on.

Corpus layout (``output_dir``)::

    corpus.json                  generation settings and the list of shards
    shard_00000.safetensors      "trajectories": BF16 [n, len(stored_steps), F, 16, H, W]
    shard_00000.json             [{"sample_id", "seed", "prompt"}, ...] for its n samples

Shard ``k`` holds samples ``[k * shard_size, (k + 1) * shard_size)``. Shards are
dealt round-robin to the ranks; finished samples wait in ``partial/`` until their
shard is packed, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..checkpoints import load_wan_base
from ..wan.scheduler import FlowMatchScheduler
from ..wan.text_encoder import TextEncoder
from .data import read_prompts
from .distributed import init_distributed, raise_if_any_failed

__all__ = ["OdeCorpus", "sample_trajectory", "train"]


@torch.no_grad()
def sample_trajectory(teacher, cond, uncond, seed, *, steps, shift, guidance_scale, stored_steps, latent_shape,
                      device):
    """One CFG Euler trajectory ``[len(stored_steps), F, 16, H, W]`` (BF16) of the bidirectional teacher.

    Guidance is applied to the x0 predictions. The teacher's flow is turned into x0
    with the sigma of the nearest 1000-step training timestep and back into a flow
    with the sampling sigma, exactly as in the original generator.
    """
    to_x0 = FlowMatchScheduler(shift=shift)
    sampler = FlowMatchScheduler(steps, shift=shift)
    num_frames, _, height, width = latent_shape
    seq_len = num_frames * height * width // 4
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, *latent_shape), generator=generator, device=device, dtype=torch.bfloat16)
    states = []
    for step, t in enumerate(sampler.timesteps):
        if step in stored_steps:
            states.append(latents[0].clone())
        timestep = torch.full((num_frames,), float(t), device=device)
        xt = latents.flatten(0, 1)
        x = latents.permute(0, 2, 1, 3, 4)
        x0_cond, x0_uncond = (
            to_x0.flow_to_x0(teacher(x, timestep[:1], context, seq_len).permute(0, 2, 1, 3, 4).flatten(0, 1),
                             xt, timestep)
            for context in (cond, uncond))
        x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        flow = sampler.x0_to_flow(x0, xt, timestep)
        latents = sampler.step(flow, timestep, xt).unflatten(0, latents.shape[:2]).to(torch.bfloat16)
    if steps in stored_steps:
        states.append(latents[0].clone())
    return torch.stack(states)


def _save_atomic(tensors, path):
    tmp = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


def _write_json_atomic(value, path):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def train(config):
    """Generate the corpus (the ``scripts/train.py`` entry point of this stage)."""
    gen = config.generation
    # Ranks finish their shards at different times; the only collective is the final one.
    ctx = init_distributed(timeout_minutes=7 * 24 * 60)
    out = Path(config.output_dir)
    (out / "partial").mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(config.data.prompts)[:config.data.get("num_samples")]
    num_shards = -(-len(prompts) // gen.shard_size)
    stored_steps = list(gen.stored_steps)
    error = None
    try:
        wan_dir = Path(config.model.checkpoint_root) / config.model.name
        teacher = load_wan_base(wan_dir, causal=False, dtype=torch.bfloat16, device=ctx.device)
        encoder = TextEncoder(wan_dir).to(ctx.device, torch.bfloat16)
        with torch.no_grad():
            uncond = encoder([gen.negative_prompt])["prompt_embeds"]
        for shard in range(ctx.rank, num_shards, ctx.world_size):
            name = f"shard_{shard:05d}"
            ids = range(shard * gen.shard_size, min((shard + 1) * gen.shard_size, len(prompts)))
            rows = out / f"{name}.json"
            if rows.exists() and len(json.loads(rows.read_text())) == len(ids):
                continue  # packed by an earlier run (a short last shard is redone if num_samples grew)
            trajectories = []
            for i in ids:
                partial = out / "partial" / f"{i:08d}.safetensors"
                if not partial.exists():
                    with torch.no_grad():
                        cond = encoder([prompts[i]])["prompt_embeds"]
                    trajectory = sample_trajectory(
                        teacher, cond, uncond, gen.seed + i, steps=gen.steps, shift=gen.shift,
                        guidance_scale=gen.guidance_scale, stored_steps=stored_steps,
                        latent_shape=tuple(gen.latent_shape), device=ctx.device)
                    _save_atomic({"trajectory": trajectory.cpu()}, partial)
                    if ctx.is_main:
                        print(f"[ode_data] sample {i} done", flush=True)
                trajectories.append(load_file(str(partial))["trajectory"])
            _save_atomic({"trajectories": torch.stack(trajectories)}, out / f"{name}.safetensors")
            _write_json_atomic([{"sample_id": i, "seed": gen.seed + i, "prompt": prompts[i]} for i in ids], rows)
            for i in ids:
                (out / "partial" / f"{i:08d}.safetensors").unlink()
    except Exception as exc:  # noqa: BLE001 - reported collectively
        error = exc
    raise_if_any_failed(error, "ODE trajectory generation")
    if ctx.is_main:
        _write_json_atomic({
            "teacher": config.model.name, "num_samples": len(prompts), "shard_size": gen.shard_size,
            "latent_shape": list(gen.latent_shape), "stored_steps": stored_steps, "steps": gen.steps,
            "shift": gen.shift, "guidance_scale": gen.guidance_scale, "negative_prompt": gen.negative_prompt,
            "seed": gen.seed, "shards": [f"shard_{k:05d}" for k in range(num_shards)],
        }, out / "corpus.json")
        if not any((out / "partial").iterdir()):
            (out / "partial").rmdir()
        print(f"[ode_data] wrote {len(prompts)} trajectories in {num_shards} shards to {out}", flush=True)


class OdeCorpus:
    """Read access to a corpus written by this stage (clean latents and prompts by sample id)."""

    def __init__(self, root, max_open=4):
        self.root = Path(root)
        self.info = json.loads((self.root / "corpus.json").read_text())
        if self.info["stored_steps"][-1] != self.info["steps"]:
            raise ValueError("corpus does not store the final clean latent")
        self.clean_index = len(self.info["stored_steps"]) - 1
        self.max_open = max_open
        self._open = OrderedDict()

    def __len__(self):
        return self.info["num_samples"]

    def _shard(self, sample_id):
        shard, offset = divmod(sample_id, self.info["shard_size"])
        name = self.info["shards"][shard]
        if name not in self._open:
            handle = safe_open(str(self.root / f"{name}.safetensors"), framework="pt", device="cpu")
            rows = json.loads((self.root / f"{name}.json").read_text())
            self._open[name] = (handle, rows)
            while len(self._open) > self.max_open:
                self._open.popitem(last=False)
        self._open.move_to_end(name)
        return self._open[name], offset

    def clean(self, sample_id):
        """Final (clean) latent ``[F, 16, H, W]`` in BF16; only these bytes are read."""
        (handle, _), offset = self._shard(sample_id)
        return handle.get_slice("trajectories")[offset:offset + 1, self.clean_index].squeeze(0)

    def prompt(self, sample_id):
        (_, rows), offset = self._shard(sample_id)
        return rows[offset]["prompt"]
