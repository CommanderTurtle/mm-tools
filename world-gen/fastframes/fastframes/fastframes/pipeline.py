#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FastFrames text-to-video inference (Sec. 3 of the paper).

Each latent frame is denoised with a few-step causal DiT that attends to the
KV cache of all previous frames. After the frame's final step:

* the pixel decoder turns the DiT's pre-head hidden states into 4 RGB frames;
* the KV refinement network replaces the frame's noisy cache entries with
  clean-context K/V (frame 0 instead receives one clean DiT pass at ``t = 0``).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from .checkpoints import build_wan, load_state_dict
from .models.kv_refinement import KVRefinementNetwork
from .models.pixel_decoder import BridgedPixelDecoder, PixelDecoder, align_frames
from .wan.causal_model import FastOps, allocate_kv_cache
from .wan.model import WAN_CONFIGS
from .wan.scheduler import FlowMatchScheduler
from .wan.text_encoder import TextEncoder

__all__ = ["FastFramesConfig", "FF_S", "FF_L", "MODELS", "ModelSpec", "FastFrames", "CONTEXT_MODES",
           "configure_backends", "to_uint8"]

# How a finished frame's cache slot is written before the next frame (``FastFrames.rollout``).
CONTEXT_MODES = ("refiner", "dit")


@dataclass(frozen=True)
class FastFramesConfig:
    """Numerical recipe of a released model.

    ``noise="sequential"`` draws each transition noise from the seeded generator when
    it is needed; ``"pregenerated"`` draws all ``3 x latent`` transition noises right
    after the initial noise; ``"reseeded"`` draws them when needed from a second
    generator with the same seed (the convention of the qualitative figures).
    ``autocast`` runs text encoding, the DiT and the KV refinement network under
    BF16 autocast. ``fast_ops`` enables fused QKV, FP32
    real-valued RoPE and GEMM patch embedding. ``contiguous_stem`` selects the memory
    layout of the refinement stem tokens (it changes GEMM kernel selection, hence bits);
    ``allow_tf32`` is the process-wide TF32 matmul setting the model was evaluated with
    (see ``configure_backends``).
    """

    name: str
    wan_model: str
    first_frame_timesteps: Sequence[float]
    timesteps: Sequence[float]
    refine_timestep: float
    noise: str
    autocast: bool
    fast_ops: bool
    decoder_dtype: torch.dtype
    contiguous_stem: bool
    allow_tf32: bool
    num_frames: int = 21
    latent_size: tuple = (60, 104)
    num_pixel_frames: int = 81


# 1.3B: CF++ frame-wise 2-step generator; shifted timesteps of nominal [1000, 750, 500, 250].
FF_S = FastFramesConfig(
    name="FF-S", wan_model="Wan2.1-T2V-1.3B",
    first_frame_timesteps=(1000.0, 937.5, 833.3333129882812, 625.0),
    timesteps=(1000.0, 833.3333129882812), refine_timestep=833.3333129882812,
    noise="sequential", autocast=False, fast_ops=False, decoder_dtype=torch.float32,
    contiguous_stem=True, allow_tf32=False)

FF_L = FastFramesConfig(
    name="FF-L", wan_model="Wan2.1-T2V-14B",
    first_frame_timesteps=(1000.0, 937.0, 833.0, 625.0), timesteps=(1000.0, 833.0),
    refine_timestep=833.0, noise="pregenerated", autocast=True, fast_ops=True,
    decoder_dtype=torch.bfloat16, contiguous_stem=False, allow_tf32=True)


@dataclass(frozen=True)
class ModelSpec:
    """A released model: its recipe and file names inside the checkpoint directory."""

    config: FastFramesConfig
    wan_dir: str
    dit: str
    refiner: str
    decoder: str


MODELS = {
    "ff-s": ModelSpec(FF_S, "Wan2.1-T2V-1.3B", "causal-forcing++/framewise-2step.pt",
                      "fastframes-s/kv_refiner.safetensors", "fastframes-s/pixel_decoder.safetensors"),
    "ff-l": ModelSpec(FF_L, "Wan2.1-T2V-14B", "fastframes-l/dit.safetensors",
                      "fastframes-l/kv_refiner.safetensors", "fastframes-l/pixel_decoder.safetensors"),
}


def configure_backends(config: FastFramesConfig):
    """Apply the process-wide matmul precision a recipe was evaluated with."""
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = True


def to_uint8(pixels):
    """``[T, 3, H, W]`` in ``[-1, 1]`` -> ``[T, H, W, 3]`` uint8."""
    return ((pixels.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class FastFrames:

    def __init__(self, config: FastFramesConfig, dit, refiner, decoder, text_encoder=None):
        self.config = config
        self.dit, self.refiner, self.decoder, self.text_encoder = dit, refiner, decoder, text_encoder
        self.device = dit.patch_embedding.weight.device
        self.scheduler = FlowMatchScheduler(shift=5.0)
        height, width = config.latent_size
        self.frame_tokens = height * width // 4
        self.fast = FastOps(dit, height, width, config.num_frames) if config.fast_ops else None
        self._time = {}
        self._refine_embedding = None

    @classmethod
    def from_pretrained(cls, config, wan_dir, dit, refiner, decoder, *, device="cuda"):
        """Build from checkpoint paths (or already-loaded state dicts)."""
        def state(value, key=None):
            return value if isinstance(value, dict) else load_state_dict(value, key)

        dit = build_wan(config.wan_model, state(dit, "generator_ema"), device=device)
        wan = WAN_CONFIGS[config.wan_model]
        refiner_state = state(refiner, "adapter")
        refiner_net = KVRefinementNetwork(
            dim=wan.dim, num_heads=wan.num_heads, head_dim=wan.dim // wan.num_heads,
            num_blocks=wan.num_layers, d_a=refiner_state["stem_in.weight"].shape[0]).to(torch.bfloat16)
        refiner_net.load_state_dict(refiner_state, strict=True)
        decoder_state = state(decoder)
        if any(k.startswith("bridge.") for k in decoder_state):
            decoder_net = BridgedPixelDecoder(in_dim=wan.dim)
        else:
            decoder_net = PixelDecoder(wan.dim)
            decoder_state = {k.removeprefix("pixel_head."): v for k, v in decoder_state.items()}
        decoder_net.load_state_dict(decoder_state, strict=True)
        text_encoder = TextEncoder(Path(wan_dir)).to(device, dtype=torch.bfloat16)
        return cls(config, dit, refiner_net.to(device).eval().requires_grad_(False),
                   decoder_net.to(device, dtype=config.decoder_dtype).eval().requires_grad_(False),
                   text_encoder)

    def autocast(self):
        if self.config.autocast:
            return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ text / noise
    @torch.inference_mode()
    def encode_prompt(self, prompt: str):
        with self.autocast():
            return self.text_encoder([prompt])["prompt_embeds"]

    def _noise(self, seed):
        cfg = self.config
        shape = (1, cfg.num_frames, 16, *cfg.latent_size)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=torch.bfloat16)
        if cfg.noise == "pregenerated":
            transition = torch.randn((3, *shape), generator=generator, device=self.device, dtype=torch.bfloat16)
            return noise, lambda step, frame: transition[step, :, frame]
        if cfg.noise == "reseeded":
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        elif cfg.noise != "sequential":
            raise ValueError(f"unknown noise convention {cfg.noise!r}")
        frame_shape = (1, 16, *cfg.latent_size)
        return noise, lambda step, frame: torch.randn(
            frame_shape, generator=generator, device=self.device, dtype=torch.bfloat16)

    # ------------------------------------------------------------------ per-frame calls
    def _time_conditioning(self, timestep):
        if timestep not in self._time:
            t = torch.full((1, 1), timestep, device=self.device, dtype=torch.float32)
            self._time[timestep] = (t,) + self.dit.time_conditioning(t, torch.bfloat16)
        return self._time[timestep]

    def _predict(self, noisy, timestep, frame, text_cache, kv_cache):
        t, e0, e_head = self._time_conditioning(timestep)
        flow, features = self.dit.forward_frame(noisy, e0, e_head, text_cache, kv_cache, frame, fast=self.fast)
        x0 = self.scheduler.flow_to_x0(flow.flatten(0, 1), noisy.flatten(0, 1), t.flatten())
        return x0.unflatten(0, flow.shape[:2]), features, e_head

    def _refine(self, x0, frame, kv_cache):
        if self._refine_embedding is None:
            self._refine_embedding = self.refiner.embed_timestep(
                self.config.refine_timestep, 1, self.device, torch.bfloat16)
        emb = self._refine_embedding
        start, end = frame * self.frame_tokens, (frame + 1) * self.frame_tokens
        stem = self.dit.patch_embedding(x0.permute(0, 2, 1, 3, 4).contiguous())
        stem = stem.flatten(2).transpose(1, 2)
        if self.config.contiguous_stem:
            stem = stem.contiguous()
        phi = self.refiner.compute_phi(stem.to(torch.bfloat16), emb)
        keys = torch.stack([c["k"][:, start:end].reshape(self.frame_tokens, -1) for c in kv_cache])
        values = torch.stack([c["v"][:, start:end].reshape(self.frame_tokens, -1) for c in kv_cache])
        keys, values = self.refiner.repair_all(keys, values, phi, emb)
        for index in self.refiner.repaired_block_indices:
            slot = kv_cache[index]["k"][:, start:end]
            kv_cache[index]["k"][:, start:end].copy_(keys[index].reshape(slot.shape).to(slot.dtype))
            kv_cache[index]["v"][:, start:end].copy_(values[index].reshape(slot.shape).to(slot.dtype))

    # ------------------------------------------------------------------ rollout
    @torch.inference_mode()
    def rollout(self, prompt_embeds, seed, context="refiner"):
        """Latents ``[1, F, 16, h, w]``, decoder features ``[1, F*L, C]`` and modulation ``[1, F, 1, C]``.

        ``context`` selects what the next frames attend to for each finished frame:
        ``"refiner"`` (FastFrames) repairs the cache slot with the KV refinement network
        after frame 0's clean DiT pass at ``t = 0``; ``"dit"`` runs that clean pass for
        every frame (Causal Forcing++).
        """
        if context not in CONTEXT_MODES:
            raise ValueError(f"context must be one of {CONTEXT_MODES}")
        cfg = self.config
        with self.autocast():
            noise, transition = self._noise(seed)
            text_cache = self.dit.text_cache(prompt_embeds)
            kv_cache = allocate_kv_cache(self.dit, cfg.num_frames, self.frame_tokens, device=self.device)
            latents, features, modulation = torch.empty_like(noise), [], []
            for frame in range(cfg.num_frames):
                schedule = cfg.first_frame_timesteps if frame == 0 else cfg.timesteps
                x = noise[:, frame:frame + 1]
                for step, timestep in enumerate(schedule):
                    x0, feature, mod = self._predict(x, timestep, frame, text_cache, kv_cache)
                    if step + 1 < len(schedule):
                        next_t = torch.full((1,), schedule[step + 1], device=self.device, dtype=torch.float32)
                        eps = transition(step, frame)
                        x = self.scheduler.add_noise(x0.flatten(0, 1), eps, next_t).unflatten(0, x0.shape[:2])
                latents[:, frame:frame + 1].copy_(x0)
                features.append(feature)
                modulation.append(mod)
                if context == "dit" or frame == 0:
                    self._predict(x0, 0.0, frame, text_cache, kv_cache)
                elif context == "refiner":
                    self._refine(x0, frame, kv_cache)
        return latents, torch.cat(features, dim=1), torch.cat(modulation, dim=1)

    @torch.inference_mode()
    def decode(self, features, modulation):
        """Pixels ``[1, 81, 3, 480, 832]`` in ``[-1, 1]``."""
        height, width = self.config.latent_size
        grid = torch.tensor([[modulation.shape[1], height // 2, width // 2]], device=self.device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pixels = self.decoder(features, modulation, grid)
        return align_frames(pixels, self.config.num_pixel_frames)

    def generate(self, prompt: str, seed: int, prompt_embeds: Optional[torch.Tensor] = None):
        """Returns ``[81, 480, 832, 3]`` uint8 frames."""
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(prompt)
        _, features, modulation = self.rollout(prompt_embeds, seed)
        return to_uint8(self.decode(features, modulation)[0]).cpu()
