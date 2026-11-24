#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FastFrames-LingBot: real-time camera-controlled world generation (Fig. 5, App. E of the paper).

LingBot-World v2 continues a first image under a prompt and a camera path. Each step
generates a chunk of four latent frames (16 RGB frames, one second at 16 fps) with four
denoising steps against a rolling KV cache of 18 latent frames, the first 6 of which are
kept as attention sinks. FastFrames-LingBot keeps the official DiT and its schedule and

* replaces the clean ``t = 0`` re-evaluation of every chunk by KV refinement (``refiner.py``;
  the first chunk keeps its clean pass),
* replaces the VAE decoder by the streaming pixel decoder (``decoder.py``),
* evaluates the DiT with the optimizations of Sec. 3.4 (``dit.py``),

on one GPU or context parallel on eight GPUs.

    pipe = FastFramesLingBot.from_pretrained("lingbot-1.3b", ...)
    pipe.start(image, prompt, seed=0)
    for chunk in range(trajectory.num_chunks):
        frames = pipe.step(trajectory.rays(chunk))    # [16, 464, 832, 3] uint8 (13 first)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fastframes.checkpoints import load_state_dict
from fastframes.models.kv_refinement import KVRefinementNetwork
from fastframes.models.pixel_decoder import PixelDecoder
from fastframes.pipeline import to_uint8

from .conditioning import CHUNK, LATENT_HEIGHT, LATENT_WIDTH, image_condition, text_context
from .decoder import CapturedDecoder, SpatialDecoder
from .dit import CACHE_FRAMES, FRAME_TOKENS, LingBotDiT, rope_frequencies, rope_table
from .refiner import KVRefiner
from .upstream import import_upstream, load_upstream

__all__ = ["LingBotModel", "MODELS", "FastFramesLingBot", "configure_backends"]

SINK_FRAMES = 6
MAX_LATENTS = 1024  # length of the upstream RoPE table


@dataclass(frozen=True)
class LingBotModel:
    """A released FastFrames-LingBot model.

    ``config``: upstream task name; ``shift``: flow shift of the official four-step schedule
    (``scheduler.timesteps[[0, 250, 500, 750]]``); ``fp32_time_weights``: evaluate the time MLP
    with the FP32 weights of the official checkpoint (otherwise with BF16-rounded weights, as
    the upstream loader); ``kv_layers_per_stream`` / ``split_kv``: KV-refinement stream grouping
    and split-KV attention factor per world size (1 or 8).
    """

    config: str
    shift: float
    fp32_time_weights: bool
    kv_layers_per_stream: dict
    split_kv: dict


MODELS = {
    "lingbot-1.3b": LingBotModel("i2v-1.3B", 5.0, True, {1: 5, 8: 30}, {1: 1, 8: 4}),
}


def configure_backends():
    """Process-wide settings FastFrames-LingBot was evaluated with."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def load_refiner(path, config, device):
    state = load_state_dict(path)
    net = KVRefinementNetwork(dim=config.dim, num_heads=config.num_heads, head_dim=config.dim // config.num_heads,
                              num_blocks=config.num_layers, d_a=state["stem_in.weight"].shape[0])
    net = net.to(torch.bfloat16)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval().requires_grad_(False)


def load_decoder(path, config, device):
    state = load_state_dict(path)
    net = PixelDecoder(config.dim)
    net.load_state_dict(state, strict=True)
    return net.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)


class FastFramesLingBot:
    """One session at a time; under context parallelism every rank makes the same calls."""

    def __init__(self, model: LingBotModel, upstream, refiner, decoder, *, world=1, rank=0, compile=True,
                 graphs=True):
        if world not in (1, 8):
            raise ValueError("FastFrames-LingBot runs on 1 or 8 GPUs")
        self.model, self.upstream, self.world, self.rank = model, upstream, world, rank
        dit = upstream.dit
        self.device = dit.patch_embedding.weight.device
        scheduler = upstream.scheduler
        scheduler.set_timesteps(upstream.config.num_train_timesteps, shift=model.shift)
        self.scheduler = scheduler
        self.schedule = scheduler.timesteps[[0, 250, 500, 750]]
        self.timesteps = tuple(int(t) for t in self.schedule)
        # sigma of every step, located like the upstream flow -> x0 conversion (float64)
        sigmas, grid = scheduler.sigmas.double(), scheduler.timesteps.double()
        self.sigmas = [sigmas[torch.argmin((grid - t).abs())].to(self.device) for t in self.schedule]
        head_dim = dit.dim // dit.num_heads
        self.freqs = rope_frequencies(head_dim, self.device)
        dit.freqs = self.freqs
        shape = (len(dit.blocks), 1, CACHE_FRAMES * FRAME_TOKENS, dit.num_heads, head_dim)
        self.cache_k = torch.zeros(shape, device=self.device, dtype=torch.bfloat16)
        self.cache_v = torch.zeros_like(self.cache_k)
        context = torch.zeros(512, 4096, device=self.device, dtype=torch.bfloat16)
        self.dit = LingBotDiT(dit, self.cache_k, self.cache_v, context, self.timesteps, world=world, rank=rank,
                              split_kv=model.split_kv[world], compile=compile, graphs=graphs)
        result = self.dit.result
        self.refiner = KVRefiner(refiner, dit.patch_embedding, self.cache_k, self.cache_v, result.key, result.value,
                                 self.timesteps[-1], world=world, rank=rank,
                                 heads_per_stream=model.kv_layers_per_stream[world], compile=compile, graphs=graphs)
        decoder_cls = CapturedDecoder if world == 1 else SpatialDecoder
        self.decoder = decoder_cls(decoder, compile=compile, graphs=graphs)
        self.prompt = self.condition = self.noise = self.generator = None
        self.noise_mode = "clip"
        self.latents = self.chunk = 0

    @classmethod
    def from_pretrained(cls, name, *, lingbot_root, lingbot_checkpoint, checkpoint_dir, assets_dir=None, world=1,
                        rank=0, compile=True, graphs=True, device="cuda"):
        """``lingbot_checkpoint``: official weights of the model (T5/VAE are also looked up in
        ``assets_dir``); ``checkpoint_dir``: ``kv_refiner.safetensors`` and ``pixel_decoder.safetensors``."""
        import_upstream(lingbot_root)
        model = MODELS[name]
        upstream = load_upstream(model.config, lingbot_checkpoint, assets_dir,
                                 fp32_time_weights=model.fp32_time_weights, device=device)
        root = Path(checkpoint_dir)
        refiner = load_refiner(root / "kv_refiner.safetensors", upstream.config, device)
        decoder = load_decoder(root / "pixel_decoder.safetensors", upstream.config, device)
        return cls(model, upstream, refiner, decoder, world=world, rank=rank, compile=compile, graphs=graphs)

    @property
    def num_chunks(self):
        return self.latents // CHUNK

    # ------------------------------------------------------------------ session
    @torch.inference_mode()
    def start(self, image, prompt, *, seed=0, latents=240, noise="clip"):
        """Begin a video from a PIL ``image`` and ``prompt`` with a horizon of ``latents`` latent frames.

        The image condition is encoded for the whole horizon, as upstream. ``noise="clip"``
        draws the initial noise of the whole horizon at the start (upstream), ``"chunk"`` draws
        it chunk by chunk; both then draw three transition noises per chunk.
        """
        if latents % CHUNK or not CHUNK <= latents <= MAX_LATENTS:
            raise ValueError(f"latents must be a multiple of {CHUNK} in [{CHUNK}, {MAX_LATENTS}]")
        if prompt != self.prompt:
            self.dit.set_context(text_context(self.upstream.text_encoder, prompt, self.device))
            self.prompt = prompt
        self.condition = image_condition(self.upstream.vae, image, latents, self.device)
        self.latents = latents
        self.restart(seed=seed, noise=noise)

    @torch.inference_mode()
    def restart(self, *, seed=0, noise=None):
        """Restart from the first image of the current session (``noise``: default, the session's mode)."""
        noise = noise or self.noise_mode
        if noise not in ("clip", "chunk"):
            raise ValueError(f"unknown noise mode {noise!r}")
        self.noise_mode = noise
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.decoder.reset()
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.noise = None
        if noise == "clip":
            self.noise = torch.randn((16, self.latents, LATENT_HEIGHT, LATENT_WIDTH), device=self.device,
                                     generator=self.generator)
        self.chunk = 0

    def _evict(self, chunk):
        """Before writing ``chunk``: drop the oldest non-sink frames of a full cache (one shift per chunk)."""
        previous = min(CHUNK * chunk, CACHE_FRAMES)
        evicted = max(0, previous + CHUNK - CACHE_FRAMES)
        if evicted:
            sink, shift = SINK_FRAMES * FRAME_TOKENS, evicted * FRAME_TOKENS
            rolled = (previous - evicted - SINK_FRAMES) * FRAME_TOKENS
            # Contiguous per-layer copies in blocks of ``shift`` tokens, moving left: no block reads a
            # region that an earlier block overwrote, so no temporary copy is needed.
            for cache in (self.cache_k, self.cache_v):
                for layer in cache:
                    for start in range(sink, sink + rolled, shift):
                        end = min(start + shift, sink + rolled)
                        layer[:, start:end].copy_(layer[:, start + shift:end + shift])

    @torch.inference_mode()
    def step(self, rays):
        """Generate the next chunk for camera ``rays`` (``Trajectory.rays`` / ``LiveCamera.rays``).

        Returns ``[16, 464, 832, 3]`` uint8 frames on rank 0 (13 for the first chunk: the pixel
        decoder's first three frames precede the first image) and ``None`` on other ranks.
        """
        chunk = self.chunk
        if self.condition is None or chunk >= self.num_chunks:
            raise RuntimeError("no session or horizon reached; call start() or restart()")
        read_frames = min(CHUNK * (chunk + 1), CACHE_FRAMES)
        self._evict(chunk)
        y = self.condition[:, CHUNK * chunk:CHUNK * (chunk + 1)]
        rope = rope_table(self.freqs, CHUNK * chunk)
        self.dit.begin_chunk(y, rays, rope)
        if self.noise is not None:
            x = self.noise[:, CHUNK * chunk:CHUNK * (chunk + 1)]
        else:
            x = torch.randn((16, CHUNK, LATENT_HEIGHT, LATENT_WIDTH), device=self.device, generator=self.generator)
        for i, timestep in enumerate(self.timesteps):
            result = self.dit(x, timestep, read_frames)
            x0 = (x.double() - self.sigmas[i] * result.prediction.double()).float()
            if i + 1 < len(self.timesteps):
                noise = torch.randn(x0.shape, device=self.device, dtype=x0.dtype, generator=self.generator)
                x = self.scheduler.add_noise(x0, noise, self.schedule[i + 1])
        if chunk == 0:
            self.dit(x0, 0, read_frames, capture=False)  # clean pass writes the first chunk's cache
        else:
            self.refiner(x0, y, rope, read_frames)
        pixels = self.decoder(result.features, result.embedding)
        self.chunk += 1
        if self.rank != 0:
            return None
        return to_uint8(pixels[0, 3:] if chunk == 0 else pixels[0])

    def close(self):
        """Release CUDA graphs (call on every rank before destroying the process group)."""
        self.refiner.close()
        self.decoder.close()
        self.dit.close()
