#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Stage ``pixel_decoder``: training of the pixel decoder (Sec. 3.2 and Appendix C of the paper).

The generator stays frozen. Every rollout runs its *primed* frame-wise rollout
(each finished frame is re-run through the DiT at ``t = 0`` to write clean KV
cache entries, the baseline of Sec. 3.3) and keeps, per latent frame, the
pre-head hidden states and head modulation of the final denoising step. The
decoder maps them to RGB and is compared with the frozen Wan VAE decoding of the
same latents; each rollout serves ``train.updates_per_rollout`` optimizer updates.

* ``phase: reconstruction``: ``L_rec = 0.5 L1 + 5 MSE + 0.05 L_grad + 0.05 L_temp
  + 0.5 L_motion`` (Eq. 4-7), with an EMA of the decoder weights.
* ``phase: adversarial``: ``L_rec`` plus the DINOv3 token GAN of Eq. 2-3
  (``dinov3_discriminator.py``) against real videos (``video_data.py``), starting
  from a reconstruction export. The first ``adversarial.warmup`` updates train
  only the discriminator.

``decoder.kind`` selects the ConvNeXt ``PixelDecoder`` (FF-S) or the
``BridgedPixelDecoder`` (FF-L: a seeded 5120 -> 1536 bridge in front of a 1.3B
decoder). With
``decoder.precision: bf16`` FSDP keeps FP32 masters, computes in BF16 and the loss
is evaluated in BF16; with ``autocast`` the FP32 decoder (DDP) and the loss run
under BF16 autocast.

Output directory: ``metrics.jsonl``, resumable ``step_<N>/`` generations (same
world size; a generation may fall inside a rollout, which is then regenerated
exactly) and ``exports/step_<N>/decoder.safetensors`` (+ ``decoder_ema.safetensors``),
loadable as ``FastFrames.from_pretrained(..., decoder=<file>)``.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..checkpoints import build_wan, load_state_dict
from ..models.pixel_decoder import BridgedPixelDecoder, PixelDecoder, align_frames
from ..pipeline import FastFrames, FastFramesConfig
from ..wan.causal_model import allocate_kv_cache
from ..wan.model import WAN_CONFIGS
from ..wan.text_encoder import TextEncoder
from ..wan.vae import VideoVAE
from . import checkpoint
from .data import PromptStream, read_prompts
from .distributed import all_reduce_mean, barrier, init_distributed, raise_if_any_failed, wrap_fsdp
from .ema import EMA
from .logging import MetricLogger

__all__ = ["PrimedRollout", "reconstruction_loss", "rollout_seed", "build_decoder", "train"]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def rollout_seed(seed, rank, rollout):
    """Noise seed of rollout ``rollout`` on ``rank``."""
    return (int(seed) + 10_000_019 * int(rollout) + int(rank)) & ((1 << 63) - 1)


# ---------------------------------------------------------------------------------------------- losses
def gradient_loss(x, y):
    """Eq. 5: horizontal and vertical finite differences."""
    dx = (x[..., :, 1:] - x[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])
    dy = (x[..., 1:, :] - x[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    return dx.abs().mean() + dy.abs().mean()


def temporal_loss(x, y):
    """Eq. 6: consecutive-frame differences."""
    return ((x[:, 1:] - x[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs().mean()


def motion_loss(x, y, gamma):
    """Eq. 7: L1 weighted by ``1 + gamma * mean_c |delta_t y|`` on frames ``t > 1``."""
    weight = 1.0 + float(gamma) * (y[:, 1:] - y[:, :-1]).abs().mean(dim=2, keepdim=True)
    return (weight * (x[:, 1:] - y[:, 1:]).abs().mean(dim=2, keepdim=True)).mean()


def reconstruction_loss(pred, target, weights):
    """``(L_rec, terms)`` of Eq. 4 on ``[B, T, 3, H, W]`` videos; zero-weight terms are skipped."""
    terms = {"l1": F.l1_loss, "mse": F.mse_loss, "grad": gradient_loss, "temporal": temporal_loss,
             "motion": partial(motion_loss, gamma=weights["motion_gamma"])}
    total, values = pred.new_zeros(()), {}
    for name, fn in terms.items():
        if float(weights[name]) > 0:
            value = fn(pred, target)
            total = total + float(weights[name]) * value
            values[f"rec/{name}"] = value.detach()
    return total, values


# ---------------------------------------------------------------------------------------------- rollout
@dataclass
class Pair:
    """Decoder input and target of one rollout."""

    features: torch.Tensor    # [1, F * H/2 * W/2, dim] pre-head hidden states
    modulation: torch.Tensor  # [1, F, 1, dim] head time modulation
    grid: torch.Tensor        # [1, 3] DiT token grid
    target: torch.Tensor      # [1, 81, 3, 8H, 8W] VAE decoding of the rollout latents


class PrimedRollout:
    """Frame-wise rollout of the frozen generator with the clean ``t = 0`` pass after each frame.

    Noise follows ``FastFramesConfig.noise`` from a generator seeded per rollout.
    """

    def __init__(self, dit, recipe: FastFramesConfig):
        self.ff = FastFrames(recipe, dit, refiner=None, decoder=None)

    @torch.no_grad()
    def __call__(self, prompt_embeds, seed):
        """Latents ``[1, F, 16, h, w]``, features ``[1, F*L, C]`` and modulation ``[1, F, 1, C]``."""
        ff, cfg = self.ff, self.ff.config
        noise, transition = ff._noise(seed)
        text_cache = ff.dit.text_cache(prompt_embeds)
        kv_cache = allocate_kv_cache(ff.dit, cfg.num_frames, ff.frame_tokens, device=ff.device)
        latents, features, modulation = torch.empty_like(noise), [], []
        for frame in range(cfg.num_frames):
            schedule = cfg.first_frame_timesteps if frame == 0 else cfg.timesteps
            x = noise[:, frame:frame + 1]
            for step, timestep in enumerate(schedule):
                x0, feature, mod = ff._predict(x, timestep, frame, text_cache, kv_cache)
                if step + 1 < len(schedule):
                    next_t = torch.full((1,), schedule[step + 1], device=ff.device, dtype=torch.float32)
                    x = ff.scheduler.add_noise(x0.flatten(0, 1), transition(step, frame), next_t).unflatten(
                        0, x0.shape[:2])
            latents[:, frame:frame + 1].copy_(x0)
            features.append(feature)
            modulation.append(mod)
            if frame + 1 < cfg.num_frames:  # the last frame's clean pass would be read by no one
                ff._predict(x0, 0.0, frame, text_cache, kv_cache)
        return latents, torch.cat(features, dim=1), torch.cat(modulation, dim=1)


# ---------------------------------------------------------------------------------------------- decoder
def build_decoder(settings, wan_name):
    """An FP32 decoder: seeded fresh weights, optionally overwritten by ``settings.init_from``.

    A ``bridged`` decoder initialised from a plain decoder keeps its fresh seeded bridge.
    """
    dim = WAN_CONFIGS[wan_name].dim
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(settings.init_seed))
        decoder = {"convnext": partial(PixelDecoder, dim),
                   "bridged": partial(BridgedPixelDecoder, in_dim=dim)}[settings.kind]()
    if settings.get("init_from"):
        state = load_state_dict(settings.init_from, settings.get("init_key"))
        if isinstance(decoder, BridgedPixelDecoder) and not any(k.startswith("bridge.") for k in state):
            decoder.pixel_head.load_state_dict({k.removeprefix("pixel_head."): v for k, v in state.items()})
        else:
            if not isinstance(decoder, BridgedPixelDecoder):
                state = {k.removeprefix("pixel_head."): v for k, v in state.items()}
            decoder.load_state_dict(state, strict=True)
    head = decoder.pixel_head if isinstance(decoder, BridgedPixelDecoder) else decoder
    head.gradient_checkpointing = bool(settings.get("gradient_checkpointing", False))
    return decoder.float()


# ---------------------------------------------------------------------------------------------- training
class Trainer:

    def __init__(self, config):
        self.config = config
        self.ctx = ctx = init_distributed()
        self.device = ctx.device
        train, model, ro = config.train, config.model, config.rollout
        torch.backends.cuda.matmul.allow_tf32 = bool(train.allow_tf32)
        torch.backends.cudnn.allow_tf32 = True
        self.adversarial = config.phase == "adversarial"
        if config.phase not in ("reconstruction", "adversarial"):
            raise ValueError(f"unknown phase {config.phase!r}")
        self.seed, self.out = int(config.seed), Path(config.output_dir)
        self.loss = dict(config.loss)

        # Frozen generator, text encoder and VAE.
        root = Path(model.checkpoint_root) / model.name
        dit = build_wan(model.name, load_state_dict(model.generator, model.get("generator_key")), device=self.device)
        recipe = FastFramesConfig(
            name=model.name, wan_model=model.name, first_frame_timesteps=tuple(ro.first_frame_timesteps),
            timesteps=tuple(ro.timesteps), refine_timestep=float(ro.timesteps[-1]), noise=ro.noise,
            autocast=False, fast_ops=False, decoder_dtype=torch.float32, contiguous_stem=False,
            allow_tf32=bool(train.allow_tf32), num_frames=int(ro.num_frames), latent_size=tuple(ro.latent_size))
        self.rollout = PrimedRollout(dit, recipe)
        self.text_encoder = TextEncoder(root).to(self.device, torch.bfloat16)
        self.vae_dtype, self.target_dtype = DTYPES[config.target.vae_dtype], DTYPES[config.target.dtype]
        self.vae = VideoVAE(root / "Wan2.1_VAE.pth").to(self.device, self.vae_dtype)
        self.num_pixel_frames = 4 * (int(ro.num_frames) - 1) + 1
        height, width = ro.latent_size
        self.grid = torch.tensor([[int(ro.num_frames), height // 2, width // 2]], dtype=torch.long,
                                 device=self.device)

        # Trainable decoder.
        dec, opt = config.decoder, config.optimizer
        self.precision = dec.precision
        net = build_decoder(dec, model.name).to(self.device).train()
        num_parameters = sum(p.numel() for p in net.parameters())  # before FSDP shards them
        if self.precision == "bf16":
            self.decoder = self.module = wrap_fsdp(net, (), ctx, sharding=train.sharding)
        elif self.precision == "autocast":
            self.decoder, self.module = DDP(net, device_ids=[ctx.local_rank], broadcast_buffers=False), net
        else:
            raise ValueError(f"unknown decoder precision {self.precision!r}")
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=float(opt.lr), betas=tuple(opt.betas),
                                           weight_decay=float(opt.weight_decay))
        self.ema = (EMA(self.decoder, opt.ema_decay, start_step=int(opt.ema_start), copy_on_start=True)
                    if opt.get("ema_decay") else None)

        if self.adversarial:
            from .dinov3_discriminator import DINOv3Backbone, TokenDiscriminator
            from .video_data import VideoFolder
            d, real = config.discriminator, config.real_data
            backbone = DINOv3Backbone(d.backbone, tuple(d.layers), d.dtype)
            self.discriminator = TokenDiscriminator(backbone, int(d.head_hidden), int(d.init_seed)).to(self.device)
            for p in self.discriminator.heads.parameters():
                dist.broadcast(p.data, src=0)
            self.d_optimizer = torch.optim.AdamW(self.discriminator.heads.parameters(), lr=float(d.lr),
                                                 betas=tuple(d.betas), weight_decay=float(d.weight_decay))
            self.real = VideoFolder(real.root, ctx.rank, ctx.world_size, num_frames=self.num_pixel_frames,
                                    height=8 * height, width=8 * width, seed=int(real.seed))

        data = config.data
        prompts = read_prompts(data.prompts)
        if data.get("num_prompts"):
            prompts = prompts[:int(data.num_prompts)]
        self.prompts = PromptStream(prompts, ctx.rank, ctx.world_size, seed=int(data.shuffle_seed))
        self.eval_prompts = read_prompts(data.eval_prompts)[:int(data.num_eval_prompts)]
        self.eval_seed = int(data.eval_seed)

        torch.manual_seed(self.seed + ctx.rank)
        self.step = self.saved_step = 0
        step, rank_state = checkpoint.load(self.out, *self._state())
        if step is not None:
            self.step = self.saved_step = step
            if self.ema is not None:
                self.ema.load_state_dict(rank_state["ema"])
            torch.set_rng_state(rank_state["rng"]["cpu"])
            torch.cuda.set_rng_state(rank_state["rng"]["cuda"], self.device)
        self.logger = MetricLogger(config, self.out, ctx.is_main)
        if ctx.is_main:
            print(f"[pixel_decoder] {config.phase}: {dec.kind} decoder, {num_parameters:,} parameters "
                  f"({self.precision}), world size {ctx.world_size}, starting at update {self.step}", flush=True)

    # ------------------------------------------------------------------ state
    def _state(self):
        models, optimizers = {"decoder": self.module}, {"decoder": (self.module, self.optimizer)}
        if self.adversarial:
            heads = self.discriminator.heads
            models["discriminator"], optimizers["discriminator"] = heads, (heads, self.d_optimizer)
        return models, optimizers

    def save(self):
        rank_state = {"ema": None if self.ema is None else self.ema.state_dict(),
                      "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device)}}
        checkpoint.save(self.out, self.step, *self._state(), rank_state=rank_state,
                        keep=int(self.config.train.keep_checkpoints), is_main=self.ctx.is_main)
        self.saved_step = self.step

    def export(self):
        path = self.out / "exports" / f"step_{self.step:08d}"
        checkpoint.export_weights(path / "decoder.safetensors", self.module, is_main=self.ctx.is_main)
        if self.ema is not None and self.ema.num_updates:
            with self.ema.applied(self.decoder):
                checkpoint.export_weights(path / "decoder_ema.safetensors", self.module, is_main=self.ctx.is_main)
        if self.ctx.is_main:
            print(f"[pixel_decoder] exported {path}", flush=True)

    # ------------------------------------------------------------------ data
    def autocast(self):
        if self.precision == "autocast":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def make_pair(self, prompt, seed):
        embeds = self.text_encoder([prompt])["prompt_embeds"]
        latents, features, modulation = self.rollout(embeds, seed)
        target = self.vae.decode_to_pixel(latents.to(self.vae_dtype)).to(self.target_dtype)
        return Pair(features, modulation, self.grid, target)

    def rollout_inputs(self, index):
        """Pair (and real clip) of rollout ``index``; a pure function of ``index`` and the rank."""
        error = pair = real = None
        try:
            self.prompts.load_state_dict({"position": index})
            pair = self.make_pair(next(self.prompts)[0], rollout_seed(self.seed, self.ctx.rank, index))
            if self.adversarial:
                real = self.real.clip(index)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"rollout {index}")
        return pair, real

    def predict(self, pair, module=None):
        with self.autocast():
            pixels = (module or self.decoder)(pair.features, pair.modulation, pair.grid)
        return align_frames(pixels, self.num_pixel_frames)

    def reconstruction(self, pred, target):
        if self.precision == "bf16":
            target = target.to(pred.dtype)
        with self.autocast():
            return reconstruction_loss(pred, target, self.loss)

    def _check(self, loss, what):
        raise_if_any_failed(None if torch.isfinite(loss) else FloatingPointError(f"{what} {loss.item()}"),
                            f"{what} at update {self.step + 1}")

    def _apply_gradients(self):
        max_norm = float(self.config.optimizer.max_grad_norm)
        if self.precision == "bf16":
            grad_norm = self.decoder.clip_grad_norm_(max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm)
        if not torch.isfinite(grad_norm):  # the norm is global, so every rank stops here
            raise FloatingPointError(f"non-finite decoder gradient norm at update {self.step + 1}")
        self.optimizer.step()
        return grad_norm

    # ------------------------------------------------------------------ updates
    def reconstruction_update(self, pair):
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.predict(pair)
        loss, metrics = self.reconstruction(pred, pair.target)
        self._check(loss, "reconstruction loss")
        loss.backward()
        return {"loss": loss.detach(), **metrics, "grad_norm": self._apply_gradients()}

    def adversarial_update(self, pair, real_video):
        from .dinov3_discriminator import (
            discriminator_step,
            generator_step,
            gradient_ratio_cap,
            sample_crops,
            step_generator,
        )
        from .video_data import to_discriminator_range
        adv, d = self.config.adversarial, self.config.discriminator
        step, rank, world = self.step, self.ctx.rank, self.ctx.world_size
        base = int(adv.seed)
        generator_update = step >= int(adv.warmup)
        with contextlib.nullcontext() if generator_update else torch.no_grad():
            pred = self.predict(pair, None if generator_update else self.module)
        crops = (int(d.clips_per_video), int(d.clip_frames), int(d.crop_size))
        fake = sample_crops(pred, *crops, step_generator(base, rank, step, 10))
        real = to_discriminator_range(sample_crops(real_video, *crops, step_generator(base, rank, step, 12)))
        real = real.to(self.device)
        metrics = discriminator_step(self.discriminator, self.d_optimizer, fake, real, d,
                                     subset_seed=base + step * world + rank,
                                     noise_generator=step_generator(base, rank, step, 11, device=self.device))
        if generator_update:
            self.optimizer.zero_grad(set_to_none=True)
            loss, rec = self.reconstruction(pred, pair.target)
            self._check(loss, "reconstruction loss")
            cap = gradient_ratio_cap(step, int(adv.warmup), int(adv.ramp), float(adv.max_gradient_ratio))
            metrics.update(generator_step(self.discriminator, pred, fake, real, loss, float(adv.weight), cap))
            metrics.update({"loss": loss.detach(), **rec, "g/ratio_cap": cap, "grad_norm": self._apply_gradients()})
        return metrics

    # ------------------------------------------------------------------ evaluation / loop
    @torch.no_grad()
    def evaluate(self):
        """``L_rec`` and PSNR against the VAE target on held-out prompts (one per rank; EMA weights if active)."""
        index = self.ctx.rank % len(self.eval_prompts)
        error = pair = None
        try:
            pair = self.make_pair(self.eval_prompts[index], self.eval_seed + index)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"evaluation rollout at update {self.step}")
        weights = self.ema.applied(self.decoder) if self.ema is not None and self.ema.num_updates else \
            contextlib.nullcontext()
        with weights:
            pred = self.predict(pair)
            loss, _ = self.reconstruction(pred, pair.target)
        mse = (pred.float().clamp(-1, 1) - pair.target.float()).square().mean()
        stats = all_reduce_mean(torch.stack([loss.float(), 10 * torch.log10(4.0 / mse)]))
        if self.ctx.is_main:
            print(f"[eval {self.step}] L_rec {stats[0].item():.5f} PSNR {stats[1].item():.2f} dB", flush=True)
        self.logger.log({"eval/rec_loss": stats[0], "eval/psnr": stats[1]}, self.step)

    def log(self, metrics, seconds):
        keys = sorted(metrics)
        values = all_reduce_mean(torch.stack([torch.as_tensor(metrics[k], dtype=torch.float32,
                                                              device=self.device) for k in keys]))
        values = dict(zip(keys, values.tolist()))
        self.logger.log({**values, "seconds": seconds}, self.step)
        if self.ctx.is_main and (self.step % int(self.config.train.print_every) == 0 or self.step == 1):
            shown = " ".join(f"{k} {values[k]:.4g}" for k in keys if k in ("loss", "grad_norm", "d/loss",
                                                                              "g/applied_gradient_ratio"))
            print(f"[update {self.step}] {shown} ({seconds:.2f}s)", flush=True)

    def fit(self):
        train = self.config.train
        per_rollout, max_updates = int(train.updates_per_rollout), int(train.max_updates)
        export_every, save_every = int(train.export_every), int(train.save_every)
        while self.step < max_updates:
            index, inner = divmod(self.step, per_rollout)
            start = time.time()
            pair, real = self.rollout_inputs(index)
            for _ in range(inner, per_rollout):
                if self.step >= max_updates:
                    break
                metrics = self.adversarial_update(pair, real) if self.adversarial else self.reconstruction_update(pair)
                self.step += 1
                if self.ema is not None:
                    self.ema.update(self.decoder, self.step)
                self.log(metrics, time.time() - start)
                start = time.time()
                if self.step % export_every == 0:
                    self.evaluate()
                    self.export()
                if self.step % save_every == 0:
                    self.save()
            del pair, real
        if self.step % export_every:
            self.evaluate()
            self.export()
        if self.step != self.saved_step:
            self.save()
        barrier()
        self.logger.finish()


def train(config):
    """``scripts/train.py`` entry point."""
    Trainer(config).fit()
