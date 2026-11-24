#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Stage ``dmd``: self-forcing distribution matching distillation of the few-step generator.

Four models take part: the trainable frame-causal generator with an EMA of its
weights, the frozen *real score* (a bidirectional Wan teacher, guided with CFG
against the negative prompt) and the trainable *fake score* (the critic), a
bidirectional Wan model that learns the generator's distribution. The generator
only ever sees prompts and noise: its clips come from self-forcing rollouts
(``self_forcing.py``) with a sampled exit step.

* Critic update: a gradient-free rollout ``x``; the fake score is trained with
  the flow-matching loss on ``x`` noised to a random score timestep.
* Generator update: a rollout ``x`` with gradient; at a random score timestep
  the detached direction ``d = (x0_fake - x0_real) / mean|x - x0_real|`` enters
  the surrogate ``0.5 * mean((x - stopgrad(x - d))^2)``, whose gradient is ``d``.

The generator update scores a gradient-free reference rollout; then every frame
is regenerated with gradient against the clean context of the reference and
back-propagates its share of an FP32 surrogate. Exit steps follow the balanced
schedule of ``self_forcing.py``; every update pre-generates its transition noise.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..checkpoints import build_wan, load_state_dict, load_wan_base
from ..wan.causal_model import CausalWanAttentionBlock
from ..wan.model import WanAttentionBlock
from ..wan.scheduler import FlowMatchScheduler
from ..wan.text_encoder import TextEncoder
from . import checkpoint
from .data import PromptStream, read_prompts
from .distributed import all_reduce_mean, init_distributed, raise_if_any_failed, wrap_fsdp
from .ema import EMA
from .logging import MetricLogger
from .self_forcing import (
    CRITIC_EXITS,
    FIRST_FRAME_CRITIC_EXITS,
    FIRST_FRAME_GENERATOR_EXITS,
    GENERATOR_EXITS,
    SelfForcingRollout,
    balanced_exit,
)

__all__ = ["train", "score", "score_timesteps", "dmd_direction", "critic_loss"]


def score(model, noisy, t, prompt_embeds, scheduler):
    """Flow and x0 of a bidirectional score network; ``noisy``: ``[B, F, C, H, W]``, ``t``: ``[B, F]``."""
    b, f, _, h, w = noisy.shape
    flow = model(noisy.permute(0, 2, 1, 3, 4), t[:, 0].to(torch.bfloat16), prompt_embeds, f * h * w // 4)
    flow = flow.permute(0, 2, 1, 3, 4)
    x0 = scheduler.flow_to_x0(flow.flatten(0, 1), noisy.flatten(0, 1), t.flatten())
    return flow, x0.unflatten(0, (b, f))


def score_timesteps(batch, frames, shift, bounds, device):
    """One score timestep per clip: ``shift(uniform{0..999} / 1000) * 1000`` clamped to ``bounds`` (FP32)."""
    sigma = torch.randint(0, 1000, (batch, 1), device=device) / 1000
    t = shift * sigma / (1 + (shift - 1) * sigma) * 1000
    return t.clamp(*bounds).expand(batch, frames)


def noised(x, scheduler, t):
    noise = torch.randn_like(x)
    return noise, scheduler.add_noise(x.flatten(0, 1), noise.flatten(0, 1), t.flatten()).unflatten(0, x.shape[:2])


@torch.no_grad()
def dmd_direction(x, real, fake, cond, uncond, scheduler, cfg):
    """Detached DMD direction at a random score timestep (BF16, like the scores)."""
    t = score_timesteps(x.shape[0], x.shape[1], cfg.shift, cfg.score_timestep_range, x.device)
    _, noisy = noised(x, scheduler, t)

    def x0(model, prompt_embeds):
        return score(model, noisy, t, prompt_embeds, scheduler)[1]

    fake_x0 = x0(fake, cond)
    if cfg.fake_guidance_scale:
        fake_x0 = fake_x0 + (fake_x0 - x0(fake, uncond)) * cfg.fake_guidance_scale
    real_x0 = x0(real, cond)
    real_x0 = real_x0 + (real_x0 - x0(real, uncond)) * cfg.guidance_scale
    normalizer = (x - real_x0).abs().mean(dim=(1, 2, 3, 4), keepdim=True)
    return torch.nan_to_num((fake_x0 - real_x0) / normalizer.clamp_min(1e-6))


def critic_loss(fake, x, cond, scheduler, cfg):
    """Flow-matching loss (FP32) of the fake score on the generated clip ``x``."""
    t = score_timesteps(x.shape[0], x.shape[1], cfg.shift, cfg.score_timestep_range, x.device)
    noise, noisy = noised(x, scheduler, t)
    flow, _ = score(fake, noisy, t, cond, scheduler)
    return (flow.float() - (noise.float() - x.float())).square().mean()


class DMDTrainer:
    """The four models and one generator / critic update of the ``dmd`` stage."""

    def __init__(self, config, ctx):
        self.config, self.ctx = config, ctx
        cfg, ro, opt = config.train, config.rollout, config.optimizer
        root = Path(config.model.checkpoint_root)
        device = ctx.device
        self.encoder = TextEncoder(root / config.model.name).to(device, torch.bfloat16)
        with torch.no_grad():
            self.uncond = self.encoder([config.train.negative_prompt])["prompt_embeds"]
        generator = build_wan(config.model.name, load_state_dict(config.model.init, config.model.get("init_key")),
                              causal=True, dtype=torch.float32, device=device)
        generator.train().requires_grad_(True).enable_gradient_checkpointing()
        fake = load_wan_base(root / config.model.fake_score, dtype=torch.float32, device=device)
        fake.train().requires_grad_(True).enable_gradient_checkpointing()
        # The real score is frozen: BF16 weights equal FSDP's BF16 compute copy of FP32 ones.
        real = load_wan_base(root / config.model.real_score, dtype=torch.bfloat16, device=device)
        self.generator = wrap_fsdp(generator, {CausalWanAttentionBlock}, ctx, sharding=cfg.sharding)
        self.fake = wrap_fsdp(fake, {WanAttentionBlock}, ctx, sharding=cfg.sharding)
        self.real = wrap_fsdp(real, {WanAttentionBlock}, ctx, sharding=cfg.sharding)
        self.generator_opt = torch.optim.AdamW(self.generator.parameters(), lr=opt.generator_lr,
                                               betas=tuple(opt.betas), weight_decay=opt.weight_decay)
        self.fake_opt = torch.optim.AdamW(self.fake.parameters(), lr=opt.critic_lr, betas=tuple(opt.betas),
                                          weight_decay=opt.weight_decay)
        self.ema = EMA(self.generator, opt.ema_decay, start_step=opt.ema_start, copy_on_start=opt.ema_copy_on_start)
        self.scheduler = FlowMatchScheduler(shift=cfg.shift)
        self.rollout = SelfForcingRollout(self.generator, self.scheduler, ro.timesteps, ro.first_frame_timesteps)
        self.shape = tuple(ro.latent_shape)

    def encode(self, prompts):
        with torch.no_grad():
            return self.encoder(prompts)["prompt_embeds"]

    def sample_noise(self, batch):
        """Initial noise ``[B, F, C, H, W]`` and the transition noise bank ``[S, B, F, C, H, W]``."""
        shape, device = (batch, *self.shape), self.ctx.device
        noise = torch.randn(shape, dtype=torch.bfloat16, device=device)
        bank = torch.randn((self.rollout.num_transitions, *shape), dtype=torch.bfloat16, device=device)
        return noise, bank

    def exits(self, index, critic):
        """(frame-0 exit, exit of the other frames) of generator / critic update ``index`` (0-based)."""
        ro = self.config.rollout
        steps, first = len(ro.timesteps), len(ro.first_frame_timesteps)
        block = self.config.train.critic_updates_per_generator if critic else None
        streams = (FIRST_FRAME_CRITIC_EXITS, CRITIC_EXITS) if critic else (FIRST_FRAME_GENERATOR_EXITS,
                                                                           GENERATOR_EXITS)
        return (balanced_exit(index, first, block or first, self.config.seed, streams[0]),
                balanced_exit(index, steps, block or steps, self.config.seed, streams[1]))

    def critic_update(self, prompts, index):
        cfg = self.config.train
        error = None
        try:
            self.fake_opt.zero_grad(set_to_none=True)
            cond = self.encode(prompts)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"preparing critic update {index + 1}")
        noise, bank = self.sample_noise(len(prompts))
        exits = self.exits(index, critic=True)
        with torch.no_grad():
            x = self.rollout(noise, cond, exits, bank)
        loss = critic_loss(self.fake, x, cond, self.scheduler, cfg)
        raise_if_any_failed(None if torch.isfinite(loss) else FloatingPointError(f"critic loss {loss.item()}"),
                            f"critic update {index + 1}")
        loss.backward()
        grad_norm = self.fake.clip_grad_norm_(self.config.optimizer.max_grad_norm)
        if not torch.isfinite(grad_norm):  # the norm is global, so every rank stops here
            raise FloatingPointError(f"non-finite critic gradient norm at critic update {index + 1}")
        self.fake_opt.step()
        return {"critic_loss": all_reduce_mean(loss.detach().float()), "critic_grad_norm": grad_norm}

    def generator_update(self, prompts, index):
        cfg = self.config.train
        error = None
        try:
            self.generator_opt.zero_grad(set_to_none=True)
            cond = self.encode(prompts)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"preparing generator update {index + 1}")
        noise, bank = self.sample_noise(len(prompts))
        exits = self.exits(index, critic=False)
        uncond = self.uncond.expand(len(prompts), -1, -1)
        with torch.no_grad():
            reference = self.rollout(noise, cond, exits, bank)
        direction = dmd_direction(reference, self.real, self.fake, cond, uncond, self.scheduler, cfg)
        loss = torch.zeros((), device=noise.device)
        for frame, x0 in self.rollout.replay(noise, reference, cond, exits, bank):
            x0 = x0.float()
            target = (x0 - direction[:, frame:frame + 1].float()).detach()
            frame_loss = (0.5 * (x0 - target).square()).mean() / noise.shape[1]
            frame_loss.backward()
            loss += frame_loss.detach()
        raise_if_any_failed(None if torch.isfinite(loss) else FloatingPointError(f"loss {loss.item()}"),
                            f"generator update {index + 1}")
        grad_norm = self.generator.clip_grad_norm_(self.config.optimizer.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite generator gradient norm at generator update {index + 1}")
        self.generator_opt.step()
        self.ema.update(self.generator, index + 1)
        return {"generator_loss": all_reduce_mean(loss.detach().float()), "generator_grad_norm": grad_norm,
                "dmd_direction_l1": all_reduce_mean(direction.abs().mean().float())}


def train(config):
    ctx = init_distributed()
    torch.manual_seed(config.seed + ctx.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(config.output_dir)
    cfg = config.train
    trainer = DMDTrainer(config, ctx)
    stream = PromptStream(read_prompts(config.data.prompts), ctx.rank, ctx.world_size,
                          seed=config.data.get("seed", config.seed), batch_size=cfg.get("batch_size", 1))
    models = {"generator": trainer.generator, "fake_score": trainer.fake}
    optimizers = {"generator": (trainer.generator, trainer.generator_opt),
                  "fake_score": (trainer.fake, trainer.fake_opt)}
    critic_updates, state = checkpoint.load(out, models, optimizers)
    generator_updates = 0
    if critic_updates is None:
        critic_updates = 0
    else:
        trainer.ema.load_state_dict(state["ema"])
        stream.load_state_dict(state["stream"])
        generator_updates = state["generator_updates"]
        torch.set_rng_state(state["rng"]["cpu"])
        torch.cuda.set_rng_state(state["rng"]["cuda"], ctx.device)
    logger = MetricLogger(config, out, ctx.is_main)

    def generator_due(completed):
        first, every = cfg.initial_critic_updates, cfg.critic_updates_per_generator
        return completed >= first and (completed - first) % every == 0

    while critic_updates < cfg.max_critic_updates:
        # A generator update follows every due critic update and scores the same prompts.
        prompts = next(stream)
        metrics = trainer.critic_update(prompts, critic_updates)
        critic_updates += 1
        if generator_due(critic_updates):
            metrics.update(trainer.generator_update(prompts, generator_updates))
            generator_updates += 1
        logger.log({**metrics, "generator_updates": generator_updates}, critic_updates)

        final = critic_updates == cfg.max_critic_updates
        if critic_updates % cfg.export_every == 0 or final:
            export = out / "exports" / f"step_{critic_updates:08d}"
            checkpoint.export_weights(export / "model.safetensors", trainer.generator, torch.bfloat16,
                                      is_main=ctx.is_main)
            if trainer.ema.num_updates:
                with trainer.ema.applied(trainer.generator):
                    checkpoint.export_weights(export / "ema.safetensors", trainer.generator, torch.bfloat16,
                                              is_main=ctx.is_main)
        if critic_updates % cfg.save_every == 0 or final:
            rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(ctx.device)}
            checkpoint.save(out, critic_updates, models, optimizers,
                            {"ema": trainer.ema.state_dict(), "stream": stream.state_dict(),
                             "generator_updates": generator_updates, "rng": rng},
                            keep=cfg.keep_checkpoints, is_main=ctx.is_main)
    logger.finish()
