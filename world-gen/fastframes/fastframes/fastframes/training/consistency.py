#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FastFrames-L stage 2: causal consistency distillation (CD) of the frame-causal teacher.

Three copies of the causal teacher's (raw) weights take part: the trainable
student, the frozen teacher and an EMA target (the student's exponential moving
average, stored as a model). For one clean latent ``x0`` per rank, one of the
adjacent intervals ``(sigma, sigma_next)`` of the ``levels``-level shifted grid is
drawn and every frame is noised to ``sigma``::

    x_t      = (1 - sigma) x0 + sigma * noise
    x_next   = x_t + v_teacher_cfg(x_t) * (sigma_next - sigma)     # one guided Euler step
    loss     = mean((x_t - sigma v_student(x_t) - stopgrad(x_next - sigma_next v_ema(x_next)))^2)

All predictions use teacher forcing on the clean latent history. The teacher is
guided (CFG) with the negative prompt, so the student learns the guided map and
is sampled conditional-only.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..checkpoints import build_wan, load_state_dict
from ..wan.causal_model import CausalWanAttentionBlock
from ..wan.text_encoder import TextEncoder
from . import checkpoint
from .data import PromptStream
from .distributed import all_reduce_mean, init_distributed, raise_if_any_failed, wrap_fsdp
from .flow_matching import CD_INTERVAL, NOISE, add_noise, learning_rate, sample_generator, shifted_sigmas
from .logging import MetricLogger
from .ode_data import OdeCorpus

__all__ = ["train"]

STREAM = "ar-cd-init-v1"


def training_sample(corpus, sample_id, epoch, seed, sigmas, device):
    """Clean/noisy latents ``[1, F, 16, H, W]`` and the drawn interval's sigmas (FP32 scalars)."""
    clean = corpus.clean(sample_id).to(device)
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=device,
                        generator=sample_generator(STREAM, seed, epoch, sample_id, NOISE, device))
    interval = int(torch.randint(0, len(sigmas) - 1, (), dtype=torch.long,
                                 generator=sample_generator(STREAM, seed, epoch, sample_id, CD_INTERVAL)))
    clean, noise = clean[None], noise[None]
    sigma, sigma_next = sigmas[interval], sigmas[interval + 1]
    noisy = add_noise(clean, noise, sigma.expand(clean.shape[:2])).to(torch.bfloat16)
    return clean, noisy, sigma, sigma_next


def consistency_loss(student, teacher, target, clean, noisy, sigma, sigma_next, context, negative, guidance_scale,
                     num_train_timesteps=1000):
    """CD loss (FP32) of one sample; the teacher step and the EMA target are gradient-free."""
    frames = clean.shape[:2]
    # The models see BF16-rounded timesteps (the noise levels themselves stay FP32).
    t = (sigma * num_train_timesteps).expand(frames).to(torch.bfloat16)
    t_next = (sigma_next * num_train_timesteps).expand(frames).to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v_cond = teacher(noisy, t, context, clean_x=clean).float()
        v_neg = teacher(noisy, t, negative, clean_x=clean).float()
        guided = v_neg + guidance_scale * (v_cond - v_neg)
        x_next = (noisy.float() + guided * (sigma_next - sigma)).to(noisy.dtype)
        v_target = target(x_next, t_next, context, clean_x=clean)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_student = student(noisy, t, context, clean_x=clean)
        x0_student = noisy.float() - sigma.float() * v_student.float()
        x0_target = x_next.float() - sigma_next.float() * v_target.float()
        return F.mse_loss(x0_student, x0_target)


@torch.no_grad()
def update_ema_target(target, student, decay):
    params = dict(student.named_parameters())
    for name, p in target.named_parameters():
        p.lerp_(params[name].detach(), 1.0 - decay)


def train(config):
    ctx = init_distributed()
    torch.manual_seed(config.seed + ctx.rank)
    out = Path(config.output_dir)
    opt_cfg = config.optimizer
    corpus = OdeCorpus(config.data.corpus)
    wan_dir = Path(config.model.checkpoint_root) / config.model.name
    encoder = TextEncoder(wan_dir).to(ctx.device, torch.bfloat16)
    with torch.no_grad():
        negative = encoder([config.train.negative_prompt])["prompt_embeds"]
    init = load_state_dict(config.model.init)

    def copy(trainable, state):
        model = build_wan(config.model.name, dict(state), causal=True, dtype=torch.float32, device=ctx.device)
        if trainable:
            model.train().requires_grad_(True).enable_gradient_checkpointing()
        return wrap_fsdp(model, {CausalWanAttentionBlock}, ctx, sharding=config.train.sharding)

    student, teacher, target = copy(True, init), copy(False, init), copy(False, init)
    del init
    optimizer = torch.optim.AdamW(student.parameters(), lr=opt_cfg.lr, betas=tuple(opt_cfg.betas),
                                  weight_decay=opt_cfg.weight_decay, eps=1e-8, foreach=False)
    stream = PromptStream(range(len(corpus)), ctx.rank, ctx.world_size, seed=config.seed)
    models = {"student": student, "target": target}
    step, state = checkpoint.load(out, models, {"student": (student, optimizer)})
    if step is None:
        step = 0
    else:
        stream.load_state_dict(state["stream"])
    logger = MetricLogger(config, out, ctx.is_main)
    sigmas = shifted_sigmas(config.train.levels, config.train.shift, ctx.device)

    while step < config.train.max_updates:
        error = None
        try:
            epoch = stream.position // stream.per_epoch
            [sample_id] = next(stream)
            clean, noisy, sigma, sigma_next = training_sample(corpus, sample_id, epoch, config.seed, sigmas,
                                                              ctx.device)
            with torch.no_grad():
                context = encoder([corpus.prompt(sample_id)])["prompt_embeds"]
            lr = learning_rate(opt_cfg.lr, step, opt_cfg.warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"preparing update {step + 1}")
        loss = consistency_loss(student, teacher, target, clean, noisy, sigma, sigma_next, context, negative,
                                config.train.guidance_scale)
        raise_if_any_failed(None if torch.isfinite(loss) else FloatingPointError(f"loss {loss.item()}"),
                            f"forward of update {step + 1}")
        loss.backward()
        grad_norm = student.clip_grad_norm_(opt_cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):  # the norm is global, so every rank stops here
            raise FloatingPointError(f"non-finite gradient norm at update {step + 1}")
        optimizer.step()
        update_ema_target(target, student, opt_cfg.ema_decay)
        step += 1
        logger.log({"loss": all_reduce_mean(loss.detach()), "grad_norm": grad_norm, "lr": lr}, step)

        final = step == config.train.max_updates
        if step % config.train.export_every == 0 or final:
            export = out / "exports" / f"step_{step:08d}"
            checkpoint.export_weights(export / "model.safetensors", student, torch.bfloat16, is_main=ctx.is_main)
            checkpoint.export_weights(export / "ema.safetensors", target, torch.bfloat16, is_main=ctx.is_main)
        if step % config.train.save_every == 0 or final:
            checkpoint.save(out, step, models, {"student": (student, optimizer)}, {"stream": stream.state_dict()},
                            keep=config.train.keep_checkpoints, is_main=ctx.is_main)
    logger.finish()
