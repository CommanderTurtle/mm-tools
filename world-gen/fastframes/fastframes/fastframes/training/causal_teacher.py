#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FastFrames-L stage 1: frame-causal Wan2.1 teacher (teacher-forcing flow matching).

The bidirectional Wan2.1 weights are fine-tuned, unchanged in shape, into a
frame-wise autoregressive model. Every update draws one clean latent per rank from
the ODE corpus (its final state; the intermediate states are not used), fresh
Gaussian noise and an independent training timestep for each latent frame. The
noisy frames are denoised under the teacher-forcing mask (noisy frame ``i`` sees
the clean frames ``< i`` and itself) and regressed onto the flow ``noise - x0``
with bell-shaped per-frame timestep weights. A fraction of the samples is trained
with the empty prompt so that the teacher supports classifier-free guidance.

All randomness of a sample is keyed by ``(seed, epoch, sample_id)``, so a resumed
run draws the same samples and noise as the uninterrupted one (at the same world size).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..checkpoints import load_wan_base
from ..wan.causal_model import CausalWanAttentionBlock
from ..wan.text_encoder import TextEncoder
from . import checkpoint
from .data import PromptStream
from .distributed import all_reduce_mean, init_distributed, raise_if_any_failed, wrap_fsdp
from .ema import EMA
from .flow_matching import (
    CFG_DROPOUT,
    NOISE,
    TIMESTEP,
    add_noise,
    bell_weights,
    learning_rate,
    sample_generator,
    shifted_sigmas,
)
from .logging import MetricLogger
from .ode_data import OdeCorpus

__all__ = ["train"]

STREAM = "causal-teacher-v1"


def training_sample(corpus, sample_id, epoch, config, sigmas, weights, device):
    """Clean/noisy latents ``[1, F, 16, H, W]``, BF16 timesteps and loss weights ``[1, F]``, and the prompt."""
    seed = config.seed
    clean = corpus.clean(sample_id).to(device)
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=device,
                        generator=sample_generator(STREAM, seed, epoch, sample_id, NOISE, device))
    index = torch.randint(0, len(sigmas), (clean.shape[0],), dtype=torch.long, device=device,
                          generator=sample_generator(STREAM, seed, epoch, sample_id, TIMESTEP, device))
    dropped = torch.rand((), generator=sample_generator(STREAM, seed, epoch, sample_id, CFG_DROPOUT)).item()
    prompt = "" if dropped < config.train.empty_prompt_probability else corpus.prompt(sample_id)
    clean, noise, sigma = clean[None], noise[None], sigmas[index][None]
    noisy = add_noise(clean, noise, sigma).to(torch.bfloat16)
    # The model sees BF16-rounded timesteps (the noise level itself stays FP32).
    timesteps = (sigma * len(sigmas)).to(torch.bfloat16)
    return clean, noisy, timesteps, noise - clean, weights[index][None], prompt


def flow_loss(model, clean, noisy, timesteps, target, weights, context):
    """Bell-weighted teacher-forcing flow-matching loss (BF16 autocast, FP32 reduction)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        flow = model(noisy, timesteps, context, clean_x=clean)
        per_frame = F.mse_loss(flow.float(), target.float(), reduction="none").mean(dim=(2, 3, 4))
        return (per_frame * weights.float()).mean()


def train(config):
    ctx = init_distributed()
    torch.manual_seed(config.seed + ctx.rank)
    out = Path(config.output_dir)
    opt_cfg = config.optimizer
    corpus = OdeCorpus(config.data.corpus)
    wan_dir = Path(config.model.checkpoint_root) / config.model.name
    encoder = TextEncoder(wan_dir).to(ctx.device, torch.bfloat16)
    model = load_wan_base(wan_dir, causal=True, dtype=torch.float32, device=ctx.device).train().requires_grad_(True)
    model.enable_gradient_checkpointing()
    model = wrap_fsdp(model, {CausalWanAttentionBlock}, ctx, sharding=config.train.sharding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.lr, betas=tuple(opt_cfg.betas),
                                  weight_decay=opt_cfg.weight_decay, eps=1e-8, foreach=False)
    ema = EMA(model, decay=opt_cfg.ema_decay)
    stream = PromptStream(range(len(corpus)), ctx.rank, ctx.world_size, seed=config.seed)
    step, state = checkpoint.load(out, {"model": model}, {"model": (model, optimizer)})
    if step is None:
        step = 0
    else:
        ema.load_state_dict(state["ema"])
        stream.load_state_dict(state["stream"])
    logger = MetricLogger(config, out, ctx.is_main)
    sigmas = shifted_sigmas(config.train.num_train_timesteps, config.train.shift, ctx.device)
    weights = bell_weights(sigmas, config.train.num_train_timesteps)

    model.train()
    while step < config.train.max_updates:
        error = None
        try:
            epoch = stream.position // stream.per_epoch
            [sample_id] = next(stream)
            clean, noisy, timesteps, target, frame_weights, prompt = training_sample(
                corpus, sample_id, epoch, config, sigmas, weights, ctx.device)
            with torch.no_grad():
                context = encoder([prompt])["prompt_embeds"]
            lr = learning_rate(opt_cfg.lr, step, opt_cfg.warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"preparing update {step + 1}")
        loss = flow_loss(model, clean, noisy, timesteps, target, frame_weights, context)
        raise_if_any_failed(None if torch.isfinite(loss) else FloatingPointError(f"loss {loss.item()}"),
                            f"forward of update {step + 1}")
        loss.backward()
        grad_norm = model.clip_grad_norm_(opt_cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):  # the norm is global, so every rank stops here
            raise FloatingPointError(f"non-finite gradient norm at update {step + 1}")
        optimizer.step()
        step += 1
        ema.update(model, step)
        logger.log({"loss": all_reduce_mean(loss.detach()), "grad_norm": grad_norm, "lr": lr}, step)

        final = step == config.train.max_updates
        if step % config.train.export_every == 0 or final:
            # Raw FP32 masters (the consistency stage starts from these) and BF16 EMA weights.
            export = out / "exports" / f"step_{step:08d}"
            checkpoint.export_weights(export / "model.safetensors", model, is_main=ctx.is_main)
            with ema.applied(model):
                checkpoint.export_weights(export / "ema.safetensors", model, torch.bfloat16, is_main=ctx.is_main)
        if step % config.train.save_every == 0 or final:
            checkpoint.save(out, step, {"model": model}, {"model": (model, optimizer)},
                            {"ema": ema.state_dict(), "stream": stream.state_dict()},
                            keep=config.train.keep_checkpoints, is_main=ctx.is_main)
    logger.finish()
