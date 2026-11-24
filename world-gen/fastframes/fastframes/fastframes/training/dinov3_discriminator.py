#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""DINOv3 token discriminator and the adversarial objectives of the pixel decoder (Eq. 2-3).

A frozen DINOv3 ViT-S/16 (Hugging Face format, loaded from a local directory)
embeds every frame of a ``[N, 3, T, 224, 224]`` crop in ``[-1, 1]``. For each
selected layer (4, 8, 12) two token heads (LayerNorm -> Linear -> SiLU ->
Linear) score every patch token: a spatial head on the token and a temporal
head on the token concatenated with its change from the previous frame (zero
for the first frame). The backbone runs in BF16 and never trains; the heads
are FP32.

* ``discriminator_step``: relativistic softplus loss averaged per head,
  ``lambda_center * mean((D(y) + D(y_hat))^2)`` and finite-difference R1/R2 slope
  penalties ``mean(((D(x + eps n) - D(x)) / eps)^2)`` on a seeded subset of the
  clips; GPU-averaged gradients, clipped.
* ``generator_step``: reconstruction loss plus the reverse relativistic term
  with stop-gradient real logits. Its RGB gradient is capped per sample at
  ``ratio`` times the L1 magnitude of the reconstruction gradient and applied
  through the surrogate ``sum(y_hat * direction)``.

Randomness (crop positions, penalty subset and noise) comes from generators
seeded per update (``step_generator``), never from the global RNG.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .distributed import raise_if_any_failed

__all__ = ["DINOv3Backbone", "TokenDiscriminator", "step_generator", "sample_crops", "discriminator_loss",
           "generator_loss", "slope_penalty", "subset_indices", "cap_gradient", "gradient_ratio_cap",
           "discriminator_step", "generator_step"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------------------------- network
class DINOv3Backbone(nn.Module):
    """Frozen DINOv3 ViT returning the patch tokens (no CLS / register tokens) of selected layers."""

    def __init__(self, path, layers=(4, 8, 12), dtype="bfloat16"):
        super().__init__()
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(path), local_files_only=True, use_safetensors=True)
        if model.config.model_type != "dinov3_vit":
            raise ValueError(f"{path} is not a DINOv3 ViT")
        self.layers = tuple(int(layer) for layer in layers)
        self.patch_size = int(model.config.patch_size)
        self.hidden_size = int(model.config.hidden_size)
        self.prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
        self.model = model.to(dtype=DTYPES[dtype]).requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)  # training mode would jitter DINOv3's patch coordinates
        return self

    def forward(self, images):
        """``[N, 3, H, W]`` in ``[-1, 1]`` -> one ``[N, tokens, hidden]`` tensor per layer."""
        x = ((images.float() + 1.0) * 0.5 - self.mean) / self.std
        states = self.model(pixel_values=x.to(next(self.model.parameters()).dtype), output_hidden_states=True,
                            return_dict=True).hidden_states
        return tuple(states[layer][:, self.prefix:, :] for layer in self.layers)


class TokenHead(nn.Module):
    """LayerNorm -> Linear -> SiLU -> Linear(1), applied to every token."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.input.weight, std=0.02)
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens):
        return self.output(self.activation(self.input(self.norm(tokens)))).squeeze(-1)


class TokenHeads(nn.Module):
    """A spatial and a temporal ``TokenHead`` per backbone layer; logits keyed ``{spatial,temporal}/layer_XX``."""

    def __init__(self, layers, hidden_size, head_hidden=128):
        super().__init__()
        self.layers = tuple(layers)
        self.spatial = nn.ModuleDict({f"layer_{i:02d}": TokenHead(hidden_size, head_hidden) for i in self.layers})
        self.temporal = nn.ModuleDict({f"layer_{i:02d}": TokenHead(2 * hidden_size, head_hidden)
                                       for i in self.layers})

    def forward(self, features):
        """``features``: one ``[B, T, tokens, hidden]`` tensor per layer."""
        logits = {}
        for layer, feature in zip(self.layers, features):
            name = f"layer_{layer:02d}"
            # (op order fixes the order in which RGB gradients accumulate through the backbone)
            logits[f"spatial/{name}"] = self.spatial[name](feature.float())
            change = torch.cat((torch.zeros_like(feature[:, :1]), feature[:, 1:] - feature[:, :-1]), dim=1)
            logits[f"temporal/{name}"] = self.temporal[name](torch.cat((feature, change), dim=-1).float())
        return logits


class TokenDiscriminator(nn.Module):
    """Frozen backbone + trainable ``heads``; ``forward`` takes clips ``[B, 3, T, H, W]`` in ``[-1, 1]``."""

    def __init__(self, backbone, head_hidden=128, seed=0):
        super().__init__()
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):  # the same fresh heads whatever the ambient RNG
            torch.manual_seed(int(seed))
            self.heads = TokenHeads(backbone.layers, backbone.hidden_size, head_hidden)

    def forward(self, clips):
        b, c, t, h, w = clips.shape
        features = self.backbone(clips.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return self.heads(tuple(f.reshape(b, t, f.shape[1], f.shape[2]) for f in features))


# ---------------------------------------------------------------------------------------------- crops
def step_generator(base_seed, rank, step, stream, device="cpu"):
    """Generator of one (rank, update, purpose) stream; stateless, hence exactly resumable."""
    seed = (int(base_seed) + 0x1E35A7BD + int(rank) * 0x9E3779B1 + int(step) * 0x85EBCA77
            + int(stream) * 0xC2B2AE3D) & ((1 << 63) - 1)
    return torch.Generator(device=device).manual_seed(seed)


def _randint(high, generator):
    return 0 if high <= 1 else int(torch.randint(0, high, (), generator=generator).item())


def sample_crops(video, clips_per_video, clip_frames, crop_size, generator):
    """Random ``clip_frames x crop_size^2`` crops of ``[B, T, 3, H, W]`` -> ``[B * clips, 3, clip_frames, S, S]``.

    Positions are drawn (start frame, top, left per clip) from the CPU ``generator``.
    """
    b, t, _, h, w = video.shape
    crops = []
    for i in range(b):
        for _ in range(clips_per_video):
            start = _randint(t - clip_frames + 1, generator)
            top = _randint(h - crop_size + 1, generator)
            left = _randint(w - crop_size + 1, generator)
            crop = video[i, start:start + clip_frames, :, top:top + crop_size, left:left + crop_size]
            crops.append(crop.permute(1, 0, 2, 3))
    return torch.stack(crops)


# ---------------------------------------------------------------------------------------------- objectives
def _mean_over_heads(values):
    return torch.stack(tuple(values)).mean()


def discriminator_loss(real, fake, center):
    """``(adversarial, centering)``: per-head means of ``softplus(D(y_hat) - D(y))`` and
    ``center * (D(y) + D(y_hat))^2``."""
    keys = sorted(real)
    adversarial = _mean_over_heads(F.softplus(-(real[k].float() - fake[k].float())).mean() for k in keys)
    centering = _mean_over_heads((real[k].float() + fake[k].float()).square().mean() for k in keys) * float(center)
    return adversarial, centering


def generator_loss(fake, real):
    """Per-head mean of ``softplus(-(D(y_hat) - sg(D(y))))``."""
    return _mean_over_heads(F.softplus(-(fake[k].float() - real[k].detach().float())).mean() for k in sorted(fake))


def slope_penalty(base, perturbed, eps):
    """Per-head mean of ``((D(x + eps n) - D(x)) / eps)^2`` (unweighted)."""
    return _mean_over_heads(((perturbed[k].float() - base[k].float()) / float(eps)).square().mean()
                            for k in sorted(base))


def subset_indices(n, fraction, seed, device):
    """``max(round(n * fraction), 1)`` clip indices from a seeded permutation."""
    count = min(max(round(n * float(fraction)), 1), n)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randperm(n, generator=generator, device=device)[:count]


def cap_gradient(reconstruction_grad, adversarial_grad, weight, ratio, floor=1e-12):
    """Scale ``weight * adversarial_grad`` per sample so its mean |.| is at most ``ratio`` times the
    reconstruction gradient's; returns ``(direction, raw_ratio, applied_ratio)``."""
    dims = tuple(range(1, reconstruction_grad.ndim))
    reference = reconstruction_grad.detach().float()
    raw = adversarial_grad.detach().float() * float(weight)
    reference_l1 = reference.abs().mean(dim=dims, keepdim=True).clamp_min(floor)
    raw_l1 = raw.abs().mean(dim=dims, keepdim=True)
    scale = (reference_l1 * float(ratio) / raw_l1.clamp_min(floor)).clamp(max=1.0)
    applied = raw * scale
    return (applied.to(adversarial_grad.dtype), (raw_l1 / reference_l1).mean(),
            (applied.abs().mean(dim=dims, keepdim=True) / reference_l1).mean())


def gradient_ratio_cap(step, warmup, ramp, ratio):
    """Cap after ``step`` completed updates: 0 during the discriminator-only warm-up, then a linear ramp."""
    if step < warmup:
        return 0.0
    return float(ratio) * (1.0 if ramp == 0 else min(1.0, (step - warmup + 1) / float(ramp)))


# ---------------------------------------------------------------------------------------------- updates
def _average_gradients(parameters):
    if dist.is_initialized() and dist.get_world_size() > 1:
        for p in parameters:
            dist.all_reduce(p.grad)
            p.grad.div_(dist.get_world_size())


def discriminator_step(discriminator, optimizer, fake, real, cfg, subset_seed, noise_generator):
    """One discriminator update on detached ``fake`` / ``real`` crops; returns detached metrics."""
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    error = None
    try:
        optimizer.zero_grad(set_to_none=True)
        fake, real = fake.detach(), real.detach()
        real_logits, fake_logits = discriminator(real), discriminator(fake)
        adversarial, centering = discriminator_loss(real_logits, fake_logits, cfg.center)
        index = subset_indices(real.shape[0], cfg.slope_fraction, subset_seed, real.device)
        real_sub, fake_sub = real.index_select(0, index), fake.index_select(0, index)
        real_noisy = real_sub + float(cfg.slope_epsilon) * torch.randn(
            real_sub.shape, dtype=real_sub.dtype, device=real_sub.device, generator=noise_generator)
        fake_noisy = fake_sub + float(cfg.slope_epsilon) * torch.randn(
            fake_sub.shape, dtype=fake_sub.dtype, device=fake_sub.device, generator=noise_generator)
        select = lambda logits: {k: v.index_select(0, index) for k, v in logits.items()}  # noqa: E731
        r1 = slope_penalty(select(real_logits), discriminator(real_noisy), cfg.slope_epsilon)
        r2 = slope_penalty(select(fake_logits), discriminator(fake_noisy), cfg.slope_epsilon)
        total = adversarial + centering + r1 * float(cfg.slope) + r2 * float(cfg.slope)
        if not torch.isfinite(total):
            raise FloatingPointError(f"discriminator loss {total.item()}")
        total.backward()
    except Exception as exc:  # noqa: BLE001 - reported collectively
        error = exc
    raise_if_any_failed(error, "discriminator forward/backward")
    _average_gradients(parameters)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(cfg.max_grad_norm))
    raise_if_any_failed(None if torch.isfinite(grad_norm) else FloatingPointError("gradient norm"),
                        "discriminator gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    accuracy = _mean_over_heads((real_logits[k] > fake_logits[k]).float().mean() for k in sorted(real_logits))
    return {"d/loss": total.detach(), "d/adversarial": adversarial.detach(), "d/centering": centering.detach(),
            "d/r1": r1.detach(), "d/r2": r2.detach(), "d/accuracy": accuracy, "d/grad_norm": grad_norm}


@contextmanager
def _frozen(parameters):
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def generator_step(discriminator, prediction, fake, real, reconstruction, weight, ratio):
    """Backward of ``reconstruction`` plus the capped adversarial RGB gradient through ``prediction``.

    ``fake`` are crops of ``prediction`` (with graph), ``real`` the real crops. The caller
    clips and applies the decoder's gradients.
    """
    error = None
    try:
        with _frozen(list(discriminator.heads.parameters())):
            with torch.no_grad():
                real_logits = discriminator(real.detach())
            adversarial = generator_loss(discriminator(fake), real_logits)
            rec_grad = torch.autograd.grad(reconstruction, prediction, retain_graph=True)[0]
            adv_grad = torch.autograd.grad(adversarial, prediction)[0]
        direction, raw_ratio, applied_ratio = cap_gradient(rec_grad, adv_grad, weight, ratio)
        surrogate = (prediction.float() * direction.detach().float()).sum()
    except Exception as exc:  # noqa: BLE001 - reported collectively
        error = exc
    raise_if_any_failed(error, "generator adversarial objective")
    (reconstruction + surrogate).backward()
    return {"g/adversarial": adversarial.detach(), "g/raw_gradient_ratio": raw_ratio,
            "g/applied_gradient_ratio": applied_ratio}
