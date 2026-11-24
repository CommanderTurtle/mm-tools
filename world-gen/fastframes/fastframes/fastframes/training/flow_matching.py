#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Flow-matching pieces shared by the causal-teacher and consistency stages.

Noise grids are computed in FP32 on the device of the batch, as in the original
runs (``torch.linspace`` is not bit-identical across devices).
"""
from __future__ import annotations

import hashlib

import torch

__all__ = ["NOISE", "TIMESTEP", "CFG_DROPOUT", "CD_INTERVAL", "shifted_sigmas", "bell_weights", "add_noise",
           "sample_generator", "learning_rate"]

# Random components of one training sample (see ``sample_generator``).
NOISE, TIMESTEP, CFG_DROPOUT, CD_INTERVAL = 1, 2, 3, 5


def shifted_sigmas(levels, shift, device):
    """``levels`` positive noise levels ``shift(linspace(1, 0, levels + 1)[:-1])`` (FP32).

    ``levels=1000`` is the training grid; ``levels=48`` the consistency grid.
    """
    base = torch.linspace(1.0, 0.0, levels + 1, dtype=torch.float32, device=device)[:-1]
    return shift * base / (1.0 + (shift - 1.0) * base)


def bell_weights(sigmas, num_train_timesteps=1000):
    """Per-timestep loss weights ``exp(-2 ((t - T/2) / T)^2)``, shifted to zero minimum and mean one."""
    t = sigmas * num_train_timesteps
    n = float(num_train_timesteps)
    raw = torch.exp(-2.0 * ((t - n / 2.0) / n) ** 2)
    shifted = raw - raw.min()
    return shifted * (n / shifted.sum())


def add_noise(clean, noise, sigma):
    """``(1 - sigma) x0 + sigma * noise`` in FP32, returned in the noise dtype; ``sigma``: ``[B, F]``."""
    sigma = sigma.float()[..., None, None, None]
    return ((1.0 - sigma) * clean.float() + sigma * noise.float()).to(noise.dtype)


def sample_generator(stream, seed, epoch, sample_id, component, device="cpu"):
    """Generator for one random component of one training sample in one corpus epoch.

    Seeds are a hash of ``(stream, seed, epoch, sample_id, component)``, so the draws do
    not depend on data order, world size or resumption.
    """
    payload = f"{stream}\0{seed}\0{epoch}\0{sample_id}\0{component}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"ct14b-v1").digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest, "little") & ((1 << 63) - 1))


def learning_rate(base_lr, completed_updates, warmup_updates):
    """Linear warm-up over the first ``warmup_updates`` updates, then constant."""
    return base_lr * min((completed_updates + 1) / max(warmup_updates, 1), 1.0)
