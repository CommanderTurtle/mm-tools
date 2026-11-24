"""The shared random state: one draw per image, read by every code position.

Each position is masked with probability MASK_PROB or holds a uniformly random code; Gaussian noise is mixed into all.
"""
import math

import torch

from . import config


def sample_random_state(mask_emb, codebook_std, batch, generator=None, rho=config.MASK_PROB, sigma=config.STATE_NOISE):
    """mask_emb [C], codebook_std [K, C] -> (state [batch, N, C], masked share of each row [batch])"""
    n, dev = config.NUM_CODES, codebook_std.device
    masked = torch.rand(batch, n, generator=generator, device=dev) < rho
    codes = torch.randint(0, codebook_std.shape[0], (batch, n), generator=generator, device=dev)
    state = torch.where(masked.unsqueeze(-1), mask_emb.view(1, 1, -1).float().expand(batch, n, -1), codebook_std[codes])
    state = math.sqrt(1 - sigma ** 2) * state + sigma * torch.randn(state.shape, generator=generator, device=dev)
    return state, masked.float().mean(1)
