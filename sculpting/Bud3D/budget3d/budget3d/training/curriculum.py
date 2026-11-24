"""The training layout curriculum: which budget and which layout each step trains on.

One checkpoint has to serve every budget from 512 tokens (6.25 %) to 8192 (100 %), so the
budget is a *training variable*.  Four modes are mixed, and the salience that places the
tokens is sometimes replaced by content-independent noise so the coarse and fine adapters
stay calibrated everywhere in [0, 1]^3 rather than only where Objaverse objects put detail.

The budget and the two split counts are drawn once per micro-batch and the *locations* stay
per-example.  Every sequence in a batch therefore has the same length and the same
per-scale token counts -- no padding tokens, no varlen kernels, and the training layout is
built by the same code path as the inference layout.
"""
import math

import torch

from ..octree import (N_CUBE, N_SOBOL, THRESHOLDS, build_layout, dense_layout, detail_scores,
                      random_scores, snap_realizable_tokens, threshold_layout_tokens, uniform_layout)


class LayoutSampler:
    def __init__(self, anchor_p=0.12, uniform_coarse_p=0.10, budget_p=0.65, random_placement_p=0.15,
                 offset_lo=-0.5, offset_hi=2.5, noise_layout_early_k=5, noise_layout_early_p=0.35,
                 thresholds=THRESHOLDS, seed=0, device="cuda"):
        self.anchor_p, self.uniform_p, self.budget_p = anchor_p, uniform_coarse_p, budget_p
        self.random_p = random_placement_p
        self.offset_lo, self.offset_hi = offset_lo, offset_hi
        self.early_k, self.early_p = noise_layout_early_k, noise_layout_early_p
        self.thresholds = thresholds
        self.gen = torch.Generator(device="cpu").manual_seed(seed + 7919)
        self.device = device

    def _u(self):
        return float(torch.rand(1, generator=self.gen).item())

    def reference_latent(self, z_t, z0_hat, k):
        """Which latent the layout is read from, per example.

        Normally the running clean estimate; at k = 0 there is none, and for the first few
        steps the layout is sometimes taken from the noisy state so the student is hardened
        against the noisy layouts that 0-warmup inference starts from.
        """
        use_noisy = (k == 0)
        early = (k < self.early_k) & (torch.rand(k.shape, generator=self.gen).to(k.device) < self.early_p)
        pick = (use_noisy | early).reshape(-1, 1, 1)
        return torch.where(pick, z_t, z0_hat)

    def __call__(self, z_ref_oct, k, perm=None):
        """z_ref_oct [B, 16, 16, 16, 2, C] -> (Layout, mode name)."""
        b = z_ref_oct.shape[0]
        dev = z_ref_oct.device
        r = self._u()
        if r < self.anchor_p:
            return dense_layout(b, dev), "anchor"
        if r < self.anchor_p + self.uniform_p:
            scale = 2 if self._u() < 0.5 else 4
            return uniform_layout(b, scale, dev), f"uniform{scale}"

        s4, s2 = detail_scores(z_ref_oct)
        if self._u() < self.random_p:
            s4, s2 = random_scores(b, dev, generator=None)

        if r < self.anchor_p + self.uniform_p + self.budget_p:
            lo = N_CUBE / N_SOBOL
            b_frac = math.exp(math.log(lo) + (math.log(1.0) - math.log(lo)) * self._u())
            n = snap_realizable_tokens(b_frac * N_SOBOL)
            mode = "budget"
        else:
            delta = self.offset_lo + (self.offset_hi - self.offset_lo) * self._u()
            n = threshold_layout_tokens(s4, s2, delta, self.thresholds)
            mode = "offset"
        return build_layout(s4, s2, n, self.thresholds), mode
