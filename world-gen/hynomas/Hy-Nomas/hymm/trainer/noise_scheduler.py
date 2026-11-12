"""Noise scheduling utilities for flow-matching training.

Shared building blocks used by all transport classes (default, teacher
forcing, diffusion forcing) to sample timesteps, apply shifts,
construct noisy inputs, and define interpolation paths.
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Interpolation path (linear flow matching)
# ---------------------------------------------------------------------------


def expand_t_like_x(t, x):
    """Reshape time ``t`` to be broadcastable against ``x``.

    Args:
        t: ``[batch_dim]`` time vector.
        x: ``[batch_dim, ...]`` data point.
    """
    dims = [1] * len(x[0].size())
    t = t.view(t.size(0), *dims)
    return t


class ICPlan:
    """Linear Coupling Plan for flow matching.

    Defines the interpolation path ``x_t = alpha_t * x1 + sigma_t * x0``
    where ``x1`` is data, ``x0`` is noise.

    Convention (non-reverse): t=0 → pure noise, t=1 → pure data.
    """

    def __init__(self, sigma=0.0, reverse=False):
        self.sigma = sigma
        self.reverse = reverse

    def compute_alpha_t(self, t):
        """Data coefficient and its time derivative."""
        if self.reverse:
            return 1 - t, -1
        return t, 1

    def compute_sigma_t(self, t):
        """Noise coefficient and its time derivative."""
        if self.reverse:
            return t, 1
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1 / t

    def compute_drift(self, x, t):
        """SDE drift (score parametrisation)."""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }
        try:
            return choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * velocity - x) / var

    def get_noise_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        return (reverse_alpha_ratio * velocity - x) / var

    def get_velocity_from_score(self, score, x, t):
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        return var * score - drift

    def compute_mu_t(self, t, x0, x1):
        """Mean of time-dependent density p_t."""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        if isinstance(x1, (list, tuple)):
            return [alpha_t[i] * x1[i] + sigma_t[i] * x0[i] for i in range(len(x1))]
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        """Vector field corresponding to p_t."""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        if isinstance(x1, (list, tuple)):
            return [d_alpha_t * x1[i] + d_sigma_t * x0[i] for i in range(len(x1))]
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------


def sample_timesteps(
    shape: Tuple[int, ...],
    device: torch.device,
    method: str = "uniform",
) -> torch.Tensor:
    """Sample raw timesteps in (0, 1).

    Args:
        shape: Output shape, e.g. ``(B,)`` or ``(B, num_chunks)``.
        device: Target device.
        method: ``"uniform"`` or ``"logit_normal"``.
    """
    if method == "logit_normal":
        return torch.sigmoid(torch.randn(shape, device=device))
    if method == "uniform":
        return torch.rand(shape, device=device)
    raise ValueError(f"Unknown timestep_sample_method: {method}")


def sample_with_snr(
    batch_size: int,
    snr_type,
    t0: float,
    t1: float,
    shift: float,
    reverse: bool,
) -> torch.Tensor:
    """Sample timesteps with SNR-aware distribution and optional shift.

    This is the core sampling logic extracted from ``Transport.sample()``.
    *snr_type* should be one of the ``SNRType`` enum values.

    Returns:
        Timestep tensor of shape ``(batch_size,)``.
    """
    from hymm.trainer.transport.transport_base import SNRType

    if snr_type == SNRType.UNIFORM:
        t = torch.rand((batch_size,)) * (t1 - t0) + t0
    elif snr_type == SNRType.LOGNORM:
        u = torch.normal(mean=0.0, std=1.0, size=(batch_size,))
        t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
    elif snr_type == SNRType.MIX:
        u = torch.normal(mean=0.0, std=1.0, size=(batch_size,))
        t_lognorm = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
        t0_ = t0 + 0.05
        t1_ = t1 - 0.05
        t_clip_uniform = torch.rand((batch_size,)) * (t1_ - t0_) + t0_
        mask = (torch.rand((batch_size,)) > 0.3).float()
        t = mask * t_lognorm + (1 - mask) * t_clip_uniform
    elif snr_type == SNRType.LOGNORM_UNIFORM_LOWT:
        # Mixture distribution over t:
        #   - with prob (1 - p_low):  standard logit-normal on [t0, t1]
        #     (identical to SNRType.LOGNORM, full range, no squeezing).
        #   - with prob p_low:        uniform on the low-t region [t0, t_split].
        # Convention in this codebase: x_t = (1 - t)*clean + t*noise, so the
        # low-t region is the near-clean region. The uniform branch densely
        # covers it (which logit-normal under-samples) without distorting
        # the lognorm branch's shape.
        t_split = 0.2
        p_low = 0.2
        # Standard logit-normal on the full [t0, t1] range.
        u = torch.normal(mean=0.0, std=1.0, size=(batch_size,))
        t_lognorm = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
        # Uniform on [t0, t_split].
        t_uniform = torch.rand((batch_size,)) * (t_split - t0) + t0
        # mask=1 -> use lognorm sample; mask=0 -> uniform low-t.
        mask = (torch.rand((batch_size,)) >= p_low).float()
        t = mask * t_lognorm + (1 - mask) * t_uniform
    else:
        raise ValueError(f"Unknown snr type: {snr_type}")

    if shift != 1.0:
        if reverse:
            t_ = (shift * t) / (1 + (shift - 1) * t)
        else:
            t_ = t / (shift - (shift - 1) * t)

        if snr_type == SNRType.MIX:
            t = mask * t_ + (1 - mask) * t
        elif snr_type == SNRType.LOGNORM_UNIFORM_LOWT:
            # Apply shift only to the lognorm (high-t) branch; the uniform
            # low-t branch is intentionally left unshifted because shift is
            # designed to push mass toward high noise and is meaningless in
            # the near-clean region we are explicitly oversampling.
            t = mask * t_ + (1 - mask) * t
        else:
            t = t_

    return t


# ---------------------------------------------------------------------------
# Timestep shift / scaling
# ---------------------------------------------------------------------------


def apply_timestep_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply flow-matching timestep shift: ``t' = t * s / (1 + (s-1)*t)``."""
    if shift > 0:
        return t * shift / (1 + (shift - 1) * t)
    return t


def get_model_t(
    t: torch.Tensor,
    training_timesteps: int = 1000,
    reverse_time_schedule: bool = False,
) -> torch.Tensor:
    """Scale timestep ``t`` in [0, 1] to model input range."""
    if reverse_time_schedule:
        return (1 - t) * training_timesteps
    return t * training_timesteps


def compute_timestep_shift(
    seq_len: int,
    base_shift: float = 0.0,
    adjust: bool = False,
) -> float:
    """Compute timestep shift, optionally adjusting based on sequence length."""
    if not adjust:
        return base_shift
    x1, x2 = 32760, 73710
    y1, y2 = 3.0, 5.0
    log_y1, log_y2 = math.log(y1), math.log(y2)
    k = (log_y2 - log_y1) / (x2 - x1)
    b = log_y1 - k * x1
    return math.exp(k * seq_len + b)


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    """Return a linear function mapping x -> y (copied from Flux.1)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    """Flux.1-style time shift."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def check_interval(
    path_sampler,
    train_eps,
    sample_eps,
    *,
    model_type=None,
    diffusion_form="SBDM",
    sde=False,
    reverse=False,
    eval=False,
    last_step_size=0.0,
):
    """Compute (t0, t1) integration bounds for a given path sampler."""
    t0 = 0
    t1 = 1
    eps = train_eps if not eval else sample_eps
    is_vp = type(path_sampler).__name__ == "VPCPlan"
    is_ic_or_gvp = type(path_sampler).__name__ in ("ICPlan", "GVPCPlan")

    if is_vp:
        t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
    elif is_ic_or_gvp and (
        model_type is not None and str(model_type) != "ModelType.VELOCITY" or sde
    ):
        t0 = (
            eps
            if (diffusion_form == "SBDM" and sde)
            or (model_type is not None and str(model_type) != "ModelType.VELOCITY")
            else 0
        )
        t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

    if reverse:
        t0, t1 = 1 - t0, 1 - t1

    return t0, t1


# ---------------------------------------------------------------------------
# Noising
# ---------------------------------------------------------------------------


def add_flow_noise(
    x_clean: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flow-matching forward process: ``x_t = (1-t)*x_1 + t*noise``.

    *t* is broadcast-expanded to match the trailing dimensions of *x_clean*.

    Returns:
        ``(x_noisy, noise)``
    """
    if noise is None:
        noise = torch.randn_like(x_clean)
    t_expanded = t
    while t_expanded.dim() < x_clean.dim():
        t_expanded = t_expanded.unsqueeze(-1)
    return (1 - t_expanded) * x_clean + t_expanded * noise, noise


# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------


def logit_normal_integral(
    t: torch.Tensor,
    num_train_timestep: int,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Logit-normal integral weighting for loss rebalancing."""
    t_normalized = t / num_train_timestep
    eps = 1e-5
    t_normalized = torch.clamp(t_normalized, eps, 1 - eps)
    logit_t = torch.log(t_normalized / (1 - t_normalized))
    log_pdf = -0.5 * ((logit_t - mean) / std) ** 2
    pdf = torch.exp(log_pdf) / (std * math.sqrt(2 * math.pi))
    jacobian = 1.0 / (t_normalized * (1 - t_normalized))
    weight = 1.0 / (pdf * jacobian + eps)
    weight = weight / weight.mean()
    return weight


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))
