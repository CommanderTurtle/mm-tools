"""Base flow-matching transport and shared enums/utilities.

Contains the ``Transport`` base class, enum types (ModelType, PathType,
WeightType, SNRType), and shared enum-string conversion maps used by all
transport factory functions.
"""

import enum

import torch as th

from hymm.trainer.noise_scheduler import (
    ICPlan,
    sample_with_snr,
    check_interval,
    get_model_t,
    compute_timestep_shift,
)

__all__ = [
    "ModelType",
    "PathType",
    "WeightType",
    "SNRType",
    "Transport",
    "_PATH_TYPE_MAP",
    "_LOSS_TYPE_MAP",
    "_MODEL_TYPE_MAP",
    "_SNR_TYPE_MAP",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class SNRType(enum.Enum):
    UNIFORM = enum.auto()
    LOGNORM = enum.auto()
    MIX = enum.auto()
    # Logit-normal on [t_split, 1.0] mixed with uniform on [0, t_split].
    # Defaults: t_split=0.2, p_low=0.2 (see sample_with_snr).
    # Goal: keep SD3/Flux-style lognorm emphasis on the noisy region while
    # uniformly covering the near-clean low-t region (which lognorm under-
    # samples), useful for AR / DF cache snapshots near t=0.
    LOGNORM_UNIFORM_LOWT = enum.auto()


# ---------------------------------------------------------------------------
# Shared string -> enum conversion maps (used by DF / TF / bidirectional factories)
# ---------------------------------------------------------------------------

_PATH_TYPE_MAP = {"Linear": PathType.LINEAR, "GVP": PathType.GVP, "VP": PathType.VP}
_LOSS_TYPE_MAP = {
    "velocity": WeightType.NONE,
    "noise": WeightType.NONE,
    "score": WeightType.VELOCITY,
}
_MODEL_TYPE_MAP = {
    "velocity": ModelType.VELOCITY,
    "noise": ModelType.NOISE,
    "score": ModelType.SCORE,
}
_SNR_TYPE_MAP = {
    "uniform": SNRType.UNIFORM,
    "lognorm": SNRType.LOGNORM,
    "mix": SNRType.MIX,
    "lognorm_uniform_lowt": SNRType.LOGNORM_UNIFORM_LOWT,
}


# ---------------------------------------------------------------------------
# Base Transport
# ---------------------------------------------------------------------------


class Transport:
    """Base flow-matching transport.

    Provides shared initialisation, noise sampling, and timestep helpers.
    Subclasses must implement ``training_losses()`` and ``forward_step()``.
    """

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        snr_type,
        training_timesteps=1000,
        reverse_time_schedule=False,
        video_shift=1.0,
        reverse=False,
    ):
        path_options = {
            PathType.LINEAR: ICPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type](reverse=reverse)
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.video_shift = video_shift
        self.reverse = reverse

        self.training_timesteps = training_timesteps
        self.reverse_time_schedule = reverse_time_schedule

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        return check_interval(
            self.path_sampler,
            train_eps,
            sample_eps,
            model_type=self.model_type,
            diffusion_form=diffusion_form,
            sde=sde,
            reverse=reverse,
            eval=eval,
            last_step_size=last_step_size,
        )

    def sample(self, x1):
        """Sample x0, t based on shape of x1."""
        if isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        t = sample_with_snr(
            len(x1), self.snr_type, t0, t1, self.video_shift, self.reverse
        )
        t = t.to(x1[0])
        return t, x0, x1

    def get_model_t(self, t):
        return get_model_t(t, self.training_timesteps, self.reverse_time_schedule)

    def _get_timestep_shift(self, seq_len: int) -> float:
        return compute_timestep_shift(
            seq_len,
            base_shift=getattr(self, "timestep_shift", 0.0),
            adjust=getattr(self, "adjust_timestep_shift", False),
        )

    def training_losses(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement training_losses()")

    def forward_step(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward_step()")
