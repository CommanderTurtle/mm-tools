# Standard library imports
import argparse
import collections.abc
import inspect
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from itertools import repeat
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

# Local imports
import random


@dataclass
class BaseStates:
    epoch: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset_epoch_based_states(self):
        pass

    def inc_epoch(self):
        """Increase (image) epoch by 1."""
        self.epoch += 1
        self.reset_epoch_based_states()
        return self.epoch

    def __repr__(self):
        max_key_length = max([len(k) for k in asdict(self).keys()])
        return "\n".join(
            [f"{k:>{max_key_length}}: {v}" for k, v in asdict(self).items()]
        )

    def to_dict(self):
        states = deepcopy(vars(self))
        # Convert defaultdict to dict
        for key, value in states.items():
            if isinstance(value, defaultdict):
                states[key] = dict(value)

        return states

    @classmethod
    def from_pretrained(
        cls,
        scalar_state,
        rank=None,
        world_size=None,
        default_rank0_ss=None,
        default=None,
    ):
        _scalar_state = default or {}
        if isinstance(scalar_state, dict):
            _scalar_state.update(scalar_state)
        elif isinstance(scalar_state, list):
            assert (
                rank is not None
                and world_size is not None
                and default_rank0_ss is not None
            ), (
                f"rank({rank}), world_size({world_size}), and default_rank0_ss({default_rank0_ss}) should be provided "
                f"when scalar_state is a list."
            )
            if len(scalar_state) != world_size:
                # Check if scalar_state all the same
                if all([scalar_state[0] == x for x in scalar_state]):
                    default_rank0_ss = True

                if default_rank0_ss:
                    from loguru import logger

                    logger.info(
                        f" len(scalar_state)={len(scalar_state)} != world_size={world_size}, Set default rank0 scalar state."
                    )
                    _scalar_state.update(scalar_state[0])
                else:
                    _scalar_state.update(scalar_state[0])
                    for i in range(len(scalar_state)):
                        if isinstance(scalar_state[i], dict):
                            for k, v in scalar_state[i].items():
                                if isinstance(v, dict):
                                    _scalar_state[k].update(v)
                    from loguru import logger

                    logger.info(
                        f"default_rank0_ss is alse, Update scalar_state={_scalar_state}"
                    )
            else:
                _scalar_state.update(scalar_state[rank])
        else:
            raise ValueError(f"Unknown scalar state type: {type(scalar_state)}")

        return cls(**_scalar_state)


@dataclass
class ScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    """

    # - Global level
    epoch: dict = field(default_factory=lambda: defaultdict(int))
    train_steps: int = 0  # Accumulated training steps
    update_steps: int = 0  # Accumulated update steps
    current_run_update_steps: int = (
        0  # Update steps in current run (Reset for every resume)
    )
    consumed_samples_total: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed samples
    epoch_consumed_samples: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed samples

    consumed_samples_per_dp: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed samples per data-parallel group
    consumed_groups_per_dp: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed packing groups per data-parallel group (token packing resume)
    consumed_tokens_total: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed tokens
    consumed_computations_attn: float = (
        0  # Accumulated consumed computations of attention + mlp
    )
    consumed_computations_total: float = 0  # Accumulated consumed computations of total

    # - Data type statistics for t2v mask type
    consumed_samples_by_data_type_total: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed samples by data type for t2v
    consumed_samples_by_data_type_per_dp: dict = field(
        default_factory=lambda: defaultdict(int)
    )  # Accumulated consumed samples by data type per data-parallel group for t2v

    lr: float = 0.0  # Current learning rate


@dataclass
class CycleStates(BaseStates):
    log_steps: int = 0
    running_loss: float = 0
    running_tokens: dict = field(default_factory=lambda: defaultdict(int))
    running_samples: dict = field(default_factory=lambda: defaultdict(int))
    # Add data_type tracking for t2v mask type
    running_loss_by_data_type: dict = field(default_factory=lambda: defaultdict(float))
    running_loss_count_by_data_type: dict = field(
        default_factory=lambda: defaultdict(int)
    )
    running_samples_by_data_type: dict = field(default_factory=lambda: defaultdict(int))
    # Per-t-range loss tracking
    running_loss_by_t_bin: dict = field(default_factory=lambda: defaultdict(float))
    running_count_by_t_bin: dict = field(default_factory=lambda: defaultdict(int))

    def reset_epoch_based_states(self):
        self.log_steps = 0
        self.running_loss = 0
        self.running_tokens = defaultdict(int)
        self.running_samples = defaultdict(int)
        self.running_loss_by_data_type = defaultdict(float)
        self.running_loss_count_by_data_type = defaultdict(int)
        self.running_samples_by_data_type = defaultdict(int)
        self.running_loss_by_t_bin = defaultdict(float)
        self.running_count_by_t_bin = defaultdict(int)

    def add_data_type_loss(self, data_type, loss_value, samples):
        """Add loss and samples for specific data_type"""
        self.running_loss_by_data_type[data_type] += loss_value
        self.running_loss_count_by_data_type[data_type] += 1
        self.running_samples_by_data_type[data_type] += samples

    def get_avg_loss_by_data_type(self):
        """获取各data_type的平均loss"""
        avg_losses = {}
        for data_type in self.running_loss_by_data_type.keys():
            count = self.running_loss_count_by_data_type[data_type]
            if count > 0:
                avg_losses[data_type] = (
                    self.running_loss_by_data_type[data_type] / count
                )
            else:
                avg_losses[data_type] = 0.0
        return avg_losses


def prepare_model_inputs(
    batch,
    device,
    model,
    vae,
    text_encoder,
    args=None,
    extra_model=None,
    seed=None,
):
    """Prepare inputs for model training/inference."""
    media, latents, *batch_args = batch
    rank = dist.get_rank() if dist.is_initialized() else 0

    if len(batch_args) == 3:
        text_ids, text_mask, kwargs = batch_args
    else:
        raise ValueError(f"Unexpected batch_args.")

    # Move batch to device
    media = media.to(device)
    latents = latents.to(device)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    # ======================================== Encode media ======================================
    # Used for 3D VAE with 2D inputs(image).
    # Prepare media shape for 2D/3D VAE
    if len(latents.shape) == 1:
        if len(media.shape) == 4:
            # media is a batch of image with shape [b, c, h, w]
            media = media.unsqueeze(2)  # [b, c, 1, h, w]
        elif len(media.shape) != 5:
            raise ValueError(
                f"Only support media with shape (b, c, h, w) or (b, c, f, h, w), but got {media.shape}."
            )

        vae_dtype = torch.float16
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32
        ):
            if args.vae.vae_encode_tile_size > 0:
                all_latents = []
                for media_chunk in torch.split(media, args.vae.vae_encode_tile_size):
                    latents_chunk = vae.encode(media_chunk).latent_dist.mode()
                    all_latents.append(latents_chunk)
                latents = torch.cat(all_latents)
            else:
                latents = vae.encode(media).latent_dist.mode()
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
                latents.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
            else:
                latents.mul_(vae.config.scaling_factor)
    elif len(latents.shape) == 5 or len(latents.shape) == 4:  # Using video/image cache
        latents = (
            latents * vae.config.scaling_factor
        )  # vae cache is not multiplied by scaling_factor
    else:
        raise ValueError(
            f"Only support media/latent with shape (b, c, h, w) or (b, c, f, h, w), but got {media.shape} {latents.shape}."
        )

    # to remove for next version
    reorg_token = "888" in args.dit.model_type or "16168" in args.dit.model_type
    if reorg_token:
        latents = torch.cat((latents[:, :, :1], latents), dim=2)
        latents = rearrange(latents, "b c f h w -> b f c h w")
        latents = rearrange(latents, "b (f n) c h w -> b f (n c) h w", n=2)
        latents = rearrange(latents, "b f c h w -> b c f h w").contiguous()

    # ======================================== Encode text ======================================
    # Autocast is handled by text_encoder itself.
    # Whether to apply text_mask is determined by args.text_encoder.use_attention_mask.
    text_outputs = text_encoder.encode(
        {"input_ids": text_ids, "attention_mask": text_mask},
        data_type=batch_args[-1]["type"][0],
    )
    text_states = text_outputs.hidden_state
    text_mask = text_outputs.attention_mask

    # ======================================== Build RoPE ======================================
    # Import here to avoid circular import
    from hymm.models.dit.modules.posemb_layers import get_nd_rotary_pos_embed

    target_ndim = 3  # n-d RoPE
    ndim = len(latents.shape) - 2
    latents_size = list(latents.shape[-ndim:])
    assert all(s % model.patch_size[i] == 0 for i, s in enumerate(latents_size)), (
        f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
        f"but got {latents_size}."
    )
    rope_sizes = [s // model.patch_size[i] for i, s in enumerate(latents_size)]
    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = model.hidden_size // model.heads_num
    rope_dim_list = model.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert sum(rope_dim_list) == head_dim, (
        "sum(rope_dim_list) should equal to head_dim of attention layer"
    )
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list=rope_dim_list,
        start=rope_sizes,
        theta=args.dit.rope_theta,
        use_real=True,
        theta_rescale_factor=1.0,
    )
    model_kwargs = dict(
        text_states=text_states,  # [b, 256, 4096]
        encoder_attention_mask=text_mask,  # [b, 256]
        vision_states=torch.zeros(
            latents.shape[0], 1, model.hidden_size, device=device
        ),
        return_dict=False,
    )

    model_kwargs["extra_kwargs"] = {}

    if latents.shape[2] == 1:
        data_type = "image"
    else:
        data_type = "video"
    return latents, model_kwargs, data_type, freqs_cos.shape[0]


def _ntuple(n):
    """Create a function that converts input to n-tuple."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


# Convenience functions for common tuple sizes
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    """Convert input to tuple."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    """Convert input to list of 2-tuples."""
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst


def dynamic_values_wrapper(anchor_sizes, values):
    """
    Dynamically mapping the size (w, h) to one of the list batch sizes.

    Args:
        anchor_sizes (tuple): list of anchor sizes.
        values (tuple): list of values for different image scale.
    """
    if not isinstance(anchor_sizes, (list, tuple)):
        raise ValueError(
            f"anchor_sizes should be a list or tuple, but got {type(anchor_sizes)}"
        )
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"batch_sizes should be a list or tuple, but got {type(values)}"
        )
    if len(anchor_sizes) != len(values):
        raise ValueError(
            f"anchor_sizes and values should have the same length, but got {len(anchor_sizes)} and "
            f"{len(values)}."
        )
    anchor_sizes = np.array(anchor_sizes)

    def get_value(size):
        sqrt_area = (size[0] * size[1]) ** 0.5
        val = values[np.argmin(np.abs(anchor_sizes - sqrt_area))]
        return val

    return get_value


def normalize_dit_input(model_type, latents):
    """Normalize DiT input based on model type."""
    if model_type == "hunyuan":
        return latents * 0.476986
    else:
        raise NotImplementedError(f"model_type {model_type} not supported")


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    """Create a linear-quadratic schedule for noise."""
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.
    Handles custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def str2bool(s):
    if isinstance(s, bool):
        return s
    elif s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected !")


def str2null(s):
    if s.lower() == "none" or s.lower() == "None" or s.lower() == "null":
        return None
    else:
        return s


def int2null(s):
    if s.lower() == "none" or s.lower() == "None" or s.lower() == "null":
        return None
    else:
        return int(s)
