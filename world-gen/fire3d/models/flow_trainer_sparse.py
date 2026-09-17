from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
from modules import sparse as sp
from modules.sampler.flow_euler import FlowEulerSampler, FlowEulerCfgSampler
from torch import nn

class FlowMatchingTrainer:
    """
    Trainer for diffusion model with flow matching objective.
    """
    def __init__(
        self,
        denoiser: nn.Module,
        trainer_name: str,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
    ):
        self.trainer_name = trainer_name
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.denoiser = denoiser

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond):
        """
        Get the conditioning data.
        """
        return cond

    def get_sampler(self) -> FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return FlowEulerSampler(self.sigma_min)

    def sample_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size, device=device, dtype=dtype)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_step(
        self,
        x_0: torch.Tensor,
        cond,
        cond_mask=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x num_cond x cond_channels] tensor of additional conditions.
            cond_mask: The [N x num_cond] tensor of boolean mask for the additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0], device=x_0.device, dtype=x_0.dtype)
        x_t = self.diffuse(x_0, t, noise=noise)

        pred = self.denoiser(x_t, t * 1000, cond, cond_mask, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        metrics_dict = {}

        loss = F.mse_loss(pred, target)

        # # Log loss with time bins - fully vectorized on GPU to avoid CPU sync
        # # Compute per-instance MSE without Python loop
        # # Flatten spatial dims and compute mean over non-batch dimensions
        # batch_size = x_0.shape[0]
        # pred_flat = pred.view(batch_size, -1)
        # target_flat = target.view(batch_size, -1)
        # mse_per_instance = ((pred_flat - target_flat) ** 2).mean(dim=1)  # [N]

        # # Compute time bins on GPU
        # bin_edges = torch.linspace(0, 1, 11, device=t.device, dtype=t.dtype)
        # # bucketize returns bin indices (0 to 10), we want 0 to 9 for 10 bins
        # time_bin = torch.bucketize(t, bin_edges[1:-1])  # [N], values 0-9

        # # Compute mean MSE per bin using scatter
        # for i in range(10):
        #     bin_mask = (time_bin == i)
        #     if bin_mask.any():
        #         # Use masked_select and mean - all on GPU, single .item() call per bin
        #         metrics_dict[f"{self.trainer_name}_bin_{i}_mse"] = mse_per_instance[bin_mask].mean().item()

        return loss, metrics_dict

    @torch.no_grad()
    def run_inference(
        self,
        cond,
        cond_mask=None,
        steps=50,
    ) -> torch.Tensor:

        # inference
        sampler = self.get_sampler()
        N = cond.shape[0]
        noise = torch.randn((N, self.denoiser.in_channels, *[self.denoiser.resolution] * 3), device=cond.device, dtype=cond.dtype)
        res = sampler.sample(
            self.denoiser,
            noise=noise,
            cond=cond,
            cond_mask=cond_mask,
            steps=steps
        )
        sample = res.samples

        return sample


class SparseFlowMatchingCFGTrainer(FlowMatchingTrainer):
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, inference=False, **kwargs):
        """
        Get the conditioning data.
        cond: list of [num_cond x cond_channels] tensors
        """
        neg_cond_template = torch.zeros_like(cond[0][:1, :])  # [1, cond_dim], single tensor
        B = len(cond)
        neg_cond = [neg_cond_template] * B  # list of references to same tensor

        if self.p_uncond > 0 and not inference:
            # randomly drop the class label
            def select(cond, neg_cond, mask):
                # Copy cond then overwrite only where mask is True (loop over ~p_uncond*B, not B)
                result = list(cond)
                for i in np.flatnonzero(mask):
                    result[i] = neg_cond[i]
                return result

            mask = np.random.rand(B) < self.p_uncond
            # keep at least one condition randomly selected if all are dropped
            if np.all(mask):
                mask[np.random.randint(B)] = False

            mask = list(mask)
            cond = select(cond, neg_cond, mask)

        return cond, neg_cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}

    def get_sampler(self, **kwargs) -> FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return FlowEulerCfgSampler(self.sigma_min)

    def training_step(
        self,
        x_0: sp.SparseTensor,
        cond,
        concat_cond: sp.SparseTensor = None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The list of [num_cond x cond_channels] N tensors of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0], device=x_0.device, dtype=x_0.dtype)
        x_t = self.diffuse(x_0, t, noise=noise)

        cond, _ = self.get_cond(cond, inference=False)

        pred = self.denoiser(x_t, t * 1000, cond, concat_cond=concat_cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        metrics_dict = {}

        loss = F.mse_loss(pred.feats, target.feats)

        return loss, metrics_dict

    @torch.no_grad()
    def run_inference(
        self,
        x_0: sp.SparseTensor,
        cond,
        concat_cond: sp.SparseTensor = None,
        steps=50,
        guidance_strength=None,
    ) -> sp.SparseTensor:

        # inference
        sampler = self.get_sampler()
        noise = x_0.replace(torch.randn_like(x_0.feats))
        cond, neg_cond = self.get_cond(cond, inference=True)
        # None keeps the sampler's own default (3.0), so the shipped protocol
        # is untouched unless a caller asks for a different guidance strength.
        extra = {} if guidance_strength is None else {
            "guidance_strength": float(guidance_strength)
        }
        res = sampler.sample(
            self.denoiser,
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
            concat_cond=concat_cond,
            steps=steps,
            **extra,
        )
        sample = res.samples

        return sample
