# Flow-matching noise schedule used by Wan2.1 and the few-step generators.
# Adapted from DiffSynth-Studio / Self-Forcing (Apache-2.0).
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ACKNOWLEDGEMENTS.
import torch

__all__ = ["FlowMatchScheduler"]


class FlowMatchScheduler:
    """Rectified-flow schedule ``x_t = (1 - sigma) x_0 + sigma * noise``.

    ``num_inference_steps`` sigmas from 1 towards 0 (the final 0 excluded), shifted by
    ``shift``, on a grid of 1000 training timesteps; a timestep ``t`` maps to the nearest
    grid entry.
    """

    def __init__(self, num_inference_steps=1000, num_train_timesteps=1000, shift=5.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100):
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def _index(self, timestep, timesteps):
        return torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = self._index(timestep, self.timesteps)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        return sample + model_output * (sigma_ - sigma)

    def add_noise(self, original_samples, noise, timestep):
        """``[N, C, H, W]`` samples/noise with ``[N]`` timesteps."""
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        sigma = self.sigmas[self._index(timestep, self.timesteps)].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    def flow_to_x0(self, flow, xt, timestep):
        """``x0 = x_t - sigma_t * flow`` in float64; inputs ``[N, C, H, W]``, timestep ``[N]``."""
        dtype = flow.dtype
        flow, xt, sigmas, timesteps = (
            x.double().to(flow.device) for x in (flow, xt, self.sigmas, self.timesteps))
        sigma = sigmas[self._index(timestep, timesteps)].reshape(-1, 1, 1, 1)
        return (xt - sigma * flow).to(dtype)

    def x0_to_flow(self, x0, xt, timestep):
        """``flow = (x_t - x0) / sigma_t`` in float64."""
        dtype = x0.dtype
        x0, xt, sigmas, timesteps = (
            x.double().to(x0.device) for x in (x0, xt, self.sigmas, self.timesteps))
        sigma = sigmas[self._index(timestep, timesteps)].reshape(-1, 1, 1, 1)
        return ((xt - x0) / sigma).to(dtype)
