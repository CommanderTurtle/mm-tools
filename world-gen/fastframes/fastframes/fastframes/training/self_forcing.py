#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Self-forcing rollouts of the frame-causal generator (DMD training).

A rollout generates a clip frame by frame with the few-step schedule. Each frame
is denoised from noise through its schedule (a dedicated one for frame 0) up to
a sampled *exit* step whose x0 prediction becomes the frame; the steps before the
exit run without gradient. Every frame attends to a *clean context*: before each
frame, the per-layer keys/values of all finished frames are recomputed by a
frame-causal forward at t = 0, without gradient. The generator loss is then
back-propagated frame by frame (``replay``).
"""
from __future__ import annotations

import torch

__all__ = ["SelfForcingRollout", "balanced_exit", "GENERATOR_EXITS", "CRITIC_EXITS",
           "FIRST_FRAME_GENERATOR_EXITS", "FIRST_FRAME_CRITIC_EXITS"]

# Independent streams of the balanced exit schedule.
GENERATOR_EXITS, CRITIC_EXITS = 0x47E1, 0xC71C
FIRST_FRAME_GENERATOR_EXITS, FIRST_FRAME_CRITIC_EXITS = 0xF147E1, 0xF1C71C


def balanced_exit(index, num_exits, block_size, seed, stream):
    """Exit step of update ``index`` (0-based), a pure function of its arguments.

    Updates form blocks of ``block_size``: each block visits every exit once in a
    shuffled order and fills its remaining slots uniformly at random.
    """
    block, offset = divmod(index, block_size)
    generator = torch.Generator().manual_seed(seed + stream + block * 1_000_003)
    order = torch.randperm(num_exits, generator=generator)
    if offset < num_exits:
        return int(order[offset])
    return int(torch.randint(0, num_exits, (block_size - num_exits,), generator=generator)[offset - num_exits])


class SelfForcingRollout:
    """Frame-wise few-step generation with the (FSDP-wrapped) causal generator.

    ``timesteps`` / ``first_frame_timesteps`` are the model timesteps of the
    steady frames and of frame 0. The generator sees BF16-rounded timesteps; its
    flow becomes x0 with the FP32 timestep.
    """

    def __init__(self, generator, scheduler, timesteps, first_frame_timesteps):
        self.generator = generator
        self.scheduler = scheduler
        self.timesteps = tuple(float(t) for t in timesteps)
        self.first_frame_timesteps = tuple(float(t) for t in first_frame_timesteps)

    @property
    def num_transitions(self):
        """Transition noises per frame of a pre-generated noise bank."""
        return max(len(self.timesteps), len(self.first_frame_timesteps)) - 1

    def add_noise(self, x0, noise, timestep):
        """Re-noise frames ``[B, F, C, H, W]`` to the (scalar) ``timestep``."""
        t = torch.full((x0.shape[0] * x0.shape[1],), timestep, dtype=torch.float32, device=x0.device)
        return self.scheduler.add_noise(x0.flatten(0, 1), noise.flatten(0, 1), t).unflatten(0, x0.shape[:2])

    def predict(self, x, timestep, prompt_embeds, context_kv, frame):
        """One generator call on frame ``frame``: its x0 and per-layer (k, v)."""
        t = torch.full(x.shape[:2], timestep, dtype=torch.float32, device=x.device)
        flow, kv = self.generator(x, t.to(torch.bfloat16), prompt_embeds, context_kv=context_kv, frame=frame)
        x0 = self.scheduler.flow_to_x0(flow.flatten(0, 1), x.flatten(0, 1), t.flatten())
        return x0.unflatten(0, x.shape[:2]), kv

    def denoise_frame(self, noise, frame, exit_index, prompt_embeds, context_kv, transitions, grad=False):
        """Denoise ``noise`` (``[B, 1, C, H, W]``) of ``frame`` up to step ``exit_index``.

        Only the exit prediction carries gradient (if ``grad``). Transition noises
        come from ``transitions`` (``[S, B, 1, C, H, W]``). Returns the exit's x0 and
        per-layer (k, v).
        """
        steps = self.first_frame_timesteps if frame == 0 else self.timesteps
        x = noise
        for step in range(exit_index + 1):
            with torch.set_grad_enabled(grad and step == exit_index):
                x0, kv = self.predict(x, steps[step], prompt_embeds, context_kv, frame)
            if step < exit_index:
                x = self.add_noise(x0, transitions[step], steps[step + 1])
        return x0, kv

    @torch.no_grad()
    def clean_context(self, frames, prompt_embeds):
        """Per-layer (keys, values) of the clean ``frames`` (``[B, F, C, H, W]``) at t = 0."""
        t = torch.zeros(frames.shape[:2], dtype=torch.bfloat16, device=frames.device)
        _, kv = self.generator(frames, t, prompt_embeds, return_kv=True)
        return [([k], [v]) for k, v in kv]

    def __call__(self, noise, prompt_embeds, exits, transitions):
        """Generate the clip for ``noise`` (``[B, F, C, H, W]``) without gradient and return its x0.

        ``exits = (frame-0 exit, exit of the other frames)``; ``transitions``:
        pre-generated noise bank ``[S, B, F, C, H, W]``.
        """
        frames = []
        for frame in range(noise.shape[1]):
            context = self.clean_context(torch.cat(frames, dim=1), prompt_embeds) if frame else None
            x0, _ = self.denoise_frame(noise[:, frame:frame + 1], frame, exits[0] if frame == 0 else exits[1],
                                       prompt_embeds, context, transitions[:, :, frame:frame + 1])
            frames.append(x0)
        return torch.cat(frames, dim=1)

    def replay(self, noise, reference, prompt_embeds, exits, transitions):
        """Redo each frame of a gradient-free rollout with gradient.

        Frame ``f`` is regenerated from the same noise against the clean context
        of ``reference[:, :f]``; yields ``(f, x0)`` so that the caller can
        backpropagate frame by frame (only one frame's graph is alive at a time).
        """
        for frame in range(noise.shape[1]):
            context = self.clean_context(reference[:, :frame], prompt_embeds) if frame else None
            x0, _ = self.denoise_frame(noise[:, frame:frame + 1], frame, exits[0] if frame == 0 else exits[1],
                                       prompt_embeds, context, transitions[:, :, frame:frame + 1], grad=True)
            yield frame, x0
