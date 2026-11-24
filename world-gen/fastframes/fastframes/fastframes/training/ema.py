#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Exponential moving average of (possibly FSDP-sharded) trainable parameters."""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _name(name: str) -> str:
    for token in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.", "module."):
        name = name.replace(token, "")
    return name


class EMA:
    """FP32 shadow of every trainable parameter (local shards under FSDP ``use_orig_params``).

    ``update`` is a no-op until ``start_step`` updates have been taken; the first
    effective update copies the parameters (``copy_on_start``) or blends as usual.
    """

    def __init__(self, model, decay: float, start_step: int = 0, copy_on_start: bool = True):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.copy_on_start = copy_on_start
        self.num_updates = 0
        self.shadow = {name: p.detach().float().clone() for name, p in self._parameters(model).items()}

    @staticmethod
    def _parameters(model):
        return {_name(n): p for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model, step: int) -> None:
        """Call after the optimizer step that completed ``step`` updates."""
        if step < self.start_step:
            return
        params = self._parameters(model)
        if step == self.start_step and self.copy_on_start:
            for name, p in params.items():
                self.shadow[name].copy_(p.detach())
        else:
            for name, p in params.items():
                self.shadow[name].lerp_(p.detach().float(), 1.0 - self.decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model) -> None:
        for name, p in self._parameters(model).items():
            p.copy_(self.shadow[name])

    @contextmanager
    def applied(self, model):
        """Temporarily swap the EMA weights into ``model``."""
        params = self._parameters(model)
        backup = {name: p.detach().clone() for name, p in params.items()}
        try:
            self.copy_to(model)
            yield model
        finally:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(backup[name])

    def state_dict(self):
        return {"decay": self.decay, "start_step": self.start_step, "num_updates": self.num_updates,
                "shadow": self.shadow}

    def load_state_dict(self, state):
        self.num_updates = int(state["num_updates"])
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value)
