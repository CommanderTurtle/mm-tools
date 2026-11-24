#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Prompt streams shared by every stage."""
from __future__ import annotations

from pathlib import Path

import torch


def read_prompts(path):
    """One prompt per line; blank lines are skipped."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class PromptStream:
    """Infinite, rank-disjoint prompt iterator reshuffled every epoch.

    Draw ``i`` on rank ``r`` is ``perm_epoch[i * world + r]``; the stream is
    resumable from ``position`` (number of draws already taken on this rank).
    """

    def __init__(self, prompts, rank=0, world_size=1, seed=0, batch_size=1, position=0):
        self.prompts = list(prompts)
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.batch_size = batch_size
        self.position = position
        self.per_epoch = len(self.prompts) // world_size
        if self.per_epoch == 0:
            raise ValueError("fewer prompts than ranks")
        self._epoch, self._perm = None, None

    def _permutation(self, epoch):
        if epoch != self._epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._perm = torch.randperm(len(self.prompts), generator=generator)
            self._epoch = epoch
        return self._perm

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            epoch, index = divmod(self.position, self.per_epoch)
            batch.append(self.prompts[self._permutation(epoch)[index * self.world_size + self.rank]])
            self.position += 1
        return batch

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = int(state["position"])
