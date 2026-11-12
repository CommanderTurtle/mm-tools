# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"SOMA convolutional decoder with numeric constraint masks."

from __future__ import annotations

import torch

from ..layers import DoubleCondDecoder as BaseDecoder


class DoubleCondDecoder(BaseDecoder):
    """Decoder that casts numeric token masks to the feature dtype."""

    def forward(
        self,
        x: torch.Tensor,
        external_cond: torch.Tensor = None,
        target_cond: torch.Tensor = None,
        has_target_cond: torch.Tensor = None,
        token_mask: torch.Tensor = None,
    ):
        if token_mask is not None:
            token_mask = token_mask.to(dtype=x.dtype)
        return super().forward(
            x, external_cond, target_cond, has_target_cond, token_mask
        )
