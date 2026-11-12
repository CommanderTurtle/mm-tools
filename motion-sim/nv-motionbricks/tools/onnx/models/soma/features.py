# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"Combine SOMA constraint masks without changing their numeric dtype."

from __future__ import annotations

import torch


def numeric_mask_union(*masks: torch.Tensor) -> torch.Tensor:
    """Combine masks by their element-wise maximum, keeping their numeric dtype."""
    if not masks:
        raise ValueError("numeric_mask_union requires at least one mask.")
    result = masks[0]
    for mask in masks[1:]:
        result = torch.maximum(result, mask)
    return result
