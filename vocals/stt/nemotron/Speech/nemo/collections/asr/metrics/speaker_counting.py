# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from typing import Tuple

import torch


def speaker_count_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute speaker-count errors over valid frames.

    Args:
        preds: Speaker probabilities with shape ``(B, T, S)``.
        targets: Hard speaker targets with shape ``(B, T, S)``.
        target_lens: Number of valid frames per sample with shape ``(B,)``.

    Returns:
        Scalar FP32 tensors containing the batch-mean speaker-count absolute
        error and exact-match accuracy. Activity outside valid frames is ignored.
    """
    num_frames = preds.shape[1]
    valid = (
        torch.arange(num_frames, device=preds.device).unsqueeze(0) < target_lens.to(preds.device).unsqueeze(1)
    ).unsqueeze(-1)

    predicted_present = ((preds > 0.5) & valid).any(dim=1)
    target_present = ((targets > 0.5) & valid).any(dim=1)

    predicted_count = predicted_present.sum(dim=1)
    target_count = target_present.sum(dim=1)
    count_error = (predicted_count - target_count).abs().float()

    return count_error.mean(), (predicted_count == target_count).float().mean()
