# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Shared output types for EasyMagpieTTS CFG distillation."""

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor


@dataclass
class TeacherOutput:
    """Outputs produced by the teacher rollout used as distillation targets."""

    codes: Tensor
    logits: Tensor
    lens: Tensor
    sample_weights: Optional[Tensor]
    rejected: Optional[Tensor]

    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]

    # CPU-observed wall times in seconds, without explicit CUDA synchronization.
    timing_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class StudentOutput:
    """Outputs produced by the student forward pass for distillation."""

    logits: Tensor
    logits_lt: Optional[Tensor]
    logits_phonemes: Optional[Tensor]
    audio_codes_gt_lt: Optional[Tensor]
    audio_codes_lens_gt_lt: Optional[Tensor]
