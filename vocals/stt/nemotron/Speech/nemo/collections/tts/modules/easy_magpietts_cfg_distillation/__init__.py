# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Public components for EasyMagpieTTS classifier-free guidance distillation."""

from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.distillation_types import (
    StudentOutput,
    TeacherOutput,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.logging_keys import (
    MONITORED_KEYS_TRAIN,
    MONITORED_KEYS_VALID,
    ON_EPOCH_TRAIN_KEYS,
    ROLLOUT_TIMING_KEYS,
    LossKey,
    LossMode,
    MetricKey,
    get_loss_key,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.teacher_inference_engine import (
    TeacherInferenceEngine,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.validation_inference_engine import (
    ValidationAudioPaths,
    ValidationInferenceEngine,
    ValidationInferenceParams,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.validation_metrics import (
    aggregate_language_metrics,
    collect_validation_outputs,
    compute_validation_metrics,
)

__all__ = [
    "StudentOutput",
    "TeacherOutput",
    "LossKey",
    "LossMode",
    "MetricKey",
    "get_loss_key",
    "MONITORED_KEYS_TRAIN",
    "MONITORED_KEYS_VALID",
    "ON_EPOCH_TRAIN_KEYS",
    "ROLLOUT_TIMING_KEYS",
    "TeacherInferenceEngine",
    "ValidationAudioPaths",
    "ValidationInferenceEngine",
    "ValidationInferenceParams",
    "aggregate_language_metrics",
    "collect_validation_outputs",
    "compute_validation_metrics",
]
