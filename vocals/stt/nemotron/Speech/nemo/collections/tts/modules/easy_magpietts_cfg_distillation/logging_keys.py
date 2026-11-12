# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Loss and metric logging keys for EasyMagpieTTS CFG distillation."""

from enum import Enum


class LossKey(str, Enum):
    """Enumeration of names used for logged loss values."""

    loss = "loss"
    kl_loss = "kl_loss"
    ce_loss = "ce_loss"
    nrmse_loss = "nrmse_loss"


class LossMode(str, Enum):
    """Enumeration of model components associated with a loss."""

    main = ""
    local_transformer = "lt"
    phoneme = "phoneme"


def get_loss_key(
    key: LossKey,
    mode: LossMode = LossMode.main,
) -> str:
    """Build a logged loss key for a model component.

    Args:
        key: Base loss key.
        mode: Model component associated with the loss.

    Returns:
        Loss key.
    """
    if mode == LossMode.main:
        return key.value
    return f"{key.value}_{mode.value}"


class MetricKey(str, Enum):
    """Enumeration of names used for metrics."""

    cer = "cer"
    wer = "wer"
    ssim = "ssim"
    utmos = "utmos"
    rejection_rate = "rejection_rate"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class LanguageMetricKey(str, Enum):
    """Enumeration of keys used to collect per-language error metrics."""

    languages = "languages"
    cer_list = "cer_list"
    wer_list = "wer_list"


ROLLOUT_TIMING_KEYS: list[str] = [
    "rollout_generation_time",
    "rollout_total_time",
]


MONITORED_KEYS_TRAIN: list[str] = [
    get_loss_key(LossKey.kl_loss),
    get_loss_key(LossKey.ce_loss),
    get_loss_key(LossKey.nrmse_loss),
    get_loss_key(key=LossKey.kl_loss, mode=LossMode.local_transformer),
    get_loss_key(key=LossKey.ce_loss, mode=LossMode.local_transformer),
    get_loss_key(key=LossKey.nrmse_loss, mode=LossMode.local_transformer),
    get_loss_key(key=LossKey.loss, mode=LossMode.phoneme),
    get_loss_key(key=LossKey.ce_loss, mode=LossMode.phoneme),
    MetricKey.rejection_rate.value,
]


MONITORED_KEYS_VALID: list[str] = [
    *MONITORED_KEYS_TRAIN,
    MetricKey.cer.value,
    MetricKey.wer.value,
    MetricKey.ssim.value,
    MetricKey.utmos.value,
]


ON_EPOCH_TRAIN_KEYS: set[str] = set(MONITORED_KEYS_TRAIN)
