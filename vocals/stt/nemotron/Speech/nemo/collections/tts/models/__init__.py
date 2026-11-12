# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.tts.models.aligner import AlignerModel
from nemo.collections.tts.models.audio_codec import AudioCodecModel
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_cfg_distillation import EasyMagpieCFGDistillation
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO
from nemo.collections.tts.models.fastpitch import FastPitchModel
from nemo.collections.tts.models.fastpitch_ssl import FastPitchModel_SSL
from nemo.collections.tts.models.hifigan import HifiGanModel
from nemo.collections.tts.models.magpietts import InferBatchOutput, MagpieTTSModel
from nemo.collections.tts.models.magpietts_cfg_distillation import OnlineCFGDistillation
from nemo.collections.tts.models.magpietts_preference_optimization import (
    MagpieTTSModelOfflinePO,
    MagpieTTSModelOfflinePODataGen,
    MagpieTTSModelOnlinePO,
)
from nemo.collections.tts.models.ssl_tts import SSLDisentangler

__all__ = [
    "AlignerModel",
    "AudioCodecModel",
    "FastPitchModel",
    "FastPitchModel_SSL",
    "SSLDisentangler",
    "HifiGanModel",
    "InferBatchOutput",
    "MagpieTTSModel",
    "OnlineCFGDistillation",
    "EasyMagpieTTSModel",
    "EasyMagpieTTSInferenceModel",
    "EasyMagpieTTSModelOnlinePO",
    "MagpieTTSModelOfflinePODataGen",
    "MagpieTTSModelOfflinePO",
    "MagpieTTSModelOnlinePO",
    "EasyMagpieCFGDistillation",
]
