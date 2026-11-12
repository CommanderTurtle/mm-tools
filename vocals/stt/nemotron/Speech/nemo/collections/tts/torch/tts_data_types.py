# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


class TTSDataType:
    """Represent TTSDataType."""

    name = None


class WithLens:
    """Represent that this data type also returns lengths for data."""


class Audio(TTSDataType, WithLens):
    name = "audio"


class Text(TTSDataType, WithLens):
    name = "text"


class LogMel(TTSDataType, WithLens):
    name = "log_mel"


class Durations(TTSDataType):
    name = "durations"


class AlignPriorMatrix(TTSDataType):
    name = "align_prior_matrix"


class Pitch(TTSDataType, WithLens):
    name = "pitch"


class Energy(TTSDataType, WithLens):
    name = "energy"


class SpeakerID(TTSDataType):
    name = "speaker_id"


class Voiced_mask(TTSDataType):
    name = "voiced_mask"


class P_voiced(TTSDataType):
    name = "p_voiced"


class LMTokens(TTSDataType):
    name = "lm_tokens"


class ReferenceAudio(TTSDataType, WithLens):
    name = "reference_audio"


MAIN_DATA_TYPES = [Audio, Text]
VALID_SUPPLEMENTARY_DATA_TYPES = [
    LogMel,
    Durations,
    AlignPriorMatrix,
    Pitch,
    Energy,
    SpeakerID,
    LMTokens,
    Voiced_mask,
    P_voiced,
    ReferenceAudio,
]
DATA_STR2DATA_CLASS = {d.name: d for d in MAIN_DATA_TYPES + VALID_SUPPLEMENTARY_DATA_TYPES}
