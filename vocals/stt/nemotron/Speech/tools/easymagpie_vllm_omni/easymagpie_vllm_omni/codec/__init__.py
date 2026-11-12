# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Native Torch implementations of the EasyMagpie spectral codec."""

from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoder, EasyMagpieCodecEncoderOutput
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig
from easymagpie_vllm_omni.codec.reference_speaker_encoder import EasyMagpieReferenceSpeakerEncoder

__all__ = [
    "EasyMagpieCodecConfig",
    "EasyMagpieCodecEncoder",
    "EasyMagpieCodecEncoderConfig",
    "EasyMagpieCodecEncoderOutput",
    "EasyMagpieReferenceSpeakerEncoder",
]
