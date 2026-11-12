# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.asr.modules.transformer.bridge_encoders import BridgeEncoder
from nemo.collections.asr.modules.transformer.perceiver_encoders import PerceiverEncoder
from nemo.collections.asr.modules.transformer.transformer_bottleneck import (
    NeMoTransformerBottleneckConfig,
    NeMoTransformerBottleneckDecoderConfig,
    NeMoTransformerBottleneckEncoderConfig,
    TransformerBottleneckEncoderNM,
)
from nemo.collections.asr.modules.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithFusionModels,
    BeamSearchSequenceGeneratorWithLanguageModel,
    EnsembleBeamSearchSequenceGenerator,
    GreedySequenceGenerator,
    TopKSequenceGenerator,
)
from nemo.collections.asr.modules.transformer.transformer_modules import AttentionBridge, TransformerEmbedding
from nemo.collections.asr.modules.transformer.transformer_utils import get_nemo_transformer

__all__ = [
    "BridgeEncoder",
    "PerceiverEncoder",
    "NeMoTransformerBottleneckConfig",
    "NeMoTransformerBottleneckDecoderConfig",
    "NeMoTransformerBottleneckEncoderConfig",
    "TransformerBottleneckEncoderNM",
    "TransformerDecoder",
    "TransformerEncoder",
    "BeamSearchSequenceGenerator",
    "BeamSearchSequenceGeneratorWithLanguageModel",
    "BeamSearchSequenceGeneratorWithFusionModels",
    "EnsembleBeamSearchSequenceGenerator",
    "GreedySequenceGenerator",
    "TopKSequenceGenerator",
    "AttentionBridge",
    "TransformerEmbedding",
    "get_nemo_transformer",
]
