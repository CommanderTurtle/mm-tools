# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.asr.modules.ssl_modules.augmentation import (
    MultiSpeakerNoiseAugmentation,
    SpeakerNoiseAugmentation,
)
from nemo.collections.asr.modules.ssl_modules.masking import ConvFeatureMaksingWrapper, RandomBlockMasking
from nemo.collections.asr.modules.ssl_modules.multi_layer_feat import ConformerMultiLayerFeaturePreprocessor
from nemo.collections.asr.modules.ssl_modules.multi_softmax_decoder import MultiSoftmaxDecoder
from nemo.collections.asr.modules.ssl_modules.quantizers import RandomProjectionVectorQuantizer

__all__ = [
    'MultiSpeakerNoiseAugmentation',
    'SpeakerNoiseAugmentation',
    'ConvFeatureMaksingWrapper',
    'RandomBlockMasking',
    'ConformerMultiLayerFeaturePreprocessor',
    'MultiSoftmaxDecoder',
    'RandomProjectionVectorQuantizer',
]
