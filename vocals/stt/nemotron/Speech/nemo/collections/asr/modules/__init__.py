# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.asr.modules.audio_preprocessing import (  # noqa: F401
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor,
    CropOrPadSpectrogramAugmentation,
    MaskedPatchAugmentation,
    SpectrogramAugmentation,
)
from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM  # noqa: F401
from nemo.collections.asr.modules.conformer_encoder import (  # noqa: F401
    ConformerEncoder,
    ConformerEncoderAdapter,
    ConformerMultiLayerFeatureExtractor,
)
from nemo.collections.asr.modules.conv_asr import (  # noqa: F401
    ConvASRDecoder,
    ConvASRDecoderClassification,
    ConvASRDecoderReconstruction,
    ConvASREncoder,
    ConvASREncoderAdapter,
    ECAPAEncoder,
    ParallelConvASREncoder,
    SpeakerDecoder,
)
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint  # noqa: F401
from nemo.collections.asr.modules.lstm_decoder import LSTMDecoder  # noqa: F401
from nemo.collections.asr.modules.parallel_expert_encoder import (  # noqa: F401
    CTCTimestampInputs,
    ParallelExpertEncoder,
)
from nemo.collections.asr.modules.rnn_encoder import RNNEncoder  # noqa: F401
from nemo.collections.asr.modules.rnnt import (  # noqa: F401
    RNNTDecoder,
    RNNTDecoderJointSSL,
    RNNTJoint,
    SampledRNNTJoint,
    StatelessTransducerDecoder,
)
from nemo.collections.asr.modules.ssl_modules import (  # noqa: F401
    ConformerMultiLayerFeaturePreprocessor,
    ConvFeatureMaksingWrapper,
    MultiSoftmaxDecoder,
    RandomBlockMasking,
    RandomProjectionVectorQuantizer,
)
from nemo.collections.asr.modules.transformer_encoder import (  # noqa: F401
    StreamingTransformerEncoder,
    TransformerEncoder,
)

__all__ = [
    "AudioToMelSpectrogramPreprocessor",
    "AudioToMFCCPreprocessor",
    "CropOrPadSpectrogramAugmentation",
    "MaskedPatchAugmentation",
    "SpectrogramAugmentation",
    "BeamSearchDecoderWithLM",
    "ConformerEncoder",
    "ConformerEncoderAdapter",
    "ConformerMultiLayerFeatureExtractor",
    "CTCTimestampInputs",
    "ParallelExpertEncoder",
    "ConvASRDecoder",
    "ConvASRDecoderClassification",
    "ConvASRDecoderReconstruction",
    "ConvASREncoder",
    "ConvASREncoderAdapter",
    "ECAPAEncoder",
    "ParallelConvASREncoder",
    "SpeakerDecoder",
    "HATJoint",
    "LSTMDecoder",
    "RNNTDecoder",
    "RNNTDecoderJointSSL",
    "RNNTJoint",
    "SampledRNNTJoint",
    "StatelessTransducerDecoder",
    "ConformerMultiLayerFeaturePreprocessor",
    "ConvFeatureMaksingWrapper",
    "MultiSoftmaxDecoder",
    "RandomBlockMasking",
    "RandomProjectionVectorQuantizer",
    "StreamingTransformerEncoder",
    "TransformerEncoder",
]
