# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from abc import ABC
from typing import Any, Dict, Optional

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import ChannelType, EncodedRepresentation, MaskType, NeuralType

__all__ = ['DecoderModule']


class DecoderModule(NeuralModule, ABC):
    """Base class for decoder neural module to be used in NLP models."""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "decoder_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "encoder_embeddings": NeuralType(('B', 'T', 'D'), ChannelType(), optional=True),
            "encoder_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "decoder_mems": NeuralType(('B', 'D', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def hidden_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def embedding(self) -> Optional[Any]:
        raise NotImplementedError

    @property
    def decoder(self) -> Optional[Any]:
        raise NotImplementedError

    @property
    def max_sequence_length(self) -> Optional[int]:
        raise NotImplementedError
