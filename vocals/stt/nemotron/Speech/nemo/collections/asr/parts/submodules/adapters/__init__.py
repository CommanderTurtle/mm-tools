# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# fmt: off
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module import (
    MHAResidualAddAdapterStrategy,
    MHAResidualAddAdapterStrategyConfig,
    MultiHeadAttentionAdapter,
    MultiHeadAttentionAdapterConfig,
    PositionalEncodingAdapter,
    PositionalEncodingAdapterConfig,
    RelPositionalEncodingAdapter,
    RelPositionalEncodingAdapterConfig,
    RelPositionMultiHeadAttentionAdapter,
    RelPositionMultiHeadAttentionAdapterConfig,
)
from nemo.collections.asr.parts.submodules.adapters.transformer_multi_head_attention_adapter_module import (
    TransformerMultiHeadAttentionAdapter,
    TransformerMultiHeadAttentionAdapterConfig,
)
from nemo.utils.serialization import register_checkpoint_schema_types

# fmt: on


register_checkpoint_schema_types(
    MultiHeadAttentionAdapterConfig,
    PositionalEncodingAdapterConfig,
    RelPositionalEncodingAdapterConfig,
    RelPositionMultiHeadAttentionAdapterConfig,
    TransformerMultiHeadAttentionAdapterConfig,
)
