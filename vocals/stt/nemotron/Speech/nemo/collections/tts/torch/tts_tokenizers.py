# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    BaseCharsTokenizer,
    BaseTokenizer,
    EnglishCharsTokenizer,
    EnglishPhonemesTokenizer,
    GermanCharsTokenizer,
    GermanPhonemesTokenizer,
    IPATokenizer,
)

__all__ = [
    'BaseCharsTokenizer',
    'BaseTokenizer',
    'EnglishCharsTokenizer',
    'EnglishPhonemesTokenizer',
    'GermanCharsTokenizer',
    'GermanPhonemesTokenizer',
    'IPATokenizer',
]

# TODO @xueyang: deprecate this file since no other places import modules from here anymore. However,
#  all checkpoints uploaded in ngc used this path. So it requires to update all ngc checkpoints path as well.
