# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import re
import unicodedata
from typing import List, Set


def extract_punctuation_from_vocab(vocab: List[str]) -> Set[str]:
    """
    Extract punctuation marks from vocabulary.

    Args:
        vocab: List of vocabulary tokens

    Returns:
        Set of punctuation marks found in the vocabulary
    """
    special_token_patterns = [
        re.compile(r'^\[.*\]$'),
        re.compile(r'^<.*>$'),
        re.compile(r'^##'),
        re.compile(r'^▁'),
        re.compile(r'^\s*$'),
    ]

    def is_special_token(token):
        return any(pattern.match(token) for pattern in special_token_patterns)

    punctuation = {
        char
        for token in vocab
        for char in token
        if unicodedata.category(char).startswith('P') and not is_special_token(token)
    }

    return punctuation


def extract_capitalized_tokens_from_vocab(vocab: List[str]) -> Set[str]:
    """
    Extract capitalized tokens from vocabulary.

    Args:
        vocab: List of vocabulary tokens

    Returns:
        Set of capitalized tokens found in the vocabulary
    """
    capitalized_tokens = {token.strip() for token in vocab if any(char.isupper() for char in token)}
    return capitalized_tokens


def define_spe_tokenizer_type(vocabulary: List[str]) -> str:
    """
    Define the tokenizer type based on the vocabulary.
    """
    if any(token.startswith("##") for token in vocabulary):
        return "wpe"
    return "bpe"
