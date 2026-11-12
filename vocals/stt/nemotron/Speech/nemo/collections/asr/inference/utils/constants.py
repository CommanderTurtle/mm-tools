# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# Precision related constants
BIG_EPSILON = 1e-5
SMALL_EPSILON = 1e-10
ROUND_PRECISION = 9

# ASR Preprocessing related constants
LOG_MEL_ZERO = -16.635532333438686  # log(2^-24), matches preprocessor output for silence

# Punctuation related constants
POST_WORD_PUNCTUATION = set(".,?")
PRE_WORD_PUNCTUATION = set("¿")
SEP_REPLACEABLE_PUNCTUATION = set("-_")
SENTENCEPIECE_UNDERSCORE = "▁"

# ITN related constants
DEFAULT_SEMIOTIC_CLASS = "name"

# Default output directory name
DEFAULT_OUTPUT_DIR_NAME = "jsons"
