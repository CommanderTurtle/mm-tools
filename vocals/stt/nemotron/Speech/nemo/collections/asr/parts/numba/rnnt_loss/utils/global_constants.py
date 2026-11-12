# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# Copyright 2018-2019, Mingkun Huang
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


import enum

import numpy as np
from numba import float32

# Internal globals
_THREADS_PER_BLOCK = 32
_WARP_SIZE = 32
_DTYPE = float32

# Constants
FP32_INF = np.inf
FP32_NEG_INF = -np.inf
THRESHOLD = 1e-1

"""
Getters
"""


def threads_per_block():
    global _THREADS_PER_BLOCK
    return _THREADS_PER_BLOCK


def warp_size():
    global _WARP_SIZE
    return _WARP_SIZE


def dtype():
    global _DTYPE
    return _DTYPE


# RNNT STATUS
class RNNTStatus(enum.Enum):
    RNNT_STATUS_SUCCESS = 0
    RNNT_STATUS_INVALID_VALUE = 1
