# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.asr.parts.numba.rnnt_loss.rnnt import rnnt_loss_cpu, rnnt_loss_gpu
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import (
    MultiblankRNNTLossNumba,
    RNNTLossNumba,
    TDTLossNumba,
)
