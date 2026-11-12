# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.asr.losses.bce_loss import BCELoss, BCEWithLogitsLoss
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.ssl_losses.contrastive import ContrastiveLoss
from nemo.collections.asr.losses.ssl_losses.ctc import CTCLossForSSL
from nemo.collections.asr.losses.ssl_losses.mlm import MLMLoss, MultiMLMLoss
from nemo.collections.asr.losses.ssl_losses.rnnt import RNNTLossForSSL

__all__ = [
    'AngularSoftmaxLoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'CTCLoss',
    'CTCLossForSSL',
    'ContrastiveLoss',
    'MLMLoss',
    'MultiMLMLoss',
    'RNNTLossForSSL',
]
