# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from torch.nn.utils import remove_weight_norm, weight_norm


def apply_weight_norm_lstm(lstm_module):
    bidirectional = lstm_module.bidirectional
    lstm_wn = weight_norm(lstm_module, name='weight_ih_l0')
    lstm_wn = weight_norm(lstm_wn, name='weight_hh_l0')
    if bidirectional:
        lstm_wn = weight_norm(lstm_wn, name='weight_ih_l0_reverse')
        lstm_wn = weight_norm(lstm_wn, name='weight_hh_l0_reverse')
    return lstm_wn


def remove_weight_norm_lstm(lstm_module):
    bidirectional = lstm_module.bidirectional
    lstm = remove_weight_norm(lstm_module, name='weight_ih_l0')
    lstm = remove_weight_norm(lstm, name='weight_hh_l0')
    if bidirectional:
        lstm = remove_weight_norm(lstm, name='weight_ih_l0_reverse')
        lstm = remove_weight_norm(lstm, name='weight_hh_l0_reverse')
    return lstm
