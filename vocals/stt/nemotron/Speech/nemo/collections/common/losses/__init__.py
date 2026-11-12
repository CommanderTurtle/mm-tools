# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.common.losses.aggregator import AggregatorLoss
from nemo.collections.common.losses.bce_logits_loss import BCEWithLogitsLoss
from nemo.collections.common.losses.cross_entropy import CrossEntropyLoss, NLLLoss
from nemo.collections.common.losses.label_smoothing import label_smoothed_log_prob_gradient
from nemo.collections.common.losses.mse_loss import MSELoss
from nemo.collections.common.losses.multi_similarity_loss import MultiSimilarityLoss
from nemo.collections.common.losses.smoothed_cross_entropy import SmoothedCrossEntropyLoss, SmoothedNLLLoss
from nemo.collections.common.losses.spanning_loss import SpanningLoss
