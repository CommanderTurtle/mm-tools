# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from torch import Tensor, nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LossType, NeuralType, RegressionValuesType

__all__ = ['MSELoss']


class MSELoss(nn.MSELoss, Serialization, Typing):
    """
    MSELoss
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "preds": NeuralType(tuple('B'), RegressionValuesType()),
            "labels": NeuralType(tuple('B'), LabelsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: type of the reduction over the batch
        """
        super().__init__(reduction=reduction)

    @typecheck()
    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            preds: output of the classifier
            labels: ground truth labels
        """
        return super().forward(preds, labels)
