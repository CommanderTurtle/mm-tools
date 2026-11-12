# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

###############################################################################

import torch


class GaussianDropout(torch.nn.Module):
    """
    Gaussian dropout using multiplicative gaussian noise.

    https://keras.io/api/layers/regularization_layers/gaussian_dropout/

    Can be an effective alternative bottleneck to VAE or VQ:

    https://www.deepmind.com/publications/gaussian-dropout-as-an-information-bottleneck-layer

    Unlike some other implementations, this takes the standard deviation of the noise as input
    instead of the 'rate' typically defined as: stdev = sqrt(rate / (1 - rate))
    """

    def __init__(self, stdev=1.0):
        super(GaussianDropout, self).__init__()
        self.stdev = stdev

    def forward(self, inputs):
        if not self.training:
            return inputs

        noise = torch.normal(mean=1.0, std=self.stdev, size=inputs.shape, device=inputs.device)
        out = noise * inputs
        return out
