# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# program from someone who is authorized to grant you that right.
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import numpy as np
import torch


def v2v(x, y):
    if torch.is_tensor(x):
        return (x - y).pow(2).sum(dim=-1).sqrt().mean()
    else:
        return np.sqrt(np.power(x - y, 2)).sum(axis=-1).mean()
