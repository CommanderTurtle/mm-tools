# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# flake8: noqa

from nemo.core.optim.adafactor import Adafactor
from nemo.core.optim.adan import Adan
from nemo.core.optim.flash_optim import patch_flashoptim_uneven_shard_support
from nemo.core.optim.lr_scheduler import (
    CosineAnnealing,
    InverseSquareRootAnnealing,
    NoamAnnealing,
    PiecewiseLinear,
    PolynomialDecayAnnealing,
    PolynomialHoldDecayAnnealing,
    SquareAnnealing,
    SquareRootAnnealing,
    T5InverseSquareRootAnnealing,
    WarmupAnnealing,
    WarmupHoldAnnealLinear,
    WarmupHoldAnnealOneMinusSquareRoot,
    WarmupHoldPolicy,
    WarmupPolicy,
    prepare_lr_scheduler,
)
from nemo.core.optim.muon import Muon
from nemo.core.optim.novograd import Novograd
from nemo.core.optim.optimizer_with_main_params import MainParamsOptimizerWrapper
from nemo.core.optim.optimizers import get_optimizer, parse_optimizer_args, register_optimizer
