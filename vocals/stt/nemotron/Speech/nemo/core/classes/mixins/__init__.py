# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.core.classes.mixins.access_mixins import AccessMixin, set_access_cfg
from nemo.core.classes.mixins.adapter_mixin_strategies import (
    ResidualAddAdapterStrategy,
    ResidualAddAdapterStrategyConfig,
    ReturnResultAdapterStrategy,
    ReturnResultAdapterStrategyConfig,
)
from nemo.core.classes.mixins.adapter_mixins import (
    AdapterModelPTMixin,
    AdapterModuleMixin,
    get_registered_adapter,
    register_adapter,
)
from nemo.core.classes.mixins.hf_io_mixin import HuggingFaceFileIO
