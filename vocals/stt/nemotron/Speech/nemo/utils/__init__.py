# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


from nemo.utils.app_state import AppState
from nemo.utils.cast_utils import (
    CastToFloat,
    CastToFloatAll,
    avoid_bfloat16_autocast_context,
    avoid_float16_autocast_context,
    cast_all,
    cast_tensor,
    monkeypatched,
)
from nemo.utils.dtype import str_to_dtype
from nemo.utils.nemo_logging import Logger as _Logger
from nemo.utils.nemo_logging import LogMode as logging_mode

try:
    from nemo.utils.serialization import register_checkpoint_config_types
except ModuleNotFoundError as error:
    if error.name != 'omegaconf':
        raise
else:
    register_checkpoint_config_types()

logging = _Logger()
try:
    from nemo.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger, preserve_trainer_callback_defaults

    add_memory_handlers_to_pl_logger()
    preserve_trainer_callback_defaults()
except ModuleNotFoundError:
    pass

try:
    import webdataset
    from nemo.utils.data_utils import wds_url_opener

    webdataset.tariterators.url_opener = wds_url_opener
except ModuleNotFoundError:
    pass
