# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging as _logging

from nemo.utils import logging as nemo_logger


def add_handlers_to_mcore_logger():
    """Add NeMo handlers to MCore loggers.

    MCore doesn't have and handlers for loggers (see
    https://docs.python.org/3/howto/logging-cookbook.html#adding-handlers-other-than-nullhandler-to-a-logger-in-a-library
    for a rationale). We have to add handlers explicitly.
    """
    mcore_logger = _logging.getLogger('megatron.core')
    for handler in nemo_logger._handlers.values():
        mcore_logger.addHandler(handler)
    mcore_logger.propagate = False
    mcore_logger.setLevel(nemo_logger._logger.level)
