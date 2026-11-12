# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


__all__ = ['experimental']


import wrapt

from nemo.utils import logging


@wrapt.decorator
def experimental(wrapped, instance, args, kwargs):
    logging.warning(f"`{wrapped}` is experimental and not ready for production yet. Use at your own risk.")
    return wrapped(*args, **kwargs)
