# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import importlib
import pathlib
import sys
from contextlib import contextmanager


@contextmanager
def python313_pathlib_pickle_compat():
    """Allow Python 3.13 ``pathlib`` pickles to load on older Python versions.

    Python 3.13 moved concrete path classes to ``pathlib._local``. Pickles
    created there therefore cannot resolve their module when loaded on older
    Python versions. Temporarily alias the missing module while unpickling.
    """
    module_name = "pathlib._local"
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    else:
        yield
        return

    sys.modules[module_name] = pathlib
    try:
        yield
    finally:
        if sys.modules.get(module_name) is pathlib:
            if previous_module is missing:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = previous_module
