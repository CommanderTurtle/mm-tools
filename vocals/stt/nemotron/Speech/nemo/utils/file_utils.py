# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import shutil
from pathlib import Path
from typing import Union


def robust_copy(src: Union[Path, str], dst: Union[Path, str]) -> str:
    """
    Copy file from src to dst, falling back to shutil.copy if shutil.copy2 fails.
    shutil.copy2 preserves metadata, but can fail on some filesystems.
    """
    try:
        return shutil.copy2(src, dst)
    except PermissionError:
        # copy2 can fail on some filesystems due to metadata copy errors
        # (e.g., permission errors on setting timestamps).
        # In such cases, we fallback to a plain copy.
        return shutil.copy(src, dst)
