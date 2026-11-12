# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


from pathlib import Path
from typing import Union

try:
    import multistorageclient as msc

    HAVE_MSC = True
except (ImportError, ModuleNotFoundError):
    msc = None

    HAVE_MSC = False


MSC_PROTOCOL = "msc://"


def is_multistorageclient_url(path: Union[str, Path]):
    """
    Check if the path is a multistorageclient URL (e.g. msc://<profile>/<path>).

    Args:
        path: str, the path to check.

    Returns:
        bool, True if the path is a multistorageclient URL, False otherwise.
    """
    if isinstance(path, Path):
        return False

    has_msc_prefix = path and str(path).startswith(MSC_PROTOCOL)

    if HAVE_MSC:
        return has_msc_prefix

    if not HAVE_MSC and has_msc_prefix:
        raise ValueError(
            "Multi-Storage Client is not installed. Please install it with "
            '"pip install multi-storage-client" to handle msc:// URLs.'
        )

    return False


def import_multistorageclient():
    """Import multistorageclient if it is installed."""
    if not HAVE_MSC:
        raise ValueError(
            "Multi-Storage Client is not installed. Please install it with " '"pip install multi-storage-client".'
        )
    return msc
