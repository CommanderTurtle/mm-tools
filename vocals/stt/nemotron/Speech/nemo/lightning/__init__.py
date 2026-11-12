# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from typing import Union

from lightning.fabric.plugins.environments import slurm
from lightning.pytorch import plugins as _pl_plugins

from nemo.lightning.base import get_vocab_size, teardown


# We monkey patch because nvidia uses a naming convention for SLURM jobs
def _is_slurm_interactive_mode():
    job_name = slurm.SLURMEnvironment.job_name()
    return job_name is None or job_name.endswith("bash") or job_name.endswith("interactive")


slurm._is_slurm_interactive_mode = _is_slurm_interactive_mode  # noqa: SLF001


_pl_plugins._PLUGIN_INPUT = Union[_pl_plugins._PLUGIN_INPUT]  # noqa: SLF001


__all__ = [
    "get_vocab_size",
    "teardown",
]
