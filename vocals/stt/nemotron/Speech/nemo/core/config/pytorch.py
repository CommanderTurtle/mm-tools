# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

__all__ = ['DataLoaderConfig']


@dataclass
class DataLoaderConfig:
    """
    Configuration of PyTorch DataLoader.

    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    batch_size: int = MISSING
    shuffle: bool = False
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    num_workers: int = 0
    collate_fn: Optional[Any] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: Optional[Any] = None
    multiprocessing_context: Optional[Any] = None
