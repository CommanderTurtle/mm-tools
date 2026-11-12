# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from dataclasses import dataclass
from typing import Optional

__all__ = ['Config']


@dataclass
class Config:
    """
    Abstract NeMo Configuration class.

    Args:
        name: name of the module/dataset/loss/model object (used in serialization, DEFAULT: None)
    """

    name: Optional[str] = None
