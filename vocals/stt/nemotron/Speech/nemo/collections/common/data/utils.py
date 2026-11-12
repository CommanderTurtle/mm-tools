# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from dataclasses import fields, is_dataclass
from typing import Any, Union

import torch


def move_data_to_device(inputs: Any, device: Union[str, torch.device], non_blocking: bool = True) -> Any:
    """Recursively moves inputs to the specified device"""
    if inputs is None:
        return None
    if isinstance(inputs, torch.Tensor):
        if inputs.dtype == torch.float64 and device.type == "mps":
            # weird dataloader behavior: in some cases, it returns empty float64 tensors
            # float64 is not supported by mps, need to force-cast to float32
            inputs = inputs.to(dtype=torch.float32)
        return inputs.to(device, non_blocking=non_blocking)
    elif isinstance(inputs, (list, tuple, set)):
        return inputs.__class__([move_data_to_device(i, device, non_blocking) for i in inputs])
    elif isinstance(inputs, dict):
        return {k: move_data_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif is_dataclass(inputs):
        return type(inputs)(
            **{
                field.name: move_data_to_device(getattr(inputs, field.name), device, non_blocking)
                for field in fields(inputs)
            }
        )
    else:
        return inputs
