# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor metadata shared by the inference engine and model contracts."""

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any

import numpy as np

from .errors import EngineDeserializationError


class TensorMode(str, Enum):
    """Whether an engine tensor is consumed or produced by inference."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Metadata for one named static inference tensor.

    Invalid metadata raises ValueError.

    Attributes:
        name: Tensor name in the TensorRT plan.
        mode: Whether inference consumes or produces the tensor.
        shape: Static shape. Every dimension must be positive.
        dtype: Element type. Object types are rejected.
    """

    name: str
    mode: TensorMode
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tensor name cannot be empty.")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"Tensor {self.name!r} has invalid static shape {self.shape}.")
        if self.dtype.hasobject:
            raise ValueError(f"Tensor {self.name!r} cannot use object dtype.")

    @property
    def element_count(self) -> int:
        """Number of elements in the tensor."""
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Size of the tensor in bytes."""
        return self.element_count * self.dtype.itemsize


def inspect_engine_tensors(engine: Any) -> tuple[TensorSpec, ...]:
    """Read and validate the static device tensors exposed by a loaded TensorRT plan.

    An unsupported tensor raises EngineDeserializationError.

    Args:
        engine: Deserialized TensorRT engine.
    """

    import tensorrt as trt

    specs: list[TensorSpec] = []
    for index in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(index))
        trt_mode = engine.get_tensor_mode(name)
        if trt_mode == trt.TensorIOMode.INPUT:
            mode = TensorMode.INPUT
        elif trt_mode == trt.TensorIOMode.OUTPUT:
            mode = TensorMode.OUTPUT
        else:
            raise EngineDeserializationError(
                f"TensorRT tensor {name!r} has unknown I/O mode {trt_mode}."
            )

        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise EngineDeserializationError(
                f"TensorRT tensor {name!r} is not device-resident."
            )
        shape = tuple(int(dimension) for dimension in engine.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            raise EngineDeserializationError(
                f"TensorRT tensor {name!r} has unsupported dynamic shape {shape}."
            )
        try:
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        except Exception as exc:
            raise EngineDeserializationError(
                f"TensorRT tensor {name!r} has an unsupported dtype: {exc}"
            ) from exc
        specs.append(TensorSpec(name, mode, shape, dtype))

    if not specs:
        raise EngineDeserializationError("The TensorRT plan exposes no input or output tensors.")
    return tuple(specs)
