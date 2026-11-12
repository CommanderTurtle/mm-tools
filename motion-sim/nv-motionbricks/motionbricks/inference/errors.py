# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Errors raised by TensorRT inference and model contracts."""


class InferenceError(RuntimeError):
    """Base class for MotionBricks inference failures."""


class EngineClosedError(InferenceError):
    """Raised when an operation is attempted on a closed engine."""


class TensorContractError(InferenceError):
    """Raised when an engine or model does not satisfy a tensor contract."""


class EngineBuildError(InferenceError):
    """Raised when a TensorRT plan cannot be built."""


class EngineDeserializationError(InferenceError):
    """Raised when a TensorRT plan cannot be loaded."""


class EngineExecutionError(InferenceError):
    """Raised when an inference launch fails."""


class EngineCacheError(InferenceError):
    """Raised when a TensorRT plan cache operation fails."""
