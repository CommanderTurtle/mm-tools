# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MotionBricks rigs, timing, and the tensor contract of each model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path

import numpy as np

from motionbricks.inference.errors import TensorContractError
from motionbricks.inference.types import TensorMode, TensorSpec


# Core model timing shared by the SOMA and G1 representations.
MODEL_SAMPLE_RATE = 30.0
MODEL_POSES_PER_TOKEN = 4
# The model reads four context poses, then a target token of up to four target poses.
MODEL_CONTEXT_AND_TARGET_POSES = 2 * MODEL_POSES_PER_TOKEN
# Fewest and most motion tokens either model can generate.
MODEL_MIN_TOKEN_COUNT = 6
MODEL_MAX_TOKEN_COUNT = 16
# Frames removed from each end of the generated motion.
MODEL_BOUNDARY_FRAMES = MODEL_POSES_PER_TOKEN - 1


def valid_frame_count(token_count: int) -> int:
    """Return the number of valid frames left after removing the boundary frames.

    Args:
        token_count: Token count, from 6 to 16. Other values raise ValueError.
    """
    if not MODEL_MIN_TOKEN_COUNT <= token_count <= MODEL_MAX_TOKEN_COUNT:
        raise ValueError(
            f"token_count must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}, got {token_count}."
        )
    return token_count * MODEL_POSES_PER_TOKEN - 2 * MODEL_BOUNDARY_FRAMES


# The longest generated motion, which sets the capacity of every output buffer.
MODEL_MAX_VALID_FRAMES = valid_frame_count(MODEL_MAX_TOKEN_COUNT)


class ModelRig(StrEnum):
    """MotionBricks rig types.

    SOMA poses hold one local joint transform per skeleton joint. Unitree G1 poses
    are joint coordinates: a root position and rotation, then the joint angles.
    """

    SOMA = "soma"
    UNITREE_G1 = "unitree_g1"


@dataclass(frozen=True, slots=True)
class MotionBricksModelContract:
    """The tensors and joints of one exported model graph.

    load_model_manifest() reads it from the model manifest. The FP32 and mixed FP16
    models share it, because their tensors match. The timing that every model shares
    lives in the MODEL_ constants above. Invalid values raise ValueError.

    Attributes:
        name: Model name from the manifest.
        onnx_filename: File name of the ONNX model, without a directory.
        joint_names: Joint names in model order. For G1, they name the 29 joint
            angles.
        tensor_specs: Metadata of every input and output tensor.
    """

    name: str
    onnx_filename: str
    joint_names: tuple[str, ...]
    tensor_specs: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model contract name cannot be empty.")
        if not self.onnx_filename or Path(self.onnx_filename).name != self.onnx_filename:
            raise ValueError("onnx_filename must be a filename without directory components.")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique.")
        tensor_names = tuple(spec.name for spec in self.tensor_specs)
        if not tensor_names or len(set(tensor_names)) != len(tensor_names):
            raise ValueError("tensor_specs must be non-empty with unique names.")

    def validate_tensor_specs(self, actual_specs: Mapping[str, TensorSpec]) -> None:
        """Check tensor metadata against the contract's names, modes, shapes, and dtypes.

        A mismatch raises TensorContractError, which lists every difference.

        Args:
            actual_specs: Tensor metadata keyed by tensor name, such as
                InferenceEngine.tensor_specs.
        """
        expected_specs = {spec.name: spec for spec in self.tensor_specs}
        errors: list[str] = []

        missing = sorted(expected_specs.keys() - actual_specs.keys())
        unexpected = sorted(actual_specs.keys() - expected_specs.keys())
        if missing:
            errors.append("missing tensors: " + ", ".join(missing))
        if unexpected:
            errors.append("unexpected tensors: " + ", ".join(unexpected))

        for name in expected_specs.keys() & actual_specs.keys():
            expected = expected_specs[name]
            actual = actual_specs[name]
            if actual.mode is not expected.mode:
                errors.append(
                    f"{name}: expected mode {expected.mode.value}, got {actual.mode.value}"
                )
            if actual.shape != expected.shape:
                errors.append(f"{name}: expected shape {expected.shape}, got {actual.shape}")
            if actual.dtype != expected.dtype:
                errors.append(f"{name}: expected dtype {expected.dtype}, got {actual.dtype}")

        if errors:
            detail = ". ".join(errors)
            raise TensorContractError(f"Engine does not match {self.name!r}: {detail}.")


# The manifest repeats the shared timing, so a model exported with other timing fails to load.
_MANIFEST_TIMING = (
    ("sample_rate", MODEL_SAMPLE_RATE),
    ("frames_per_token", MODEL_POSES_PER_TOKEN),
    ("context_frames", MODEL_POSES_PER_TOKEN),
    ("target_capacity", MODEL_POSES_PER_TOKEN),
    ("min_pred_tokens", MODEL_MIN_TOKEN_COUNT),
    ("max_pred_tokens", MODEL_MAX_TOKEN_COUNT),
)


def load_model_manifest(path: Path) -> tuple[MotionBricksModelContract, Path, Path]:
    """Load a model manifest and return its contract, FP32 ONNX path, and mixed FP16 ONNX path.

    The ONNX files sit next to the manifest. A manifest whose timing differs from the
    MODEL_ constants raises ValueError.

    Args:
        path: Model manifest JSON file.
    """

    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load model manifest {manifest_path}: {exc}") from exc

    try:
        if manifest["schema_version"] != 1:
            raise ValueError(f"unsupported schema_version {manifest['schema_version']!r}")
        for field, value in _MANIFEST_TIMING:
            if manifest[field] != value:
                raise ValueError(f"{field} must be {value:g}")
        tensor_specs = tuple(
            TensorSpec(
                tensor["name"],
                TensorMode(tensor["mode"]),
                tuple(tensor["shape"]),
                np.dtype(tensor["dtype"]),
            )
            for tensor in manifest["tensor_specs"]
        )
        contract = MotionBricksModelContract(
            name=manifest["name"],
            onnx_filename=manifest["onnx_filename"],
            joint_names=tuple(manifest["joint_names"]),
            tensor_specs=tensor_specs,
        )
        fp16_onnx_filename = manifest["precision_variants"]["fp16"]["onnx_filename"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model manifest {manifest_path}: {exc}") from exc
    return (
        contract,
        manifest_path.with_name(contract.onnx_filename),
        manifest_path.with_name(fp16_onnx_filename),
    )
