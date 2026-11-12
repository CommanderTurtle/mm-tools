# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender armature helpers for SOMA procedural-control transforms."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .definition import (
    ProceduralDefinition,
    compile_rotation_rows,
    compile_translation_rows,
    load_definition,
)
from .evaluator import SomaProceduralTransformEvaluator

DEFINITION_FILENAME = "SOMA_procedural_transforms.json"
PROP_DEFINITION_PATH = "soma_procedural_definition_path"
PROP_SCHEMA_VERSION = "soma_procedural_schema_version"
PROP_DEFINITION_NAME = "soma_procedural_definition_name"
PROP_ENABLED = "soma_procedural_enabled"
PROP_PUBLIC_JOINT_NAMES = "soma_procedural_public_joint_names"
PROP_TWIST_JOINT_NAMES = "soma_procedural_twist_joint_names"
PROP_SEGMENTS = "soma_procedural_segments"
PROP_MATRIX_ROWS = "soma_procedural_matrix_rows"
PROP_PUBLIC_BASELINE_MATRICES = "soma_procedural_public_baseline_matrices"
PROP_PUBLIC_BIND_WORLD_MATRICES = "soma_procedural_public_bind_world_matrices"
PROP_TWIST_BASELINE_MATRICES = "soma_procedural_twist_baseline_matrices"
PROP_OUTPUT_INDEX = "soma_procedural_output_index"
PROP_START_JOINT = "soma_procedural_start_joint"
PROP_END_JOINT = "soma_procedural_end_joint"
PROP_SOURCE_AXIS = "soma_procedural_source_axis"
PROP_SOURCE_AXIS_ID = "soma_procedural_source_axis_id"
PROP_SOURCE_SIGN = "soma_procedural_source_sign"
PROP_REVERSE = "soma_procedural_reverse"

_EVALUATOR_CACHE: dict[str, SomaProceduralTransformEvaluator] = {}
_IS_EVALUATING = False


def find_repo_definition(start: str | Path | None = None) -> Path | None:
    """Find the checked-in SOMA procedural definition by walking parent dirs."""

    current = Path(__file__).resolve() if start is None else Path(start).resolve()
    for parent in (current, *current.parents):
        candidate = parent / "assets" / DEFINITION_FILENAME
        if candidate.exists():
            return candidate
    return None


def _require_armature_object(armature_object: Any) -> None:
    if getattr(armature_object, "type", None) != "ARMATURE":
        name = getattr(armature_object, "name", repr(armature_object))
        raise TypeError(f"{name} is not a Blender armature object")


def _definition_path_or_default(definition_path: str | Path | None = None) -> Path:
    if definition_path:
        return Path(definition_path).expanduser().resolve()
    found = find_repo_definition()
    if found is None:
        raise ValueError(
            "Could not find assets/SOMA_procedural_transforms.json. "
            "Pass a definition path explicitly."
        )
    return found


def _matrix_payload(definition: ProceduralDefinition) -> str:
    payload = {
        "rotation": compile_rotation_rows(definition),
        "translation": compile_translation_rows(definition),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _segment_payload(definition: ProceduralDefinition) -> list[dict[str, Any]]:
    return [
        {
            "start_joint": segment.start_joint,
            "end_joint": segment.end_joint,
            "parent_joint": segment.parent_joint,
            "twist_joints": segment.twist_joints,
            "reverse": segment.reverse,
            "source_axis": segment.source_axis,
            "source_axis_id": segment.source_axis_id,
            "source_sign": segment.source_sign,
        }
        for segment in definition.segments
    ]


def _missing_pose_bones(armature_object: Any, names: tuple[str, ...]) -> list[str]:
    pose_bones = armature_object.pose.bones
    return [name for name in names if name not in pose_bones]


def configure_armature(
    armature_object: Any,
    definition_path: str | Path | None = None,
    enabled: bool = True,
) -> dict[str, tuple[str, ...] | list[str]]:
    """Attach SOMA procedural metadata to a Blender armature.

    The armature must already contain the public SOMA pose bones and the
    generated twist pose bones. The add-on writes the generated transforms onto
    those existing twist bones during handler evaluation.
    """

    _require_armature_object(armature_object)
    resolved_path = _definition_path_or_default(definition_path)
    definition = load_definition(resolved_path)

    missing_public = _missing_pose_bones(armature_object, definition.public_joint_names)
    missing_twist = _missing_pose_bones(armature_object, definition.twist_joint_names)
    if missing_public or missing_twist:
        messages = []
        if missing_public:
            messages.append(f"missing public pose bones: {missing_public[:8]}")
        if missing_twist:
            messages.append(f"missing procedural twist pose bones: {missing_twist[:8]}")
        raise ValueError("; ".join(messages))

    armature_object[PROP_DEFINITION_PATH] = str(resolved_path)
    armature_object[PROP_SCHEMA_VERSION] = definition.schema_version
    armature_object[PROP_DEFINITION_NAME] = definition.definition_name
    armature_object[PROP_ENABLED] = bool(enabled)
    armature_object[PROP_PUBLIC_JOINT_NAMES] = json.dumps(definition.public_joint_names)
    armature_object[PROP_TWIST_JOINT_NAMES] = json.dumps(definition.twist_joint_names)
    armature_object[PROP_SEGMENTS] = json.dumps(_segment_payload(definition), sort_keys=True)
    armature_object[PROP_MATRIX_ROWS] = _matrix_payload(definition)
    armature_object[PROP_PUBLIC_BASELINE_MATRICES] = json.dumps(
        [
            _matrix_to_payload(armature_object.pose.bones[joint_name].matrix_basis)
            for joint_name in definition.public_joint_names
        ],
        separators=(",", ":"),
    )
    armature_object[PROP_PUBLIC_BIND_WORLD_MATRICES] = json.dumps(
        [
            _matrix_to_payload(armature_object.pose.bones[joint_name].matrix)
            for joint_name in definition.public_joint_names
        ],
        separators=(",", ":"),
    )
    armature_object[PROP_TWIST_BASELINE_MATRICES] = json.dumps(
        [
            _matrix_to_payload(armature_object.pose.bones[joint_name].matrix_basis)
            for joint_name in definition.twist_joint_names
        ],
        separators=(",", ":"),
    )

    output_index = 0
    for segment in definition.segments:
        for twist_joint in segment.twist_joints:
            pose_bone = armature_object.pose.bones[twist_joint]
            pose_bone[PROP_OUTPUT_INDEX] = output_index
            pose_bone[PROP_START_JOINT] = segment.start_joint
            pose_bone[PROP_END_JOINT] = segment.end_joint
            pose_bone[PROP_SOURCE_AXIS] = segment.source_axis
            pose_bone[PROP_SOURCE_AXIS_ID] = segment.source_axis_id
            pose_bone[PROP_SOURCE_SIGN] = segment.source_sign
            pose_bone[PROP_REVERSE] = segment.reverse
            output_index += 1

    return {
        "public_joint_names": definition.public_joint_names,
        "twist_joint_names": definition.twist_joint_names,
        "missing_public": missing_public,
        "missing_twist": missing_twist,
    }


def clear_armature_configuration(armature_object: Any) -> None:
    """Disable SOMA procedural evaluation and remove armature-level metadata."""

    _require_armature_object(armature_object)
    for prop_name in (
        PROP_DEFINITION_PATH,
        PROP_SCHEMA_VERSION,
        PROP_DEFINITION_NAME,
        PROP_ENABLED,
        PROP_PUBLIC_JOINT_NAMES,
        PROP_TWIST_JOINT_NAMES,
        PROP_SEGMENTS,
        PROP_MATRIX_ROWS,
        PROP_PUBLIC_BASELINE_MATRICES,
        PROP_PUBLIC_BIND_WORLD_MATRICES,
        PROP_TWIST_BASELINE_MATRICES,
    ):
        if prop_name in armature_object:
            del armature_object[prop_name]
    for pose_bone in armature_object.pose.bones:
        for prop_name in (
            PROP_OUTPUT_INDEX,
            PROP_START_JOINT,
            PROP_END_JOINT,
            PROP_SOURCE_AXIS,
            PROP_SOURCE_AXIS_ID,
            PROP_SOURCE_SIGN,
            PROP_REVERSE,
        ):
            if prop_name in pose_bone:
                del pose_bone[prop_name]


def _get_evaluator(definition_path: str) -> SomaProceduralTransformEvaluator:
    key = str(Path(definition_path).expanduser().resolve())
    evaluator = _EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = SomaProceduralTransformEvaluator.from_path(key)
        _EVALUATOR_CACHE[key] = evaluator
    return evaluator


def _matrix_to_numpy(matrix: Any) -> np.ndarray:
    return np.array(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _matrix_to_payload(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _matrix_payload_to_numpy(payload: Any, prop_name: str, index: int) -> np.ndarray:
    matrix = np.array(payload, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{prop_name}[{index}] must be a 4x4 matrix")
    return matrix


def _load_baseline_matrices(
    armature_object: Any,
    prop_name: str,
    count: int,
) -> np.ndarray:
    raw_payload = armature_object.get(prop_name)
    if not raw_payload:
        return np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()

    payload = json.loads(str(raw_payload))
    if not isinstance(payload, list) or len(payload) != count:
        raise ValueError(f"{armature_object.name} has invalid {prop_name}")
    return np.stack(
        [_matrix_payload_to_numpy(matrix, prop_name, index) for index, matrix in enumerate(payload)]
    )


def _numpy_to_blender_matrix(matrix: np.ndarray) -> Any:
    from mathutils import Matrix

    return Matrix(
        tuple(tuple(float(matrix[row, column]) for column in range(4)) for row in range(4))
    )


def _translation_vector_from_numpy(matrix: np.ndarray) -> Any:
    from mathutils import Vector

    return Vector((float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])))


def _pose_source_local_matrix(pose_bone: Any) -> Any:
    """Return the posed local transform that SOMA uses for source joints.

    Blender pose channels store matrix_basis as a delta from the edit-bone rest
    transform. SOMA procedural twist extraction expects the absolute local
    transform, matching Maya joint.matrix and SOMALayer absolute_pose=True.
    """

    if pose_bone.parent is None:
        return pose_bone.matrix.copy()
    return pose_bone.parent.matrix.inverted_safe() @ pose_bone.matrix


def _pose_base_matrix(pose_bone: Any) -> Any:
    if pose_bone.parent is None:
        return pose_bone.bone.matrix_local.copy()
    return (
        pose_bone.parent.matrix
        @ pose_bone.parent.bone.matrix_local.inverted_safe()
        @ pose_bone.bone.matrix_local
    )


def _matrix_basis_from_pose_matrix(pose_bone: Any, pose_matrix: Any) -> Any:
    if pose_bone.parent is None:
        return pose_bone.bone.matrix_local.inverted_safe() @ pose_matrix
    return (
        pose_bone.bone.matrix_local.inverted_safe()
        @ pose_bone.parent.bone.matrix_local
        @ pose_bone.parent.matrix.inverted_safe()
        @ pose_matrix
    )


def _write_procedural_pose_matrix(
    pose_bone: Any,
    procedural_matrix: np.ndarray,
    baseline_matrix: np.ndarray,
) -> None:
    local_rotation = np.eye(4, dtype=np.float64)
    local_rotation[:3, :3] = procedural_matrix[:3, :3]
    local_rotation_matrix = _numpy_to_blender_matrix(local_rotation)
    baseline = _numpy_to_blender_matrix(baseline_matrix)
    if pose_bone.parent is None:
        pose_matrix = local_rotation_matrix @ pose_bone.bone.matrix_local @ baseline
    else:
        pose_matrix = (
            pose_bone.parent.matrix
            @ pose_bone.parent.bone.matrix_local.inverted_safe()
            @ local_rotation_matrix
            @ pose_bone.bone.matrix_local
            @ baseline
        )
    pose_bone.matrix_basis = _matrix_basis_from_pose_matrix(pose_bone, pose_matrix)


def evaluate_armature(armature_object: Any) -> tuple[str, ...]:
    """Evaluate and write procedural twist pose-bone transforms on an armature."""

    _require_armature_object(armature_object)
    if not bool(armature_object.get(PROP_ENABLED, False)):
        return ()

    definition_path = armature_object.get(PROP_DEFINITION_PATH)
    if not definition_path:
        raise ValueError(f"{armature_object.name} is missing {PROP_DEFINITION_PATH}")

    evaluator = _get_evaluator(str(definition_path))
    public_count = len(evaluator.public_joint_names)
    twist_count = len(evaluator.twist_joint_names)
    twist_baseline_matrices = _load_baseline_matrices(
        armature_object,
        PROP_TWIST_BASELINE_MATRICES,
        twist_count,
    )
    public_bind_world_matrices = _load_baseline_matrices(
        armature_object,
        PROP_PUBLIC_BIND_WORLD_MATRICES,
        public_count,
    )
    local_matrices = np.broadcast_to(np.eye(4, dtype=np.float64), (public_count, 4, 4)).copy()
    object_matrices = np.broadcast_to(np.eye(4, dtype=np.float64), (public_count, 4, 4)).copy()

    for index, joint_name in enumerate(evaluator.public_joint_names):
        pose_bone = armature_object.pose.bones.get(joint_name)
        if pose_bone is None:
            raise ValueError(f"{armature_object.name} is missing public pose bone {joint_name!r}")
        local_matrices[index] = _matrix_to_numpy(_pose_source_local_matrix(pose_bone))
        object_matrices[index] = _matrix_to_numpy(pose_bone.matrix)

    evaluation = evaluator.evaluate(
        local_matrices,
        source_world_matrices=object_matrices,
        source_bind_world_matrices=public_bind_world_matrices,
    )
    written = []
    for index, joint_name in enumerate(evaluation.joint_names):
        pose_bone = armature_object.pose.bones.get(joint_name)
        if pose_bone is None:
            raise ValueError(f"{armature_object.name} is missing twist pose bone {joint_name!r}")
        _write_procedural_pose_matrix(
            pose_bone,
            evaluation.matrices[index],
            twist_baseline_matrices[index],
        )
        written.append(joint_name)
    return tuple(written)


def iter_configured_armatures(scene: Any) -> list[Any]:
    """Return enabled armature objects in a Blender scene."""

    return [
        obj
        for obj in scene.objects
        if getattr(obj, "type", None) == "ARMATURE" and bool(obj.get(PROP_ENABLED, False))
    ]


def update_scene(scene: Any) -> int:
    """Evaluate all configured SOMA procedural armatures in a scene."""

    global _IS_EVALUATING
    if _IS_EVALUATING:
        return 0

    _IS_EVALUATING = True
    updated = 0
    try:
        for armature_object in iter_configured_armatures(scene):
            evaluate_armature(armature_object)
            updated += 1
    finally:
        _IS_EVALUATING = False
    return updated
