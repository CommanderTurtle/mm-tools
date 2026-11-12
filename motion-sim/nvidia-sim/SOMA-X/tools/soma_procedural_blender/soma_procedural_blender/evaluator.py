# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy evaluator for the SOMA procedural Blender add-on."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .definition import MatrixEntry, ProceduralDefinition, load_definition


@dataclass(frozen=True)
class ProceduralEvaluation:
    """Transforms generated for the procedural twist joints."""

    joint_names: tuple[str, ...]
    matrices: np.ndarray


def _matrix_to_euler_xyz(rotations: np.ndarray) -> np.ndarray:
    sy = np.clip(-rotations[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(sy)
    x = np.arctan2(rotations[..., 2, 1], rotations[..., 2, 2])
    z = np.arctan2(rotations[..., 1, 0], rotations[..., 0, 0])
    return np.stack((x, y, z), axis=-1)


def _local_x_euler_from_matrix(rotations: np.ndarray) -> np.ndarray:
    return np.arctan2(rotations[..., 2, 1], rotations[..., 1, 1])


def _matrix_to_quaternion_xyzw(rotations: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m00 = rotations[..., 0, 0]
    m01 = rotations[..., 0, 1]
    m02 = rotations[..., 0, 2]
    m10 = rotations[..., 1, 0]
    m11 = rotations[..., 1, 1]
    m12 = rotations[..., 1, 2]
    m20 = rotations[..., 2, 0]
    m21 = rotations[..., 2, 1]
    m22 = rotations[..., 2, 2]

    qw = 0.5 * np.sqrt(np.maximum(1.0 + m00 + m11 + m22, 0.0))
    qx = 0.5 * np.copysign(np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)), m21 - m12)
    qy = 0.5 * np.copysign(np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)), m02 - m20)
    qz = 0.5 * np.copysign(np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)), m10 - m01)
    quaternions = np.stack((qx, qy, qz, qw), axis=-1)
    norm = np.maximum(np.linalg.norm(quaternions, axis=-1, keepdims=True), eps)
    return quaternions / norm


def _swing_twist_channels_from_matrix(
    rotations: np.ndarray,
    axis_ids: np.ndarray,
    axis_signs: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    quaternions = _matrix_to_quaternion_xyzw(rotations, eps=eps)
    q_vec = quaternions[..., :3]
    half_w = quaternions[..., 3] + 1.0
    half_norm = np.maximum(np.sqrt(np.square(half_w) + np.square(q_vec).sum(axis=-1)), eps)
    half_w = half_w / half_norm
    half_vec = q_vec / half_norm[..., None]
    twist_imag = np.take_along_axis(half_vec, axis_ids.reshape(-1, 1), axis=-1).reshape(-1)
    twist_half_angles = 2.0 * np.arctan2(twist_imag, half_w)
    return 2.0 * twist_half_angles * axis_signs


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), eps)


def _project_to_plane(vectors: np.ndarray, normals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    projected = vectors - np.sum(vectors * normals, axis=-1, keepdims=True) * normals
    return _normalize_vectors(projected, eps=eps)


def _quaternion_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _quaternion_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def _quaternion_normalize_xyzw(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)


def _bind_alignment_quaternions(
    bind_world_matrices: np.ndarray,
    start_ids: np.ndarray,
    end_ids: np.ndarray,
) -> np.ndarray:
    start = bind_world_matrices[start_ids]
    end = bind_world_matrices[end_ids]
    start_rot = start[..., :3, :3]
    span = _normalize_vectors(end[..., :3, 3] - start[..., :3, 3])

    local_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    local_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_x = start_rot @ local_x
    x_axis = span * np.where(np.sum(up_x * span, axis=-1, keepdims=True) >= 0.0, 1.0, -1.0)

    candidates = (
        start_rot @ local_y,
        start_rot @ local_z,
        np.broadcast_to(local_y, x_axis.shape),
        np.broadcast_to(local_z, x_axis.shape),
    )
    y_axis = _project_to_plane(candidates[-1], x_axis)
    for candidate in reversed(candidates[:-1]):
        projected_raw = candidate - np.sum(candidate * x_axis, axis=-1, keepdims=True) * x_axis
        valid = np.linalg.norm(projected_raw, axis=-1, keepdims=True) > 1e-8
        y_axis = np.where(valid, _normalize_vectors(projected_raw), y_axis)

    z_axis = _normalize_vectors(np.cross(x_axis, y_axis))
    y_axis = _normalize_vectors(np.cross(z_axis, x_axis))
    align_rot = np.stack((x_axis, y_axis, z_axis), axis=-1)
    return _matrix_to_quaternion_xyzw(align_rot)


def _aligned_virtual_quaternions(
    world_rotations: np.ndarray,
    bind_quaternions: np.ndarray,
    align_quaternions: np.ndarray,
    segment_ids: np.ndarray,
    joint_ids: np.ndarray,
) -> np.ndarray:
    q_current = _matrix_to_quaternion_xyzw(world_rotations[joint_ids])
    q_bind_inv = _quaternion_conjugate_xyzw(bind_quaternions[joint_ids])
    q_align = align_quaternions[segment_ids]
    q = _quaternion_multiply_xyzw(_quaternion_multiply_xyzw(q_current, q_bind_inv), q_align)
    return _quaternion_normalize_xyzw(q)


def _aligned_relative_twist_angles(
    source_world_matrices: np.ndarray,
    source_bind_world_matrices: np.ndarray,
    segment_ids: np.ndarray,
    driver_ids: np.ndarray,
    reference_ids: np.ndarray,
    start_ids: np.ndarray,
    end_ids: np.ndarray,
) -> np.ndarray:
    bind_quaternions = _matrix_to_quaternion_xyzw(source_bind_world_matrices[:, :3, :3])
    align_quaternions = _bind_alignment_quaternions(
        source_bind_world_matrices,
        start_ids,
        end_ids,
    )
    q_driver = _aligned_virtual_quaternions(
        source_world_matrices[:, :3, :3],
        bind_quaternions,
        align_quaternions,
        segment_ids,
        driver_ids,
    )
    q_reference = _aligned_virtual_quaternions(
        source_world_matrices[:, :3, :3],
        bind_quaternions,
        align_quaternions,
        segment_ids,
        reference_ids,
    )
    q_rel = _quaternion_multiply_xyzw(_quaternion_conjugate_xyzw(q_reference), q_driver)
    q_rel = _quaternion_normalize_xyzw(q_rel)
    q_rel = np.where(q_rel[..., 3:] < 0.0, -q_rel, q_rel)
    q_half = _quaternion_normalize_xyzw(
        np.concatenate((q_rel[..., :3], q_rel[..., 3:] + 1.0), axis=-1)
    )
    twist_half_angles = 2.0 * np.arctan2(q_half[..., 0], q_half[..., 3])
    return 2.0 * twist_half_angles


def _axis_rotation_matrices(
    angles: np.ndarray,
    axis_ids: np.ndarray,
    axis_signs: np.ndarray,
) -> np.ndarray:
    signed = angles * axis_signs
    cos = np.cos(signed)
    sin = np.sin(signed)
    matrices = np.broadcast_to(np.eye(3, dtype=np.float64), (len(angles), 3, 3)).copy()

    x_mask = axis_ids == 0
    matrices[x_mask, 1, 1] = cos[x_mask]
    matrices[x_mask, 1, 2] = -sin[x_mask]
    matrices[x_mask, 2, 1] = sin[x_mask]
    matrices[x_mask, 2, 2] = cos[x_mask]

    y_mask = axis_ids == 1
    matrices[y_mask, 0, 0] = cos[y_mask]
    matrices[y_mask, 0, 2] = sin[y_mask]
    matrices[y_mask, 2, 0] = -sin[y_mask]
    matrices[y_mask, 2, 2] = cos[y_mask]

    z_mask = axis_ids == 2
    matrices[z_mask, 0, 0] = cos[z_mask]
    matrices[z_mask, 0, 1] = -sin[z_mask]
    matrices[z_mask, 1, 0] = sin[z_mask]
    matrices[z_mask, 1, 1] = cos[z_mask]

    return matrices


def _identity_transform_array(count: int) -> np.ndarray:
    return np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()


def _as_transform_array(value: np.ndarray, count: int, field: str) -> np.ndarray:
    transforms = np.asarray(value, dtype=np.float64)
    if transforms.shape != (count, 4, 4):
        raise ValueError(f"{field} must have shape ({count}, 4, 4), got {transforms.shape}")
    return transforms


def _dense_matrix(
    entries: tuple[MatrixEntry, ...],
    row_names: tuple[str, ...],
    column_names: tuple[str, ...],
    matrix_name: str,
) -> np.ndarray:
    columns = {name: index for index, name in enumerate(column_names)}
    rows = {name: index for index, name in enumerate(row_names)}
    matrix = np.zeros((len(row_names), len(column_names)), dtype=np.float64)
    unsupported_columns = []
    for entry in entries:
        column_index = columns.get(entry.column)
        if column_index is None:
            unsupported_columns.append(entry.column)
            continue
        matrix[rows[entry.row], column_index] = entry.value
    if unsupported_columns:
        unique_columns = sorted(set(unsupported_columns))
        raise ValueError(
            f"{matrix_name} matrix references non-input columns: {unique_columns}. "
            "The Blender add-on only consumes public joint inputs."
        )
    return matrix


def _rotation_parameter_matrices_by_mode(
    rotation_parameter_matrix: np.ndarray,
    mode_names: tuple[str, ...],
    rotation_extraction_modes: tuple[str, ...],
) -> np.ndarray:
    mode_to_idx = {mode: index for index, mode in enumerate(mode_names)}
    matrices = np.zeros((len(mode_names), *rotation_parameter_matrix.shape), dtype=np.float64)
    for row, mode in enumerate(rotation_extraction_modes):
        matrices[mode_to_idx[mode], row] = rotation_parameter_matrix[row]
    return matrices


class SomaProceduralTransformEvaluator:
    """Evaluate SOMA procedural twist transforms with dense NumPy matrices."""

    def __init__(
        self,
        definition: ProceduralDefinition,
    ) -> None:
        self.definition = definition
        self.public_joint_names = definition.public_joint_names
        self.twist_joint_names = definition.twist_joint_names
        rotation_extraction_modes = definition.rotation_extraction_modes
        mode_names = tuple(
            mode_name for mode_name in definition.modes if mode_name in rotation_extraction_modes
        )
        self.mode = (
            rotation_extraction_modes[0] if len(set(rotation_extraction_modes)) == 1 else None
        )
        self.rotation_extraction_modes = rotation_extraction_modes
        self.rotation_extraction_mode_names = mode_names
        self.rotation_parameter_matrix = _dense_matrix(
            definition.rotation_entries,
            self.twist_joint_names,
            self.public_joint_names,
            "rotation",
        )
        self.translation_parameter_matrix = _dense_matrix(
            definition.translation_entries,
            self.twist_joint_names,
            self.public_joint_names,
            "translation",
        )
        self.rotation_parameter_matrices_by_mode = _rotation_parameter_matrices_by_mode(
            self.rotation_parameter_matrix,
            mode_names,
            rotation_extraction_modes,
        )
        self.source_axis_ids, self.source_axis_signs = self._source_twist_channels()
        self.twist_axis_ids = np.array(
            [
                segment.source_axis_id
                for segment in definition.segments
                for _ in segment.twist_joints
            ],
            dtype=np.int64,
        )
        self.twist_axis_signs = np.array(
            [
                float(segment.source_sign)
                for segment in definition.segments
                for _ in segment.twist_joints
            ],
            dtype=np.float64,
        )
        source_by_name = {name: index for index, name in enumerate(self.public_joint_names)}
        self.segment_start_ids = np.array(
            [source_by_name[segment.start_joint] for segment in definition.segments],
            dtype=np.int64,
        )
        self.segment_end_ids = np.array(
            [source_by_name[segment.end_joint] for segment in definition.segments],
            dtype=np.int64,
        )
        self.segment_parent_ids = np.array(
            [
                source_by_name[segment.parent_joint]
                if segment.parent_joint is not None
                else source_by_name[segment.start_joint]
                for segment in definition.segments
            ],
            dtype=np.int64,
        )
        self.segment_ids = np.arange(len(definition.segments), dtype=np.int64)

    @classmethod
    def from_path(
        cls,
        definition_path: str | Path,
    ) -> "SomaProceduralTransformEvaluator":
        return cls(load_definition(definition_path))

    def _source_twist_channels(self) -> tuple[np.ndarray, np.ndarray]:
        axis_ids = np.zeros((len(self.public_joint_names),), dtype=np.int64)
        signs = np.ones((len(self.public_joint_names),), dtype=np.float64)
        source_by_name = {name: index for index, name in enumerate(self.public_joint_names)}
        assigned: dict[str, tuple[int, float]] = {}
        for segment in self.definition.segments:
            for name in (segment.start_joint, segment.end_joint):
                spec = (segment.source_axis_id, float(segment.source_sign))
                if name in assigned and assigned[name] != spec:
                    raise ValueError(f"Conflicting SOMA twist source channel for joint {name!r}")
                assigned[name] = spec
        for name, (axis_id, sign) in assigned.items():
            index = source_by_name[name]
            axis_ids[index] = axis_id
            signs[index] = sign
        return axis_ids, signs

    def evaluate(
        self,
        source_local_matrices: np.ndarray,
        source_world_matrices: np.ndarray | None = None,
        source_bind_world_matrices: np.ndarray | None = None,
    ) -> ProceduralEvaluation:
        """Return procedural twist transforms from public joint inputs.

        ``source_local_matrices`` are used for local twist extraction. If
        ``source_world_matrices`` is provided, its translations are multiplied by
        the sidecar translation parameter matrix; otherwise local translations
        are used as a fallback for simple reference scenes.
        """

        public_count = len(self.public_joint_names)
        local_matrices = _as_transform_array(
            source_local_matrices,
            public_count,
            "source_local_matrices",
        )
        if source_world_matrices is None:
            world_matrices = local_matrices
        else:
            world_matrices = _as_transform_array(
                source_world_matrices,
                public_count,
                "source_world_matrices",
            )
        if source_bind_world_matrices is None:
            bind_world_matrices = world_matrices
        else:
            bind_world_matrices = _as_transform_array(
                source_bind_world_matrices,
                public_count,
                "source_bind_world_matrices",
            )

        source_rotations = local_matrices[:, :3, :3]
        twist_angles = np.zeros((len(self.twist_joint_names),), dtype=np.float64)
        for mode_index, mode in enumerate(self.rotation_extraction_mode_names):
            if mode == "local_x_euler":
                euler_channels = _matrix_to_euler_xyz(source_rotations)
                euler_channels[:, 0] = _local_x_euler_from_matrix(source_rotations)
                twist_values = (
                    np.take_along_axis(
                        euler_channels,
                        self.source_axis_ids.reshape(-1, 1),
                        axis=-1,
                    )
                    .reshape(-1)
                    .astype(np.float64)
                )
                twist_values *= self.source_axis_signs
            elif mode == "local_x_swing_twist":
                twist_values = _swing_twist_channels_from_matrix(
                    source_rotations,
                    self.source_axis_ids,
                    self.source_axis_signs,
                )
            elif mode == "aligned_x_swing_twist":
                local_twist = _aligned_relative_twist_angles(
                    world_matrices,
                    bind_world_matrices,
                    self.segment_ids,
                    self.segment_end_ids,
                    self.segment_start_ids,
                    self.segment_start_ids,
                    self.segment_end_ids,
                )
                inherited_twist = _aligned_relative_twist_angles(
                    world_matrices,
                    bind_world_matrices,
                    self.segment_ids,
                    self.segment_start_ids,
                    self.segment_parent_ids,
                    self.segment_start_ids,
                    self.segment_end_ids,
                )
                twist_values = np.zeros((len(self.public_joint_names),), dtype=np.float64)
                for segment_index, segment in enumerate(self.definition.segments):
                    if segment.reverse:
                        twist_values[self.segment_start_ids[segment_index]] = inherited_twist[
                            segment_index
                        ]
                    twist_values[self.segment_end_ids[segment_index]] = local_twist[segment_index]
            else:
                raise RuntimeError(f"Unsupported SOMA procedural twist mode: {mode!r}")
            twist_angles += twist_values @ self.rotation_parameter_matrices_by_mode[mode_index].T
        matrices = _identity_transform_array(len(self.twist_joint_names))
        matrices[:, :3, :3] = _axis_rotation_matrices(
            twist_angles,
            self.twist_axis_ids,
            self.twist_axis_signs,
        )
        matrices[:, :3, 3] = self.translation_parameter_matrix @ world_matrices[:, :3, 3]
        return ProceduralEvaluation(joint_names=self.twist_joint_names, matrices=matrices)
