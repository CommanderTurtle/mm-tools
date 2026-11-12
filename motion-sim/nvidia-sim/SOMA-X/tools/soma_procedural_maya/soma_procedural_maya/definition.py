# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python reader for SOMA procedural-control rig definitions.

This module intentionally avoids importing :mod:`soma` so it can run inside
Maya's Python environment without Torch, SciPy, or the rest of SOMA-X installed.
It is a reference consumer for ``assets/SOMA_procedural_transforms.json``.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}
SUPPORTED_MODES = ("local_x_euler", "local_x_swing_twist", "aligned_x_swing_twist")


@dataclass(frozen=True)
class Segment:
    """One public segment with generated procedural twist joints."""

    start_joint: str
    end_joint: str
    parent_joint: str | None
    twist_joints: tuple[str, ...]
    reverse: bool
    source_axis: str
    source_axis_id: int
    source_sign: float


@dataclass(frozen=True)
class MatrixEntry:
    """One named sparse matrix entry."""

    row: str
    column: str
    value: float


@dataclass(frozen=True)
class ProceduralDefinition:
    """Validated SOMA procedural-control definition."""

    path: Path
    schema_version: str
    definition_name: str
    modes: tuple[str, ...]
    rotation_extraction_modes: tuple[str, ...]
    public_joint_names: tuple[str, ...]
    segments: tuple[Segment, ...]
    rotation_entries: tuple[MatrixEntry, ...]
    translation_entries: tuple[MatrixEntry, ...]
    raw: Mapping[str, Any]

    @property
    def twist_joint_names(self) -> tuple[str, ...]:
        return tuple(joint for segment in self.segments for joint in segment.twist_joints)

    @property
    def main_joint_names(self) -> tuple[str, ...]:
        """Main non-procedural joint names from the portable JSON schema."""
        return self.public_joint_names


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_mode(value: Any, field: str, modes: Sequence[str]) -> str:
    mode = _require_string(value, field)
    if mode not in modes:
        raise ValueError(f"{field} must be one of {tuple(modes)}, got {mode!r}")
    return mode


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    items = tuple(_require_string(item, f"{field}[]") for item in _require_sequence(value, field))
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise ValueError(f"{field} contains duplicate names: {duplicates}")
    return items


def _axis(value: Any, field: str) -> tuple[str, int]:
    axis = _require_string(value, field).lower()
    if axis not in AXIS_TO_ID:
        raise ValueError(f"{field} must be one of 'x', 'y', or 'z'")
    return axis, AXIS_TO_ID[axis]


def _sign(value: Any, field: str) -> float:
    if value in (-1, -1.0):
        return -1.0
    if value in (1, 1.0):
        return 1.0
    raise ValueError(f"{field} must be -1.0 or 1.0")


def _parse_rotation_extraction_modes(
    root: Mapping[str, Any],
    modes: Sequence[str],
    twist_joint_names: Sequence[str],
) -> tuple[str, ...]:
    raw = root.get("rotation_extraction")
    if raw is None:
        raise ValueError("rotation_extraction is required")
    if isinstance(raw, str):
        return (_require_mode(raw, "rotation_extraction", modes),) * len(twist_joint_names)

    config = _require_mapping(raw, "rotation_extraction")
    per_joint = _require_mapping(
        config.get("per_procedural_joint", {}),
        "rotation_extraction.per_procedural_joint",
    )
    unknown_joints = sorted(set(per_joint) - set(twist_joint_names))
    if unknown_joints:
        raise ValueError(
            "rotation_extraction.per_procedural_joint references unknown procedural joints: "
            f"{unknown_joints}"
        )

    default = config.get("default")
    if default is None:
        missing = [name for name in twist_joint_names if name not in per_joint]
        if missing:
            raise ValueError(
                "rotation_extraction.default is required unless every procedural joint "
                f"has an override; missing: {missing}"
            )
        default_mode = None
    else:
        default_mode = _require_mode(default, "rotation_extraction.default", modes)

    parsed_modes = []
    for joint_name in twist_joint_names:
        parsed_modes.append(
            _require_mode(
                per_joint.get(joint_name, default_mode),
                f"rotation_extraction.per_procedural_joint[{joint_name}]",
                modes,
            )
        )
    return tuple(parsed_modes)


def _parse_entries(
    matrix: Mapping[str, Any],
    matrix_name: str,
    valid_rows: set[str],
    valid_columns: set[str],
) -> tuple[MatrixEntry, ...]:
    if matrix.get("format") != "sparse_coo_named":
        raise ValueError(f"parameter_matrices.{matrix_name}.format must be 'sparse_coo_named'")
    if matrix.get("dtype") != "float32":
        raise ValueError(f"parameter_matrices.{matrix_name}.dtype must be 'float32'")

    entries = []
    seen = set()
    for index, raw_entry in enumerate(
        _require_sequence(matrix.get("entries"), f"parameter_matrices.{matrix_name}.entries")
    ):
        entry = _require_mapping(raw_entry, f"parameter_matrices.{matrix_name}.entries[{index}]")
        row = _require_string(entry.get("row"), f"{matrix_name}.row")
        column = _require_string(entry.get("column"), f"{matrix_name}.column")
        if row not in valid_rows:
            raise ValueError(f"unknown {matrix_name} matrix row: {row!r}")
        if column not in valid_columns:
            raise ValueError(f"unknown {matrix_name} matrix column: {column!r}")
        key = (row, column)
        if key in seen:
            raise ValueError(f"duplicate {matrix_name} matrix entry for {row!r}, {column!r}")
        seen.add(key)
        try:
            value = float(entry["value"])
        except KeyError as e:
            raise ValueError(f"parameter_matrices.{matrix_name}.entries[].value is required") from e
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"parameter_matrices.{matrix_name}.entries[].value must be numeric"
            ) from e
        entries.append(MatrixEntry(row=row, column=column, value=value))
    if not entries:
        raise ValueError(f"parameter_matrices.{matrix_name}.entries must not be empty")
    return tuple(entries)


def parse_definition(data: Mapping[str, Any], path: str | Path) -> ProceduralDefinition:
    """Validate a JSON-loaded SOMA procedural-control definition."""

    root = _require_mapping(data, "definition")
    schema_version = _require_string(root.get("schema_version"), "schema_version")
    definition_name = _require_string(root.get("definition_name"), "definition_name")

    modes = _string_tuple(root.get("modes"), "modes")
    unknown_modes = sorted(set(modes) - set(SUPPORTED_MODES))
    if unknown_modes:
        raise ValueError(f"unknown procedural modes: {unknown_modes}")

    public_rig = _require_mapping(root.get("public_rig_derivation"), "public_rig_derivation")
    public_joint_names = _string_tuple(
        public_rig.get("main_joint_names"),
        "public_rig_derivation.main_joint_names",
    )
    public_joint_set = set(public_joint_names)

    channel_extractors = _require_mapping(root.get("channel_extractors"), "channel_extractors")
    missing_extractors = [mode for mode in modes if mode not in channel_extractors]
    if missing_extractors:
        raise ValueError(f"missing channel extractors for modes: {missing_extractors}")

    segments = []
    for index, raw_segment in enumerate(_require_sequence(root.get("segments"), "segments")):
        segment = _require_mapping(raw_segment, f"segments[{index}]")
        source_axis, source_axis_id = _axis(
            segment.get("source_axis", "x"), "segments[].source_axis"
        )
        start_joint = _require_string(segment.get("start_joint"), "segments[].start_joint")
        end_joint = _require_string(segment.get("end_joint"), "segments[].end_joint")
        parent_joint_raw = segment.get("parent_joint")
        parent_joint = (
            _require_string(parent_joint_raw, "segments[].parent_joint")
            if parent_joint_raw is not None
            else None
        )
        control_names = (start_joint, end_joint)
        if parent_joint is not None:
            control_names = (*control_names, parent_joint)
        missing = [name for name in control_names if name not in public_joint_set]
        if missing:
            raise ValueError(f"segments[{index}] references unknown public joints: {missing}")
        segments.append(
            Segment(
                start_joint=start_joint,
                end_joint=end_joint,
                parent_joint=parent_joint,
                twist_joints=_string_tuple(segment.get("twist_joints"), "segments[].twist_joints"),
                reverse=bool(segment.get("reverse", False)),
                source_axis=source_axis,
                source_axis_id=source_axis_id,
                source_sign=_sign(segment.get("source_sign", 1.0), "segments[].source_sign"),
            )
        )

    twist_joint_names = tuple(joint for segment in segments for joint in segment.twist_joints)
    duplicate_outputs = sorted(
        {name for name in twist_joint_names if twist_joint_names.count(name) > 1}
    )
    if duplicate_outputs:
        raise ValueError(f"duplicate procedural outputs: {duplicate_outputs}")
    rotation_extraction_modes = _parse_rotation_extraction_modes(root, modes, twist_joint_names)

    parameter_matrices = _require_mapping(root.get("parameter_matrices"), "parameter_matrices")
    rotation_entries = _parse_entries(
        _require_mapping(parameter_matrices.get("rotation"), "parameter_matrices.rotation"),
        "rotation",
        set(twist_joint_names),
        public_joint_set,
    )
    translation_entries = _parse_entries(
        _require_mapping(parameter_matrices.get("translation"), "parameter_matrices.translation"),
        "translation",
        set(twist_joint_names),
        public_joint_set | set(twist_joint_names),
    )

    return ProceduralDefinition(
        path=Path(path),
        schema_version=schema_version,
        definition_name=definition_name,
        modes=modes,
        rotation_extraction_modes=rotation_extraction_modes,
        public_joint_names=public_joint_names,
        segments=tuple(segments),
        rotation_entries=rotation_entries,
        translation_entries=translation_entries,
        raw=root,
    )


def load_definition(path: str | Path) -> ProceduralDefinition:
    """Load and validate a SOMA procedural-control JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_definition(data, path)


def compile_rotation_rows(definition: ProceduralDefinition) -> dict[str, dict[str, float]]:
    """Return rotation matrix rows keyed by output twist joint then source joint."""

    rows: dict[str, dict[str, float]] = {name: {} for name in definition.twist_joint_names}
    for entry in definition.rotation_entries:
        rows[entry.row][entry.column] = entry.value
    return rows


def compile_translation_rows(definition: ProceduralDefinition) -> dict[str, dict[str, float]]:
    """Return translation matrix override rows keyed by output twist joint."""

    rows: dict[str, dict[str, float]] = {name: {} for name in definition.twist_joint_names}
    for entry in definition.translation_entries:
        rows[entry.row][entry.column] = entry.value
    return rows
