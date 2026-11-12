# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL-family rig layers backed by SMPL and SMPL-X assets."""

from pathlib import Path
from typing import Any

import torch

from .._smpl_family_loader import load_smpl_family_model
from ..geometry.batched_skinning import BatchedSkinning
from ..geometry.lbs import batch_rodrigues
from ..units import Unit
from .transfer import (
    SMPLFamilyPoseTransferResult,
    SMPLFamilyTopologyBridge,
    transfer_smpl_family_pose_parameters,
)

__all__ = [
    "SMPL_JOINT_NAMES",
    "SMPLX_JOINT_NAMES",
    "SMPLFamilyPoseTransferResult",
    "SMPLFamilyTopologyBridge",
    "SMPLLayer",
    "SMPLXLayer",
    "create_smpl_family_layer",
    "transfer_smpl_family_pose_parameters",
]

SMPL_JOINT_NAMES = [
    "Pelvis",
    "LeftHip",
    "RightHip",
    "Spine1",
    "LeftKnee",
    "RightKnee",
    "Spine2",
    "LeftAnkle",
    "RightAnkle",
    "Spine3",
    "LeftFoot",
    "RightFoot",
    "Neck",
    "LeftCollar",
    "RightCollar",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
    "LeftHand",
    "RightHand",
]

SMPLX_JOINT_NAMES = [
    "Pelvis",
    "LeftHip",
    "RightHip",
    "Spine1",
    "LeftKnee",
    "RightKnee",
    "Spine2",
    "LeftAnkle",
    "RightAnkle",
    "Spine3",
    "LeftFoot",
    "RightFoot",
    "Neck",
    "LeftCollar",
    "RightCollar",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftHand",
    "RightHand",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftIndex1",
    "LeftIndex2",
    "LeftIndex3",
    "LeftMiddle1",
    "LeftMiddle2",
    "LeftMiddle3",
    "LeftPinky1",
    "LeftPinky2",
    "LeftPinky3",
    "LeftRing1",
    "LeftRing2",
    "LeftRing3",
    "LeftThumb1",
    "LeftThumb2",
    "LeftThumb3",
    "RightIndex1",
    "RightIndex2",
    "RightIndex3",
    "RightMiddle1",
    "RightMiddle2",
    "RightMiddle3",
    "RightPinky1",
    "RightPinky2",
    "RightPinky3",
    "RightRing1",
    "RightRing2",
    "RightRing3",
    "RightThumb1",
    "RightThumb2",
    "RightThumb3",
]


def _coerce_unit(unit: Unit | str) -> Unit:
    if isinstance(unit, Unit):
        return unit
    return Unit.from_name(unit)


def _identity_coeffs(
    values: torch.Tensor | None,
    *,
    num_coeffs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if values is None:
        return torch.zeros(1, num_coeffs, dtype=dtype, device=device)

    coeffs = values.to(dtype=dtype, device=device)
    if coeffs.ndim == 1:
        coeffs = coeffs.unsqueeze(0)
    if coeffs.ndim != 2:
        raise ValueError(f"Expected identity coefficients with shape (B, C), got {coeffs.shape}.")

    if coeffs.shape[1] == num_coeffs:
        return coeffs
    if coeffs.shape[1] > num_coeffs:
        return coeffs[:, :num_coeffs]

    pad = torch.zeros(
        coeffs.shape[0],
        num_coeffs - coeffs.shape[1],
        dtype=coeffs.dtype,
        device=coeffs.device,
    )
    return torch.cat([coeffs, pad], dim=1)


class _SMPLFamilyLBSLayer(torch.nn.Module):
    """Shared PoseInversion-compatible LBS wrapper for native SMPL-family rigs."""

    NATIVE_UNIT = Unit.METERS

    def __init__(
        self,
        data_root: str | Path,
        *,
        device: str | torch.device = "cpu",
        mode: str = "warp",
        output_unit: Unit | str = Unit.METERS,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.device = torch.device(device)
        self.mode = mode
        self.output_unit = _coerce_unit(output_unit)
        self._native_to_output_scale = (
            self.NATIVE_UNIT.meters_per_unit / self.output_unit.meters_per_unit
        )
        self.low_lod = False
        self.nv_lod_mid_to_low = None
        self.root_joint_idx = 0
        self.register_buffer(
            "excluded_vert_ids",
            torch.empty(0, dtype=torch.long, device=self.device),
            persistent=False,
        )

    @property
    def num_joints(self) -> int:
        return int(self.joint_parent_ids.shape[0])

    def _shape_native(self, identity_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _shape(self, identity_coeffs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        coeffs = _identity_coeffs(
            identity_coeffs,
            num_coeffs=self.num_identity_coeffs,
            device=self.device,
            dtype=self._v_template.dtype,
        )
        rest_shape, joints = self._shape_native(coeffs)
        scale = self._native_to_output_scale
        return rest_shape * scale, joints * scale

    def _make_bind_world(self, joints: torch.Tensor) -> torch.Tensor:
        batch_size, num_joints, _ = joints.shape
        bind_world = torch.eye(4, dtype=joints.dtype, device=joints.device).repeat(
            batch_size, num_joints, 1, 1
        )
        bind_world[:, :, :3, 3] = joints
        return bind_world

    def _pose_corrective_offsets(self, rotations: torch.Tensor) -> torch.Tensor:
        batch_size = rotations.shape[0]
        ident = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
        pose_feature = (rotations[:, 1:] - ident).reshape(batch_size, -1)
        posedirs = self.posedirs.to(dtype=rotations.dtype, device=rotations.device)

        if posedirs.ndim == 3:
            posedirs = posedirs.permute(2, 0, 1).reshape(posedirs.shape[2], -1)
        elif posedirs.ndim != 2:
            raise ValueError(f"Expected posedirs to be rank 2 or 3, got shape {posedirs.shape}.")

        if (
            posedirs.shape[0] != pose_feature.shape[1]
            and posedirs.shape[1] == pose_feature.shape[1]
        ):
            posedirs = posedirs.transpose(0, 1)
        if posedirs.shape[0] != pose_feature.shape[1]:
            raise ValueError(
                "SMPL-family posedirs do not match the pose feature width: "
                f"{posedirs.shape[0]} vs {pose_feature.shape[1]}."
            )

        return (pose_feature @ posedirs).reshape(batch_size, -1, 3) * self._native_to_output_scale

    def prepare_identity(
        self,
        identity_coeffs: torch.Tensor | None,
        scale_params: torch.Tensor | None = None,
        repose_to_bind_pose: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        del scale_params, repose_to_bind_pose, kwargs
        rest_shape, joints = self._shape(identity_coeffs)
        bind_world = self._make_bind_world(joints)

        self._cached_rest_shape = rest_shape
        self._cached_bind_transforms_world = bind_world
        self.bind_shape = rest_shape[0]
        self.bind_pose_world = bind_world[0]
        self.t_pose_world = bind_world[0].clone()
        self.rig_data["bind_shape"] = self.bind_shape.detach().cpu().numpy()

        self.batched_skinning = BatchedSkinning(
            joint_parent_ids=self.joint_parent_ids,
            skinning_weights=self.skinning_weights,
            bind_world_transforms=bind_world,
            bind_shapes=rest_shape,
            joint_orient=self.t_pose_world,
            mode=self.mode,
            global_translation_joint_idx=self.root_joint_idx,
        )
        self._identity_prepared = True

    def pose(
        self,
        poses: torch.Tensor,
        pose2rot: bool = True,
        apply_correctives: bool = True,
        absolute_pose: bool = False,
        global_translation: torch.Tensor | None = None,
        fk_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not getattr(self, "_identity_prepared", False):
            raise RuntimeError("Call prepare_identity() before pose().")

        batch_size = poses.shape[0]
        if pose2rot:
            rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(
                batch_size, self.num_joints, 3, 3
            )
        else:
            rotations = poses.reshape(batch_size, self.num_joints, 3, 3)
        rotations = rotations.to(dtype=self.bind_shape.dtype, device=self.device)

        if global_translation is None:
            global_translation = torch.zeros(
                batch_size,
                3,
                dtype=rotations.dtype,
                device=rotations.device,
            )
        else:
            global_translation = global_translation.to(
                dtype=rotations.dtype,
                device=rotations.device,
            )

        rest_shape = self._cached_rest_shape
        bind_transforms = self._cached_bind_transforms_world
        if apply_correctives and not fk_only:
            rest_shape = rest_shape + self._pose_corrective_offsets(rotations)

        if bind_transforms.shape[0] == 1 and rest_shape.shape[0] > 1:
            bind_transforms = bind_transforms.expand(rest_shape.shape[0], -1, -1, -1)
        elif rest_shape.shape[0] == 1 and bind_transforms.shape[0] > 1:
            rest_shape = rest_shape.expand(bind_transforms.shape[0], -1, -1)
        self.batched_skinning.rebind(bind_transforms, rest_shape)

        if fk_only:
            transforms = self.batched_skinning.forward_kinematics(
                rotations,
                global_translation=global_translation,
                absolute_pose=absolute_pose,
            )
            return {"joints": transforms[:, :, :3, 3], "transforms": transforms}

        vertices, transforms = self.batched_skinning.pose(
            rotations,
            global_translation=global_translation,
            absolute_pose=absolute_pose,
            return_transforms=True,
        )
        return {
            "vertices": vertices,
            "joints": transforms[:, :, :3, 3],
            "transforms": transforms,
        }

    def forward(
        self,
        poses: torch.Tensor,
        identity_coeffs: torch.Tensor | None = None,
        pose2rot: bool = True,
        apply_correctives: bool = True,
        absolute_pose: bool = False,
        global_translation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.prepare_identity(identity_coeffs)
        return self.pose(
            poses,
            pose2rot=pose2rot,
            apply_correctives=apply_correctives,
            absolute_pose=absolute_pose,
            global_translation=global_translation,
        )


class _SMPLFamilyRigLayer(_SMPLFamilyLBSLayer):
    _MODEL_TYPE = ""
    _JOINT_NAMES: list[str] = []

    def __init__(
        self,
        data_root: str | Path,
        *,
        device: str | torch.device = "cpu",
        mode: str = "warp",
        output_unit: Unit | str = Unit.METERS,
        gender: str = "neutral",
        model_path: str | Path | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(data_root, device=device, mode=mode, output_unit=output_unit)
        self.model_type = self._MODEL_TYPE
        self.model_spec = self.model_type
        self.gender = gender.lower()
        self.topology_family = "body"
        self.identity_model_type = f"{self.model_type}_native"
        self.identity_model_kwargs = {"model_type": self.model_type, "gender": self.gender}
        self.rig_data = {"joint_names": self._JOINT_NAMES}
        self.default_skin_mesh_name = self.model_type
        model_dir = self.data_root / self.model_type.upper()
        self.base_mesh_path = model_dir / "base_body.obj"
        self.wrap_mesh_path = model_dir / "SOMA_wrap.obj"

        resolved_model_path = self._resolve_model_path(model_path)
        self.model_path = resolved_model_path
        tensors = self._load_smpl_family_model(resolved_model_path, model_kwargs)
        parent_ids = tensors["parents"].clone().to(dtype=torch.long)
        parent_ids[0] = 0
        if parent_ids.numel() != len(self._JOINT_NAMES):
            raise ValueError(
                f"Expected {len(self._JOINT_NAMES)} {self.model_type.upper()} joints, "
                f"got {parent_ids.numel()}."
            )

        self.num_identity_coeffs = int(tensors["shapedirs"].shape[2])
        self.register_buffer("_v_template", tensors["v_template"], persistent=False)
        self.register_buffer("_shapedirs", tensors["shapedirs"], persistent=False)
        self.register_buffer("_J_regressor", tensors["J_regressor"], persistent=False)
        self.register_buffer("skinning_weights", tensors["lbs_weights"], persistent=False)
        self.register_buffer("joint_parent_ids", parent_ids.to(self.device), persistent=False)
        self.register_buffer("faces", tensors["faces"], persistent=False)
        self.register_buffer("posedirs", tensors["posedirs"], persistent=False)

        self.prepare_identity(None)

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        if model_path is not None:
            return Path(model_path)
        model_dir = self.data_root / self.model_type.upper()
        prefix = self.model_type.upper()
        gender = self.gender.upper()
        for suffix in ("npz", "pkl"):
            candidate = model_dir / f"{prefix}_{gender}.{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find {prefix}_{gender}.npz or {prefix}_{gender}.pkl in {model_dir}."
        )

    def _load_smpl_family_model(
        self, model_path: Path, model_kwargs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        num_betas = int(model_kwargs.pop("num_betas", 10))
        tensors = load_smpl_family_model(
            model_path,
            model_type=self.model_type,
            num_betas=num_betas,
        )
        return {
            "v_template": torch.from_numpy(tensors["v_template"]).to(self.device),
            "shapedirs": torch.from_numpy(tensors["shapedirs"]).to(self.device),
            "J_regressor": torch.from_numpy(tensors["J_regressor"]).to(self.device),
            "lbs_weights": torch.from_numpy(tensors["lbs_weights"]).to(self.device),
            "parents": torch.from_numpy(tensors["parents"]).to(self.device),
            "faces": torch.from_numpy(tensors["faces"]).to(dtype=torch.long, device=self.device),
            "posedirs": torch.from_numpy(tensors["posedirs"]).to(self.device),
        }

    def _shape_native(self, identity_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blend = torch.einsum("bk,vdk->bvd", identity_coeffs, self._shapedirs)
        verts = self._v_template.unsqueeze(0) + blend
        joints = torch.einsum("jv,bvd->bjd", self._J_regressor, verts)
        return verts, joints


class SMPLLayer(_SMPLFamilyRigLayer):
    """SMPL LBS rig adapter implementing the PoseInversion layer contract."""

    _MODEL_TYPE = "smpl"
    _JOINT_NAMES = SMPL_JOINT_NAMES


class SMPLXLayer(_SMPLFamilyRigLayer):
    """SMPL-X LBS rig adapter implementing the PoseInversion layer contract."""

    _MODEL_TYPE = "smplx"
    _JOINT_NAMES = SMPLX_JOINT_NAMES


def create_smpl_family_layer(
    model: str,
    data_root: str | Path,
    *,
    device: str | torch.device = "cpu",
    mode: str = "warp",
    output_unit: Unit | str = Unit.METERS,
    **kwargs: Any,
) -> _SMPLFamilyLBSLayer:
    """Create an SMPL-family rig layer from a compact model spec."""
    spec = model.lower().replace("_", "-")
    if spec in {"smpl", "body-smpl"}:
        return SMPLLayer(data_root, device=device, mode=mode, output_unit=output_unit, **kwargs)
    if spec in {"smplx", "smpl-x", "body-smplx", "body-smpl-x"}:
        return SMPLXLayer(data_root, device=device, mode=mode, output_unit=output_unit, **kwargs)
    if spec in {"mano-left", "left-mano", "mano:l", "mano-l"}:
        from ..hand.mano import MANOLayer

        return MANOLayer(data_root, "left", device=device, mode=mode, output_unit=output_unit)
    if spec in {"mano-right", "right-mano", "mano:r", "mano-r"}:
        from ..hand.mano import MANOLayer

        return MANOLayer(data_root, "right", device=device, mode=mode, output_unit=output_unit)
    raise ValueError(
        f"Unsupported SMPL-family model {model!r}. Use 'smpl', 'smplx', "
        "'mano-left', or 'mano-right'."
    )
