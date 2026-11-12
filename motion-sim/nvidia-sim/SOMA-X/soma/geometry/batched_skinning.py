# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched forward kinematics and linear blend skinning utilities."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ._utils import mask_1d, one_hot_1d, one_hot_2d, require_torch_tensors
from .lbs import lbs
from .rig_utils import (
    apply_joint_orient_local,
    compute_skeleton_levels,
    joint_local_to_world_levelorder,
    joint_world_to_local,
    precompute_joint_orient,
)
from .transforms import SE3_from_Rt


@dataclass(frozen=True)
class FKTopology:
    """Optional FK-only source topology for a target LBS rig.

    ``BatchedSkinning`` always owns one target topology for LBS.  A source
    topology describes a related FK rig whose world transforms can be expanded
    into the target topology by caller-owned logic.
    """

    parent_ids: Sequence[int]
    target_joint_indices: Sequence[int] | torch.Tensor | None = None
    joint_orient: torch.Tensor | None = None
    global_translation_joint_idx: int | None = None
    bind_world_transforms: torch.Tensor | None = None


def topk_skinning(
    W: torch.Tensor,
    K: int = 8,
    weight_eps: float = 1e-12,
    sort_desc: bool = True,
    pad_index: int = -1,
    dtype_idx: torch.dtype = torch.int32,
    dtype_w: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense skinning weights (N, J) -> sparse top-K jointIndices/jointWeights.

    Args:
      W: (N, J) float tensor of skinning weights per vertex.
      K: number of influences per vertex to keep.
      weight_eps: prune tiny weights; also avoids div-by-zero in normalization.
      sort_desc: sort the chosen K influences by descending weight (stable output).
      pad_index: index used when fewer than K nonzero weights exist for a vertex.
      dtype_idx: dtype for the joint-index output.
      dtype_w: dtype for the joint-weight output.

    Returns:
      idx_mat: (N, K) int32
      w_mat:   (N, K) float32
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError("W must be a torch.Tensor.")
    N, J = W.shape
    K_eff = min(K, J)

    W_masked = torch.where(W > weight_eps, W, torch.zeros_like(W))

    w_topk, idx_topk = torch.topk(W_masked, K_eff, dim=1, largest=True, sorted=sort_desc)

    if K_eff < K:
        pad_cols = K - K_eff
        idx_pad = torch.full((N, pad_cols), pad_index, device=W.device, dtype=idx_topk.dtype)
        w_pad = torch.zeros((N, pad_cols), device=W.device, dtype=w_topk.dtype)
        idx_mat = torch.cat([idx_topk, idx_pad], dim=1)
        w_mat = torch.cat([w_topk, w_pad], dim=1)
    else:
        idx_mat, w_mat = idx_topk, w_topk

    s = w_mat.sum(dim=1, keepdim=True)
    nonzero = s > 0
    w_mat = torch.where(nonzero, w_mat / torch.clamp(s, min=1e-20), torch.zeros_like(w_mat))

    idx_mat = idx_mat.to(dtype_idx)
    w_mat = w_mat.to(dtype_w)

    return idx_mat, w_mat


class BatchedSkinning:
    """Cache-friendly FK + LBS wrapper for repeated posing against a fixed rig topology."""

    def __init__(
        self,
        joint_parent_ids: Sequence[int],
        skinning_weights: torch.Tensor,
        bind_world_transforms: torch.Tensor,
        bind_shapes: torch.Tensor,
        joint_orient: torch.Tensor | None = None,
        mode: str = "warp",
        global_translation_joint_idx: int | None = None,
        root_joint_idx: int | None = None,
        *,
        source_fk: FKTopology | None = None,
    ) -> None:
        """Initialize a BatchedSkinning instance for posing meshes using LBS.

        Args:
            joint_parent_ids: ``(J,)`` int array of joint parent indices across all characters.
            skinning_weights: ``(V, J)`` array of skinning weights across all characters.
            bind_world_transforms: ``(B, J, 4, 4)`` array of joint bind poses in world space.
            bind_shapes: ``(B, V, 3)`` array of vertex positions in bind pose.
            joint_orient: ``None`` or ``(J, M, M)`` where ``M > 3``, specifying the
                initial world-space orientation of each joint. Poses are relative to
                this orientation if given.
            mode: ``"warp"`` for sparse Warp-based LBS (default) or ``"dense"`` for dense LBS.
            global_translation_joint_idx: Index of the joint that receives
                ``global_translation``. Default 1 for the full-body model where
                joint 0 is a dummy ``Root`` and joint 1 is ``Hips``; use 0 for
                models whose global translation is applied to joint 0.
            root_joint_idx: Deprecated alias for ``global_translation_joint_idx``.
            source_fk: Optional FK-only source topology. When provided,
                ``forward_source_kinematics`` runs FK over this topology while
                LBS remains on the target topology.
        """
        if global_translation_joint_idx is None:
            global_translation_joint_idx = root_joint_idx if root_joint_idx is not None else 1
        self.dtype, self.device = require_torch_tensors(
            skinning_weights, bind_world_transforms, bind_shapes, name="BatchedSkinning inputs"
        )
        self.mode = mode

        batched = bind_shapes.ndim == 3
        num_joints = len(joint_parent_ids)
        if num_joints != bind_world_transforms.shape[1 if batched else 0]:
            raise ValueError(
                "joint_parent_ids and bind_world_transforms must have the same number of joints."
            )
        if batched and bind_world_transforms.shape[0] != bind_shapes.shape[0]:
            raise ValueError("bind_world_transforms and bind_shapes must have the same batch size.")
        self.bind_batched = batched
        self.num_joints = len(joint_parent_ids)
        self.global_translation_joint_idx = global_translation_joint_idx
        self.joint_parent_ids = (
            joint_parent_ids
            if isinstance(joint_parent_ids, list)
            else joint_parent_ids.tolist()
            if hasattr(joint_parent_ids, "tolist")
            else list(joint_parent_ids)
        )
        self.skinning_weights = skinning_weights
        self.bind_world_transforms = bind_world_transforms
        bind_local_transforms, self.inverse_bind_transform = joint_world_to_local(
            bind_world_transforms, joint_parent_ids, return_inverse=True
        )
        self.local_rotations = bind_local_transforms[..., :3, :3]
        self.local_translations = bind_local_transforms[..., :3, 3]
        self.bind_shapes = bind_shapes
        self.joint_orient = None
        self._orient_parent_T = None
        if joint_orient is not None:
            if num_joints != joint_orient.shape[0]:
                raise ValueError(
                    "joint_orient must have the same number of joints as joint_parent_ids."
                )
            jo = joint_orient.to(dtype=self.dtype, device=self.device)
            self.joint_orient, self._orient_parent_T = precompute_joint_orient(
                jo, self.joint_parent_ids
            )

        self._levels = compute_skeleton_levels(self.joint_parent_ids, device=self.device)

        self._bone_weights = None
        self._bone_indices = None
        if self.mode == "warp":
            self._prepare_warp_data()
        self.source_joint_parent_ids = None
        self.source_target_joint_indices = None
        self.source_joint_indices = None
        self.source_num_joints = 0
        self.source_global_translation_joint_idx = 0
        self.source_bind_world_transforms = None
        self.source_bind_batched = False
        self.source_local_rotations = None
        self.source_local_translations = None
        self.source_joint_orient = None
        self._source_orient_parent_T = None
        self._source_levels = None
        self._configure_source_fk(source_fk)

    @staticmethod
    def _as_parent_list(joint_parent_ids: Sequence[int]) -> list[int]:
        return (
            joint_parent_ids
            if isinstance(joint_parent_ids, list)
            else joint_parent_ids.tolist()
            if hasattr(joint_parent_ids, "tolist")
            else list(joint_parent_ids)
        )

    @staticmethod
    def _bind_joint_count(bind_world_transforms: torch.Tensor) -> int:
        return (
            bind_world_transforms.shape[1]
            if bind_world_transforms.ndim == 4
            else (bind_world_transforms.shape[0])
        )

    @staticmethod
    def _is_batched_bind(bind_world_transforms: torch.Tensor) -> bool:
        return bind_world_transforms.ndim == 4

    def _target_bind_subset(self, bind_world_transforms: torch.Tensor) -> torch.Tensor:
        if self.source_target_joint_indices is None:
            raise RuntimeError(
                "source_fk.target_joint_indices are required to derive source bind transforms"
            )
        return bind_world_transforms[..., self.source_target_joint_indices, :, :]

    def _set_source_bind_world_transforms(
        self,
        source_bind_world_transforms: torch.Tensor,
    ) -> None:
        if self.source_joint_parent_ids is None:
            raise RuntimeError("Cannot set source bind transforms without source topology")
        if self._bind_joint_count(source_bind_world_transforms) != self.source_num_joints:
            raise ValueError(
                "source_bind_world_transforms must have the same number of joints as "
                "source_joint_parent_ids."
            )
        source_bind_world_transforms = source_bind_world_transforms.to(
            dtype=self.dtype,
            device=self.device,
        )
        self.source_bind_world_transforms = source_bind_world_transforms
        self.source_bind_batched = self._is_batched_bind(source_bind_world_transforms)
        source_local_transforms = joint_world_to_local(
            source_bind_world_transforms,
            self.source_joint_parent_ids,
        )
        self.source_local_rotations = source_local_transforms[..., :3, :3]
        self.source_local_translations = source_local_transforms[..., :3, 3]

    def _configure_source_fk(self, source_fk: FKTopology | None) -> None:
        if source_fk is None:
            return
        if source_fk.parent_ids is None:
            raise ValueError("source_fk.parent_ids are required for source FK config.")
        if source_fk.target_joint_indices is None and source_fk.bind_world_transforms is None:
            raise ValueError(
                "source FK config requires source_fk.target_joint_indices or "
                "source_fk.bind_world_transforms."
            )

        self.source_joint_parent_ids = self._as_parent_list(source_fk.parent_ids)
        self.source_num_joints = len(self.source_joint_parent_ids)
        if source_fk.target_joint_indices is not None:
            self.source_target_joint_indices = torch.as_tensor(
                source_fk.target_joint_indices,
                dtype=torch.long,
                device=self.device,
            )
            # Backward-compatible alias for older callers/tests.
            self.source_joint_indices = self.source_target_joint_indices
            if self.source_target_joint_indices.numel() != self.source_num_joints:
                raise ValueError(
                    "source_fk.target_joint_indices must have the same length as "
                    "source_fk.parent_ids."
                )
        source_global_translation_joint_idx = source_fk.global_translation_joint_idx
        if source_global_translation_joint_idx is None:
            source_global_translation_joint_idx = (
                self.global_translation_joint_idx
                if self.global_translation_joint_idx < self.source_num_joints
                else 0
            )
        self.source_global_translation_joint_idx = source_global_translation_joint_idx

        source_bind_world_transforms = source_fk.bind_world_transforms
        if source_bind_world_transforms is None:
            source_bind_world_transforms = self._target_bind_subset(self.bind_world_transforms)
        self._set_source_bind_world_transforms(source_bind_world_transforms)

        if source_fk.joint_orient is not None:
            if self.source_num_joints != source_fk.joint_orient.shape[0]:
                raise ValueError(
                    "source_fk.joint_orient must have the same number of joints as "
                    "source_fk.parent_ids."
                )
            jo = source_fk.joint_orient.to(dtype=self.dtype, device=self.device)
            self.source_joint_orient, self._source_orient_parent_T = precompute_joint_orient(
                jo,
                self.source_joint_parent_ids,
            )
        self._source_levels = compute_skeleton_levels(
            self.source_joint_parent_ids,
            device=self.device,
        )

    def rebind(self, bind_world_transforms: torch.Tensor, bind_shapes: torch.Tensor) -> None:
        """Rebind the skeleton to new bind poses and shapes
        Args:
            bind_world_transforms: (B, J, 4, 4) array of new joint bind poses in world space for multiple characters
            bind_shapes: (B, V, 3) array of new vertex positions in bind pose for multiple characters
        """
        batched = bind_shapes.ndim == 3
        self.bind_batched = batched
        self.bind_world_transforms = bind_world_transforms
        bind_local_transforms, self.inverse_bind_transform = joint_world_to_local(
            bind_world_transforms, self.joint_parent_ids, return_inverse=True
        )
        self.local_rotations = bind_local_transforms[..., :3, :3]
        self.local_translations = bind_local_transforms[..., :3, 3]
        self.bind_shapes = bind_shapes
        if (
            self.source_joint_parent_ids is not None
            and self.source_target_joint_indices is not None
        ):
            self._set_source_bind_world_transforms(self._target_bind_subset(bind_world_transforms))

    def _prepare_warp_data(self):
        """Prepare sparse bone weights and indices for warp-based LBS."""
        bone_indices, bone_weights = topk_skinning(self.skinning_weights)
        self._bone_indices = bone_indices.to(device=self.device)
        self._bone_weights = bone_weights.to(dtype=self.dtype, device=self.device)

    def get_bone_weights(self) -> torch.Tensor:
        """Get bone weights. For warp mode, returns sparse weights (V, K). For dense mode, returns full weights (V, J)."""
        if self.mode == "warp":
            if self._bone_weights is None:
                self._prepare_warp_data()
            return self._bone_weights
        else:
            return self.skinning_weights

    def get_bone_indices(self) -> torch.Tensor | None:
        """Get bone indices. For warp mode, returns sparse indices (V, K). For dense mode, returns None."""
        if self.mode == "warp":
            if self._bone_indices is None:
                self._prepare_warp_data()
            return self._bone_indices
        else:
            return None

    def forward_kinematics(
        self,
        local_rotations: torch.Tensor,
        global_translation: torch.Tensor | None = None,
        align_translation: torch.Tensor | None = None,
        absolute_pose: bool = False,
        *,
        hips_translations: torch.Tensor | None = None,
        local_translations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run batched forward kinematics and return per-joint world transforms.

        Supports:
        - many characters x one pose
        - one character x many poses
        - N characters x N poses (i-th character with i-th pose)
        - one character x one pose

        Args:
            local_rotations: (J,3,3) or (B,J,3,3)
            global_translation: (3,) or (B,3) -- translation applied to the joint
                specified by global_translation_joint_idx.
            align_translation: None or (3,)
            absolute_pose: bool, whether local_rotations are absolute rotations (True) or relative to joint orient (False, default)
            hips_translations: deprecated alias for global_translation.
            local_translations: (J,3) or (B,J,3) -- per-call override for
                self.local_translations.  When provided, replaces the bind-pose
                local translations for this call only.  Useful for bone scaling
                or custom offsets.  Default None uses self.local_translations.

        Returns:
            world_transforms: (B, J, 4, 4)
        """
        return self._forward_kinematics_impl(
            local_rotations=local_rotations,
            global_translation=global_translation,
            align_translation=align_translation,
            absolute_pose=absolute_pose,
            hips_translations=hips_translations,
            local_translations=local_translations,
            num_joints=self.num_joints,
            bind_world_transforms=self.bind_world_transforms,
            bind_batched=self.bind_batched,
            default_local_translations=self.local_translations,
            global_translation_joint_idx=self.global_translation_joint_idx,
            levels=self._levels,
            joint_orient=self.joint_orient,
            orient_parent_T=self._orient_parent_T,
        )

    def forward_source_kinematics(
        self,
        local_rotations: torch.Tensor,
        global_translation: torch.Tensor | None = None,
        align_translation: torch.Tensor | None = None,
        absolute_pose: bool = False,
        *,
        hips_translations: torch.Tensor | None = None,
        local_translations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run FK over the optional source/public topology."""
        if self.source_joint_parent_ids is None:
            raise RuntimeError("forward_source_kinematics requires source FK config.")
        return self._forward_kinematics_impl(
            local_rotations=local_rotations,
            global_translation=global_translation,
            align_translation=align_translation,
            absolute_pose=absolute_pose,
            hips_translations=hips_translations,
            local_translations=local_translations,
            num_joints=self.source_num_joints,
            bind_world_transforms=self.source_bind_world_transforms,
            bind_batched=self.source_bind_batched,
            default_local_translations=self.source_local_translations,
            global_translation_joint_idx=self.source_global_translation_joint_idx,
            levels=self._source_levels,
            joint_orient=self.source_joint_orient,
            orient_parent_T=self._source_orient_parent_T,
        )

    def expand_source_world_transforms(
        self,
        source_rotations: torch.Tensor,
        source_world_transforms: torch.Tensor,
        transform_expander: Callable[..., torch.Tensor],
        *,
        target_local_translations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand source FK output to target world transforms with a caller-supplied hook."""
        if self.source_joint_parent_ids is None:
            raise RuntimeError("expand_source_world_transforms requires source FK config.")
        if target_local_translations is None:
            target_local_translations = self.local_translations
        return transform_expander(
            source_rotations=source_rotations,
            source_world_transforms=source_world_transforms,
            target_local_rotations=self.local_rotations,
            target_local_translations=target_local_translations,
            target_joint_count=self.num_joints,
        )

    def _forward_kinematics_impl(
        self,
        *,
        local_rotations: torch.Tensor,
        global_translation: torch.Tensor | None,
        align_translation: torch.Tensor | None,
        absolute_pose: bool,
        hips_translations: torch.Tensor | None,
        local_translations: torch.Tensor | None,
        num_joints: int,
        bind_world_transforms: torch.Tensor,
        bind_batched: bool,
        default_local_translations: torch.Tensor,
        global_translation_joint_idx: int,
        levels: torch.Tensor,
        joint_orient: torch.Tensor | None,
        orient_parent_T: torch.Tensor | None,
    ) -> torch.Tensor:
        if global_translation is None and hips_translations is not None:
            global_translation = hips_translations
        if local_rotations.shape[-3:] != (num_joints, 3, 3):
            raise ValueError(
                f"Expected local_rotations to have shape (...,{num_joints},3,3); "
                f"got {local_rotations.shape}"
            )

        if local_rotations.ndim == 3:
            local_rotations = local_rotations[None, :, :, :]

        rot_batch = local_rotations.shape[0]
        bind_batch = bind_world_transforms.shape[0] if bind_batched else 1

        if global_translation is None:
            global_translation = torch.zeros(
                rot_batch,
                3,
                dtype=local_rotations.dtype,
                device=local_rotations.device,
            )
        if global_translation.shape not in [(3,), (rot_batch, 3)]:
            raise ValueError(
                f"Expected global_translation to have shape (3,) or ({rot_batch},3); got {global_translation.shape}"
            )

        if rot_batch == 1 and bind_batch > 1:
            batch_size = bind_batch
            local_rotations = local_rotations.to(dtype=self.dtype, device=self.device).expand(
                batch_size, num_joints, 3, 3
            )
        elif rot_batch >= 1 and bind_batch == 1:
            batch_size = rot_batch
            local_rotations = local_rotations.to(dtype=self.dtype, device=self.device)
        elif rot_batch == bind_batch:
            batch_size = rot_batch
            local_rotations = local_rotations.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError(
                f"Incompatible batches: rotations={rot_batch}, bind={bind_batch}. "
                "Provide (1xB), (Bx1), (1x1), or (BxB) with equal B."
            )

        if align_translation is not None:
            align_translation = align_translation.to(dtype=self.dtype, device=self.device)

        if local_translations is not None:
            local_t = local_translations.to(dtype=self.dtype, device=self.device)
        else:
            local_t = default_local_translations.to(dtype=self.dtype, device=self.device)
        if bind_batched:
            if rot_batch >= 1 and bind_batch == 1:
                local_t = local_t.expand(batch_size, num_joints, 3)
        else:
            local_t = local_t.unsqueeze(0).expand(batch_size, num_joints, 3)

        j_mask = one_hot_1d(
            num_joints,
            global_translation_joint_idx,
            dtype=self.dtype,
            device=self.device,
        )[None, :, None]
        if align_translation is not None:
            comp_m = mask_1d(3, [0, 2], dtype=self.dtype, device=self.device)[None, None, :]
            M = j_mask * comp_m
            local_t = local_t * (1 - M) + align_translation[None, None, :] * M
        else:
            comp_m = mask_1d(3, [0, 1, 2], dtype=self.dtype, device=self.device)[None, None, :]
            M = j_mask * comp_m
            gt = global_translation.to(dtype=self.dtype, device=self.device)
            if gt.ndim == 1:
                gt = gt[None, :]
            local_t = local_t * (1 - j_mask) + gt[:, None, :] * j_mask

        if joint_orient is not None and not absolute_pose:
            local_rotations = apply_joint_orient_local(
                local_rotations,
                joint_orient,
                orient_parent_T,
            )
        T_local = SE3_from_Rt(local_rotations, local_t)

        T_world = joint_local_to_world_levelorder(T_local, levels)

        if align_translation is not None:
            y_world = T_world[..., 1, 3]
            y_offset = y_world.min(dim=1, keepdim=True).values
            shift = y_offset + align_translation[1]

            new_y_world = y_world - shift
            delta_w = (new_y_world - y_world)[..., None, None]
            E13 = one_hot_2d(4, 4, 1, 3, dtype=self.dtype, device=self.device)[None, None, ...]
            T_world = T_world + delta_w * E13

            y_local = T_local[..., 1, 3]
            new_y_local = y_local - shift
            delta_l = (new_y_local - y_local)[..., None, None]
            j_only = one_hot_1d(num_joints, 1, dtype=self.dtype, device=self.device)[
                None, :, None, None
            ]
            T_local = T_local + delta_l * E13 * j_only

        return T_world

    def linear_blend_skinning(
        self,
        world_transforms: torch.Tensor,
        *,
        bind_shapes: torch.Tensor | None = None,
        inverse_bind_transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply linear blend skinning from already-computed world transforms.

        Args:
            world_transforms: (J,4,4) or (B,J,4,4) FK world transforms.
            bind_shapes: optional override for bind-pose vertices. Defaults to
                the shapes most recently supplied to ``__init__`` or ``rebind``.
            inverse_bind_transform: optional override for inverse bind
                transforms. Defaults to the transforms most recently supplied to
                ``__init__`` or ``rebind``.

        Returns:
            posed_shapes: (B, V, 3)
        """
        if world_transforms.shape[-3:] != (self.num_joints, 4, 4):
            raise ValueError(
                f"Expected world_transforms to have shape (...,{self.num_joints},4,4); "
                f"got {world_transforms.shape}"
            )
        if world_transforms.ndim == 3:
            world_transforms = world_transforms[None, :, :, :]

        world_transforms = world_transforms.to(dtype=self.dtype, device=self.device)
        batch_size = world_transforms.shape[0]

        if bind_shapes is None:
            bind_shapes = self.bind_shapes
        bind_shapes = bind_shapes.to(dtype=self.dtype, device=self.device)

        if inverse_bind_transform is None:
            inverse_bind_transform = self.inverse_bind_transform
        inverse_bind_transform = inverse_bind_transform.to(dtype=self.dtype, device=self.device)

        bind_batch = bind_shapes.shape[0] if bind_shapes.ndim == 3 else 1
        if bind_batch > 1 and batch_size == 1:
            batch_size = bind_batch
            world_transforms = world_transforms.expand(batch_size, self.num_joints, 4, 4)
        elif bind_batch not in (1, batch_size):
            raise ValueError(
                f"Incompatible batches: world_transforms={batch_size}, bind_shapes={bind_batch}."
            )

        if bind_shapes.ndim == 2:
            bind_shapes = bind_shapes.unsqueeze(0).expand(batch_size, -1, -1)
        elif bind_shapes.shape[0] == 1 and batch_size > 1:
            bind_shapes = bind_shapes.expand(batch_size, -1, -1)

        inverse_bind_batch = (
            inverse_bind_transform.shape[0] if inverse_bind_transform.ndim == 4 else 1
        )
        if inverse_bind_batch > 1 and batch_size == 1:
            batch_size = inverse_bind_batch
            world_transforms = world_transforms.expand(batch_size, self.num_joints, 4, 4)
            if bind_shapes.shape[0] == 1:
                bind_shapes = bind_shapes.expand(batch_size, -1, -1)
        elif inverse_bind_batch not in (1, batch_size):
            raise ValueError(
                "Incompatible batches: "
                f"world_transforms={batch_size}, inverse_bind_transform={inverse_bind_batch}."
            )

        if inverse_bind_transform.ndim == 3:
            inverse_bind_transform = inverse_bind_transform.unsqueeze(0).expand(
                batch_size, self.num_joints, 4, 4
            )
        elif inverse_bind_transform.shape[0] == 1 and batch_size > 1:
            inverse_bind_transform = inverse_bind_transform.expand(
                batch_size, self.num_joints, 4, 4
            )

        bone_transforms = world_transforms @ inverse_bind_transform
        if self.mode == "warp":
            return self._warp_skinning(bind_shapes, bone_transforms)
        return lbs(
            bind_shapes,
            self.skinning_weights,
            bone_transforms,
        )

    def pose(
        self,
        local_rotations: torch.Tensor,
        global_translation: torch.Tensor | None = None,
        align_translation: torch.Tensor | None = None,
        return_transforms: bool = False,
        absolute_pose: bool = False,
        fk_only: bool = False,
        *,
        hips_translations: torch.Tensor | None = None,
        local_translations: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Pose the meshes using Linear Blend Skinning (LBS), autograd-safe (no in-place).

        Supports:
        - many characters x one pose
        - one character x many poses
        - N characters x N poses (i-th character with i-th pose)
        - one character x one pose

        Args:
            local_rotations: (J,3,3) or (B,J,3,3)
            global_translation: (3,) or (B,3) -- translation applied to the joint
                specified by global_translation_joint_idx.
            align_translation: None or (3,)
            return_transforms: bool, whether to return world transforms
            absolute_pose: bool, whether local_rotations are absolute rotations (True) or relative to joint orient (False, default)
            fk_only: if True, run forward kinematics only and skip LBS. Returns
                the per-joint world transforms. When True, the returned tensor is always (B, J, 4, 4);
                ``return_transforms`` is ignored.
            hips_translations: deprecated alias for global_translation.
            local_translations: (J,3) or (B,J,3) -- per-call override for
                self.local_translations.  When provided, replaces the bind-pose
                local translations for this call only.  Useful for bone scaling
                or custom offsets.  Default None uses self.local_translations.
        Returns:
            posed_shapes: (..., V, 3)
            (optional) world_transforms: (..., J, 4, 4)
            (fk_only=True): world_transforms only, (B, J, 4, 4)
        """
        T_world = self.forward_kinematics(
            local_rotations=local_rotations,
            global_translation=global_translation,
            align_translation=align_translation,
            absolute_pose=absolute_pose,
            hips_translations=hips_translations,
            local_translations=local_translations,
        )
        if fk_only:
            return T_world

        posed_shapes = self.linear_blend_skinning(T_world)
        if return_transforms:
            return posed_shapes, T_world
        return posed_shapes

    def _warp_skinning(
        self, bind_verts: torch.Tensor, bone_transforms: torch.Tensor
    ) -> torch.Tensor:
        """Warp-based skinning."""
        from .lbs_warp import linear_blend_skinning

        return linear_blend_skinning(
            bind_verts, self._bone_weights, self._bone_indices, bone_transforms
        )
