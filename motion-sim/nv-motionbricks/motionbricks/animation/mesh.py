# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triangle meshes, skinned meshes, and skeletal meshes."""

import numpy as np
import warp as wp

from motionbricks.animation.skeleton import Skeleton


class Mesh:
    """Triangle mesh with vertex positions and triangle indices."""

    def __init__(self, points: np.ndarray, indices: np.ndarray) -> None:
        """Create a mesh from host vertex positions and triangle indices.

        Args:
            points: Vertex positions with shape (N, 3).
            indices: Triangle indices as a flat array, three per triangle.
        """
        self._points = wp.array(points, dtype=wp.vec3)
        self._indices = wp.array(indices, dtype=wp.int32)

    @property
    def points(self) -> wp.array:
        """Vertex positions, one vec3 per vertex."""
        return self._points

    @property
    def indices(self) -> wp.array:
        """Triangle indices as a flat int32 array, three per triangle."""
        return self._indices

    @property
    def num_points(self) -> int:
        """Number of vertices in the mesh."""
        return self._points.size

    @property
    def num_indices(self) -> int:
        """Number of triangle indices in the mesh."""
        return self._indices.size


class SkinnedMesh(Mesh):
    """Triangle mesh for linear blend skinning, with K influences per vertex.

    Joint indices and weights are flat arrays ordered by vertex. For N vertices
    and K influences per vertex, both arrays contain N * K elements.
    """

    def __init__(
        self,
        points: np.ndarray,
        indices: np.ndarray,
        joint_indices: np.ndarray,
        joint_weights: np.ndarray,
    ) -> None:
        """Create a skinned mesh from host geometry and skinning data.

        Args:
            points: Vertex positions with shape (N, 3).
            indices: Triangle indices as a flat array, three per triangle.
            joint_indices: Joint index of every vertex influence.
            joint_weights: Weight of every vertex influence.
        """
        super().__init__(points, indices)
        self._joint_indices = wp.array(joint_indices, dtype=wp.int32)
        self._joint_weights = wp.array(joint_weights, dtype=wp.float32)
        self._num_influences = (
            self._joint_indices.size // self._points.size
            if self._points.size > 0
            else 0
        )

    @property
    def num_influences(self) -> int:
        """Number of joint influences stored for each vertex."""
        return self._num_influences

    @property
    def joint_indices(self) -> wp.array:
        """Joint index of every vertex influence, as a flat int32 array."""
        return self._joint_indices

    @property
    def joint_weights(self) -> wp.array:
        """Weight of every vertex influence, as a flat float32 array."""
        return self._joint_weights


class SkeletalMesh:
    """Skinned meshes driven by one skeleton, ready to render.

    Each inverse bind transform maps vertices from skeleton space into the
    joint's bind space. Skinning composes it with the joint's current world
    transform to move vertices into the current pose.

    Override poses and masks change how the mesh is drawn, such as relaxing
    the fingers. An override pose holds one local transform per joint. A mask
    holds one weight per joint from 0 to 1. Renderers pick one of each by name.
    """

    def __init__(
        self,
        skinned_meshes: list[SkinnedMesh],
        skeleton: Skeleton,
        inverse_bind_transforms: list[wp.transform],
        override_poses: dict[str, wp.array] | None = None,
        override_pose_masks: dict[str, wp.array] | None = None,
    ) -> None:
        """Create a skeletal mesh from mesh parts and the skeleton that drives them.

        Args:
            skinned_meshes: Mesh parts deformed by the skeleton.
            skeleton: Skeleton that drives every mesh part.
            inverse_bind_transforms: One inverse bind transform per skeleton joint.
            override_poses: Override poses by name. None means none.
            override_pose_masks: Override pose masks by name. None means none.
        """
        if len(inverse_bind_transforms) != skeleton.num_joints:
            raise ValueError(
                "inverse_bind_transforms must contain one transform per joint."
            )

        self._skinned_meshes = skinned_meshes
        self._skeleton = skeleton
        self._inverse_bind_transforms = wp.array(
            inverse_bind_transforms,
            dtype=wp.transform,
        )
        self._override_poses = dict(override_poses or {})
        self._override_pose_masks = dict(override_pose_masks or {})

    @property
    def skinned_meshes(self) -> list[SkinnedMesh]:
        """Skinned mesh parts driven by :attr:`skeleton`."""
        return self._skinned_meshes

    @property
    def skeleton(self) -> Skeleton:
        """Skeleton that drives the mesh parts."""
        return self._skeleton

    @property
    def inverse_bind_transforms(self) -> wp.array:
        """One inverse bind transform per joint, as a Warp array."""
        return self._inverse_bind_transforms

    @property
    def override_poses(self) -> dict[str, wp.array]:
        """Override poses by name. Each holds one local transform per joint."""
        return self._override_poses

    @property
    def override_pose_masks(self) -> dict[str, wp.array]:
        """Override pose masks by name. Each holds one weight per joint from 0 to 1."""
        return self._override_pose_masks
