# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw SOMA and G1 skeletons in the Newton viewer, with each bone as two pyramids of lines."""

from abc import ABC, abstractmethod

import newton
import warp as wp

from motionbricks.animation.skeleton import Skeleton
from motionbricks.renderers.line_renderer import LineRenderer
from motionbricks.utils import pose_utils

_PYRAMID_VERTICES = (
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, 1.0, 1.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(-1.0, 1.0, 1.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(-1.0, -1.0, 1.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, -1.0, 1.0),
    wp.vec3(1.0, 1.0, 1.0),
    wp.vec3(1.0, -1.0, 1.0),
    wp.vec3(1.0, -1.0, 1.0),
    wp.vec3(-1.0, -1.0, 1.0),
    wp.vec3(-1.0, -1.0, 1.0),
    wp.vec3(-1.0, 1.0, 1.0),
    wp.vec3(-1.0, 1.0, 1.0),
    wp.vec3(1.0, 1.0, 1.0),
)

# Each pair of vertices is one line.
_PYRAMID_LINE_COUNT = wp.constant(len(_PYRAMID_VERTICES) // 2)
# A bone is two pyramids that share a square base. The base sits 80 percent of the way
# from the child joint to its parent, and its half width is a twentieth of the bone length.
_BONE_LINE_COUNT = wp.constant(2 * _PYRAMID_LINE_COUNT)
_CHILD_PYRAMID_HEIGHT = wp.constant(0.8)
_PARENT_PYRAMID_HEIGHT = wp.constant(1.0 - 0.8)
_BONE_LENGTH_PER_HALF_WIDTH = wp.constant(20.0)


@wp.func
def _compute_pyramid_lines(
    in_pyramid_vertices: wp.array(dtype=wp.vec3),
    in_tx: wp.transform,
    in_scale: wp.vec3,
    in_offset: wp.int32,
    out_line_starts: wp.array(dtype=wp.vec3),
    out_line_ends: wp.array(dtype=wp.vec3),
):
    for i in range(_PYRAMID_LINE_COUNT):
        idx = in_offset + i
        out_line_starts[idx] = wp.transform_point(
            in_tx, wp.cw_mul(in_pyramid_vertices[i * 2], in_scale)
        )
        out_line_ends[idx] = wp.transform_point(
            in_tx, wp.cw_mul(in_pyramid_vertices[1 + i * 2], in_scale)
        )


@wp.func
def _compute_bone_lines(
    in_pyramid_vertices: wp.array(dtype=wp.vec3),
    in_offset: wp.int32,
    in_joint_t: wp.vec3,
    in_parent_t: wp.vec3,
    out_line_starts: wp.array(dtype=wp.vec3),
    out_line_ends: wp.array(dtype=wp.vec3),
):
    diff = wp.sub(in_parent_t, in_joint_t)
    q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.normalize(diff))

    length = wp.length(diff)
    width = length / _BONE_LENGTH_PER_HALF_WIDTH

    _compute_pyramid_lines(
        in_pyramid_vertices,
        wp.transform(in_joint_t, q),
        wp.vec3(width, width, length * _CHILD_PYRAMID_HEIGHT),
        in_offset,
        out_line_starts,
        out_line_ends,
    )

    _compute_pyramid_lines(
        in_pyramid_vertices,
        wp.transform(in_parent_t, q),
        wp.vec3(width, width, -length * _PARENT_PYRAMID_HEIGHT),
        in_offset + _PYRAMID_LINE_COUNT,
        out_line_starts,
        out_line_ends,
    )


@wp.kernel
def _update_bone_lines_batch_kernel(
    in_pyramid_vertices: wp.array(dtype=wp.vec3),
    in_world_transforms: wp.array2d(dtype=wp.transform),
    in_bone_indices: wp.array(dtype=wp.vec2i),
    in_colors: wp.array(dtype=wp.vec3),
    out_line_starts: wp.array(dtype=wp.vec3),
    out_line_ends: wp.array(dtype=wp.vec3),
    out_line_colors: wp.array(dtype=wp.vec3),
):
    pose, bone_index = wp.tid()
    bone = in_bone_indices[bone_index]
    bone_count = in_bone_indices.shape[0]
    offset = (pose * bone_count + bone_index) * _BONE_LINE_COUNT

    _compute_bone_lines(
        in_pyramid_vertices,
        offset,
        in_world_transforms[pose, bone.y].p,
        in_world_transforms[pose, bone.x].p,
        out_line_starts,
        out_line_ends,
    )

    for line in range(_BONE_LINE_COUNT):
        out_line_colors[offset + line] = in_colors[pose]


class BaseSkeletonRenderer(LineRenderer, ABC):
    """Draw batches of poses as bones between their world transforms.

    A subclass names its bones as (parent, child) index pairs into each pose's world
    transforms. It checks the poses it draws, and writes their world transforms into
    world_poses.
    """

    def __init__(self, bones, transforms_per_pose: int, max_instances: int, device=None) -> None:
        """Upload the bones and allocate the line buffers.

        Args:
            bones: (parent, child) index pairs into each pose's world transforms.
            transforms_per_pose: Number of world transforms in each pose.
            max_instances: Number of poses the buffers hold at first. They grow when a
                batch needs more.
            device: Device of the buffers. None uses the current device.
        """
        self._transforms_per_pose = transforms_per_pose
        self.pyramid_vertices = wp.array(_PYRAMID_VERTICES, dtype=wp.vec3, device=device)
        self.bones = wp.array(list(bones), dtype=wp.vec2i, device=device)
        super().__init__(len(self.bones) * _BONE_LINE_COUNT, max_instances, device=device)

    def draw_batch(
        self,
        viewer,
        poses: wp.array,
        num_poses: int,
        xform: wp.transform,
        colors: wp.array,
        instance_id: int | str,
    ) -> None:
        """Draw the first num_poses poses as one batch.

        Args:
            viewer: Newton viewer to draw into.
            poses: Poses laid out as [pose, value], in the renderer's pose layout.
            num_poses: Number of poses to draw. Zero hides the batch.
            xform: World transform applied to every pose.
            colors: One vec3 color per pose.
            instance_id: Identifier of the drawn batch. Drawing the same identifier again
                updates it.
        """
        self._check_poses(poses)
        if not 0 <= num_poses <= poses.shape[0]:
            raise ValueError(f"num_poses must be in [0, {poses.shape[0]}].")
        if colors.dtype != wp.vec3 or colors.ndim != 1 or len(colors) < num_poses:
            raise ValueError("colors must provide one wp.vec3 per active pose.")
        if num_poses == 0:
            self.hide(viewer, instance_id)
            return
        self._ensure_capacity(num_poses)
        self._compute_world_poses(poses, num_poses, xform)
        wp.launch(
            _update_bone_lines_batch_kernel,
            dim=(num_poses, len(self.bones)),
            inputs=[self.pyramid_vertices, self.world_poses, self.bones, colors],
            outputs=[self.line_starts, self.line_ends, self.line_colors],
            device=self._device,
        )
        self._log_batch(viewer, instance_id, num_poses)

    @abstractmethod
    def _check_poses(self, poses: wp.array) -> None:
        """Raise ValueError unless poses holds [pose, value] poses in the renderer's layout."""

    @abstractmethod
    def _compute_world_poses(self, poses: wp.array, num_poses: int, xform: wp.transform) -> None:
        """Write the world transforms of the first num_poses poses, placed by xform, into world_poses."""

    def _allocate_buffers(self, capacity: int) -> None:
        super()._allocate_buffers(capacity)
        self.world_poses = wp.empty(
            (capacity, self._transforms_per_pose), dtype=wp.transform, device=self._device,
        )


class SkeletonRenderer(BaseSkeletonRenderer):
    """Draw poses of local joint transforms along a skeleton's joint hierarchy."""

    _NAME_PREFIX = "/skeleton_"

    def __init__(self, skeleton: Skeleton, max_instances: int = 1):
        """Collect the skeleton's bones and allocate the line buffers on its device.

        Args:
            skeleton: Skeleton to draw.
            max_instances: Number of poses the buffers hold at first. They grow when a
                batch needs more.
        """
        self.skeleton = skeleton
        super().__init__(
            [(skeleton.joint_parent(joint), joint) for joint in range(1, skeleton.num_joints)],
            skeleton.num_joints,
            max_instances,
            device=skeleton.device,
        )

    def _check_poses(self, poses: wp.array) -> None:
        if poses.dtype != wp.transform or poses.ndim != 2:
            raise ValueError("poses must be a 2D Warp array of wp.transform.")
        if poses.shape[1] != self.skeleton.num_joints:
            raise ValueError(f"Expected {self.skeleton.num_joints} joints, got {poses.shape[1]}.")

    def _compute_world_poses(self, poses: wp.array, num_poses: int, xform: wp.transform) -> None:
        # compute_world_poses() launches on the current device.
        with wp.ScopedDevice(self.skeleton.device):
            pose_utils.compute_world_poses(
                poses, self.skeleton.parent_indices, self.world_poses, xform, num_poses=num_poses,
            )


@wp.kernel
def _place_bodies_kernel(
    xform: wp.transform,
    body_q: wp.array(dtype=wp.transform),
    placed_body_q: wp.array(dtype=wp.transform),
):
    body = wp.tid()
    placed_body_q[body] = wp.mul(xform, body_q[body])


class DOFSkeletonRenderer(BaseSkeletonRenderer):
    """Draw poses of joint coordinates along the Newton model's joint hierarchy.

    Each pose holds all of the Newton model's joint_q coordinates. Forward kinematics
    uses a separate state, so drawing does not affect playback. Bones connect body
    origins along the joint hierarchy.
    """

    _NAME_PREFIX = "/dof_skeleton_"

    def __init__(self, model: newton.Model, max_instances: int = 1) -> None:
        """Collect the model's bones and allocate the line buffers on its device.

        Args:
            model: Newton model whose joint hierarchy to draw.
            max_instances: Number of poses the buffers hold at first. They grow when a
                batch needs more.
        """
        self.model = model
        self.state = model.state()
        super().__init__(
            [(parent, child) for parent, child in zip(model.joint_parent.numpy(), model.joint_child.numpy())
             if parent >= 0 and child >= 0],
            model.body_count,
            max_instances,
            device=model.device,
        )

    def _check_poses(self, poses: wp.array) -> None:
        if poses.dtype != wp.float32 or poses.ndim != 2:
            raise ValueError("poses must be a 2D Warp array of float32.")
        if poses.shape[1] != self.model.joint_coord_count:
            raise ValueError(f"Expected {self.model.joint_coord_count} joint coordinates, got {poses.shape[1]}.")

    def _compute_world_poses(self, poses: wp.array, num_poses: int, xform: wp.transform) -> None:
        for frame in range(num_poses):
            newton.eval_fk(self.model, poses[frame], self.model.joint_qd, self.state)
            wp.launch(
                _place_bodies_kernel, dim=self.model.body_count,
                inputs=[xform, self.state.body_q], outputs=[self.world_poses[frame]],
                device=self.model.device,
            )


__all__ = ["BaseSkeletonRenderer", "DOFSkeletonRenderer", "SkeletonRenderer"]
