# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw and pick skeletal meshes with linear blend skinning."""

import warp as wp

from motionbricks.animation.mesh import SkeletalMesh
from motionbricks.renderers.base_renderer import BaseRenderer
from motionbricks.utils.pose_utils import blend_transform


@wp.kernel
def _skin_mesh_kernel(
    points: wp.array(dtype=wp.vec3),
    joint_indices: wp.array(dtype=wp.int32),
    joint_weights: wp.array(dtype=wp.float32),
    num_influences: wp.int32,
    skinning_transforms: wp.array(dtype=wp.transform),
    output_points: wp.array(dtype=wp.vec3),
):
    point = wp.tid()
    skinned_point = wp.vec3(0.0, 0.0, 0.0)
    for influence in range(num_influences):
        index = point * num_influences + influence
        skinned_point += (
            wp.transform_point(
                skinning_transforms[joint_indices[index]],
                points[point],
            )
            * joint_weights[index]
        )
    output_points[point] = skinned_point


@wp.kernel
def _raycast_meshes_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    origin: wp.vec3,
    direction: wp.vec3,
    max_distance: float,
    distances: wp.array(dtype=float),
):
    index = wp.tid()
    hit = wp.mesh_query_ray(mesh_ids[index], origin, direction, max_distance)
    distances[index] = wp.inf
    if hit.result:
        distances[index] = hit.t


@wp.kernel
def _blend_override_pose_kernel(
    local_transforms: wp.array(dtype=wp.transform),
    override_pose: wp.array(dtype=wp.transform),
    override_pose_mask: wp.array(dtype=wp.float32),
    blended_transforms: wp.array(dtype=wp.transform),
):
    joint = wp.tid()
    weight = wp.clamp(override_pose_mask[joint], 0.0, 1.0)
    blended_transforms[joint] = blend_transform(local_transforms[joint], override_pose[joint], weight)


@wp.kernel
def _compute_skinning_transforms_kernel(
    world_transforms: wp.array(dtype=wp.transform),
    inverse_bind_transforms: wp.array(dtype=wp.transform),
    skinning_transforms: wp.array(dtype=wp.transform),
):
    # A skinning transform takes a vertex from the bind pose to the drawn pose.
    joint = wp.tid()
    skinning_transforms[joint] = wp.mul(world_transforms[joint], inverse_bind_transforms[joint])


class SkeletalMeshRenderer(BaseRenderer):
    """Draw a skeletal mesh with linear blend skinning.

    set_override_pose() selects one of the mesh's override poses with a mask.
    Each joint then blends toward the override pose by its mask weight. The
    poses passed in are never changed.
    """

    def __init__(self, skeletal_mesh: SkeletalMesh) -> None:
        """Allocate the skinning buffers for a skeletal mesh on its skeleton's device.

        Args:
            skeletal_mesh: Skeletal mesh to draw.
        """
        super().__init__()
        self.skeletal_mesh = skeletal_mesh
        self._device = skeletal_mesh.skeleton.device
        self.skinning_transforms = wp.empty(
            skeletal_mesh.skeleton.num_joints,
            dtype=wp.transform,
            device=self._device,
        )
        self._blended_pose = wp.empty_like(self.skinning_transforms)
        self._world_pose = wp.empty_like(self.skinning_transforms)
        self._override: tuple[wp.array, wp.array] | None = None
        self.skinned_points = [
            wp.empty(
                mesh.num_points,
                dtype=wp.vec3,
                device=self._device,
            )
            for mesh in skeletal_mesh.skinned_meshes
        ]
        self._pick_meshes = None

    @staticmethod
    def _name(instance_id: int | str, mesh_index: int) -> str:
        return f"/skeletal_mesh_{instance_id}_{mesh_index}"

    def set_override_pose(self, pose: str | None, mask: str | None = None) -> None:
        """Select a mesh override pose and mask by name.

        Drawing clamps mask weights to the range from 0 to 1.

        Args:
            pose: Name of an override pose on the mesh. None draws poses as given.
            mask: Name of a mask on the mesh, required with a pose. It holds one
                float32 weight per joint on the renderer's device.
        """
        if pose is None:
            self._override = None
            return
        if mask is None:
            raise ValueError("An override pose needs a mask.")
        pose_array = self.skeletal_mesh.override_poses[pose]
        mask_array = self.skeletal_mesh.override_pose_masks[mask]
        shape, device = self.skinning_transforms.shape, self.skinning_transforms.device
        if pose_array.dtype != wp.transform or pose_array.shape != shape or pose_array.device != device:
            raise ValueError(f"Override pose {pose} must hold one transform per joint on {device}.")
        if mask_array.dtype != wp.float32 or mask_array.shape != shape or mask_array.device != device:
            raise ValueError(f"Override pose mask {mask} must hold one float32 weight per joint on {device}.")
        self._override = (pose_array, mask_array)

    def draw_pose(
        self,
        viewer,
        local_transforms: wp.array,
        xform: wp.transform,
        color: wp.vec3,
        instance_id: int | str,
    ) -> None:
        """Skin and draw one pose of the mesh.

        Args:
            viewer: Newton viewer to draw into.
            local_transforms: Local joint transforms of the pose.
            xform: World transform applied to the pose.
            color: RGB color of the mesh.
            instance_id: Identifier of the drawn instance. Drawing the same identifier
                again updates it.
        """
        self._skin_pose(local_transforms, xform)
        for mesh_index, mesh in enumerate(self.skeletal_mesh.skinned_meshes):
            if mesh.num_points == 0:
                continue
            self._log_visible(
                viewer.log_mesh,
                self._name(instance_id, mesh_index),
                self.skinned_points[mesh_index],
                mesh.indices,
                color=color,
            )

    def raycast_pose(
        self,
        local_transforms: wp.array,
        xform: wp.transform,
        origin: wp.vec3,
        direction: wp.vec3,
        max_distance: float = 1.0e6,
    ) -> float:
        """Return the distance to the skinned mesh along a ray.

        Picking skins the pose on demand, so it adds no work to normal rendering.

        Args:
            local_transforms: Local joint transforms of the pose.
            xform: World transform applied to the pose.
            origin: Ray origin in world space.
            direction: Unit ray direction in world space.
            max_distance: Longest distance to test, in meters. A miss returns it.
        """
        self._skin_pose(local_transforms, xform)
        if self._pick_meshes is None:
            self._pick_meshes = [
                wp.Mesh(points=points, indices=mesh.indices)
                for points, mesh in zip(self.skinned_points, self.skeletal_mesh.skinned_meshes)
                if mesh.num_indices > 0
            ]
            self._pick_mesh_ids = wp.array(
                [mesh.id for mesh in self._pick_meshes], dtype=wp.uint64, device=self._device,
            )
            self._pick_distances = wp.empty(len(self._pick_meshes), dtype=float, device=self._device)
        if not self._pick_meshes:
            return max_distance
        for mesh in self._pick_meshes:
            mesh.refit()
        wp.launch(
            _raycast_meshes_kernel,
            dim=len(self._pick_meshes),
            inputs=[self._pick_mesh_ids, origin, direction, max_distance],
            outputs=[self._pick_distances],
            device=self._device,
        )
        return min(max_distance, float(self._pick_distances.numpy().min()))

    def _skin_pose(self, local_transforms: wp.array, xform: wp.transform) -> None:
        skeleton = self.skeletal_mesh.skeleton
        if (
            local_transforms.dtype != wp.transform
            or local_transforms.ndim != 1
            or local_transforms.shape[0] != skeleton.num_joints
        ):
            raise ValueError("local_transforms must contain one transform per joint.")
        with wp.ScopedDevice(self._device):
            if self._override is not None:
                override_pose, override_pose_mask = self._override
                wp.launch(
                    _blend_override_pose_kernel,
                    dim=skeleton.num_joints,
                    inputs=[local_transforms, override_pose, override_pose_mask],
                    outputs=[self._blended_pose],
                )
                local_transforms = self._blended_pose
            skeleton.compute_world_transforms(local_transforms, self._world_pose, root_tx=xform)
            wp.launch(
                _compute_skinning_transforms_kernel,
                dim=skeleton.num_joints,
                inputs=[self._world_pose, self.skeletal_mesh.inverse_bind_transforms],
                outputs=[self.skinning_transforms],
            )

            for mesh_index, mesh in enumerate(self.skeletal_mesh.skinned_meshes):
                if mesh.num_points == 0:
                    continue
                wp.launch(
                    _skin_mesh_kernel,
                    dim=mesh.num_points,
                    inputs=[
                        mesh.points,
                        mesh.joint_indices,
                        mesh.joint_weights,
                        mesh.num_influences,
                        self.skinning_transforms,
                    ],
                    outputs=[self.skinned_points[mesh_index]],
                )

    def hide(self, viewer, instance_id: int | str) -> None:
        """Hide one drawn instance, keeping its data for the next draw.

        Args:
            viewer: Newton viewer that shows the drawing.
            instance_id: Identifier of the instance to hide.
        """
        for mesh_index, mesh in enumerate(self.skeletal_mesh.skinned_meshes):
            name = self._name(instance_id, mesh_index)
            self._log_hidden_once(
                viewer.log_mesh,
                name,
                self.skinned_points[mesh_index],
                mesh.indices,
            )

    def clear(self, viewer) -> None:
        """Remove every drawn instance from the viewer.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._clear(viewer.objects)
