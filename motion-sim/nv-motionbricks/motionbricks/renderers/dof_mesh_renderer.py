# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw and pick a robot mesh using its joint coordinates."""

import newton
import numpy as np
import warp as wp
from newton import GeoType
from newton._src.geometry.raycast import map_ray_to_local, ray_intersect_mesh, ray_intersect_shape
from newton._src.viewer.gl.opengl import update_vbo_transforms
from newton._src.viewer.kernels import update_shape_xforms

from motionbricks.renderers.base_renderer import BaseRenderer


@wp.kernel
def _raycast_shapes(
    shapes: wp.array(dtype=int),
    transforms: wp.array(dtype=wp.transform),
    types: wp.array(dtype=int),
    scales: wp.array(dtype=wp.vec3),
    meshes: wp.array(dtype=wp.uint64),
    origin: wp.vec3,
    direction: wp.vec3,
    max_distance: float,
    distances: wp.array(dtype=float),
):
    index = wp.tid()
    shape = shapes[index]
    if types[shape] == GeoType.MESH or types[shape] == GeoType.CONVEX_MESH:
        local_origin, local_direction = map_ray_to_local(transforms[shape], origin, direction, scales[shape])
        distance, normal, u, v, face = ray_intersect_mesh(
            local_origin, local_direction, scales[shape], meshes[shape], False, max_distance,
        )
    else:
        distance, normal = ray_intersect_shape(
            transforms[shape], scales[shape], types[shape], origin, direction, False,
        )
    distances[index] = wp.where(distance >= 0.0, wp.min(distance, max_distance), max_distance)


class DOFMeshRenderer(BaseRenderer):
    """Draw a Newton model's visible shapes using its joint coordinates.

    Geometry is uploaded once. All pose instances share it. FK and picking
    use scratch buffers, so the Newton model's joint coordinates and playback state
    stay unchanged.
    """

    def __init__(self, model: newton.Model) -> None:
        """Collect the model's visible shapes and allocate scratch buffers.

        Args:
            model: Newton model to draw.
        """
        super().__init__()
        self.model = model
        self.state = model.state()
        self.shapes = np.flatnonzero(model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)).tolist()
        self.shape_indices = wp.array(self.shapes, dtype=int, device=model.device)
        self.world_transforms = wp.empty_like(model.shape_transform)
        self.distances = wp.empty(len(self.shapes), dtype=float, device=model.device)
        # GL uploads colors and materials from the host, so these constants need no GPU round trip.
        self.color = wp.empty(1, dtype=wp.vec3, device="cpu")
        self.shape_colors = wp.array(model.shape_color.numpy(), dtype=wp.vec3, device="cpu")
        self.material = wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device="cpu")
        self.vbo_transforms = wp.empty(model.shape_count, dtype=wp.mat44, device=model.device)
        types, scales = model.shape_type.numpy(), model.shape_scale.numpy().copy()
        margins, solids = model.shape_margin.numpy(), model.shape_is_solid.numpy()
        self.geometry = {
            shape: dict(
                geo_type=int(types[shape]), geo_scale=tuple(float(v) for v in scales[shape]),
                geo_thickness=float(margins[shape]), geo_is_solid=bool(solids[shape]),
                geo_src=model.shape_source[shape],
            )
            for shape in self.shapes
        }
        # log_geo bakes primitive sizes into the geometry, and mesh scales belong to the instances.
        scales[~np.isin(types, [GeoType.MESH, GeoType.CONVEX_MESH])] = 1.0
        self.scales = wp.array(scales, dtype=wp.vec3, device=model.device)
        self._geometry_loaded = False

    @staticmethod
    def _name(instance_id: int | str, shape: int) -> str:
        return f"/dof_mesh_{instance_id}_{shape}"

    def _place_shapes(self, joint_q: wp.array, xform: wp.transform) -> None:
        if joint_q.dtype != wp.float32 or joint_q.shape != (self.model.joint_coord_count,):
            raise ValueError("joint_q must contain one pose with all of the Newton model's joint coordinates.")
        newton.eval_fk(self.model, joint_q, self.model.joint_qd, self.state)
        wp.launch(
            update_shape_xforms, dim=self.model.shape_count,
            inputs=[self.model.shape_transform, self.model.shape_body, self.state.body_q,
                    self.model.shape_world, None, xform],
            outputs=[self.world_transforms], device=self.model.device,
        )

    def draw_pose(
        self, viewer, joint_q: wp.array, xform: wp.transform, color: wp.vec3 | None, instance_id: int | str,
    ) -> None:
        """Draw one pose of the model.

        Args:
            viewer: Newton viewer to draw into.
            joint_q: Joint coordinates of the pose, with all of the model's coordinates.
            xform: World transform applied to the pose.
            color: RGB color of every shape. None uses each shape's own color.
            instance_id: Identifier of the drawn instance. Drawing the same identifier
                again updates it.
        """
        if not self._geometry_loaded:
            for shape, geometry in self.geometry.items():
                viewer.log_geo(f"/dof_geometry_{shape}", **geometry, hidden=True)
            self._geometry_loaded = True
        self._place_shapes(joint_q, xform)
        # Pack every part together, matching Newton's batched model-rendering path.
        # A single host copy avoids a separate GPU synchronization for each part.
        wp.launch(
            update_vbo_transforms, dim=self.model.shape_count,
            inputs=[self.world_transforms, self.scales], outputs=[self.vbo_transforms],
            device=self.model.device,
        )
        transforms = self.vbo_transforms.numpy()
        if color is not None:
            self.color.fill_(color)
        for shape in self.shapes:
            colors = self.color if color is not None else self.shape_colors[shape:shape + 1]
            name = self._name(instance_id, shape)
            if name not in self._registered_names:
                self._log_visible(
                    viewer.log_instances, name, f"/dof_geometry_{shape}",
                    self.world_transforms[shape:shape + 1], self.scales[shape:shape + 1],
                    colors, self.material,
                )
            else:
                # A viewer layer keeps its objects under the layer's name prefix.
                instance = viewer.objects[viewer.layer.name_prefix + name]
                instance.update_from_pinned(transforms[shape:shape + 1], 1, colors, self.material)
                instance.hidden = False
                self._hidden_names.discard(name)

    def raycast_pose(
        self, joint_q: wp.array, xform: wp.transform, origin: wp.vec3,
        direction: wp.vec3, max_distance: float = 1.0e6,
    ) -> float:
        """Return the distance to the nearest visible shape along a ray.

        Args:
            joint_q: Joint coordinates of the pose, with all of the model's coordinates.
            xform: World transform applied to the pose.
            origin: Ray origin in world space.
            direction: Unit ray direction in world space.
            max_distance: Longest distance to test, in meters. A miss returns it.
        """
        if not self.shapes:
            return max_distance
        self._place_shapes(joint_q, xform)
        wp.launch(
            _raycast_shapes, dim=len(self.shapes),
            inputs=[self.shape_indices, self.world_transforms, self.model.shape_type,
                    self.model.shape_scale, self.model.shape_source_ptr, origin, direction, max_distance],
            outputs=[self.distances], device=self.model.device,
        )
        return float(self.distances.numpy().min())

    def hide(self, viewer, instance_id: int | str) -> None:
        """Hide one drawn instance, keeping its data for the next draw.

        Args:
            viewer: Newton viewer that shows the drawing.
            instance_id: Identifier of the instance to hide.
        """
        for shape in self.shapes:
            self._log_hidden_once(
                viewer.log_instances, self._name(instance_id, shape), f"/dof_geometry_{shape}",
                None, None, None, None,
            )

    def clear(self, viewer) -> None:
        """Remove every drawn instance from the viewer.

        The shared geometry stays loaded, so drawing again does not upload it.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._clear(viewer.objects)
