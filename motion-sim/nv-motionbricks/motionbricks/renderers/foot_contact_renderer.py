# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Outline the heels and toes in the Newton viewer where the model predicts foot contact."""

from collections.abc import Sequence
from typing import NamedTuple

import newton
from newton import GeoType
import numpy as np
from scipy.spatial import ConvexHull
import warp as wp

from motionbricks.animation.foot_contacts import FOOT_CONTACT_COUNT
from motionbricks.animation.skeleton import Skeleton
from motionbricks.renderers.base_renderer import BaseRenderer
from motionbricks.utils.pose_utils import compute_world_poses

_OUTLINE_LIFT = 0.004
_FLOOR_CLEARANCE = 0.003


class ContactOutline(NamedTuple):
    """A closed loop of points that shows one foot contact.

    The points move with one of the transforms passed to draw(). The transform
    field picks which one.
    """

    contact: int
    transform: int
    points: np.ndarray


@wp.kernel
def _update_outlines_kernel(
    transforms: wp.array(dtype=wp.transform),
    contacts: wp.array(dtype=wp.float32),
    threshold: wp.float32,
    segment_transforms: wp.array(dtype=wp.int32),
    segment_contacts: wp.array(dtype=wp.int32),
    local_starts: wp.array(dtype=wp.vec3),
    local_ends: wp.array(dtype=wp.vec3),
    floor: wp.float32,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    segment = wp.tid()
    contact = wp.clamp(contacts[segment_contacts[segment]], 0.0, 1.0)
    transform = transforms[segment_transforms[segment]]
    start = wp.transform_point(transform, local_starts[segment])
    end = wp.transform_point(transform, local_ends[segment])
    start = wp.vec3(start[0], start[1], wp.max(start[2], floor))
    end = wp.vec3(end[0], end[1], wp.max(end[2], floor))
    starts[segment] = start
    # A segment of zero length draws nothing.
    ends[segment] = wp.where(contact > wp.clamp(threshold, 0.0, 1.0), end, start)
    colors[segment] = wp.vec3(1.0, 1.0 - contact, 0.0)


class FootContactRenderer(BaseRenderer):
    """Draw outlines for foot contacts above the threshold.

    See motionbricks.animation.foot_contacts for the foot contact order and values.
    Colors blend from yellow at 0 to red at 1. A foot contact of 0 never shows.
    Outlines stay just above the floor at height 0.
    """

    DEFAULT_THRESHOLD = 0.5

    def __init__(self, name: str, outlines: Sequence[ContactOutline], *, device=None) -> None:
        """Upload the outline points.

        The threshold attribute starts at DEFAULT_THRESHOLD.

        Args:
            name: Viewer name of the outline lines.
            outlines: Outlines from skeletal_mesh_sole_outlines() or
                body_sole_outlines().
            device: Device of the drawn poses, where the outlines and lines also live.
                None uses the current device.
        """
        super().__init__()
        if not outlines or any(not 0 <= outline.contact < FOOT_CONTACT_COUNT for outline in outlines):
            raise ValueError(f"outlines must be non-empty, with foot contacts in [0, {FOOT_CONTACT_COUNT}).")
        repeat = [len(outline.points) for outline in outlines]
        self._name = f"/{name}"
        self._device = wp.get_device(device)
        self.threshold = self.DEFAULT_THRESHOLD
        self._inputs = [
            wp.array(np.repeat([outline.transform for outline in outlines], repeat), dtype=wp.int32, device=device),
            wp.array(np.repeat([outline.contact for outline in outlines], repeat), dtype=wp.int32, device=device),
            wp.array(np.concatenate([outline.points for outline in outlines]), dtype=wp.vec3, device=device),
            wp.array(
                np.concatenate([np.roll(outline.points, -1, axis=0) for outline in outlines]),
                dtype=wp.vec3, device=device,
            ),
            _FLOOR_CLEARANCE,
        ]
        self._lines = tuple(wp.empty(sum(repeat), dtype=wp.vec3, device=device) for _ in range(3))

    def draw(self, viewer, transforms: wp.array, contacts: wp.array) -> None:
        """Draw the outlines for one pose.

        Args:
            viewer: Newton viewer to draw into.
            transforms: World transforms of the pose's joints or bodies, as the outlines
                index them.
            contacts: The pose's four sampled foot contacts.
        """
        wp.launch(
            _update_outlines_kernel,
            dim=len(self._lines[0]),
            inputs=[transforms, contacts, self.threshold, *self._inputs],
            outputs=list(self._lines),
            device=self._device,
        )
        self._log_visible(viewer.log_lines, self._name, *self._lines)

    def hide(self, viewer) -> None:
        """Hide the outlines, keeping their data for the next draw.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._log_hidden_once(viewer.log_lines, self._name, *self._lines)

    def clear(self, viewer) -> None:
        """Remove the outlines from the viewer.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._clear(viewer.lines)


class SkeletonFootContactRenderer(FootContactRenderer):
    """Draw foot contact outlines on a pose of local joint transforms, such as a SOMA pose."""

    def __init__(self, name: str, skeleton: Skeleton, outlines: Sequence[ContactOutline]) -> None:
        """Upload the outline points and allocate the world joint transforms.

        Args:
            name: Viewer name of the outline lines.
            skeleton: Skeleton of the drawn poses.
            outlines: Outlines from skeletal_mesh_sole_outlines(), relative to the
                skeleton's joints.
        """
        super().__init__(name, outlines, device=skeleton.device)
        self._parent_indices = skeleton.parent_indices
        self._world_transforms = wp.empty(skeleton.num_joints, dtype=wp.transform, device=skeleton.device)

    def draw_pose(self, viewer, local_pose: wp.array, contacts: wp.array) -> None:
        """Draw the outlines on one pose of local joint transforms.

        Args:
            viewer: Newton viewer to draw into.
            local_pose: Local joint transforms of the pose.
            contacts: The pose's four sampled foot contacts.
        """
        with wp.ScopedDevice(self._device):
            compute_world_poses(local_pose, self._parent_indices, self._world_transforms)
        self.draw(viewer, self._world_transforms, contacts)


class DOFFootContactRenderer(FootContactRenderer):
    """Draw foot contact outlines on a pose of joint coordinates, such as a G1 pose.

    Forward kinematics runs in the renderer's own Newton state, so the outlines never
    depend on a mesh draw.
    """

    def __init__(self, name: str, model, outlines: Sequence[ContactOutline]) -> None:
        """Upload the outline points and create the state for forward kinematics.

        Args:
            name: Viewer name of the outline lines.
            model: Newton model of the drawn poses.
            outlines: Outlines from body_sole_outlines(), relative to the model's bodies.
        """
        super().__init__(name, outlines, device=model.device)
        self._model = model
        self._state = model.state()

    def draw_pose(self, viewer, joint_q: wp.array, contacts: wp.array) -> None:
        """Draw the outlines on one pose of joint coordinates.

        Args:
            viewer: Newton viewer to draw into.
            joint_q: Joint coordinates of the pose, in the model's joint_q order.
            contacts: The pose's four sampled foot contacts.
        """
        newton.eval_fk(self._model, joint_q, self._model.joint_qd, self._state)
        self.draw(viewer, self._state.body_q, contacts)


def skeletal_mesh_sole_outlines(
    skeletal_mesh,
    contact_joints: Sequence[str],
    *,
    margin: float = 0.015,
    rings: int = 3,
    spacing: float = 0.005,
) -> list[ContactOutline]:
    """Outline each foot of a skinned mesh, seen from above in its Y up bind pose.

    A foot's vertices are those that its heel joint and the joints below it mainly
    move. The outline splits at the toe joint into a heel part and a toe part.
    Points are stored relative to their joint. Draw them with world joint
    transforms. The bind pose must face +Z.

    Args:
        skeletal_mesh: Skeletal mesh whose feet to outline.
        contact_joints: Joint of each foot contact, in foot contact order.
        margin: Distance the first ring runs outside the foot, in meters.
        rings: Number of rings around each foot.
        spacing: Distance between rings, in meters.
    """
    names = list(skeletal_mesh.skeleton.joint_names)
    parents = skeletal_mesh.skeleton.parent_indices.numpy()
    inverse_bind = skeletal_mesh.inverse_bind_transforms.numpy()
    outlines = []
    for foot, (heel_name, toe_name) in enumerate(zip(contact_joints[0::2], contact_joints[1::2], strict=True)):
        heel, toe = names.index(heel_name), names.index(toe_name)
        joints = [heel]
        for joint in range(len(names)):
            if parents[joint] in joints:
                joints.append(joint)
        foot_points = np.concatenate([_vertices_moved_by(mesh, joints) for mesh in skeletal_mesh.skinned_meshes])
        split = wp.transform_inverse(wp.transform(*inverse_bind[toe])).p[2]
        height = foot_points[:, 1].min() + _OUTLINE_LIFT
        # Columns z and x hold the forward and side coordinates.
        for loops in _rings(foot_points[:, [2, 0]], split, margin, rings, spacing):
            for contact, (joint, loop) in enumerate(zip((heel, toe), loops), start=2 * foot):
                bind = np.column_stack([loop[:, 1], np.full(len(loop), height), loop[:, 0]])
                joint_from_bind = wp.transform(*inverse_bind[joint])
                points = np.array([wp.transform_point(joint_from_bind, wp.vec3(*point)) for point in bind])
                outlines.append(ContactOutline(contact, joint, points))
    return outlines


def body_sole_outlines(
    model,
    contact_bodies: Sequence[str],
    contact_offsets: Sequence[Sequence[float]],
    *,
    margin: float = 0.015,
    rings: int = 3,
    spacing: float = 0.005,
) -> list[ContactOutline]:
    """Outline each foot body of a Newton model, seen from above in its Z up space.

    A foot's heel and toe share one body. The body must face +X. The outline splits
    halfway between the heel and toe points. Points are stored relative to their
    body. Draw them with world body transforms.

    Args:
        model: Newton model with the foot bodies.
        contact_bodies: Body of each foot contact, in foot contact order.
        contact_offsets: Each contact point in its body's local space, in meters.
        margin: Distance the first ring runs outside the foot, in meters.
        rings: Number of rings around each foot.
        spacing: Distance between rings, in meters.
    """
    labels = [label.rsplit("/", 1)[-1] for label in model.body_label]
    shape_bodies = model.shape_body.numpy()
    is_mesh = np.isin(model.shape_type.numpy(), (GeoType.MESH, GeoType.CONVEX_MESH))
    shape_transforms = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    outlines = []
    for foot in range(len(contact_bodies) // 2):
        name, toe_name = contact_bodies[2 * foot], contact_bodies[2 * foot + 1]
        if name != toe_name:
            raise ValueError(f"The heel and toe contacts of {name!r} must share one body.")
        body = labels.index(name)
        split = (contact_offsets[2 * foot][0] + contact_offsets[2 * foot + 1][0]) / 2.0
        points = np.array([
            wp.transform_point(wp.transform(*shape_transforms[shape]), wp.vec3(*vertex))
            for shape in np.flatnonzero((shape_bodies == body) & is_mesh)
            for vertex in np.asarray(model.shape_source[shape].vertices) * scales[shape]
        ])
        if len(points) == 0:
            raise ValueError(f"Body {name!r} has no mesh to outline.")
        height = points[:, 2].min() + _OUTLINE_LIFT
        for loops in _rings(points[:, :2], split, margin, rings, spacing):
            for contact, loop in enumerate(loops, start=2 * foot):
                outlines.append(ContactOutline(contact, body, np.column_stack([loop, np.full(len(loop), height)])))
    return outlines


def _rings(points: np.ndarray, split: float, margin: float, rings: int, spacing: float) -> list:
    """Return the heel and toe loops of each ring around (forward, side) points.

    The loops are cut at split. Each ring scales the convex hull about its center.
    Its ends and sides move out by margin, plus spacing for each ring inside it.
    """
    hull = points[ConvexHull(points).vertices]
    low, high = hull.min(axis=0), hull.max(axis=0)
    center, reach = (low + high) / 2.0, (high - low) / 2.0
    grown = [center + (hull - center) * (1.0 + (margin + ring * spacing) / reach) for ring in range(rings)]
    return [(_clip(loop, split, 1.0), _clip(loop, split, -1.0)) for loop in grown]


def _clip(polygon: np.ndarray, split: float, side: float) -> np.ndarray:
    """Keep the part of a convex polygon where side * (forward - split) is at most zero."""
    kept = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        a, b = side * (start[0] - split), side * (end[0] - split)
        if a <= 0.0:
            kept.append(start)
        if a * b < 0.0:
            kept.append(start + (end - start) * a / (a - b))
    return np.array(kept)


def _vertices_moved_by(mesh, joints: Sequence[int]) -> np.ndarray:
    """Return the vertices of a skinned mesh part that the given joints mainly move."""
    indices = mesh.joint_indices.numpy().reshape(mesh.num_points, -1)
    weights = mesh.joint_weights.numpy().reshape(mesh.num_points, -1)
    return mesh.points.numpy()[np.where(np.isin(indices, joints), weights, 0.0).sum(axis=1) > 0.5]
