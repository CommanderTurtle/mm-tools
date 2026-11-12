# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Load each rig's character and clip registry, place its model space in world space, and draw it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import warp as wp

import motionbricks.io.utils as io_utils
from motionbricks.model.contract import ModelRig
from motionbricks.model.soma.contract import SOMA_N1_FOOT_CONTACT_JOINTS
from motionbricks.model.unitree_g1.contract import (
    UNITREE_G1_N1_CONTRACT,
    UNITREE_G1_N1_FOOT_CONTACT_BODIES,
    UNITREE_G1_N1_FOOT_CONTACT_OFFSETS,
)
from motionbricks.renderers.dof_mesh_renderer import DOFMeshRenderer
from motionbricks.renderers.foot_contact_renderer import (
    DOFFootContactRenderer,
    SkeletonFootContactRenderer,
    body_sole_outlines,
    skeletal_mesh_sole_outlines,
)
from motionbricks.renderers.skeletal_mesh_renderer import SkeletalMeshRenderer
from motionbricks.renderers.skeleton_renderer import DOFSkeletonRenderer, SkeletonRenderer
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

from .clip_registry import ClipRegistry

if TYPE_CHECKING:
    import newton

    from motionbricks.animation.mesh import SkeletalMesh


# The repository's assets folder, separate from the assets of the motionbricks package.
ASSETS_ROOT = Path(__file__).resolve().parents[3] / "assets"
RIG_LABELS = {ModelRig.SOMA: "SOMA", ModelRig.UNITREE_G1: "Unitree G1"}
# SOMA draws in gray. None draws the robot in its own colors.
CHARACTER_COLORS = {ModelRig.SOMA: wp.vec3(0.7, 0.7, 0.7), ModelRig.UNITREE_G1: None}


def model_to_runtime(rig: ModelRig) -> wp.transform:
    """Return the transform from a rig's model space to world space, which is Z up.

    SOMA model space and BVH clips are Y up. The MuJoCo convention turns them to
    face the same way as G1 and its CSV clips, which need no conversion.
    """
    if rig is ModelRig.SOMA:
        return SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
    return wp.transform_identity()


@dataclass(frozen=True)
class RigAssets:
    """What load_rig_assets() loads for one rig.

    Attributes:
        model: The SOMA skeletal mesh, or the G1 Newton model.
        clip_registry: Registry for the rig's clips. It starts empty.
        reference_pose: Reference pose of the rig. It sets the pose layout.
        model_to_runtime: Transform from the rig's model space to world space.
    """

    model: SkeletalMesh | newton.Model
    clip_registry: ClipRegistry
    reference_pose: wp.array
    model_to_runtime: wp.transform


def load_rig_assets(rig: ModelRig) -> RigAssets:
    """Load a rig's skeletal mesh or Newton model, and create an empty clip registry for it."""
    if rig is ModelRig.SOMA:
        skeletal_mesh = io_utils.get_soma_human_model()
        return RigAssets(
            skeletal_mesh,
            ClipRegistry(rig, skeleton=skeletal_mesh.skeleton),
            skeletal_mesh.skeleton.reference_local_transforms,
            model_to_runtime(rig),
        )
    robot = io_utils.get_unitree_g1_model()
    return RigAssets(
        robot,
        ClipRegistry(rig, joint_names=UNITREE_G1_N1_CONTRACT.joint_names),
        wp.clone(robot.joint_q),
        model_to_runtime(rig),
    )


def create_mesh_renderer(rig: ModelRig, model) -> SkeletalMeshRenderer | DOFMeshRenderer:
    """Create the renderer that draws the rig's skeletal mesh or robot.

    Args:
        rig: Rig of the model.
        model: The SOMA skeletal mesh, or the G1 Newton model.
    """
    if rig is ModelRig.SOMA:
        renderer = SkeletalMeshRenderer(model)
        # Draw relaxed fingers. The pose data keeps the motion's fingers.
        renderer.set_override_pose("RelaxedHands", mask="Fingers")
        return renderer
    return DOFMeshRenderer(model)


def create_skeleton_renderer(rig: ModelRig, model, *, max_instances: int = 1) -> SkeletonRenderer | DOFSkeletonRenderer:
    """Create the renderer that draws the rig's poses as bone lines.

    Args:
        rig: Rig of the model.
        model: The SOMA skeletal mesh, or the G1 Newton model.
        max_instances: Number of poses one draw call can hold.
    """
    if rig is ModelRig.SOMA:
        return SkeletonRenderer(model.skeleton, max_instances=max_instances)
    return DOFSkeletonRenderer(model, max_instances=max_instances)


def create_foot_contact_renderer(
    rig: ModelRig, name: str, model,
) -> SkeletonFootContactRenderer | DOFFootContactRenderer:
    """Create the renderer that draws the rig's foot contacts on its soles.

    Args:
        rig: Rig of the model.
        name: Viewer name of the outline lines.
        model: The SOMA skeletal mesh, or the G1 Newton model.
    """
    if rig is ModelRig.SOMA:
        return SkeletonFootContactRenderer(
            name, model.skeleton, skeletal_mesh_sole_outlines(model, SOMA_N1_FOOT_CONTACT_JOINTS),
        )
    return DOFFootContactRenderer(
        name, model, body_sole_outlines(model, UNITREE_G1_N1_FOOT_CONTACT_BODIES, UNITREE_G1_N1_FOOT_CONTACT_OFFSETS),
    )


__all__ = [
    "ASSETS_ROOT",
    "CHARACTER_COLORS",
    "RIG_LABELS",
    "RigAssets",
    "create_foot_contact_renderer",
    "create_mesh_renderer",
    "create_skeleton_renderer",
    "load_rig_assets",
    "model_to_runtime",
]
