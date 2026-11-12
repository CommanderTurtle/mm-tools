# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SOMA character animated by the MotionBricks SOMA model."""

from __future__ import annotations

from pathlib import Path

import warp as wp

from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.runtime.motion_player import GeneratedMotionPlayer
from motionbricks.runtime.target_pose import (
    TargetPoseIKCorrector,
    create_soma_target_pose_ik_rig,
)

from ..common.character import BaseCharacter
from ..common.interaction import InteractionProbe
from ..common.io.smart_primitive_document import SmartPrimitiveAssetPool
from ..common.motion import LINEAR_LOCOMOTION_MODE, LocomotionMode
from .presets import (
    DEFAULT_CROUCH_IDLE_CLIPS, DEFAULT_IDLE_CLIPS, DEFAULT_LOCOMOTION_MODES, INTERACTION_PROBE,
)
from .target_pose_blender import SOMATargetPoseBlender


class SOMACharacter(BaseCharacter):
    """SOMA character animated by the MotionBricks SOMA model, with target pose IK for pinned hands and feet."""

    root_forward_axis = wp.vec3(0.0, 1.0, 0.0)

    def __init__(
        self,
        engine: MotionBricksSOMAEngine,
        *,
        asset_pool: SmartPrimitiveAssetPool,
        initial_transform: wp.transform = wp.transform_identity(),
        modes: tuple[LocomotionMode, ...] = DEFAULT_LOCOMOTION_MODES,
        initial_mode: str = LINEAR_LOCOMOTION_MODE,
        idle_paths: tuple[str | Path, ...] = DEFAULT_IDLE_CLIPS,
        crouch_idle_paths: tuple[str | Path, ...] = DEFAULT_CROUCH_IDLE_CLIPS,
        interaction_probe: InteractionProbe = INTERACTION_PROBE,
        current_pose: wp.array | None = None,
    ) -> None:
        """Create the SOMA character with its target pose blender and target pose IK.

        The arguments are as BaseCharacter takes them. The defaults are the SOMA presets.
        """
        self.skeleton = asset_pool.clip_registry.skeleton
        self.target_pose_blender = SOMATargetPoseBlender(self.skeleton.reference_local_transforms)
        super().__init__(
            engine,
            asset_pool=asset_pool,
            initial_transform=initial_transform,
            modes=modes,
            initial_mode=initial_mode,
            idle_paths=idle_paths,
            crouch_idle_paths=crouch_idle_paths,
            interaction_probe=interaction_probe,
            target_pose_ik_corrector=TargetPoseIKCorrector(
                self.skeleton,
                create_soma_target_pose_ik_rig(self.skeleton),
            ),
            current_pose=current_pose,
        )

    def _create_player(self) -> GeneratedMotionPlayer:
        return GeneratedMotionPlayer(self.engine, self.skeleton)


__all__ = ["SOMACharacter"]
