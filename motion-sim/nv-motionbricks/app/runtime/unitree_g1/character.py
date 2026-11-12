# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unitree G1 character animated by the MotionBricks G1 model."""

from __future__ import annotations

from pathlib import Path

import warp as wp

from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine
from motionbricks.runtime.motion_player import GeneratedDOFMotionPlayer

from ..common.character import BaseCharacter
from ..common.interaction import InteractionProbe
from ..common.io.smart_primitive_document import SmartPrimitiveAssetPool
from ..common.motion import LINEAR_LOCOMOTION_MODE, LocomotionMode
from .presets import (
    DEFAULT_CROUCH_IDLE_CLIPS, DEFAULT_IDLE_CLIPS, DEFAULT_LOCOMOTION_MODES, INTERACTION_PROBE,
)
from .target_pose_blender import G1TargetPoseBlender


class UnitreeG1Character(BaseCharacter):
    """Unitree G1 character animated by the MotionBricks G1 model."""

    root_forward_axis = wp.vec3(1.0, 0.0, 0.0)

    def __init__(
        self,
        engine: MotionBricksUnitreeG1Engine,
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
        """Create the G1 character with its target pose blender.

        The arguments are as BaseCharacter takes them. The defaults are the G1 presets.
        """
        self.target_pose_blender = G1TargetPoseBlender()
        super().__init__(
            engine,
            asset_pool=asset_pool,
            initial_transform=initial_transform,
            modes=modes,
            initial_mode=initial_mode,
            idle_paths=idle_paths,
            crouch_idle_paths=crouch_idle_paths,
            interaction_probe=interaction_probe,
            current_pose=current_pose,
        )

    def _create_player(self) -> GeneratedDOFMotionPlayer:
        return GeneratedDOFMotionPlayer(self.engine, self.current_pose)


__all__ = ["UnitreeG1Character"]
