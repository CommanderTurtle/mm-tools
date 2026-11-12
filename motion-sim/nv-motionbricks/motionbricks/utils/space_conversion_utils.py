# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert transforms from Y up source data to Newton's Z up space."""

from enum import IntEnum, auto

import warp as wp


class FacingDirectionType(IntEnum):
    """How converted characters face, following Maya or MuJoCo conventions."""
    MAYA = auto()
    MUJOCO = auto()


class SpaceConverter:
    """Convert source data from Y up to Newton's Z up space.

    The facing direction type sets which way characters face after conversion.
    """
    def __init__(self, facing_direction: FacingDirectionType):
        """Choose the rotation for a facing direction convention.

        Args:
            facing_direction: MAYA or MUJOCO. Any other value keeps transforms
                unchanged.
        """
        if facing_direction == FacingDirectionType.MUJOCO:
            self.converter = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.radians(90.0))
        elif facing_direction == FacingDirectionType.MAYA:
            q1 = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), wp.radians(90.0))
            q2 = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.radians(90.0))
            self.converter = q1 * q2
        else:
            self.converter = wp.quat_identity()

    def convert_transform(self, transform):
        """Convert a transform from the source space to Newton's space.

        Args:
            transform: Transform in the source space.
        """
        return wp.mul(wp.transform(wp.vec3(0, 0, 0), self.converter), transform)
