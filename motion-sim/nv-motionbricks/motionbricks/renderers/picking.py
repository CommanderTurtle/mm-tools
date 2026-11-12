# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ray helpers for selecting thin rendered lines."""

import warp as wp
from newton._src.geometry.raycast import map_ray_to_local, ray_intersect_capsule

_LINE_PICK_RADIUS = 0.025


def raycast_line(
    start: wp.vec3,
    end: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    max_distance: float,
) -> float:
    """Return the distance where a ray hits a line, treated as a thin capsule.

    Args:
        start: Start of the line.
        end: End of the line.
        origin: Ray origin, in the line's space.
        direction: Unit ray direction, in the line's space.
        max_distance: Longest distance to test, in meters. A miss returns it.
    """
    length = wp.length(end - start)
    # quat_between_vectors() normalizes the line, and a zero length line gives the identity.
    rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), end - start)
    local_origin, local_direction = map_ray_to_local(wp.transform((start + end) * 0.5, rotation), origin, direction)
    distance, _ = ray_intersect_capsule(local_origin, local_direction, _LINE_PICK_RADIUS, length * 0.5)
    return distance if 0 <= distance < max_distance else max_distance


def raycast_lines(
    transform: wp.transform,
    lines,
    origin: wp.vec3,
    direction: wp.vec3,
    max_distance: float,
) -> float:
    """Return the distance where a ray first hits one of several lines, each a thin capsule.

    Args:
        transform: World transform of the space the lines are in.
        lines: (start, end) pairs in that space.
        origin: Ray origin in world space.
        direction: Unit ray direction in world space.
        max_distance: Longest distance to test, in meters. A miss returns it.
    """
    origin, direction = map_ray_to_local(transform, origin, direction)
    for start, end in lines:
        max_distance = raycast_line(start, end, origin, direction, max_distance)
    return max_distance
