"""Tier-1 post-hoc pruning: drop detections larger than the room itself.

A perception failure under distribution shift (seen at 5x frame density on
scannetpp `07ff1c45bb`) merges several instances into one room-spanning blob:
OBB volume 3.1x the room-box prior's volume, versus <=14% for the largest real
object in the same scene. One such instance spoils every render and metric.

This gate prunes any NON-background instance whose OBB volume exceeds
``ratio`` times the room-box prior's volume. The room instance is exempt by
identity -- the prior already names it -- which is what makes a size rule safe:
size alone cannot distinguish "the room" from "a blob", identity can. There is
deliberately nothing subtler here: containment-style merge detectors eat nested
real objects (a book inside a bookshelf), and support-density gates eat
partially observed ones. An instance bigger than half the room, that is not the
room, is the one case that is unambiguous.

Requires the room-box prior; without it (GT runs, baselines) the gate is a
no-op, because there is nothing safe to normalise size against.
"""

from __future__ import annotations

import numpy as np

# 1.0 = "bigger than the room itself", which is physically impossible for a
# real object. Measured margins on scannetpp 07ff1c45bb: the merged blob is
# 4.9x the room-box volume; the largest real objects (edge-2.98 OBB cubes) are
# 0.475x -- so 0.5 would keep them by only 5%, while 1.0 keeps a 2.1x margin
# below the gate and 4.9x above it. Note the "volume" is (OBB cube edge)^3,
# which overestimates flat/elongated furniture; that is fine at 1.0 and is
# exactly why 0.5 was too tight.
DEFAULT_VOLUME_RATIO = 1.0


def prune_oversized_instances(
    *,
    local_ids: np.ndarray,
    object_ids: list[int],
    object_names: list[str],
    point_counts: list[int],
    scene_to_object: list[np.ndarray],
    normalized_obbs: list[dict],
    retained_raw_obbs: list[dict],
    background_prior: dict,
    volume_ratio: float = DEFAULT_VOLUME_RATIO,
) -> tuple[np.ndarray, dict]:
    """Prune in place-compatible fashion; returns (new local_ids, audit).

    All per-instance lists are mutated to the kept subset (order preserved),
    the label array is remapped, and the prior's background local id is updated
    to its new index.
    """

    box = background_prior.get("box") or {}
    extents = np.asarray(box.get("full_extents", ()), dtype=np.float64).reshape(-1)
    if extents.size != 3 or not np.all(extents > 0):
        return local_ids, {"applied": False, "reason": "no_room_box_extents"}
    room_volume = float(np.prod(extents))
    background_local = int(background_prior["background_local_instance_id"])

    pruned, audit_rows = [], []
    for local_id, obb in enumerate(normalized_obbs):
        if local_id == background_local:
            continue
        edge = float(np.asarray(obb["scale"], dtype=np.float64).reshape(-1)[0])
        volume = edge ** 3
        if volume > volume_ratio * room_volume:
            pruned.append(local_id)
            audit_rows.append({
                "local_id": local_id,
                "object_id": int(object_ids[local_id]),
                "object_name": object_names[local_id],
                "obb_edge": edge,
                "obb_volume": volume,
                "room_volume": room_volume,
                "volume_ratio": volume / room_volume,
                "num_points": int(point_counts[local_id]),
            })
    audit = {
        "applied": True,
        "volume_ratio_threshold": float(volume_ratio),
        "room_volume": room_volume,
        "num_pruned": len(pruned),
        "pruned": audit_rows,
    }
    if not pruned:
        return local_ids, audit

    keep = [i for i in range(len(object_ids)) if i not in set(pruned)]
    remap = np.full(len(object_ids), -1, dtype=np.int64)
    for new_id, old_id in enumerate(keep):
        remap[old_id] = new_id
    new_local_ids = np.where(
        local_ids >= 0, remap[np.clip(local_ids, 0, len(remap) - 1)], -1
    )

    def take(values: list) -> list:
        return [values[i] for i in keep]

    object_ids[:] = take(object_ids)
    object_names[:] = take(object_names)
    point_counts[:] = take(point_counts)
    scene_to_object[:] = take(scene_to_object)
    normalized_obbs[:] = take(normalized_obbs)
    retained_raw_obbs[:] = take(retained_raw_obbs)
    background_prior["background_local_instance_id"] = int(remap[background_local])
    return new_local_ids, audit
