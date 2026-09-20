"""Drive a native SOMA-X body from ARDY motion and export skinned animation.

Two pure-numpy pieces, both GPU-free so they stay unit-testable off-box:

- ``solve_soma_pose`` converts world-space joint positions from an ARDY
  humanoid skeleton (cskel27 / somaskel30 / somaskel77) into SOMA public
  pose parameters: 77 axis-angle joints relative to the fitted T-pose plus
  the Hips translation (exactly ``SOMALayer.pose`` input). Each mapped bone
  is rescaled onto its own SOMA rest length and rotated by the minimum
  rotation from the rest local direction to the current local direction, so
  differing rest shapes and units are absorbed per bone. Unmapped joints
  (fingers, jaw, eyes, ...) hold their rest articulation and simply follow
    their nearest mapped ancestor. This is visualization-grade IK-lite: the
    twist gauge around each bone axis is the minimum solution, not a
    constrained solve. Sources whose arm chains stay at rest while the legs
    walk get a gait-locked procedural arm swing instead of a frozen V-pose;
    see ``swing`` on ``solve_soma_pose``.

- ``build_animated_glb`` writes a minimal standard glTF 2.0 binary: one
  static mesh (rest vertices, normals, top-4 skin weights), one skin over
  every public joint, one linear animation of per-joint local TRS tracks.
  Any conforming glTF loader plays it back with built-in skinning.
"""

from __future__ import annotations

import json
import struct

import numpy as np

# cskel27 (ARDY Core) joint names that differ from the SOMA public rig.
# Every other name matches exactly between the two rigs (Hips, RightArm,
# LeftForeArm, RightFoot, ...). Spine chains align by depth: Core's five
# trunk joints map onto SOMA's six (Chest absorbs Core's extra bend);
# legs map UpLeg->Leg and Leg->Shin.
_CORE_TO_SOMA = {
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Chest",
    "Spine3": "Neck1",
    "Neck": "Neck2",
    "RightUpLeg": "RightLeg",
    "LeftUpLeg": "LeftLeg",
    "RightLeg": "RightShin",
    "LeftLeg": "LeftShin",
}

# Joint counts the ARDY registry knows for humanoid rigs (g1skel34 is a
# robot rig and is rejected by the caller).
HUMANOID_JOINT_COUNTS = (27, 30, 77)

# Public index range of the trunk used for the root-orientation estimate
# (Spine1..Neck2; the head sits at index 7 and is excluded so head turns
# do not steer the body).
_TRUNK_RANGE = range(2, 7)

_IDENTITY3 = np.eye(3, dtype=np.float64)

# Gait-locked arm synthesis. ARDY Core walking output keeps the arm chain
# at its rest articulation, so sources where the arms are static while the
# legs walk get a procedural swing instead of a frozen V: the upper arm
# pitches in antiphase to the ipsilateral thigh, and the forearm carries a
# constant flexion plus a small follow-through. The effect fades as the
# source arms move on their own, so clips with real arm motion are solved
# bit-for-bit identically to before.
_ARM_SWING_GAIN = 0.55    # fraction of the ipsilateral thigh pitch
_ARM_SWING_LIMIT = 0.6    # rad clamp on the synthesized upper-arm pitch
_FOREARM_FOLLOW = 0.30    # fraction of the upper-arm pitch carried by the forearm
_FOREARM_FLEXION = 0.30   # rad constant elbow bend added when the forearm is static
_ARM_STATIC_DEG = 4.0     # below this RMS direction variation the arm counts as static
_ARM_FADE_DEG = 10.0      # fully faded out at this RMS variation
_LEG_ACTIVE_DEG = 8.0     # minimum thigh pitch amplitude (deg) before any synthesis

def soma_correspondence(source_names, soma_names, source_rig=""):
    """Map each source joint name to a SOMA public joint index (or None).

    Exact-name matching with the cskel27 aliases applied only when the
    source rig is ARDY Core; SOMA-native rigs already use the public
    names and must not be re-aliased. A target claimed by an earlier
    source joint is skipped rather than double-driven.
    """
    soma_index = {str(name): index for index, name in enumerate(soma_names)}
    aliases = _CORE_TO_SOMA if str(source_rig) == "cskel27" else {}
    mapping: list[int | None] = []
    used: set[int] = set()
    for name in source_names:
        target = aliases.get(str(name), str(name))
        index = soma_index.get(target)
        if index is None or index in used:
            mapping.append(None)
        else:
            mapping.append(index)
            used.add(index)
    return mapping


def _min_rotation(a, b):
    """Minimum rotations taking directions ``a`` onto ``b``.

    Both [F, 3]; returns [F, 3, 3]. Anti-parallel pairs take a 180-degree
    turn about an arbitrary perpendicular axis (the gauge freedom of the
    minimum solution).
    """
    a = _unit(a.astype(np.float64, copy=False))
    b = _unit(b.astype(np.float64, copy=False))
    f = a.shape[0]
    c = (a * b).sum(axis=-1)
    v = np.cross(a, b)
    s = np.linalg.norm(v, axis=-1)
    out = np.broadcast_to(_IDENTITY3, (f, 3, 3)).copy()
    regular = s > 1e-9
    if regular.any():
        w = (1.0 - c[regular]) / np.maximum(s[regular] ** 2, 1e-18)
        k = _skew(v[regular])
        out[regular] = _IDENTITY3 + k + (k @ k) * w[:, None, None]
    anti = ~regular & (c <= 0.0)
    if anti.any():
        ref_x = np.where(np.abs(a[anti, 0]) < 0.9, 1.0, 0.0)
        ref_axis = np.stack([ref_x, 1.0 - ref_x, np.zeros_like(ref_x)], axis=-1)
        axis = _unit(np.cross(a[anti], ref_axis))
        out[anti] = 2.0 * np.einsum("fi,fj->fij", axis, axis) - _IDENTITY3
    return out


def _mat_to_axis_angle(r):
    """[F, 3, 3] rotation matrices -> [F, 3] axis-angle vectors."""
    f = r.shape[0]
    theta = np.arccos(np.clip(((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]) - 1.0) / 2.0, -1.0, 1.0))
    sin_theta = np.sin(theta)
    out = np.zeros((f, 3), dtype=np.float64)
    regular = sin_theta > 1e-6
    if regular.any():
        i = np.nonzero(regular)[0]
        axis = np.stack([r[i, 2, 1] - r[i, 1, 2], r[i, 0, 2] - r[i, 2, 0], r[i, 1, 0] - r[i, 0, 1]], axis=-1)
        out[regular] = axis / (2.0 * sin_theta[regular, None]) * theta[regular, None]
    flip = sin_theta < -1e-6  # near pi: axis from the diagonal
    if flip.any():
        i = np.nonzero(flip)[0]
        diag = np.stack([r[i, 0, 0], r[i, 1, 1], r[i, 2, 2]], axis=-1) + 1.0
        pick = np.argmax(diag, axis=-1)
        axis = np.zeros((len(pick), 3))
        axis[np.arange(len(pick)), pick] = 1.0
        second = (pick + 1) % 3
        third = (pick + 2) % 3
        axis[:, second] = r[i, second, third]
        axis[:, third] = r[i, third, second]
        out[flip] = _unit(axis) * np.pi
    return out


def _mat_to_quat(r):
    """[F, 3, 3] rotation matrices -> [F, 4] quaternions (x, y, z, w)."""
    f = r.shape[0]
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    branches = np.stack([trace, r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]], axis=-1)
    pick = np.argmax(branches, axis=-1)
    out = np.zeros((f, 4), dtype=np.float64)
    for branch in range(4):
        mask = pick == branch
        if not mask.any():
            continue
        i = np.nonzero(mask)[0]
        if branch == 0:
            w = np.sqrt(np.maximum(1.0 + trace[mask], 0.0)) / 2.0
            x = (r[i, 2, 1] - r[i, 1, 2]) / (4.0 * w)
            y = (r[i, 0, 2] - r[i, 2, 0]) / (4.0 * w)
            z = (r[i, 1, 0] - r[i, 0, 1]) / (4.0 * w)
        elif branch == 1:
            x = np.sqrt(np.maximum(1.0 + r[i, 0, 0] - r[i, 1, 1] - r[i, 2, 2], 0.0)) / 2.0
            y = (r[i, 0, 1] + r[i, 1, 0]) / (4.0 * x)
            z = (r[i, 0, 2] + r[i, 2, 0]) / (4.0 * x)
            w = (r[i, 2, 1] - r[i, 1, 2]) / (4.0 * x)
        elif branch == 2:
            y = np.sqrt(np.maximum(1.0 - r[i, 0, 0] + r[i, 1, 1] - r[i, 2, 2], 0.0)) / 2.0
            x = (r[i, 0, 1] + r[i, 1, 0]) / (4.0 * y)
            z = (r[i, 1, 2] + r[i, 2, 1]) / (4.0 * y)
            w = (r[i, 0, 2] - r[i, 2, 0]) / (4.0 * y)
        else:
            z = np.sqrt(np.maximum(1.0 - r[i, 0, 0] - r[i, 1, 1] + r[i, 2, 2], 0.0)) / 2.0
            x = (r[i, 0, 2] + r[i, 2, 0]) / (4.0 * z)
            y = (r[i, 1, 2] + r[i, 2, 1]) / (4.0 * z)
            w = (r[i, 1, 0] - r[i, 0, 1]) / (4.0 * z)
        out[i] = np.stack([x, y, z, w], axis=-1)
    out /= np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12, None)
    negative = out[:, 3] < 0.0
    out[negative] *= -1.0  # non-negative w keeps linear interpolation short-arc
    return out


def _rot_about(axis, angle):
    """Batched Rodrigues rotations: [3] unit axis + [F] radians -> [F, 3, 3]."""
    axis = _unit(np.asarray(axis, dtype=np.float64))
    k = _skew(axis)
    c = np.cos(angle)[:, None, None]
    s = np.sin(angle)[:, None, None]
    return c * _IDENTITY3 + s * k + (1.0 - c) * (k @ k)


def _direction_rms_deg(directions):
    """RMS deviation (degrees) of a [F, 3] direction field from its mean.

    Returns ``inf`` when the directions cancel out (no stable mean), which
    callers treat as "not static".
    """
    directions = _unit(np.asarray(directions, dtype=np.float64))
    mean = directions.mean(axis=0)
    if float(np.linalg.norm(mean)) < 1e-6:
        return float("inf")
    cosines = np.clip((directions * _unit(mean)).sum(axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    return float(np.sqrt(np.mean(angles ** 2)))


def _apply_gait_swing(local_rot, targets, pos_cur, rot_cur, mapped, rest_pos,
                      rest_rot, parent, trunk, swing):
    """Compose a gait-locked arm swing into solved local rotations in place.

    The pelvis rest frame supplies the body axes: up follows the rest trunk
    direction, sagittal is the horizontal pelvis axis along which the knees
    vary most, and the swing rotates about the pelvis lateral axis expressed
    in each arm joint's parent bind frame, so it stays body-locked while the
    torso turns. Sides whose source arms already move (or whose legs do not)
    are skipped entirely, keeping those outputs bit-identical.
    """
    required = set(int(i) for i in swing["thighs"])
    for pair in swing["arms"]:
        required.update(int(i) for i in pair)
    if not all(joint in mapped for joint in required):
        return
    f = int(local_rot.shape[0])
    if f == 0:
        return
    pelvis_rot = rest_rot[1]
    up_dir = _unit(rest_pos[trunk] - rest_pos[1]) if trunk > 1 else _unit(rest_pos[7] - rest_pos[1])
    # The knee offsets below live in pelvis-frame coordinates, so the body
    # basis must be expressed there too: the trunk direction in pelvis
    # coordinates picks the up axis and fixes its sign. Taking the world
    # coordinate axes instead only coincides when the pelvis frame is
    # axis-aligned with the world, which SOMA's canonical rest pose is not
    # (its local X is world up); the mixed-frame dot products then make the
    # pitch gauge wind through the atan2 branch cut every stride.
    up_local = pelvis_rot.T @ up_dir
    up_axis = int(np.argmax(np.abs(up_local)))
    horiz = [i for i in range(3) if i != up_axis]
    up_f = np.eye(3)[up_axis]
    up_f *= 1.0 if float(up_local[up_axis]) >= 0.0 else -1.0
    hips_target = targets[:, mapped[1]]
    offsets = {
        "left": np.einsum("fji,fj->fi", rot_cur[:, 1],
                          targets[:, mapped[int(swing["thighs"][0])]] - hips_target),
        "right": np.einsum("fji,fj->fi", rot_cur[:, 1],
                           targets[:, mapped[int(swing["thighs"][1])]] - hips_target),
    }
    # Sagittal = horizontal axis with the largest knee variance; the other
    # horizontal axis is the lateral swing rotation axis.
    sag_axis = max(horiz, key=lambda h: offsets["left"][:, h].var() + offsets["right"][:, h].var())
    sag_f = np.eye(3)[sag_axis]
    lat_f = np.cross(sag_f, up_f)

    def pitch(v):
        return np.arctan2(v @ sag_f, v @ (-up_f))

    theta_leg = {"left": pitch(offsets["left"]), "right": pitch(offsets["right"])}
    leg_amp_deg = float(np.degrees(np.ptp(theta_leg["left"]) + np.ptp(theta_leg["right"])) / 4.0)
    axis_world = pelvis_rot @ lat_f

    for side, (upper, fore, wrist) in zip(("left", "right"), swing["arms"]):
        upper_i, fore_i, wrist_i = int(upper), int(fore), int(wrist)
        if upper_i not in mapped or fore_i not in mapped or wrist_i not in mapped:
            continue
        parent_u = int(parent[upper_i])
        # Fade gauges: the upper-arm bone (elbow about the shoulder-end joint)
        # and the forearm bone (wrist about the elbow), measured on the
        # pre-swing pose. A source arm that moves on its own fades the
        # synthesis out bit-for-bit; a resting arm stays fully synthesized.
        dir_u = np.einsum("fji,fj->fi", rot_cur[:, upper_i],
                          targets[:, mapped[fore_i]] - pos_cur[:, upper_i])
        w = float(np.clip((_ARM_FADE_DEG - _direction_rms_deg(dir_u)) /
                          (_ARM_FADE_DEG - _ARM_STATIC_DEG), 0.0, 1.0))
        w *= float(np.clip(leg_amp_deg / _LEG_ACTIVE_DEG, 0.0, 1.0))
        if w <= 0.0:
            continue
        theta_arm = np.clip(-_ARM_SWING_GAIN * theta_leg[side],
                           -_ARM_SWING_LIMIT, _ARM_SWING_LIMIT) * w
        axis_u = rest_rot[parent_u].T @ axis_world
        local_rot[:, upper_i] = _rot_about(axis_u, theta_arm) @ local_rot[:, upper_i]
        dir_f = np.einsum("fji,fj->fi", rot_cur[:, fore_i],
                          targets[:, mapped[wrist_i]] - pos_cur[:, fore_i])
        fore_w = w * float(np.clip((_ARM_FADE_DEG - _direction_rms_deg(dir_f)) /
                                   (_ARM_FADE_DEG - _ARM_STATIC_DEG), 0.0, 1.0))
        if fore_w <= 0.0:
            continue
        theta_fore = _FOREARM_FOLLOW * theta_arm + _FOREARM_FLEXION * fore_w
        axis_f = rest_rot[upper_i].T @ axis_world
        local_rot[:, fore_i] = _rot_about(axis_f, theta_fore) @ local_rot[:, fore_i]


def solve_soma_pose(targets, correspondence, rest_world, parent_ids, swing=None):
    """Convert ARDY world joint positions into SOMA pose parameters.

    Args:
        targets: [F, Jm, 3] world positions of the mapped source joints only,
            ordered like the non-None entries of ``correspondence``.
        correspondence: [Jsrc] int | None, source joint -> SOMA public index.
        rest_world: [78, 4, 4] world transforms of the layer's zero-input
            evaluation state (0 = virtual Root, 1 = Hips), e.g.
            ``SOMALayer.pose(zeros, zeros).transforms[0]``. The solved poses
            are only consumed correctly when evaluated against this exact
            rest; the fitted reposed bind is a different frame and must not
            be used here.
        parent_ids: [78] public joint parents (SOMA convention: the virtual
            Root points at itself, index 0; -1 is tolerated).
        swing: optional dict wiring the gait-locked arm synthesis to SOMA
            public indices: ``{"thighs": (left_knee, right_knee), "arms":
            ((left_upper, left_fore, left_wrist), (right_upper, right_fore,
            right_wrist))}`` -- the shoulder-end/elbow/wrist triples per side.
            When given, clips whose arms stay at rest while the legs walk
            get a procedural gait-coherent arm swing; anything that moves
            already solves exactly as before. ``None`` disables it.

    Returns:
        dict with ``poses_aa`` [F, 77, 3], ``transl`` [F, 3],
        ``local_trans`` [F, 78, 3] and ``local_quat`` [F, 78, 4]
        (float32, ready for ``SOMALayer.pose`` and glTF tracks).
    """
    f = int(targets.shape[0])
    joints = int(rest_world.shape[0])
    if joints != 78:
        raise ValueError(f"Expected 78 public SOMA joints, got {joints}.")
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    par = parent_ids.copy()
    if int(par[0]) < 0:
        par[0] = 0
    pc = np.where(par < 0, 0, par)
    for index in range(1, joints):
        if int(pc[index]) >= index:
            raise ValueError("Public joint order must be parent-first.")

    rest_pos = rest_world[..., :3, 3].astype(np.float64)          # [78,3]
    rest_rot = rest_world[..., :3, :3].astype(np.float64)         # [78,3,3]
    rest_inv = np.linalg.inv(rest_rot)
    rest_local_rot = rest_inv[pc] @ rest_rot                      # [78,3,3]
    rest_local_rot[0] = _IDENTITY3
    rest_offset = np.einsum("pab,pb->pa", rest_inv[pc], rest_pos - rest_pos[pc])
    rest_offset[0] = 0.0

    mapped = {int(correspondence[index]): index
              for index in range(len(correspondence)) if correspondence[index] is not None}
    if 1 not in mapped:
        raise ValueError("Source motion does not drive the Hips joint.")
    hips_target = targets[:, mapped[1]]                           # [F,3]

    # Root orientation: align the rest trunk direction (Hips -> deepest
    # mapped trunk joint, head excluded) with the target trunk direction.
    mapped_values = sorted(int(value) for value in correspondence if value is not None)
    trunk_candidates = [i for i in mapped_values if i in _TRUNK_RANGE]
    trunk = max(trunk_candidates) if trunk_candidates else 1
    root_rest_dir = _unit(rest_pos[trunk] - rest_pos[1])[None]
    root_target_dir = _unit(soma_target_at(mapped, trunk, targets) - hips_target)
    root_rot = _min_rotation(np.broadcast_to(root_rest_dir, (f, 3)), root_target_dir)

    pos_cur = np.broadcast_to(rest_pos[None], (f, joints, 3)).copy()
    rot_cur = np.zeros((f, joints, 3, 3), dtype=np.float64)
    local_rot = np.zeros((f, joints, 3, 3), dtype=np.float64)
    local_trans = np.zeros((f, joints, 3), dtype=np.float64)
    hips_world = root_rot @ rest_rot[1][None]
    pos_cur[:, 1] = hips_target
    rot_cur[:, 0] = _IDENTITY3
    rot_cur[:, 1] = hips_world
    local_rot[:, 0] = _IDENTITY3
    local_rot[:, 1] = hips_world
    local_trans[:, 1] = hips_target

    for joint in range(2, joints):
        parent = int(pc[joint])
        if joint in mapped:
            delta = targets[:, mapped[joint]] - pos_cur[:, parent]
            rest_length = float(np.linalg.norm(rest_pos[joint] - rest_pos[parent]))
            local_offset = np.einsum("fji,fj->fi", rot_cur[:, parent], delta)
            scale = rest_length / np.maximum(np.linalg.norm(local_offset, axis=-1), 1e-12)
            local_offset = local_offset * scale[:, None]
            relative = _min_rotation(np.broadcast_to(rest_offset[joint], (f, 3)), local_offset)
            local_rot[:, joint] = relative @ rest_local_rot[joint][None]
        else:
            local_offset = rest_offset[joint][None]
            local_rot[:, joint] = np.broadcast_to(rest_local_rot[joint], (f, 3, 3))
        rot_cur[:, joint] = rot_cur[:, parent] @ local_rot[:, joint]
        pos_cur[:, joint] = pos_cur[:, parent] + np.einsum("fij,fj->fi", rot_cur[:, parent], local_offset)
        local_trans[:, joint] = np.einsum("fij,fj->fi", rot_cur[:, parent],
                                          pos_cur[:, joint] - pos_cur[:, parent])

    if swing is not None:
        _apply_gait_swing(local_rot, targets, pos_cur, rot_cur, mapped,
                          rest_pos, rest_rot, pc, trunk, swing)

    # Emission contract: SOMALayer.pose composes O_p^-1 @ P_j @ O_j per joint
    # (apply_joint_orient_local), so emitting P_j = O_p L_j O_j^-1 makes the
    # consumed layer rotation exactly the solved local rotation L_j.
    poses_aa = np.zeros((f, 77, 3), dtype=np.float64)
    for slot, joint in enumerate(range(1, joints)):
        parent = int(pc[joint])
        relative = rest_rot[parent][None] @ local_rot[:, joint] @ rest_inv[joint][None]
        poses_aa[:, slot] = _mat_to_axis_angle(relative)

    local_quat = _mat_to_quat(local_rot.reshape(f * joints, 3, 3)).reshape(f, joints, 4)
    return {
        "poses_aa": poses_aa.astype(np.float32),
        "transl": local_trans[:, 1].astype(np.float32),
        "local_trans": local_trans.astype(np.float32),
        "local_quat": local_quat.astype(np.float32),
    }


def soma_target_at(mapped, soma_index, targets):
    """Target positions for one SOMA joint from the mapped-source layout."""
    return targets[:, mapped[soma_index]]


def build_animated_glb(vertices, normals, faces, joint_indices, joint_weights,
                       joint_rest_world, anim_trans, anim_quat, times,
                       parent_ids, material_color=(0.72, 0.74, 0.78, 1.0)):
    """Assemble a skinned, single-animation glTF 2.0 GLB.

    Args:
        vertices: [V, 3] float32 rest (bind) positions.
        normals: [V, 3] float32 rest normals.
        faces: [T, 3] integer triangle indices.
        joint_indices: [V, 4] uint8 per-vertex influence joint indices.
        joint_weights: [V, 4] float32 per-vertex influence weights.
        joint_rest_world: [J, 4, 4] float32 bind transforms in world space.
        anim_trans: [F, J, 3] float32 animated local translations.
        anim_quat: [F, J, 4] float32 animated local quaternions (xyzw).
        times: [F] float32 second timestamps.
        parent_ids: [J] joint parents (-1 for the scene root).
        material_color: baseColorFactor RGBA.

    Returns:
        GLB bytes.
    """
    v_count = int(vertices.shape[0])
    j_count = int(joint_rest_world.shape[0])
    frame_count = int(times.shape[0])
    index_dtype = np.uint16 if v_count <= 0xFFFF else np.uint32

    sections: list[dict] = []

    def add(payload, component_type, type_name, count, min_max=False):
        raw = np.ascontiguousarray(payload).tobytes()
        padded = raw + b"\0" * (-len(raw) % 4)
        info = {
            "offset": sum(len(part["data"]) for part in sections),
            "data": padded,
            "componentType": component_type,
            "type": type_name,
            "count": count,
        }
        if min_max:
            flat = payload.astype(np.float32).reshape(count, -1)
            info["min"] = [float(value) for value in flat.min(axis=0)]
            info["max"] = [float(value) for value in flat.max(axis=0)]
        sections.append(info)
        return len(sections) - 1

    acc_position = add(vertices, 5126, "VEC3", v_count, min_max=True)
    acc_normal = add(normals, 5126, "VEC3", v_count)
    acc_joint = add(joint_indices, 5121, "VEC4", v_count)
    acc_weight = add(joint_weights, 5126, "VEC4", v_count)
    acc_faces = add(faces.astype(index_dtype), 5123 if index_dtype is np.uint16 else 5125, "SCALAR", int(faces.size))
    inverse_bind = np.linalg.inv(joint_rest_world).transpose(0, 2, 1).reshape(j_count, 16)
    acc_inverse_bind = add(inverse_bind, 5126, "MAT4", j_count)
    acc_times = add(times, 5126, "SCALAR", frame_count)

    samplers = []
    channels = []
    for joint in range(j_count):
        acc_translation = add(anim_trans[:, joint], 5126, "VEC3", frame_count)
        acc_rotation = add(anim_quat[:, joint], 5126, "VEC4", frame_count)
        samplers.append({"input": acc_times, "output": acc_translation, "interpolation": "LINEAR"})
        samplers.append({"input": acc_times, "output": acc_rotation, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 2, "target": {"node": joint, "path": "translation"}})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": joint, "path": "rotation"}})

    buffer_views = []
    accessors = []
    for section in sections:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": section["offset"],
            "byteLength": len(section["data"]),
        })
        accessor = {
            "bufferView": len(accessors),
            "componentType": section["componentType"],
            "type": section["type"],
            "count": section["count"],
        }
        if "min" in section:
            accessor["min"] = section["min"]
            accessor["max"] = section["max"]
        accessors.append(accessor)

    rest_local = _local_rest_transforms(joint_rest_world, parent_ids)
    children_by_parent: dict[int, list[int]] = {}
    for joint in range(1, j_count):
        children_by_parent.setdefault(int(parent_ids[joint]), []).append(joint)
    nodes = []
    for joint in range(j_count):
        translation, rotation = _decompose_trs(rest_local[joint])
        node: dict = {}
        if bool(np.any(np.abs(translation) > 1e-8)):
            node["translation"] = [float(x) for x in translation]
        if not np.allclose(rotation, (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            node["rotation"] = [float(x) for x in rotation]
        kids = children_by_parent.get(joint, [])
        if kids:
            node["children"] = kids
        nodes.append(node)

    document = {
        "asset": {"version": "2.0", "generator": "mm-tools nvidia-sim"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": acc_position,
                    "NORMAL": acc_normal,
                    "JOINTS_0": acc_joint,
                    "WEIGHTS_0": acc_weight,
                },
                "indices": acc_faces,
                "material": 0,
                "mode": 4,
                "skin": 0,
            }]
        }],
        "materials": [{
            "name": "soma",
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(x) for x in material_color],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.85,
            },
        }],
        "skins": [{
            "name": "soma-skin",
            "joints": list(range(j_count)),
            "inverseBindMatrices": acc_inverse_bind,
        }],
        "animations": [{
            "name": "motion",
            "channels": channels,
            "samplers": samplers,
        }],
        "buffers": [{"byteLength": sum(len(part["data"]) for part in sections)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_padded = json_bytes + b"\0" * (-len(json_bytes) % 4)
    binary = b"".join(part["data"] for part in sections)
    total = 12 + 8 + len(json_padded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(json_padded), b"JSON") + json_padded
        + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    )


def _local_rest_transforms(rest_world, parent_ids):
    """World bind transforms -> local bind transforms (parent-relative)."""
    rest_world = np.asarray(rest_world, dtype=np.float64)
    count = rest_world.shape[0]
    local = np.zeros_like(rest_world)
    local[0] = rest_world[0]
    for joint in range(1, count):
        local[joint] = np.linalg.inv(rest_world[int(parent_ids[joint])]) @ rest_world[joint]
    return local


def _decompose_trs(matrix):
    """Rigid 4x4 transform -> (translation xyz, quaternion xyzw)."""
    rotation = matrix[:3, :3].astype(np.float64)
    if float(np.linalg.det(rotation)) < 0:
        rotation = rotation.copy()
        rotation[:, 0] *= -1.0
    quat = _mat_to_quat(rotation[None])[0]
    return matrix[:3, 3], quat


def decompose_trs_batch(matrices):
    """[N, 4, 4] rigid transforms -> (translation [N, 3], quaternion [N, 4] xyzw)."""
    matrices = np.asarray(matrices, dtype=np.float64)
    flat = matrices.reshape(-1, 4, 4)
    rotation = flat[:, :3, :3]
    det = np.linalg.det(rotation)
    flip = det < 0
    if flip.any():
        rotation = rotation.copy()
        rotation[flip, :, 0] *= -1.0
    quat = _mat_to_quat(rotation)
    return flat[:, :3, 3].reshape(matrices.shape[:-2] + (3,)), quat.reshape(matrices.shape[:-2] + (4,))


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def _skew(v):
    """[.., 3] -> [.., 3, 3] cross-product matrices."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    out[..., 0, 1] = -z
    out[..., 0, 2] = y
    out[..., 1, 0] = z
    out[..., 1, 2] = -x
    out[..., 2, 0] = -y
    out[..., 2, 1] = x
    return out
