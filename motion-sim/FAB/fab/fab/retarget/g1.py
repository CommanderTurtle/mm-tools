"""Unitree G1 (29 DoF) kinematics from its URDF and a batched SMPL -> G1 retargeting IK (torch, GPU).

G1Kinematics: forward kinematics of every link from a root pose (R [B, 3, 3], t [B, 3]) and the 29 revolute joint
angles in URDF order (left leg 6, right leg 6, waist 3, left arm 7, right arm 7: Unitree's order).

retarget_smpl: per-frame IK of G1 onto SMPL joints (24, any frame: the root rotation absorbs it). The human skeleton
is scaled by one factor per sample (G1 / human limb lengths), the targets are pelvis-relative, joint angles stay
inside the URDF limits (sigmoid parameterisation), the root rotation starts from the hips / spine frame. The
SMPL -> G1 correspondence (SMPL_TO_G1) is FAB's: pelvis, hips (hip pitch link), knees, ankles and toes (FAB's ankle / toe
points, scripts/build_g1.py), shoulders (shoulder roll link), elbows and hands (Unitree's rubber-hand frame).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import torch

# SMPL joint index -> G1 link or FAB frame (the URDF of scripts/build_g1.py)
SMPL_TO_G1 = {
    0: "pelvis_contour_link", 1: "left_hip_pitch_link", 2: "right_hip_pitch_link", 4: "left_knee_link",
    5: "right_knee_link", 7: "left_ankle", 8: "right_ankle",
    10: "left_toe", 11: "right_toe", 16: "left_shoulder_roll_link",
    17: "right_shoulder_roll_link", 18: "left_elbow_link", 19: "right_elbow_link", 20: "left_rubber_hand",
    21: "right_rubber_hand",
}
END_EFFECTORS = (7, 8, 10, 11, 20, 21)
BONES = [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21), (0, 16), (0, 17), (1, 2)]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def axis_angle(axis, q):
    """Rotation matrices [B, 3, 3] about a unit axis (3,) by angles q [B]."""
    x, y, z = axis
    c, s = torch.cos(q), torch.sin(q)
    C = 1 - c
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)


def rot6d_to_matrix(x):
    a, b = x[..., :3], x[..., 3:]
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


class G1Kinematics(torch.nn.Module):
    def __init__(self, urdf):
        super().__init__()
        root = ET.parse(urdf).getroot()
        joints = []
        for j in root.findall("joint"):
            if j.get("type") not in ("revolute", "fixed"):
                continue
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            lim = j.find("limit")
            joints.append({"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                           "child": j.find("child").get("link"), "xyz": xyz, "R": _rpy(*rpy),
                           "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else None,
                           "lower": float(lim.get("lower")) if lim is not None else 0.0,
                           "upper": float(lim.get("upper")) if lim is not None else 0.0})
        self.revolute = [j["name"] for j in joints if j["type"] == "revolute"]
        children = {}
        for j in joints:
            children.setdefault(j["parent"], []).append(j)
        order, stack = [], ["pelvis"]
        while stack:  # parents before children
            link = stack.pop()
            for j in children.get(link, []):
                order.append(j)
                stack.append(j["child"])
        self.order = order
        self.links = ["pelvis"] + [j["child"] for j in order]
        self.register_buffer("lower", torch.tensor([j["lower"] for j in joints if j["type"] == "revolute"]))
        self.register_buffer("upper", torch.tensor([j["upper"] for j in joints if j["type"] == "revolute"]))
        self.register_buffer("origin_t", torch.tensor([j["xyz"] for j in order]))
        self.register_buffer("origin_R", torch.tensor([j["R"] for j in order]))
        self.qi = [self.revolute.index(j["name"]) if j["type"] == "revolute" else -1 for j in order]
        self.axes = [j["axis"] for j in order]

    def forward(self, R, t, q):
        """Link positions {name: [B, 3]} and rotations {name: [B, 3, 3]} in the root's frame of reference."""
        pos, rot = {"pelvis": t}, {"pelvis": R}
        for k, j in enumerate(self.order):
            Rp, tp = rot[j["parent"]], pos[j["parent"]]
            Rc = Rp @ self.origin_R[k]
            if self.qi[k] >= 0:
                Rc = Rc @ axis_angle(self.axes[k], q[:, self.qi[k]])
            pos[j["child"]] = tp + torch.einsum("bij,j->bi", Rp, self.origin_t[k])
            rot[j["child"]] = Rc
        return pos, rot

    def rest_lengths(self):
        """G1 lengths of the BONES (zero pose)."""
        B = 1
        dev = self.lower.device
        pos, _ = self(torch.eye(3, device=dev)[None], torch.zeros(B, 3, device=dev), torch.zeros(B, 29, device=dev))
        return torch.stack([(pos[SMPL_TO_G1[a]] - pos[SMPL_TO_G1[b]]).norm(dim=-1)[0] for a, b in BONES])


def root_frame(J):
    """Initial G1 root rotation [B, 3, 3] from SMPL joints [B, 24, 3]: columns forward, left, up."""
    left = torch.nn.functional.normalize(J[:, 1] - J[:, 2], dim=-1)
    up = J[:, [16, 17]].mean(1) - J[:, 0]
    up = torch.nn.functional.normalize(up - (up * left).sum(-1, keepdim=True) * left, dim=-1)
    fwd = torch.cross(left, up, dim=-1)
    return torch.stack([fwd, left, up], -1)


def retarget_smpl(kin, J, iters=150, lr=0.08, reg=1e-3, init=None, graph=False):
    """G1 joint angles q [B, 29], root rotation [B, 3, 3] (in J's frame), scale [B] and per-sample mean residual (m,
    G1 scale) for SMPL joints J [B, 24, 3]. init: optional (q [B, 29], R [B, 3, 3]) start instead of the zero pose and
    the hips / spine frame. graph (CUDA): the iterations as a CUDA graph (fab/retarget/cuda_graph.py; bitwise the loop)."""
    B, dev = J.shape[0], J.device
    g1_len = kin.rest_lengths()
    hum_len = torch.stack([(J[:, a] - J[:, b]).norm(dim=-1) for a, b in BONES], -1)
    scale = g1_len.sum() / hum_len.sum(-1).clamp_min(1e-3)
    ids = list(SMPL_TO_G1)
    target = (J[:, ids] - J[:, :1]) * scale[:, None, None]
    w = torch.tensor([2.0 if i in END_EFFECTORS else 1.0 for i in ids], device=dev)
    lo, hi = kin.lower, kin.upper
    if init is None:
        z0 = torch.logit(((0 - lo) / (hi - lo)).clamp(0.02, 0.98))
        z = z0.expand(B, -1).clone().requires_grad_(True)
        R0 = root_frame(J)
    else:
        z = torch.logit(((init[0] - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
        R0 = init[1]
    r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
    opt = torch.optim.Adam([z, r6], lr=lr)
    zeros = torch.zeros(B, 3, device=dev)

    def loss_fn():
        q = lo + (hi - lo) * torch.sigmoid(z)
        R = rot6d_to_matrix(r6)
        pos, _ = kin(R, zeros, q)
        P = torch.stack([pos[SMPL_TO_G1[i]] for i in ids], 1)
        P = P - P[:, :1]
        err = ((P - target) ** 2).sum(-1)
        return (err * w).sum(-1).mean() + reg * (q ** 2).sum(-1).mean()

    if graph and J.is_cuda:
        from fab.retarget.cuda_graph import graph_steps

        graph_steps([z, r6], opt, loss_fn, iters)
    for _ in range(0 if graph and J.is_cuda else iters):
        loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = lo + (hi - lo) * torch.sigmoid(z)
        R = rot6d_to_matrix(r6)
        pos, _ = kin(R, zeros, q)
        P = torch.stack([pos[SMPL_TO_G1[i]] for i in ids], 1)
        res = ((P - P[:, :1]) - target).norm(dim=-1).mean(-1)
    return q.detach(), R.detach(), scale.detach(), res


def retarget_smpl_stable(kin, J, reg_first=0.03, iters_first=150, iters=100, lr=0.03, reg=1e-3, init=None):
    """Per-frame IK with a consistent choice among near-equal solutions (a label maker for single images): a strongly
    regularised solve (the redundant waist / wrist / hip-yaw angles pulled to the zero pose), then the usual weak
    regularisation started from it at a lower learning rate, which keeps the fit while staying near that choice."""
    q0, R0, _, _ = retarget_smpl(kin, J, iters=iters_first, reg=reg_first, init=init)
    return retarget_smpl(kin, J, iters=iters, lr=lr, reg=reg, init=(q0, R0))


def flexion_init(kin, J):
    """A per-frame IK start (q [B, 29], R [B, 3, 3]) with the knees and hip pitches set from the human's thigh / shank
    directions (root_frame for the root). From the zero pose the IK can lock a knee at its straight stop in seated,
    kneeling or inverted poses (the hip then swings the straight leg towards the knee target) and never bend it."""
    R0 = root_frame(J)
    q0 = torch.zeros(len(J), len(kin.revolute), device=J.device)
    for side, (hip, knee, ankle) in (("left", (1, 4, 7)), ("right", (2, 5, 8))):
        th = torch.nn.functional.normalize(J[:, knee] - J[:, hip], dim=-1)
        sh = torch.nn.functional.normalize(J[:, ankle] - J[:, knee], dim=-1)
        flex = torch.arccos((th * sh).sum(-1).clamp(-1, 1))
        fwd, up = R0[..., 0], R0[..., 2]
        pitch = -torch.atan2((th * fwd).sum(-1), -(th * up).sum(-1))  # thigh forward of the pelvis -> negative
        q0[:, kin.revolute.index(f"{side}_knee_joint")] = flex
        q0[:, kin.revolute.index(f"{side}_hip_pitch_joint")] = pitch
    q0 = torch.minimum(torch.maximum(q0, kin.lower + 0.02), kin.upper - 0.02)
    return q0, R0


def retarget_sequence(kin, J, iters=300, lr=0.03, reg=1e-3, w_vel=0.0, w_acc=2.0, init_iters=150, flex_init=False, graph=False):
    """Temporally consistent IK of a motion: SMPL joints J [T, 24, 3] (one frame of reference) -> G1 joint angles
    q [T, 29], root rotations [T, 3, 3], the sequence's scale and per-frame residual. One human -> G1 scale for the
    whole clip (median of the per-frame bone-length ratios), the per-frame IK (retarget_smpl) as the start, then all
    frames refined jointly with velocity / acceleration penalties on the joint angles and root rotation (6D): the
    per-frame IK's redundant degrees of freedom (waist, wrists, hip yaw) otherwise flicker between equal-cost
    solutions. graph (CUDA): both stages' iterations as CUDA graphs (fab/retarget/cuda_graph.py; bitwise the loops)."""
    T, dev = J.shape[0], J.device
    g1_len = kin.rest_lengths()
    hum_len = torch.stack([(J[:, a] - J[:, b]).norm(dim=-1) for a, b in BONES], -1)
    scale = (g1_len.sum() / hum_len.sum(-1).clamp_min(1e-3)).median()
    ids = list(SMPL_TO_G1)
    target = (J[:, ids] - J[:, :1]) * scale
    w = torch.tensor([2.0 if i in END_EFFECTORS else 1.0 for i in ids], device=dev)
    lo, hi = kin.lower, kin.upper
    q0, R0, _, _ = retarget_smpl(kin, J, iters=init_iters, init=flexion_init(kin, J) if flex_init else None, graph=graph)
    z = torch.logit(((q0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
    opt = torch.optim.Adam([z, r6], lr=lr)
    zeros = torch.zeros(T, 3, device=dev)

    def fk(z, r6):
        q = lo + (hi - lo) * torch.sigmoid(z)
        R = rot6d_to_matrix(r6)
        pos, _ = kin(R, zeros, q)
        P = torch.stack([pos[SMPL_TO_G1[i]] for i in ids], 1)
        return q, R, P - P[:, :1]

    def loss_fn():
        q, R, P = fk(z, r6)
        loss = (((P - target) ** 2).sum(-1) * w).sum(-1).mean() + reg * (q ** 2).sum(-1).mean()
        if T > 2:
            x = torch.cat([q, R[..., 0], R[..., 1]], -1)  # the normalised 6D (raw r6 could shrink to cheat)
            if w_vel > 0:
                loss = loss + w_vel * ((x[1:] - x[:-1]) ** 2).sum(-1).mean()
            loss = loss + w_acc * ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).sum(-1).mean()
        return loss

    if graph and J.is_cuda:
        from fab.retarget.cuda_graph import graph_steps

        graph_steps([z, r6], opt, loss_fn, iters)
    for _ in range(0 if graph and J.is_cuda else iters):
        loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        q, R, P = fk(z, r6)
        res = (P - target).norm(dim=-1).mean(-1)
    return q.detach(), R.detach(), scale.detach(), res
