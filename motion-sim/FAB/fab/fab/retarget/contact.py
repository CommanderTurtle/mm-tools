"""Contact-aware cleanup of a G1 reference motion in world space (before tracking).

Retargeted references put the G1's lowest point on the floor frame by frame, but nothing keeps the other supports
consistent: in a push-up the feet touch while the hands float, in a crouch a knee sinks into the floor, a stance foot
slides because the human -> G1 scale of the root path and of the limbs differ. Given per-frame contact labels for six
support groups (feet, hands, knees; from the human motion, no ground truth), the joint angles and the root translation
are adjusted jointly (root orientation kept) so that
  - every group in contact touches the floor (its lowest support point at z = 0) and stays put (no skate within a
    contact segment),
  - no group penetrates the floor,
  - the pose stays close to the input (joint angles, root path) and smooth (acceleration penalty).
Support points come from the robot model (scripts/build_g1.py): each foot's four sole contact spheres and its toe point
(centre - radius), the extreme mesh vertices of the rubber hand plus its thumb and pinky points, and the knee links' extreme
mesh vertices, in link coordinates.
"""

import numpy as np
import torch

GROUPS = {"lfoot": [f"left_sole_{i}" for i in range(1, 5)] + ["left_toe"],
          "rfoot": [f"right_sole_{i}" for i in range(1, 5)] + ["right_toe"],
          "lhand": ["left_rubber_hand", "left_thumb", "left_pinky"],
          "rhand": ["right_rubber_hand", "right_thumb", "right_pinky"],
          "lknee": ["left_knee_link"], "rknee": ["right_knee_link"]}


def support_points(mjcf):
    """{group: [(link, (P, 3) points in link coordinates, radius)]} from the MJCF geoms (spheres: centre + radius;
    meshes: the 6 extreme vertices along the link axes)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    out = {}
    for g, links in GROUPS.items():
        out[g] = []
        for link in links:
            b = m.body(link).id
            pts, rad = [], 0.0
            for gi in np.flatnonzero(m.geom_bodyid == b):
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[gi])
                R = R.reshape(3, 3)
                if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    pts.append(m.geom_pos[gi][None])
                    rad = float(m.geom_size[gi][0])
                elif m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[gi]
                    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
                    v = v @ R.T + m.geom_pos[gi]
                    ext = [v[v[:, k].argmin()] for k in range(3)] + [v[v[:, k].argmax()] for k in range(3)]
                    pts.append(np.stack(ext))
            if pts:
                out[g].append((link, np.concatenate(pts).astype(np.float32), rad))
    return out


def group_points(kin_pos, kin_rot, sup, g):
    """World support points [T, P, 3] of group g (sphere points lowered by their radius)."""
    ps = []
    for link, local, rad in sup[g]:
        p = kin_pos[link][:, None] + torch.einsum("tij,pj->tpi", kin_rot[link], torch.as_tensor(local, device=kin_pos[link].device))
        if rad:
            p = p - torch.tensor([0.0, 0.0, rad], device=p.device)
        ps.append(p)
    return torch.cat(ps, 1)


def contact_cleanup(kin, q0, R, t0, contact, sup, iters=300, lr=0.01, w_contact=200.0, w_skate=50.0, w_pen=2000.0,
                    w_q=1.0, w_t=20.0, w_acc=20.0):
    """q0 [T, 29], R [T, 3, 3], t0 [T, 3] (world, z-up, floor z = 0), contact {group: [T] bool} -> q, t and a report.
    Contact segments are the runs of True per group; within a segment the group's lowest support point is pulled to
    z = 0 and its horizontal position to the segment mean."""
    dev = q0.device
    lo, hi = kin.lower, kin.upper
    dq = torch.zeros_like(q0, requires_grad=True)
    dt = torch.zeros_like(t0, requires_grad=True)
    opt = torch.optim.Adam([dq, dt], lr=lr)
    segs = {}
    for g, c in contact.items():
        c = np.asarray(c, bool)
        edges = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
        segs[g] = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 3]
    masks = {g: torch.as_tensor(np.asarray(c, bool), device=dev) for g, c in contact.items()}

    def lowest(pts):
        k = pts[..., 2].argmin(-1)
        return torch.gather(pts, 1, k[:, None, None].expand(-1, 1, 3))[:, 0]

    def evaluate(q, t):
        pos, rot = kin(R, t, q)
        return {g: lowest(group_points(pos, rot, sup, g)) for g in sup}

    for _ in range(iters):
        q = torch.minimum(torch.maximum(q0 + dq, lo), hi)
        t = t0 + dt
        low = evaluate(q, t)
        loss = w_q * (dq ** 2).sum(-1).mean() + w_t * (dt ** 2).sum(-1).mean()
        x = torch.cat([q, t], -1)
        loss = loss + w_acc * ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).sum(-1).mean()
        for g, p in low.items():
            loss = loss + w_pen * (torch.relu(-p[:, 2]) ** 2).mean()
            m = masks.get(g)
            if m is None or not m.any():
                continue
            loss = loss + w_contact * (p[m, 2] ** 2).mean()
            for a, b in segs[g]:
                xy = p[a:b, :2]
                loss = loss + w_skate * ((xy - xy.mean(0, keepdim=True)) ** 2).sum(-1).mean() * (b - a) / len(q0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = torch.minimum(torch.maximum(q0 + dq, lo), hi)
        t = t0 + dt
        before, after = evaluate(q0, t0), evaluate(q, t)
        rep = {}
        for g in sup:
            m = masks.get(g, torch.zeros(len(q0), dtype=torch.bool, device=dev))
            rep[g] = {"contact_frames": int(m.sum()),
                      "height_in_contact_mm": [float(before[g][m, 2].abs().mean() * 1000) if m.any() else 0.0,
                                               float(after[g][m, 2].abs().mean() * 1000) if m.any() else 0.0],
                      "penetration_mm": [float(torch.relu(-before[g][:, 2]).max() * 1000),
                                         float(torch.relu(-after[g][:, 2]).max() * 1000)],
                      "skate_mm_per_frame": [skate(before[g], segs.get(g, [])), skate(after[g], segs.get(g, []))]}
        rep["dq_deg_mean"] = float(torch.rad2deg((q - q0).abs()).mean())
        rep["dt_mm_mean"] = float((t - t0).norm(dim=-1).mean() * 1000)
    return q.detach(), t.detach(), rep


def skate(p, segs):
    v = [(p[a + 1:b, :2] - p[a:b - 1, :2]).norm(dim=-1) for a, b in segs if b - a > 1]
    return float(torch.cat(v).mean() * 1000) if v else 0.0


HUMAN_PARTS = {"lfoot": (7, 10), "rfoot": (8, 11), "lhand": (20, 22), "rhand": (21, 23), "lknee": (4,), "rknee": (5,)}


def human_contact_labels(V, part, fps, h_contact=0.03, v_contact=0.5):
    """Contact labels of the support groups from a human mesh sequence V [T, 6890, 3] (z-up, local floor at z = 0):
    the group's lowest vertex within h_contact of the floor and its centroid slower than v_contact (m/s); part =
    the SMPL skinning-weight argmax per vertex."""
    out = {}
    for g, js in HUMAN_PARTS.items():
        idx = np.flatnonzero(np.isin(part, js))
        h = V[:, idx, 2].min(1)
        c = V[:, idx, :2].mean(1)
        v = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=-1) * fps]
        out[g] = (h < h_contact) & (v < v_contact)
    return out


def human_support_heights(V, part):
    """Height of each support group's lowest vertex above the local floor, per frame: {group: [T]} (V z-up, floor 0)."""
    return {g: V[:, np.flatnonzero(np.isin(part, js)), 2].min(1) for g, js in HUMAN_PARTS.items()}


def human_support_speeds(V, part, fps):
    """Horizontal speed (m/s) of each support group's vertex centroid per frame: {group: [T]}."""
    out = {}
    for g, js in HUMAN_PARTS.items():
        c = V[:, np.flatnonzero(np.isin(part, js)), :2].mean(1)
        out[g] = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=-1) * fps]
    return out
