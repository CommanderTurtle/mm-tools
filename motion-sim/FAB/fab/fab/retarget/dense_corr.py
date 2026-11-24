"""Dense-point -> Unitree G1 surface-anchor correspondence of the dense-point retargeter (fab-retarget's points mode; after the
learned point-cloud correspondence of UMR, Cao et al., arXiv 2609.02134, github.com/hanyang9/UMR; this one is a deterministic,
part-constrained construction).

Both bodies in a matched T-pose: SMPL's template (zero pose: arms horizontal) and the G1 at UMR's tpose_qpos (shoulder roll
+-pi/2, elbow pi/2, everything else 0), in one z-up frame (x forward, y left). Each FAB dense point (a fixed SMPL template vertex,
the model's dense_head.surface_idx) has a part: its dominant skinning joint's segment (fab.body_parts.SMPL_JOINT_TO_PART,
14 parts). Each part maps to a group of G1 visual meshes (PART_LINKS). Inside a part, points on both bodies get cylindrical
coordinates about the part's bone axis (PART_AXES: SMPL joints a -> b; G1 axis points from link origins / mesh extremes): s = the
fraction along the axis, theta = the azimuth about it from a fixed reference direction (forward, or up for forward-pointing axes).
A human point's anchor is the G1 surface sample of its group with the least (s_h - s_g)^2 + ((theta_h - theta_g) / pi)^2.
G1 surface samples: area-uniform on the visual meshes (fixed seed), stored in each body's local frame (FK-ready).

The file is built from your SMPL model, the G1 description and the FAB model (it holds SMPL template positions, so it is not
distributed); fab-retarget's points mode builds it on first use and again when the installed G1 MJCF changes:

    python -m fab.retarget.dense_corr [--ckpt <FAB model.pt>] [--mjcf <g1_29dof.xml>] [--smpl <SMPL_NEUTRAL.pkl>] [--out <npz>]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

PART_NAMES = ["head", "torso", "l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm", "l_hand", "r_hand", "l_thigh",
              "r_thigh", "l_calf", "r_calf", "l_foot", "r_foot"]  # ids 1..14 (fab.body_parts)
# part -> G1 visual meshes (body, mesh name)
PART_LINKS = {
    "head": [("torso_link", "head_link")],
    "torso": [("pelvis", "pelvis"), ("pelvis_contour_link", "pelvis_contour_link"), ("waist_yaw_link", "waist_yaw_link"),
              ("waist_roll_link", "waist_roll_link"), ("torso_link", "torso_link"), ("torso_link", "logo_link"),
              ("waist_support_link", "waist_support_link")],
}
for s, S in (("l", "left"), ("r", "right")):
    PART_LINKS.update({
        f"{s}_upper_arm": [(f"{S}_shoulder_{j}_link", f"{S}_shoulder_{j}_link") for j in ("pitch", "roll", "yaw")],
        f"{s}_forearm": [(f"{S}_elbow_link", f"{S}_elbow_link"), (f"{S}_wrist_roll_link", f"{S}_wrist_roll_link")],
        f"{s}_hand": [(f"{S}_wrist_pitch_link", f"{S}_wrist_pitch_link"), (f"{S}_wrist_yaw_link", f"{S}_wrist_yaw_link"),
                      (f"{S}_rubber_hand", f"{S}_rubber_hand")],
        f"{s}_thigh": [(f"{S}_hip_{j}_link", f"{S}_hip_{j}_link") for j in ("pitch", "roll", "yaw")],
        f"{s}_calf": [(f"{S}_knee_link", f"{S}_knee_link")],
        f"{s}_foot": [(f"{S}_ankle_pitch_link", f"{S}_ankle_pitch_link"), (f"{S}_ankle_roll_link", f"{S}_ankle_roll_link")],
    })
# the torso part's pelvis / chest points (dominant SMPL joint) -> the meshes of the rigid G1 body they move with (the waist joints
# lie between); spine1 / spine2 points use the whole torso group
TORSO_SPLIT = [((0,), [("pelvis", "pelvis"), ("pelvis_contour_link", "pelvis_contour_link")]),
               ((9, 13, 14), [("torso_link", "torso_link"), ("torso_link", "logo_link"), ("waist_support_link", "waist_support_link")])]
# part -> axis (human start, human end, G1 start, G1 end): SMPL joint ids / G1 body origins, a tuple = their midpoint.
# End segments (head, hands, feet) end at a "tip": the start moved along a shared direction (up, or the human joints
# a -> b) to the part's farthest point along it, on each body.
SH = ("left_shoulder_roll_link", "right_shoulder_roll_link")
PART_AXES = {"torso": ((1, 2), (16, 17), ("left_hip_pitch_link", "right_hip_pitch_link"), SH),
             "head": ((16, 17), ("tip", None), SH, ("tip", None))}
for s, S, o in (("l", "left", 0), ("r", "right", 1)):
    PART_AXES.update({
        f"{s}_upper_arm": (16 + o, 18 + o, f"{S}_shoulder_roll_link", f"{S}_elbow_link"),
        f"{s}_forearm": (18 + o, 20 + o, f"{S}_elbow_link", f"{S}_wrist_pitch_link"),
        f"{s}_hand": (20 + o, ("tip", (20 + o, 22 + o)), f"{S}_wrist_pitch_link", ("tip", (20 + o, 22 + o))),
        f"{s}_thigh": (1 + o, 4 + o, f"{S}_hip_pitch_link", f"{S}_knee_link"),
        f"{s}_calf": (4 + o, 7 + o, f"{S}_knee_link", f"{S}_ankle"),
        f"{s}_foot": (7 + o, ("tip", (7 + o, 10 + o)), f"{S}_ankle", ("tip", (7 + o, 10 + o))),
    })
TPOSE = {"left_shoulder_roll_joint": np.pi / 2, "left_elbow_joint": np.pi / 2, "right_shoulder_roll_joint": -np.pi / 2,
         "right_elbow_joint": np.pi / 2}  # UMR's G1 tpose_qpos
# the solver's dense-point groups: (G1 frame link, part, SMPL joints of the points' dominant skinning joint or None = the whole
# part). The frame link is the group's most distal G1 link (its rotation carries the segment's twist). Pelvis and chest split
# the torso part (the waist joints lie between); spine1 / spine2 points and the head (fixed to the G1 torso) have no group.
FRAME_GROUPS = [("pelvis", "torso", (0,)), ("torso_link", "torso", (9, 13, 14))]
for s, S in (("l", "left"), ("r", "right")):
    FRAME_GROUPS += [(f"{S}_shoulder_yaw_link", f"{s}_upper_arm", None), (f"{S}_wrist_roll_link", f"{s}_forearm", None),
                     (f"{S}_wrist_yaw_link", f"{s}_hand", None), (f"{S}_hip_yaw_link", f"{s}_thigh", None),
                     (f"{S}_knee_link", f"{s}_calf", None), (f"{S}_ankle_roll_link", f"{s}_foot", None)]
# each group's bone (SMPL joints a -> b, b a tuple = their midpoint): the solver turns the points' rotation so that the group's
# G1 bone lies along this human bone (cfg pts_align: swing from the joints, as G's bone terms; the points give the rotation about
# it); pelvis: none
# and the G1 bone it is matched to (link origins a -> b; rigid in the frame link; as fab.retarget.motion.LINK_BONES)
GROUP_AXES = {"pelvis": None, "torso_link": (0, (16, 17))}
G1_BONES = {"torso_link": ("torso_link", ("left_shoulder_roll_link", "right_shoulder_roll_link"))}
for S, o in (("left", 0), ("right", 1)):
    GROUP_AXES.update({f"{S}_shoulder_yaw_link": (16 + o, 18 + o), f"{S}_wrist_roll_link": (18 + o, 20 + o),
                       f"{S}_wrist_yaw_link": (20 + o, 22 + o), f"{S}_hip_yaw_link": (1 + o, 4 + o),
                       f"{S}_knee_link": (4 + o, 7 + o), f"{S}_ankle_roll_link": (7 + o, 10 + o)})
    G1_BONES.update({f"{S}_shoulder_yaw_link": (f"{S}_shoulder_roll_link", f"{S}_elbow_link"),
                     f"{S}_wrist_roll_link": (f"{S}_elbow_link", f"{S}_wrist_yaw_link"),
                     f"{S}_wrist_yaw_link": (f"{S}_wrist_yaw_link", f"{S}_rubber_hand"),
                     f"{S}_hip_yaw_link": (f"{S}_hip_roll_link", f"{S}_knee_link"),
                     f"{S}_knee_link": (f"{S}_knee_link", f"{S}_ankle_roll_link"),
                     f"{S}_ankle_roll_link": (f"{S}_ankle_roll_link", f"{S}_toe")})


def bone(J, ax):
    """Unit bone direction(s) of joints J (..., 24, 3) for a GROUP_AXES entry."""
    a, b = ax
    v = (J[..., list(b), :].mean(-2) if isinstance(b, tuple) else J[..., b, :]) - J[..., a, :]
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def align(R, u, v):
    """R (T, 3, 3) left-multiplied by the minimal rotation taking unit u (T, 3) to unit v (T, 3) (swing correction)."""
    w = np.cross(u, v)
    c = (u * v).sum(-1)
    K = np.zeros((len(u), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    K -= K.transpose(0, 2, 1)
    M = np.eye(3) + K + np.einsum("tij,tjk->tik", K, K) / np.maximum(1 + c, 1e-6)[:, None, None]
    return np.einsum("tij,tjk->tik", M, R)


def groups(c):
    """[(frame link, point indices into c's arrays)] of a dense_corr npz c (FRAME_GROUPS)."""
    out = []
    for link, pname, js in FRAME_GROUPS:
        m = np.asarray(c["part"]) == PART_NAMES.index(pname) + 1
        if js is not None:
            m &= np.isin(np.asarray(c["joint"]), js)
        if m.any():
            out.append((link, np.flatnonzero(m)))
    return out


def g1_tpose_bones(kin):
    """{frame link: unit G1 bone (G1_BONES) at the T-pose, world (3,)}."""
    import torch

    q = torch.zeros(1, len(kin.revolute), dtype=kin.lower.dtype, device=kin.lower.device)
    for j, v in TPOSE.items():
        q[0, kin.revolute.index(j)] = v
    pos, _ = kin(torch.eye(3, dtype=q.dtype, device=q.device)[None], torch.zeros(1, 3, dtype=q.dtype, device=q.device), q)
    pt = lambda k: np.mean([pos[x][0].double().cpu().numpy() for x in k], 0) if isinstance(k, tuple) else pos[k][0].double().cpu().numpy()  # noqa: E731
    out = {}
    for l, (a, b) in G1_BONES.items():
        v = pt(b) - pt(a)
        out[l] = v / np.linalg.norm(v)
    return out


def solver_groups(c, kin):
    """Per group (frame link, point indices, anchors in the frame link's T-pose frame (Ng, 3), the link's T-pose rotation
    (3, 3)) for the solver; kin: fab.retarget.g1.G1Kinematics (URDF frames = the MJCF bodies' frames)."""
    import torch

    q = torch.zeros(1, len(kin.revolute), dtype=kin.lower.dtype, device=kin.lower.device)
    for j, v in TPOSE.items():
        q[0, kin.revolute.index(j)] = v
    eye = torch.eye(3, dtype=q.dtype, device=q.device)[None]
    pos, rot = kin(eye, torch.zeros(1, 3, dtype=q.dtype, device=q.device), q)
    pos = {k: v[0].double().cpu().numpy() for k, v in pos.items()}
    rot = {k: v[0].double().cpu().numpy() for k, v in rot.items()}
    body = [str(b) for b in c["body"]]
    aw = np.stack([pos[b] + rot[b] @ a for b, a in zip(body, np.asarray(c["anchor_local"], np.float64))])
    return [(link, ix, (aw[ix] - pos[link]) @ rot[link], rot[link]) for link, ix in groups(c)]


def kabsch(X, P):
    """Rotations R (T, 3, 3) with P[t] - mean ~ R[t] (X - mean): X (N, 3) template points, P (T, N, 3)."""
    X0 = X - X.mean(0)
    P0 = P - P.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum("ni,tnj->tij", X0, P0))
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.tile(np.eye(3), (len(P), 1, 1))
    D[:, 2, 2] = d
    return np.einsum("tij,tjk,tkl->til", Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1))


def cyl(p, a, b):
    """(s, theta, r) of points p (N, 3) about the axis a -> b (z-up frame; azimuth reference: forward x, or up z when the axis
    points forward)."""
    u = (b - a) / np.linalg.norm(b - a)
    L = np.linalg.norm(b - a)
    ref = np.array([1.0, 0, 0]) if abs(u[0]) < 0.8 else np.array([0, 0, 1.0])
    e1 = ref - (ref @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    d = p - a
    s = d @ u / L
    q = d - np.outer(d @ u, u)
    return s, np.arctan2(q @ e2, q @ e1), np.linalg.norm(q, axis=1)


def g1_surface(mjcf, per_m2=20000, seed=0):
    """G1 visual-mesh samples at the T-pose: world points / normals, their body + mesh, body-local points / normals."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    for j, v in TPOSE.items():
        d.qpos[m.joint(j).qposadr[0]] = v
    d.qpos[2] = 0.8
    mujoco.mj_kinematics(m, d)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("p", "n", "body", "mesh", "p_loc", "n_loc")}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:  # visual meshes
            continue
        mid, b = m.geom_dataid[g], m.geom_bodyid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[v0:v0 + nv].astype(np.float64)
        F = m.mesh_face[f0:f0 + nf]
        Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        Vw = V @ Rg.T + pg
        tri = Vw[F]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(cr, axis=1)
        k = max(int(round(area.sum() * per_m2)), 1)
        fi = rng.choice(len(F), size=k, p=area / area.sum())
        r1, r2 = rng.random(k), rng.random(k)
        sq = np.sqrt(r1)
        P = (1 - sq)[:, None] * tri[fi, 0] + (sq * (1 - r2))[:, None] * tri[fi, 1] + (sq * r2)[:, None] * tri[fi, 2]
        N = cr[fi] / np.maximum(np.linalg.norm(cr[fi], axis=1, keepdims=True), 1e-12)
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        out["p"].append(P); out["n"].append(N)
        out["body"] += [m.body(b).name] * k; out["mesh"] += [m.mesh(mid).name] * k
        out["p_loc"].append((P - pb) @ Rb); out["n_loc"].append(N @ Rb)
    for k in ("p", "n", "p_loc", "n_loc"):
        out[k] = np.concatenate(out[k])
    out["body"], out["mesh"] = np.array(out["body"]), np.array(out["mesh"])
    origin = {m.body(i).name: d.xpos[i].copy() for i in range(m.nbody)}
    return out, origin


def human_tpose(smpl_pkl):
    """SMPL template vertices / joints / part per vertex, in the z-up frame (x forward, y left): SMPL (x left, y up, z forward)."""
    from bench.body_models import SMPL
    from fab.body_parts import SMPL_JOINT_TO_PART

    s = SMPL(smpl_pkl)
    V = s.v_template.double().numpy()
    W = s.weights.double().numpy()
    J = s.J_regressor.double().numpy() @ V

    part = np.asarray(SMPL_JOINT_TO_PART)[W.argmax(1)]  # 1..14
    to_z = lambda x: x[:, [2, 0, 1]]  # noqa: E731  (z_s, x_s, y_s)
    return to_z(V), to_z(J), part, W.argmax(1)


def build(surface_idx, smpl_pkl, mjcf):
    V, J, part, joint = human_tpose(smpl_pkl)
    G, origin = g1_surface(mjcf)
    hpt = lambda k: J[list(k)].mean(0) if isinstance(k, tuple) else J[k]  # noqa: E731
    gpt = lambda k: np.mean([origin[b] for b in k], 0) if isinstance(k, tuple) else origin[k]  # noqa: E731

    def tip(a, key, pts):
        u = np.array([0, 0, 1.0]) if key[1] is None else J[key[1][1]] - J[key[1][0]]
        u = u / np.linalg.norm(u)
        return a + u * ((pts - a) @ u).max()

    rows = []
    for pid, pname in enumerate(PART_NAMES, start=1):
        hi = np.array([i for i in surface_idx if part[i] == pid], int)
        if not len(hi):
            continue
        keep = np.zeros(len(G["body"]), bool)
        for body, mesh in PART_LINKS[pname]:
            keep |= (G["body"] == body) & (G["mesh"] == mesh)
        gp = G["p"][keep]
        ha, hb, ga, gb = PART_AXES[pname]
        a_h, a_g = hpt(ha), gpt(ga)
        hp = V[part == pid]  # the whole part (not only the dense ids) sets the human tip
        b_h = tip(a_h, hb, hp) if isinstance(hb, tuple) and hb[0] == "tip" else hpt(hb)
        b_g = tip(a_g, gb, gp) if isinstance(gb, tuple) and gb[0] == "tip" else gpt(gb)
        s_h, t_h, r_h = cyl(V[hi], a_h, b_h)
        s_g, t_g, r_g = cyl(gp, a_g, b_g)
        gidx = np.flatnonzero(keep)
        for k, v in enumerate(hi):
            dt = np.angle(np.exp(1j * (t_h[k] - t_g)))
            c = (s_h[k] - s_g) ** 2 + (dt / np.pi) ** 2
            sub = next((m for js, m in TORSO_SPLIT if joint[v] in js), None) if pname == "torso" else None
            if sub is not None:  # pelvis / chest points only on their own rigid G1 body's meshes
                c = np.where(np.isin(G["mesh"][gidx], [mm for _, mm in sub]) & np.isin(G["body"][gidx], [bb for bb, _ in sub]), c, np.inf)
            j = int(c.argmin())
            rows.append((int(v), pid, int(gidx[j]), float(c[j]), float(s_h[k]), float(t_h[k]), float(r_h[k]), float(s_g[j]),
                         float(t_g[j]), float(r_g[j])))
    rows.sort(key=lambda r: list(surface_idx).index(r[0]))
    g = np.array([r[2] for r in rows])
    return {"surface_idx": np.array([r[0] for r in rows]), "part": np.array([r[1] for r in rows]),
            "joint": joint[[r[0] for r in rows]], "human_tpose_joints": J.astype(np.float32),
            "body": G["body"][g], "anchor_local": G["p_loc"][g].astype(np.float32), "normal_local": G["n_loc"][g].astype(np.float32),
            "anchor_tpose": G["p"][g].astype(np.float32), "human_tpose": V[[r[0] for r in rows]].astype(np.float32),
            "cost": np.array([r[3] for r in rows], np.float32),
            "cyl_human": np.array([r[4:7] for r in rows], np.float32), "cyl_g1": np.array([r[7:10] for r in rows], np.float32)}


def default_path():
    """The correspondence file fab-retarget reads: $FAB_HOME/extras/dense_corr_g1.npz."""
    from fab import paths

    return paths.home() / "extras" / "dense_corr_g1.npz"


def surface_idx_of(ckpt):
    """The FAB model's dense surface points (SMPL template vertex ids) from its model.pt."""
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    return sd["dense_head.surface_idx"].numpy()


def file_sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stale(path, mjcf=None):
    """True when the correspondence file was built from another G1 MJCF than the installed one (or records none)."""
    from fab.retarget import g1_files

    c = np.load(path)
    return "mjcf_sha256" not in c.files or str(c["mjcf_sha256"]) != file_sha256(mjcf or g1_files.mjcf())


def make(ckpt=None, mjcf=None, smpl=None, out=None):
    """Build and save the correspondence (defaults: the released FAB-L model, the installed G1 MJCF, your SMPL neutral model,
    default_path()); returns the output path."""
    from fab import paths
    from fab.retarget import g1_files

    ckpt = str(ckpt or paths.model_file("fab-l"))
    mjcf = str(mjcf or g1_files.mjcf())
    smpl = str(smpl or paths.smpl_dir() / "SMPL_NEUTRAL.pkl")
    out = Path(out or default_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    c = build(surface_idx_of(ckpt), smpl, mjcf)
    tmp = out.with_name(f"{out.stem}.tmp{os.getpid()}.npz")  # written whole, then moved into place (no partial file on a crash)
    np.savez(tmp, **c, ckpt=ckpt, mjcf=mjcf, mjcf_sha256=file_sha256(mjcf), tpose=json.dumps({k: float(v) for k, v in TPOSE.items()}))
    os.replace(tmp, out)
    per = {PART_NAMES[i - 1]: int((c["part"] == i).sum()) for i in range(1, 15)}
    print(f"[dense_corr] {len(c['surface_idx'])} points -> {out}; per part {per}; mean cost {float(c['cost'].mean()):.4f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description="Build the dense-point -> G1 anchor correspondence (fab-retarget's points mode).")
    p.add_argument("--ckpt", default="", help="FAB model.pt (dense_head.surface_idx; default: the released FAB-L)")
    p.add_argument("--mjcf", default="", help="G1 MJCF (default: the installed g1_29dof.xml, models/fab_assets.py)")
    p.add_argument("--smpl", default="", help="SMPL_NEUTRAL.pkl (default: the installed SMPL model)")
    p.add_argument("--out", default="", help="default: $FAB_HOME/extras/dense_corr_g1.npz")
    a = p.parse_args()
    make(a.ckpt or None, a.mjcf or None, a.smpl or None, a.out or None)


if __name__ == "__main__":
    main()
