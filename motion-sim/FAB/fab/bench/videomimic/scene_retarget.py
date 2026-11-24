"""Scene-aware motion retargeting: the motion-based SMPL -> G1 retargeter (fab/retarget/motion.py, its direction-first
core) with the floor replaced by the reconstructed terrain,
for VideoMimic-style clips (stairs, slopes, benches). Same objective as retarget_motion, except that every height is
measured above the terrain under the point (a heightfield h(x, y), bilinear, differentiable):
  contact / skate  labelled support groups touch the terrain and do not slide (feet, hands, knees, and a pelvis group
                   for sitting: G1 pelvis-link extreme vertices <-> SMPL buttock vertices on the terrain)
  floor            no support point below the terrain
  world            (new) no G1 link (origin, radius r_link) below the terrain: shins / thighs / torso on stair edges,
                   seat edges (VideoMimic's world-collision cost against its heightmap)
  ground           (cfg w_ground) the lowest support point on the terrain while the human touches it
The human world and the terrain are scaled by ONE human -> G1 factor s (as the limbs and the root path in
retarget_motion), so stairs keep their proportion to the robot's legs; the G1-world terrain is written next to the motion
for the simulator.

Input: bench/videomimic/recon.py outputs (<name>.npz: joints, verts, contacts, fps; <name>_hfield.npz). --mode skeleton: the
mesh's three uses (contacts, the seated pelvis, the root start) from the joints alone (fab/retarget/skeleton.py); the input then
needs only joints, fps and frames. On CUDA the solve's iterations (and the start IK's) run as CUDA graphs
(fab/retarget/cuda_graph.py), bitwise the plain loop; cfg graph 0 runs the plain loop.
Output: <out>/<name>_motion.npz (fps 50, g1_dof, g1_joint_names, g1_root_pos, g1_root_quat_xyzw, scale, frames,
contacts at 50 fps) and <out>/<name>_terrain_g1.npz (x0, y0, res, height: the heightfield x s).

    python -m bench.videomimic.scene_retarget --recon <fab-world out>/work/joint/recon --out <dir> [--only ...]
fab-retarget (scripts/fab_retarget.py) runs it with the released configuration (fab/retarget/release_config.py).
"""

import argparse
import hashlib
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from fab.retarget.contact import GROUPS, group_points, support_points
from fab.retarget.cuda_graph import graph_steps
from fab.retarget.g1 import END_EFFECTORS, SMPL_TO_G1, G1Kinematics, retarget_sequence, rot6d_to_matrix
from fab.retarget.motion import DEFAULTS, LINK_BONES, velocity_limits

from fab import paths
from fab.retarget import g1_files

ROOT = Path(__file__).resolve().parents[2]
URDF = g1_files.urdf()
MJCF = str(g1_files.mjcf())
CORE = {"w_dir": 1.0, "dir_mode": "links", "w_rrot": 1.0, "w_pose": 0.1,  # the direction-first core's weights
        "lr_sched": "cos", "lr_t": 0.005}  # + cosine step decay / root step (less solver jitter)
SCENE = {"w_world": 1000.0, "r_link": 0.04, "w_contact_pelvis": 50.0}
HARNESS = str(g1_files.harness_mjcf())
SMPL_PKL = str(paths.smpl_dir() / "SMPL_NEUTRAL.pkl")
DENSE_CORR = str(paths.home() / "extras" / "dense_corr_g1.npz")  # fab.retarget.dense_corr (cfg w_pts; fab-retarget's points mode)
_TEMPLATE_UP_PITCH = []


def template_up_pitch():
    """hum_up "template": the pitch (rad, + = leaning back) of R_hum's up axis (spine1 - pelvis, J3 - J0,
    orthogonal to the hip axis J1 - J2) in the SMPL neutral template's rest pose (y up, facing +z): 13.8 deg. A constant of the
    body model, not fitted on data."""
    if not _TEMPLATE_UP_PITCH:
        from bench.body_models import load_smpl_pkl

        sm = load_smpl_pkl(SMPL_PKL)
        Jr = (sm["J_regressor"] @ sm["v_template"]).astype(np.float64)
        left = (Jr[1] - Jr[2]) / np.linalg.norm(Jr[1] - Jr[2])
        up = Jr[3] - Jr[0]
        up = up - (up @ left) * left
        up /= np.linalg.norm(up)
        _TEMPLATE_UP_PITCH.append(float(np.arcsin(np.clip(np.cross(left, up)[1], -1.0, 1.0))))  # the frame's forward axis, upward part
    return _TEMPLATE_UP_PITCH[0]


PT_GROUPS = ("heel", "toe", "heel_t", "toe_t")  # per-point foot groups (l / r prefix): pinned, and touching (_t)
PT_CHANNELS = ("lheel", "ltoe", "rheel", "rtoe")  # the order of the per-point label files (l_heel, l_toe, r_heel, r_toe)
SELF_ARM = [f"{s}_{l}_link" for s in ("left", "right") for l in ("shoulder_yaw", "elbow", "wrist_pitch", "wrist_yaw")]
SELF_BODY = ["pelvis", "torso_link"] + [f"{s}_{l}_link" for s in ("left", "right") for l in ("hip_roll", "hip_yaw", "knee")]
SELF_LEG = {s: [f"{s}_{l}_link" for l in ("hip_yaw", "knee", "ankle_roll")] for s in ("left", "right")}
SELF_HAND = {s: [f"{s}_{l}_link" for l in ("elbow", "wrist_pitch", "wrist_yaw")] for s in ("left", "right")}
SELF_WRIST = [f"{s}_{l}_link" for s in ("left", "right") for l in ("wrist_pitch", "wrist_yaw")]
SELF_HIP = ["pelvis"] + [f"{s}_{l}_link" for s in ("left", "right") for l in ("hip_roll", "hip_yaw", "knee")]


def self_capsules(bodies, k=5, mjcf=HARNESS):
    """The Model 15 harness MJCF's collision capsules / spheres of `bodies` as k points along each axis (body frame) +
    radius: [(body, (k, 3) points, radius)]."""
    key = ("caps", tuple(bodies), k, str(mjcf))
    if key not in _CACHE:
        _CACHE[key] = _self_capsules(bodies, k, mjcf)
    return [(b, p.copy(), r) for b, p, r in _CACHE[key]]


_CACHE = {}  # per process: results that depend only on their inputs (fab-retarget's in-process chain reuses them across steps)


def _self_capsules(bodies, k, mjcf):
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    out = []
    for b in bodies:
        for g in np.flatnonzero(m.geom_bodyid == m.body(b).id):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            hl = float(m.geom_size[g][1]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            ax = R.reshape(3, 3)[:, 2]
            pts = m.geom_pos[g][None] + np.linspace(-hl, hl, k if hl > 0 else 1)[:, None] * ax[None]
            out.append((b, pts.astype(np.float32), float(m.geom_size[g][0])))
    return out


class Terrain(torch.nn.Module):
    """Bilinear, differentiable in xy. With `levels` (multi-level terrain) and a query
    height z: each cell's highest level <= z + 0.3 m (as scene_metrics.HF)."""

    def __init__(self, hf, scale=1.0, device="cuda"):
        super().__init__()
        self.x0, self.y0, self.res = float(hf["x0"]) * scale, float(hf["y0"]) * scale, float(hf["res"]) * scale
        self.H = torch.as_tensor(np.asarray(hf["height"]) * scale, dtype=torch.float32, device=device)
        lv = np.asarray(hf["levels"]) * scale if "levels" in (hf.files if hasattr(hf, "files") else hf) else None
        lv = lv if lv is not None and lv.ndim == 3 else None  # simready `levels` are 1-D (stair levels)
        self.L = None if lv is None else torch.as_tensor(lv, dtype=torch.float32, device=device)

    def forward(self, xy, z=None):
        ny, nx = self.H.shape
        if self.L is None and xy.is_cuda and xy.dtype == torch.float32 and os.environ.get("SOLVE_TERR_FUSED", "1") != "0":
            from fab.retarget.req_kernel import AVAILABLE, terrain_fused

            if AVAILABLE:  # one fused kernel: the same values and gradient
                return terrain_fused(self.H, self.x0, self.y0, self.res, xy)
        xy = torch.nan_to_num(xy)
        u = ((xy[..., 0] - self.x0) / self.res).clamp(0, nx - 1.001)
        v = ((xy[..., 1] - self.y0) / self.res).clamp(0, ny - 1.001)
        i, j = v.floor().long(), u.floor().long()
        a, b = v - i, u - j
        if self.L is not None and z is not None:
            zq = torch.nan_to_num(z).detach()[..., None] + 0.3

            def pick(ii, jj):
                lv = self.L[ii, jj]
                h = torch.where(lv <= zq, lv, torch.full_like(lv, -1e9)).max(-1).values
                return torch.where(h > -1e8, h, torch.full_like(h, -10.0))

            c = [pick(i, j), pick(i, j + 1), pick(i + 1, j), pick(i + 1, j + 1)]
            return (1 - a) * (1 - b) * c[0] + (1 - a) * b * c[1] + a * (1 - b) * c[2] + a * b * c[3]
        H = self.H
        return (1 - a) * (1 - b) * H[i, j] + (1 - a) * b * H[i, j + 1] + a * (1 - b) * H[i + 1, j] + a * b * H[i + 1, j + 1]


def pelvis_support(mjcf, hips=False):
    """[(pelvis, (P, 3) extreme mesh vertices in link coordinates, 0)] in contact.support_points' format; hips: + the
    hip pitch / roll / yaw links' mesh vertices (every ~32nd, in their own link coordinates)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    out = []
    links = ["pelvis"] + ([f"{s}_hip_{j}_link" for s in ("left", "right") for j in ("pitch", "roll", "yaw")] if hips else [])
    for link in links:
        pts = []
        for gi in np.flatnonzero(m.geom_bodyid == m.body(link).id):
            if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[gi])
            mid = m.geom_dataid[gi]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            v = v @ R.reshape(3, 3).T + m.geom_pos[gi]
            if link == "pelvis":
                low = v[v[:, 2] < v[:, 2].min() + 0.02]  # the underside
                pts.append(low[np.linspace(0, len(low) - 1, min(len(low), 8)).astype(int)])
            else:  # a hip link rotates with the leg: no fixed underside
                pts.append(v[:: max(1, len(v) // 32)])
        if pts:
            out.append((link, np.concatenate(pts).astype(np.float32), 0.0))
    return out


def retarget_scene(kin, J, fps, contacts, sup, terr, urdf, cfg=None, links=None, heights=None, init=None, root_z0=None,
                   t_start=None, dense=None, ee_targets=None):
    """J [T, 24, 3] SMPL joints (human scale, z-up, terrain world); terr: G1-world Terrain (heights x s); contacts
    {group: [T] bool} incl. optional 'pelvis'; sup: support points (+ 'pelvis'); heights {group: [T]}: the human
    support group's lowest vertex above the terrain (human scale; w_foot_h). Terrain versions of the core's terms:
    still_sigma / still_v (stillness-adaptive smoothing of the human joints),
    w_foot_h (each sole's lowest point at s x the human foot's height above the terrain), w_swing (a foot off its
    contact labels keeps its sole >= swing_clear above the terrain). init: (q, R, s) of retarget_sequence (computed if
    None); root_z0 [T]: the root height start / prior (G1 world; default the scaled human pelvis); t_start [T, 3]: another
    start of the root path only (the priors unchanged; e.g. another retargeter's solution). ee_targets {group: ([T, 3] G1-world
    target, [T] weight)} for lhand / rhand / lfoot / rfoot: the end-effector position term of fab-retarget-ee (w_ee_hand /
    w_ee_foot; None: off). -> q, R, t, s, report."""
    c = {**DEFAULTS, **CORE, **SCENE, **(cfg or {})}
    dev = J.device
    T = len(J)
    q0, R0, s = init if init is not None else retarget_sequence(kin, J, graph=bool(c.get("graph", 1)))[:3]  # the init and the one scale (as retarget_motion's caller)
    if c["still_sigma"] > 0:  # as fab.retarget.motion.retarget_motion
        from scipy.ndimage import gaussian_filter1d

        Jn = J.detach().cpu().numpy()
        Js = gaussian_filter1d(Jn, c["still_sigma"], axis=0, mode="nearest")
        sp = np.linalg.norm(np.gradient(Js, axis=0), axis=-1) * fps
        a_ = np.clip(1.0 - sp / c["still_v"], 0.0, 1.0)[..., None]
        J = torch.as_tensor(a_ * Js + (1 - a_) * Jn, dtype=J.dtype, device=dev)
    lo, hi = kin.lower, kin.upper
    ids = list(SMPL_TO_G1)
    w = torch.tensor([2.0 if i in END_EFFECTORS else 1.0 for i in ids], device=dev)
    target_rel = (J[:, ids] - J[:, :1]) * s
    ee = [i for i in ids if i in END_EFFECTORS]
    ee_k = [ids.index(i) for i in ee]
    ee_kt = torch.tensor(ee_k, device=dev)  # the index on the device (a list index copies host -> device per call)
    hum_ee_v = (J[1:, ee] - J[:-1, ee]) * s * fps
    hum_root_v = (J[1:, 0] - J[:-1, 0]) * s * fps
    t_xy0 = J[:, 0, :2] * s
    hb = torch.stack([J[:, b] - J[:, a] for (a, b), _ in LINK_BONES] + [J[:, [16, 17]].mean(1) - J[:, 0]], 1)
    hum_dir = torch.nn.functional.normalize(hb, dim=-1)
    left = torch.nn.functional.normalize(J[:, 1] - J[:, 2], dim=-1)
    up0 = J[:, 3] - J[:, 0]
    up = torch.nn.functional.normalize(up0 - (up0 * left).sum(-1, keepdim=True) * left, dim=-1)
    R_hum = torch.stack([torch.cross(left, up, dim=-1), left, up], -1)
    if c.get("hum_up") == "template":  # the up axis turned forward about the hip axis by the template's
        # constant rest pitch, so a template-posed human gives an upright rrot target (the G1 zero pose's pelvis frame)
        th = template_up_pitch()
        up_t = float(np.cos(th)) * up + float(np.sin(th)) * torch.cross(left, up, dim=-1)
        R_hum = torch.stack([torch.cross(left, up_t, dim=-1), left, up_t], -1)
    # root start: the scaled human pelvis, its height kept above the terrain the human pelvis has (lowest-point rule
    # is not defined on stairs); the solve fixes contacts
    t0 = (J[:, 0] * s).clone()
    if root_z0 is not None:  # the lowest-point rule on the terrain (root_lowest), as retarget_motion's root_z0
        t0[:, 2] = torch.as_tensor(np.asarray(root_z0), dtype=J.dtype, device=dev)
    vmax = velocity_limits(urdf, kin.revolute).to(dev) * c["vlim_frac"]
    z = torch.logit(((q0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
    t = (t0 if t_start is None else torch.as_tensor(np.asarray(t_start), dtype=J.dtype, device=dev)).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [z, r6]}, {"params": [t], "lr": c["lr_t"] or c["lr"]}], lr=c["lr"])
    lr0 = [g["lr"] for g in opt.param_groups]
    masks = {g: torch.as_tensor(np.asarray(m, bool), device=dev) for g, m in contacts.items()}
    m_any = {g: bool(np.asarray(m, bool).any()) for g, m in contacts.items()}  # host-side, once
    m_f = {g: m.float() for g, m in masks.items()}
    m_n = {g: float(max(np.asarray(contacts[g], bool).sum(), 1)) for g in masks}
    gmode = [False]  # True while the CUDA-graph loop runs (masked means instead of boolean indexing)
    segs = {}
    for g, m in contacts.items():
        e = np.flatnonzero(np.diff(np.r_[0, np.asarray(m, int), 0]))
        segs[g] = [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= 3]
    links = links or [l for l in kin.links if l != "pelvis"]
    if not c.get("world_support", 1):  # world_support 0: the link-clearance term skips the support groups' own links (sole
        # spheres, hand, knee links) and the ankle-roll links: the floor / contact terms hold those on their geometry, and a
        # 4 cm clearance on a sole sphere lifts a planted foot against its contact term
        own = {l for g in sup.values() for l, _, _ in g} | {"left_ankle_roll_link", "right_ankle_roll_link"}
        links = [l for l in links if l not in own]
    w_link = None
    if c.get("world_support", 1) == "contact":  # the support links keep the clearance except on the frames their group is
        # labelled in contact (the contact term holds them on the surface there); feet: the sole points + the ankle roll
        w_link = torch.ones(T, len(links), device=dev)
        for g, m in masks.items():
            own = {l for l, _, _ in sup.get(g, [])} | ({"left_ankle_roll_link", "right_ankle_roll_link"} if g == "anyfoot" else
                                                     {f"{'left' if g[0] == 'l' else 'right'}_ankle_roll_link"} if g.endswith("foot") else set())
            for k_, l in enumerate(links):
                if l in own:
                    w_link[m, k_] = 0.0
    hts = {g: torch.as_tensor(np.asarray(v, np.float32), device=dev) for g, v in (heights or {}).items()}
    if c.get("w_self", 0) > 0:  # harness MJCF capsules: arm (shoulder yaw .. wrist) vs pelvis / torso / hip / knee; with
        # self_legs, also left vs right leg (hip yaw, knee, ankle / foot) and left vs right forearm / hand
        pairs = [(self_capsules(SELF_ARM), self_capsules(SELF_BODY))]
        if c.get("self_legs", 0):
            pairs += [(self_capsules(SELF_LEG["left"]), self_capsules(SELF_LEG["right"])),
                      (self_capsules(SELF_HAND["left"]), self_capsules(SELF_HAND["right"]))]
        margins = [c.get("self_margin", 0.01)] * len(pairs)
        arm_j = [i for i, n_ in enumerate(kin.revolute) if any(k in n_ for k in ("shoulder", "elbow", "wrist"))]
        if c.get("self_margin_hand", 0):  # hands (wrist pitch / yaw) kept visibly clear of the hips / thighs
            pairs.append((self_capsules(SELF_WRIST), self_capsules(SELF_HIP)))
            margins.append(float(c["self_margin_hand"]))

        _caps_t = {}

        def caps_world(pos, rot, caps):  # the capsule points / radii on the device once per capsule set
            key = id(caps)
            if key not in _caps_t:
                _caps_t[key] = ([torch.as_tensor(p, device=dev) for _, p, _ in caps],
                                torch.cat([torch.full((len(p),), r_, device=dev) for _, p, r_ in caps]))
            pts_, r = _caps_t[key]
            P = torch.cat([pos[b][:, None] + torch.einsum("tij,pj->tpi", rot[b], pt) for (b, _, _), pt in zip(caps, pts_)], 1)
            return P, r

        with torch.no_grad():  # pairs already touching in the Model 15 default pose (shoulder on torso) are not penalised
            from fab.retarget.g1_model15 import DEFAULT as dflt

            q_d = torch.tensor([[dflt[n] for n in kin.revolute]], device=dev, dtype=torch.float32)
            pd, rd = kin(torch.eye(3, device=dev)[None], torch.zeros(1, 3, device=dev), q_d)
            self_mask = []
            for (arm, body), mg in zip(pairs, margins):
                Pa, ra = caps_world(pd, rd, arm)
                Pb, rb = caps_world(pd, rd, body)
                # only pairs already touching (1 cm) in the default pose are exempt, also for a larger margin
                self_mask.append(((torch.cdist(Pa, Pb)[0] - ra[:, None] - rb[None]) > c.get("self_margin", 0.01)).float())

    self_pos = [None, None]  # the last forward's link poses (the self-collision / ZMP terms)
    dgr = []
    if dense is not None and c.get("w_pts", 0) > 0:  # dense-point term: each dense-point group's G1 frame link turned as the
        # group's human points turn from the matched T-pose (Kabsch), measured at the group's G1 surface anchors in that link
        # (fab.retarget.dense_corr: correspondence, groups); proportion-free (rotations only, G1 lever arms)
        # pts_align (default 1): each group's bone turned onto the skeleton's (swing from the joints, as the bone terms; the points
        # give the rotation about it and the pelvis frame)
        from fab.retarget.dense_corr import GROUP_AXES, align, bone, g1_tpose_bones, kabsch, solver_groups

        Pd, dcorr = dense
        assert len(Pd) == T, (len(Pd), T)
        Xt = np.asarray(dcorr["human_tpose"], np.float64)
        Jn_ = J.detach().cpu().numpy().astype(np.float64)
        bT_ = g1_tpose_bones(kin)
        for link, ix, a_l, rT in solver_groups(dcorr, kin):
            Rh_ = kabsch(Xt[ix], np.asarray(Pd, np.float64)[:, ix])
            if c.get("pts_align", 1) and GROUP_AXES.get(link) is not None:  # the G1 bone (T-pose, turned by Rh_) onto the human's
                Rh_ = align(Rh_, np.einsum("tij,j->ti", Rh_, bT_[link]), bone(Jn_, GROUP_AXES[link]))
            dgr.append((link, torch.as_tensor(Rh_ @ rT, dtype=J.dtype, device=dev), torch.as_tensor(a_l.T, dtype=J.dtype, device=dev)))
    if c.get("w_zmp", 0) > 0:  # cart-table ZMP of the Model 15 masses inside the stance support (the foot capsules)
        import mujoco

        mh = mujoco.MjModel.from_xml_path(HARNESS)
        names = {mh.body(i).name: i for i in range(mh.nbody)}
        zl = [l for l in kin.links if l in names]
        zm = torch.tensor([float(mh.body_mass[names[l]]) for l in zl], device=dev)
        zi = torch.tensor(np.stack([mh.body_ipos[names[l]] for l in zl]), dtype=torch.float32, device=dev)
        with torch.no_grad():  # each foot's capsule: its two most distant support points (heel / toe) at the init pose
            p0, r0 = kin(R0[:1], (J[:1, 0] * s), q0[:1])
            ends = {}
            for g in ("lfoot", "rfoot"):
                P = group_points(p0, r0, sup, g)[0]
                dd = torch.cdist(P[:, :2], P[:, :2])
                k = int(dd.argmax())
                ends[g] = (k // len(P), k % len(P))
        stance = {g: masks[g] for g in ("lfoot", "rfoot") if g in masks}
        with torch.no_grad():  # cart-table offset -h a_xy / (g + a_z) of the human's CoM (joint mean, x s, 0.1 s smoothing)
            from scipy.ndimage import gaussian_filter1d

            ch = gaussian_filter1d((J.mean(1) * s).cpu().numpy().astype(np.float64), 0.1 * fps, axis=0, mode="nearest")
            ah = np.gradient(np.gradient(ch, axis=0), axis=0) * fps * fps
            hh = np.clip(ch[:, 2] - terr(torch.as_tensor(ch[:, :2], dtype=torch.float32, device=dev),
                                         torch.as_tensor(ch[:, 2] - 0.6, dtype=torch.float32, device=dev)).cpu().numpy(), 0.2, None)
            zmp_off = torch.as_tensor(-hh[:, None] * ah[:, :2] / np.clip(9.81 + ah[:, 2:3], 1.0, None), dtype=torch.float32, device=dev)
            zmp_off = zmp_off.clamp(-0.3, 0.3) * float(c.get("zmp_dyn", 1.0))  # zmp_dyn 0: the static CoM over the support

        def seg_dist(x, a, b):
            ab = b - a
            u = (((x - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0, 1)
            return (x - (a + u[..., None] * ab)).norm(dim=-1)

    # out-and-back root spikes: the bilinear heightfield's xy slope is piecewise constant and huge at
    # scan risers / clutter, and it kicked single frames sideways. terr_xy_grad 0: the terrain terms act vertically only;
    # terr_grad_sigma > 0: heights from the grid, xy gradients from the grid blurred by that many cells (straight-through;
    # terr_grad_sigma_m: the blur in metres, the same on every grid);
    # world_xy_grad 0: only the link-clearance (world) term acts vertically
    gs = float(c.get("terr_grad_sigma", 0) or 0) or float(c.get("terr_grad_sigma_m", 0) or 0) / terr.res  # cells, or metres (G1 world)
    if gs > 0 and terr.L is None:
        from scipy.ndimage import gaussian_filter

        terr_s = Terrain({"x0": terr.x0, "y0": terr.y0, "res": terr.res,
                          "height": gaussian_filter(terr.H.cpu().numpy().astype(np.float64), gs, mode="nearest")}, 1.0, device=dev)
    else:
        terr_s = None

    def hgt(p, xy_grad=True):
        xy = p[..., :2] if (c.get("terr_xy_grad", 1) and xy_grad) else p[..., :2].detach()
        if terr_s is not None and xy.requires_grad:
            hs = terr_s(xy, p[..., 2])
            return p[..., 2] - (terr(xy.detach(), p[..., 2]) + hs - hs.detach())
        return p[..., 2] - terr(xy, p[..., 2])

    pts = bool(c.get("points"))  # heel / toe per point
    # feet_gate: the feet groups' terrain terms (floor, contact, skate, foot_h, swing) on the
    # stance pin's foot geometry and surface: the gate MJCF's foot capsule sample points (stance_pin.collision_points, the pin's
    # foot_k / cap_eps) with their radii, each point's height = its centre z - stance_pin.Surface.req (the lowest centre that clears
    # the package grid max the gate's slot grid within the point's footprint); the slot from the scaled human pelvis path; the
    # xy gradient through the same blur as hgt (straight-through on the terr_grad_sigma blurred grid)
    gfeet, low_r = {}, {}
    if c.get("feet_gate"):
        from bench.stance_pin import GATE_XML, MjFK, Surface, collision_points, surfaces

        assert terr.L is None and not pts, "feet_gate: single-level terrain, no per-point groups"
        fkg = MjFK(GATE_XML, list(kin.revolute), "cpu")
        Bc, Pc, Rc, _ = collision_points(fkg, k_foot=int(c.get("gate_foot_k", 20)), eps=float(c.get("gate_cap_eps", 0.001)))
        for g_, f_ in (("lfoot", "left_ankle_roll_link"), ("rfoot", "right_ankle_roll_link")):
            sel = Bc == fkg.names.index(f_)
            gfeet[g_] = (f_, torch.as_tensor(Pc[sel], device=dev), torch.as_tensor(Rc[sel], device=dev))
        rxy = t_xy0.detach().cpu().numpy().astype(np.float64)
        tgd = {"height": terr.H.cpu().numpy().astype(np.float64), "x0": terr.x0, "y0": terr.y0, "res": terr.res}
        gsurf = Surface([surfaces(tgd, rxy)], dev)
        gsurf_s = Surface([surfaces(dict(tgd, height=terr_s.H.cpu().numpy().astype(np.float64)), rxy)], dev) if terr_s is not None else None

    def gate_gap(p, r, xy_grad=True):  # p (T, n, 3) foot point centres, r (n,) or (T, n) radii -> (T, n)
        cid = torch.zeros(p.shape[:-1], dtype=torch.long, device=dev)
        r = r.expand(p.shape[:-1])
        xy = p[..., :2] if (c.get("terr_xy_grad", 1) and xy_grad) else p[..., :2].detach()
        if gsurf_s is not None and xy.requires_grad:
            hs = gsurf_s.req(xy, r, cid)
            return p[..., 2] - (gsurf.req(xy.detach(), r, cid) + hs - hs.detach())
        return p[..., 2] - gsurf.req(xy, r, cid)

    def terr_normal(xy, d=0.02):  # the (blurred, when terr_s) terrain's unit normal at xy, by central differences
        tr = terr_s if terr_s is not None else terr
        ex, ey = torch.tensor([d, 0.0], device=xy.device), torch.tensor([0.0, d], device=xy.device)
        gx = (tr(xy + ex) - tr(xy - ex)) / (2 * d)
        gy = (tr(xy + ey) - tr(xy - ey)) / (2 * d)
        return torch.nn.functional.normalize(torch.stack([-gx, -gy, torch.ones_like(gx)], -1), dim=-1)

    _sup_t = {}

    def gpoints(pos, rot, g):  # fab.retarget.contact.group_points with its locals / radius offsets on the device once
        if g not in _sup_t:
            _sup_t[g] = [(link, torch.as_tensor(local, device=dev), torch.tensor([0.0, 0.0, rad], device=dev) if rad else None)
                         for link, local, rad in sup[g]]
        ps = []
        for link, lt, off in _sup_t[g]:
            p_ = pos[link][:, None] + torch.einsum("tij,pj->tpi", rot[link], lt)
            ps.append(p_ - off if off is not None else p_)
        return torch.cat(ps, 1)

    eet = None if ee_targets is None else {g: (torch.as_tensor(tg, dtype=torch.float32, device=dev),
                                                torch.as_tensor(wk, dtype=torch.float32, device=dev)) for g, (tg, wk) in ee_targets.items()}
    _ee_t = {}

    def ee_point(pos, rot, g):  # fab-retarget-ee's end-effector point of a support group (G1 world)
        if g not in _ee_t:
            sel = ("_thumb", "_pinky") if g.endswith("hand") else tuple(f"_sole_{i}" for i in range(1, 5))
            _ee_t[g] = [(link, torch.as_tensor(local, device=dev), torch.tensor([0.0, 0.0, rad], device=dev) if rad else None)
                        for link, local, rad in sup[g] if link.endswith(sel)]
        ps = []
        for link, lt, off in _ee_t[g]:
            p_ = pos[link][:, None] + torch.einsum("tij,pj->tpi", rot[link], lt)
            ps.append(p_ - off if off is not None else p_)
        return torch.cat(ps, 1).mean(1)

    def lowest(p):
        k = hgt(p).argmin(-1)
        return torch.gather(p, 1, k[:, None, None].expand(-1, 1, 3))[:, 0]

    def forward(z, r6, t):
        q = lo + (hi - lo) * torch.sigmoid(z)
        R = rot6d_to_matrix(r6)
        pos, rot = kin(R, t, q)
        P = torch.stack([pos[SMPL_TO_G1[i]] for i in ids], 1)
        low = {g: lowest(gpoints(pos, rot, g)) for g in sup}
        for g, (f_, pf, rf) in gfeet.items():  # feet_gate: the foot point that is lowest above the pin's surface, and its radius
            Pg = pos[f_][:, None] + torch.einsum("tij,pj->tpi", rot[f_], pf)
            k = gate_gap(Pg.detach(), rf).argmin(-1)
            low[g] = torch.gather(Pg, 1, k[:, None, None].expand(-1, 1, 3))[:, 0]
            low_r[g] = rf[k]
        gb = torch.stack([pos[lb] - pos[la] for _, (la, lb) in LINK_BONES] +
                         [(pos["left_shoulder_roll_link"] + pos["right_shoulder_roll_link"]) / 2 - pos["pelvis"]], 1)
        Lp = torch.stack([pos[l] for l in links], 1)
        self_pos[:] = [pos, rot]
        return q, R, P, low, torch.nn.functional.normalize(gb, dim=-1), Lp

    def terms(q, R, P, low, ub, Lp, t):
        fx = bool(c.get("feet_xy_grad", 1))  # feet_xy_grad 0: the feet's terrain terms (floor, contact, foot_h, swing) vertical only
        hg = lambda g, p_: hgt(p_, fx or g not in ("lfoot", "rfoot"))  # noqa: E731
        if gfeet:  # feet_gate: the feet groups' heights above the pin's surface, each at its lowest point's radius
            hg = lambda g, p_: (gate_gap(p_[:, None], low_r[g][:, None], fx)[:, 0] if g in gfeet  # noqa: E731
                                else hgt(p_, fx or g not in ("lfoot", "rfoot")))
        e = {"dir": ((ub - hum_dir) ** 2).sum(-1).sum(-1).mean(), "rrot": ((R - R_hum) ** 2).sum((-1, -2)).mean(),
             "pose": (((P - P[:, :1] - target_rel) ** 2).sum(-1) * w).sum(-1).mean(),
             "vel": (((P[1:, ee_kt] - P[:-1, ee_kt]) * fps - hum_ee_v) ** 2).sum(-1).mean(),
             "root_vel": (((t[1:] - t[:-1]) * fps - hum_root_v) ** 2).sum(-1).mean(),
             "floor": sum((torch.relu(-hg(g, p)) ** 2).mean() for g, p in low.items()),
             "world": ((torch.relu(c["r_link"] - hgt(Lp, bool(c.get("world_xy_grad", 1)))) ** 2) * (1.0 if w_link is None else w_link)).mean()}
        con, sk, conp = 0.0, 0.0, 0.0
        for g, p in low.items():
            m = masks.get(g)
            if m is None or not m_any[g]:
                continue
            if pts and g in ("lfoot", "rfoot"):  # points: the heel / toe groups below hold the foot instead
                continue
            wp = 0.5 if pts and g[1:] in PT_GROUPS else 1.0  # per point: half the foot's weight (two points per foot)
            if g.endswith("_t") and not c.get("touch_h", 1):  # touch_h 0: touching drives the sole alignment only (no height pull)
                continue
            h2 = (hg(g, p)[m] ** 2).mean() if not gmode[0] else ((hg(g, p) ** 2) * m_f[g]).sum() / m_n[g]
            if g == "pelvis":
                conp = conp + h2
            else:
                con = con + wp * h2
            if g.endswith("_t"):  # touching only: on the terrain, free to roll / slide (pitch follows the human foot)
                continue
            if g == "anyfoot":  # --ground_aware's no-flight group: the lower foot on the terrain, height only
                continue
            for a, b in segs[g]:
                xy = p[a:b, :2]
                sk = sk + wp * ((xy - xy.mean(0, keepdim=True)) ** 2).sum(-1).mean() * (b - a) / T
        e["contact"], e["skate"], e["contact_pelvis"] = con, sk, conp
        if pts and c.get("w_sole", 0) > 0:  # heel and toe both touching: the sole (ankle-roll z) along the local terrain normal
            es = 0.0
            for side, link in (("l", "left_ankle_roll_link"), ("r", "right_ankle_roll_link")):
                both = masks.get(side + "heel_t")
                if both is None:
                    continue
                both = both & masks[side + "toe_t"]
                if both.any():
                    n_s = self_pos[1][link][both][:, :, 2]
                    es = es + (1.0 - (n_s * terr_normal(self_pos[0][link][both][:, :2].detach())).sum(-1)).mean()
            e["sole"] = es
        if not pts and c.get("points_sole_only") and c.get("w_sole", 0) > 0:  # points_sole_only: the same sole term on the
            # per-point touching labels alone; the feet's contact / skate / swing / clearance and the pins keep their own labels
            es = 0.0
            for side, link in (("l", "left_ankle_roll_link"), ("r", "right_ankle_roll_link")):
                both = masks.get(side + "heel_t")
                if both is None:
                    continue
                both = both & masks[side + "toe_t"]
                if both.any():
                    n_s = self_pos[1][link][both][:, :, 2]
                    es = es + (1.0 - (n_s * terr_normal(self_pos[0][link][both][:, :2].detach())).sum(-1)).mean()
            e["sole"] = es
        if c["w_foot_h"] > 0 and hts:
            e["foot_h"] = sum(((hg(g, low[g]) - s * hts[g]) ** 2).mean() for g in ("lfoot", "rfoot") if g in low and g in hts)
        if c["w_swing"] > 0:
            e["swing"] = sum((torch.relu(c["swing_clear"] - hg(g, low[g])[~masks[g]]) ** 2).mean()
                             for g in ("lfoot", "rfoot") if g in low and g in masks and (~masks[g]).any())
        if c.get("w_self", 0) > 0:
            e["self"] = 0.0
            for k_, ((arm, body), mk, mg) in enumerate(zip(pairs, self_mask, margins)):
                Pa, ra = caps_world(*self_pos, arm)
                Pb, rb = caps_world(*self_pos, body)
                if c.get("self_margin_hand", 0) and k_ == len(pairs) - 1:  # the hand margin moves the arm joints only:
                    # through the root / waist / legs it rolled the stance foot into the ground
                    qm = q.detach().clone()
                    qm[:, arm_j] = q[:, arm_j]
                    Pa, ra = caps_world(*kin(R.detach(), t.detach(), qm), arm)
                    Pb = Pb.detach()
                dd = torch.cdist(Pa, Pb) - ra[:, None] - rb[None]
                e["self"] = e["self"] + ((torch.relu(mg - dd) ** 2) * mk).sum((-1, -2)).mean()
        if c.get("w_zmp", 0) > 0 and stance:
            pos_, rot_ = self_pos
            Pl = torch.stack([pos_[l] for l in zl], 1) + torch.einsum("tlij,lj->tli", torch.stack([rot_[l] for l in zl], 1), zi)
            com = (Pl * zm[None, :, None]).sum(1) / zm.sum()
            zmp = com[:, :2] + zmp_off  # the dynamic offset comes from the human's own CoM path (fixed): the term moves
            # the G1's CoM over its support and never creates accelerations (an offset from the G1's own CoM diverges)
            dist, fc = [], {}
            for g, m_ in stance.items():
                Pg = group_points(pos_, rot_, sup, g)
                a_, b_ = Pg[:, ends[g][0], :2], Pg[:, ends[g][1], :2]
                fc[g] = (a_ + b_) / 2
                dist.append(torch.where(m_, seg_dist(zmp, a_, b_) - c.get("zmp_r", 0.04), torch.full_like(zmp[:, 0], 1e3)))
            if len(fc) == 2:  # double support: the band between the feet
                both = stance["lfoot"] & stance["rfoot"]
                dist.append(torch.where(both, seg_dist(zmp, fc["lfoot"], fc["rfoot"]) - c.get("zmp_bridge_r", 0.08),
                                        torch.full_like(zmp[:, 0], 1e3)))
            dmin = torch.stack(dist, 0).min(0).values
            on = dmin < 1e2
            e["zmp"] = (torch.relu(dmin[on]) ** 2).mean() if on.any() else torch.zeros((), device=dev)
        if dgr:  # dense-point term: anchors of each group in its frame link, current rotation vs the points' target rotation (m^2)
            e["pts"] = sum(((torch.matmul(self_pos[1][l_] - Rt_, a_) ** 2).sum(1)).mean() for l_, Rt_, a_ in dgr) / len(dgr)
        e["vlim"] = (torch.relu((q[1:] - q[:-1]).abs() * fps - vmax) ** 2).sum(-1).mean()
        x = torch.cat([q, R[..., 0], R[..., 1], t], -1)
        e["acc"] = ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).sum(-1).mean()
        e["root_xy"] = ((t[:, :2] - t_xy0) ** 2).sum(-1).mean()
        e["root_z"] = ((t[:, 2] - t0[:, 2]) ** 2).mean()
        if eet is not None:  # fab-retarget-ee: squared distance of each end-effector point to its target, weighted mean over frames
            for kind in ("hand", "foot"):
                gs = [g for g in eet if g.endswith(kind)]
                if gs:
                    e["ee_" + kind] = sum((((ee_point(self_pos[0], self_pos[1], g) - eet[g][0]) ** 2).sum(-1) * eet[g][1]).sum()
                                          / eet[g][1].sum().clamp_min(1.0) for g in gs) / len(gs)
        return e

    keys = {"pose": "w_pose", "vel": "w_vel", "root_vel": "w_root_vel", "floor": "w_floor", "contact": "w_contact",
            "skate": "w_skate", "vlim": "w_vlim", "acc": "w_acc", "root_xy": "w_root_xy", "root_z": "w_root_z",
            "dir": "w_dir", "rrot": "w_rrot", "world": "w_world", "contact_pelvis": "w_contact_pelvis",
            "foot_h": "w_foot_h", "swing": "w_swing", "self": "w_self", "zmp": "w_zmp", "sole": "w_sole", "pts": "w_pts",
            "ee_hand": "w_ee_hand", "ee_foot": "w_ee_foot"}
    with torch.no_grad():
        e0 = {k: float(v) for k, v in terms(*forward(z, r6, t), t).items()}
    # the CUDA-graph path (fab/retarget/cuda_graph.py; cfg graph 0: the plain loop below): forward, terms and backward captured
    # once and replayed, the Adam step eager with the same python learning rates, the contact means as masked sums (no
    # boolean indexing inside the graph): bitwise the plain loop. Terms with host-side or data-dependent steps stay on the plain loop.
    graph_ok = bool(c.get("graph", 1)) and str(dev).startswith("cuda") and not c.get("w_zmp", 0) and not c.get("w_sole", 0) \
        and not c.get("feet_gate") and not (c.get("self_margin_hand", 0) and c.get("w_self", 0) > 0) \
        and not c.get("w_swing", 0) and not c.get("points")
    if graph_ok:
        def set_lr(it_):
            if c["lr_sched"] == "cos":
                for g, l0 in zip(opt.param_groups, lr0):
                    g["lr"] = l0 * 0.5 * (1 + np.cos(np.pi * it_ / int(c["iters"])))

        gmode[0] = True
        graph_steps([z, r6, t], opt, lambda: sum(c[keys[k]] * v for k, v in terms(*forward(z, r6, t), t).items()), int(c["iters"]),
                    pre=set_lr)
        gmode[0] = False
    for it_ in range(0 if graph_ok else int(c["iters"])):
        if c["lr_sched"] == "cos":
            for g, l0 in zip(opt.param_groups, lr0):
                g["lr"] = l0 * 0.5 * (1 + np.cos(np.pi * it_ / int(c["iters"])))
        out = forward(z, r6, t)
        e = terms(*out, t)
        loss = sum(c[keys[k]] * v for k, v in e.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        q, R, P, low, ub, Lp = forward(z, r6, t)
        e1 = {k: float(v) for k, v in terms(q, R, P, low, ub, Lp, t).items()}
        res = (P - P[:, :1] - target_rel).norm(dim=-1).mean(-1)
    return q.detach(), R.detach(), t.detach(), s.detach(), {"before": e0, "after": e1, "residual_mm": float(res.mean() * 1000),
                                                             "cfg": {k: v for k, v in c.items() if not callable(v)}, "graph": bool(graph_ok)}


def export_scene(recon_npz, out, s):
    """The clip's scene mesh in the G1 world (x s) for simulators that take a mesh; an empty scene (ViPE depth failure)
    becomes a 20 m flat square at z = 0."""
    import trimesh

    scene = Path(recon_npz[:-4] + "_scene.ply")
    tm = trimesh.load(scene, process=False, force="mesh") if scene.exists() else None
    if tm is None or not hasattr(tm, "vertices") or len(tm.faces) == 0:
        tm = trimesh.Trimesh(np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0.0]]) / s,
                             np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    tm.vertices = np.asarray(tm.vertices) * s
    tm.export(out)


def export_terrain_mesh(terr, out):
    """The G1-world heightfield the motion was retargeted on, as a triangulated grid mesh (for simulators that take a
    mesh: the same terrain as the retargeting)."""
    import trimesh

    H = terr.H.cpu().numpy().astype(np.float64)
    ny, nx = H.shape
    X, Y = np.meshgrid(terr.x0 + terr.res * np.arange(nx), terr.y0 + terr.res * np.arange(ny))
    V = np.stack([X.ravel(), Y.ravel(), H.ravel()], 1)
    i = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None]).ravel()
    F = np.concatenate([np.stack([i, i + 1, i + nx + 1], 1), np.stack([i, i + nx + 1, i + nx], 1)])
    trimesh.Trimesh(V, F, process=False).export(out)


def resample(x, n_out, fps_in, fps_out):
    tq = np.arange(n_out) / fps_out * fps_in
    i0 = np.clip(np.floor(tq).astype(int), 0, len(x) - 1)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    a = (tq - i0).reshape(-1, *([1] * (x.ndim - 1)))
    return (1 - a) * x[i0] + a * x[i1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="mesh", choices=("mesh", "skeleton"),
                   help="mesh: contacts / seated pelvis / root start from the body mesh; skeleton: from the joints alone")
    p.add_argument("--only", default="")
    p.add_argument("--cfg", default="", help="JSON overrides of the solver weights")
    p.add_argument("--h_pelvis", type=float, default=0.05, help="buttock vertices within this of the terrain: seated")
    p.add_argument("--sigma_in", type=float, default=0.0, help="Gaussian smoothing (frames at 50 fps) of the human joints")
    p.add_argument("--clean", action="store_true", help="level-snap the terrain first (recon.clean_heightfield)")
    p.add_argument("--sigma_root_xy", type=float, default=0.0,
                   help="extra Gaussian smoothing (frames at 50 fps) of the pelvis path in the ground plane")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--terrain_solve", default="", help="solve on a clean terrain instead (<dir>/<clip>/"
                   "terrain_hfield_2cm_clean.npz, package frame, - the package dz); the outputs keep the recon terrain")
    p.add_argument("--pkg", default="", help="the packages' clips dir (<pkg>/<clip>/package.json dz_m), with --terrain_solve")
    p.add_argument("--terrain_file", default="terrain_hfield_2cm_clean.npz", help="the file name under <terrain_solve>/<clip>/")
    p.add_argument("--reverse", action="store_true", help="the shard's clips in reverse order (a second worker on the same shard; done clips are skipped)")
    args = p.parse_args()
    from bench.body_models import SMPL

    os.makedirs(args.out, exist_ok=True)
    kin = G1Kinematics(URDF).cuda()
    sup = support_points(MJCF)
    sup["pelvis"] = pelvis_support(MJCF)
    part = SMPL(SMPL_PKL).weights.argmax(1).numpy()
    butt = np.flatnonzero(np.isin(part, (0, 1, 2)))
    files = sorted(f for f in glob.glob(f"{args.recon}/*.npz") if not f.endswith("_hfield.npz"))
    files = [f for f in files if not args.only or any(o in f for o in args.only.split(","))][args.shard::args.num_shards]
    files = files[::-1] if args.reverse else files
    for f in files:
        name = os.path.basename(f)[:-4]
        out = Path(args.out) / f"{name}_motion.npz"
        if out.exists():
            if not (Path(args.out) / f"{name}_scene_g1.obj").exists():
                export_scene(f, Path(args.out) / f"{name}_scene_g1.obj", float(np.load(out)["scale"]))
            continue
        z = np.load(f)
        hf = dict(np.load(f[:-4] + "_hfield.npz"))
        if args.clean:  # level snapping of the reconstructed terrain (bench/videomimic/recon.clean_heightfield)
            from bench.videomimic.recon import clean_heightfield

            key = ("level", hashlib.sha1(np.ascontiguousarray(hf["height"]).tobytes() + np.ascontiguousarray(z["joints"][:, 0, :2]).tobytes()
                                         + json.dumps({k: float(hf[k]) for k in ("x0", "y0", "res")}).encode()).hexdigest())
            if key not in _CACHE:
                _CACHE[key] = clean_heightfield(hf, z["joints"][:, 0, :2])
            hf = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in _CACHE[key].items()}
        fps_in = float(z["fps"])
        n_in = len(z["joints"])
        n = int(np.floor((n_in - 1) / fps_in * 50.0)) + 1
        J = resample(z["joints"].astype(np.float64), n, fps_in, 50.0)
        if args.sigma_in > 0:  # zero-phase smoothing of the human joints at 50 fps (positions)
            from scipy.ndimage import gaussian_filter1d

            J = gaussian_filter1d(J, args.sigma_in, axis=0, mode="nearest")
        if args.sigma_root_xy > 0:  # the pelvis path in the ground plane smoothed on its own (whole skeleton shifted)
            from scipy.ndimage import gaussian_filter1d

            r = J[:, 0, :2].copy()
            J[:, :, :2] += (gaussian_filter1d(r, args.sigma_root_xy, axis=0, mode="nearest") - r)[:, None]
        skel = args.mode == "skeleton"
        con = {} if skel else {g: np.asarray(v, bool) for g, v in json.loads(str(z["contacts"])).items()}
        # seated: buttock vertices on the terrain and slow (at the input rate), from the human mesh
        V = None if skel else z["verts"].astype(np.float64)
        from bench.videomimic.scene_metrics import HF

        terr_h = HF(hf)
        cfg = json.loads(args.cfg or "{}")
        if skel:  # every label from the joints (fab/retarget/skeleton.py rule 1, the mesh rule's thresholds)
            from fab.retarget.skeleton import skeleton_rules

            assert not (cfg.get("points") or cfg.get("points_sole_only") or cfg.get("w_pts", 0) > 0 or cfg.get("w_foot_h", 0) > 0
                        or cfg.get("root_lowest")), "skeleton mode: no per-point / dense / foot-height / lowest-point terms"
            Jin = z["joints"].astype(np.float64)[:, :24]
            sk = skeleton_rules(Jin, terr_h, fps_in, cfg.pop("h_contact", 0.04), cfg.pop("v_contact", 0.15))
            con = sk["contacts"]
        elif "h_contact" in cfg or "v_contact" in cfg:  # relabel on the terrain with the config's thresholds (the mesh rule)
            from fab.retarget.contact import human_contact_labels

            Vrel = V.copy()
            Vrel[..., 2] -= terr_h(V[..., :2], V[..., 2])
            con = human_contact_labels(Vrel, part, fps_in, cfg.pop("h_contact", 0.03), cfg.pop("v_contact", 0.5))
        if cfg.get("contacts_dir"):  # contact-head files (<dir>/<clip>__0.contact.npz): heel or toe
            # static-contact probability >= contact_thr per foot; contact_mode replace (default) | and | or with the labels
            base = name
            for suf in ("_w06", "_w20", "_full"):
                base = base[: -len(suf)] if base.endswith(suf) else base
            cf = Path(cfg["contacts_dir"]) / f"{base}__0.contact.npz"
            if cf.exists():
                cz = np.load(cf, allow_pickle=True)
                key = cfg.get("contact_key", "video_static_prob")
                key = key if key in cz.files else key.replace("video", "image")
                prob = np.asarray(cz[key], np.float64)
                rows = {int(fr): i for i, fr in enumerate(np.asarray(cz["frame"], int))}
                kk = np.array([rows.get(int(fr), -1) for fr in np.asarray(z["frames"], int)])
                okh = kk >= 0
                mode = cfg.get("contact_mode", "replace")
                for g, cols in (("lfoot", [0, 1]), ("rfoot", [2, 3])):
                    h = np.zeros(len(kk), bool)
                    h[okh] = prob[kk[okh]][:, cols].max(1) >= float(cfg.get("contact_thr", 0.5))
                    new = h if mode == "replace" else (h & con[g]) if mode == "and" else (h | con[g])
                    con[g] = np.where(okh, new, con[g])
                hk = key.replace("_static_prob", "_hk_static_prob")
                if cfg.get("contact_hk") and hk in cz.files:  # hands / knees from the same head's static channels (one rule
                    # for every group; hk_channels l_hand, r_hand, l_knee, r_knee)
                    phk = np.asarray(cz[hk], np.float64)
                    for g, col in (("lhand", 0), ("rhand", 1), ("lknee", 2), ("rknee", 3)):
                        h = np.zeros(len(kk), bool)
                        h[okh] = phk[kk[okh]][:, col] >= float(cfg.get("contact_thr", 0.5))
                        con[g] = np.where(okh, h, np.asarray(con.get(g, np.zeros(len(kk), bool)), bool))
                print(f"[scene_rt] {name}: contact heads {key}{' + ' + hk if cfg.get('contact_hk') else ''} on {okh.mean() * 100:.0f} % "
                      f"of frames ({mode})", flush=True)
            else:
                print(f"[scene_rt] {name}: no contact-head file {cf}, heuristic labels", flush=True)
        if cfg.get("contact_labels_dir"):  # {lfoot, rfoot: (n_in,) 0/1} per clip (<dir>/<clip>.json), e.g. ground-truth kinematic labels
            lf = Path(cfg["contact_labels_dir"]) / f"{name}.json"
            if lf.exists():
                lj = json.loads(lf.read_text())
                for g in ("lfoot", "rfoot"):
                    if g in lj:
                        con[g] = np.asarray(lj[g], bool)[:len(con[g])]
                print(f"[scene_rt] {name}: contact labels from {lf}", flush=True)
        if cfg.get("points"):  # heel / toe per point from one source: <points_dir>/<clip>__0.contact.npz with
            # frame (the recon frames), video_static_prob (pin) and video_any_prob (touching), channels l_heel, l_toe, r_heel, r_toe;
            # the feet's lfoot / rfoot = heel or toe touching (swing, the link-clearance exemption); hands / knees unchanged
            pz = np.load(Path(cfg["points_dir"]) / f"{name}__0.contact.npz", allow_pickle=True)
            rows = {int(fr): i for i, fr in enumerate(np.asarray(pz["frame"], int))}
            kk = np.array([rows.get(int(fr), -1) for fr in np.asarray(z["frames"], int)])
            okp = kk >= 0
            for key, suf in (("video_static_prob", ""), ("video_any_prob", "_t")):
                prob = np.asarray(pz[key], np.float64)
                for j, g in enumerate(PT_CHANNELS):
                    h = np.zeros(len(kk), bool)
                    h[okp] = prob[kk[okp], j] >= 0.5
                    con[g + suf] = h
            con["lfoot"], con["rfoot"] = con["lheel_t"] | con["ltoe_t"], con["rheel_t"] | con["rtoe_t"]
            print(f"[scene_rt] {name}: points from {str(pz['source']) if 'source' in pz.files else cfg['points_dir']} on {okp.mean() * 100:.0f} % of frames", flush=True)
        if cfg.get("points_sole_only") and not cfg.get("points"):  # only the per-point touching labels
            # (video_any_prob of <points_dir>/<clip>__0.contact.npz) as <side>heel_t / <side>toe_t, read by the sole term alone; no
            # heel / toe support groups, lfoot / rfoot unchanged (the mesh rule)
            pz = np.load(Path(cfg["points_dir"]) / f"{name}__0.contact.npz", allow_pickle=True)
            rows = {int(fr): i for i, fr in enumerate(np.asarray(pz["frame"], int))}
            kk = np.array([rows.get(int(fr), -1) for fr in np.asarray(z["frames"], int)])
            okp = kk >= 0
            prob = np.asarray(pz["video_any_prob"], np.float64)
            for j, g in enumerate(PT_CHANNELS):
                h = np.zeros(len(kk), bool)
                h[okp] = prob[kk[okp], j] >= 0.5
                con[g + "_t"] = h
            print(f"[scene_rt] {name}: sole-only touching labels from {cfg['points_dir']} on {okp.mean() * 100:.0f} % of frames", flush=True)
        if cfg.get("hand_windows"):  # fab-retarget-ee: hand-floor windows {windows: {left | right: [{frames: [first, last]}]}} of
            # the input's frame ids: the hands' support points held on the terrain there by the contact / skate terms
            fr_in = np.asarray(z["frames"], int)
            hw = json.loads(Path(cfg["hand_windows"]).read_text())["windows"]
            for g, side in (("lhand", "left"), ("rhand", "right")):
                m_ = np.zeros(len(fr_in), bool)
                for w_ in hw.get(side, []):
                    m_ |= (fr_in >= w_["frames"][0]) & (fr_in <= w_["frames"][1])
                con[g] = m_
                if g + "_t" in con:
                    con[g + "_t"] = m_.copy()
        if cfg.get("contact_groups"):  # only these groups' labels (e.g. "lfoot,rfoot": hands / knees off)
            keep = set(cfg["contact_groups"].split(",")) | {g + "_t" for g in cfg["contact_groups"].split(",")}
            con = {g: (v if g in keep or g.removesuffix("_t") in keep else np.zeros_like(np.asarray(v, bool))) for g, v in con.items()}
        if cfg.get("contact_debounce"):  # every group's labels with the stance-run rule (gaps < 0.1 s filled, runs < 0.2 s dropped;
            # bench/g1_contact_check.stance_runs): single-frame contacts pull the root
            gap, mn = int(round(0.1 * fps_in)), int(round(0.2 * fps_in))
            for g, v in list(con.items()):
                v = np.asarray(v, bool).copy()
                e = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
                a_, b_ = e[::2], e[1::2]
                for k in range(len(a_) - 1):
                    if a_[k + 1] - b_[k] < gap:
                        v[b_[k]: a_[k + 1]] = True
                e = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
                for a2, b2 in zip(e[::2], e[1::2]):
                    if b2 - a2 < mn:
                        v[a2:b2] = False
                con[g] = v
        if cfg.get("no_flight") and not skel:  # --ground_aware: frames with no foot labelled where the person's lowest foot vertex is
            # < no_flight above the terrain or (no_flight_key) FAB's video contact says a foot touches, debounced as above -> group
            # anyfoot: the lower G1 foot's lowest point held at the terrain
            from fab.retarget.contact import HUMAN_PARTS

            fv = np.flatnonzero(np.isin(part, HUMAN_PARTS["lfoot"] + HUMAN_PARTS["rfoot"]))
            hlow = (V[:, fv, 2] - terr_h(V[:, fv, :2], V[:, fv, 2])).min(1)
            near = hlow < float(cfg["no_flight"])
            if cfg.get("no_flight_key") and cfg.get("contacts_dir"):
                cf_ = Path(cfg["contacts_dir"]) / f"{base}__0.contact.npz"
                if cf_.exists():
                    cz_ = np.load(cf_, allow_pickle=True)
                    k_ = cfg["no_flight_key"] if cfg["no_flight_key"] in cz_.files else cfg["no_flight_key"].replace("video", "image")
                    pa = np.asarray(cz_[k_], np.float64)
                    rows_ = {int(fr): i for i, fr in enumerate(np.asarray(cz_["frame"], int))}
                    kk_ = np.array([rows_.get(int(fr), -1) for fr in np.asarray(z["frames"], int)])
                    tch = np.zeros(len(kk_), bool)
                    tch[kk_ >= 0] = pa[kk_[kk_ >= 0]].max(1) >= float(cfg.get("contact_thr", 0.5))
                    near = near | tch
            nf = (~np.asarray(con["lfoot"], bool)) & (~np.asarray(con["rfoot"], bool)) & near
            gap_, mn_ = int(round(0.1 * fps_in)), int(round(0.2 * fps_in))
            e = np.flatnonzero(np.diff(np.r_[0, nf.astype(int), 0]))
            a_, b_ = e[::2], e[1::2]
            for k in range(len(a_) - 1):
                if a_[k + 1] - b_[k] < gap_:
                    nf[b_[k]: a_[k + 1]] = True
            e = np.flatnonzero(np.diff(np.r_[0, nf.astype(int), 0]))
            for a2, b2 in zip(e[::2], e[1::2]):
                if b2 - a2 < mn_:
                    nf[a2:b2] = False
            con["anyfoot"] = nf & ~np.asarray(con["lfoot"], bool) & ~np.asarray(con["rfoot"], bool)
            print(f"[scene_rt] {name}: no-flight frames (no foot labelled, the person's foot < {cfg['no_flight']} m"
                  f"{' or ' + cfg['no_flight_key'] + ' touching' if cfg.get('no_flight_key') else ''}) "
                  f"{int(con['anyfoot'].sum())} of {len(nf)}", flush=True)
        if skel:  # rule 2: the pelvis / thigh spheres' lowest point, the pelvis joint's speed
            hb, cb = sk["pelvis_h"], Jin[:, 0, :2]
        else:
            hb = (V[:, butt, 2] - terr_h(V[:, butt, :2], V[:, butt, 2])).min(1)
            cb = V[:, butt, :2].mean(1)
        vb = np.r_[0.0, np.linalg.norm(np.diff(cb, axis=0), axis=-1) * fps_in]
        pel = z["joints"][:, 0].astype(np.float64)
        con["pelvis"] = (hb < args.h_pelvis) & (vb < 0.5) & (pel[:, 2] - terr_h(pel[:, :2], pel[:, 2] - 0.2) < 0.25)
        idx = np.clip(np.round(np.arange(n) / 50.0 * fps_in).astype(int), 0, n_in - 1)
        con50 = {g: v[idx] for g, v in con.items()}
        hts = None
        if cfg.get("w_foot_h", 0) > 0:  # the human feet's lowest vertex above the terrain
            from fab.retarget.contact import HUMAN_PARTS

            hts = {}
            for g in ("lfoot", "rfoot"):
                vi = np.flatnonzero(np.isin(part, HUMAN_PARTS[g]))
                # clamped at 0: a reconstructed foot below the terrain is an error, not a target
                hts[g] = resample(np.maximum((V[:, vi, 2] - terr_h(V[:, vi, :2], V[:, vi, 2])).min(1), 0.0), n, fps_in, 50.0)
        Jt = torch.as_tensor(J, dtype=torch.float32, device="cuda")
        # s is the bone-length ratio of retarget_sequence; the terrain is scaled by the same s
        with torch.no_grad():
            from fab.retarget.g1 import BONES

            hum_len = torch.stack([(Jt[:, a] - Jt[:, b]).norm(dim=-1) for a, b in BONES], -1)
            s = float((kin.rest_lengths().sum() / hum_len.sum(-1).clamp_min(1e-3)).median())
        terr = Terrain(hf, s)
        # seat_hips: the seat contact / floor terms act on the lowest of the pelvis and hip links (the hip motors hang
        # below the pelvis link and otherwise go 8-11 cm into seats)
        sup_c = dict(sup, pelvis=pelvis_support(MJCF, hips=True)) if cfg.get("seat_hips", 0) else sup
        if cfg.get("no_flight"):  # --ground_aware: the no-flight group's support points = both feet's
            sup_c = dict(sup_c, anyfoot=list(sup_c["lfoot"]) + list(sup_c["rfoot"]))
        if cfg.get("points"):  # heel = sole points 1-2 (x -5 cm in the ankle-roll frame), toe = sole points 3-4 and the toe (x 12-14 cm)
            sup_c = dict(sup_c)
            for side, g in (("l", "lfoot"), ("r", "rfoot")):
                heel = [x for x in sup_c[g] if x[0].endswith(("_sole_1", "_sole_2"))]
                toe = [x for x in sup_c[g] if not x[0].endswith(("_sole_1", "_sole_2"))]
                sup_c.update({side + "heel": heel, side + "toe": toe, side + "heel_t": heel, side + "toe_t": toe})
        terr_o = terr
        if args.terrain_solve:  # the clean terrain, already in the G1 world (+ dz): the solve stands on it
            cf_ = Path(args.terrain_solve) / name / args.terrain_file
            ck = np.load(cf_)
            dz_ = float(json.loads((Path(args.pkg) / name / "package.json").read_text())["dz_m"])
            terr = Terrain({"x0": ck["x0"], "y0": ck["y0"], "res": ck["res"], "height": np.asarray(ck["height"], np.float64) - dz_}, 1.0)
        init, rz0 = None, None
        if cfg.get("flex_init") or cfg.get("root_lowest"):  # the paper retargeter's start (bench/sonic_motion --ik_motion)
            key = ("init", hashlib.sha1(J.tobytes()).hexdigest(), bool(cfg.get("flex_init")), bool(cfg.get("graph", 1)), URDF)
            if key not in _CACHE:  # the init depends on the human joints alone: a chain's second solve reuses the first's
                _CACHE[key] = tuple(x.clone() for x in retarget_sequence(kin, Jt, flex_init=bool(cfg.get("flex_init")),
                                                                         graph=bool(cfg.get("graph", 1)))[:3])
            init = tuple(x.clone() for x in _CACHE[key])
        if cfg.get("root_lowest"):  # root height: the G1's lowest point (init pose, all geoms) at s x the absolute height
            # of the human vertex lowest above the terrain (on a flat floor: s x the human's elevation, the lowest-point rule)
            from scipy.ndimage import gaussian_filter1d

            from bench.sonic_motion import g1_lowest

            k_ = (V[..., 2] - terr_h(V[..., :2], V[..., 2])).argmin(1)
            z_low = resample(V[np.arange(len(V)), k_, 2], n, fps_in, 50.0)
            Q0 = Rotation.from_matrix(init[1].cpu().numpy().astype(np.float64)).as_quat()
            low0 = g1_lowest(MJCF, list(kin.revolute), Q0[:, [3, 0, 1, 2]], init[0].cpu().numpy().astype(np.float64))
            rz0 = gaussian_filter1d(float(init[2]) * z_low - low0, 1.0, mode="nearest")
        if cfg.get("root_init") == "terrain":  # root start / root_z prior = the scaled human pelvis, raised per
            # frame by how far a human foot's lowest point sits below the SOLVE terrain (the one the G1 stands on) at that foot's
            # centre, so that point starts on it. A foot counts only while its leg crosses the surface there (knee joint above it):
            # under a 2.5D overhang (table / seat top) or beside a block edge the heightfield is not the foot's support. One-sided
            # (a foot below the terrain is an error, as w_foot_h's clamp; above it can be real: swing, jumps); the deeper foot; the
            # depth gets the input's own smoother (--sigma_in)
            assert rz0 is None, "root_init terrain and root_lowest are exclusive"
            from fab.retarget.contact import HUMAN_PARTS

            sink = np.zeros(n)
            for g_, kn in (("lfoot", 4), ("rfoot", 5)):
                if skel:  # rule 3: the foot's lowest point of rule 1, at the ankle / foot joint centre
                    lowz = resample(sk["foot_low"][g_], n, fps_in, 50.0) * s
                    cxy = resample(sk["foot_xy"][g_], n, fps_in, 50.0) * s
                else:
                    Vf = resample(V[:, np.flatnonzero(np.isin(part, HUMAN_PARTS[g_]))], n, fps_in, 50.0) * s
                    lowz, cxy = Vf[..., 2].min(1), Vf[..., :2].mean(1)
                with torch.no_grad():
                    tc = terr(torch.as_tensor(cxy, dtype=torch.float32, device="cuda"),
                              torch.as_tensor(lowz, dtype=torch.float32, device="cuda")).cpu().numpy().astype(np.float64)
                sink = np.maximum(sink, np.where(J[:, kn, 2] * s > tc, tc - lowz, 0.0))
            if args.sigma_in > 0:
                from scipy.ndimage import gaussian_filter1d

                sink = gaussian_filter1d(sink, args.sigma_in, mode="nearest")
            rz0 = J[:, 0, 2] * s + sink
            print(f"[scene_rt] {name}: root_init terrain, foot depth below the solve terrain f0 {100 * sink[0]:.1f} cm, median "
                  f"{100 * np.median(sink):.1f} cm, max {100 * sink.max():.1f} cm", flush=True)
        t_st = None
        if cfg.get("init_from"):  # the same objective started from another retargeter's solution (<dir>:<fmt>,
            # {c} = clip; its 50 fps G1 motion in this world): joint angles, root rotation and root path; priors unchanged
            d_i, f_i = cfg["init_from"].rsplit(":", 1)
            zi = np.load(Path(d_i) / f"{f_i.format(c=name)}_motion.npz")
            jn_i = [str(j) for j in zi["g1_joint_names"]]
            assert len(zi["g1_dof"]) >= n, (len(zi["g1_dof"]), n)
            qi = torch.as_tensor(np.asarray(zi["g1_dof"], np.float64)[:n][:, [jn_i.index(j) for j in kin.revolute]],
                                 dtype=torch.float32, device="cuda")
            qi = torch.minimum(torch.maximum(qi, kin.lower), kin.upper)
            Ri = torch.as_tensor(Rotation.from_quat(np.asarray(zi["g1_root_quat_xyzw"], np.float64)[:n]).as_matrix(),
                                 dtype=torch.float32, device="cuda")
            s_i = init[2] if init is not None else retarget_sequence(kin, Jt)[2]
            init, t_st = (qi, Ri, s_i), np.asarray(zi["g1_root_pos"], np.float64)[:n]
        dense = None
        if cfg.get("w_pts", 0) > 0:  # dense-point term: FAB's dense points registered on the input body (pts, pts_idx; fab-retarget
            # points mode) or, for an input without them, the input body's own surface at the same SMPL vertex ids; the joints'
            # resampling and --sigma_in smoother
            dcorr = dict(np.load(cfg.get("dense_corr", DENSE_CORR)))
            if "pts" in z.files:
                assert np.array_equal(np.asarray(z["pts_idx"]), dcorr["surface_idx"]), name
                Pd = np.asarray(z["pts"], np.float64)
            else:
                Pd = V[:, dcorr["surface_idx"]]
            Pd = resample(Pd, n, fps_in, 50.0)
            if args.sigma_in > 0:
                from scipy.ndimage import gaussian_filter1d

                Pd = gaussian_filter1d(Pd, args.sigma_in, axis=0, mode="nearest")
            dense = (Pd, dcorr)
            print(f"[scene_rt] {name}: dense points ({'FAB pts' if 'pts' in z.files else 'input body surface'}, "
                  f"{Pd.shape[1]}), w_pts {cfg['w_pts']}", flush=True)
        eeg = None
        if cfg.get("ee_targets"):  # fab-retarget-ee: {frames, <side>_<hand | foot> (T_in, 3) G1-world targets, + _valid} -> 50 fps
            ez = np.load(cfg["ee_targets"], allow_pickle=True)
            rows = {int(f_): i for i, f_ in enumerate(np.asarray(ez["frames"], int))}
            kk = np.array([rows.get(int(f_), -1) for f_ in np.asarray(z["frames"], int)])
            i50 = np.clip(np.round(np.arange(n) / 50.0 * fps_in).astype(int), 0, n_in - 1)
            eeg = {}
            for g, key in (("lhand", "left_hand"), ("rhand", "right_hand"), ("lfoot", "left_foot"), ("rfoot", "right_foot")):
                if key not in ez.files or not float(cfg.get("w_ee_" + g[1:], 0)):
                    continue
                P_in = np.where(kk[:, None] >= 0, np.asarray(ez[key], np.float64)[np.clip(kk, 0, None)], 0.0)
                v_in = (kk >= 0) & np.asarray(ez[key + "_valid"], bool)[np.clip(kk, 0, None)]
                eeg[g] = (resample(P_in, n, fps_in, 50.0), (v_in[i50] & v_in[np.clip(i50 + 1, 0, n_in - 1)]).astype(np.float32))
            print(f"[scene_rt] {name}: end-effector targets {sorted(eeg)} (w_ee_hand {cfg.get('w_ee_hand')}, w_ee_foot "
                  f"{cfg.get('w_ee_foot')})", flush=True)
        q, R, t, s2, rep = retarget_scene(kin, Jt, 50.0, con50, sup_c, terr, URDF, cfg, heights=hts, init=init, root_z0=rz0,
                                          t_start=t_st, dense=dense, ee_targets=eeg)
        if skel:
            rep["skeleton"] = sk["info"]
        if dense is not None:
            rep["dense"] = {"source": "pts" if "pts" in z.files else "verts", "corr": cfg.get("dense_corr", DENSE_CORR),
                            "pts_info": json.loads(str(z["pts_info"])) if "pts_info" in z.files else None}
        if args.terrain_solve:
            rep["terrain_solve"] = {"file": str(cf_), "dz_m": dz_}
            terr = terr_o
        assert abs(float(s2) - s) < 1e-4, (float(s2), s)
        quat = Rotation.from_matrix(R.cpu().numpy().astype(np.float64)).as_quat()
        np.savez(out, fps=50.0, g1_dof=q.cpu().numpy().astype(np.float32), g1_joint_names=np.array(list(kin.revolute)),
                 g1_root_pos=t.cpu().numpy().astype(np.float32), g1_root_quat_xyzw=quat.astype(np.float32), scale=s,
                 frames=idx, contacts=json.dumps({g: v.astype(int).tolist() for g, v in con50.items()}),
                 report=json.dumps(rep, default=float))
        extra = {}
        if terr.L is not None:  # multi-level terrain: the level stack and the physics boxes, x s
            extra["levels"] = terr.L.cpu().numpy()
            if "boxes" in hf:
                extra["boxes"] = np.asarray(hf["boxes"], np.float64) * s
        np.savez(Path(args.out) / f"{name}_terrain_g1.npz", x0=terr.x0, y0=terr.y0, res=terr.res,
                 height=terr.H.cpu().numpy(), scale=s, **extra)
        export_scene(f, Path(args.out) / f"{name}_scene_g1.obj", s)
        export_terrain_mesh(terr, Path(args.out) / f"{name}_terrain_g1.obj")
        print(f"[scene_rt] {name}: {n} frames @50, s {s:.3f}, residual {rep['residual_mm']:.0f} mm, world "
              f"{rep['before']['world']:.2e} -> {rep['after']['world']:.2e}, contact {rep['before']['contact']:.4f} -> "
              f"{rep['after']['contact']:.4f}, pelvis {rep['after']['contact_pelvis']:.4f} "
              f"({int(con['pelvis'].sum())} seated frames)", flush=True)


if __name__ == "__main__":
    main()
