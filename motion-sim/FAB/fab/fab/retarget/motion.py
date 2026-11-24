"""Motion-based SMPL -> Unitree G1 retargeting (whole clip, world space).

Per-frame retargeting matches static poses; a tracker has to follow the motion. This solver optimises the joint
angles, root rotation and root translation of the whole clip jointly against motion terms:
  pose        pelvis-relative G1 links vs the human joints scaled by ONE human -> G1 factor for the clip
  velocity    world velocities of the end effectors (hands, ankles, toes) and of the pelvis vs the scaled human's
  contact     support groups the human has on the floor (labels from the human motion: lowest part vertex within
              h_contact of the local floor, slow) touch the floor and do not slide (variance of their position over
              each contact segment): planted feet stay planted
  floor       no support point below the floor
  limits      joint angles inside the URDF limits (sigmoid parameterisation), joint speeds under the URDF velocity
              limits (hinge penalty), acceleration penalty on joints / root (smoothness, torque proxy)
  root        weak priors keeping the root on the scaled human path and the initial height
  directions  (w_dir, off by default) G1 bone directions (unit vectors between the mapped links: legs, feet, arms,
              pelvis -> shoulders) vs the human's: the proportions differ, so matching scaled positions alone bends
              the G1's bones away from the human's directions, which is what a tracker is scored on
  root rot    (w_rrot, off by default) the G1 pelvis orientation vs the human pelvis frame (hips left-right axis,
              pelvis -> spine up axis): the scored bone directions are measured in the root's heading frame, and
              position-only IK lets the pelvis yaw / tilt drift while the waist compensates
  init prior  (w_init, with t_init) refinement of another retargeter's per-frame G1 motion (e.g. SOMA, GMR): start
              from its joint angles / root rotation / root path, keep close to its pose (joint angles and root
              rotation), and the root path / height priors follow its root instead of the scaled human pelvis
  foot height (w_foot_h, off by default) every frame, each G1 sole's lowest point at the human foot's height above
              the floor x the human -> G1 scale: swing clearance and stance contact both follow the human (a pose
              change that lowers the legs does not drag the feet along the floor)
  link accel  (w_lacc, off by default) acceleration of the mapped G1 links, pelvis-relative in the root's heading
              frame ((m/s^2)^2): smoothness where the tracker feels it, root-yaw jitter included
  init feet   (init_contacts, off by default; refinement) the per-frame retargeter's own planted feet (its sole within
              2 cm of the floor and slower than 0.2 m/s) become foot contacts: "replace" the human labels of the feet,
              "or" add to them, "and" keep only the frames both agree on
  swing       (w_swing, off by default) a foot the human does not have on the floor keeps its sole at least swing_clear
              above it: low swing feet read as foot skate to a tracker (and to the skate metric)
  foot speed  (w_footvel, off by default) a sole's horizontal speed, on frames where the human's foot is also slow
              (< footvel_v m/s) and low (< footvel_h m): a soft per-frame stand-in for the segment lock
  ground      (w_ground, off by default) while the human touches the floor (its lowest point within 2 cm of the
              local floor), the G1's lowest support point does too: no floating body in planks / kneeling
  balance     (w_com, off by default) while the human has support contacts, the G1 centre of mass (URDF link masses)
              stays over the contact groups' support points: horizontal distance to their centroid beyond their
              mean spread (a smooth stand-in for the support polygon; static balance)
Initialised from the temporally consistent root-relative IK (g1.retarget_sequence) and the lowest-point root height.
"""

import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

from fab.retarget.contact import group_points
from fab.retarget.g1 import END_EFFECTORS, SMPL_TO_G1, retarget_sequence, rot6d_to_matrix

DEFAULTS = {"w_pose": 1.0, "w_vel": 0.05, "w_root_vel": 0.05, "w_contact": 50.0, "w_skate": 50.0, "w_floor": 1000.0,
            "w_vlim": 0.01, "w_acc": 2.0, "w_root_xy": 0.01, "w_root_z": 0.1, "vlim_frac": 0.8, "iters": 400, "lr": 0.02,
            "w_com": 0.0, "w_dir": 0.0, "dir_mode": "mapped", "w_rrot": 0.0, "w_init": 0.0, "w_ground": 0.0, "w_foot_h": 0.0, "w_lacc": 0.0, "init_contacts": "off", "w_swing": 0.0, "swing_clear": 0.03, "swing_h": 0.0, "w_footvel": 0.0, "footvel_v": 0.3, "footvel_h": 0.05, "footvel_gate": "human",
            "lr_sched": "const", "lr_t": None, "rrot_sigma": 0.0, "w_racc": 0.0, "c2f_frac": 0.0, "still_sigma": 0.0,
            "still_v": 0.3, "w_headg": 0.0, "headg_deg": 20.0, "w_hrate": 0.0, "hrate_free": 1.0}
# w_hrate / hrate_free: the heading rate (azimuth of the pelvis forward axis's ground projection, what trackers take as
# heading) above hrate_free rad/s, squared: near-vertical forward axes keep a steady azimuth instead of spinning
# w_headg / headg_deg: heading guard, the pelvis forward axis kept >= headg_deg from vertical (trackers and the scorer
# take the heading from its ground projection: prone poses with it near vertical spin the heading frame)
# still_sigma / still_v: stillness-adaptive smoothing of the human joints (per joint, a blend towards a Gaussian-smoothed
# copy, sigma still_sigma frames, weight 1 - speed / still_v): near-static poses lose the estimator's jitter, motion keeps
# its timing
# rrot_sigma: Gaussian smoothing (frames) of the human pelvis frame w_rrot follows; w_racc: root orientation
# acceleration in physical units (1/s^2); c2f_frac: coarse-to-fine, the first fraction of the iterations moves only the
# root and the leg / waist joints (arms held at the init)
# lr_sched "cos": cosine decay of the step size to 0 (Adam's normalised steps otherwise leave a per-frame random walk
# of ~lr in every variable: jitter and foot skate); lr_t: the root translation's own step size (m)
DIR_BONES = [(1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11), (16, 18), (18, 20), (17, 19), (19, 21), (0, 16), (0, 17)]
# dir_mode "links": the limb bones between the G1 links a tracker is scored on (hip roll -> knee -> ankle roll,
# shoulder roll -> elbow -> wrist yaw) vs the human's (hip -> knee -> ankle,
# shoulder -> elbow -> wrist), plus the feet and pelvis -> shoulder midpoint
LINK_BONES = [((16, 18), ("left_shoulder_roll_link", "left_elbow_link")),
              ((18, 20), ("left_elbow_link", "left_wrist_yaw_link")),
              ((17, 19), ("right_shoulder_roll_link", "right_elbow_link")),
              ((19, 21), ("right_elbow_link", "right_wrist_yaw_link")),
              ((1, 4), ("left_hip_roll_link", "left_knee_link")), ((4, 7), ("left_knee_link", "left_ankle_roll_link")),
              ((2, 5), ("right_hip_roll_link", "right_knee_link")), ((5, 8), ("right_knee_link", "right_ankle_roll_link")),
              ((7, 10), ("left_ankle_roll_link", "left_toe")),
              ((8, 11), ("right_ankle_roll_link", "right_toe"))]


def velocity_limits(urdf, names):
    lim = {j.get("name"): float(j.find("limit").get("velocity")) for j in ET.parse(urdf).getroot().findall("joint")
           if j.get("type") == "revolute" and j.find("limit") is not None and j.find("limit").get("velocity")}
    return torch.tensor([lim.get(n, 30.0) for n in names])


def link_inertials(urdf):
    """{link: (mass, centre of mass in link coordinates)} from the URDF."""
    out = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        i = link.find("inertial")
        if i is None or i.find("mass") is None:
            continue
        o = i.find("origin")
        out[link.get("name")] = (float(i.find("mass").get("value")),
                                 [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
    return out


def retarget_motion(kin, J, fps, contacts, sup, root_z0, urdf, init=None, cfg=None, t_init=None, grounded=None,
                    heights=None, speeds=None, contact_z=None, root_free=None, contact_xy=None):
    """J [T, 24, 3] SMPL joints, z-up world, local floor at z = 0 (human scale); contacts {group: [T] bool};
    sup: support points (contact.support_points); root_z0 [T] initial root height (lowest point rule, at the same
    scale); init (q, R, scale) from g1.retarget_sequence (computed if None). The one scale is used for the limbs AND
    the root path (time-consistent, no limb / path mismatch). t_init [T, 3]: the root path of a per-frame retargeter
    whose (q, R) are passed as init (refinement mode). contact_z {group: [T]}: the height (G1 scale) a labelled group
    is held at while in contact (the scene surface under it; default the floor, 0); root_free [T] bool: frames without
    the root-height prior (e.g. hands on a support: the contacts place the root, not the lowest-point rule);
    contact_xy {group: [T, 2]} (G1 scale, NaN = free): where a labelled group is also held horizontally (thin supports).
    -> q [T, 29], R [T, 3, 3], t [T, 3], scale, report."""
    c = {**DEFAULTS, **(cfg or {})}
    dev = J.device
    T = len(J)
    if c["still_sigma"] > 0:
        from scipy.ndimage import gaussian_filter1d

        Jn = J.detach().cpu().numpy()
        Js = gaussian_filter1d(Jn, c["still_sigma"], axis=0, mode="nearest")
        sp = np.linalg.norm(np.gradient(Js, axis=0), axis=-1) * fps  # [T, 24] m/s (human scale)
        a = np.clip(1.0 - sp / c["still_v"], 0.0, 1.0)[..., None]
        J = torch.as_tensor(a * Js + (1 - a) * Jn, dtype=J.dtype, device=dev)
    q0, R0, s = init if init is not None else retarget_sequence(kin, J)[:3]
    lo, hi = kin.lower, kin.upper
    ids = list(SMPL_TO_G1)
    w = torch.tensor([2.0 if i in END_EFFECTORS else 1.0 for i in ids], device=dev)
    target_rel = (J[:, ids] - J[:, :1]) * s
    ee = [i for i in ids if i in END_EFFECTORS]
    ee_k = [ids.index(i) for i in ee]
    hum_ee_v = (J[1:, ee] - J[:-1, ee]) * s * fps
    hum_root_v = (J[1:, 0] - J[:-1, 0]) * s * fps
    t_xy0 = J[:, 0, :2] * s
    dir_a, dir_b = [ids.index(a) for a, _ in DIR_BONES], [ids.index(b) for _, b in DIR_BONES]
    hum_dir = torch.nn.functional.normalize(J[:, [b for _, b in DIR_BONES]] - J[:, [a for a, _ in DIR_BONES]], dim=-1)
    left = J[:, 1] - J[:, 2]
    up0 = J[:, 3] - J[:, 0]
    if c["rrot_sigma"] > 0:
        from scipy.ndimage import gaussian_filter1d

        sm = lambda a: torch.as_tensor(gaussian_filter1d(a.detach().cpu().numpy(), c["rrot_sigma"], axis=0, mode="nearest"),  # noqa: E731
                                       dtype=J.dtype, device=dev)
        left, up0 = sm(left), sm(up0)
    left = torch.nn.functional.normalize(left, dim=-1)
    up = torch.nn.functional.normalize(up0 - (up0 * left).sum(-1, keepdim=True) * left, dim=-1)
    R_hum = torch.stack([torch.cross(left, up, dim=-1), left, up], -1)
    if c["dir_mode"] == "links":
        hb = torch.stack([J[:, b] - J[:, a] for (a, b), _ in LINK_BONES] + [J[:, [16, 17]].mean(1) - J[:, 0]], 1)
        hum_dir = torch.nn.functional.normalize(hb, dim=-1)
    t0 = torch.cat([t_xy0, torch.as_tensor(root_z0, dtype=J.dtype, device=dev)[:, None]], -1)
    if t_init is not None:  # refinement: the per-frame retargeter's own root path is the start and the prior
        t0 = torch.as_tensor(t_init, dtype=J.dtype, device=dev).clone()
        t_xy0 = t0[:, :2].clone()
    q_init, R_init = q0.clone(), R0.clone()
    vmax = velocity_limits(urdf, kin.revolute).to(dev) * c["vlim_frac"]
    z = torch.logit(((q0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
    t = t0.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [z, r6]}, {"params": [t], "lr": c["lr_t"] or c["lr"]}], lr=c["lr"])
    lr0 = [g["lr"] for g in opt.param_groups]
    if c["init_contacts"] != "off" and t_init is not None:
        with torch.no_grad():
            pos_i, rot_i = kin(R0, t0, q0)
            contacts = dict(contacts)
            for g in ("lfoot", "rfoot"):
                if g not in sup:
                    continue
                pts = group_points(pos_i, rot_i, sup, g)
                lowz = pts[..., 2].min(-1).values
                xy = pts[..., :2].mean(1)
                v = torch.cat([torch.zeros(1, device=dev), (xy[1:] - xy[:-1]).norm(dim=-1) * fps])
                lab = ((lowz < 0.02) & (v < 0.2)).cpu().numpy()
                hum = np.asarray(contacts.get(g, np.zeros(T, bool)), bool)
                contacts[g] = {"replace": lab, "or": lab | hum, "and": lab & hum}[c["init_contacts"]]
    masks = {g: torch.as_tensor(np.asarray(m, bool), device=dev) for g, m in contacts.items()}
    cz_t = {g: torch.as_tensor(np.asarray(v, np.float32), dtype=J.dtype, device=dev) for g, v in (contact_z or {}).items()}
    cxy_t = {g: torch.as_tensor(np.asarray(v, np.float32), dtype=J.dtype, device=dev) for g, v in (contact_xy or {}).items()}
    rz_w = torch.ones(T, dtype=J.dtype, device=dev)
    if root_free is not None:
        rz_w[torch.as_tensor(np.asarray(root_free, bool), device=dev)] = 0.0
    segs = {}
    for g, m in contacts.items():
        e = np.flatnonzero(np.diff(np.r_[0, np.asarray(m, int), 0]))
        segs[g] = [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= 3]

    def lowest(p):
        k = p[..., 2].argmin(-1)
        return torch.gather(p, 1, k[:, None, None].expand(-1, 1, 3))[:, 0]

    inert = {k: v for k, v in link_inertials(urdf).items() if k in kin.links}
    m_link = torch.tensor([v[0] for v in inert.values()], device=dev)
    c_link = torch.tensor([v[1] for v in inert.values()], device=dev)
    any_c = torch.zeros(T, dtype=torch.bool, device=dev)
    for m in masks.values():
        any_c |= m

    def forward(z, r6, t):
        q = lo + (hi - lo) * torch.sigmoid(z)
        R = rot6d_to_matrix(r6)
        pos, rot = kin(R, t, q)
        P = torch.stack([pos[SMPL_TO_G1[i]] for i in ids], 1)
        pts = {g: group_points(pos, rot, sup, g) for g in sup}
        low = {g: lowest(p) for g, p in pts.items()}
        if c["w_footvel"] > 0:  # sole centroids (the lowest point jumps between sole points)
            low["_feetxy"] = {g: pts[g][..., :2].mean(1) for g in ("lfoot", "rfoot") if g in pts}
        if c["w_dir"] > 0 and c["dir_mode"] == "links":
            gb = torch.stack([pos[lb] - pos[la] for _, (la, lb) in LINK_BONES] +
                             [(pos["left_shoulder_roll_link"] + pos["right_shoulder_roll_link"]) / 2 - pos["pelvis"]], 1)
            low["_dir"] = torch.nn.functional.normalize(gb, dim=-1)
        if c["w_com"] > 0:
            lp = torch.stack([pos[k] for k in inert], 1)
            lr = torch.stack([rot[k] for k in inert], 1)
            com = ((lp + torch.einsum("tlij,lj->tli", lr, c_link)) * m_link[:, None]).sum(1) / m_link.sum()
            low["_com"], low["_pts"] = com, pts
        return q, R, P, low

    def balance(low):
        com, pts = low.pop("_com"), low.pop("_pts")
        num, den = 0.0, 0.0
        for g, m in masks.items():
            if g in pts and m.any():
                w = m.float()[:, None]
                num = num + (pts[g][..., :2] * w[..., None]).sum(1)
                den = den + w * pts[g].shape[1]
        if not torch.is_tensor(den) or not any_c.any():
            return torch.zeros((), device=dev)
        cen = num / den.clamp_min(1)
        spread = 0.0
        for g, m in masks.items():
            if g in pts and m.any():
                spread = spread + ((pts[g][..., :2] - cen[:, None]).norm(dim=-1) * m.float()[:, None]).sum(1)
        spread = spread / den[:, 0].clamp_min(1)
        d = torch.relu((com[:, :2] - cen).norm(dim=-1) - spread)
        return (d[any_c] ** 2).mean()

    def terms(q, R, P, low, t):
        e = {}
        if c["w_dir"] > 0:
            ub = low.pop("_dir") if "_dir" in low else torch.nn.functional.normalize(P[:, dir_b] - P[:, dir_a], dim=-1)
            e["dir"] = ((ub - hum_dir) ** 2).sum(-1).sum(-1).mean()
        if c["w_ground"] > 0 and grounded is not None and grounded.any():
            g_low = torch.stack([p[:, 2] for g, p in low.items() if not g.startswith("_")], -1).min(-1).values
            e["ground"] = (g_low[grounded] ** 2).mean()
        if c["w_footvel"] > 0 and heights is not None and speeds is not None:
            fv = 0.0
            for g in ("lfoot", "rfoot"):
                if g not in low.get("_feetxy", {}):
                    continue
                slow = (torch.as_tensor(speeds[g], device=dev) < c["footvel_v"]) & \
                    (torch.as_tensor(heights[g], device=dev) < c["footvel_h"])
                if c["footvel_gate"] == "labels" and g in masks:  # only where the contact labels say stance too
                    slow = slow & masks[g]
                fx = low["_feetxy"][g]
                v = (fx[1:] - fx[:-1]).norm(dim=-1) * fps
                m = slow[1:] & slow[:-1]
                if m.any():
                    fv = fv + (v[m] ** 2).mean()
            e["footvel"] = fv
        low.pop("_feetxy", None)
        if c["w_swing"] > 0:  # swing: the human foot clearly off the floor (swing_h), else off its contact labels
            sw = {g: (torch.as_tensor(heights[g], device=dev) > c["swing_h"]) if c["swing_h"] > 0 and heights is not None
                  else ~masks[g] for g in ("lfoot", "rfoot") if g in low and g in masks}
            e["swing"] = sum((torch.relu(c["swing_clear"] - low[g][m, 2]) ** 2).mean() for g, m in sw.items() if m.any())
        if c["w_lacc"] > 0:
            # pelvis-relative, in the root's heading frame (as trackers / the metrics see the body): root-yaw jitter
            # shakes every link there even when the world positions are smooth
            fx = torch.nn.functional.normalize(R[:, :2, 0], dim=-1)
            Pr = P - P[:, :1]
            Pr = torch.stack([fx[:, None, 0] * Pr[..., 0] + fx[:, None, 1] * Pr[..., 1],
                              -fx[:, None, 1] * Pr[..., 0] + fx[:, None, 0] * Pr[..., 1], Pr[..., 2]], -1)
            e["lacc"] = (((Pr[2:] - 2 * Pr[1:-1] + Pr[:-2]) * fps ** 2) ** 2).sum(-1).mean()
        if c["w_foot_h"] > 0 and heights is not None:
            e["foot_h"] = sum(((low[g][:, 2] - s * torch.as_tensor(heights[g], dtype=J.dtype, device=dev)) ** 2).mean()
                              for g in ("lfoot", "rfoot") if g in low and g in heights)
        if c["w_init"] > 0:
            e["init"] = ((q - q_init) ** 2).sum(-1).mean() + ((R - R_init) ** 2).sum((-1, -2)).mean()
        if c["w_rrot"] > 0:
            e["rrot"] = ((R - R_hum) ** 2).sum((-1, -2)).mean()
        if c["w_racc"] > 0:
            e["racc"] = (((R[2:] - 2 * R[1:-1] + R[:-2]) * fps ** 2) ** 2).sum((-1, -2)).mean()
        if c["w_headg"] > 0:
            e["headg"] = (torch.relu(R[:, 2, 0].abs() - np.cos(np.radians(c["headg_deg"]))) ** 2).mean()
        if c["w_hrate"] > 0:
            fx, fy = R[:, 0, 0], R[:, 1, 0]
            dh = torch.atan2(fx[:-1] * fy[1:] - fy[:-1] * fx[1:], fx[:-1] * fx[1:] + fy[:-1] * fy[1:] + 1e-6)
            e["hrate"] = (torch.relu(dh.abs() * fps - c["hrate_free"]) ** 2).mean()
        if "_com" in low:
            e["com"] = balance(low)
        e["pose"] = (((P - P[:, :1] - target_rel) ** 2).sum(-1) * w).sum(-1).mean()
        e["vel"] = (((P[1:, ee_k] - P[:-1, ee_k]) * fps - hum_ee_v) ** 2).sum(-1).mean()
        e["root_vel"] = (((t[1:] - t[:-1]) * fps - hum_root_v) ** 2).sum(-1).mean()
        e["floor"] = sum((torch.relu(-p[:, 2]) ** 2).mean() for p in low.values())
        con, sk = 0.0, 0.0
        for g, p in low.items():
            m = masks.get(g)
            if m is None or not m.any():
                continue
            cz = cz_t.get(g)
            con = con + ((p[m, 2] - (cz[m] if cz is not None else 0.0)) ** 2).mean()
            cxy = cxy_t.get(g)
            if cxy is not None:
                mx = m & torch.isfinite(cxy[:, 0])
                if mx.any():
                    con = con + ((p[mx, :2] - cxy[mx]) ** 2).sum(-1).mean()
            for a, b in segs[g]:
                xy = p[a:b, :2]
                sk = sk + ((xy - xy.mean(0, keepdim=True)) ** 2).sum(-1).mean() * (b - a) / T
        e["contact"], e["skate"] = con, sk
        e["vlim"] = (torch.relu((q[1:] - q[:-1]).abs() * fps - vmax) ** 2).sum(-1).mean()
        x = torch.cat([q, R[..., 0], R[..., 1], t], -1)
        e["acc"] = ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).sum(-1).mean()
        e["root_xy"] = ((t[:, :2] - t_xy0) ** 2).sum(-1).mean()
        e["root_z"] = (((t[:, 2] - t0[:, 2]) ** 2) * rz_w).mean()
        return e

    keys = {"pose": "w_pose", "vel": "w_vel", "root_vel": "w_root_vel", "floor": "w_floor", "contact": "w_contact",
            "skate": "w_skate", "vlim": "w_vlim", "acc": "w_acc", "root_xy": "w_root_xy", "root_z": "w_root_z",
            "com": "w_com", "dir": "w_dir", "rrot": "w_rrot", "init": "w_init", "ground": "w_ground", "foot_h": "w_foot_h", "lacc": "w_lacc", "swing": "w_swing", "footvel": "w_footvel", "racc": "w_racc", "headg": "w_headg", "hrate": "w_hrate"}
    arms = torch.tensor([bool(re.search("shoulder|elbow|wrist", n)) for n in kin.revolute], device=dev)
    n_c2f = int(c["c2f_frac"] * int(c["iters"]))
    with torch.no_grad():
        e0 = {k: float(v) for k, v in terms(*forward(z, r6, t), t).items()}
    for it_ in range(int(c["iters"])):
        if c["lr_sched"] == "cos":
            for g, l0 in zip(opt.param_groups, lr0):
                g["lr"] = l0 * 0.5 * (1 + np.cos(np.pi * it_ / int(c["iters"])))
        q, R, P, low = forward(z, r6, t)
        e = terms(q, R, P, low, t)
        loss = sum(c[keys[k]] * v for k, v in e.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if it_ < n_c2f:
            z.grad[:, arms] = 0
        opt.step()
    with torch.no_grad():
        q, R, P, low = forward(z, r6, t)
        e1 = {k: float(v) for k, v in terms(q, R, P, low, t).items()}
        res = (P - P[:, :1] - target_rel).norm(dim=-1).mean(-1)
    return q.detach(), R.detach(), t.detach(), s.detach(), {"before": e0, "after": e1, "residual_mm": float(res.mean() * 1000)}


def retarget_frames_dir(kin, J, init=None, w_dir=1.0, w_rrot=1.0, w_pose=0.1, passes=((0.03, 150, 0.08), (1e-3, 100, 0.03))):
    """Per-frame direction-first IK (the pose terms of retarget_motion without the motion / contact terms; a label maker
    for single images): SMPL joints J [B, 24, 3] (any frame) -> q [B, 29], R [B, 3, 3], scale [B], position residual [B]
    (m, G1 scale). Link-bone directions (LINK_BONES + pelvis -> shoulders) and the pelvis frame vs the human's, scaled
    pelvis-relative positions as a weak term; passes of (joint-angle regulariser, iterations, step): a strongly
    regularised solve then the weak one from it, as retarget_smpl_stable. init (q, R): e.g. g1.flexion_init."""
    from fab.retarget.g1 import BONES, root_frame

    B, dev = J.shape[0], J.device
    hum_len = torch.stack([(J[:, a] - J[:, b]).norm(dim=-1) for a, b in BONES], -1)
    scale = kin.rest_lengths().sum() / hum_len.sum(-1).clamp_min(1e-3)
    ids = list(SMPL_TO_G1)
    target = (J[:, ids] - J[:, :1]) * scale[:, None, None]
    w = torch.tensor([2.0 if i in END_EFFECTORS else 1.0 for i in ids], device=dev)
    hb = torch.stack([J[:, b] - J[:, a] for (a, b), _ in LINK_BONES] + [J[:, [16, 17]].mean(1) - J[:, 0]], 1)
    hum_dir = torch.nn.functional.normalize(hb, dim=-1)
    left = torch.nn.functional.normalize(J[:, 1] - J[:, 2], dim=-1)
    up0 = J[:, 3] - J[:, 0]
    up = torch.nn.functional.normalize(up0 - (up0 * left).sum(-1, keepdim=True) * left, dim=-1)
    R_hum = torch.stack([torch.cross(left, up, dim=-1), left, up], -1)
    lo, hi = kin.lower, kin.upper
    q0, R0 = init if init is not None else (torch.zeros(B, len(kin.revolute), device=dev), root_frame(J))
    zeros = torch.zeros(B, 3, device=dev)

    def fk(z, r6):
        q = lo + (hi - lo) * torch.sigmoid(z)
        R = rot6d_to_matrix(r6)
        pos, _ = kin(R, zeros, q)
        P = torch.stack([pos[SMPL_TO_G1[i]] for i in ids], 1)
        gb = torch.stack([pos[lb] - pos[la] for _, (la, lb) in LINK_BONES] +
                         [(pos["left_shoulder_roll_link"] + pos["right_shoulder_roll_link"]) / 2 - pos["pelvis"]], 1)
        return q, R, P - P[:, :1], torch.nn.functional.normalize(gb, dim=-1)

    for reg, iters, lr in passes:
        z = torch.logit(((q0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
        r6 = torch.cat([R0[..., 0], R0[..., 1]], -1).clone().requires_grad_(True)
        opt = torch.optim.Adam([z, r6], lr=lr)
        for _ in range(iters):
            q, R, P, d = fk(z, r6)
            loss = (w_dir * ((d - hum_dir) ** 2).sum(-1).sum(-1) + w_rrot * ((R - R_hum) ** 2).sum((-1, -2))
                    + w_pose * (((P - target) ** 2).sum(-1) * w).sum(-1) + reg * (q ** 2).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            q0, R0 = lo + (hi - lo) * torch.sigmoid(z), rot6d_to_matrix(r6)
    with torch.no_grad():
        q, R, P, _ = fk(torch.logit(((q0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)), torch.cat([R0[..., 0], R0[..., 1]], -1))
        res = (P - target).norm(dim=-1).mean(-1)
    return q.detach(), R.detach(), scale.detach(), res
