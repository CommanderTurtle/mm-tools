"""fab-retarget's skeleton mode: the mesh mode's inputs from human joints alone (bench/videomimic/scene_retarget.py --mode
skeleton). The mesh mode reads the fitted SMPL mesh in three places; each gets one joint rule, with the SMPL neutral
template's constants scaled by the input's leg length over the template's (no per-clip tuning):
  1 contacts   per group, spheres at its joints with template radii; the group's lowest point = min over spheres (z - r); in
               contact when that point is < h_contact (4 cm) above the terrain and the joints' centroid moves < v_contact
               (0.15 m/s) in the ground plane (the mesh rule's lowest vertex / vertex centroid)
                 lfoot / rfoot  ankle 7 / 8 and foot 10 / 11, radius = the joint's height above the foot's lowest vertex in the
                                rest pose, + the template's toe-tip point carried in the foot frame (ankle -> toe base, the
                                shin's orthogonal part)
                 lhand / rhand  wrist 20 / 21, hand 22 / 23 (radius: median distance of the hand part's vertices)
                 lknee / rknee  knee 4 / 5 and ankle 7 / 8 of the shin capsule (radius: median distance of the part's vertices
                                to the knee-ankle axis)
  2 seated     the pelvis / thigh spheres (pelvis 0: its height above the pelvis part's lowest vertex; hips 1 / 2 and knees
               4 / 5: the thigh capsule radius): their lowest point above the terrain
  3 root_init  each foot's lowest point of rule 1, at the ankle / foot joint centre
Joints: SMPL-24 order, metres, z-up world (docs/retarget.md: the input contract and a map for other skeletons).
"""
import numpy as np

from fab import paths

CENTROID = {"lfoot": (7, 10), "rfoot": (8, 11), "lhand": (20, 22), "rhand": (21, 23), "lknee": (4, 7), "rknee": (5, 8)}
LEG = ((1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11))  # hip -> knee -> ankle -> foot, both sides
_TEMPLATE = {}


def skeleton_template():
    """The rules' constants from the SMPL neutral template (rest pose, y up): {groups: {group: [(joint, radius m)]}, tips:
    {foot: (ankle, foot, knee, a, b, r)}, leg: the template's leg length (m, one side)}."""
    if not _TEMPLATE:
        from bench.body_models import load_smpl_pkl

        sm = load_smpl_pkl(paths.smpl_dir() / "SMPL_NEUTRAL.pkl")
        v = sm["v_template"].astype(np.float64)
        J = (sm["J_regressor"] @ v).astype(np.float64)
        part = sm["weights"].argmax(1)

        def low(j, parts):  # the joint's height above the parts' lowest vertex (rest pose, y up)
            return float(J[j, 1] - v[np.isin(part, parts), 1].min())

        def blob(j, parts):  # median distance of the parts' vertices to the joint
            return float(np.median(np.linalg.norm(v[np.isin(part, parts)] - J[j], axis=1)))

        def tube(a, b, parts):  # median distance of the parts' vertices to the a -> b segment
            P = v[np.isin(part, parts)]
            d = J[b] - J[a]
            t = np.clip((P - J[a]) @ d / (d @ d), 0, 1)
            return float(np.median(np.linalg.norm(P - (J[a] + t[:, None] * d), axis=1)))

        def tip(an, ft, kn):  # the foot's front-most vertex in the foot frame (u = ankle -> toe base, n = the shin's part
            # orthogonal to u): (a, b) with tip = toe base + a u + b n, and its height above the foot's lowest vertex
            P = v[np.isin(part, (an, ft))]
            u = (J[ft] - J[an]) / np.linalg.norm(J[ft] - J[an])
            sh = J[kn] - J[an]
            nn = sh - (sh @ u) * u
            nn /= np.linalg.norm(nn)
            k_ = int(np.argmax(P[:, 2]))  # the template faces +z
            d = P[k_] - J[ft]
            return float(d @ u), float(d @ nn), float(P[k_, 1] - P[:, 1].min())

        g, tips = {}, {}
        for side, (hip, kn, an, ft, wr, hd) in (("l", (1, 4, 7, 10, 20, 22)), ("r", (2, 5, 8, 11, 21, 23))):
            g[side + "foot"] = [(an, low(an, (an, ft))), (ft, low(ft, (an, ft)))]
            tips[side + "foot"] = (an, ft, kn) + tip(an, ft, kn)
            g[side + "hand"] = [(wr, blob(wr, (wr, hd))), (hd, blob(hd, (wr, hd)))]
            r_sh = tube(kn, an, (kn,))
            g[side + "knee"] = [(kn, r_sh), (an, r_sh)]
            r_th = tube(hip, kn, (hip,))
            g.setdefault("seat", [(0, low(0, (0,)))])
            g["seat"] += [(hip, r_th), (kn, r_th)]
        leg = np.mean([np.linalg.norm(J[a] - J[b]) for a, b in LEG]) * 3
        _TEMPLATE.update(groups=g, tips=tips, leg=float(leg))
    return _TEMPLATE


def skeleton_rules(J, terr_h, fps, h_contact, v_contact):
    """J (T, 24, 3) input-rate joints, z-up; terr_h = scene_metrics.HF(hfield) -> contacts {group: (T,) bool} (rule 1),
    pelvis_h (T,) the seat spheres' lowest point above the terrain (rule 2), foot_low / foot_xy {lfoot, rfoot} (rule 3), info."""
    tp = skeleton_template()
    leg = np.median(np.sum([np.linalg.norm(J[:, a] - J[:, b], axis=-1) for a, b in LEG], axis=0)) / 2
    k = float(leg / tp["leg"])  # one side's hip -> knee -> ankle -> foot length, input / template

    def lowest(spheres, extra=None):  # (T,) absolute z of the lowest sphere bottom, (T,) its height above the terrain under it;
        # extra: (T, 3) one more point (the toe tip) with its radius
        pts = [(J[:, j], k * r) for j, r in spheres] + ([extra] if extra is not None else [])
        zs = np.stack([p_[:, 2] - r for p_, r in pts], 1)
        hs = np.stack([zs[:, i] - terr_h(p_[:, :2], zs[:, i]) for i, (p_, _) in enumerate(pts)], 1)
        return zs.min(1), hs.min(1)

    def toe_tip(g):  # the template's toe-tip point carried by the ankle / toe-base / knee joints (foot frame, scaled by k)
        an, ft, kn, a_, b_, r_ = tp["tips"][g]
        u = J[:, ft] - J[:, an]
        u /= np.linalg.norm(u, axis=-1, keepdims=True).clip(1e-6)
        sh = J[:, kn] - J[:, an]
        nn = sh - (sh * u).sum(-1, keepdims=True) * u
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True).clip(1e-6)
        return J[:, ft] + k * (a_ * u + b_ * nn), k * r_

    con, foot_low, foot_xy = {}, {}, {}
    for g, sph in tp["groups"].items():
        if g == "seat":
            continue
        zlow, h = lowest(sph, toe_tip(g) if g in tp["tips"] else None)
        c = J[:, list(CENTROID[g]), :2].mean(1)
        v = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=-1) * fps]
        con[g] = (h < h_contact) & (v < v_contact)
        if g.endswith("foot"):
            foot_low[g], foot_xy[g] = zlow, c
    _, hp = lowest(tp["groups"]["seat"])
    info = {"k_scale": k, "h_contact": h_contact, "v_contact": v_contact,
            "radii_m": {g: [[int(j), round(k * r, 4)] for j, r in s] for g, s in tp["groups"].items()},
            "toe_tip": {g: [round(float(x), 4) for x in t[3:]] for g, t in tp["tips"].items()},
            "contact_share": {g: float(np.mean(v)) for g, v in con.items()}}
    return {"contacts": con, "pelvis_h": hp, "foot_low": foot_low, "foot_xy": foot_xy, "info": info}
