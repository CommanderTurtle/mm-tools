"""Named extra dense-head surface points: appended after the farthest-point
surface set (dense_head.symmetric_subset), so a head initialised from a 468-point model keeps its points in place
(train_fab --dense_remap starts the new ones from their nearest old point). The set stays closed under the template's
left <-> right mirror map (flip augmentation).
  sole  per foot: heel, big toe, small toe + farthest-point samples of the sole (vertices skinned
        to the ankle / foot joints within 1.5 cm of the foot's lowest template point)
  tips  the 5 fingertips per hand (SMPL-H vertex ids, smplx vertex_ids["smplh"])
"""

import numpy as np

from fab.dense_head import farthest_points, mirror_map

FOOT_L = (3387, 3216, 3226)  # heel, big toe, small toe (left)
TIPS_L = (2746, 2319, 2445, 2556, 2673)  # thumb, index, middle, ring, pinky (left)
LEFT_FOOT_JOINTS = (7, 10)  # SMPL left ankle, left foot


def sole_points(verts, weights, k=8, tol=0.015):
    """Left-foot sole vertices: k farthest-point samples of the lowest vertices skinned to the left ankle / foot."""
    verts, part = np.asarray(verts, np.float64), np.asarray(weights).argmax(1)
    foot = np.nonzero(np.isin(part, LEFT_FOOT_JOINTS))[0]
    sole = foot[verts[foot, 1] <= verts[foot, 1].min() + tol]
    return sole[farthest_points(verts[sole], min(k, len(sole)))]


def extra_points(verts, weights, idx, names, k_sole=8):
    """(idx, mirror positions) of the surface set idx with the named points appended (left ones and their mirrors,
    those not in idx yet), as dense_head.symmetric_subset returns them."""
    mirror, _, _ = mirror_map(verts)
    left = []
    for name in names:
        if name == "sole":
            left += list(FOOT_L) + [int(v) for v in sole_points(verts, weights, k_sole)]
        elif name == "tips":
            left += list(TIPS_L)
        elif name:
            raise ValueError(f"unknown dense extra points {name!r} (sole, tips)")
    out = [int(v) for v in idx]
    have = set(out)
    for v in left:
        for u in (v, int(mirror[v])):
            if u not in have:
                out.append(u)
                have.add(u)
    while True:  # closed under the mirror map
        more = [int(mirror[v]) for v in out if int(mirror[v]) not in have]
        if not more:
            break
        for u in more:
            if u not in have:
                out.append(u)
                have.add(u)
    pos = {v: i for i, v in enumerate(out)}
    return np.asarray(out), np.asarray([pos[int(mirror[v])] for v in out])


def foot_positions(verts, weights, idx):
    """Positions in the surface list idx of the foot points (skinned to an ankle / foot joint, below the ankle joint
    height of the template): the points whose 3D the training takes only from trusted GT (--dense_foot_untrusted)."""
    verts, part = np.asarray(verts, np.float64), np.asarray(weights).argmax(1)
    idx = np.asarray(idx)
    foot = np.isin(part[idx], (7, 8, 10, 11))
    ank = np.asarray(weights)[:, [7, 8]].max(1) > 0.5  # mostly-ankle vertices: their height bounds the foot from above
    y_ank = np.median(verts[ank, 1])
    return np.nonzero(foot & (verts[idx, 1] <= y_ank))[0]
