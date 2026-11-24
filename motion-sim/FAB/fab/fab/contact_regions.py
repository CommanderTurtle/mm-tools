"""Contact channels on the SMPL template: left / right heel and toe, hands, knees (CONTACT_NAMES order)."""
from pathlib import Path

import numpy as np

from fab import paths

CONTACT_NAMES = ["l_heel", "l_toe", "r_heel", "r_toe", "l_hand", "r_hand", "l_knee", "r_knee"]


def smpl_regions(bm_dir="", halves=False):
    """Vertex index lists of the contact channels on the SMPL template (y up, z forward). halves: the whole rear / front
    half of each foot instead of the sole ends (for per-vertex contact labels such as RICH's human-scene contact)."""
    from bench.body_models import load_smpl_pkl

    m = load_smpl_pkl((Path(bm_dir) if bm_dir else paths.body_models_dir()) / "smpl" / "SMPL_NEUTRAL.pkl")
    v, part = m["v_template"].astype(np.float64), m["weights"].argmax(1)
    J = m["J_regressor"] @ v
    out = []
    for parts in ((7, 10), (8, 11)):
        ix = np.flatnonzero(np.isin(part, parts))
        y, z = v[ix, 1], v[ix, 2]
        sole = ix[y < y.min() + 0.02]
        zs = v[sole, 2]
        L = z.max() - z.min()
        if halves:
            out += [ix[z < z.min() + 0.5 * L], ix[z >= z.min() + 0.5 * L]]
            continue
        out += [sole[zs < z.min() + 0.3 * L], sole[zs > z.min() + 0.7 * L]]
    for parts in ((20, 22), (21, 23)):  # hands: every 4th vertex is plenty for the lowest point / centroid
        out.append(np.flatnonzero(np.isin(part, parts))[::4])
    for j in (4, 5):
        d = np.linalg.norm(v - J[j], axis=1)
        out.append(np.flatnonzero((d < 0.07) & (v[:, 2] > J[j, 2])))
    return out
