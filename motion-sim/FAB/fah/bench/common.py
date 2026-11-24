"""Shared benchmark utilities: MANO GT layer, prediction format, matching and metrics.

Conventions (all datasets / methods):
  - camera frame: x right, y down, z forward, meters
  - 21 hand joints in OpenPose order (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4),
    MANO definition: 16 MANO joints + 5 fingertip vertices
  - vertices: MANO topology (778) when a method provides them, else None
Per-frame prediction = list of dicts:
  {"is_right": bool, "score": float, "bbox": (4,) xyxy, "joints3d": (21,3) | None,
   "verts3d": (778,3) | None, "joints2d": (21,2) | None}
"""
import numpy as np
import torch

# smplx MANO joint order: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
# + appended fingertips 16 thumb, 17 index, 18 middle, 19 ring, 20 pinky
MANO21_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
# fingertip vertex ids used by HOT3D (thumb, index, middle, ring, pinky)
MANO_TIP_VERTS = [744, 320, 443, 554, 671]


class ManoLayer:
    """MANO forward kinematics (smplx) for both hands -> 21 OpenPose joints + 778 verts."""

    def __init__(self, model_dir, num_pca_comps=None, flat_hand_mean=False):
        import smplx

        use_pca = num_pca_comps is not None
        kw = dict(use_pca=use_pca, flat_hand_mean=flat_hand_mean)
        if use_pca:
            kw["num_pca_comps"] = num_pca_comps
        self.layers = {
            True: smplx.create(f"{model_dir}/MANO_RIGHT.pkl", "mano", is_rhand=True, **kw),
            False: smplx.create(f"{model_dir}/MANO_LEFT.pkl", "mano", is_rhand=False, **kw),
        }
        # smplx left-hand shapedirs bug (https://github.com/vchoutas/smplx/issues/48)
        l, r = self.layers[False], self.layers[True]
        if torch.sum(torch.abs(l.shapedirs[:, 0, :] - r.shapedirs[:, 0, :])) < 1:
            l.shapedirs[:, 0, :] *= -1
        self.faces = {k: v.faces.astype(np.int64) for k, v in self.layers.items()}

    @torch.no_grad()
    def __call__(self, is_right, betas, hand_pose, global_orient, transl):
        def t(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)

        out = self.layers[bool(is_right)](
            betas=t(betas), hand_pose=t(hand_pose), global_orient=t(global_orient), transl=t(transl)
        )
        verts = out.vertices[0].numpy()
        joints = np.concatenate([out.joints[0, :16].numpy(), verts[MANO_TIP_VERTS]], 0)
        return joints[MANO21_TO_OPENPOSE], verts


# ---------------------------------------------------------------------------- alignment
def procrustes(pred, gt):
    """Similarity-align pred (N,3) to gt (N,3); returns aligned pred."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p, g = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(p.T @ g)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / max((p**2).sum(), 1e-12)
    return s * p @ R.T + mu_g


def procrustes_params(pred, gt):
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p, g = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(p.T @ g)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / max((p**2).sum(), 1e-12)
    return s, R, mu_p, mu_g


def f_score(pred, gt, thresh):
    """FreiHAND-style F-score between two point sets (after alignment)."""
    from scipy.spatial import cKDTree

    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    precision = (d_pg < thresh).mean()
    recall = (d_gp < thresh).mean()
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def hand_metrics(pred_j, gt_j, pred_v=None, gt_v=None):
    """Per-hand errors in mm (F-scores in [0,1])."""
    m = {
        "pa_mpjpe": np.linalg.norm(procrustes(pred_j, gt_j) - gt_j, axis=1).mean() * 1000,
        "rr_mpjpe": np.linalg.norm((pred_j - pred_j[:1]) - (gt_j - gt_j[:1]), axis=1).mean() * 1000,
        "cam_mpjpe": np.linalg.norm(pred_j - gt_j, axis=1).mean() * 1000,
        "root_err": np.linalg.norm(pred_j[0] - gt_j[0]) * 1000,
    }
    if pred_v is not None and gt_v is not None:
        # align with the vertex-based similarity (FreiHAND protocol)
        pv = procrustes(pred_v, gt_v)
        m["pa_mpvpe"] = np.linalg.norm(pv - gt_v, axis=1).mean() * 1000
        m["f5"] = f_score(pv, gt_v, 0.005)
        m["f15"] = f_score(pv, gt_v, 0.015)
    return m


# ---------------------------------------------------------------------------- matching
def box_iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    area = lambda r: max(r[2] - r[0], 0) * max(r[3] - r[1], 0)
    return inter / (area(a) + area(b) - inter + 1e-9)


def match_hands(preds, gts, iou_thresh=0.3):
    """Greedy IoU matching of predicted to GT hands, ignoring handedness (so handedness
    errors are measured, not hidden). Returns list of (gt_idx, pred_idx or None)."""
    pairs = sorted(
        ((box_iou(p["bbox"], g["bbox"]), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds)),
        reverse=True,
    )
    used_g, used_p, out = set(), set(), {}
    for iou, gi, pi in pairs:
        if iou < iou_thresh or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out[gi] = pi
    return [(gi, out.get(gi)) for gi in range(len(gts))], len(preds) - len(used_p)


def accel_error(pred_seq, gt_seq):
    """Mean acceleration error (mm/frame^2) of root-relative joints over a contiguous track."""
    if len(pred_seq) < 3:
        return None
    p = pred_seq - pred_seq[:, :1]
    g = gt_seq - gt_seq[:, :1]
    acc_p = p[2:] - 2 * p[1:-1] + p[:-2]
    acc_g = g[2:] - 2 * g[1:-1] + g[:-2]
    return float(np.linalg.norm(acc_p - acc_g, axis=-1).mean() * 1000)


class VertexJoints:
    """Identical joint definition for every MANO-topology mesh (GT and predictions): MANO's
    16-joint regressor applied to the posed vertices + HOT3D fingertip vertices, OpenPose order.
    Removes per-method joint/fingertip convention differences from joint metrics."""

    def __init__(self, model_dir):
        import pickle

        d = pickle.load(open(f"{model_dir}/MANO_RIGHT.pkl", "rb"))
        J = d["J_regressor"]
        self.J = np.asarray(J.todense() if hasattr(J, "todense") else J, dtype=np.float64)  # (16, 778)

    def __call__(self, verts):
        j = np.concatenate([self.J @ verts, verts[MANO_TIP_VERTS]], 0)
        return j[MANO21_TO_OPENPOSE]
