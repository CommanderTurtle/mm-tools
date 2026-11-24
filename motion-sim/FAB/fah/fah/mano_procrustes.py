"""Closed-form MANO mesh from a dense MANO-topology surface prediction (per-part Procrustes): every MANO bone gets
the rotation that best maps its own template vertices (largest skinning weight) onto the predicted ones (Kabsch),
the joints follow the kinematic chain from the predicted wrist, one scale per hand, mean shape; the mesh is linear
blend skinning of the template with those rigid transforms. A valid, watertight hand that follows the dense surface
(no folded triangles from points sliding along the surface), in well under a millisecond per hand, no iterations.
"""
import os
import pickle

import numpy as np


class ManoProcrustes:
    def __init__(self, mano_dir=None):
        mano_dir = mano_dir or os.environ.get("MANO_DIR", "data/mano")
        m = pickle.load(open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
        dense = lambda x: np.asarray(x.todense() if hasattr(x, "todense") else x, np.float64)
        self.v = dense(m["v_template"])  # (778, 3) flat hand
        self.W = dense(m["weights"])  # (778, 16)
        self.J = dense(m["J_regressor"]) @ self.v  # (16, 3) rest joints
        self.parents = np.asarray(m["kintree_table"][0], np.int64)
        self.parents[0] = -1
        self.part = self.W.argmax(1)
        self.groups = [np.where(self.part == k)[0] for k in range(16)]
        self.faces = np.asarray(m["f"], np.int64)

    @staticmethod
    def kabsch(a, b):
        """Rotation R with R (a - mean a) ~ (b - mean b); a, b (n, 3)."""
        a, b = a - a.mean(0), b - b.mean(0)
        U, _, Vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    def rotations(self, P):
        """Per-part global rotations (16, 3, 3) and the hand scale for a predicted surface P (778, 3); all parts in
        one batched SVD (part covariances from a one-hot part matrix)."""
        P = np.asarray(P, np.float64)
        if not hasattr(self, "_oh"):
            self._oh = np.eye(16)[self.part]  # (778, 16)
            self._cnt = self._oh.sum(0)
            self._vc = self.v - (self._oh @ (self._oh.T @ self.v / self._cnt[:, None]))  # template, centred per part
            self._big = self._cnt >= 20
            self._vs = np.einsum("vk,vc->k", self._oh, self._vc ** 2)
        Pc = P - self._oh @ (self._oh.T @ P / self._cnt[:, None])
        C = (self._oh.T @ (self._vc[:, :, None] * Pc[:, None, :]).reshape(778, 9)).reshape(16, 3, 3)  # part covariances (BLAS)
        U, _, Vt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)))
        D = np.ones((16, 3))
        D[:, 2] = d
        R = Vt.transpose(0, 2, 1) @ (D[..., None] * U.transpose(0, 2, 1))
        s = float(np.median(np.sqrt((self._oh.T @ (Pc ** 2).sum(1))[self._big] / self._vs[self._big])))
        return R, s

    def local_pose(self, R):
        """Global part rotations (16, 3, 3) -> root rotation (3, 3) and MANO hand pose (15 x axis-angle, flat-hand
        convention, 45)."""
        par = self.parents[1:]
        L = R[par].transpose(0, 2, 1) @ R[1:]  # (15, 3, 3) local rotations
        cos = np.clip((np.trace(L, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
        ang = np.arccos(cos)
        w = np.stack([L[:, 2, 1] - L[:, 1, 2], L[:, 0, 2] - L[:, 2, 0], L[:, 1, 0] - L[:, 0, 1]], 1)  # 2 sin(ang) axis
        sn = np.sin(ang)
        f = np.where(sn > 1e-6, ang / (2 * np.maximum(sn, 1e-12)), 0.5)  # log map (angles < pi; ~pi bends do not occur)
        return R[0], (w * f[:, None]).reshape(-1)

    def __call__(self, P, wrist=None):
        """P (778, 3) predicted surface (right-hand frame, camera coordinates); wrist (3,) predicted wrist joint
        (default: the template wrist mapped by the root part). Returns the MANO-topology mesh (778, 3)."""
        P = np.asarray(P, np.float64)
        g0 = self.groups[0]
        spread = lambda x: ((x - x.mean(0)) ** 2).sum()
        s = float(np.median([np.sqrt(spread(P[g]) / spread(self.v[g])) for g in self.groups if len(g) >= 20]))  # rigid parts: pose-invariant size
        R = [None] * 16
        for k in range(16):
            R[k] = self.kabsch(self.v[self.groups[k]], P[self.groups[k]])
        Jp = np.zeros((16, 3))
        if wrist is None:  # the root part's rigid map applied to the rest wrist
            wrist = P[g0].mean(0) + s * R[0] @ (self.J[0] - self.v[g0].mean(0))
        Jp[0] = wrist
        for k in range(1, 16):  # chain: each joint from its parent along the parent's rotated rest bone
            p = self.parents[k]
            Jp[k] = Jp[p] + s * R[p] @ (self.J[k] - self.J[p])
        # LBS: v' = sum_k W[v, k] (s R_k (v - J_k) + J'_k)
        Rs = np.stack(R) * s  # (16, 3, 3)
        loc = self.v[:, None, :] - self.J[None]  # (778, 16, 3)
        posed = np.einsum("kij,vkj->vki", Rs, loc) + Jp[None]
        return (self.W[..., None] * posed).sum(1)
