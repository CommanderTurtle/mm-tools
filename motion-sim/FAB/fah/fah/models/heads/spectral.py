"""Non-parametric smooth surface for the dense keypoint model: the K lowest-frequency eigenfunctions of the template
mesh's cotangent Laplacian (generalised problem L u = lambda M u, M = lumped vertex areas) span smooth functions on
the surface; projecting the 778 predicted surface points (3D or 2D) onto them is a fixed low-pass filter (no shape
model -- any smooth surface is representable; spikes are high-frequency). K = 256 reconstructs real hand meshes
(HOT3D GT) to 0.61 mm mean / 1.7 mm p95. A weighted least-squares fit in the basis lets hidden / low-confidence
points take their values from their neighbours. The basis is computed at run time from the template's
connectivity and rest geometry (nothing of a body model is stored in checkpoints)."""
import numpy as np
import torch

_CACHE = {}


def cotangent_basis(X, F, k):
    """X (V, 3) rest vertices, F (T, 3) faces -> (U (V, k) M-orthonormal eigenfunctions, M (V,) vertex areas)."""
    import scipy.linalg

    key = (X.shape, F.shape, float(np.abs(X).sum()), k)
    if key not in _CACHE:
        n = len(X)
        L, M = np.zeros((n, n)), np.zeros(n)
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            i, j, o = F[:, a], F[:, b], F[:, c]
            u, v = X[i] - X[o], X[j] - X[o]
            cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
            np.add.at(L, (i, j), -0.5 * cot)
            np.add.at(L, (j, i), -0.5 * cot)
        L[np.diag_indices(n)] = -L.sum(1)
        area = np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1) / 2
        for a in range(3):
            np.add.at(M, F[:, a], area / 3)
        _, U = scipy.linalg.eigh(L, np.diag(M))
        # rescaled so that U^T diag(M / mean M) U = I (the area weights are normalised to mean 1)
        _CACHE[key] = ((U[:, :k] * np.sqrt(M.mean())).astype(np.float32), (M / M.mean()).astype(np.float32))
    return _CACHE[key]


def smooth_surface(x, basis, w=None, lam=1e-4):
    """x (B, V, C) per-vertex values -> their low-pass projection: c = argmin sum_i m_i w_i |U_i c - x_i|^2 + lam |c|^2,
    returns U c. basis = (U, M) from cotangent_basis; w (B, V) optional per-point weights (visibility / confidence)."""
    U, M = basis
    U = torch.as_tensor(U, device=x.device, dtype=torch.float32)
    m = torch.as_tensor(M, device=x.device, dtype=torch.float32)[None]
    if w is None:  # U is M-orthonormal: plain projection
        return U[None] @ (U.T[None] @ (m[..., None] * x.float()))
    w = m * w.float()
    UW = U.T[None] * w[:, None, :]  # (B, k, V)
    H = UW @ U[None] + lam * torch.eye(U.shape[1], device=x.device)
    return U[None] @ torch.linalg.solve(H, UW @ x.float())


def template_basis(mano_dir, k=256):
    """Basis on the MANO-topology template (right hand; the same connectivity serves mirrored left hands)."""
    from bench.common import ManoLayer

    ml = ManoLayer(mano_dir)
    _, rest = ml(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
    return cotangent_basis(rest.astype(np.float64), ml.faces[True], k)
