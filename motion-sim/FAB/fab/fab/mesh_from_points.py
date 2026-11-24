"""Full SMPL mesh from the dense head's surface points (a parameter-free mesh output): a linear map A (6890 x K) with
V ~= A P, fitted by ridge regression on posed SMPL bodies. Both sides are centred on
the points' mean, so the map commutes with translation."""

import numpy as np
import torch


def fit_map(V, idx, lam=1e-3):
    """V (N, 6890, 3) meshes, idx (K,) point vertices -> A (6890, K)."""
    V = V - V[:, idx].mean(1, keepdims=True)
    P = V[:, idx]  # (N, K, 3)
    Y = V.transpose(1, 0, 2).reshape(V.shape[1], -1)  # (6890, 3N)
    X = P.transpose(1, 0, 2).reshape(len(idx), -1)  # (K, 3N)
    G = X @ X.T
    A = (Y @ X.T) @ np.linalg.inv(G + lam * np.trace(G) / len(idx) * np.eye(len(idx)))
    return A


class PointsToMesh(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(np.asarray(A), dtype=torch.float32))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["A"]), z["idx"]

    def forward(self, points):
        """points (B, K, 3) -> (B, 6890, 3)."""
        c = points.float().mean(1, keepdim=True)
        return torch.einsum("vk,bkc->bvc", self.A, points.float() - c) + c
