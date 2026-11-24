"""MHR (LOD1, 18439 vertices) -> SMPL topology (6890 vertices) by MHR's official barycentric mapping.

Uses tools/mhr_smpl_conversion/assets from facebookresearch/MHR: `mhr2smpl_mapping.npz` gives, for every SMPL
vertex, an MHR triangle id and barycentric weights; the triangle's vertex ids come from `mhr_face_mask.ply`.
Deterministic and fit-free, so any model that outputs MHR vertices is scored the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh


class MHRToSMPL(torch.nn.Module):
    """target "smpl" (6890 vertices) or "smplx" (10475 vertices, mhr2smplx_mapping.npz)."""

    def __init__(self, mhr_assets_dir: str | Path, target: str = "smpl"):
        super().__init__()
        assets = Path(mhr_assets_dir)
        mapping = np.load(assets / f"mhr2{target}_mapping.npz")
        faces = trimesh.load(assets / "mhr_face_mask.ply", process=False).faces
        vert_ids = faces[mapping["triangle_ids"]]  # [6890, 3]
        self.register_buffer("vert_ids", torch.from_numpy(vert_ids.astype(np.int64)))
        self.register_buffer("bary", torch.from_numpy(mapping["baryc_coords"].astype(np.float32)))

    def forward(self, mhr_verts: torch.Tensor) -> torch.Tensor:
        """[..., 18439, 3] -> [..., 6890 (SMPL) or 10475 (SMPL-X), 3]."""
        tri = mhr_verts[..., self.vert_ids, :]  # [..., 6890, 3, 3]
        return (tri * self.bary[..., None]).sum(dim=-2)
