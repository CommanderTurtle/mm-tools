"""Signed distance of hand vertices to an object mesh: the one label rule of every contact pool / clip.

Exact point-to-triangle distance (Open3D RaycastingScene), negative inside: the generalised winding number > 0.5 (libigl) on
every point inside the mesh's axis-aligned box padded by 1 mm (points outside the box are outside the mesh). A vertex is in
contact iff its signed distance is < 10 mm; penetration counts at any depth. Distances are stored in mm, clipped to +-500.
Used by train.build_hoi4d_contact, train.build_arctic_contact and train.build_arctic_video.
"""
import numpy as np

THRESH_MM = 10.0
CLIP_MM = 500.0


def signed_mm(sc, V, F, local):
    """Unclipped signed distance (mm) of points `local` (n, 3, the mesh frame, metres) to the mesh (V, F) whose Open3D
    RaycastingScene is sc: exact point-to-triangle distance, negative where the winding number > 0.5 (every point inside
    the mesh's AABB + 1 mm is tested)."""
    import igl
    import open3d as o3d

    local = np.ascontiguousarray(local, np.float64)
    d = sc.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy().astype(np.float64)
    k = np.all((local >= V.min(0) - 0.001) & (local <= V.max(0) + 0.001), 1)
    if k.any():
        w = igl.fast_winding_number(np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64), np.ascontiguousarray(local[k]))
        d[k] = np.where(w > 0.5, -d[k], d[k])
    return d * 1000


class MeshSDF:
    """Signed distance (mm) of points to one triangle mesh (V metres, object frame), built once per mesh."""

    def __init__(self, V, F):
        import open3d as o3d

        self.V, self.F = np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64)
        self.sc = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.sc.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))

    def __call__(self, local):
        """local (n, 3) points in the mesh frame (metres) -> signed distance (n,) mm, clipped to +-CLIP_MM."""
        return np.clip(signed_mm(self.sc, self.V, self.F, local), -CLIP_MM, CLIP_MM)

    def posed(self, pts, T):
        """pts (n, 3) camera frame, T (4, 4) object -> camera."""
        return self((np.asarray(pts, np.float64) - T[:3, 3]) @ T[:3, :3])


def mesh_sdf(V, F, pts):
    """One-shot signed distance (mm) of pts to the mesh (V, F) in the same frame."""
    return MeshSDF(V, F)(pts)
