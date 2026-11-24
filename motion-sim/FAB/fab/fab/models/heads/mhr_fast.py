"""Eager evaluation of the MHR TorchScript rig restricted to what the caller needs.

Ported from FAH (github.com/noahcao/FAH, its eager MHR hand rig) and generalised to any vertex subset: skeleton forward kinematics
(EagerSkeleton) plus linear blend skinning and pose correctives of a vertex subset. Numerically matches the
full TorchScript rig on those outputs, and, unlike the TorchScript module, it is
visible to torch.compile and free of host syncs, so it can be fused and captured in CUDA graphs.
"""

import roma
import torch
import torch.nn as nn

from .fk_eager import EagerSkeleton


def compose_skel_state(a, b):
    """Compose similarity transforms stored as [t(3), q_xyzw(4), s(1)] (a after b)."""
    ta, qa, sa = a[..., :3], a[..., 3:7], a[..., 7:]
    tb, qb, sb = b[..., :3], b[..., 3:7], b[..., 7:]
    t = ta + sa * roma.quat_action(qa, tb, is_normalized=True)
    q = roma.quat_product(qa, qb)
    return torch.cat([t, q, sa * sb], dim=-1)


def euler_xyz_to_6d(r):
    """First two columns [R00, R10, R20, R01, R11, R21] of the XYZ-Euler rotation (== mhr_utils.batch6DFromXYZ,
    written without in-place writes into an empty tensor so torch.compile / inductor can lower it)."""
    c, s = torch.cos(r), torch.sin(r)
    cx, cy, cz = c.unbind(-1)
    sx, sy, sz = s.unbind(-1)
    return torch.stack([cy * cz, cy * sz, -sy,
                        -cx * sz + sx * sy * cz, cx * cz + sx * sy * sz, sx * cy], dim=-1)


def pose_features(joint_params):
    """pymomentum MHR _pose_features_from_joint_params in eager PyTorch (the TorchScript version
    host-syncs): 6D features of the euler angles of joints 2.., minus identity."""
    euler = joint_params.reshape(joint_params.shape[0], -1, 7)[:, 2:, 3:6]
    feat = euler_xyz_to_6d(euler)
    feat[:, :, 0] -= 1
    feat[:, :, 4] -= 1
    return feat.flatten(1, 2)


class SubsetMHR(nn.Module):
    """Skeleton FK and the skinned vertices of `vert_idx` (None = all vertices)."""

    def __init__(self, mhr, vert_idx=None):
        super().__init__()
        ct = mhr.character_torch
        num_verts = ct.mesh.rest_vertices.shape[0]
        vert_idx = torch.arange(num_verts) if vert_idx is None else torch.as_tensor(vert_idx, dtype=torch.long)
        self.num_identity = ct.blend_shape.shape_vectors.shape[0]
        self.fk = EagerSkeleton(ct)
        self.register_buffer("vert_idx", vert_idx, False)
        self.register_buffer("base_shape", ct.blend_shape.base_shape.cpu()[vert_idx].clone(), False)
        self.register_buffer("shape_vectors", ct.blend_shape.shape_vectors.cpu()[:, vert_idx].clone(), False)

        # Linear blend skinning restricted to the vertex subset
        lbs = ct.linear_blend_skinning
        remap = -torch.ones(num_verts, dtype=torch.long)
        remap[vert_idx] = torch.arange(len(vert_idx))
        v = remap[lbs.vert_indices_flattened.long().cpu()]
        keep = v >= 0
        self.register_buffer("skin_vert", v[keep], False)
        self.register_buffer("skin_joint", lbs.skin_indices_flattened.long().cpu()[keep], False)
        self.register_buffer("skin_weight", lbs.skin_weights_flattened.cpu()[keep].clone(), False)
        self.register_buffer("inverse_bind_pose", lbs.inverse_bind_pose.cpu().clone(), False)

        # Pose correctives restricted to the subset (sparse -> ReLU -> dense)
        seq = mhr.pose_correctives_model.pose_dirs_predictor
        dense = getattr(seq, "2").weight.detach().cpu()  # (num_verts * 3, hidden)
        rows = (vert_idx[:, None] * 3 + torch.arange(3)[None]).flatten()
        dense = dense[rows]
        hidden = torch.nonzero(dense.abs().sum(0) > 0).flatten()
        sp = getattr(seq, "0")
        w0 = torch.sparse_coo_tensor(
            sp.sparse_indices.cpu(), sp.sparse_weight.detach().cpu(), list(sp.sparse_shape), check_invariants=True
        ).to_dense()
        self.register_buffer("corr_w0", w0[hidden].clone(), False)
        self.register_buffer("corr_w2", dense[:, hidden].clone(), False)
        self.corr_w2_csr = None

    def sparse_correctives(self):
        """Inference: keep the corrective output matrix as CSR (the full mesh's is ~4 % non-zero, 483 MB dense, a
        about 2 ms matrix-vector product per frame). Same products up to FP32 summation order."""
        self.corr_w2_csr = self.corr_w2.to_sparse_csr()
        return self

    def skeleton(self, model_params):
        zeros = torch.zeros(model_params.shape[0], self.num_identity, device=model_params.device)
        return self.fk(torch.cat([model_params, zeros], dim=1))

    def vertices(self, shape_params, joint_params, skel_state, correctives=True):
        batch = shape_params.shape[0]
        rest = self.base_shape[None] + torch.einsum("nvd,bn->bvd", self.shape_vectors, shape_params)
        if correctives:
            hidden = torch.relu(pose_features(joint_params) @ self.corr_w0.T)
            if self.corr_w2_csr is not None:
                rest = rest + (self.corr_w2_csr @ hidden.T).T.reshape(batch, -1, 3)
            else:
                rest = rest + (hidden @ self.corr_w2.T).view(batch, -1, 3)

        transforms = compose_skel_state(skel_state, self.inverse_bind_pose[None])  # B x J x 8
        tk = transforms[:, self.skin_joint]  # B x K x 8
        pk = rest[:, self.skin_vert]  # B x K x 3
        moved = tk[..., :3] + tk[..., 7:] * roma.quat_action(tk[..., 3:7], pk, is_normalized=True)
        out = torch.zeros_like(rest)
        out.index_add_(1, self.skin_vert, moved * self.skin_weight[None, :, None])
        return out
