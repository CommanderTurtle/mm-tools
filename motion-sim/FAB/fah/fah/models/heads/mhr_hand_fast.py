import numpy as np
import roma
import torch
import torch.nn as nn

MHR_RIGHT_WRIST_TWIST_JOINT = 41
MHR_RIGHT_WRIST_JOINT = 42


def hand_vertex_mask(mhr, thresh=0.5, include_wrist_twist=True):
    """Vertices of the MHR mesh skinned (mostly) to the right hand joints."""
    skel = mhr.character_torch.skeleton
    lbs = mhr.character_torch.linear_blend_skinning
    parents = skel.joint_parents.cpu().numpy()
    hand_joints = {MHR_RIGHT_WRIST_JOINT}
    changed = True
    while changed:
        changed = False
        for j, p in enumerate(parents):
            if p in hand_joints and j not in hand_joints:
                hand_joints.add(j)
                changed = True
    if include_wrist_twist:
        hand_joints.add(MHR_RIGHT_WRIST_TWIST_JOINT)

    joint_idx = lbs.skin_indices_flattened.long().cpu().numpy()
    weights = lbs.skin_weights_flattened.cpu().numpy()
    vert_idx = lbs.vert_indices_flattened.cpu().numpy()
    num_verts = mhr.character_torch.mesh.rest_vertices.shape[0]
    sel = np.isin(joint_idx, sorted(hand_joints))
    weight_sum = np.zeros(num_verts, dtype=np.float32)
    np.add.at(weight_sum, vert_idx[sel], weights[sel])
    return weight_sum > thresh


def compose_skel_state(a, b):
    """Compose similarity transforms stored as [t(3), q_xyzw(4), s(1)] (a after b)."""
    ta, qa, sa = a[..., :3], a[..., 3:7], a[..., 7:]
    tb, qb, sb = b[..., :3], b[..., 3:7], b[..., 7:]
    t = ta + sa * roma.quat_action(qa, tb, is_normalized=True)
    q = roma.quat_product(qa, qb)
    return torch.cat([t, q, sa * sb], dim=-1)



# pose_features and HandMHR port parts of pymomentum's MHR rig (https://github.com/facebookresearch/momentum; MIT License,
# Copyright (c) Meta Platforms, Inc. and affiliates) and of the MHR TorchScript model (https://github.com/facebookresearch/MHR;
# Apache License 2.0); THIRD_PARTY_NOTICES.md
def pose_features(joint_params):
    """pymomentum MHR _pose_features_from_joint_params in eager PyTorch (the TorchScript version
    host-syncs): 6D features of the euler angles of joints 2.., minus identity."""
    from ..modules.mhr_utils import batch6DFromXYZ

    euler = joint_params.reshape(joint_params.shape[0], -1, 7)[:, 2:, 3:6]
    feat = batch6DFromXYZ(euler)
    feat[:, :, 0] -= 1
    feat[:, :, 4] -= 1
    return feat.flatten(1, 2)

class HandMHR(nn.Module):
    """Evaluates only what the hand decoder needs from the MHR TorchScript rig:
    skeleton FK (joint states) and, optionally, the skinned vertices of a subset of
    vertices (the hand). Numerically matches the full rig on those outputs."""

    def __init__(self, mhr, vert_idx):
        super().__init__()
        ct = mhr.character_torch
        vert_idx = torch.as_tensor(vert_idx, dtype=torch.long)
        self.num_identity = ct.blend_shape.shape_vectors.shape[0]
        # eager port of the TorchScript FK (fusable by torch.compile; checked by scripts/check_fk_eager.py)
        from .fk_eager import EagerSkeleton

        self.fk = EagerSkeleton(ct)
        self.register_buffer("vert_idx", vert_idx, False)
        self.register_buffer("base_shape", ct.blend_shape.base_shape.cpu()[vert_idx].clone(), False)
        self.register_buffer(
            "shape_vectors", ct.blend_shape.shape_vectors.cpu()[:, vert_idx].clone(), False
        )

        # Linear blend skinning restricted to the vertex subset
        lbs = ct.linear_blend_skinning
        remap = -torch.ones(ct.mesh.rest_vertices.shape[0], dtype=torch.long)  # CPU
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
        W0 = torch.sparse_coo_tensor(
            sp.sparse_indices.cpu(), sp.sparse_weight.detach().cpu(), list(sp.sparse_shape), check_invariants=True
        ).to_dense()
        self.register_buffer("corr_w0", W0[hidden].clone(), False)
        self.register_buffer("corr_w2", dense[:, hidden].clone(), False)

    def skeleton(self, mhr, model_params):
        ct = mhr.character_torch
        zeros = torch.zeros(
            model_params.shape[0], self.num_identity, device=model_params.device
        )
        joint_params, skel_state = self.fk(torch.cat([model_params, zeros], dim=1))
        return joint_params, skel_state

    def vertices(self, mhr, shape_params, joint_params, skel_state):
        B = shape_params.shape[0]
        rest = self.base_shape[None] + torch.einsum(
            "nvd,bn->bvd", self.shape_vectors, shape_params
        )
        feats = pose_features(joint_params)
        hidden = torch.relu(feats @ self.corr_w0.T)
        rest = rest + (hidden @ self.corr_w2.T).view(B, -1, 3)

        T = compose_skel_state(skel_state, self.inverse_bind_pose[None])  # B x J x 8
        Tk = T[:, self.skin_joint]  # B x K x 8
        pk = rest[:, self.skin_vert]  # B x K x 3
        moved = Tk[..., :3] + Tk[..., 7:] * roma.quat_action(
            Tk[..., 3:7], pk, is_normalized=True
        )
        out = torch.zeros_like(rest)
        out.index_add_(1, self.skin_vert, moved * self.skin_weight[None, :, None])
        return out
