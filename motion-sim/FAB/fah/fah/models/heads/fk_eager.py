"""Eager-PyTorch port of the MHR (pymomentum) TorchScript skeleton FK, so torch.compile / inductor can
fuse it (TorchScript calls are opaque to dynamo). Numerically checked against the TorchScript rig by
scripts/check_fk_eager.py. Skeleton state per joint: [t (3), q (4, xyzw), s (1)]. Ported from pymomentum
(https://github.com/facebookresearch/momentum; MIT License, Copyright (c) Meta Platforms, Inc. and affiliates) and the
MHR rig (https://github.com/facebookresearch/MHR; Apache License 2.0); THIRD_PARTY_NOTICES.md."""
import torch


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], -1)


def quat_rotate(q, v):
    qv, w = q[..., :3], q[..., 3:]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def euler_xyz_to_quat(e):
    """Quaternion of R = Rz(e2) @ Ry(e1) @ Rx(e0) (x applied first)."""
    h = e * 0.5
    c, s = torch.cos(h), torch.sin(h)
    zero = torch.zeros_like(c[..., :1])
    qx = torch.cat([s[..., 0:1], zero, zero, c[..., 0:1]], -1)
    qy = torch.cat([zero, s[..., 1:2], zero, c[..., 1:2]], -1)
    qz = torch.cat([zero, zero, s[..., 2:3], c[..., 2:3]], -1)
    return quat_mul(qz, quat_mul(qy, qx))


def compose(a, b):
    """Skeleton-state product a * b (b expressed in a's frame)."""
    ta, qa, sa = a[..., :3], a[..., 3:7], a[..., 7:]
    tb, qb, sb = b[..., :3], b[..., 3:7], b[..., 7:]
    return torch.cat([ta + sa * quat_rotate(qa, tb), quat_mul(qa, qb), sa * sb], -1)


class EagerSkeleton(torch.nn.Module):
    """model params -> joint params -> local states -> global states (prefix scan over `pmi`)."""

    def __init__(self, character_torch):
        super().__init__()
        ct = character_torch
        sk = ct.skeleton
        self.register_buffer("param_transform", ct.parameter_transform.parameter_transform.detach().cpu().clone(), False)
        self.register_buffer("offsets", sk.joint_translation_offsets.detach().cpu().clone(), False)
        self.register_buffer("prerot", sk.joint_prerotations.detach().cpu().clone(), False)
        levels = torch.split(sk.pmi.detach().cpu(), list(sk._pmi_buffer_sizes), 1)
        for k, lv in enumerate(levels):
            self.register_buffer(f"src{k}", lv[0].clone(), False)
            self.register_buffer(f"tgt{k}", lv[1].clone(), False)
        self.num_levels = len(levels)

    def joint_params(self, model_params):
        return torch.einsum("dn,...n->...d", self.param_transform, model_params)

    def local_state(self, joint_params):
        jp = joint_params.reshape(*joint_params.shape[:-1], -1, 7)
        t = jp[..., :3] + self.offsets
        q = quat_mul(self.prerot.expand(*jp.shape[:-1], 4), euler_xyz_to_quat(jp[..., 3:6]))
        s = torch.exp(jp[..., 6:7] * 0.69314718246459961)
        return torch.cat([t, q, s], -1)

    def global_state(self, local):
        g = local
        for k in range(self.num_levels):
            src, tgt = getattr(self, f"src{k}"), getattr(self, f"tgt{k}")
            g = g.index_copy(-2, src, compose(g.index_select(-2, tgt), g.index_select(-2, src)))
        return g

    def forward(self, model_params):
        jp = self.joint_params(model_params)
        return jp, self.global_state(self.local_state(jp))
