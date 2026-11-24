"""Host-sync-free versions of ops used in the decoder/rig hot path.

Boolean-mask indexing, torch.nonzero and indexing CUDA tensors with CPU index tensors each force a
device->host sync (or a pageable host->device copy). Besides stalling the CPU, they make CUDA-graph
capture impossible. The functions here are numerically identical replacements.
"""
import math

import roma
import torch

_IDX_CACHE = {}


def dev_index(idx, ref):
    """CPU index list / long tensor / bool mask -> cached long index tensor on ref's device.
    The first call per (index, device) copies to the device; later calls (and CUDA-graph replays)
    reuse the cached tensor."""
    if torch.is_tensor(idx) and idx.device == ref.device and idx.dtype != torch.bool:
        return idx
    t = torch.as_tensor(idx).detach().cpu()
    if t.dtype == torch.bool:
        t = t.nonzero().flatten()
    t = t.long()
    key = (t.numpy().tobytes(), tuple(t.shape), str(ref.device))
    out = _IDX_CACHE.get(key)
    if out is None:
        out = _IDX_CACHE[key] = t.to(ref.device)
    return out


# rotmat_to_unitquat and unitquat_to_euler adapt RoMa (https://github.com/naver/roma; BSD 3-Clause License, Copyright (c)
# 2020 NAVER Corp.; THIRD_PARTY_NOTICES.md)
def rotmat_to_unitquat(R):
    """roma.rotmat_to_unitquat (XYZW) without nonzero(): all four candidates, then gather."""
    m, batch_shape = roma.internal.flatten_batch_dims(R, end_dim=-3)
    diag = m.diagonal(dim1=1, dim2=2)
    tr = diag.sum(1)
    choices = torch.cat([diag, tr[:, None]], 1).argmax(1)
    cands = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        q = [None] * 4
        q[i] = 1 - tr + 2 * m[:, i, i]
        q[j] = m[:, j, i] + m[:, i, j]
        q[k] = m[:, k, i] + m[:, i, k]
        q[3] = m[:, k, j] - m[:, j, k]
        cands.append(torch.stack(q, 1))
    cands.append(torch.stack([m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1], 1 + tr], 1))
    quat = torch.stack(cands, 1).gather(1, choices[:, None, None].expand(-1, 1, 4))[:, 0]
    quat = quat / torch.norm(quat, dim=1)[:, None]
    return roma.internal.unflatten_batch_dims(quat, batch_shape)


def unitquat_to_euler(convention, quat, epsilon=1e-7):
    """roma.unitquat_to_euler (radians, stacked) with torch.where instead of masked assignment."""
    pi, lamb = math.pi, math.pi / 2
    extrinsic = convention.islower()
    if not extrinsic:
        convention = convention.lower()[::-1]
    quat, batch_shape = roma.internal.flatten_batch_dims(quat, end_dim=-2)
    i, j, k = ("xyz".index(c) for c in convention)
    symmetric = i == k
    if symmetric:
        k = 3 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2
    if symmetric:
        a, b, c, d = quat[:, 3], quat[:, i], quat[:, j], quat[:, k] * sign
    else:
        a = quat[:, 3] - quat[:, j]
        b = quat[:, i] + quat[:, k] * sign
        c = quat[:, j] + quat[:, 3]
        d = quat[:, k] * sign - quat[:, i]
    first, third = (0, 2) if extrinsic else (2, 0)
    angles = [None] * 3
    angles[1] = 2 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))
    case1 = torch.abs(angles[1]) <= epsilon
    case2 = torch.abs(angles[1] - pi) <= epsilon
    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)
    angles[first] = half_sum - half_diff
    angles[third] = half_sum + half_diff
    zero = torch.zeros_like(half_sum)
    angles[2] = torch.where(case1 | case2, zero, angles[2])
    angles[0] = torch.where(case1, 2 * half_sum, angles[0])
    angles[0] = torch.where(case2, 2 * (-1 if extrinsic else 1) * half_diff, angles[0])
    if not symmetric:
        angles[third] = angles[third] * sign
        angles[1] = angles[1] - lamb
    out = []
    for x in angles:
        x = torch.where(x < -pi, x + 2 * pi, x)
        x = torch.where(x > pi, x - 2 * pi, x)
        out.append(roma.internal.unflatten_batch_dims(x, batch_shape))
    return torch.stack(out, dim=-1)


def rotmat_to_euler(convention, rotmat, epsilon=1e-7):
    return unitquat_to_euler(convention, rotmat_to_unitquat(rotmat), epsilon=epsilon)
