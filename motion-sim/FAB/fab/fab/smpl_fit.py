"""SMPL parameters from the dense head's points (optional output: the network itself predicts no body-model
parameters). Batched fit: L-BFGS on axis-angle pose, betas and translation to the surface points
(and optionally the 14 evaluation joints), from a template whose global rotation is the Kabsch alignment of the
template points to the prediction. A weak L2 prior keeps the body pose and betas small."""

import torch


def kabsch(src, dst):
    """Rotation R (B, 3, 3) minimising |R src - dst| for centred point sets (B, N, 3)."""
    H = torch.einsum("bni,bnj->bij", src, dst)
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vh.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return Vh.transpose(1, 2) @ D @ U.transpose(1, 2)


def rotmat_to_quat(R):
    """(B, 3, 3) -> unit quaternions (B, 4) wxyz with w >= 0 (Shepperd: the largest of the four squared components is
    taken directly, the others from it, so the conversion stays exact near 180 degrees)."""
    m = R.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    t = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], 1)
    r = t.clamp_min(1e-12).sqrt()  # 2 x |component|
    cand = torch.stack([
        torch.stack([r[:, 0], (m21 - m12) / r[:, 0], (m02 - m20) / r[:, 0], (m10 - m01) / r[:, 0]], 1),
        torch.stack([(m21 - m12) / r[:, 1], r[:, 1], (m01 + m10) / r[:, 1], (m02 + m20) / r[:, 1]], 1),
        torch.stack([(m02 - m20) / r[:, 2], (m01 + m10) / r[:, 2], r[:, 2], (m12 + m21) / r[:, 2]], 1),
        torch.stack([(m10 - m01) / r[:, 3], (m02 + m20) / r[:, 3], (m12 + m21) / r[:, 3], r[:, 3]], 1)], 1)
    q = cand[torch.arange(len(m), device=m.device), t.argmax(1)]
    q = q / q.norm(dim=1, keepdim=True)
    return torch.where(q[:, :1] < 0, -q, q)


def rotmat_to_aa(R):
    """(B, 3, 3) -> axis-angle (B, 3), angle in [0, pi]; stable at 0 and at 180 degrees."""
    q = rotmat_to_quat(R)
    v = q[:, 1:]
    s = v.norm(dim=1, keepdim=True)
    ang = 2 * torch.atan2(s, q[:, :1])
    return torch.where(s > 1e-7, v / s.clamp_min(1e-12) * ang, 2 * v)  # small angles: aa ~= 2 v


def fit_smpl(body, points, idx, joints=None, j14=None, iters=100, w_pose=1e-5, w_betas=1e-5, init=None):
    """body: bench.body_models.SMPL; points (B, K, 3) predicted surface points (camera frame) at template vertices
    idx (K,); joints (B, 14, 3) and j14 (14, 6890) regressor optional. Returns (pose (B, 72), betas (B, 10),
    trans (B, 3), vertices (B, 6890, 3))."""
    B, dev = points.shape[0], points.device
    pts = points.float()
    tmpl = body.v_template[idx].float()
    c_t, c_p = tmpl.mean(0), pts.mean(1, keepdim=True)
    if init is None:
        R0 = kabsch((tmpl - c_t)[None].expand(B, -1, -1), pts - c_p)
        go = rotmat_to_aa(R0).detach().clone().requires_grad_(True)
        pose = torch.zeros(B, 69, device=dev, requires_grad=True)
        betas = torch.zeros(B, 10, device=dev, requires_grad=True)
        trans = (c_p[:, 0] - torch.einsum("bij,j->bi", R0, c_t)).detach().clone().requires_grad_(True)
    else:  # learned one-pass fitter (fab/smpl_fitter.py): joint rotations (B, 24, 3, 3), betas (B, 10)
        R, b0 = init
        aa = rotmat_to_aa(R.reshape(-1, 3, 3).float()).view(B, 72)
        go = aa[:, :3].detach().clone().requires_grad_(True)
        pose = aa[:, 3:].detach().clone().requires_grad_(True)
        betas = b0.float().detach().clone().requires_grad_(True)
        with torch.no_grad():
            v0, _ = body(go, pose, betas, None)
        trans = (c_p[:, 0] - v0[:, idx].mean(1)).detach().clone().requires_grad_(True)
    params = [go, pose, betas, trans]

    def objective():
        v, _ = body(go, pose, betas, trans)
        loss = (v[:, idx] - pts).pow(2).sum(-1).mean(-1).sum()  # per body, summed: bodies stay independent
        if joints is not None:
            loss = loss + (torch.einsum("jv,bvc->bjc", j14, v) - joints.float()).pow(2).sum(-1).mean(-1).sum()
        return loss + w_pose * pose.pow(2).sum() / 69 + w_betas * betas.pow(2).sum() / 10

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        v, _ = body(go, pose, betas, trans)
    return torch.cat([go, pose], 1).detach(), betas.detach(), trans.detach(), v
