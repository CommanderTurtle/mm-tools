"""The one-pass SMPL fit (fab/smpl_fitter.py) on the deployment path: the same network and outputs, with SMPL
restricted to what the fit needs (joints from precomputed shape directions, kinematics batched over the tree's 9
depth levels instead of 23 sequential steps, the refinement stages skinning only the dense head's surface points), and
the whole fit -- Kabsch alignment, MLPs, both refinement stages, the final full mesh -- captured as one CUDA graph per
person count.

    fit = GraphFit(load_fitter(path), SMPL(".../SMPL_NEUTRAL.pkl").cuda(), surface_idx)
    R, betas, verts = fit(points)  # (N, K, 3) camera frame -> (N, 24, 3, 3), (N, 10), (N, 6890, 3) placed at the points
"""

import torch

from fab.retarget.g1 import rot6d_to_matrix


def kabsch(src, dst, squarings=24):
    """fab/smpl_fit.kabsch without the SVD (which copies to the host, so it cannot be graph-captured): Horn's
    quaternion method, the top eigenvector of the 4x4 profile matrix found by repeated squaring of its PSD shift."""
    return kabsch_S(torch.einsum("bni,bnj->bij", src, dst), squarings)


def kabsch_S(S, squarings=24):
    """kabsch from the (B, 3, 3) correlation sum_n src_n dst_n^T (several problems batched into one solve)."""
    S = S.double()
    xx, xy, xz, yx, yy, yz, zx, zy, zz = S.flatten(1).unbind(1)
    N = torch.stack([xx + yy + zz, yz - zy, zx - xz, xy - yx,
                     yz - zy, xx - yy - zz, xy + yx, zx + xz,
                     zx - xz, xy + yx, yy - xx - zz, yz + zy,
                     xy - yx, zx + xz, yz + zy, zz - xx - yy], 1).view(-1, 4, 4)
    M = N + N.flatten(1).norm(dim=1)[:, None, None] * torch.eye(4, dtype=N.dtype, device=N.device)
    for _ in range(squarings):  # M^(2^k) / |.| -> q q^T
        M = M @ M
        M = M / M.flatten(1).norm(dim=1).clamp_min(1e-300)[:, None, None]
    q = M.gather(2, M.diagonal(dim1=1, dim2=2).argmax(1)[:, None, None].expand(-1, 4, 1))[..., 0]
    w, x, y, z = (q / q.norm(dim=1, keepdim=True)).float().unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)


class FastSMPL(torch.nn.Module):
    """bench/body_models.SMPL with rotation matrices given (no translation), for the full mesh or a vertex subset."""

    def __init__(self, body, idx):
        super().__init__()
        J = body.J_regressor.float()
        self.register_buffer("J_t", J @ body.v_template)  # (24, 3)
        self.register_buffer("J_dirs", torch.einsum("jv,vcl->ljc", J, body.shapedirs).flatten(1))  # (10, 72)
        cols = (torch.as_tensor(idx, device=J.device)[:, None] * 3 + torch.arange(3, device=J.device)).flatten()
        blend = torch.cat([body.shapedirs.permute(2, 0, 1).flatten(1), body.posedirs], 0)  # (10 + 207, V * 3)
        self.register_buffer("tmpl_full", body.v_template.clone())
        self.register_buffer("blend_full", blend.clone())
        self.register_buffer("w_full", body.weights.clone())
        self.register_buffer("tmpl_sub", body.v_template[idx].clone())
        self.register_buffer("blend_sub", blend[:, cols].clone())
        self.register_buffer("w_sub", body.weights[idx].clone())
        parents = body.parents
        depth = [0] * 24
        for j in range(1, 24):
            depth[j] = depth[parents[j]] + 1
        order, pos, self.levels = [0], {0: 0}, []
        for d in range(1, max(depth) + 1):
            js = [j for j in range(24) if depth[j] == d]
            self.levels.append((torch.tensor(js, device=J.device), torch.tensor([pos[parents[j]] for j in js], device=J.device)))
            for j in js:
                pos[j] = len(order)
                order.append(j)
        self.register_buffer("inv", torch.argsort(torch.tensor(order, device=J.device)))
        self.register_buffer("parents_t", torch.tensor([0] + parents[1:], device=J.device))
        self.register_buffer("eye", torch.eye(3, device=J.device))

    def forward(self, R, betas, full=True):
        """R (B, 24, 3, 3), betas (B, 10) -> vertices (B, V, 3) of the full mesh or the subset, joints (B, 24, 3)."""
        B = len(R)
        betas = betas[:, :10]
        J = self.J_t + (betas @ self.J_dirs).view(B, 24, 3)
        rel = J - torch.cat([torch.zeros_like(J[:, :1]), J[:, self.parents_t[1:]]], 1)
        GR, Gt = R[:, :1], rel[:, :1]
        for js, pp in self.levels:
            Rp = GR[:, pp]
            GR = torch.cat([GR, Rp @ R[:, js]], 1)
            Gt = torch.cat([Gt, (Rp @ rel[:, js, :, None])[..., 0] + Gt[:, pp]], 1)
        GR, Gt = GR[:, self.inv], Gt[:, self.inv]
        T = torch.cat([GR, (Gt - (GR @ J[..., None])[..., 0])[..., None]], -1).flatten(2)  # (B, 24, 12)
        tmpl, blend, w = (self.tmpl_full, self.blend_full, self.w_full) if full else (self.tmpl_sub, self.blend_sub, self.w_sub)
        feat = torch.cat([betas, (R[:, 1:] - self.eye).flatten(1)], 1)
        v = tmpl + (feat @ blend).view(B, -1, 3)
        skin = (w @ T).view(B, -1, 3, 4)
        return (skin[..., :3] @ v[..., None])[..., 0] + skin[..., 3], Gt


class FastFit(torch.nn.Module):
    """SMPLFitter.forward with FastSMPL; also returns the full mesh placed at the points' centroid."""

    def __init__(self, fitter, body, idx):
        super().__init__()
        self.fitter = fitter
        self.smpl = FastSMPL(body, idx)

    def forward(self, points):
        f = self.fitter
        c = points.mean(1, keepdim=True)
        x = points - c
        R0 = None
        if len(f.tmpl):
            t = f.tmpl - f.tmpl.mean(0)
            R0 = kabsch(t[None].expand(len(x), -1, -1), x)
            x = torch.einsum("bji,bnj->bni", R0, x)
        y = f.net((x * 5.0).flatten(1))
        rot6d, betas = y[:, :144], y[:, 144:]
        for r in f.refine:
            R = rot6d_to_matrix(rot6d.reshape(-1, 6)).view(-1, 24, 3, 3)
            vp, _ = self.smpl(R, betas, full=False)
            vp = vp - vp.mean(1, keepdim=True)
            d = r(torch.cat([(x * 5.0).flatten(1), (vp * 5.0).flatten(1), rot6d, betas], 1))
            rot6d, betas = rot6d + d[:, :144], betas + d[:, 144:]
        R = rot6d_to_matrix(rot6d.reshape(-1, 6)).view(-1, 24, 3, 3)
        if R0 is not None:
            R = torch.cat([(R0 @ R[:, 0])[:, None], R[:, 1:]], 1)
        v, _ = self.smpl(R, betas)
        return R, betas, v


class GraphFit:
    """FastFit replayed from a CUDA graph captured per person count (outputs are the graph's buffers: copy to keep)."""

    def __init__(self, fitter, body, idx, graph=True):
        self.fit = FastFit(fitter, body, torch.as_tensor(idx)).eval()
        self.idx = torch.as_tensor(idx, device=self.fit.smpl.eye.device)
        self.graph = graph
        self.graphs = {}

    @torch.no_grad()
    def __call__(self, points):
        n = len(points)
        if not self.graph:
            return self._place(points, *self.fit(points))
        if n not in self.graphs:
            static = points.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._place(static, *self.fit(static))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._place(static, *self.fit(static))
            self.graphs[n] = (g, static, out)
        g, static, out = self.graphs[n]
        static.copy_(points)
        g.replay()
        return out

    def _place(self, points, R, betas, v):
        return R, betas, v - v[:, self.idx].mean(1, keepdim=True) + points.mean(1, keepdim=True)
