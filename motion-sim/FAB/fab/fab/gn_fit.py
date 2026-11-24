"""Gauss-Newton refinement of a SMPL fit to dense keypoints: a few damped Gauss-Newton
steps after the one-pass learned fit (fab/fast_fit.py), on any linear point set of the SMPL mesh (vertices:
the dense head's surface points or the parametric mesh's own vertices; regressed points: the H36M joints, MHR
keypoints), with an optional reprojection term on the head's own 2D. Every step is fixed-shape tensor work
(forward-mode Jacobians, batched Cholesky), so the whole refinement captures into one CUDA graph.

Unknowns per body: 24 local rotation increments (R_j <- R_j exp(w_j)), 10 betas, a translation for the 3D term (the
dense 3D is root-relative: its frame's translation is a nuisance) and, with 2D, the camera translation of the body.

    gn = GNFit(body, vert_idx, reg)      # points = [V[vert_idx], reg @ V]
    R, betas, t3, t2 = gn(R0, betas0, X, w3, uv=uv, w2=w2, Kc=Kc)   # Kc: intrinsics of the crop (crop px)
"""

import torch

from fab.fast_fit import FastSMPL


def skew(w):
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).view(*w.shape[:-1], 3, 3)


def so3_exp(w):
    th2 = (w * w).sum(-1)[..., None, None]
    th = th2.clamp_min(1e-12).sqrt()
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
    K = skew(w)
    return torch.eye(3, device=w.device, dtype=w.dtype) + a * K + b * (K @ K)


def crop_camera(K, A):
    """Intrinsics (B, 3, 3) and image -> crop affine (B, 2, 3) -> the crop's own 3x3 camera (projects to crop px)."""
    A3 = torch.cat([A, torch.tensor([0.0, 0.0, 1.0], device=A.device, dtype=A.dtype).expand(len(A), 1, 3)], 1)
    return A3 @ K


def project(P, Kc):
    q = torch.einsum("bij,bnj->bni", Kc, P)
    return q[..., :2] / q[..., 2:].clamp_min(1e-3)


def closed_form_t(P, uv, Kc, w):
    """Translation t (B, 3) minimising the weighted algebraic reprojection error of P + t (B, N, 3) onto uv (B, N, 2)
    under the crop camera Kc: (k_x - u k_z) . (P + t) = 0, (k_y - v k_z) . (P + t) = 0 (linear least squares)."""
    ax = Kc[:, None, 0] - uv[..., :1] * Kc[:, None, 2]  # (B, N, 3)
    ay = Kc[:, None, 1] - uv[..., 1:] * Kc[:, None, 2]
    Am = torch.cat([ax, ay], 1)
    bm = -torch.cat([(ax * P).sum(-1), (ay * P).sum(-1)], 1)
    ww = torch.cat([w, w], 1)
    H = torch.einsum("bni,bn,bnj->bij", Am, ww, Am) + 1e-9 * torch.eye(3, device=P.device)
    g = torch.einsum("bni,bn,bn->bi", Am, ww, bm)
    return torch.linalg.solve_ex(H, g)[0]


class GNFit(torch.nn.Module):
    """Damped Gauss-Newton on SMPL (rotation matrices, betas) to observed points. vert_idx (Nv,) SMPL vertices; reg
    (Nr, 6890) regressor rows (H36M joints, MHR keypoints); point order [vertices, regressed]."""

    def __init__(self, body, vert_idx, reg=None, iters=3, lam=1e-3, w_betas=1e-5, w_step=1e-6, w_2d=1.0, edges=None,
                 w_edge=1.0, edge_w=None):
        super().__init__()
        V = body.v_template.shape[0]
        vert_idx = torch.as_tensor(vert_idx, dtype=torch.long, device=body.v_template.device)
        reg = torch.zeros(0, V, device=vert_idx.device) if reg is None else torch.as_tensor(reg).float().to(vert_idx.device)
        # edges (E, 2): mesh edge vectors v[b] - v[a] as a translation-free term (the official MHR -> SMPL fit's edge loss)
        edges = torch.zeros(0, 2, dtype=torch.long, device=vert_idx.device) if edges is None else \
            torch.as_tensor(edges, dtype=torch.long, device=vert_idx.device)
        cols = torch.zeros(V, dtype=torch.bool, device=vert_idx.device)
        cols[vert_idx] = True
        cols[edges.flatten()] = True
        cols |= reg.abs().sum(0) > 0
        support = torch.nonzero(cols).flatten()
        pos = torch.full((V,), -1, dtype=torch.long, device=vert_idx.device)
        pos[support] = torch.arange(len(support), device=vert_idx.device)
        self.smpl = FastSMPL(body, support)
        self.register_buffer("vpos", pos[vert_idx])
        self.register_buffer("reg", reg[:, support])
        self.register_buffer("ea", pos[edges[:, 0]])
        self.register_buffer("eb", pos[edges[:, 1]])
        ew = torch.ones(len(edges), device=vert_idx.device) if edge_w is None else torch.as_tensor(edge_w).float().to(vert_idx.device)
        self.register_buffer("edge_w", ew / ew.mean().clamp_min(1e-9))  # relative per-edge weights (mean 1)
        self.n_points = len(vert_idx) + len(reg)
        self.iters, self.lam, self.w_betas, self.w_step, self.w_2d = iters, lam, w_betas, w_step, w_2d
        self.w_edge = w_edge

    def points(self, R, betas, with_edges=False):
        v, _ = self.smpl(R, betas, full=False)
        P = torch.cat([v[:, self.vpos], torch.einsum("ns,bsc->bnc", self.reg, v)], 1)
        return (P, v[:, self.eb] - v[:, self.ea]) if with_edges else P

    def forward(self, R, betas, X, w3, uv=None, w2=None, Kc=None, t2=None, Xe=None):
        """R (B, 24, 3, 3), betas (B, 10) initial fit; X (B, N, 3) observed 3D (any translation), w3 (B, N) weights;
        2D (optional): uv (B, N, 2) crop px, w2 (B, N), Kc (B, 3, 3) crop camera, t2 (B, 3) initial camera
        translation (None: closed form from the initial fit); Xe (B, E, 3) observed edge vectors (with `edges`).
        -> R, betas, t3 (3D-frame translation), t2."""
        B = len(R)
        use2d = uv is not None
        n = 85 + (3 if use2d else 0)
        eye = torch.eye(3, device=R.device)
        s3 = (w3 / w3.sum(1, keepdim=True).clamp_min(1e-9)).sqrt()[..., None]
        P0 = self.points(R, betas)
        t3 = ((X - P0) * s3 ** 2).sum(1)
        if use2d:
            s2 = (w2 / w2.sum(1, keepdim=True).clamp_min(1e-9) * self.w_2d).sqrt()[..., None]
            fc = (Kc[:, 0, 0] * Kc[:, 1, 1] - Kc[:, 0, 1] * Kc[:, 1, 0]).abs().sqrt()
            if t2 is None:
                t2 = closed_form_t(P0, uv, Kc, w2)
        sb, ss = self.w_betas ** 0.5, self.w_step ** 0.5
        use_e = Xe is not None and len(self.ea) > 0
        se = (self.w_edge / max(len(self.ea), 1)) ** 0.5
        d0 = torch.zeros(B, n, device=R.device)
        basis = torch.eye(n, device=R.device)[:, None].expand(n, B, n)
        for _ in range(self.iters):
            if use2d:
                with torch.no_grad():
                    zs = ((self.points(R, betas) + t2[:, None])[..., 2:] / fc[:, None, None]).clamp_min(1e-3)

            def res(d, R=R, betas=betas, t3=t3, t2=t2):
                Rn = R @ (eye + skew(d[:, :72].reshape(B, 24, 3)))
                bn = betas + d[:, 72:82]
                P, E = self.points(Rn, bn, with_edges=True)
                r = [((P + (t3 + d[:, 82:85])[:, None] - X) * s3).flatten(1), sb * bn, ss * d[:, 3:72]]
                if use_e:
                    r.append(((E - Xe) * se * self.edge_w[None, :, None].sqrt()).flatten(1))
                if use2d:
                    r.append(((project(P + (t2 + d[:, 85:88])[:, None], Kc) - uv) * zs * s2).flatten(1))
                return torch.cat(r, 1)

            r0 = res(d0)
            J = torch.func.vmap(lambda v: torch.func.jvp(res, (d0,), (v,))[1])(basis).permute(1, 2, 0)  # (B, M, n)
            H = (J.transpose(1, 2) @ J).double()
            H = H + self.lam * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2)) + 1e-12 * torch.eye(n, device=R.device)
            g = (J.transpose(1, 2) @ r0[..., None]).double()
            L, _ = torch.linalg.cholesky_ex(H)
            d = -torch.cholesky_solve(g, L)[..., 0].float()
            R = R @ so3_exp(d[:, :72].reshape(B, 24, 3))
            betas = betas + d[:, 72:82]
            t3 = t3 + d[:, 82:85]
            if use2d:
                t2 = t2 + d[:, 85:88]
        return R, betas, t3, t2


def normal_eq(J, w, r, chunks=256):
    """Weighted Gauss-Newton normal equations H = J^T W J (B, n, n), g = J^T W r (B, n) for J (B, M, n), w (B, M) or (M,),
    r (B, M): the long M axis split into chunks (one small GEMM each, summed) -- a single (n x M) @ (M x n) GEMM per
    body runs on one or two thread blocks (cuBLAS 128x128 tiles), ~2.5 ms for M = 60K at 2 people."""
    B, M, n = J.shape
    w = w.expand(B, M) if w.dim() == 1 else w
    pad = (-M) % chunks
    if pad:
        J = torch.nn.functional.pad(J, (0, 0, 0, pad))
        w = torch.nn.functional.pad(w, (0, pad))
        r = torch.nn.functional.pad(r, (0, pad))
    Jc = J.view(B, chunks, -1, n)
    Jw = (Jc * w.view(B, chunks, -1, 1)).transpose(2, 3)
    return (Jw @ Jc).sum(1), (Jw @ r.view(B, chunks, -1, 1)).sum(1)[..., 0]


def cholesky_cusolver(H):
    """torch.linalg.cholesky_ex through cuSOLVER (the default MAGMA path for small batches is ~10x slower at 2-4 people)."""
    lib = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        return torch.linalg.cholesky_ex(H)
    finally:
        torch.backends.cuda.preferred_linalg_library(lib)


class FastGN(torch.nn.Module):
    """GNFit's damped Gauss-Newton with analytic linear-blend-skinning Jacobians instead of forward-mode AD (the
    deployment path; the pose-corrective blend shapes enter the residuals but not the Jacobian): rotation increments
    w_j of joint j move every vertex by (A_j w_j) x u_ij, u_ij = its skinned position over the joints below j minus
    their weight times joint j's posed position; betas move the template and the joints along the chain. 3D terms
    only: vertices / regressed points (translation-coupled) and edge vectors (translation-free)."""

    def __init__(self, body, vert_idx, reg=None, edges=None, iters=5, lam=1e-3, w_betas=0.0, w_step=0.0, w_edge=1.0,
                 edge_w=None, w_root=0.0, edge_first=False, edge_l1=False, w_beta_step=0.0):
        super().__init__()
        V = body.v_template.shape[0]
        dev = body.v_template.device
        vert_idx = torch.as_tensor(vert_idx, dtype=torch.long, device=dev)
        reg = torch.zeros(0, V, device=dev) if reg is None else torch.as_tensor(reg).float().to(dev)
        edges = torch.zeros(0, 2, dtype=torch.long, device=dev) if edges is None else \
            torch.as_tensor(edges, dtype=torch.long, device=dev)
        cols = torch.zeros(V, dtype=torch.bool, device=dev)
        cols[vert_idx] = True
        cols[edges.flatten()] = True
        cols |= reg.abs().sum(0) > 0
        support = torch.nonzero(cols).flatten()
        pos = torch.full((V,), -1, dtype=torch.long, device=dev)
        pos[support] = torch.arange(len(support), device=dev)
        self.smpl = FastSMPL(body, support)
        S = len(support)
        self.register_buffer("vpos", pos[vert_idx])
        self.register_buffer("reg", reg[:, support])
        self.register_buffer("ea", pos[edges[:, 0]])
        self.register_buffer("eb", pos[edges[:, 1]])
        ew = torch.ones(len(edges), device=dev) if edge_w is None else torch.as_tensor(edge_w).float().to(dev)
        self.register_buffer("edge_w", ew / ew.mean().clamp_min(1e-9))
        self.register_buffer("sdirs", self.smpl.blend_sub[:10].view(10, S, 3).permute(1, 2, 0).contiguous())  # (S, 3, 10)
        self.register_buffer("dJ", self.smpl.J_dirs.view(10, 24, 3).permute(1, 2, 0).contiguous())  # (24, 3, 10)
        parents = body.parents
        D = torch.eye(24, device=dev)
        for k in range(1, 24):  # D[j, k] = 1 when k is j or below it
            j = parents[k]
            while j >= 0:
                D[j, k] = 1.0
                j = parents[j] if j > 0 else -1
        self.register_buffer("D", D)
        self.register_buffer("Wd", self.smpl.w_sub @ D.T)  # (S, 24): skinning weight of the joints below j
        # sparse skinning influences (exact: every non-zero weight kept): K joints per vertex
        w_sub = self.smpl.w_sub
        K = int((w_sub > 0).sum(1).max())
        topw, topj = w_sub.topk(K, dim=1)
        self.register_buffer("topw", topw)
        self.register_buffer("topj", topj)
        self.register_buffer("Dsel", D[:, topj].permute(1, 0, 2) * topw[:, None, :])  # (S, 24, K)
        self.iters, self.lam, self.w_betas, self.w_step, self.w_edge = iters, lam, w_betas, w_step, w_edge
        self.w_root = w_root  # damping of the global rotation's steps (large: keep the initial fit's orientation)
        # edge_first: the first step on the edges alone (vertex term x 1e-3), as the official MHR -> SMPL fit's schedule;
        # edge_l1: robust edges (iteratively reweighted, 1 / max(|r|, 1 mm): the official fit's L1 edge loss)
        self.edge_first, self.edge_l1 = edge_first, edge_l1
        self.w_beta_step = w_beta_step  # damping of the betas' steps (keep the initial fit's body shape)

    def skin(self, R, betas):
        """-> support vertices (B, S, 3) and their Jacobian (B, S, 3, 82) w.r.t. [local rotation increments (72), betas]."""
        m = self.smpl
        B = len(R)
        betas = betas[:, :10]
        J = m.J_t + (betas @ m.J_dirs).view(B, 24, 3)
        rel = J - torch.cat([torch.zeros_like(J[:, :1]), J[:, m.parents_t[1:]]], 1)
        dJ = self.dJ.expand(B, -1, -1, -1)
        drel = dJ - torch.cat([torch.zeros_like(dJ[:, :1]), dJ[:, m.parents_t[1:]]], 1)
        GR, Gt, dP = R[:, :1], rel[:, :1], drel[:, :1]
        for js, pp in m.levels:
            Rp = GR[:, pp]
            GR = torch.cat([GR, Rp @ R[:, js]], 1)
            Gt = torch.cat([Gt, (Rp @ rel[:, js, :, None])[..., 0] + Gt[:, pp]], 1)
            dP = torch.cat([dP, Rp @ drel[:, js] + dP[:, pp]], 1)
        GR, Gt, dP = GR[:, m.inv], Gt[:, m.inv], dP[:, m.inv]
        feat = torch.cat([betas, (R[:, 1:] - m.eye).flatten(1)], 1)
        vbar = m.tmpl_sub + (feat @ m.blend_sub).view(B, -1, 3)  # (B, S, 3) rest-pose vertices
        b = Gt - (GR @ J[..., None])[..., 0]
        Gk = GR[:, self.topj]  # (B, S, K, 3, 3) rotations of each vertex's influencing joints
        # (elementwise products instead of ~10^5 batched 3x3 GEMMs, which cuBLAS runs slowly)
        x = (Gk * vbar[:, :, None, None, :]).sum(-1) + b[:, self.topj]  # (B, S, K, 3) vertex under joint k alone
        tw = self.topw[None, :, :, None]
        v = (tw * x).sum(2)
        u = torch.einsum("sjk,bskc->bsjc", self.Dsel, x) - self.Wd[None, :, :, None] * Gt[:, None]
        # d v / d w_j = -[u]x A_j: column c is A_j[:, c] x u -> (B, S, 24, c, xyz)
        Jr = torch.linalg.cross(GR.transpose(-1, -2)[:, None], u[:, :, :, None, :].expand(-1, -1, -1, 3, -1), dim=-1)
        Jr = Jr.permute(0, 1, 4, 2, 3).reshape(B, -1, 3, 72)
        Abar = (tw[..., None] * Gk).sum(2)
        Jb = (Abar[..., None] * self.sdirs[None, :, None]).sum(-2) + (tw[..., None] * (dP - GR @ dJ)[:, self.topj]).sum(2)
        return v, torch.cat([Jr, Jb], -1)

    def forward(self, R, betas, X, w3, Xe=None, uv=None, w2=None, Kc=None, t2=None, w_2d=1.0, rows2d=None, R0=None, b0=None,
                w_anchor=0.0, w_anchor_b=0.0, R1=None, b1=None, w1=None, w1_b=None, w1_root=0.0, ray=None, w_ray=1.0):
        """R (B, 24, 3, 3), betas (B, 10); X (B, N, 3) observed points [vertices, regressed] (any translation), w3 (B, N);
        Xe (B, E, 3) observed edge vectors; 2D (optional): uv (B, N, 2) crop px of the same points, w2 (B, N), Kc
        (B, 3, 3) crop camera, t2 (B, 3) camera translation (None: closed form), w_2d total 2D weight, rows2d (static
        index of the points that have 2D; default all); R0 / b0 + w_anchor / w_anchor_b: a prior toward a reference fit
        (w_anchor |log(R0^T R)|^2 per joint + w_anchor_b |betas - b0|^2; the fitter work's anchor to the one-pass fit);
        R1 / b1 + per-sample weights w1 / w1_b (B,): a second prior of that form (video: the previous frame's fit of the
        same person; w1 0 = none), w1_root its weight on the global rotation relative to w1; ray (B, N, 3) unit camera
        rays of the observed points + w_ray: each 3D residual's along-ray (depth) part weighted w_ray (anisotropic)
        -> R, betas, t3 (translation of the 3D frame), t2 (None without 2D)."""
        B = len(R)
        use2d = uv is not None
        n = 88 if use2d else 85
        s3 = (w3 / w3.sum(1, keepdim=True).clamp_min(1e-9))[..., None, None]  # squared weights
        se = self.w_edge / max(len(self.ea), 1)
        use_e = Xe is not None and len(self.ea) > 0
        I3 = torch.eye(3, device=R.device).expand(B, 1, 3, 3)
        if use2d:
            if rows2d is not None:
                uv, w2 = uv[:, rows2d], w2[:, rows2d]
            s2 = w2 / w2.sum(1, keepdim=True).clamp_min(1e-9) * w_2d
            fc = (Kc[:, 0, 0] * Kc[:, 1, 1] - Kc[:, 0, 1] * Kc[:, 1, 0]).abs().sqrt()
        t3 = None
        for it in range(self.iters):  # iters >= 1
            v, Jv = self.skin(R, betas)
            P = torch.cat([v[:, self.vpos], torch.einsum("ns,bsc->bnc", self.reg, v)], 1)
            JP = torch.cat([Jv[:, self.vpos], torch.einsum("ns,bscl->bncl", self.reg, Jv)], 1)  # (B, N, 3, 82)
            if t3 is None:
                t3 = ((X - P) * s3[..., 0]).sum(1)
                if use2d and t2 is None:
                    t2 = closed_form_t(P if rows2d is None else P[:, rows2d], uv, Kc, w2)
            r = P + t3[:, None] - X
            N = JP.shape[1]
            Jf = torch.cat([JP, I3.expand(-1, N, -1, -1)] + ([torch.zeros_like(JP[..., :3])] if use2d else []), -1)
            if ray is not None and w_ray != 1.0:  # r, J -> (I - (1 - sqrt(w_ray)) d d^T) r, J: the depth part down-weighted
                c_ = 1.0 - w_ray ** 0.5
                r = r - c_ * (r * ray).sum(-1, keepdim=True) * ray
                Jf = Jf - c_ * ray[..., None] * torch.einsum("bnc,bncl->bnl", ray, Jf)[:, :, None]
            wv = s3[..., 0, 0] * (1e-3 if self.edge_first and it == 0 else 1.0)
            H, g = normal_eq(Jf.flatten(1, 2), wv.repeat_interleave(3, 1), r.flatten(1))
            if use2d:  # reprojection in metres at each point's depth: (u - uv) z / f
                P2, JP2 = (P, JP) if rows2d is None else (P[:, rows2d], JP[:, rows2d])
                Q = P2 + t2[:, None]
                q = torch.einsum("bij,bnj->bni", Kc, Q)
                z = q[..., 2:].clamp_min(1e-3)
                u = q[..., :2] / z
                du = (Kc[:, None, :2] - u[..., None] * Kc[:, None, 2:]) / z[..., None]  # (B, N, 2, 3) d u / d Q
                zs = z / fc[:, None, None]
                J2 = torch.cat([du @ JP2, torch.zeros_like(du), du], -1) * zs[..., None]  # (B, N2, 2, 88)
                r2 = (u - uv) * zs
                H2, g2 = normal_eq(J2.flatten(1, 2), s2.repeat_interleave(2, 1), r2.flatten(1), chunks=1)
                H, g = H + H2, g + g2
            if use_e:
                Je = Jv[:, self.eb] - Jv[:, self.ea]
                re = v[:, self.eb] - v[:, self.ea] - Xe
                we = (se * self.edge_w).repeat_interleave(3)
                if self.edge_l1:  # IRLS weights of an L1 loss on the edge-vector components
                    we = we / re.flatten(1).abs().clamp_min(1e-3) * 1e-3
                He, ge = normal_eq(Je.flatten(1, 2), we, re.flatten(1))
                H[:, :82, :82] = H[:, :82, :82] + He
                g[:, :82] = g[:, :82] + ge
            prior = torch.zeros(n, device=R.device)
            prior[:3] = self.w_root
            prior[3:72] = self.w_step
            prior[72:82] = self.w_betas + self.w_beta_step
            g[:, 72:82] = g[:, 72:82] + self.w_betas * betas[:, :10]
            if R0 is not None and w_anchor > 0:  # body joints only (the global rotation stays free)
                from fab.smpl_fit import rotmat_to_aa

                e = rotmat_to_aa((R0.transpose(-1, -2) @ R).reshape(-1, 3, 3)).reshape(B, 72)
                g[:, 3:72] = g[:, 3:72] + w_anchor * e[:, 3:]
                prior[3:72] = prior[3:72] + w_anchor
            if b0 is not None and w_anchor_b > 0:
                g[:, 72:82] = g[:, 72:82] + w_anchor_b * (betas[:, :10] - b0[:, :10])
                prior[72:82] = prior[72:82] + w_anchor_b
            if R1 is not None and w1 is not None:  # per-sample weights (B,), joint rotations (+ the root at w1_root)
                from fab.smpl_fit import rotmat_to_aa

                e1 = rotmat_to_aa((R1.transpose(-1, -2) @ R).reshape(-1, 3, 3)).reshape(B, 72)
                wj = torch.cat([w1[:, None].expand(-1, 3) * w1_root, w1[:, None].expand(-1, 69)], 1)
                g[:, :72] = g[:, :72] + wj * e1
                prior = prior[None].expand(B, -1).clone()
                prior[:, :72] = prior[:, :72] + wj
                if b1 is not None and w1_b is not None:
                    g[:, 72:82] = g[:, 72:82] + w1_b[:, None] * (betas[:, :10] - b1[:, :10])
                    prior[:, 72:82] = prior[:, 72:82] + w1_b[:, None]
            H = H + torch.diag_embed(self.lam * torch.diagonal(H, dim1=1, dim2=2) + prior + 1e-12)
            dsc = torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-20).rsqrt()  # Jacobi scaling: float32 Cholesky
            L, _ = cholesky_cusolver(H * dsc[:, :, None] * dsc[:, None, :])
            y = torch.linalg.solve_triangular(L, (g * dsc)[..., None], upper=False)  # (cholesky_solve's batched
            d = -torch.linalg.solve_triangular(L.transpose(1, 2), y, upper=True)[..., 0] * dsc  # MAGMA path can't be captured)
            R = R @ so3_exp(d[:, :72].reshape(B, 24, 3))
            betas = betas + d[:, 72:82]
            t3 = t3 + d[:, 82:85]
            if use2d:
                t2 = t2 + d[:, 85:88]
        return R, betas, t3, t2


class DenseFit(torch.nn.Module):
    """The refined SMPL fit of a dense head (FAB-L / FAB-S): the one-pass learned fit of the dense surface, then
    FastGN steps on the dense-refined mesh (the parametric mesh with the dense surface offsets interpolated to every
    vertex: all vertices + all mesh edges, L1 edges as the official MHR -> SMPL fit) and the dense MHR body keypoints
    (3D + their 2D at w_2d), placed in the camera by the 2D. The betas stay the one-pass fit's (w_beta_step damps their
    steps): with a fitter taught the official fit, that keeps its body shape. Inputs per person (mesh frame = the MHR mesh's root-relative frame, camera axes):
      surf (N, K, 3) dense surface points (anchors added, hip shift removed); mesh (N, 6890, 3) the MHR mesh in SMPL
      topology; kp (N, 23, 3) dense MHR body keypoints (BODY23, anchored on reg23 @ mesh); kp_uv (N, 23, 2) their 2D
      (crop px); Kc (N, 3, 3) crop camera.
    -> R (N, 24, 3, 3), betas (N, 10), vertices (N, 6890, 3) camera frame."""

    def __init__(self, onepass, body, surface_idx, reg23, iters=2, w_edge=100.0, w_2d=0.1, lam=1e-3, compile=False,
                 fit_pos=None, edge_l1=True, w_beta_step=1.0):
        super().__init__()
        self.onepass = onepass  # fab/fast_fit.FastFit
        V = body.v_template.shape[0]
        dev = body.v_template.device
        f = body.faces.long()
        e = torch.unique(torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(1)[0], dim=0)
        self.gn = FastGN(body, torch.arange(V, device=dev), reg23, edges=e, iters=iters, lam=lam, w_edge=w_edge,
                         edge_l1=edge_l1, w_beta_step=w_beta_step)
        if compile:  # inductor-fused Gauss-Newton steps, captured inside GraphDenseFit's CUDA graph
            self.gn.forward = torch.compile(self.gn.forward, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.register_buffer("sidx", torch.as_tensor(surface_idx, dtype=torch.long, device=dev))
        # the one-pass fitter's points among the surface points (all of them unless the head has more)
        self.register_buffer("fit_pos", torch.arange(len(self.sidx), device=dev) if fit_pos is None
                             else torch.as_tensor(fit_pos, dtype=torch.long, device=dev))
        self.register_buffer("edges", e)
        tmpl = body.v_template
        dd, nn_ = (tmpl[:, None] - tmpl[self.sidx][None]).norm(dim=-1).topk(8, dim=1, largest=False)
        wgt = torch.exp(-(dd / 0.04) ** 2) + 1e-9
        self.register_buffer("nn", nn_)
        self.register_buffer("wgt", wgt / wgt.sum(1, keepdim=True))
        self.n_kp, self.w_2d, self.V = reg23.shape[0], w_2d, V
        self.register_buffer("rows2d", torch.arange(V, V + reg23.shape[0], device=dev))

    def forward(self, surf, mesh, kp, kp_uv, Kc):
        R, betas, v = self.onepass(surf[:, self.fit_pos])
        off = surf - mesh[:, self.sidx]
        dm = mesh + (off[:, self.nn] * self.wgt[None, ..., None]).sum(2)  # dense-refined mesh
        B = len(surf)
        X = torch.cat([dm, kp], 1)
        w3 = torch.cat([torch.full((B, self.V), 468.0 / self.V, device=surf.device),
                        torch.ones(B, self.n_kp, device=surf.device)], 1)
        uv = torch.cat([torch.zeros(B, self.V, 2, device=surf.device), kp_uv], 1)
        w2 = torch.cat([torch.zeros(B, self.V, device=surf.device), torch.ones(B, self.n_kp, device=surf.device)], 1)
        Xe = dm[:, self.edges[:, 1]] - dm[:, self.edges[:, 0]]
        R, betas, _, t2 = self.gn(R, betas, X, w3, Xe=Xe, uv=uv, w2=w2, Kc=Kc, w_2d=self.w_2d, rows2d=self.rows2d)
        v, _ = self.gn.smpl(R, betas, full=True)
        return R, betas, v + t2[:, None]


class GraphDenseFit:
    """DenseFit replayed from a CUDA graph captured per person count (outputs are the graph's buffers: copy to keep)."""

    def __init__(self, fit, graph=True):
        self.fit, self.graph, self.graphs = fit.eval(), graph, {}

    @torch.no_grad()
    def __call__(self, surf, mesh, kp, kp_uv, Kc):
        x = (surf, mesh, kp, kp_uv, Kc)
        if not self.graph:
            return self.fit(*x)
        n = len(surf)
        if n not in self.graphs:
            static = [t.clone() for t in x]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.fit(*static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.fit(*static)
            self.graphs[n] = (g, static, out)
        g, static, out = self.graphs[n]
        for a, b in zip(static, x):
            a.copy_(b)
        g.replay()
        return out
