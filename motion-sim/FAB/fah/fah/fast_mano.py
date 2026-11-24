"""Fast batched MANO (right hand) in torch for per-frame mesh fitting: the kinematic chain is evaluated level by
level (all joints of one depth in one batched matmul: 3 levels instead of 15 sequential products), no Python-side
tensor indexing (CUDA-graph capturable). Same math as smplx.MANO with use_pca=False; poses are axis-angle in the
flat-hand convention (MANO's mean pose not added).

MeshFitter: Adam on pose / shape / global rotation / translation such that the mesh matches a predicted MANO-
topology surface and the 21 joints match the predicted joints (Huber); the whole iteration loop is captured in one
CUDA graph per batch size, so a 2-hand fit costs a few milliseconds.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

MANO21_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MANO_TIP_VERTS = [744, 320, 443, 554, 671]


def rodrigues(w, eye=None):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3), safe at zero. eye: a (3, 3) identity to use (inside
    compiled vmap / jacfwd, factory calls with a device fail: pass it as an input)."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    z = k[..., 0] * 0
    K = torch.stack([z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z], -1).reshape(*w.shape[:-1], 3, 3)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, device=w.device, dtype=w.dtype) if eye is None else eye
    return eye + s * K + (1 - c) * (K @ K)


class FastMano(torch.nn.Module):
    def __init__(self, mano_dir=None, device="cuda", subset=None):
        """subset: vertex indices to evaluate (the fingertip vertices are appended; joints stay exact: the joint
        regressor is folded into the template / shape basis) -> forward returns those vertices (+ tips) only."""
        super().__init__()
        mano_dir = mano_dir or os.environ.get("MANO_DIR", "data/mano")
        m = pickle.load(open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
        dense = lambda x: torch.as_tensor(np.asarray(x.todense() if hasattr(x, "todense") else x, np.float32))
        v_t, sd, pd, Jr, W = dense(m["v_template"]), dense(m["shapedirs"])[..., :10], dense(m["posedirs"]), dense(m["J_regressor"]), dense(m["weights"])
        self.register_buffer("J_t", Jr @ v_t)  # (16, 3) rest joints of the mean shape
        self.register_buffer("J_dirs", torch.einsum("jv,vcs->jcs", Jr, sd))  # (16, 3, 10)
        idx = np.arange(778) if subset is None else np.concatenate([np.asarray(subset, np.int64), MANO_TIP_VERTS])
        self.n = len(idx)
        self.tip_slot = list(range(self.n - 5, self.n)) if subset is not None else MANO_TIP_VERTS
        idx_t = torch.as_tensor(idx)
        self.register_buffer("v_t", v_t[idx_t])  # (n, 3)
        self.register_buffer("shapedirs", sd[idx_t])  # (n, 3, 10)
        self.register_buffer("posedirs", pd[idx_t].reshape(self.n * 3, 135).T.contiguous())  # (135, 3n)
        self.register_buffer("W", W[idx_t])  # (n, 16)
        par = np.asarray(m["kintree_table"][0], np.int64)
        par[0] = -1
        depth = np.zeros(16, np.int64)
        for k in range(1, 16):
            depth[k] = depth[par[k]] + 1
        self.levels = [[int(k) for k in np.where(depth == d)[0]] for d in range(1, depth.max() + 1)]  # joints per tree depth
        self.par = [int(p) for p in par]  # plain Python ints (torch.compile treats numpy arrays as tensors)
        for d, idx in enumerate(self.levels):  # device index tensors (CPU index lists copy during CUDA graph capture)
            self.register_buffer(f"lvi{d}", torch.tensor(idx))
            self.register_buffer(f"lpi{d}", torch.tensor([self.par[k] for k in idx]))
        self.register_buffer("tips", torch.tensor(self.tip_slot))
        self.register_buffer("eye3", torch.eye(3))
        self.register_buffer("order", torch.tensor(MANO21_TO_OPENPOSE))
        self.to(device)

    def bufs(self):
        """All model tensors as a dict (passed explicitly into compiled / vmapped functions)."""
        return dict(self.named_buffers())

    def forward(self, pose, betas, b=None):
        """pose (B, 45) flat-hand axis-angle of joints 1..15, betas (B, 10) -> root-relative vertices (B, 778, 3),
        joints (B, 21, 3, OpenPose order), both relative to the wrist and with identity global orientation.
        b: the buffers as a dict (default: this module's)."""
        b = b if b is not None else self.bufs()
        B = pose.shape[0]
        v = b["v_t"][None] + torch.einsum("vcs,bs->bvc", b["shapedirs"], betas)
        J = b["J_t"][None] + torch.einsum("jcs,bs->bjc", b["J_dirs"], betas)  # (B, 16, 3)
        eye = b["eye3"]
        R = rodrigues(pose.reshape(B, 15, 3), eye)  # (B, 15, 3, 3)
        v = v + ((R - eye).reshape(B, 135) @ b["posedirs"]).reshape(B, self.n, 3)
        Rall = torch.cat([eye.expand(B, 1, 3, 3), R], 1)  # (B, 16, 3, 3)
        GRl, GTl = [Rall[:, 0]] + [None] * 15, [J[:, 0]] + [None] * 15  # global transforms, level by level (out of place: vmap / compile)
        for d, idx in enumerate(self.levels):
            pidx = [self.par[k] for k in idx]
            li, lp = b[f"lvi{d}"], b[f"lpi{d}"]
            pr, pt = torch.stack([GRl[p] for p in pidx], 1), torch.stack([GTl[p] for p in pidx], 1)
            nr = pr @ Rall.index_select(1, li)
            nt = pt + (pr @ (J.index_select(1, li) - J.index_select(1, lp))[..., None])[..., 0]
            for c, k in enumerate(idx):
                GRl[k], GTl[k] = nr[:, c], nt[:, c]
        GR, GT = torch.stack(GRl, 1), torch.stack(GTl, 1)
        # LBS: x' = sum_k W_vk (GR_k (x - J_k) + GT_k)
        A = torch.einsum("vk,bkij->bvij", b["W"], GR)  # (B, 778, 3, 3)
        off = torch.einsum("vk,bki->bvi", b["W"], GT - (GR @ J[..., None])[..., 0])
        verts = (A @ v[..., None])[..., 0] + off
        joints = torch.cat([GT, verts[:, b["tips"]]], 1)[:, b["order"]]
        root = joints[:, :1]
        return verts - root, joints - root


class MeshFitter:
    """Batched MANO fit to predicted surfaces / joints (3D terms only: on HOT3D the 2D-joint term and pose / shape
    priors pull the mesh away from the GT). One functional Adam step (torch.func.grad) compiled with inductor; the
    whole iteration loop captured in one CUDA graph per batch size."""

    def __init__(self, mano_dir=None, device="cuda", iters=30, lr=0.03, wv=1.0, wj=1.0, graph=True, compile=True,
                 solver="adam", damping=1e-3, subset=None, jac="ad", fd_eps=1e-3, centre=False):
        """subset: fit only these vertices of the predicted surface (e.g. the points a subset model predicts); the
        output mesh is always the full 778. centre: fit() relative to the target wrist (as MeshStage centre)."""
        self.mano = FastMano(mano_dir, device)
        self.subset = None if subset is None else np.asarray(subset, np.int64)
        self.fm = self.mano if subset is None else FastMano(mano_dir, device, subset=self.subset)
        self.nv = 778 if subset is None else len(self.subset)
        self.dev, self.iters, self.wv, self.wj = torch.device(device), iters, wv, wj
        self.graph = graph and self.dev.type == "cuda"
        self.lr = torch.tensor([lr] * 45 + [lr * 0.5] * 10 + [lr] * 3 + [lr * 0.05] * 3, device=self.dev)
        self.solver, self.damping = solver, damping
        # Gauss-Newton Jacobian: "ad" (vmap(jacfwd)), "fd" (central differences) or "fdb" (= fd with the perturbed sets as one
        # MANO batch instead of vmap: dense matmuls rather than hundreds of tiny ones, 3x faster on the CPU)
        self.jac, self.fd_eps, self.centre = jac, fd_eps, centre
        fn = self._gn_step if solver == "gn" else self._step
        self.B = self.fm.bufs()  # model tensors as explicit inputs (inductor + vmap(jacfwd) fail on module buffers)
        self.B["rscale"] = torch.cat([torch.full((3 * self.nv,), (wv / self.nv) ** 0.5), torch.full((63,), (wj / 21) ** 0.5)]).to(self.dev)
        self.B["eye61"] = torch.eye(61, device=self.dev)
        compile = compile and (solver != "gn" or jac in ("fd", "fdb"))  # inductor fails on vmap(jacfwd) here (FakeTensor device propagation): AD-GN runs eager in the CUDA graph
        self.step = torch.compile(fn, fullgraph=True, dynamic=False) if compile else fn
        self.graphs = {}

    def _residual(self, x, tV, tJ, R0, b):
        """One hand: x (61,), tV (nv, 3), tJ (21, 3), R0 (3, 3), b model tensors -> weighted residuals (3 nv + 63,)."""
        verts, joints = self.fm(x[None, :45], x[None, 45:55], b)
        R = R0 @ rodrigues(x[55:58], b["eye3"])
        verts = verts[0, :self.nv] @ R.T + x[58:61]
        joints = joints[0] @ R.T + x[58:61]
        return torch.cat([(verts - tV).reshape(-1) * (self.wv / self.nv) ** 0.5, (joints - tJ).reshape(-1) * (self.wj / 21) ** 0.5])

    def _residuals(self, X, tV, tJ, R0, b):
        """_residual for a batch of parameter sets X (N, 61) and their targets, without vmap."""
        verts, joints = self.fm(X[:, :45], X[:, 45:55], b)
        R = R0 @ rodrigues(X[:, 55:58], b["eye3"])
        t = X[:, None, 58:61]
        verts = (verts[:, :self.nv, None, :] * R[:, None]).sum(-1) + t  # rows @ R^T, elementwise (no batched 3x3 matmul)
        joints = (joints[:, :, None, :] * R[:, None]).sum(-1) + t
        return torch.cat([(verts - tV).reshape(len(X), -1) * (self.wv / self.nv) ** 0.5,
                          (joints - tJ).reshape(len(X), -1) * (self.wj / 21) ** 0.5], 1)

    def _gn_step(self, x, m, v2, k, tV, tJ, R0, b):
        """Damped Gauss-Newton with Huber (IRLS) weights, batched over hands; m, v2 unused (same signature as Adam).
        jac "ad": forward-mode Jacobian (vmap(jacfwd)); "fd": central differences, all 2 x 61 perturbed parameter sets in
        one batched forward (far fewer kernels)."""
        res = torch.func.vmap(self._residual, in_dims=(0, 0, 0, 0, None))
        if self.jac in ("fd", "fdb"):
            B, n = x.shape
            E = b["eye61"] * self.fd_eps
            X = torch.cat([x[:, None], x[:, None] + E[None], x[:, None] - E[None]], 1).reshape(-1, n)  # (B (1 + 2 n), n)
            rep = lambda t: t[:, None].expand(B, 1 + 2 * n, *t.shape[1:]).reshape(-1, *t.shape[1:])
            ra = (self._residuals if self.jac == "fdb" else res)(X, rep(tV), rep(tJ), rep(R0), b).reshape(B, 1 + 2 * n, -1)
            r = ra[:, 0]
            Jac = ((ra[:, 1:1 + n] - ra[:, 1 + n:]) / (2 * self.fd_eps)).mT  # (B, ., 61)
        else:
            r = res(x, tV, tJ, R0, b)  # (B, 3 nv + 63)
            Jac = torch.func.vmap(torch.func.jacfwd(self._residual), in_dims=(0, 0, 0, 0, None))(x, tV, tJ, R0, b)  # (B, ., 61)
        n = r.abs() / b["rscale"]
        w = torch.clamp(0.01 / n.clamp(min=1e-9), max=1.0)  # Huber delta 1 cm
        Jw = Jac * w[..., None]
        H = Jac.mT @ Jw
        g = Jw.mT @ r[..., None]
        H = H + self.damping * torch.diag_embed(torch.diagonal(H, dim1=-2, dim2=-1)) + 1e-6 * b["eye61"]
        return x - torch.linalg.solve_ex(H, g)[0][..., 0], m, v2  # solve_ex: no info check (host sync breaks capture)

    def _loss(self, x, tV, tJ, R0):
        verts, joints = self.fm(x[:, :45], x[:, 45:55])
        verts = verts[:, :self.nv]
        R = R0 @ rodrigues(x[:, 55:58])
        verts = verts @ R.mT + x[:, None, 58:61]
        joints = joints @ R.mT + x[:, None, 58:61]
        lv = F.huber_loss(verts, tV, delta=0.01, reduction="none").sum(-1).mean(1)
        lj = F.huber_loss(joints, tJ, delta=0.01, reduction="none").sum(-1).mean(1)
        return (self.wv * lv + self.wj * lj).sum()

    def _step(self, x, m, v2, k, tV, tJ, R0, b=None):
        g = torch.func.grad(self._loss)(x, tV, tJ, R0)
        m = 0.9 * m + 0.1 * g
        v2 = 0.999 * v2 + 0.001 * g * g
        return x - self.lr * (m / (1 - 0.9 ** k)) / ((v2 / (1 - 0.999 ** k)).sqrt() + 1e-8), m, v2

    def _loop(self, s, iters):
        x, m, v2 = s["x0"], torch.zeros_like(s["x0"]), torch.zeros_like(s["x0"])
        for it in range(1, iters + 1):
            x, m, v2 = self.step(x, m, v2, s["k"][it - 1], s["tV"], s["tJ"], s["R0"], self.B)
        with torch.no_grad():
            verts, _ = self.mano(x[:, :45], x[:, 45:55])
            R = s["R0"] @ rodrigues(x[:, 55:58])
            s["out_v"].copy_(verts @ R.mT + x[:, None, 58:61])
            s["out_R"].copy_(R)
            s["out_x"].copy_(x)

    def fit(self, tV, tJ, R0, pose0, betas0=None, iters=None, params=False):
        """tV (B, 778, 3), tJ (B, 21, 3): predicted surface / joints (right-hand frame, camera coordinates);
        R0 (B, 3, 3), pose0 (B, 45, flat-hand convention), betas0 (B, 10): init (e.g. fah.mano_procrustes, or the
        previous frame's fit in a video with fewer iterations). -> mesh (B, 778, 3) [, (R (B, 3, 3), pose, betas)]."""
        B, iters = len(tV), iters or self.iters
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)
        key = (B, iters)
        if key not in self.graphs:
            s = dict(tV=torch.zeros(B, self.nv, 3, device=self.dev), tJ=torch.zeros(B, 21, 3, device=self.dev),
                     R0=torch.eye(3, device=self.dev).repeat(B, 1, 1), x0=torch.zeros(B, 61, device=self.dev),
                     k=torch.arange(1, iters + 1, device=self.dev, dtype=torch.float32)[:, None],
                     out_v=torch.zeros(B, 778, 3, device=self.dev), out_R=torch.zeros(B, 3, 3, device=self.dev),
                     out_x=torch.zeros(B, 61, device=self.dev))
            g = None
            if self.graph:
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    self._loop(s, iters)  # compile + allocations outside the capture
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._loop(s, iters)
            self.graphs[key] = (s, g)
        s, g = self.graphs[key]
        c = T(tJ)[:, :1] if self.centre else 0.0
        s["tV"].copy_(T(tV if self.subset is None or len(tV[0]) == self.nv else np.asarray(tV)[:, self.subset]) - c)
        s["tJ"].copy_(T(tJ) - c)
        s["R0"].copy_(T(R0))
        x0 = torch.zeros(B, 61, device=self.dev)
        x0[:, :45] = T(pose0)
        if betas0 is not None:
            x0[:, 45:55] = T(betas0)
        x0[:, 58:] = s["tJ"][:, 0]
        s["x0"].copy_(x0)
        if g is not None:
            g.replay()
        else:
            self._loop(s, iters)
        v = (s["out_v"] + c).cpu().numpy()
        if not params:
            return v
        x = s["out_x"].cpu().numpy()
        return v, (s["out_R"].cpu().numpy(), x[:, :45], x[:, 45:55])


def polar_rot(C, iters=10):
    """Rotation part of C (..., 3, 3) = U V^T of its SVD, by Newton's polar iteration X <- (X + X^-T) / 2 (batched
    3x3 inverses without host syncs: CUDA-graph friendly; converges quadratically, also for thin point sets)."""
    X = C / (C.square().sum((-2, -1), keepdim=True).sqrt() + 1e-12)
    for _ in range(iters):
        X = 0.5 * (X + torch.linalg.inv_ex(X)[0].mT)
    return X


def kabsch_horn(C, squarings=12):
    """Kabsch rotation R (R a ~ b) for C = sum a b^T (..., 3, 3) by Horn's quaternion method: the dominant eigenvector of
    the 4x4 matrix N(C), found by repeated squaring of N + |N| I (matmuls only: CUDA-graph friendly, no SVD; exact for
    thin / near-planar point sets, where the polar iteration needs a reflection fix)."""
    Sxx, Sxy, Sxz = C[..., 0, 0], C[..., 0, 1], C[..., 0, 2]
    Syx, Syy, Syz = C[..., 1, 0], C[..., 1, 1], C[..., 1, 2]
    Szx, Szy, Szz = C[..., 2, 0], C[..., 2, 1], C[..., 2, 2]
    N = torch.stack([
        torch.stack([Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], -1),
        torch.stack([Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz], -1),
        torch.stack([Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], -1),
        torch.stack([Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz], -1)], -2)
    M = N + N.square().sum((-2, -1), keepdim=True).sqrt() * torch.eye(4, device=C.device, dtype=C.dtype)
    for _ in range(squarings):
        M = M @ M
        M = M / (M.square().sum((-2, -1), keepdim=True).sqrt() + 1e-30)
    q = M[..., :, 0] + M[..., :, 1] + M[..., :, 2] + M[..., :, 3]  # any column of the rank-1 limit (sum avoids a zero column)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-30)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def kabsch_rot(C):
    """Kabsch rotation V diag(1, 1, d) U^T for C = U S V^T (..., 3, 3) without an SVD: polar factor Q = U V^T, and for
    a reflection (det C < 0, thin point sets) the smallest singular direction flipped (inverse iteration on Q^T C)."""
    Q = polar_rot(C)
    H = Q.mT @ C  # V S V^T
    eps = 1e-9 * H.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] + 1e-15
    Hi = torch.linalg.inv_ex(H + eps * torch.eye(3, device=C.device, dtype=C.dtype))[0]
    x = torch.ones_like(C[..., 0]) / 3 ** 0.5
    for _ in range(3):
        x = (Hi @ x[..., None])[..., 0]
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-30)
    v3, u3 = x, (Q @ x[..., None])[..., 0]
    d = torch.sign(torch.linalg.det(Q))
    return Q.mT - (1 - d)[..., None, None] * v3[..., :, None] * u3[..., None, :]


class MeshStage:
    """The whole mesh stage on the device in one CUDA graph per batch size: closed-form per-part Procrustes init
    (part covariances + Newton-Schulz polar rotations, log map to the flat-hand pose), optional learned one-pass fitter
    (train/mesh_fitter.py), then Gauss-Newton steps (MeshFitter). One host-to-device and one device-to-host copy."""

    def __init__(self, mano_dir=None, device="cuda", iters=1, wv=0.3, lmf=None, stride=1, jac="ad", centre=False, fd_eps=1e-3):
        """stride > 1: the Gauss-Newton residuals on every stride-th vertex of the surface (cheaper Jacobian); the
        init, the learned fitter and the output mesh stay full. centre: Gauss-Newton residuals relative to the target
        wrist instead of in camera coordinates (same objective; float32 rounding of metre-scale coordinates otherwise
        dominates the finite-difference Jacobian of far hands); fd_eps: the finite-difference step (jac fd / fdb)."""
        mano_dir = mano_dir or os.environ.get("MANO_DIR", "data/mano")
        self.dev = torch.device(device)
        self.gn_idx = np.arange(0, 778, stride) if stride > 1 else None
        self.fitter = MeshFitter(mano_dir, device, iters=max(iters, 1), solver="gn", wv=wv, subset=self.gn_idx, jac=jac, fd_eps=fd_eps)
        self.iters, self.centre = iters, centre
        m = pickle.load(open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
        dense = lambda x: np.asarray(x.todense() if hasattr(x, "todense") else x, np.float64)
        v = dense(m["v_template"])
        part = dense(m["weights"]).argmax(1)
        oh = np.eye(16)[part]
        cnt = oh.sum(0)
        vc = v - oh @ (oh.T @ v / cnt[:, None])
        par = np.asarray(m["kintree_table"][0], np.int64)
        T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.dev)
        self.oh, self.ohm, self.vc = T(oh), T(oh / cnt[None]), T(vc)  # (778, 16), mean weights, centred template
        self.par = T(par[1:], torch.int64)
        self.lmf = None
        if lmf:
            from train.mesh_fitter import SUB, Fitter

            ck = torch.load(lmf, map_location=self.dev, weights_only=False)
            self.lmf = Fitter(ck["d_in"], ck.get("hidden", 1024)).to(self.dev).eval()
            self.lmf.load_state_dict(ck["model"])
            self.sub = T(SUB, torch.int64)
        self.gn_sel = None if self.gn_idx is None else torch.as_tensor(self.gn_idx, device=self.dev)
        self.graphs = {}
        if self.dev.type == "cpu":  # no CUDA graph: the whole stage as one compiled graph instead (eager init / fitter / output ops)
            self._run = torch.compile(self._run, dynamic=False)

    def _init(self, tV, tJ):
        """Closed-form init: R0 (B, 3, 3) root rotation, p0 (B, 45) flat-hand pose."""
        Pc = tV - torch.einsum("vk,bkc->bvc", self.oh, torch.einsum("vk,bvc->bkc", self.ohm, tV))
        C = torch.einsum("vk,vi,bvj->bkij", self.oh, self.vc, Pc)  # (B, 16, 3, 3) template x prediction
        R = kabsch_horn(C)
        L = R[:, self.par].mT @ R[:, 1:]  # (B, 15, 3, 3) local rotations
        cos = ((L.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
        ang = torch.acos(cos)
        w = torch.stack([L[..., 2, 1] - L[..., 1, 2], L[..., 0, 2] - L[..., 2, 0], L[..., 1, 0] - L[..., 0, 1]], -1)
        p0 = (w * (ang / (2 * torch.sin(ang).clamp(min=1e-6)))[..., None]).reshape(len(tV), 45)
        return R[:, 0], p0

    def _run(self, s):
        with torch.no_grad():
            R0, p0 = self._init(s["tV"], s["tJ"])
            b0 = torch.zeros(len(p0), 10, device=self.dev)
            if self.lmf is not None:
                w = s["tJ"][:, :1]
                x = torch.cat([(s["tV"][:, self.sub] - w) @ R0, (s["tJ"] - w) @ R0], 1).flatten(1)
                p0, b0, dr, _ = self.lmf(x, p0)
                R0 = R0 @ rodrigues(dr)
        f = self.fitter
        c = s["tJ"][:, :1] if self.centre else torch.zeros_like(s["tJ"][:, :1])
        tV, tJ = s["tV"] - c, s["tJ"] - c
        x0 = torch.cat([p0, b0, torch.zeros(len(p0), 3, device=self.dev), tJ[:, 0]], 1)
        f_s = s["fs"]
        f_s["tV"].copy_(tV if self.gn_idx is None else tV[:, self.gn_sel]), f_s["tJ"].copy_(tJ), f_s["R0"].copy_(R0), f_s["x0"].copy_(x0)
        if self.iters > 0:
            f._loop(f_s, self.iters)
            s["out"].copy_(f_s["out_v"] + c)
        else:
            with torch.no_grad():
                v, _ = f.mano(p0, b0)
                s["out"].copy_(v @ R0.mT + s["tJ"][:, None, 0])

    def __call__(self, tV, tJ):
        """tV (B, 778, 3), tJ (B, 21, 3) numpy, right-hand frame -> MANO meshes (B, 778, 3) numpy. Chunks of <= 2 hands
        (one graph per size 1 / 2; the batched linear solve takes a non-capturable path for larger batches)."""
        if len(tV) > 2:
            return np.concatenate([self(tV[i:i + 2], tJ[i:i + 2]) for i in range(0, len(tV), 2)])
        B = len(tV)
        if B not in self.graphs:
            Z = lambda *sh: torch.zeros(*sh, device=self.dev)
            fs = dict(tV=Z(B, 778 if self.gn_idx is None else len(self.gn_idx), 3), tJ=Z(B, 21, 3), R0=Z(B, 3, 3), x0=Z(B, 61), out_v=Z(B, 778, 3), out_R=Z(B, 3, 3),
                      out_x=Z(B, 61), k=torch.arange(1, max(self.iters, 1) + 1, device=self.dev, dtype=torch.float32)[:, None])
            s = dict(tV=Z(B, 778, 3), tJ=Z(B, 21, 3), out=Z(B, 778, 3), fs=fs)
            s["tV"].copy_(torch.as_tensor(np.asarray(tV), dtype=torch.float32))
            s["tJ"].copy_(torch.as_tensor(np.asarray(tJ), dtype=torch.float32))
            g = None
            if self.dev.type == "cuda":
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    self._run(s)
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._run(s)
            self.graphs[B] = (s, g)
        s, g = self.graphs[B]
        inp = torch.as_tensor(np.concatenate([np.asarray(tV, np.float32).reshape(B, -1), np.asarray(tJ, np.float32).reshape(B, -1)], 1))
        inp = inp.to(self.dev, non_blocking=True)
        s["tV"].copy_(inp[:, :778 * 3].reshape(B, 778, 3))
        s["tJ"].copy_(inp[:, 778 * 3:].reshape(B, 21, 3))
        if g is not None:
            g.replay()
        else:
            self._run(s)
        return s["out"].cpu().numpy().copy()  # copy: on the CPU .cpu() is the reused buffer itself
