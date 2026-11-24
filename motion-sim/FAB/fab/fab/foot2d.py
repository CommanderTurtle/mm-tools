"""2D foot-direction step after the foot refine, on the dense head's own 2D foot keypoints (MHR70 heel / big toe / small
toe): each fitted foot is rotated about its ankle until its projected heel -> toe direction lies on the dense 2D direction,
with the smallest 3D rotation: its heel -> toe vector is projected onto the plane through the camera centre and the 2D line
drawn from the fitted heel's own projection along the dense direction. Per foot weight w in [0, 1] scales the rotation
angle (0 = keep: invisible / truncated / too short in 2D). The result is a target mesh for
fab/foot_refine.FootRefine._closed, which then re-solves the ankle / toe rotations. Elementwise ops only (CUDA graphs).
"""

import math

import torch

FOOT_V = ((3387, (3216, 3226)), (6787, (6617, 6624)))  # SMPL heel, (big toe, small toe) vertices: left, right
ANKLE = (7, 8)
FOOT_MHR = ((17, 15, 16), (20, 18, 19))  # MHR70 heel, big toe, small toe (left, right)


def foot_vertex_sets(body):
    """Per side the SMPL vertices skinned mostly to that ankle / foot (joints 7, 10 left; 8, 11 right)."""
    part = body.weights.argmax(1)
    return [torch.nonzero(torch.isin(part, torch.tensor(j, device=part.device))).flatten() for j in ((7, 10), (8, 11))]


def _ray(uv, K):
    return torch.cat([(uv - K[:, :2, 2]) / torch.stack([K[:, 0, 0], K[:, 1, 1]], -1), torch.ones_like(uv[:, :1])], -1)


def _proj(x, K):
    p = torch.einsum("bij,bj->bi", K, x)
    return p[:, :2] / p[:, 2:].clamp_min(1e-6)


def _axis_angle(v):
    """(B, 3) axis-angle -> (B, 3, 3) rotation (Rodrigues)."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / th
    K_ = torch.zeros(len(v), 3, 3, device=v.device, dtype=v.dtype)
    K_[:, 0, 1], K_[:, 0, 2], K_[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K_[:, 1, 2], K_[:, 2, 0], K_[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=v.device, dtype=v.dtype) + s * K_ + (1 - c) * (K_ @ K_)


def _between(a, b, frac=None):
    """(B, 3) unit vectors -> the minimal rotation a -> b (B, 3, 3), its angle scaled by frac (B,) when given."""
    ax = torch.cross(a, b, dim=-1)
    ang = torch.atan2(ax.norm(dim=-1), (a * b).sum(-1))
    if frac is not None:
        ang = ang * frac
    return _axis_angle(ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-12) * ang[:, None])


def foot2d_target(verts, joints, uv_feet, K, w, foot_sets, max_deg=60.0, iters=3, min_deg=0.0):
    """verts (B, 6890, 3) camera frame; joints (B, 24, 3) their SMPL joints; uv_feet (B, 2, 3, 2) the dense 2D heel /
    big toe / small toe per side (px of K); K (B, 3, 3); w (B, 2) per-foot weights -> target mesh (B, 6890, 3).
    max_deg: feet whose projected heel -> toe is more than this off the dense 2D direction stay as fitted (left / right
    swaps, flipped feet: the in-plane projection would point the toe backwards). The rotation is about the ankle, which
    moves the heel (and the plane through its projection): iters fixed-point steps. min_deg > 0: only feet whose fitted
    image direction is more than min_deg off the dense one are turned (the cheap / conservative variant)."""
    tgt = verts.clone()
    for s in range(2):
        heel, toes = FOOT_V[s]
        h3 = verts[:, heel]
        d3 = (verts[:, toes[0]] + verts[:, toes[1]]) / 2 - h3  # int indices: no index tensor (CUDA graph capture)
        d2 = uv_feet[:, s, 1:].mean(1) - uv_feet[:, s, 0]  # the dense head's heel -> toe direction in the image
        dn = d2 / d2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ph = _proj(h3, K)
        f2 = _proj(h3 + d3, K) - ph  # the fitted foot's own image direction
        c2 = (f2 * d2).sum(-1) / (f2.norm(dim=-1) * d2.norm(dim=-1)).clamp_min(1e-9)
        anc = joints[:, ANKLE[s]]
        a = d3 / d3.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        b = a
        for _ in range(iters):
            hk = anc + torch.einsum("bij,bj->bi", _between(a, b), h3 - anc)  # the heel after the current rotation
            pk = _proj(hk, K)
            n = torch.cross(_ray(pk, K), _ray(pk + dn * 20.0, K), dim=-1)
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dp = d3 - (d3 * n).sum(-1, keepdim=True) * n  # the fitted direction's closest one in that plane
            b = dp / dp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ws = w[:, s] * (c2 > math.cos(math.radians(max_deg))).float()
        if min_deg > 0:
            ws = ws * (c2 < math.cos(math.radians(min_deg))).float()
        Rf = _between(a, b, ws)
        idx = foot_sets[s]
        tgt[:, idx] = torch.einsum("bij,bnj->bni", Rf, verts[:, idx] - anc[:, None]) + anc[:, None]
    return tgt


def foot2d_step(refine, body, R, betas, verts, uv70, K, size, strength=1.0, vis=None, max_deg=60.0, foot_sets=None,
                iters=3, min_deg=0.0, target_fn=None):
    """The step on a fitted SMPL: verts (B, 6890, 3) camera frame, uv70 (B, >= 21, 2) the dense MHR70 2D (px of K, a
    crop of `size` px), vis (B, >= 21) P(visible) or None; refine(R, betas, verts, target) the closed-form FootRefine
    (or its graph) -> R, verts. target_fn(verts, J, uvf, K, w): foot2d_target with the remaining arguments bound (e.g.
    a CUDA graph of it, GraphFoot2D)."""
    fm = torch.as_tensor(FOOT_MHR, device=uv70.device)
    uvf = uv70[:, fm]  # (B, 2, 3, 2)
    w = foot2d_weights(uvf, size, vis=None if vis is None else vis[:, fm]) * strength
    J = torch.einsum("jv,bvc->bjc", body.J_regressor.float(), verts)
    if target_fn is not None:
        tgt = target_fn(verts, J, uvf, K, w)
    else:
        tgt = foot2d_target(verts, J, uvf, K, w, foot_sets if foot_sets is not None else foot_vertex_sets(body), max_deg,
                            iters, min_deg)
    return refine(R, betas, verts, tgt)


class GraphFoot2D:
    """foot2d_target replayed from a CUDA graph per person count (its ~10^2 elementwise kernels launched as one)."""

    def __init__(self, foot_sets, max_deg=60.0, iters=3, min_deg=0.0):
        self.args, self.graphs = (foot_sets, max_deg, iters, min_deg), {}

    def _run(self, verts, J, uvf, K, w):
        fs, mx, it, mn = self.args
        return foot2d_target(verts, J, uvf, K, w, fs, mx, it, mn)

    @torch.no_grad()
    def __call__(self, verts, J, uvf, K, w):
        x = (verts, J, uvf, K, w)
        n = len(verts)
        if not verts.is_cuda:
            return self._run(*x)
        if n not in self.graphs:
            static = [t.float().clone() for t in x]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._run(*static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._run(*static)
            self.graphs[n] = (g, static, out)
        g, static, out = self.graphs[n]
        for dst, src in zip(static, x):
            dst.copy_(src)
        g.replay()
        return out.clone()


def foot2d_weights(uv_feet, size, margin=0.02, min_len=6.0, vis=None):
    """(B, 2) weights: 1 for a foot whose 3 dense 2D points lie inside the crop (margin of the crop size) with a
    heel -> toe length >= min_len px, times min P(visible) over its points (vis (B, 2, 3) or None)."""
    lo, hi = margin * size, (1 - margin) * size
    inside = ((uv_feet >= lo) & (uv_feet <= hi)).all(-1).all(-1)
    long_ = (uv_feet[:, :, 1:].mean(2) - uv_feet[:, :, 0]).norm(dim=-1) >= min_len
    w = (inside & long_).float()
    if vis is not None:
        w = w * vis.min(-1).values
    return w


class GraphFoldTarget:
    """foot_2d folded into the foot refine (fab/predictor.py foot_2d_mode "fold"), one CUDA graph per person
    count: the refine's target mesh (mesh frame) placed by the MHR cam_t, its SMPL joints, the per-foot weights (crop,
    length, x P(visible)) and foot2d_target, back in the mesh frame."""

    def __init__(self, J_regressor, foot_sets, max_deg, iters, size, strength):
        # only the ankles (SMPL 7, 8) are read by foot2d_target: regress those two rows
        self.J2 = J_regressor[list(ANKLE)].clone()
        self.args, self.size, self.strength, self.graphs = (foot_sets, max_deg, iters), size, strength, {}
        self.fm = torch.as_tensor(FOOT_MHR, device=J_regressor.device)

    def _run(self, ftm, cam_t, uv01, cam_int, affine, vis_logits=None):
        bottom = torch.zeros_like(affine[:, :1])  # the crop camera without a host tensor (graph capture)
        bottom[..., 2] = 1.0
        Kc = torch.cat([affine, bottom], 1) @ cam_int
        tg = ftm + cam_t[:, None]
        uvf = (uv01 * self.size)[:, self.fm]
        pv = None if vis_logits is None else torch.softmax(vis_logits, -1)[:, :70, 0][:, self.fm]
        w = foot2d_weights(uvf, self.size, vis=pv) * self.strength
        J = torch.zeros(len(tg), 24, 3, device=tg.device, dtype=tg.dtype)
        J[:, ANKLE[0] : ANKLE[1] + 1] = torch.einsum("jv,bvc->bjc", self.J2, tg)
        fs, mx, it = self.args
        return foot2d_target(tg, J, uvf, Kc, w, fs, mx, it) - cam_t[:, None]

    @torch.no_grad()
    def __call__(self, ftm, cam_t, uv01, cam_int, affine, vis_logits=None):
        x = tuple(t for t in (ftm, cam_t, uv01, cam_int, affine, vis_logits) if t is not None)
        n, key = len(ftm), (len(ftm), vis_logits is not None)
        if not ftm.is_cuda:
            return self._run(*x)
        if key not in self.graphs:
            static = [t.float().clone() for t in x]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._run(*static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._run(*static)
            self.graphs[key] = (g, static, out)
        del n
        g, static, out = self.graphs[key]
        for dst, src in zip(static, x):
            dst.copy_(src)
        g.replay()
        return out.clone()
