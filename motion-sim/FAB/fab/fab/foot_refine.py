"""Foot refinement of a fitted SMPL: the one-pass fit loses the feet of extreme poses
(MOYO's pointed feet: big-toe PA 92 vs 66 mm for the MHR mesh it is fitted next to). Only the ankle and foot joint
rotations (SMPL 7, 8, 10, 11) are re-solved so that each fitted foot, aligned at its ankle, matches a target mesh's foot
(the MHR mesh in SMPL topology); the rest of the body, its joints and the ankle positions stay as fitted.

    refine = FootRefine(SMPL(".../SMPL_NEUTRAL.pkl").cuda())
    R, verts = refine(R, betas, verts, target)  # (N, 24, 3, 3) local rotations, fitted (N, 6890, 3), target (N, 6890, 3)

iters > 0: Adam on the four joints' axis-angle offsets (vertex loss, ankle-aligned feet). iters = 0: closed form (the
deployment path, two SMPL passes): per foot, the rotation about the ankle that best maps the fitted foot onto the
target's (Kabsch on ankle-relative vertices) updates the ankle's global rotation, then the same for the toes about the
foot joint.
"""

import math

import torch

from bench.body_models import _rodrigues

FOOT_JOINTS = (7, 8, 10, 11)
FOOT_V = {"l": (3387, 3216, 3226), "r": (6787, 6617, 6624)}  # SMPL heel, big toe, small toe vertices
SIDES = ((7, (7, 10)), (8, (8, 11)))  # ankle joint, skinning parts of the foot


class FootRefine:
    def __init__(self, body, iters=40, lr=0.03, w_prior=1e-3, passes=1, squarings=12, mode="kabsch", gate=0.0,
                 gate_hi=20.0, sole_w=0.0, sole_ny=-0.5):
        """mode kabsch: the whole foot (and toes) rotated onto the target's; swing: only the heel -> toe direction is
        turned onto the target's (minimal rotation about the ankle; the fit's roll about the foot axis and its toes are
        kept, so the sole plane follows the fit); hybrid: swing, and kabsch for feet whose fitted direction is more than
        gate_hi (deg) off the target's (extreme foot poses the fit regularised away). gate (deg): feet whose fitted
        direction is within gate of the target's are left as fitted."""
        self.body, self.iters, self.lr, self.w_prior, self.passes = body, iters, lr, w_prior, passes
        self.mode, self.gate, self.gate_hi = mode, gate, gate_hi
        self.squarings = squarings  # Horn's eigenvector by repeated squaring: 12 suffices for near-aligned feet
        part = body.weights.argmax(1)
        self.feet = [(a, torch.nonzero(torch.isin(part, torch.tensor(p, device=part.device)))[:, 0]) for a, p in SIDES]
        self.toes = [torch.nonzero(part == f)[:, 0] for f in (10, 11)]
        # sole_w > 0: the Kabsch weights the plantar vertices (rest-pose normal facing down: n_y < sole_ny) by
        # 1 + sole_w, so the fitted sole plane follows the target's (the deployed batched kabsch mode only)
        self.sole_w = float(sole_w)
        if self.sole_w > 0:
            v, f = body.v_template.float(), body.faces.long()
            fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
            vn = torch.zeros_like(v).index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
            vn = vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if ((v - v.mean(0)) * vn).sum() < 0:  # outward
                vn = -vn
            self.w_feet = [1.0 + self.sole_w * (vn[ix, 1] < sole_ny).float() for _, ix in self.feet]
            self.w_toes = [w[torch.isin(ix, t)] for (_, ix), w, t in zip(self.feet, self.w_feet, self.toes)]

    def _verts(self, R, betas):
        z = torch.zeros(len(R), 3, device=R.device)
        return self.body(z, z.repeat(1, 23), betas, None, rotmats=R)

    def __call__(self, R, betas, verts, target):
        with torch.inference_mode(False):  # callers may run under inference mode; the solve needs autograd
            R, betas, verts, target = (x.float().clone() for x in (R, betas, verts, target))
            return self._solve(R, betas, verts, target)

    def _closed(self, R, betas, verts, target):
        from fab.fast_fit import FastSMPL, kabsch, kabsch_S

        if getattr(self, "fast", None) is None:  # batched kinematics, skinning only the feet (fab/fast_fit.py)
            idx = torch.cat([ix for _, ix in self.feet])
            self.fast = FastSMPL(self.body, idx).to(idx.device)
            pos = {int(v): i for i, v in enumerate(idx.tolist())}
            self.sub = [torch.tensor([pos[int(v)] for v in ix.tolist()], device=idx.device) for _, ix in self.feet]
            self.sub_toe = [torch.tensor([pos[int(v)] for v in t.tolist()], device=idx.device) for t in self.toes]
            self.kp = [torch.tensor([pos[v] for v in FOOT_V[s_]], device=idx.device) for s_ in "lr"]
            self.kp_full = [torch.tensor(FOOT_V[s_], device=idx.device) for s_ in "lr"]
            self.idx = idx
        fs = self.fast
        with torch.no_grad():
            J = self.body.J_regressor.float()
            jt = torch.einsum("jv,bvc->bjc", J, target)
            Rn = R.clone()
            first = None
            for _ in range(self.passes):
                v0, j0 = fs(Rn, betas, full=False)
                first = v0 if first is None else first
                G = Rn[:, :1]
                for js, pp in fs.levels:
                    G = torch.cat([G, G[:, pp] @ Rn[:, js]], 1)
                G = G[:, fs.inv]
                par = self.body.parents
                if self.mode == "kabsch" and self.gate <= 0 and getattr(self, "batched", True):  # the deployed mode:
                    # both feet in one solve, then both
                    # toes (the sides are independent: G / v0 / j0 are from before either update; ~half the kernels)
                    B = len(Rn)
                    S = torch.cat([torch.einsum("bni,bnj->bij", v0[:, si] - j0[:, a : a + 1], target[:, ix] - jt[:, a : a + 1])
                                   for (a, ix), si in zip(self.feet, self.sub)]) if self.sole_w <= 0 else \
                        torch.cat([torch.einsum("bni,bnj->bij", (v0[:, si] - j0[:, a : a + 1]) * w[None, :, None],
                                                target[:, ix] - jt[:, a : a + 1])
                                   for (a, ix), si, w in zip(self.feet, self.sub, self.w_feet)])
                    dRs = kabsch_S(S, self.squarings).split(B)
                    Gas = [dR @ G[:, a] for (a, _), dR in zip(self.feet, dRs)]
                    for (a, _), Ga in zip(self.feet, Gas):
                        Rn[:, a] = G[:, par[a]].transpose(1, 2) @ Ga
                    wt = self.w_toes if self.sole_w > 0 else [None, None]
                    S2 = torch.cat([torch.einsum("bni,bnj->bij", torch.einsum("bij,bnj->bni", dR, v0[:, st] - j0[:, f : f + 1])
                                                 * (1.0 if w is None else w[None, :, None]), target[:, tix] - jt[:, f : f + 1])
                                    for dR, f, st, tix, w in zip(dRs, (10, 11), self.sub_toe, self.toes, wt)])
                    dR2s = kabsch_S(S2, self.squarings).split(B)
                    for dR, Ga, dR2, f in zip(dRs, Gas, dR2s, (10, 11)):
                        Rn[:, f] = Ga.transpose(1, 2) @ (dR2 @ dR @ G[:, f])
                    continue
                assert self.sole_w <= 0, "sole_w: the batched kabsch mode only"
                for (a, ix), f, si, st, tix, kp, kpf in zip(self.feet, (10, 11), self.sub, self.sub_toe, self.toes,
                                                             self.kp, self.kp_full):
                    df = v0[:, kp[1:]].mean(1) - v0[:, kp[:1]].mean(1)  # fitted heel -> toe midpoint (no 0-d index:
                    dt = target[:, kpf[1:]].mean(1) - target[:, kpf[:1]].mean(1)  # it syncs, breaking graph capture)
                    df, dt = df / df.norm(dim=-1, keepdim=True).clamp_min(1e-9), dt / dt.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    eye = torch.eye(3, device=df.device).expand(len(df), 3, 3)
                    if self.mode == "swing":
                        dR = rotation_between(df, dt)
                    else:
                        dR = kabsch(v0[:, si] - j0[:, a : a + 1], target[:, ix] - jt[:, a : a + 1], self.squarings)
                    full = None
                    if self.mode == "hybrid":  # kabsch only where the fit is far off the target
                        full = (df * dt).sum(-1) < math.cos(math.radians(self.gate_hi))
                        dR = torch.where(full[:, None, None], dR, rotation_between(df, dt))
                    if self.gate > 0:  # near-agreeing feet stay as fitted
                        keep = (df * dt).sum(-1) > math.cos(math.radians(self.gate))
                        dR = torch.where(keep[:, None, None], eye, dR)
                    Ga = dR @ G[:, a]
                    Rf = Rn[:, f].clone()
                    Rn[:, a] = G[:, par[a]].transpose(1, 2) @ Ga
                    if self.mode == "swing":
                        continue  # the toes keep the fit's rotation relative to the foot
                    src = torch.einsum("bij,bnj->bni", dR, v0[:, st] - j0[:, f : f + 1])
                    dR2 = kabsch(src, target[:, tix] - jt[:, f : f + 1], self.squarings)
                    if self.gate > 0:
                        dR2 = torch.where(keep[:, None, None], eye, dR2)
                    Rn[:, f] = Ga.transpose(1, 2) @ (dR2 @ dR @ G[:, f])
                    if full is not None:  # swung feet keep the fit's toes
                        Rn[:, f] = torch.where(full[:, None, None], Rn[:, f], Rf)
            offset = (verts[:, self.idx] - first).mean(1, keepdim=True)  # the fit's placement (a translation)
            v, _ = fs(Rn, betas, full=True)
        return Rn, v + offset

    def _solve(self, R, betas, verts, target):
        if self.iters == 0:
            return self._closed(R, betas, verts, target)
        with torch.no_grad():
            v0, _ = self._verts(R, betas)
            offset = (verts - v0).mean(1, keepdim=True)  # the fit's placement (a translation)
            J = self.body.J_regressor.float()
            jt = torch.einsum("jv,bvc->bjc", J, target)
            tgt = [target[:, ix] - jt[:, a : a + 1] for a, ix in self.feet]
        d = torch.zeros(len(R), len(FOOT_JOINTS), 3, device=R.device, requires_grad=True)
        opt = torch.optim.Adam([d], lr=self.lr)
        with torch.enable_grad():
            for _ in range(self.iters):
                Rn = R.clone()
                Rn[:, list(FOOT_JOINTS)] = R[:, list(FOOT_JOINTS)] @ _rodrigues(d.reshape(-1, 3)).view(-1, len(FOOT_JOINTS), 3, 3)
                v, j = self._verts(Rn, betas)
                loss = sum(((v[:, ix] - j[:, a : a + 1] - t) ** 2).sum(-1).mean() for (a, ix), t in zip(self.feet, tgt))
                loss = loss + self.w_prior * (d ** 2).sum((1, 2)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            Rn = R.clone()
            Rn[:, list(FOOT_JOINTS)] = R[:, list(FOOT_JOINTS)] @ _rodrigues(d.reshape(-1, 3)).view(-1, len(FOOT_JOINTS), 3, 3)
            v, _ = self._verts(Rn, betas)
        return Rn, v + offset


def rotation_between(a, b):
    """(B, 3) unit vectors -> (B, 3, 3) the minimal rotation taking a to b (Rodrigues; graph-capturable)."""
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    z = torch.zeros_like(c)
    vx = torch.stack([z, -v[:, 2], v[:, 1], v[:, 2], z, -v[:, 0], -v[:, 1], v[:, 0], z], -1).view(-1, 3, 3)
    eye = torch.eye(3, device=a.device).expand_as(vx)
    return eye + vx + vx @ vx / (1 + c).clamp_min(1e-6)[:, None, None]


class GraphFootRefine:
    """The closed-form FootRefine replayed from a CUDA graph per person count (as fab/fast_fit.GraphFit;
    outputs are the graph's buffers: copy to keep)."""

    def __init__(self, body, passes=1, mode="kabsch", gate=0.0, gate_hi=20.0, sole_w=0.0):
        self.refine = FootRefine(body, iters=0, passes=passes, mode=mode, gate=gate, gate_hi=gate_hi, sole_w=sole_w)
        self.graphs = {}

    @torch.no_grad()
    def __call__(self, R, betas, verts, target):
        n = len(R)
        if not verts.is_cuda:  # CPU (FABPredictor(device="cpu")): no CUDA graph, as foot2d's graph classes
            return self.refine._closed(R, betas, verts, target)
        if n not in self.graphs:
            static = [x.float().clone() for x in (R, betas, verts, target)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.refine._closed(*static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.refine._closed(*static)
            self.graphs[n] = (g, static, out)
        g, static, out = self.graphs[n]
        for dst, src in zip(static, (R, betas, verts, target)):
            dst.copy_(src)
        g.replay()
        return out
