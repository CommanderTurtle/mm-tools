"""The learned one-pass SMPL fitter (inference): an MLP from the dense head's surface points (centred, camera axes) to
SMPL joint rotations (6D) and betas. FAB-L / FAB-S use the points fitter (weights/shared/smpl_fitter.pt), FAB-H the
mesh fitter (weights/shared/smpl_fitter_mesh.pt) on its mesh's surface points.
"""
import torch
import torch.nn as nn

from fab.retarget.g1 import rot6d_to_matrix


class SMPLFitter(nn.Module):
    """One MLP on the centred points. With a template (tmpl), the points are first rotated into the template's frame
    by the Kabsch rotation of the template points (the MLP then sees a rotation-normalised body; the global rotation is
    composed back). stages > 0: iterative refinement -- each stage sees the points and the current fit's points and
    predicts parameter updates (a learned Gauss-Newton step; needs `body` at forward)."""

    def __init__(self, n_points, hidden=1024, tmpl=None, stages=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_points * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 24 * 6 + 10))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.copy_(torch.cat([torch.tensor([1.0, 0, 0, 0, 1.0, 0]).repeat(24), torch.zeros(10)]))
        self.register_buffer("tmpl", torch.zeros(0, 3) if tmpl is None else torch.as_tensor(tmpl, dtype=torch.float32))
        self.refine = nn.ModuleList()
        for _ in range(stages):
            r = nn.Sequential(nn.Linear(n_points * 6 + 154, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                              nn.Linear(hidden, 154))
            nn.init.zeros_(r[-1].weight)
            nn.init.zeros_(r[-1].bias)
            self.refine.append(r)

    def forward(self, points, body=None, idx=None, detach_rot=False):
        """points (B, K, 3) -> joint rotations (B, 24, 3, 3), betas (B, 10); refinement needs body (SMPL) and idx.
        detach_rot: no gradient through the canonicalising Kabsch rotation (losses through the fit: the SVD's
        gradient is unstable for near-degenerate point sets)."""
        from fab.smpl_fit import kabsch

        c = points.mean(1, keepdim=True)
        x = points - c
        R0 = None
        if len(self.tmpl):  # rotate into the template frame (points_canonical = R0^T x)
            t = self.tmpl - self.tmpl.mean(0)
            R0 = kabsch(t[None].expand(len(x), -1, -1), x)
            if detach_rot:
                R0 = R0.detach()
            x = torch.einsum("bji,bnj->bni", R0, x)
        y = self.net((x * 5.0).flatten(1))
        rot6d, betas = y[:, :144], y[:, 144:]
        for r in self.refine:
            R = rot6d_to_matrix(rot6d.reshape(-1, 6)).view(-1, 24, 3, 3)
            z = torch.zeros(len(x), 3, device=x.device)
            v, _ = body(z, z.repeat(1, 23), betas, None, rotmats=R)
            vp = v[:, idx] - v[:, idx].mean(1, keepdim=True)  # current fit, canonical frame (global rot not applied)
            d = r(torch.cat([(x * 5.0).flatten(1), (vp * 5.0).flatten(1), rot6d, betas], 1))
            rot6d, betas = rot6d + d[:, :144], betas + d[:, 144:]
        R = rot6d_to_matrix(rot6d.reshape(-1, 6)).view(-1, 24, 3, 3)
        if R0 is not None:
            R = torch.cat([(R0 @ R[:, 0])[:, None], R[:, 1:]], 1)
        return R, betas


def load_fitter(path, device="cuda"):
    st = torch.load(path, map_location=device, weights_only=False)
    tmpl = st["model"].get("tmpl")
    m = SMPLFitter(st["n_points"], st.get("hidden", 1024), tmpl if tmpl is not None and len(tmpl) else None,
                   st.get("stages", 0)).to(device).eval()
    m.load_state_dict(st["model"], strict=False)
    m.idx = st.get("idx")  # the template vertices of its input points
    return m
