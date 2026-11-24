"""Parametric MANO head for the decoder-free dense keypoint model: a learned IK step from the model's own 2D points
(21 joints + 40 bone points, crop-normalised, centred on the wrist), the image features pooled under their
heatmaps and the pooled hand token to right-hand MANO parameters (global orientation as 6D, 15 joint rotations as
axis-angle residuals to MANO's mean pose, 10 shape coefficients). The mesh is always a valid hand; the dense
surface points stay a separate output. The MANO model is loaded from `mano_dir` at run time and is not a
parameter of this module (not saved in checkpoints)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MANO21_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MANO_TIP_VERTS = [744, 320, 443, 554, 671]  # HOT3D fingertip vertices (thumb, index, middle, ring, pinky)


_PARENTS = {}  # MANO kinematic tree as a Python list (indexing a list with a CUDA tensor syncs, breaking CUDA graphs)


def _rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """Forward kinematics along the joint tree, used in place of smplx's batch_rigid_transform (same arguments and
    outputs): a joint's world transform is its parent's times its local transform [R | offset from the parent], and its
    skinning transform is the world transform with the rest-pose joint location taken out. The tree is walked with the
    parent indices from a cached Python list (no tensor -> int syncs: capturable in a CUDA graph)."""
    par = _PARENTS.get(parents.shape[0])
    if par is None:
        par = _PARENTS[parents.shape[0]] = parents.tolist()
    n = joints.shape[1]
    j = joints[..., None]  # (B, n, 3, 1)
    off = torch.cat([j[:, :1], j[:, 1:] - j[:, parents[1:]]], 1)
    rot = rot_mats.reshape(-1, n, 3, 3)
    last = torch.zeros_like(off[..., :1, :]).expand(-1, -1, -1, 4).clone()
    last[..., 3] = 1
    local = torch.cat([torch.cat([rot, off], -1), last], -2)  # (B, n, 4, 4)
    world = [local[:, 0]]
    for k in range(1, len(par)):
        world.append(torch.matmul(world[par[k]], local[:, k]))
    world = torch.stack(world, 1)
    rest = torch.matmul(world, torch.cat([j, torch.zeros_like(j[..., :1, :])], -2))  # world rotation applied to the rest joint
    skin = torch.cat([world[..., :3], world[..., 3:] - rest], -1)
    return world[..., :3, 3], skin


def rot6d(x):
    a, b = x[..., :3], x[..., 3:]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)  # columns


class ManoParamHead(nn.Module):
    def __init__(self, tok_dim=1024, pt_dim=64, n_pts=61, hidden=1024, mano_dir="data/mano"):
        super().__init__()
        self.n_pts, self.mano_dir = n_pts, mano_dir
        self.feat = nn.Sequential(nn.LayerNorm(pt_dim), nn.Linear(pt_dim, 32), nn.GELU())
        d = tok_dim + n_pts * (2 + 32)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 6 + 45 + 10))
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            self.mlp[-1].bias[:6] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
        self._mano = []  # smplx MANO (right), kept out of the state dict
        self._idx = {}

    def mano(self, device):
        if not self._mano:
            import smplx
            import smplx.lbs

            smplx.lbs.batch_rigid_transform = _rigid_transform  # capturable in a CUDA graph (no tensor -> int syncs)
            self._mano.append(smplx.create(f"{self.mano_dir}/MANO_RIGHT.pkl", "mano", is_rhand=True, use_pca=False,
                                           flat_hand_mean=False).to(device))
            _PARENTS[self._mano[0].parents.shape[0]] = self._mano[0].parents.tolist()
        if self._mano[0].shapedirs.device != device:
            self._mano[0] = self._mano[0].to(device)
        return self._mano[0]

    def forward(self, tok, uv, pooled):
        """tok (B, tok_dim); uv (B, >= n_pts, 2) crop-normalised 2D points; pooled (B, >= n_pts, pt_dim).
        Returns root-relative vertices (B, 778, 3), joints (B, 21, 3, OpenPose order), parameters (B, 61)."""
        B = tok.shape[0]
        u = (uv[:, :self.n_pts] - uv[:, :1]).float() * 4.0
        f = self.feat(pooled[:, :self.n_pts].float())
        p = self.mlp(torch.cat([tok.float(), u.flatten(1), f.flatten(1)], -1))
        R = rot6d(p[:, :6])
        m = self.mano(tok.device)
        out = m(global_orient=torch.zeros(B, 3, device=tok.device), hand_pose=p[:, 6:51], betas=p[:, 51:61])
        v = out.vertices
        if tok.device not in self._idx:  # index tensors on the device (CPU index lists copy during CUDA graph capture)
            self._idx[tok.device] = (torch.tensor(MANO_TIP_VERTS, device=tok.device), torch.tensor(MANO21_TO_OPENPOSE, device=tok.device))
        tips, order = self._idx[tok.device]
        j = torch.cat([out.joints[:, :16], v[:, tips]], 1)[:, order]
        v, j = v - j[:, :1], j - j[:, :1]
        return v @ R.mT, j @ R.mT, p
