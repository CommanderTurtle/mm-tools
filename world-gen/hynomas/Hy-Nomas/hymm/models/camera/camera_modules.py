"""
Camera control helper modules and utility functions.

Consolidates shared components from the three camera injection modes:
- action_attn: ActionModule cross-attention on raw action_onehot
- extrinsics_attn: ActionModule cross-attention on extrinsics embedding
- residual_norm: Plücker ray embedding added to img tokens
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version as pver
from einops import rearrange


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3]  # [B,3,3]
    trans = T[:, :3, 3:]  # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(
        T.shape[0], 1, 1
    )
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = True,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(
            SE3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""
    # c2w: B, V, 4, 4
    # K: B, V, 4
    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    return plucker


def rotmat_to_6d(R):
    """Convert rotation matrix to 6D representation. R: (..., 3, 3)"""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


# ---------------------------------------------------------------------------
# Camera embedding modules
# ---------------------------------------------------------------------------


class CameraEmbed(nn.Module):
    def __init__(self, cam_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(cam_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.SiLU()
        # zero init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cam_vec):
        x = self.act(self.fc1(cam_vec))
        return self.fc2(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class SimpleAdapter(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        downscale_factor=8,
        num_residual_blocks=1,
    ):
        super(SimpleAdapter, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_dim * downscale_factor * downscale_factor,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        x_unshuffled = self.pixel_unshuffle(x)
        x_conv = self.conv(x_unshuffled)
        out = self.residual_blocks(x_conv)
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))
        out = out.permute(0, 2, 1, 3, 4)
        return out
