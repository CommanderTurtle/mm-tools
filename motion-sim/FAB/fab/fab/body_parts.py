"""Body part segmentation labels from a mesh (rendered on the GPU in the trainer so every augmentation stays
consistent): SMPL vertices get the part of their dominant skinning joint,
faces the majority part of their vertices, and the camera-frame mesh is z-buffered into the crop at the head's
heatmap resolution. Rendering = dense barycentric samples per face, nearest sample per pixel (scatter-min on depth),
which is exact enough at 128 x 128 for a 6890-vertex body."""

import torch

PART_NAMES = ["head", "torso", "l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm", "l_hand", "r_hand", "l_thigh",
              "r_thigh", "l_calf", "r_calf", "l_foot", "r_foot"]  # ids 1..14, 0 = background
# SMPL joint (segment starting at that joint) -> part id
SMPL_JOINT_TO_PART = [2, 9, 10, 2, 11, 12, 2, 13, 14, 2, 13, 14, 1, 2, 2, 1, 3, 4, 5, 6, 7, 8, 7, 8]
# part id permutation of a mirrored body (left <-> right)
PART_FLIP = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
N_PARTS = len(PART_NAMES) + 1


def face_parts(weights, faces):
    """weights (V, 24) SMPL skinning weights, faces (F, 3) -> part id per face (F,)."""
    vpart = torch.as_tensor(SMPL_JOINT_TO_PART, device=weights.device)[weights.argmax(1)]
    fp = vpart[faces.long()]  # (F, 3): majority, ties -> the first vertex
    maj = torch.where(fp[:, 1] == fp[:, 2], fp[:, 1], fp[:, 0])
    return maj


def barycentric_grid(n):
    """Barycentric sample weights (S, 3) on a regular grid of a triangle (n per edge)."""
    w = [(i / n, j / n, 1 - (i + j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return torch.tensor(w, dtype=torch.float32)


def adaptive_bary(uv, faces, hw, lo=3, hi=12):
    """Barycentric samples dense enough for this batch (close-ups have triangles of many pixels: a fixed 3-per-edge
    grid leaves holes): n per edge = the 95th percentile of the projected edge lengths in pixels, clipped to [lo, hi]."""
    t = uv[:, faces] * torch.tensor([hw[1], hw[0]], device=uv.device, dtype=uv.dtype)  # (B, F, 3, 2) pixels
    e = (t - t.roll(1, dims=2)).norm(dim=-1).flatten()
    e = e[torch.isfinite(e)]
    n = int(min(max(torch.quantile(e[:: max(len(e) // 100000, 1)].float(), 0.95).ceil().item(), lo), hi)) if len(e) else lo
    return barycentric_grid(n).to(uv.device)


def render_depth(uv, z, faces, hw, bary):
    """Z-buffer of a mesh as render_parts: uv (B, V, 2) crop units, z (B, V), faces (F, 3) -> nearest depth per pixel
    (B, H, W) (inf where the mesh does not cover the pixel)."""
    B, H, W = uv.shape[0], hw[0], hw[1]
    su = torch.einsum("sk,bfkc->bfsc", bary, uv[:, faces]).reshape(B, -1, 2)
    sz = torch.einsum("sk,bfk->bfs", bary, z[:, faces]).reshape(B, -1)
    x = (su[..., 0] * W).floor().long()
    y = (su[..., 1] * H).floor().long()
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (sz > 0.05)
    pix = torch.where(ok, y * W + x, torch.full_like(x, H * W))
    flat = pix + torch.arange(B, device=uv.device)[:, None] * (H * W + 1)
    zmin = torch.full((B * (H * W + 1),), float("inf"), device=uv.device)
    zmin.scatter_reduce_(0, flat.reshape(-1), sz.reshape(-1).float(), reduce="amin")
    return zmin.view(B, H * W + 1)[:, : H * W].view(B, H, W)


def render_parts(uv, z, faces, fpart, hw, bary, flip_parts=None):
    """uv (B, V, 2) mesh vertices in crop units [0, 1], z (B, V) depth, faces (F, 3), fpart (F,), hw (H, W),
    bary (S, 3) -> part ids (B, H, W) (0 background). flip_parts (B,) bool: swap left / right ids (mirrored bodies)."""
    B, H, W = uv.shape[0], hw[0], hw[1]
    tri_uv = uv[:, faces]  # (B, F, 3, 2)
    tri_z = z[:, faces]
    su = torch.einsum("sk,bfkc->bfsc", bary, tri_uv).reshape(B, -1, 2)
    sz = torch.einsum("sk,bfk->bfs", bary, tri_z).reshape(B, -1)
    sp = fpart[:, None].expand(-1, bary.shape[0]).reshape(-1)
    x = (su[..., 0] * W).floor().long()
    y = (su[..., 1] * H).floor().long()
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (sz > 0.05)
    pix = torch.where(ok, y * W + x, torch.full_like(x, H * W))  # off-crop samples -> a dump bin
    flat = pix + torch.arange(B, device=uv.device)[:, None] * (H * W + 1)
    zmin = torch.full((B * (H * W + 1),), float("inf"), device=uv.device)
    zmin.scatter_reduce_(0, flat.reshape(-1), sz.reshape(-1), reduce="amin")
    win = (sz.reshape(-1) <= zmin[flat.reshape(-1)]) & ok.reshape(-1)
    lab = torch.zeros(B * (H * W + 1), dtype=torch.long, device=uv.device)
    lab[flat.reshape(-1)[win]] = sp.repeat(B)[win].long()
    lab = lab.view(B, H * W + 1)[:, : H * W].view(B, H, W)
    if flip_parts is not None:
        perm = torch.as_tensor(PART_FLIP, device=uv.device)
        lab = torch.where(flip_parts[:, None, None], perm[lab], lab)
    return lab
