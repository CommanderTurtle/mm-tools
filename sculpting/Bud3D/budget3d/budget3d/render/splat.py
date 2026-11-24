"""A small, exact 3D Gaussian Splatting rasterizer in plain PyTorch (no CUDA extension).

Same image formation as the reference 3DGS rasterizer: EWA projection of each Gaussian to a
2D conic (+0.3 px^2 low-pass), alpha = min(0.99, o * exp(-q/2)) dropped below 1/255, and
front-to-back compositing in depth order.  Gaussians are binned into 16 px tiles; each tile's
depth-sorted list is composited in chunks with a cumulative product *inside* the chunk, so the
over operator is applied in exact order (chunking only changes the batching, not the result).
"""
import math

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


def look_at(campos, target, up):
    """World->camera rotation, rows = camera x (right), y (down), z (forward): OpenCV."""
    f = F.normalize(target - campos, dim=0)
    r = F.normalize(torch.linalg.cross(f, up), dim=0)
    d = torch.linalg.cross(f, r)
    return torch.stack([r, d, f])


def quat_to_rot(q):
    """(w, x, y, z) unit quaternions [N, 4] -> rotation matrices [N, 3, 3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


class Splats:
    """Activated Gaussians kept on the GPU with their view-independent parts precomputed."""

    def __init__(self, gaussian):
        keep = gaussian.opacity[:, 0] > 1.0 / 255
        self.xyz = gaussian.xyz[keep].float()
        self.rgb = (0.5 + SH_C0 * gaussian.features_dc[keep, 0].float()).clamp(0, 1)
        self.opacity = gaussian.opacity[keep, 0].float()
        m = quat_to_rot(gaussian.rotation[keep].float()) * gaussian.scaling[keep].float()[:, None, :]
        self.cov3 = m @ m.transpose(1, 2)
        self.n = self.xyz.shape[0]


@torch.no_grad()
def render(splats, campos, fov_deg, size=256, bg=1.0, up=(0.0, 0.0, 1.0), tile=16, chunk=96):
    """Render [size, size, 3] in [0, 1] from ``campos`` looking at the origin (z-up world)."""
    dev = splats.xyz.device
    H = W = size
    campos = torch.as_tensor(campos, dtype=torch.float32, device=dev)
    R = look_at(campos, torch.zeros(3, device=dev), torch.tensor(up, device=dev))
    p = (splats.xyz - campos) @ R.T
    z = p[:, 2]
    f = 0.5 * H / math.tan(math.radians(fov_deg) / 2)

    vis = z > 0.02
    p, z = p[vis], z[vis]
    idx = vis.nonzero()[:, 0]
    u = f * p[:, 0] / z + 0.5 * W
    v = f * p[:, 1] / z + 0.5 * H

    covc = R @ splats.cov3[idx] @ R.T
    J = torch.zeros(len(z), 2, 3, device=dev)
    J[:, 0, 0] = f / z
    J[:, 0, 2] = -f * p[:, 0] / z ** 2
    J[:, 1, 1] = f / z
    J[:, 1, 2] = -f * p[:, 1] / z ** 2
    cov2 = J @ covc @ J.transpose(1, 2)
    a, b, c = cov2[:, 0, 0] + 0.3, cov2[:, 0, 1], cov2[:, 1, 1] + 0.3
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mid = 0.5 * (a + c)
    lam = mid + (mid * mid - det).clamp_min(0.1).sqrt()
    rad = (3.0 * lam.sqrt()).ceil().clamp(max=max(H, W))

    on = (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H)
    u, v, z, rad, conic, idx = u[on], v[on], z[on], rad[on], conic[on], idx[on]
    n = len(z)
    TW, TH = W // tile, H // tile
    bg_img = torch.full((H, W, 3), bg, device=dev)
    if n == 0:
        return bg_img

    # --- bin every Gaussian into the tiles its 3-sigma box touches, sort by (tile, depth)
    tx0 = ((u - rad) / tile).floor().clamp(0, TW - 1).long()
    tx1 = ((u + rad) / tile).floor().clamp(0, TW - 1).long()
    ty0 = ((v - rad) / tile).floor().clamp(0, TH - 1).long()
    ty1 = ((v + rad) / tile).floor().clamp(0, TH - 1).long()
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    cnt = nx * ny
    gid = torch.repeat_interleave(torch.arange(n, device=dev), cnt)
    local = torch.arange(len(gid), device=dev) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    tid = (ty0[gid] + local // nx[gid]) * TW + (tx0[gid] + local % nx[gid])
    rank = torch.empty(n, dtype=torch.long, device=dev)
    rank[z.argsort()] = torch.arange(n, device=dev)
    order = (tid * n + rank[gid]).argsort()
    tid, gid = tid[order], gid[order]
    T = TW * TH
    counts = torch.bincount(tid, minlength=T)
    pos = torch.arange(len(tid), device=dev) - (counts.cumsum(0) - counts)[tid]
    K = int(counts.max())
    table = torch.full((T, K), n, dtype=torch.long, device=dev)   # n = transparent padding
    table[tid, pos] = gid

    mean = torch.cat([torch.stack([u, v], -1), torch.zeros(1, 2, device=dev)])
    con = torch.cat([conic, torch.ones(1, 3, device=dev)])
    op = torch.cat([splats.opacity[idx], torch.zeros(1, device=dev)])
    col = torch.cat([splats.rgb[idx], torch.zeros(1, 3, device=dev)])

    ys, xs = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev), indexing="ij")
    ty, tx = torch.div(torch.arange(T, device=dev), TW, rounding_mode="floor"), torch.arange(T, device=dev) % TW
    px = (tx[:, None] * tile + xs.reshape(1, -1)).float() + 0.5          # [T, P] pixel centres
    py = (ty[:, None] * tile + ys.reshape(1, -1)).float() + 0.5

    # only tiles that still have Gaussians and light left are processed in each chunk
    C = torch.zeros(T, tile * tile, 3, device=dev)
    Tacc = torch.ones(T, tile * tile, device=dev)
    active = torch.arange(T, device=dev)
    for k0 in range(0, K, chunk):
        active = active[(counts[active] > k0) & (Tacc[active].amax(1) > 1e-4)]
        if len(active) == 0:
            break
        g = table[active, k0:k0 + chunk]                                   # [A, Kc]
        dx = px[active][:, None, :] - mean[g][..., 0:1]                     # [A, Kc, P]
        dy = py[active][:, None, :] - mean[g][..., 1:2]
        cg = con[g]
        power = -0.5 * (cg[..., 0:1] * dx * dx + cg[..., 2:3] * dy * dy) - cg[..., 1:2] * dx * dy
        alpha = (op[g][..., None] * power.exp()).clamp(max=0.99)
        alpha = torch.where((power <= 0) & (alpha >= 1.0 / 255), alpha, torch.zeros_like(alpha))
        trans = torch.cumprod(1 - alpha, dim=1)                             # inclusive, in depth order
        excl = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], 1)
        w = alpha * excl * Tacc[active][:, None, :]
        C[active] += torch.einsum("akp,akc->apc", w, col[g])
        Tacc[active] = Tacc[active] * trans[:, -1]

    img = C + Tacc[..., None] * bg
    img = img.view(TH, TW, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(H, W, 3)
    return img.clamp(0, 1)


def orbit_positions(yaw, pitch, radius, n_frames):
    """Camera centres for one full turn about the z axis that starts and ends at (yaw, pitch).

    Idol's camera decode gives the unit direction (dx, dy, dz) from the object to the camera
    with yaw = atan2(dx, dy) and pitch = asin(dz)."""
    out = []
    for i in range(n_frames):
        y = yaw + 2 * math.pi * i / n_frames
        out.append((radius * math.cos(pitch) * math.sin(y), radius * math.cos(pitch) * math.cos(y),
                    radius * math.sin(pitch)))
    return out
