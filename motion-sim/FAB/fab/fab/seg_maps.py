"""Part / person-mask maps of FAB-L / FAB-S crops pasted into the image (FABPredictor's `parts` / `mask` outputs): the crop maps
at res x res on the GPU (part argmax of the upsampled logits, mask logits > 0), one device-to-host copy for all of them,
then each person's map warped (nearest) into only its crop's region of the image, zeros elsewhere."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def stack_maps(maps, dim):
    """Stack (B, r, r) uint8 crop maps of possibly different r (a full-res mask with 256 part maps): the coarser ones
    repeated (nearest) up to the finest."""
    r = max(m.shape[-1] for m in maps)
    return torch.stack([m if m.shape[-1] == r else m.repeat_interleave(r // m.shape[-2], -2).repeat_interleave(
        r // m.shape[-1], -1) for m in maps], dim)


def crop_maps_gpu(part_logits=None, mask_logits=None, res=256):
    """The crop maps on the device (graph-capturable: no host sync): (part ids (B, res, res) uint8 or None, mask
    (B, res, res) uint8 or None; a finer mask decoder ("+full") keeps its own resolution)."""
    pm = mm = None
    if part_logits is not None:
        pm = F.interpolate(part_logits.float(), size=(res, res), mode="bilinear", align_corners=False).argmax(1).to(torch.uint8)
    if mask_logits is not None:
        ml = mask_logits.float()
        if ml.shape[-1] < res:
            ml = F.interpolate(ml, size=(res, res), mode="bilinear", align_corners=False)
        mm = (ml[:, 0] > 0).to(torch.uint8)
    return pm, mm


@torch.inference_mode()
def crop_maps(part_logits=None, mask_logits=None, res=256):
    """part_logits (B, P, h, w) / mask_logits (B, 1, h', w') or None -> (parts (B, res, res) uint8 or None,
    mask (B, res, res) uint8 or None), on the CPU."""
    maps = []
    if part_logits is not None:
        maps.append(F.interpolate(part_logits.float(), size=(res, res), mode="bilinear", align_corners=False).argmax(1))
    if mask_logits is not None:
        ml = mask_logits.float()
        if ml.shape[-1] < res:
            ml = F.interpolate(ml, size=(res, res), mode="bilinear", align_corners=False)
        maps.append(ml[:, 0] > 0)
    if not maps:
        return None, None
    cpu = stack_maps([m.to(torch.uint8) for m in maps], 1).cpu().numpy()
    parts = cpu[:, 0] if part_logits is not None else None
    mask = cpu[:, -1] if mask_logits is not None else None
    return parts, mask


def paste_roi(m, A, S, H, W):
    """m (res, res) or (res, res, C) uint8 crop maps, A (2, 3) image -> crop (S px) affine -> (the maps warped (nearest)
    into the crop's region of the image, (x0, y0, x1, y1) of that region), without a full-image array; several maps as
    channels cost one warp."""
    res = m.shape[0]
    k = S / res  # map cell -> crop pixels (cell centres at k * i + (k - 1) / 2)
    G = np.array([[k, 0, (k - 1) / 2], [0, k, (k - 1) / 2], [0, 0, 1.0]])
    Ai = np.linalg.inv(np.vstack([np.asarray(A, np.float64), [0, 0, 1]])) @ G  # map cell -> image pixel
    c = Ai @ np.array([[-0.5, -0.5, 1], [res - 0.5, -0.5, 1], [-0.5, res - 0.5, 1], [res - 0.5, res - 0.5, 1]]).T
    x0, y0 = max(int(np.floor(c[0].min())), 0), max(int(np.floor(c[1].min())), 0)
    x1, y1 = min(int(np.ceil(c[0].max())) + 1, W), min(int(np.ceil(c[1].max())) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0) + m.shape[2:], np.uint8), (0, 0, 0, 0)
    T = Ai[:2].copy()
    T[:, 2] -= (x0, y0)
    roi = cv2.warpAffine(np.ascontiguousarray(m), T, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST, borderValue=0)
    return roi.reshape((y1 - y0, x1 - x0) + m.shape[2:]), (x0, y0, x1, y1)


def paste_roi_gpu(m, A, S, H, W):
    """paste_roi on the device: m (res, res, C) uint8 tensor -> (roi (h, w, C) uint8 tensor, (x0, y0, x1, y1)); nearest
    cell per image pixel by direct indexing (no CPU warp: cv2's thread pool wakes up slowly inside a GPU pipeline)."""
    res = m.shape[0]
    k = S / res
    G = np.array([[k, 0, (k - 1) / 2], [0, k, (k - 1) / 2], [0, 0, 1.0]])
    A3 = np.vstack([np.asarray(A, np.float64), [0, 0, 1]])
    Ai = np.linalg.inv(A3) @ G
    c = Ai @ np.array([[-0.5, -0.5, 1], [res - 0.5, -0.5, 1], [-0.5, res - 0.5, 1], [res - 0.5, res - 0.5, 1]]).T
    x0, y0 = max(int(np.floor(c[0].min())), 0), max(int(np.floor(c[1].min())), 0)
    x1, y1 = min(int(np.ceil(c[0].max())) + 1, W), min(int(np.ceil(c[1].max())) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return m.new_zeros((0, 0) + tuple(m.shape[2:])), (0, 0, 0, 0)
    M = np.linalg.inv(G) @ A3  # image pixel -> map cell
    xs = torch.arange(x0, x1, device=m.device, dtype=torch.float64)
    ys = torch.arange(y0, y1, device=m.device, dtype=torch.float64)
    u = torch.floor(M[0, 0] * xs[None] + M[0, 1] * ys[:, None] + M[0, 2] + 0.5).long()
    v = torch.floor(M[1, 0] * xs[None] + M[1, 1] * ys[:, None] + M[1, 2] + 0.5).long()
    ok = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    roi = m[v.clamp(0, res - 1), u.clamp(0, res - 1)]
    roi = roi * ok.view(ok.shape + (1,) * (roi.dim() - 2)).to(roi.dtype)
    return roi, (x0, y0, x1, y1)


def full_map(roi, box, H, W):
    """A paste_roi map as the (H, W) image map (0 outside the region)."""
    out = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = box
    if x1 > x0:
        out[y0:y1, x0:x1] = roi
    return out


def paste(m, A, S, H, W):
    """m (res, res) uint8 crop map, A (2, 3) image -> crop (S px) affine -> (H, W) uint8: the map warped into the crop's
    region of the image (nearest), 0 elsewhere."""
    res = m.shape[0]
    k = S / res  # map cell -> crop pixels (cell centres at k * i + (k - 1) / 2)
    G = np.array([[k, 0, (k - 1) / 2], [0, k, (k - 1) / 2], [0, 0, 1.0]])
    Ai = np.linalg.inv(np.vstack([np.asarray(A, np.float64), [0, 0, 1]])) @ G  # map cell -> image pixel
    c = Ai @ np.array([[-0.5, -0.5, 1], [res - 0.5, -0.5, 1], [-0.5, res - 0.5, 1], [res - 0.5, res - 0.5, 1]]).T
    x0, y0 = max(int(np.floor(c[0].min())), 0), max(int(np.floor(c[1].min())), 0)
    x1, y1 = min(int(np.ceil(c[0].max())) + 1, W), min(int(np.ceil(c[1].max())) + 1, H)
    out = np.zeros((H, W), np.uint8)
    if x1 > x0 and y1 > y0:
        T = Ai[:2].copy()
        T[:, 2] -= (x0, y0)
        out[y0:y1, x0:x1] = cv2.warpAffine(m, T, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST, borderValue=0)
    return out
