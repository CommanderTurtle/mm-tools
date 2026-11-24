"""Dense prediction: from the spatial queries' task-head outputs to full maps.

Each query predicts a window of 2 x 2 cells centred on its own 32 x 32-pixel cell; overlapping windows are blended.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import config

# Colours of the pictures made from the predictions, as in the paper's figures.
PALETTE = np.array([[230, 159, 0], [204, 121, 167], [240, 228, 186], [213, 94, 0], [166, 118, 82], [86, 180, 233], [0, 158, 115],
                    [240, 200, 60], [180, 70, 110], [120, 90, 200], [40, 40, 40], [200, 205, 215], [110, 200, 230]],
                   dtype=np.uint8)                                     # one colour per class of config.NYU13
DEPTH_RAMP = np.array([[252, 253, 191], [252, 137, 97], [183, 55, 121], [81, 18, 124], [0, 0, 4]], dtype=np.float32)   # near to far


def assemble(o, gh, gw, p, c):
    """the queries' outputs o [n, gh*gw, w*w*c], w = 2p -> maps [n, c, gh*p, gw*p] by tent-weighted overlap-add"""
    n = o.shape[0]
    w = 2 * p
    t = 1.0 - (torch.arange(w, device=o.device, dtype=torch.float32) + 0.5 - p).abs() / p      # tent, (0, 1]
    win = (t[:, None] * t[None, :]).flatten()                                                   # [w*w]
    x = o.float().view(n, gh * gw, w, w, c).permute(0, 4, 2, 3, 1).reshape(n, c * w * w, gh * gw)
    wt = win.repeat(c).view(1, c * w * w, 1)
    kw = dict(output_size=(gh * p, gw * p), kernel_size=w, stride=p, padding=p // 2)
    num = F.fold(x * wt, **kw)
    den = F.fold(wt.expand(1, -1, gh * gw), **kw)
    return num / den


def resize_for_vision(image_u8):
    """uint8 [B, H, W, 3] -> uint8 [B, 576, 768, 3]: the picture as the vision tower reads it for dense prediction (bilinear)"""
    x = F.interpolate(image_u8.permute(0, 3, 1, 2).float(), size=config.DENSE_INPUT_HW, mode="bilinear", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def to_predictions(task, raw):
    """a task's raw map [B, C, H, W] (class logits / log-depth / unnormalised normal) -> the prediction:
    seg: class index [B, H, W] (config.NYU13); depth: meters [B, H, W]; normal: unit vectors [B, H, W, 3]"""
    if task == "seg":
        return raw.argmax(1)
    if task == "depth":
        return raw[:, 0].clamp(-5, 5).exp()
    return F.normalize(raw.permute(0, 2, 3, 1), dim=-1)


def seg_view(seg, picture=None):
    """class indices [H, W] -> uint8 [H, W, 3]; the picture (uint8 [H, W, 3]), when given, shows through at 15 %"""
    out = PALETTE[seg].astype(np.float32)
    if picture is not None:
        out = 0.85 * out + 0.15 * picture
    return out.astype(np.uint8)


def depth_view(depth):
    """depth [H, W] -> uint8 [H, W, 3]: nearer is brighter, over the picture's 2-98 % range"""
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    v = np.clip((depth - lo) / max(float(hi - lo), 1e-3), 0, 1) * (len(DEPTH_RAMP) - 1)
    i = np.minimum(v.astype(int), len(DEPTH_RAMP) - 2)
    f = (v - i)[..., None]
    return (DEPTH_RAMP[i] * (1 - f) + DEPTH_RAMP[i + 1] * f).astype(np.uint8)


def normal_view(normal):
    """unit normals [H, W, 3] -> uint8 [H, W, 3]: (n + 1) / 2 as RGB"""
    return ((np.clip(normal, -1, 1) + 1) * 127.5).astype(np.uint8)
