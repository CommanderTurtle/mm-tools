"""Image input and output helpers."""
import numpy as np
import torch
from PIL import Image


def load_square(path_or_image, size=512):
    """an image file or PIL image -> uint8 [size, size, 3]: RGB, shorter side to `size` (bicubic), centre crop"""
    im = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    im = im.convert("RGB")
    w, h = im.size
    if (w, h) != (size, size):
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        l, t = (im.size[0] - size) // 2, (im.size[1] - size) // 2
        im = im.crop((l, t, l + size, t + size))
    return np.asarray(im, dtype=np.uint8)


def to_gray(img):
    """uint8 [H, W, 3] -> uint8 [H, W, 3] with three equal channels: luma (ITU-R BT.601), the colorization task's input"""
    y = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).round().clip(0, 255).astype(np.uint8)
    return np.repeat(y[..., None], 3, axis=2)


def to_uint8(images):
    """float images [B, 3, H, W] -> uint8 numpy [B, H, W, 3] (clamped to [0, 1])"""
    return (images.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
