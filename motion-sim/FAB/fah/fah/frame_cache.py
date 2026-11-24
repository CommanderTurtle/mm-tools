"""The current frame uploaded to the device once and shared by the detector letterbox and the pose crops."""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch

_FRAME = ContextVar("fah_frame_cache", default=None)


@contextmanager
def frame_cache():
    """Share one upload within an inference call, including nested detector/crop passes.

    The image must stay unchanged inside this scope. A new outer call always starts
    fresh, even when a camera fills the same numpy buffer for every frame.
    """
    if _FRAME.get() is not None:
        yield
        return
    token = _FRAME.set({})
    try:
        yield
    finally:
        _FRAME.reset(token)


def frame_tensor(img, device):
    """img (H, W, 3) uint8 numpy -> (1, 3, H, W) float32 on device (0..255, BGR).

    Cache only inside frame_cache(); standalone calls always upload fresh pixels.
    Retain the array itself while cached so recycled addresses cannot match it.
    """
    cache = _FRAME.get()
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:  # "cuda" and "cuda:0" are one device: one upload, not two
        device = torch.device("cuda", torch.cuda.current_device())
    key = (img.shape, img.strides, str(device))
    if cache is not None and cache.get("image") is img and cache.get("key") == key:
        return cache["tensor"]
    t = torch.from_numpy(np.ascontiguousarray(img)).to(device, non_blocking=True)
    t = t.permute(2, 0, 1)[None].float()
    if cache is not None:
        cache.update(image=img, key=key, tensor=t)
    return t
