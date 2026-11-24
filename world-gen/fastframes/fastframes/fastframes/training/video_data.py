#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Real videos for the adversarial pixel-decoder phase: a folder of video files.

Every ``.mp4``/``.mov``/``.m4v`` file below ``root`` is a candidate (H.264 video,
decoded with OpenH264); rank ``r`` of ``W`` owns files ``r, r + W, ...`` of the
sorted list (disjoint shards). Clip ``i`` of a rank is file ``perm_epoch[i mod n]``
of its shard (reshuffled every epoch): ``num_frames`` consecutive frames from a
random start, center-cropped to the output aspect ratio, resized (bilinear,
antialiased) and returned as uint8 ``[1, T, 3, H, W]`` on the CPU. A clip is a
pure function of ``(seed, rank, i)``, so resuming needs no loader state. Files
with fewer than ``num_frames`` frames (or that fail to decode, e.g. another
codec) are skipped in favour of the next file of the order.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from fastframes.video import count_frames, read_video

__all__ = ["VideoFolder", "cover_crop_resize", "to_discriminator_range"]

EXTENSIONS = (".mp4", ".mov", ".m4v")  # H.264 in an MP4/QuickTime container


def cover_crop_resize(frames, height, width, antialias=True):
    """``[..., H, W]`` float -> ``[..., height, width]``: centered crop to the target aspect, then resize."""
    src_h, src_w = int(frames.shape[-2]), int(frames.shape[-1])
    if src_w / src_h > width / height:
        crop_h, crop_w = src_h, int(src_h * (width / height))
        top, left = 0, (src_w - crop_w) // 2
    else:
        crop_w, crop_h = src_w, int(src_w / (width / height))
        top, left = (src_h - crop_h) // 2, 0
    return TF.resized_crop(frames, top=top, left=left, height=crop_h, width=crop_w, size=[height, width],
                           interpolation=InterpolationMode.BILINEAR, antialias=antialias)


def to_discriminator_range(clips):
    """uint8 crops -> FP32 in ``[-1, 1]``."""
    return clips.float().div(127.5).sub(1.0)


def _decode(path, start, count):
    """``count`` RGB frames ``[T, H, W, 3]`` (uint8) starting at frame ``start``."""
    return torch.from_numpy(read_video(path, start, count))


class VideoFolder:
    """Stateless, rank-sharded clip source (see the module docstring)."""

    def __init__(self, root, rank=0, world_size=1, *, num_frames=81, height=480, width=832, seed=0,
                 antialias=True, max_skips=64):
        files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
        self.files = files[rank::world_size]
        if not self.files:
            raise ValueError(f"no videos for rank {rank} of {world_size} under {root}")
        self.rank, self.num_frames, self.height, self.width = rank, num_frames, height, width
        self.seed, self.antialias, self.max_skips = int(seed), antialias, max_skips

    def _order(self, epoch):
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return torch.randperm(len(self.files), generator=generator)

    def clip(self, index):
        """Clip ``index`` of this rank: uint8 ``[1, num_frames, 3, height, width]``."""
        n = len(self.files)
        errors = []
        for attempt in range(min(self.max_skips, n)):
            position = index + attempt
            path = self.files[self._order(position // n)[position % n]]
            try:
                total = count_frames(path)
                if total < self.num_frames:
                    raise ValueError(f"{path}: {total} < {self.num_frames} frames")
                generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.rank * 7919 + index)
                start = int(torch.randint(0, total - self.num_frames + 1, (), generator=generator))
                frames = _decode(path, start, self.num_frames)
            except Exception as exc:  # noqa: BLE001 - skipped deterministically
                errors.append(repr(exc))
                continue
            video = frames.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W] in [0, 1]
            video = cover_crop_resize(video, self.height, self.width, self.antialias)
            video = video.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        raise RuntimeError(f"no decodable clip for index {index} on rank {self.rank}: {errors[-3:]}")
