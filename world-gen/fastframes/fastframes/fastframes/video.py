#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""H.264 MP4 writing and reading with OpenH264 (see ``fastframes.openh264``) and ``fastframes.mp4``.

Files are written with ``moov`` before ``mdat`` so browsers can stream them, in BT.601 limited range (labeled
as such). Reading follows the matrix and range signaled in the stream and assumes BT.601 limited range when they
are absent.
"""
from __future__ import annotations

import numpy as np

from . import mp4, openh264

__all__ = ["DEFAULT_QP", "save_video", "read_video", "count_frames", "rgb_to_i420", "i420_to_rgb", "sps_color"]

DEFAULT_QP = 20
_KR, _KB = 0.299, 0.114
_KG = 1 - _KR - _KB
# (Kr, Kb) of the H.264 matrix_coefficients values; anything else is read as BT.601.
_MATRICES = {1: (0.2126, 0.0722), 4: (0.30, 0.11), 5: (_KR, _KB), 6: (_KR, _KB), 7: (0.212, 0.087),
             9: (0.2627, 0.0593), 10: (0.2627, 0.0593)}


def rgb_to_i420(frame):
    """``[H, W, 3]`` uint8 RGB (even ``H`` and ``W``) -> uint8 planes ``y [H, W]``, ``u``, ``v [H/2, W/2]``.

    Chroma is the mean of each 2x2 block."""
    rgb = np.asarray(frame)
    r, g, b = (rgb[..., k].astype(np.float32) for k in range(3))
    y = (_KR * 219 / 255) * r + (_KG * 219 / 255) * g + (_KB * 219 / 255) * b + 16.5
    r, g, b = ((c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2] + c[1::2, 1::2]) * 0.25 for c in (r, g, b))
    luma = _KR * r + _KG * g + _KB * b
    u = (b - luma) * (224 / 255 / (2 - 2 * _KB)) + 128.5
    v = (r - luma) * (224 / 255 / (2 - 2 * _KR)) + 128.5
    return tuple(np.clip(plane, 0, 255).astype(np.uint8) for plane in (y, u, v))


def i420_to_rgb(y, u, v, *, matrix=6, full_range=False):
    """uint8 I420 planes -> ``[H, W, 3]`` uint8 RGB; each chroma sample covers its 2x2 block.

    ``matrix`` is the H.264 ``matrix_coefficients`` value (6: BT.601, 1: BT.709, 9: BT.2020)."""
    kr, kb = _MATRICES.get(matrix, (_KR, _KB))
    kg = 1 - kr - kb
    h, w = y.shape
    if full_range:
        luma, scale = y.astype(np.float32) + 0.5, 1.0
    else:
        luma, scale = (y.astype(np.float32) - 16) * (255 / 219) + 0.5, 255 / 224
    cb = (u.astype(np.float32) - 128) * scale
    cr = (v.astype(np.float32) - 128) * scale
    offsets = ((2 - 2 * kr) * cr, -(kb * (2 - 2 * kb) / kg) * cb - (kr * (2 - 2 * kr) / kg) * cr, (2 - 2 * kb) * cb)
    rgb = np.empty((h, w, 3), np.uint8)
    for k, offset in enumerate(offsets):
        channel = luma + np.repeat(np.repeat(offset, 2, axis=0), 2, axis=1)[:h, :w]
        rgb[..., k] = np.clip(channel, 0, 255)
    return rgb


def sps_color(sps):
    """``(matrix_coefficients, full_range)`` signaled in an H.264 SPS NAL unit; ``(2, False)`` if absent."""
    data, zeros = bytearray(), 0
    for byte in sps[1:]:  # drop emulation prevention bytes
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        data.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    bits, total, position = int.from_bytes(data, "big"), 8 * len(data), 0

    def u(n):
        nonlocal position
        position += n
        if position > total:
            raise ValueError("truncated SPS")
        return (bits >> (total - position)) & ((1 << n) - 1)

    def ue():
        n = 0
        while not u(1):
            n += 1
        return (1 << n) - 1 + u(n)

    def se():
        value = ue()
        return (value + 1) // 2 if value & 1 else -(value // 2)

    profile = u(8)
    u(16)  # constraint flags, level
    ue()  # seq_parameter_set_id
    if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma_format = ue()
        if chroma_format == 3:
            u(1)
        ue(), ue(), u(1)  # bit depths, transform bypass
        if u(1):  # scaling matrices
            for index in range(12 if chroma_format == 3 else 8):
                if u(1):
                    last = following = 8
                    for _ in range(16 if index < 6 else 64):
                        if following:
                            following = (last + se()) % 256
                        last = following or last
    ue()  # log2_max_frame_num_minus4
    poc_type = ue()
    if poc_type == 0:
        ue()
    elif poc_type == 1:
        u(1), se(), se()
        for _ in range(ue()):
            se()
    ue(), u(1), ue(), ue()  # reference frames, gaps, width, height
    if not u(1):  # frame_mbs_only_flag
        u(1)
    u(1)  # direct_8x8_inference_flag
    if u(1):  # cropping
        ue(), ue(), ue(), ue()
    if not u(1):  # vui_parameters_present_flag
        return 2, False
    if u(1) and u(8) == 255:  # aspect ratio
        u(32)
    if u(1):  # overscan
        u(1)
    if not u(1):  # video_signal_type_present_flag
        return 2, False
    u(3)  # video_format
    full_range = bool(u(1))
    matrix = 2
    if u(1):  # colour_description_present_flag
        u(8), u(8)
        matrix = u(8)
    return matrix, full_range


def save_video(path, frames, fps=16, *, qp=DEFAULT_QP):
    """Write ``[T, H, W, 3]`` uint8 RGB frames (numpy or CPU tensor) as a faststart H.264 MP4."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or not len(frames):
        raise ValueError(f"expected [T, H, W, 3] uint8 frames, got {frames.shape} {frames.dtype}")
    height, width = frames.shape[1:3]
    access_units = []
    with openh264.Encoder(width, height, fps, qp=qp) as encoder:
        for frame in frames:
            access_units.append(encoder.encode(*rgb_to_i420(frame))[0])
    sps = next(n for n in access_units[0] if (n[0] & 0x1F) == 7)
    pps = next(n for n in access_units[0] if (n[0] & 0x1F) == 8)
    return mp4.write_h264(path, access_units, width=width, height=height, fps=fps, sps=sps, pps=pps)


def count_frames(path):
    """Number of frames of the first video track."""
    return mp4.probe(path)["frames"]


def read_video(path, start=0, count=None):
    """RGB frames ``[count, H, W, 3]`` (uint8, display order) of an H.264 MP4, from frame ``start``.

    Decoding starts at the last IDR picture at or before ``start``."""
    track = mp4.read_h264(path)
    total = len(track.sizes)
    count = total - start if count is None else count
    if start < 0 or count < 0 or start + count > total:
        raise ValueError(f"{path}: frames [{start}, {start + count}) out of range for {total} frames")
    first = max([i for i in track.sync if i <= start], default=0)
    while first > 0 and not track.is_idr(first):
        first = max([i for i in track.sync if i < first], default=0)
    frames, skip = [], start - first
    matrix, full_range = sps_color(track.sps[0])

    def keep(pictures):
        nonlocal skip
        for picture in pictures:
            if len(frames) == count:
                return
            if skip:
                skip -= 1
            else:
                frames.append(i420_to_rgb(*picture, matrix=matrix, full_range=full_range))

    prefix = b"".join(b"\x00\x00\x00\x01" + nal for nal in track.sps + track.pps)
    with openh264.Decoder() as decoder:
        for index, sample in enumerate(track.samples(first)):
            if len(frames) == count:
                break
            keep(decoder.decode((prefix if index == 0 else b"") + track.annexb(sample)))
        else:
            keep(decoder.flush())
    if len(frames) != count:
        raise ValueError(f"{path}: decoded {len(frames)} of {count} frames from {start}")
    return np.stack(frames) if frames else np.zeros((0, track.height, track.width, 3), np.uint8)
