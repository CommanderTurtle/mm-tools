#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Minimal MP4 (ISO BMFF) writing and reading for one H.264 video track; standard library only.

``write_h264`` stores length-prefixed access units with ``moov`` before ``mdat`` (faststart), so
browsers can start playing before the whole file has arrived. ``probe`` reports what the media checks
need (codec, size, frame count, average frame rate, top-level box order) and ``read_h264`` gives the
parameter sets and the access units of the first video track in decode order. Fragmented MP4 is not
supported.
"""
from __future__ import annotations

import os
import secrets
import struct
from fractions import Fraction
from pathlib import Path

__all__ = ["write_h264", "probe", "read_h264", "top_level_boxes", "H264Track"]

CODECS = {b"avc1": "h264", b"avc3": "h264", b"hvc1": "hevc", b"hev1": "hevc", b"vp09": "vp9", b"av01": "av1"}
MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


def _box(kind, *payload):
    body = b"".join(payload)
    return struct.pack(">I4s", 8 + len(body), kind) + body


def _full_box(kind, version, flags, *payload):
    return _box(kind, struct.pack(">I", (version << 24) | flags), *payload)


def _timing(fps):
    """``(timescale, sample_delta)`` for a constant frame rate."""
    rate = (Fraction(fps) if isinstance(fps, (int, Fraction)) else Fraction(float(fps))).limit_denominator(1001)
    if rate <= 0:
        raise ValueError(f"invalid frame rate {fps}")
    if rate.denominator == 1:
        return rate.numerator * 1000, 1000
    return rate.numerator, rate.denominator


def _avcc(sps, pps):
    profile, compatibility, level = sps[1], sps[2], sps[3]
    out = struct.pack(">BBBBBB", 1, profile, compatibility, level, 0xFC | 3, 0xE0 | 1)
    out += struct.pack(">H", len(sps)) + sps + struct.pack(">BH", 1, len(pps)) + pps
    if profile in (100, 110, 122, 144):  # chroma 4:2:0, 8-bit, no SPS extensions
        out += bytes([0xFC | 1, 0xF8, 0xF8, 0])
    return _box(b"avcC", out)


def write_h264(path, access_units, *, width, height, fps, sps, pps):
    """Write ``access_units`` (lists of NAL units without start codes, decode order = display order;
    the first one an IDR picture) as a faststart MP4. SPS/PPS NAL units are taken from ``sps``/``pps``
    and dropped from the samples. The file is written next to ``path`` and renamed into place."""
    samples, sync = [], []
    for index, nals in enumerate(access_units):
        kept = [n for n in nals if (n[0] & 0x1F) not in (7, 8)]
        for n in nals:
            if (n[0] & 0x1F) == 7 and n != sps or (n[0] & 0x1F) == 8 and n != pps:
                raise ValueError(f"parameter sets change at access unit {index}")
        samples.append(b"".join(struct.pack(">I", len(n)) + n for n in kept))
        if any((n[0] & 0x1F) == 5 for n in kept):
            sync.append(index + 1)
    if not samples or sync[:1] != [1]:
        raise ValueError("the first access unit must be an IDR picture")
    timescale, delta = _timing(fps)
    count, duration = len(samples), len(samples) * delta
    movie_duration = round(duration * 1000 / timescale)

    def moov(chunk_offset):
        entry = (bytes(6) + struct.pack(">H", 1) + bytes(16) + struct.pack(">HHIIIH", width, height, 0x480000,
                                                                          0x480000, 0, 1)
                 + bytes(32) + struct.pack(">Hh", 0x18, -1) + _avcc(sps, pps))
        stbl = _box(b"stbl",
                    _full_box(b"stsd", 0, 0, struct.pack(">I", 1), _box(b"avc1", entry)),
                    _full_box(b"stts", 0, 0, struct.pack(">III", 1, count, delta)),
                    _full_box(b"stss", 0, 0, struct.pack(f">I{len(sync)}I", len(sync), *sync))
                    if len(sync) < count else b"",
                    _full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, count, 1)),
                    _full_box(b"stsz", 0, 0, struct.pack(f">II{count}I", 0, count, *map(len, samples))),
                    _full_box(b"stco", 0, 0, struct.pack(">II", 1, chunk_offset)))
        minf = _box(b"minf", _full_box(b"vmhd", 0, 1, bytes(8)),
                    _box(b"dinf", _full_box(b"dref", 0, 0, struct.pack(">I", 1), _full_box(b"url ", 0, 1))),
                    stbl)
        mdia = _box(b"mdia",
                    _full_box(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, timescale, duration, 0x55C4, 0)),
                    _full_box(b"hdlr", 0, 0, bytes(4), b"vide", bytes(12), b"VideoHandler\x00"), minf)
        tkhd = _full_box(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, 1, 0, movie_duration), bytes(8),
                         struct.pack(">hhhH", 0, 0, 0, 0), MATRIX, struct.pack(">II", width << 16, height << 16))
        mvhd = _full_box(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, 1000, movie_duration),
                         struct.pack(">IH", 0x10000, 0x100), bytes(10), MATRIX, bytes(24), struct.pack(">I", 2))
        return _box(b"moov", mvhd, _box(b"trak", tkhd, mdia))

    ftyp = _box(b"ftyp", b"isom", struct.pack(">I", 0x200), b"isom", b"iso2", b"avc1", b"mp41")
    payload = sum(map(len, samples))
    offset = len(ftyp) + len(moov(0)) + 8
    if offset + payload >= 2**32:
        raise ValueError("MP4 files of 4 GiB or more are not supported")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(staged, "xb") as f:
            f.write(ftyp + moov(offset) + struct.pack(">I4s", 8 + payload, b"mdat"))
            for sample in samples:
                f.write(sample)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return path


def _iter_boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    offset = start
    while offset + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, offset)
        header = 8
        if size == 1:
            size, header = struct.unpack_from(">Q", data, offset + 8)[0], 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            raise ValueError(f"invalid MP4 box {kind!r} at offset {offset}")
        yield kind, offset + header, offset + size
        offset += size


def _child(data, start, end, kind):
    for k, s, e in _iter_boxes(data, start, end):
        if k == kind:
            return s, e
    return None


def top_level_boxes(path):
    """Types of the top-level boxes, in file order (reads only the box headers)."""
    boxes, total = [], os.path.getsize(path)
    with open(path, "rb") as handle:
        offset = 0
        while offset + 8 <= total:
            handle.seek(offset)
            size, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if size == 1:
                size, header = struct.unpack(">Q", handle.read(8))[0], 16
            elif size == 0:
                size = total - offset
            if size < header or offset + size > total:
                raise ValueError(f"invalid MP4 box {kind!r} in {path}")
            boxes.append(kind.decode("latin1"))
            offset += size
    return boxes


def _moov(path):
    total = os.path.getsize(path)
    with open(path, "rb") as handle:
        offset = 0
        while offset + 8 <= total:
            handle.seek(offset)
            size, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if size == 1:
                size, header = struct.unpack(">Q", handle.read(8))[0], 16
            elif size == 0:
                size = total - offset
            if size < header or offset + size > total:
                raise ValueError(f"invalid MP4 box {kind!r} in {path}")
            if kind == b"moov":
                handle.seek(offset)
                return handle.read(size)
            offset += size
    raise ValueError(f"no moov box in {path}")


def _video_track(path):
    """Sample table of the first video track: dict with codec, entry, timescale, sizes, offsets, deltas, sync."""
    try:
        return _parse_video_track(path, _moov(path))
    except (struct.error, TypeError, IndexError, StopIteration) as exc:
        raise ValueError(f"unsupported or malformed MP4: {path} ({exc!r})") from None


def _parse_video_track(path, moov):
    header = 16 if struct.unpack_from(">I", moov)[0] == 1 else 8
    for kind, start, end in _iter_boxes(moov, header):
        if kind != b"trak":
            continue
        mdia = _child(moov, start, end, b"mdia")
        hdlr = mdia and _child(moov, *mdia, b"hdlr")
        if not hdlr or moov[hdlr[0] + 8:hdlr[0] + 12] != b"vide":
            continue
        mdhd = _child(moov, *mdia, b"mdhd")
        version = moov[mdhd[0]]
        timescale = struct.unpack_from(">I", moov, mdhd[0] + (20 if version == 1 else 12))[0]
        stbl = _child(moov, *_child(moov, *mdia, b"minf"), b"stbl")
        stsd = _child(moov, *stbl, b"stsd")
        entry_kind, entry_start, entry_end = next(_iter_boxes(moov, stsd[0] + 8, stsd[1]))
        width, height = struct.unpack_from(">HH", moov, entry_start + 24)
        stsz = _child(moov, *stbl, b"stsz")
        uniform, count = struct.unpack_from(">II", moov, stsz[0] + 4)
        sizes = [uniform] * count if uniform else list(struct.unpack_from(f">{count}I", moov, stsz[0] + 12))
        stts = _child(moov, *stbl, b"stts")
        n = struct.unpack_from(">I", moov, stts[0] + 4)[0]
        runs = struct.unpack_from(f">{2 * n}I", moov, stts[0] + 8)
        deltas = [d for c, d in zip(runs[::2], runs[1::2]) for _ in range(c)]
        stco = _child(moov, *stbl, b"stco")
        if stco:
            n = struct.unpack_from(">I", moov, stco[0] + 4)[0]
            chunks = list(struct.unpack_from(f">{n}I", moov, stco[0] + 8))
        else:
            co64 = _child(moov, *stbl, b"co64")
            n = struct.unpack_from(">I", moov, co64[0] + 4)[0]
            chunks = list(struct.unpack_from(f">{n}Q", moov, co64[0] + 8))
        stsc = _child(moov, *stbl, b"stsc")
        n = struct.unpack_from(">I", moov, stsc[0] + 4)[0]
        rows = [struct.unpack_from(">III", moov, stsc[0] + 8 + 12 * i) for i in range(n)]
        offsets, sample = [], 0
        for i, (first, per_chunk, _) in enumerate(rows):
            last = rows[i + 1][0] - 1 if i + 1 < len(rows) else len(chunks)
            for chunk in range(first, last + 1):
                position = chunks[chunk - 1]
                for _ in range(per_chunk):
                    if sample >= count:
                        break
                    offsets.append(position)
                    position += sizes[sample]
                    sample += 1
        if len(offsets) != count:
            raise ValueError(f"inconsistent sample table in {path}")
        stss = _child(moov, *stbl, b"stss")
        if stss:
            n = struct.unpack_from(">I", moov, stss[0] + 4)[0]
            sync = [i - 1 for i in struct.unpack_from(f">{n}I", moov, stss[0] + 8)]
        else:
            sync = list(range(count))
        return dict(codec=CODECS.get(entry_kind, entry_kind.decode("latin1")), entry=(moov, entry_start, entry_end),
                    width=width, height=height, timescale=timescale, sizes=sizes, offsets=offsets, deltas=deltas,
                    sync=sync)
    raise ValueError(f"no video track in {path}")


def probe(path):
    """``codec``, ``width``, ``height``, ``frames`` and ``fps`` (average) of the first video track."""
    track = _video_track(path)
    duration = sum(track["deltas"])
    frames = len(track["sizes"])
    fps = Fraction(frames * track["timescale"], duration) if duration else Fraction(0)
    return {"codec": track["codec"], "width": track["width"], "height": track["height"], "frames": frames,
            "fps": fps}


class H264Track:
    """Parameter sets and sample table of an H.264 video track (see ``read_h264``)."""

    def __init__(self, path, track):
        moov, start, end = track["entry"]
        avcc = _child(moov, start + 78, end, b"avcC")
        if avcc is None:
            raise ValueError(f"{path}: no avcC box")
        config = moov[avcc[0]:avcc[1]]
        self.path, self.length_size = path, (config[4] & 3) + 1
        self.width, self.height = track["width"], track["height"]
        self.sizes, self.offsets, self.sync = track["sizes"], track["offsets"], track["sync"]
        position, self.sps, self.pps = 6, [], []
        for _ in range(config[5] & 0x1F):
            n = struct.unpack_from(">H", config, position)[0]
            self.sps.append(bytes(config[position + 2:position + 2 + n]))
            position += 2 + n
        for _ in range(config[position]):
            n = struct.unpack_from(">H", config, position + 1)[0]
            self.pps.append(bytes(config[position + 3:position + 3 + n]))
            position += 2 + n

    def samples(self, start=0):
        """Access units (length-prefixed NAL units) in decode order, from sample ``start``."""
        with open(self.path, "rb") as handle:
            for offset, size in zip(self.offsets[start:], self.sizes[start:]):
                handle.seek(offset)
                data = handle.read(size)
                if len(data) != size:
                    raise ValueError(f"{self.path}: truncated sample at offset {offset}")
                yield data

    def nal_units(self, sample):
        out, position, n = [], 0, self.length_size
        while position + n <= len(sample):
            size = int.from_bytes(sample[position:position + n], "big")
            out.append(sample[position + n:position + n + size])
            position += n + size
        if position != len(sample):
            raise ValueError(f"{self.path}: malformed access unit")
        return out

    def annexb(self, sample):
        """``sample`` with start codes instead of length prefixes."""
        return b"".join(b"\x00\x00\x00\x01" + nal for nal in self.nal_units(sample))

    def is_idr(self, index):
        return any(nal and (nal[0] & 0x1F) == 5 for nal in self.nal_units(next(self.samples(index))))


def read_h264(path):
    """``H264Track`` of the first video track; raises ``ValueError`` if that track is not H.264."""
    track = _video_track(path)
    if track["codec"] != "h264":
        raise ValueError(f"{path}: video codec is {track['codec']}, not H.264")
    return H264Track(path, track)
