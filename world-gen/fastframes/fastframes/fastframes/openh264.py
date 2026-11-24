#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""H.264 encoding and decoding through Cisco's OpenH264 library (ctypes; no compiled extension).

The library is not bundled. It is looked up in ``$OPENH264_LIBRARY``, then in the download location,
then on the system library path; when none of these has version 2.7 (and ``$OPENH264_LIBRARY`` is unset),
Cisco's prebuilt Linux x86-64 build of OpenH264 2.7.0 is downloaded on first use and its SHA-256 checked
(``python -m fastframes.openh264`` downloads it ahead of time). The structure layouts below follow the 2.7
headers, so other versions are rejected.

Encoder: one spatial and one temporal layer, constant QP (rate control off), CABAC (High profile, no B
frames), an IDR picture every ``idr_interval`` frames, and the same SPS/PPS for the whole stream.
Decoder: Constrained Baseline, Main and High profile streams (including B frames), output in display order.
For some B-frame streams (temporal direct prediction, B-pyramids) it differs from other decoders by a few
levels in small areas.
"""
from __future__ import annotations

import argparse
import bz2
import ctypes
import ctypes.util
import hashlib
import http.client
import os
import secrets
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["Encoder", "Decoder", "load_library", "split_annexb", "download"]

VERSION = (2, 7)
BINARY = "libopenh264-2.7.0-linux64.8.so"
BINARY_URL = f"https://ciscobinary.openh264.org/{BINARY}.bz2"
BINARY_BZ2_SHA256 = "44ad0dd2c5af2bac84d26f89f2eb4e268708b8739323d15d33837f7221ce4980"
BINARY_SHA256 = "1522f873a482923c5ec018a4415e1866898197528828c50f90fd300c94620df3"  # published by Cisco
DEFAULT_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fastframes" / "openh264"

# Constants of codec_app_def.h / codec_def.h (OpenH264 2.7).
MAX_SLICES_NUM_TMP = 35
MAX_SPATIAL_LAYER_NUM = 4
MAX_LAYER_NUM_OF_FRAME = 128
VIDEO_FORMAT_I420 = 23
FRAME_TYPE_SKIP = 4
CAMERA_VIDEO_REAL_TIME = 0
RC_OFF_MODE = -1
HIGH_COMPLEXITY = 2
PRO_HIGH = 100
CONSTANT_ID = 0
MIN_QP = 18  # lower QPs can overflow the library's bitstream buffer on noisy frames
VIDEO_FORMAT_UNDEFINED, SMPTE170M = 5, 6  # VUI video_format, colour description
ENCODER_OPTION_DATAFORMAT = 0
ENCODER_OPTION_TRACE_LEVEL = 25
DECODER_OPTION_ERROR_CON_IDC = 8
DECODER_OPTION_TRACE_LEVEL = 9
DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER = 18
WELS_LOG_ERROR = 1
VIDEO_BITSTREAM_AVC = 0
ERROR_CON_DISABLE = 0
DS_ERROR_MASK = 0x04 | 0x08 | 0x10 | 0x1000 | 0x2000 | 0x4000 | 0x8000  # failures (not "frame pending")


class SliceArgument(ctypes.Structure):
    _fields_ = [("uiSliceMode", ctypes.c_int), ("uiSliceNum", ctypes.c_uint),
                ("uiSliceMbNum", ctypes.c_uint * MAX_SLICES_NUM_TMP), ("uiSliceSizeConstraint", ctypes.c_uint)]


class SpatialLayerConfig(ctypes.Structure):
    _fields_ = [("iVideoWidth", ctypes.c_int), ("iVideoHeight", ctypes.c_int), ("fFrameRate", ctypes.c_float),
                ("iSpatialBitrate", ctypes.c_int), ("iMaxSpatialBitrate", ctypes.c_int),
                ("uiProfileIdc", ctypes.c_int), ("uiLevelIdc", ctypes.c_int), ("iDLayerQp", ctypes.c_int),
                ("sSliceArgument", SliceArgument), ("bVideoSignalTypePresent", ctypes.c_bool),
                ("uiVideoFormat", ctypes.c_ubyte), ("bFullRange", ctypes.c_bool),
                ("bColorDescriptionPresent", ctypes.c_bool), ("uiColorPrimaries", ctypes.c_ubyte),
                ("uiTransferCharacteristics", ctypes.c_ubyte), ("uiColorMatrix", ctypes.c_ubyte),
                ("bAspectRatioPresent", ctypes.c_bool), ("eAspectRatio", ctypes.c_int),
                ("sAspectRatioExtWidth", ctypes.c_ushort), ("sAspectRatioExtHeight", ctypes.c_ushort)]


class EncParamExt(ctypes.Structure):
    _fields_ = [("iUsageType", ctypes.c_int), ("iPicWidth", ctypes.c_int), ("iPicHeight", ctypes.c_int),
                ("iTargetBitrate", ctypes.c_int), ("iRCMode", ctypes.c_int), ("fMaxFrameRate", ctypes.c_float),
                ("iTemporalLayerNum", ctypes.c_int), ("iSpatialLayerNum", ctypes.c_int),
                ("sSpatialLayers", SpatialLayerConfig * MAX_SPATIAL_LAYER_NUM),
                ("iComplexityMode", ctypes.c_int), ("uiIntraPeriod", ctypes.c_uint), ("iNumRefFrame", ctypes.c_int),
                ("eSpsPpsIdStrategy", ctypes.c_int), ("bPrefixNalAddingCtrl", ctypes.c_bool),
                ("bEnableSSEI", ctypes.c_bool), ("bSimulcastAVC", ctypes.c_bool), ("iPaddingFlag", ctypes.c_int),
                ("iEntropyCodingModeFlag", ctypes.c_int), ("bEnableFrameSkip", ctypes.c_bool),
                ("iMaxBitrate", ctypes.c_int), ("iMaxQp", ctypes.c_int), ("iMinQp", ctypes.c_int),
                ("uiMaxNalSize", ctypes.c_uint), ("bEnableLongTermReference", ctypes.c_bool),
                ("iLTRRefNum", ctypes.c_int), ("iLtrMarkPeriod", ctypes.c_uint),
                ("iMultipleThreadIdc", ctypes.c_ushort), ("bUseLoadBalancing", ctypes.c_bool),
                ("iLoopFilterDisableIdc", ctypes.c_int), ("iLoopFilterAlphaC0Offset", ctypes.c_int),
                ("iLoopFilterBetaOffset", ctypes.c_int), ("bEnableDenoise", ctypes.c_bool),
                ("bEnableBackgroundDetection", ctypes.c_bool), ("bEnableAdaptiveQuant", ctypes.c_bool),
                ("bEnableFrameCroppingFlag", ctypes.c_bool), ("bEnableSceneChangeDetect", ctypes.c_bool),
                ("bIsLosslessLink", ctypes.c_bool), ("bFixRCOverShoot", ctypes.c_bool),
                ("iIdrBitrateRatio", ctypes.c_int), ("bPsnrY", ctypes.c_bool), ("bPsnrU", ctypes.c_bool),
                ("bPsnrV", ctypes.c_bool)]


class SourcePicture(ctypes.Structure):
    _fields_ = [("iColorFormat", ctypes.c_int), ("iStride", ctypes.c_int * 4),
                ("pData", ctypes.POINTER(ctypes.c_ubyte) * 4), ("iPicWidth", ctypes.c_int),
                ("iPicHeight", ctypes.c_int), ("uiTimeStamp", ctypes.c_longlong), ("bPsnrY", ctypes.c_bool),
                ("bPsnrU", ctypes.c_bool), ("bPsnrV", ctypes.c_bool)]


class LayerBSInfo(ctypes.Structure):
    _fields_ = [("uiTemporalId", ctypes.c_ubyte), ("uiSpatialId", ctypes.c_ubyte), ("uiQualityId", ctypes.c_ubyte),
                ("eFrameType", ctypes.c_int), ("uiLayerType", ctypes.c_ubyte), ("iSubSeqId", ctypes.c_int),
                ("iNalCount", ctypes.c_int), ("pNalLengthInByte", ctypes.POINTER(ctypes.c_int)),
                ("pBsBuf", ctypes.POINTER(ctypes.c_ubyte)), ("rPsnr", ctypes.c_float * 3)]


class FrameBSInfo(ctypes.Structure):
    _fields_ = [("iLayerNum", ctypes.c_int), ("sLayerInfo", LayerBSInfo * MAX_LAYER_NUM_OF_FRAME),
                ("eFrameType", ctypes.c_int), ("iFrameSizeInBytes", ctypes.c_int), ("uiTimeStamp", ctypes.c_longlong)]


class VideoProperty(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint), ("eVideoBsType", ctypes.c_int)]


class DecodingParam(ctypes.Structure):
    _fields_ = [("pFileNameRestructed", ctypes.c_char_p), ("uiCpuLoad", ctypes.c_uint),
                ("uiTargetDqLayer", ctypes.c_ubyte), ("eEcActiveIdc", ctypes.c_int), ("bParseOnly", ctypes.c_bool),
                ("sVideoProperty", VideoProperty)]


class SysMemBuffer(ctypes.Structure):
    _fields_ = [("iWidth", ctypes.c_int), ("iHeight", ctypes.c_int), ("iFormat", ctypes.c_int),
                ("iStride", ctypes.c_int * 2)]


class BufferInfo(ctypes.Structure):
    _fields_ = [("iBufferStatus", ctypes.c_int), ("uiInBsTimeStamp", ctypes.c_ulonglong),
                ("uiOutYuvTimeStamp", ctypes.c_ulonglong), ("sSystemBuffer", SysMemBuffer),
                ("pDst", ctypes.POINTER(ctypes.c_ubyte) * 3)]


class Version(ctypes.Structure):
    _fields_ = [("uMajor", ctypes.c_uint), ("uMinor", ctypes.c_uint), ("uRevision", ctypes.c_uint),
                ("uReserved", ctypes.c_uint)]


LAYOUT = {  # sizeof() of the 2.7 headers, checked at load time
    SliceArgument: 152, SpatialLayerConfig: 200, EncParamExt: 924, SourcePicture: 80, LayerBSInfo: 56,
    FrameBSInfo: 7192, DecodingParam: 32, BufferInfo: 72,
}

_P = ctypes.c_void_p
_BYTES = ctypes.POINTER(ctypes.c_ubyte)


class EncoderVtbl(ctypes.Structure):
    _fields_ = [
        ("Initialize", ctypes.CFUNCTYPE(ctypes.c_int, _P, _P)),
        ("InitializeExt", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.POINTER(EncParamExt))),
        ("GetDefaultParams", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.POINTER(EncParamExt))),
        ("Uninitialize", ctypes.CFUNCTYPE(ctypes.c_int, _P)),
        ("EncodeFrame", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.POINTER(SourcePicture),
                                         ctypes.POINTER(FrameBSInfo))),
        ("EncodeParameterSets", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.POINTER(FrameBSInfo))),
        ("ForceIntraFrame", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.c_bool)),
        ("SetOption", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.c_int, _P)),
        ("GetOption", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.c_int, _P)),
    ]


class DecoderVtbl(ctypes.Structure):
    _fields_ = [
        ("Initialize", ctypes.CFUNCTYPE(ctypes.c_long, _P, ctypes.POINTER(DecodingParam))),
        ("Uninitialize", ctypes.CFUNCTYPE(ctypes.c_long, _P)),
        ("DecodeFrame", _P),
        ("DecodeFrameNoDelay", ctypes.CFUNCTYPE(ctypes.c_int, _P, _BYTES, ctypes.c_int,
                                                ctypes.POINTER(_BYTES), ctypes.POINTER(BufferInfo))),
        ("DecodeFrame2", _P),
        ("FlushFrame", ctypes.CFUNCTYPE(ctypes.c_int, _P, ctypes.POINTER(_BYTES), ctypes.POINTER(BufferInfo))),
        ("DecodeParser", _P),
        ("DecodeFrameEx", _P),
        ("SetOption", ctypes.CFUNCTYPE(ctypes.c_long, _P, ctypes.c_int, _P)),
        ("GetOption", ctypes.CFUNCTYPE(ctypes.c_long, _P, ctypes.c_int, _P)),
    ]


_LIBRARY = None


def _candidates():
    if os.environ.get("OPENH264_LIBRARY"):
        return [os.environ["OPENH264_LIBRARY"]]
    found = [str(DEFAULT_DIR / BINARY)] if (DEFAULT_DIR / BINARY).is_file() else []
    system = ctypes.util.find_library("openh264")
    return found + ([system] if system else [])


def load_library():
    """The loaded OpenH264 library (version 2.7), cached."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    for cls, size in LAYOUT.items():
        assert ctypes.sizeof(cls) == size, (cls.__name__, ctypes.sizeof(cls), size)
    errors = []
    for candidate in _candidates():
        lib = _open(candidate, errors)
        if lib is not None:
            break
    else:
        lib = None
        if not os.environ.get("OPENH264_LIBRARY"):
            try:
                lib = _open(str(download()), errors)
            except (OSError, http.client.HTTPException) as exc:  # network errors (URLError is an OSError)
                errors.append(f"download from {BINARY_URL}: {exc}")
    if lib is None:
        raise RuntimeError("OpenH264 2.7 not found. Set OPENH264_LIBRARY to libopenh264.so (version 2.7). "
                           + "; ".join(errors))
    lib.WelsCreateSVCEncoder.argtypes = [ctypes.POINTER(_P)]
    lib.WelsDestroySVCEncoder.argtypes = [_P]
    lib.WelsCreateDecoder.argtypes = [ctypes.POINTER(_P)]
    lib.WelsCreateDecoder.restype = ctypes.c_long
    lib.WelsDestroyDecoder.argtypes = [_P]
    _LIBRARY = lib
    return lib


def _open(candidate, errors):
    """``candidate`` loaded with ctypes if it is OpenH264 2.7, else None (with the reason in ``errors``)."""
    try:
        lib = ctypes.CDLL(candidate)
    except OSError as exc:
        errors.append(f"{candidate}: {exc}")
        return None
    version = Version()
    lib.WelsGetCodecVersionEx(ctypes.byref(version))
    if (version.uMajor, version.uMinor) != VERSION:
        errors.append(f"{candidate}: version {version.uMajor}.{version.uMinor}.{version.uRevision}, "
                      f"need {VERSION[0]}.{VERSION[1]}")
        return None
    return lib


def _vtable(handle, cls):
    return ctypes.cast(handle, ctypes.POINTER(ctypes.POINTER(cls))).contents.contents


def split_annexb(data):
    """NAL units (without start codes) of an Annex B byte stream."""
    data = bytes(data)
    starts, i, n = [], 0, len(data)
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0:
            break
        starts.append(i + 3)
        i += 3
    nals = []
    for k, start in enumerate(starts):
        end = starts[k + 1] - 3 if k + 1 < len(starts) else n
        nal = data[start:end]
        if k + 1 < len(starts) and nal.endswith(b"\x00"):
            nal = nal[:-1]  # the zero_byte of the next 4-byte start code
        if nal:
            nals.append(nal)
    return nals


class Encoder:
    """Constant-QP H.264 encoder for I420 pictures (``width`` and ``height`` even)."""

    def __init__(self, width, height, fps, *, qp=20, idr_interval=None):
        if width % 2 or height % 2:
            raise ValueError(f"H.264 4:2:0 needs even dimensions, got {width}x{height}")
        if not MIN_QP <= qp <= 51:
            raise ValueError(f"qp must be in [{MIN_QP}, 51], got {qp}")
        lib = load_library()
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self._lib, self._handle = lib, _P()
        if lib.WelsCreateSVCEncoder(ctypes.byref(self._handle)) != 0 or not self._handle:
            raise RuntimeError("WelsCreateSVCEncoder failed")
        self._vt = _vtable(self._handle, EncoderVtbl)
        level = ctypes.c_int(WELS_LOG_ERROR)
        self._vt.SetOption(self._handle, ENCODER_OPTION_TRACE_LEVEL, ctypes.cast(ctypes.byref(level), _P))
        param = EncParamExt()
        self._vt.GetDefaultParams(self._handle, ctypes.byref(param))
        param.iUsageType = CAMERA_VIDEO_REAL_TIME
        param.iPicWidth, param.iPicHeight, param.fMaxFrameRate = self.width, self.height, self.fps
        param.iRCMode, param.iTargetBitrate = RC_OFF_MODE, 0
        param.iTemporalLayerNum, param.iSpatialLayerNum = 1, 1
        param.iComplexityMode = HIGH_COMPLEXITY
        param.uiIntraPeriod = int(idr_interval if idr_interval is not None else max(1, round(2 * self.fps)))
        param.eSpsPpsIdStrategy = CONSTANT_ID
        param.bPrefixNalAddingCtrl = param.bEnableSSEI = param.bSimulcastAVC = False
        param.iEntropyCodingModeFlag = 1  # CABAC
        param.bEnableFrameSkip = False
        param.iMinQp = param.iMaxQp = int(qp)
        param.iMultipleThreadIdc = 1
        param.bEnableDenoise = param.bEnableBackgroundDetection = param.bEnableSceneChangeDetect = False
        param.bEnableAdaptiveQuant = False
        param.bEnableFrameCroppingFlag = True
        layer = param.sSpatialLayers[0]
        layer.iVideoWidth, layer.iVideoHeight, layer.fFrameRate = self.width, self.height, self.fps
        layer.uiProfileIdc, layer.iDLayerQp = PRO_HIGH, int(qp)
        layer.sSliceArgument.uiSliceMode, layer.sSliceArgument.uiSliceNum = 0, 1
        layer.bVideoSignalTypePresent = layer.bColorDescriptionPresent = True  # label as BT.601 limited range
        layer.uiVideoFormat, layer.bFullRange = VIDEO_FORMAT_UNDEFINED, False
        layer.uiColorPrimaries = layer.uiTransferCharacteristics = layer.uiColorMatrix = SMPTE170M
        if self._vt.InitializeExt(self._handle, ctypes.byref(param)) != 0:
            self.close()
            raise RuntimeError(f"OpenH264 encoder initialization failed ({width}x{height}, {fps} fps, qp {qp})")
        fmt = ctypes.c_int(VIDEO_FORMAT_I420)
        self._vt.SetOption(self._handle, ENCODER_OPTION_DATAFORMAT, ctypes.cast(ctypes.byref(fmt), _P))
        self._info, self._index = FrameBSInfo(), 0

    def encode(self, y, u, v):
        """One picture (uint8 planes ``[H, W]``, ``[H/2, W/2]`` x2) -> ``(nal_units, is_idr)``."""
        planes = [np.ascontiguousarray(p, dtype=np.uint8) for p in (y, u, v)]
        if planes[0].shape != (self.height, self.width) or any(
                p.shape != (self.height // 2, self.width // 2) for p in planes[1:]):
            raise ValueError("plane shapes do not match the encoder size")
        pic = SourcePicture()
        pic.iColorFormat, pic.iPicWidth, pic.iPicHeight = VIDEO_FORMAT_I420, self.width, self.height
        for k, plane in enumerate(planes):
            pic.iStride[k] = plane.strides[0]
            pic.pData[k] = plane.ctypes.data_as(_BYTES)
        pic.uiTimeStamp = round(self._index * 1000 / self.fps)
        self._index += 1
        info = self._info
        ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
        if self._vt.EncodeFrame(self._handle, ctypes.byref(pic), ctypes.byref(info)) != 0:
            raise RuntimeError(f"OpenH264 EncodeFrame failed at frame {self._index - 1}")
        if info.eFrameType == FRAME_TYPE_SKIP:
            raise RuntimeError("OpenH264 skipped a frame")
        nals = []
        for i in range(info.iLayerNum):
            layer = info.sLayerInfo[i]
            size = sum(layer.pNalLengthInByte[k] for k in range(layer.iNalCount))
            nals += split_annexb(ctypes.string_at(layer.pBsBuf, size))
        return nals, any((nal[0] & 0x1F) == 5 for nal in nals)

    def close(self):
        if self._handle:
            self._vt.Uninitialize(self._handle)
            self._lib.WelsDestroySVCEncoder(self._handle)
            self._handle = _P()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Decoder:
    """H.264 decoder: Annex B access units in decode order -> I420 pictures in display order."""

    def __init__(self):
        lib = load_library()
        self._lib, self._handle = lib, _P()
        if lib.WelsCreateDecoder(ctypes.byref(self._handle)) != 0 or not self._handle:
            raise RuntimeError("WelsCreateDecoder failed")
        self._vt = _vtable(self._handle, DecoderVtbl)
        level = ctypes.c_int(WELS_LOG_ERROR)
        self._vt.SetOption(self._handle, DECODER_OPTION_TRACE_LEVEL, ctypes.cast(ctypes.byref(level), _P))
        param = DecodingParam()
        param.eEcActiveIdc = ERROR_CON_DISABLE
        param.sVideoProperty.size = ctypes.sizeof(VideoProperty)
        param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC
        if self._vt.Initialize(self._handle, ctypes.byref(param)) != 0:
            self.close()
            raise RuntimeError("OpenH264 decoder initialization failed")

    @staticmethod
    def _picture(info, dst):
        if info.iBufferStatus != 1:
            return None
        buf = info.sSystemBuffer
        if buf.iFormat != VIDEO_FORMAT_I420:
            raise RuntimeError(f"OpenH264 decoded an unsupported picture format ({buf.iFormat})")
        w, h, sy, sc = buf.iWidth, buf.iHeight, buf.iStride[0], buf.iStride[1]

        def plane(ptr, stride, rows, cols):
            flat = np.ctypeslib.as_array(ptr, shape=(stride * rows,))
            return flat.reshape(rows, stride)[:, :cols].copy()

        return (plane(dst[0], sy, h, w), plane(dst[1], sc, (h + 1) // 2, (w + 1) // 2),
                plane(dst[2], sc, (h + 1) // 2, (w + 1) // 2))

    def decode(self, access_unit):
        """Decode one access unit (Annex B bytes); returns a list of ``(y, u, v)`` pictures (0 or 1)."""
        data = (ctypes.c_ubyte * len(access_unit)).from_buffer_copy(access_unit)
        dst, info = (_BYTES * 3)(), BufferInfo()
        state = self._vt.DecodeFrameNoDelay(self._handle, data, len(access_unit), dst, ctypes.byref(info))
        if state & DS_ERROR_MASK:
            raise RuntimeError(f"OpenH264 decoding error (state 0x{state:x})")
        picture = self._picture(info, dst)
        return [picture] if picture is not None else []

    def flush(self):
        """Pictures still held for reordering at the end of the stream."""
        out = []
        while True:
            remaining = ctypes.c_int(0)
            self._vt.GetOption(self._handle, DECODER_OPTION_NUM_OF_FRAMES_REMAINING_IN_BUFFER,
                               ctypes.cast(ctypes.byref(remaining), _P))
            if remaining.value <= 0:
                return out
            dst, info = (_BYTES * 3)(), BufferInfo()
            self._vt.FlushFrame(self._handle, dst, ctypes.byref(info))
            picture = self._picture(info, dst)
            if picture is None:
                return out
            out.append(picture)

    def close(self):
        if self._handle:
            self._vt.Uninitialize(self._handle)
            self._lib.WelsDestroyDecoder(self._handle)
            self._handle = _P()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def download(dest_dir=DEFAULT_DIR):
    """Download Cisco's OpenH264 2.7.0 build for Linux x86-64 into ``dest_dir`` (SHA-256 checked)."""
    dest_dir = Path(dest_dir)
    target = dest_dir / BINARY
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == BINARY_SHA256:
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BINARY_URL, timeout=120) as response:
        compressed = response.read()
    if hashlib.sha256(compressed).hexdigest() != BINARY_BZ2_SHA256:
        raise RuntimeError(f"SHA-256 mismatch for {BINARY_URL}")
    binary = bz2.decompress(compressed)
    if hashlib.sha256(binary).hexdigest() != BINARY_SHA256:
        raise RuntimeError(f"SHA-256 mismatch for {BINARY}")
    staged = dest_dir / f".{BINARY}.{secrets.token_hex(4)}.tmp"
    try:
        staged.write_bytes(binary)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description="Download Cisco's OpenH264 2.7.0 library (Linux x86-64).")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DIR, help=f"directory (default: {DEFAULT_DIR})")
    args = parser.parse_args()
    path = download(args.dest)
    if args.dest != DEFAULT_DIR:
        print(f"set OPENH264_LIBRARY={path}")
    print(path)


if __name__ == "__main__":
    main()
