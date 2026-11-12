"""INT8 storage for segment-AR KV, with BF16 attention inputs.

Each token/head has one FP32 scale for K and another for V. The CUDA kernels
write directly into the persistent cache and decode one layer's valid prefix
into a short-lived BF16 tensor before the existing attention implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from hymm.utils.inference_metrics import profiled


KV_KEYS = ("k", "v")
STORAGE_KEYS = ("k", "v", "k_scale", "v_scale")
ROWS_PER_PROGRAM = 8


@triton.jit
def _quantize_rows(
    X, Q, S,
    L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr, XS3: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, QS3: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
    START, TOTAL: tl.constexpr,
    ROWS: tl.constexpr, BD: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, BD)
    batch = row // (L * H)
    token = (row // H) % L
    head = row % H
    mask = (row[:, None] < TOTAL) & (dim[None, :] < D)
    x = tl.load(
        X + batch[:, None] * XS0 + token[:, None] * XS1
        + head[:, None] * XS2 + dim[None, :] * XS3,
        mask=mask, other=0,
    ).to(tl.float32)
    peak = tl.max(tl.abs(x), 1)
    scale = tl.where(peak > 0, peak / 127.0, 1.0)
    scaled = x / scale[:, None]
    rounded = tl.where(
        scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5)
    )
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(
        Q + batch[:, None] * QS0 + (token[:, None] + START) * QS1
        + head[:, None] * QS2 + dim[None, :] * QS3,
        quantized, mask=mask,
    )
    tl.store(
        S + batch * SS0 + (token + START) * SS1 + head * SS2,
        scale, mask=row < TOTAL,
    )


@triton.jit
def _dequantize_rows(
    Q, S, Y,
    L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, QS3: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
    YS0: tl.constexpr, YS1: tl.constexpr, YS2: tl.constexpr, YS3: tl.constexpr,
    TOTAL: tl.constexpr, ROWS: tl.constexpr, BD: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, BD)
    batch = row // (L * H)
    token = (row // H) % L
    head = row % H
    mask = (row[:, None] < TOTAL) & (dim[None, :] < D)
    q = tl.load(
        Q + batch[:, None] * QS0 + token[:, None] * QS1
        + head[:, None] * QS2 + dim[None, :] * QS3,
        mask=mask, other=0,
    ).to(tl.float32)
    scale = tl.load(
        S + batch * SS0 + token * SS1 + head * SS2,
        mask=row < TOTAL, other=1,
    )
    tl.store(
        Y + batch[:, None] * YS0 + token[:, None] * YS1
        + head[:, None] * YS2 + dim[None, :] * YS3,
        (q * scale[:, None]).to(tl.bfloat16), mask=mask,
    )


def make_int8_kv_block(
    batch: int, capacity: int, heads: int, dim: int,
    *, device: torch.device | str, pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    value_shape = (batch, capacity, heads, dim)
    scale_shape = (batch, capacity, heads, 1)
    return {
        "k": torch.empty(value_shape, device=device, dtype=torch.int8,
                         pin_memory=pin_memory),
        "v": torch.empty(value_shape, device=device, dtype=torch.int8,
                         pin_memory=pin_memory),
        "k_scale": torch.empty(scale_shape, device=device, dtype=torch.float32,
                               pin_memory=pin_memory),
        "v_scale": torch.empty(scale_shape, device=device, dtype=torch.float32,
                               pin_memory=pin_memory),
    }


def is_int8_kv_cache(cache: dict) -> bool:
    return cache["k"].dtype == torch.int8


def _validate(cache: dict, key: str) -> None:
    if key not in KV_KEYS:
        raise ValueError(f"Invalid KV key: {key}")
    q = cache[key]
    scale = cache[f"{key}_scale"]
    if q.dtype != torch.int8 or scale.dtype != torch.float32:
        raise TypeError("INT8 KV data and FP32 scale are required")
    if scale.shape != (*q.shape[:-1], 1) or scale.device != q.device:
        raise ValueError("INT8 KV scale shape/device mismatch")


def quantize_kv_into(cache: dict, key: str, values: torch.Tensor,
                     start: int, *, assume_finite: bool = False) -> None:
    """Append BF16 values into INT8 storage at token offset ``start``."""
    _validate(cache, key)
    q = cache[key]
    scale = cache[f"{key}_scale"]
    if values.ndim != 4 or values.dtype != torch.bfloat16:
        raise TypeError("KV quantization expects [B,L,H,D] BF16 values")
    batch, length, heads, dim = values.shape
    if (batch != q.shape[0] or heads != q.shape[2] or dim != q.shape[3]
            or start < 0 or start + length > q.shape[1]
            or values.device != q.device):
        raise ValueError("KV quantization shape, capacity, or device mismatch")
    if length == 0:
        return
    if not assume_finite and not bool(torch.isfinite(values).all()):
        raise ValueError("Cannot quantize non-finite KV values")
    if values.is_cuda:
        total = batch * length * heads
        _quantize_rows[(triton.cdiv(total, ROWS_PER_PROGRAM),)](
            values, q, scale, length, heads, dim,
            *values.stride(), *q.stride(), *scale.stride()[:3],
            start, total, ROWS_PER_PROGRAM, triton.next_power_of_2(dim),
        )
    else:
        src = values.float()
        peak = src.abs().amax(dim=-1, keepdim=True)
        s = torch.where(peak > 0, peak / 127.0, torch.ones_like(peak))
        scaled = src / s
        rounded = torch.where(
            scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5)
        )
        q[:, start:start + length].copy_(rounded.clamp(-127, 127).to(torch.int8))
        scale[:, start:start + length].copy_(s)


@profiled("int8_encode")
def quantize_kv_pair_into(cache: dict, k: torch.Tensor, v: torch.Tensor,
                          start: int) -> None:
    """Validate K and V before writing either persistent payload or scale."""
    if k.shape != v.shape or k.dtype != v.dtype or k.device != v.device:
        raise ValueError("K/V shapes, dtypes, and devices must match")
    for key, values in (("k", k), ("v", v)):
        _validate(cache, key)
        q = cache[key]
        if (values.ndim != 4 or values.dtype != torch.bfloat16
                or values.shape[0] != q.shape[0]
                or values.shape[2:] != q.shape[2:]
                or start < 0 or start + values.shape[1] > q.shape[1]
                or values.device != q.device):
            raise ValueError("KV pair shape, capacity, dtype, or device mismatch")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Cannot quantize non-finite KV values")
    quantize_kv_into(cache, "k", k, start, assume_finite=True)
    quantize_kv_into(cache, "v", v, start, assume_finite=True)


@profiled("int8_decode")
def dequantize_kv(cache: dict, key: str, valid_len: int) -> torch.Tensor:
    """Decode only the valid prefix into a BF16 attention tensor."""
    _validate(cache, key)
    q = cache[key]
    scale = cache[f"{key}_scale"]
    if valid_len < 0 or valid_len > q.shape[1]:
        raise ValueError("Invalid KV valid length")
    batch, _, heads, dim = q.shape
    result = torch.empty((batch, valid_len, heads, dim),
                         device=q.device, dtype=torch.bfloat16)
    if valid_len == 0:
        return result
    if q.is_cuda:
        total = batch * valid_len * heads
        _dequantize_rows[(triton.cdiv(total, ROWS_PER_PROGRAM),)](
            q, scale, result, valid_len, heads, dim,
            *q.stride(), *scale.stride()[:3], *result.stride(),
            total, ROWS_PER_PROGRAM, triton.next_power_of_2(dim),
        )
    else:
        result.copy_((q[:, :valid_len].float()
                      * scale[:, :valid_len]).to(torch.bfloat16))
    return result


def kv_storage_bytes(cache: dict) -> int:
    keys = STORAGE_KEYS if is_int8_kv_cache(cache) else KV_KEYS
    return sum(cache[key].numel() * cache[key].element_size() for key in keys)
