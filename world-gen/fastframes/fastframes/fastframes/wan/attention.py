# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ACKNOWLEDGEMENTS.
# Modified for FastFrames: attention uses PyTorch SDPA with the cuDNN backend
# (fastest on B200; PyTorch would otherwise auto-select FLASH).
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ["attention", "cross_attention", "set_cross_attention_backend"]

_cross_attention_backend = "cudnn"


def attention(q, k, v, causal=False, dropout_p=0.0, backend="cudnn"):
    """BF16 multi-head attention on ``[B, L, H, D]`` tensors.

    ``backend="auto"`` leaves the kernel choice to PyTorch. The result is
    ``[B, L, H, D]`` in the dtype of ``q``.
    """
    out_dtype = q.dtype
    q = q.transpose(1, 2).to(torch.bfloat16)
    k = k.transpose(1, 2).to(torch.bfloat16)
    v = v.transpose(1, 2).to(torch.bfloat16)
    if backend == "auto":
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
    else:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def set_cross_attention_backend(name):
    """Kernel of the text cross-attention: ``"cudnn"`` (default, used by all released
    inference) or ``"auto"`` (PyTorch's own dispatch, as in the original Wan2.1-14B
    ODE, DMD and KV-refinement training runs)."""
    global _cross_attention_backend
    if name not in ("cudnn", "auto"):
        raise ValueError(f"unknown cross-attention backend {name!r}")
    _cross_attention_backend = name


def cross_attention(q, k, v):
    return attention(q, k, v, backend=_cross_attention_backend)
