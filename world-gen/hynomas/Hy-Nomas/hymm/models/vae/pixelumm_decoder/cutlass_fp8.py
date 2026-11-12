"""Opt-in SM89 inference: CUTLASS E4M3 GEMM, 128x128 scaling on both inputs.

Unlike DeepGEMM's per-token activation scales, this backend uses blockwise
activation scales. Validate reconstructed video quality when enabling it.
Replaced layers retain only FP8 buffers: this is an inference-only model and
must not be saved as a BF16 training checkpoint.
"""
from collections import Counter
from functools import lru_cache
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_calls = Counter()


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load
    root = Path(os.environ.get("CUTLASS_PATH", ""))
    if not (root / "include/cutlass/gemm/device/gemm_blockwise.h").is_file():
        raise RuntimeError("Set CUTLASS_PATH to a CUTLASS checkout containing Ada GemmBlockwise (example 94)")
    return load(
        name="hymm_pixelumm_cutlass_sm89_fp8",
        sources=[str(Path(__file__).parent / "csrc/cutlass_fp8.cu")],
        extra_include_paths=[str(root.resolve() / "include")],
        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89"],
        verbose=False,
    )


def _quantize_blockwise(x):
    m, k = x.shape
    padded = F.pad(x, (0, 0, 0, (-m) % 128))
    blocks = padded.float().reshape(-1, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(-1, k)[:m].contiguous(), scale.contiguous()


quantize_blockwise = torch.compile(_quantize_blockwise, fullgraph=True, dynamic=False)


@torch.library.custom_op("hymm_pixelumm_cutlass_fp8::linear", mutates_args=())
def _linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q, s = quantize_blockwise(x)
    result = load_extension().gemm(q, weight, s, scale)
    _calls[(x.device.index, x.shape[0], x.shape[1], weight.shape[0])] += 1
    return result


@_linear.register_fake
def _linear_fake(x, weight, scale):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


class CutlassFP8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        # One-time startup conversion: neither weight quantization nor compilation enters timing.
        q, s = _quantize_blockwise(linear.weight.detach())
        self.register_buffer("fp8_weight", q)
        self.register_buffer("fp8_scale", s)
        self.train(False)

    @classmethod
    def from_quantized(cls, weight, scale, expected_shape):
        """Install serialized buffers unchanged, without weight quantization."""
        from .prequantized import validate_serialized_tensors

        validate_serialized_tensors(weight, scale, expected_shape)
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.register_buffer("fp8_weight", weight.contiguous())
        module.register_buffer("fp8_scale", scale.contiguous())
        return module.eval()

    def _apply(self, fn, recurse=True):
        # Device/offload moves must preserve the serialized FP8 and FP32 dtypes.
        for name in ("fp8_weight", "fp8_scale"):
            tensor = self._buffers[name]
            target = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
            self._buffers[name] = tensor.to(device=target.device)
        return self

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("CUTLASS FP8 supports eval + no_grad inference only")
        if x.dtype != torch.bfloat16:
            if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
                raise TypeError("CUTLASS FP8 requires BF16 inputs or BF16 CUDA autocast")
            x = x.to(torch.bfloat16)
        y = _linear(x.reshape(-1, x.shape[-1]).contiguous(), self.fp8_weight, self.fp8_scale)
        return y.reshape(*x.shape[:-1], self.fp8_weight.shape[0])


@torch.no_grad()
def enable_cutlass_fp8(decoder):
    from .model import DCAEVPixelUMMDecoder
    if type(decoder) is not DCAEVPixelUMMDecoder or decoder.training:
        raise ValueError("Expected an eval-mode DCAEVPixelUMMDecoder")
    targets = []
    for layer in decoder.layers:
        targets.extend((layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj"))
        targets.extend((layer.attn, name) for name in ("q_proj", "o_proj"))
    for parent, name in targets:
        linear = getattr(parent, name)
        if not isinstance(linear, nn.Linear) or linear.bias is not None:
            raise ValueError("Expected an unmodified bias-free linear")
        w = linear.weight
        if not w.is_cuda or w.dtype != torch.bfloat16 or any(d % 128 for d in w.shape):
            raise ValueError("Expected CUDA BF16 weights with dimensions divisible by 128")
        if torch.cuda.get_device_capability(w.device) != (8, 9):
            raise ValueError("CUTLASS FP8 backend requires SM89")
    load_extension()
    for parent, name in targets:
        setattr(parent, name, CutlassFP8Linear(getattr(parent, name)))
    return len(targets)


def execution_stats():
    return [{"device": d, "M": m, "K": k, "N": n, "calls": count}
            for (d, m, k, n), count in sorted(_calls.items())]
