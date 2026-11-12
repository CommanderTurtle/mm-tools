"""Direct loading of complete PixelUMM FP8/BF16 single-file checkpoints."""
import json

import torch

FORMAT = "pixelumm_fp8_bf16_single_v1"


def is_prequantized_pixelumm(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return (handle.metadata() or {}).get("format") == FORMAT


def validate_serialized_tensors(weight, scale, expected_shape):
    shape = tuple(expected_shape)
    if (len(shape) != 2 or any(d % 128 for d in shape)
            or weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != shape
            or scale.dtype != torch.float32
            or tuple(scale.shape) != tuple(d // 128 for d in shape)):
        raise ValueError("Invalid serialized FP8 weight/scale")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("FP8 scales must be finite and positive")


def install_serialized_fp8(decoder, state, metadata, backend):
    if backend == "cutlass":
        from .cutlass_fp8 import CutlassFP8Linear as linear_class
    elif backend == "deepgemm":
        from .fp8 import FP8Linear as linear_class
    else:
        raise ValueError("Unsupported serialized PixelUMM FP8 backend: " + str(backend))

    if (metadata.get("format") != FORMAT or metadata.get("block_size") != "128"
            or metadata.get("weight_dtype") != "float8_e4m3fn"
            or metadata.get("scale_dtype") != "float32"):
        raise ValueError("Unsupported PixelUMM FP8 checkpoint metadata")
    targets = [
        f"layers.{i}.{name}"
        for i in range(len(decoder.layers))
        for name in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                     "attn.q_proj", "attn.o_proj")
    ]
    declared = json.loads(metadata.get("quantized_source_keys", "[]"))
    expected = {"decoder." + name + ".weight" for name in targets}
    if (set(declared) != expected or len(declared) != len(expected)
            or int(metadata.get("quantized_layers", -1)) != len(targets)):
        raise ValueError("Incomplete or incompatible PixelUMM FP8 layer manifest")
    for name in targets:
        linear = decoder.get_submodule(name)
        if not isinstance(linear, torch.nn.Linear) or linear.bias is not None:
            raise ValueError("Expected a bias-free PixelUMM linear: " + name)
        module = linear_class.from_quantized(
            state[name + ".fp8_weight"], state[name + ".fp8_scale"],
            linear.weight.shape,
        )
        parent, _, leaf = name.rpartition(".")
        setattr(decoder.get_submodule(parent), leaf, module)
    for key, tensor in state.items():
        if not key.endswith((".fp8_weight", ".fp8_scale")) and tensor.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 unquantized tensor: " + key)
    decoder._prequantized_fp8_backend = backend
    decoder._prequantized_fp8 = True
    decoder._prequantized_fp8_layers = len(targets)
