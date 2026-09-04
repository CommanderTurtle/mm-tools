"""Private nodes for the original, local Ideogram per-row FP8 checkpoints.

No checkpoint conversion, network lookup, or modifications to bundled Comfy.
"""
import os
from pathlib import Path

import torch
import comfy.ops
import comfy.sd
from safetensors import safe_open


class RowFP8Ops(comfy.ops.manual_cast):
    class Linear(comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # Allocate at load, not a second full BF16 model during construction.
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = in_features, out_features
            self.has_bias = bias
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.register_buffer("weight_scale", None)
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def _load_from_state_dict(self, sd, prefix, metadata, strict, missing, unexpected, errors):
            weight, scale, bias = (sd.get(prefix + name) for name in ("weight", "weight_scale", "bias"))
            if weight is None or tuple(weight.shape) != (self.out_features, self.in_features):
                errors.append(f"{prefix}weight missing or has incompatible dimensions")
                return
            is_fp8 = weight.dtype == torch.float8_e4m3fn
            if is_fp8 and (scale is None or tuple(scale.shape) != (self.out_features,)):
                errors.append(f"{prefix}FP8 requires one scale per output row")
                return
            if scale is not None and (not is_fp8 or scale.dtype != torch.float32):
                errors.append(f"{prefix}unsupported quantization layout")
                return
            if self.has_bias and (bias is None or tuple(bias.shape) != (self.out_features,)):
                errors.append(f"{prefix}bias missing or has incompatible dimensions")
                return
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.bias = torch.nn.Parameter(bias, requires_grad=False) if self.has_bias else None
            self.weight_scale = scale
            # ModelPatcher must retain FP8 storage, casting only for each matmul.
            self.weight_comfy_model_dtype = weight.dtype
            self.bias_comfy_model_dtype = bias.dtype if self.has_bias else None
            for key in sd:
                if key.startswith(prefix) and key[len(prefix):] not in {"weight", "bias", "weight_scale"}:
                    unexpected.append(key)

        def forward_comfy_cast_weights(self, input):
            with comfy.ops.CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                if self.weight_scale is not None:
                    # Identical dequantization order to ideogram4.quantized_loading.Fp8Linear.
                    weight = weight * self.weight_scale.to(device=input.device, dtype=input.dtype)[:, None]
                return torch.nn.functional.linear(input, weight, bias)


def checkpoint(component, filename, text_only=False):
    root = Path(os.environ["IDEOGRAM4_FP8_MODEL"]).expanduser()
    path = root / component / filename
    if not path.is_file():
        raise RuntimeError(f"Local Ideogram checkpoint missing: {path}")
    # Read only language tensors for conditioning; magic captioning has its own small VL model.
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if text_only:
            return {"model." + k[len("language_model."):]: f.get_tensor(k)
                    for k in f.keys() if k.startswith("language_model.") and k != "language_model.norm.weight"}
        return {k: f.get_tensor(k) for k in f.keys()}


def layout(state):
    return {key: tuple(value.shape) for key, value in state.items()}


def verify_loaded(expected, module):
    actual = layout(module.state_dict())
    if actual != expected:
        mismatch = [key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)]
        raise RuntimeError("Ideogram checkpoint/runtime layout mismatch: " + ", ".join(sorted(mismatch)[:12]))


class LocalModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"component": (["conditional", "unconditional"],)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "mm-tools/Ideogram"

    def load(self, component):
        if component not in {"conditional", "unconditional"}:
            raise ValueError("Invalid Ideogram component")
        part = "transformer" if component == "conditional" else "unconditional_transformer"
        state = checkpoint(part, "diffusion_pytorch_model.safetensors")
        expected = layout(state)
        model = comfy.sd.load_diffusion_model_state_dict(
            state,
            model_options={"custom_operations": RowFP8Ops, "dtype": torch.bfloat16},
            disable_dynamic=True,
        )
        if model is None:
            raise RuntimeError("Bundled Comfy does not recognize the local Ideogram architecture")
        verify_loaded(expected, model.model.diffusion_model)
        return (model,)


class LocalText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "mm-tools/Ideogram"

    def load(self):
        state = checkpoint("text_encoder", "model.safetensors", text_only=True)
        expected = layout(state)
        clip = comfy.sd.load_text_encoder_state_dicts(
            [state],
            clip_type=comfy.sd.CLIPType.IDEOGRAM4,
            model_options={"custom_operations": RowFP8Ops, "dtype": torch.bfloat16},
            disable_dynamic=True,
        )
        verify_loaded(expected, clip.cond_stage_model.qwen3vl_8b.transformer)
        return (clip,)


class LocalVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "mm-tools/Ideogram"

    def load(self):
        vae = comfy.sd.VAE(sd=checkpoint("vae", "diffusion_pytorch_model.safetensors"))
        vae.throw_exception_if_invalid()
        return (vae,)


NODE_CLASS_MAPPINGS = {"MMToolsIdeogramLocal": LocalModel,
                       "MMToolsIdeogramText": LocalText, "MMToolsIdeogramVAE": LocalVAE}
