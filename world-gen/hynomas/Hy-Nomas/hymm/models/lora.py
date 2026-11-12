"""Fixed wide-target LoRA adapters for segment SFT and inference."""

from fnmatch import fnmatchcase
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from safetensors.torch import load_file, save_file


# The same fixed scope is used by training, adapter loading, and checkpoint
# metadata. Do not match leaf names: refiner/head layers must remain frozen.
LORA_TARGETS = (
    "double_blocks.*.img_attn_q",
    "double_blocks.*.img_attn_k",
    "double_blocks.*.img_attn_v",
    "double_blocks.*.img_attn_proj",
    "double_blocks.*.txt_attn_q",
    "double_blocks.*.txt_attn_k",
    "double_blocks.*.txt_attn_v",
    "double_blocks.*.txt_attn_proj",
    "double_blocks.*.img_mlp.fc1",
    "double_blocks.*.img_mlp.fc2",
    "double_blocks.*.txt_mlp.fc1",
    "double_blocks.*.txt_mlp.fc2",
    "double_blocks.*.img_mod.linear",
    "double_blocks.*.txt_mod.linear",
    "action_modules.*.img_attn_q",
    "action_modules.*.keyboard_attn_kv",
    "action_modules.*.proj_keyboard",
)


def _canonical_adapter_name(name):
    """Remove activation-checkpoint wrapper components from adapter keys."""
    return name.replace("._checkpoint_wrapped_module.", ".").replace(
        "_checkpoint_wrapped_module.", ""
    )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        # Preserve the nn.Linear metadata expected by FLOPs accounting and
        # other model-introspection helpers after the projection is wrapped.
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_B = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * (
            self.alpha / self.rank
        )


def inject_lora(model, *, rank=8, alpha=8.0, dropout=0.0):
    """Adapt DiT Q/K/V/O, MLPs, modulation, and camera projections only."""
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("LoRA requires rank > 0, alpha > 0, and 0 <= dropout < 1")
    model.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(
            fnmatchcase(_canonical_adapter_name(name), target)
            for target in LORA_TARGETS
        ):
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    if not replaced:
        raise ValueError("The model has no supported wide LoRA projections")
    return replaced


def _adapter_parameters(model):
    return {
        name: param
        for name, param in model.named_parameters()
        if name.endswith(("lora_A.weight", "lora_B.weight"))
    }


def save_lora_adapter(model, directory, *, rank, alpha, dropout):
    """Gather only adapter tensors; all distributed ranks must call this."""
    directory = Path(directory)
    tensors = {}
    for name, param in _adapter_parameters(model).items():
        value = param.full_tensor() if isinstance(param, DTensor) else param
        canonical_name = _canonical_adapter_name(name)
        if canonical_name in tensors:
            raise ValueError(f"Duplicate canonical LoRA tensor: {canonical_name}")
        tensors[canonical_name] = value.detach().cpu().contiguous()
    if not tensors:
        raise ValueError("No LoRA parameters found to save")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        directory.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(directory / "adapter.safetensors"))
        (directory / "adapter_config.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "rank": int(rank),
                    "alpha": float(alpha),
                    "dropout": float(dropout),
                    "targets": list(LORA_TARGETS),
                },
                indent=2,
            )
            + "\n"
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_lora_adapter(model, directory):
    """Load wide adapters; legacy target metadata cannot change the scope."""
    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text())
    if config.get("format_version") != 1:
        raise ValueError("Unsupported LoRA adapter format version")
    inject_lora(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config["dropout"],
    )
    raw_tensors = load_file(str(directory / "adapter.safetensors"))
    tensors = {}
    for name, tensor in raw_tensors.items():
        canonical_name = _canonical_adapter_name(name)
        if canonical_name in tensors:
            raise ValueError(f"Duplicate canonical LoRA tensor: {canonical_name}")
        tensors[canonical_name] = tensor
    params = _adapter_parameters(model)
    canonical_params = {
        _canonical_adapter_name(name): param for name, param in params.items()
    }
    if set(tensors) != set(canonical_params):
        raise ValueError(
            "LoRA adapter does not match the fixed wide LoRA scope of the model; "
            "attention-only adapters cannot be loaded as wide adapters"
        )
    with torch.no_grad():
        for name, param in canonical_params.items():
            if tensors[name].shape != param.shape:
                raise ValueError(f"LoRA tensor shape mismatch: {name}")
            param.copy_(tensors[name].to(device=param.device, dtype=param.dtype))
    return {key: config[key] for key in ("rank", "alpha", "dropout")}
