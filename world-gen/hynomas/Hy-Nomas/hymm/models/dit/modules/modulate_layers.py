from typing import Callable
from einops import rearrange

import torch
import torch.nn as nn


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(
            hidden_size, factor * hidden_size, bias=True, **factory_kwargs
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    """Apply gating to the input tensor.

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


def modulate2(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        scale = scale.unsqueeze(0)
        scale = rearrange(scale, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = scale.shape[1]  # latent length
        token_length = x.shape[1] // latent_length
        # operate on the hidden_states
        scale = scale.repeat_interleave(token_length, dim=1).type_as(x)
        return x * (1 + scale)

    elif scale is None:
        shift = shift.unsqueeze(0)
        shift = rearrange(shift, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = shift.shape[1]  # latent length
        token_length = x.shape[1] // latent_length
        # operate on the hidden_states
        shift = shift.repeat_interleave(token_length, dim=1).type_as(x)
        return x + shift

    else:
        shift = shift.unsqueeze(0)
        scale = scale.unsqueeze(0)
        shift = rearrange(shift, "B (N T) C -> (B N) T C", N=x.shape[0])
        scale = rearrange(scale, "B (N T) C -> (B N) T C", N=x.shape[0])

        latent_length = shift.shape[1]  # latent length
        token_length = x.shape[1] // latent_length

        scale = scale.repeat_interleave(token_length, dim=1).type_as(x)
        shift = shift.repeat_interleave(token_length, dim=1).type_as(x)
        return x * (1 + scale) + shift


def apply_gate2(x, gate=None, tanh=False):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    gate = gate.unsqueeze(0)
    gate = rearrange(gate, "B (N T) C -> (B N) T C", N=x.shape[0])
    latent_length = gate.shape[1]  # latent length
    token_length = x.shape[1] // latent_length
    gate = gate.repeat_interleave(token_length, dim=1).type_as(x)
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
