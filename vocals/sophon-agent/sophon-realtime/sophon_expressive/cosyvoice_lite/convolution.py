# Vendored from FunAudioLLM/CosyVoice (Apache-2.0, see LICENSE in this directory); inference-only subset, imports adjusted.
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        causal_type: str = 'left',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride=1,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        assert causal_type in ['left', 'right']
        self.causal_type = causal_type

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor]:
        input_timestep = x.shape[2]
        if cache.size(2) == 0:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding).to(x)
        assert cache.size(2) == self.causal_padding
        if self.causal_type == 'left':
            x = torch.concat([cache, x], dim=2)
        else:
            x = torch.concat([x, cache], dim=2)
        x = super(CausalConv1d, self).forward(x)
        assert x.shape[2] == input_timestep
        return x


class CausalConv1dDownSample(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1dDownSample, self).__init__(in_channels, out_channels,
                                                     kernel_size, stride,
                                                     padding=0, dilation=dilation,
                                                     groups=groups, bias=bias,
                                                     padding_mode=padding_mode,
                                                     device=device, dtype=dtype)
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super(CausalConv1dDownSample, self).forward(x)
        return x


class CausalConv1dUpsample(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1dUpsample, self).__init__(in_channels, out_channels,
                                                   kernel_size, 1,
                                                   padding=0, dilation=dilation,
                                                   groups=groups, bias=bias,
                                                   padding_mode=padding_mode,
                                                   device=device, dtype=dtype)
        assert dilation == 1
        self.causal_padding = kernel_size - 1
        self.upsample = torch.nn.Upsample(scale_factor=stride, mode='nearest')

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.upsample(x)
        input_timestep = x.shape[2]
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super(CausalConv1dUpsample, self).forward(x)
        assert input_timestep == x.shape[2]
        return x
