# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# The residual blocks and Encoder are adapted from T2M-GPT through MotionGPT.
# https://github.com/Mael-zys/T2M-GPT
# https://github.com/OpenMotionLab/MotionGPT
# Copyright 2023 tencent.
# NVIDIA modified these layers, including adding token-mask handling.
# NVIDIA additions retain the copyright and Apache-2.0 terms above.
"Transformer embeddings and convolutional motion encoders and decoders."

from __future__ import annotations

import math

import torch
import torch as t
from torch import nn


class FCBlock(t.nn.Module):
    """Stack of fully connected layers with LeakyReLU activations."""

    def __init__(
        self,
        num_layers: int,
        layer_width: int,
        size_in: int,
        size_out: int,
        dropout: float = 0.0,
    ):
        super(FCBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.fc_layers = [t.nn.Linear(size_in, layer_width)]
        self.relu_layers = [t.nn.LeakyReLU(inplace=True)]
        if dropout > 0.0:
            self.fc_layers.append(t.nn.Dropout(p=dropout))
            self.relu_layers.append(t.nn.Identity())
        self.fc_layers += [
            t.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)
        ]
        self.relu_layers += [
            t.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)
        ]
        self.forward_projection = t.nn.Linear(layer_width, size_out)
        self.fc_layers = t.nn.ModuleList(self.fc_layers)
        self.relu_layers = t.nn.ModuleList(self.relu_layers)

    def forward(self, x: t.Tensor):
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        return f


def PE1d_sincos(seq_length, dim):
    """Return a sine and cosine positional encoding of shape [seq_length, dim].

    dim must be even.
    """
    if dim % 2 != 0:
        raise ValueError(
            "Cannot use sine and cosine positional encoding with an odd dim ({:d}).".format(
                dim
            )
        )
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """Add a fixed sine and cosine positional encoding to a sequence."""

    def __init__(self, seq_length, dim, dropout: float = 0.0, grad=False):
        super().__init__()
        self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        l = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


class nonlinearity(nn.Module):
    """SiLU activation, x * sigmoid(x)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """Residual block of two 1D convolutions with an optional norm."""

    def __init__(
        self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None
    ):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-06, affine=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-06, affine=True
            )
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-06, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-06, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor = None):
        x = x * token_mask[:, None, :] if token_mask is not None else x
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
        x = x * token_mask[:, None, :] if token_mask is not None else x
        x = self.conv1(x)
        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        x = x * token_mask[:, None, :] if token_mask is not None else x
        x = self.conv2(x)
        x = x + x_orig
        x = x * token_mask[:, None, :] if token_mask is not None else x
        return x


class Resnet1D(nn.Module):
    """Stack of residual 1D convolution blocks with growing dilation."""

    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor = None):
        for block in self.model:
            x = block(x, token_mask)
        return x


class Encoder(nn.Module):
    """Convolutional encoder that downsamples motion features in time."""

    def __init__(
        self,
        input_emb_width,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = (stride_t * 2, stride_t // 2)
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        input_dim = width
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width, depth, dilation_growth_rate, activation=activation, norm=norm
                ),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DoubleCondDecoder(nn.Module):
    """Convolutional decoder that upsamples motion tokens into motion features.

    It is conditioned on the target poses and on external features.
    """

    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 3,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm: str = None,
        target_cond_dim: int = -1,
        external_cond_dim: int = -1,
        cond_fusion_last_layer=False,
    ):
        super().__init__()
        self._down_t, self._width, self._input_emb_width, self._output_emb_width = (
            down_t,
            width,
            input_emb_width,
            output_emb_width,
        )
        self._target_cond_dim, self._external_cond_dim = (
            target_cond_dim,
            external_cond_dim,
        )
        self._HAS_EXTERNAL_COND, self._HAS_TARGET_COND = (
            external_cond_dim > 0,
            target_cond_dim > 0,
        )
        self._COND_FUSION_LAST_LAYER = cond_fusion_last_layer
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        if self._HAS_EXTERNAL_COND:
            external_cond_blocks = []
            for i in range(down_t + (1 if self._COND_FUSION_LAST_LAYER else 0)):
                cond_feat_dim = 2 ** (down_t - i) * self._external_cond_dim
                external_cond_blocks.append(nn.Linear(cond_feat_dim + width, width))
                external_cond_blocks.append(nn.ReLU())
            self.external_cond_blocks = nn.ModuleList(external_cond_blocks)
        if self._HAS_TARGET_COND:
            target_cond_blocks = []
            assert width % 2**down_t == 0, (
                "width % (2 ** down_t) needs to 0 so that hidden can be split for each frame in earlier layers."
            )
            for i in range(down_t + (1 if self._COND_FUSION_LAST_LAYER else 0)):
                target_cond_blocks.append(
                    nn.Linear(self._target_cond_dim, int(width / 2 ** (down_t - i)))
                )
                target_cond_blocks.append(nn.ReLU())
            self.target_cond_blocks = nn.ModuleList(target_cond_blocks)

    def forward(
        self,
        x: torch.Tensor,
        external_cond: torch.Tensor = None,
        target_cond: torch.Tensor = None,
        has_target_cond: torch.Tensor = None,
        token_mask: torch.Tensor = None,
    ):
        batch_size = x.shape[0]
        x = x * token_mask[:, None, :] if token_mask is not None else x
        h = self.model[0](x)
        h = self.model[1](h)
        for i in range(self._down_t + (1 if self._COND_FUSION_LAST_LAYER else 0)):
            numFrames_per_position = 2 ** (self._down_t - i)
            numPositions = h.shape[-1]
            timesteps = numPositions * numFrames_per_position
            if (
                not self._HAS_TARGET_COND
                or target_cond is None
                or has_target_cond is None
            ):
                pass
            else:
                h_target_cond = self.target_cond_blocks[i * 2](target_cond)
                h_target_cond = self.target_cond_blocks[i * 2 + 1](h_target_cond)
                h = h.transpose(1, 2)
                h = h.reshape(
                    [
                        batch_size,
                        numPositions * numFrames_per_position,
                        self._width // numFrames_per_position,
                    ]
                )
                h = torch.where(has_target_cond[:, :, None], h_target_cond, h)
                h = h.reshape([batch_size, numPositions, self._width]).transpose(1, 2)
            if self._HAS_EXTERNAL_COND:
                assert external_cond is not None and external_cond.shape[1] == timesteps
                h_cond = external_cond.reshape([batch_size, numPositions, -1])
                h = torch.cat([h.transpose(1, 2), h_cond], dim=-1)
                h = self.external_cond_blocks[i * 2](h).transpose(1, 2)
                h = self.external_cond_blocks[i * 2 + 1](h)
            if i == self._down_t:
                continue
            h = self.model[i + 2][0](h, token_mask)
            h = self.model[i + 2][1](h)
            token_mask = (
                token_mask.repeat_interleave(2, 1) if token_mask is not None else None
            )
            h = h * token_mask[:, None, :] if token_mask is not None else h
            h = self.model[i + 2][2](h)
        h = h * token_mask[:, None, :] if token_mask is not None else h
        h = self.model[2 + self._down_t](h)
        h = self.model[2 + self._down_t + 1](h)
        h = h * token_mask[:, None, :] if token_mask is not None else h
        h = self.model[2 + self._down_t + 2](h)
        h = h * token_mask[:, None, :] if token_mask is not None else h
        return h
