"""
NVIDIA DCAE-V (DC-AE Video) — single-file implementation.

All low-level 3-D convolution ops, encoder/decoder/top-level model, and the
high-level ``AutoencoderDCAEV`` wrapper are consolidated here so no separate
``dcae_v/`` subpackage is needed.

Compression ratios  (f32t4c32 variant)
--------------------------------------
* Temporal  :   4   (T -> T/4)
* Spatial   :  32   (H -> H/32, W -> W/32)
* Latent ch :  32

Checkpoint format
-----------------
A single ``.pt`` file with key ``model_state_dict`` containing an
``OrderedDict`` of ``encoder.*`` / ``decoder.*`` weights in ``bfloat16``.
There is **no** embedded config — the architecture is hard-coded via
``dc_ae_v_f32t4c32_config()``.

Usage
-----
>>> vae = AutoencoderDCAEV.from_checkpoint("/path/to/dc-ae-v-1.0-f32t4c32-bf16.pt")
>>> vae = vae.to("cuda").eval()
>>>
>>> # encode:  [B, 3, T, H, W]  float in [-1, 1]
>>> latent = vae.encode(video).latent_dist.mode()
>>> # decode:  [B, 32, T', H', W']  ->  [B, 3, T, H, W]  float in [-1, 1]
>>> recon  = vae.decode(latent).sample
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================================== #
#  Utilities
# =========================================================================== #
def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, (tuple, list)):
        return tuple(get_same_padding(ks) for ks in kernel_size)
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def val2list(x: list | tuple | Any, repeat_time: int = 1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights


# =========================================================================== #
#  Activation / Normalization registries
# =========================================================================== #
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
    "mish": nn.Mish,
    "linear": nn.Identity,
    "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
}


def build_act(name: Optional[str]) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        return REGISTERED_ACT_DICT[name]()
    return None


class RMSNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = x / torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        x = x.permute(0, 4, 1, 2, 3)
        return x


REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "rms3d": RMSNorm3d,
}


def build_norm(
    name: Optional[str] = "bn2d", num_features=None, **kwargs
) -> Optional[nn.Module]:
    if name in ("ln", "rms3d"):
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        return REGISTERED_NORM_DICT[name](**kwargs)
    return None


# =========================================================================== #
#  Basic blocks
# =========================================================================== #
class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            return x
        if self.shortcut is None:
            return self.forward_main(x)
        res = self.forward_main(x) + self.shortcut(x)
        if self.post_act:
            res = self.post_act(res)
        return res


# =========================================================================== #
#  Custom Conv3d (handles large tensors that exceed int32 element count)
# =========================================================================== #
def _conv3d_split_channel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    num_in_channel_chunks: int,
    num_out_channel_chunks: int,
) -> torch.Tensor:
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    in_channels_per_split = in_channels // num_in_channel_chunks
    out_channels_per_split = out_channels // num_out_channel_chunks

    output = []
    for i in range(num_out_channel_chunks):
        oc_s = i * out_channels_per_split
        oc_e = (i + 1) * out_channels_per_split
        output_i: torch.Tensor | int = 0
        for j in range(num_in_channel_chunks):
            ic_s = j * in_channels_per_split
            ic_e = (j + 1) * in_channels_per_split
            output_i = output_i + F.conv3d(
                x[:, ic_s:ic_e],
                weight[oc_s:oc_e, ic_s:ic_e],
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
            )
        output.append(output_i)
    result = torch.cat(output, dim=1)
    if bias is not None:
        result = result + bias[:, None, None, None]
    return result


def _custom_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    groups: int,
) -> torch.Tensor:
    input_numel = input[0].numel()
    output_numel = (
        weight.shape[0]
        * (input.shape[2] // stride[0])
        * (input.shape[3] // stride[1])
        * (input.shape[4] // stride[2])
    )
    if (input_numel >= 1 << 31 or output_numel >= 1 << 31) and groups == 1:
        nic, noc = 1, 1
        while input_numel // nic >= 1 << 31:
            nic *= 2
        while output_numel // noc >= 1 << 31:
            noc *= 2
        return _conv3d_split_channel(
            input, weight, bias, stride, padding, dilation, nic, noc
        )
    return F.conv3d(input, weight, bias, stride, padding, dilation, groups)


class CustomConv3d(nn.Conv3d):
    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        padding: Optional[tuple[int, ...]] = None,
    ):
        assert self.padding_mode == "zeros"
        return _custom_conv3d(
            input,
            weight,
            bias,
            self.stride,
            padding or self.padding,
            self.dilation,
            self.groups,
        )

    def forward(
        self, input: torch.Tensor, padding: Optional[tuple[int, ...]] = None
    ) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias, padding)


# =========================================================================== #
#  ConvLayer3d – core 3D conv with optional norm/act and chunk-causal support
# =========================================================================== #
class ConvLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        groups: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
        zero_out: bool = False,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = val2tuple(kernel_size, 3)
        stride = val2tuple(stride, 3)
        padding = get_same_padding(kernel_size)
        self.causal = causal
        self.causal_chunk_length = causal_chunk_length

        if causal:
            self.custom_padding = (0, 0, 0, 0, 2 * padding[0], 0)
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = (
                "constant"
                if temporal_padding_mode == "zeros"
                else temporal_padding_mode
            )
        elif causal_chunk_length is not None:
            self.custom_padding = None
            self.custom_padding_mode = None
        elif spatial_padding_mode != temporal_padding_mode:
            self.custom_padding = (0, 0, 0, 0, padding[0], padding[0])
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = (
                "constant"
                if temporal_padding_mode == "zeros"
                else temporal_padding_mode
            )
        else:
            self.custom_padding = None
            self.custom_padding_mode = None

        self.conv = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=use_bias,
            padding_mode=spatial_padding_mode,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

        self.zero_out = zero_out
        if zero_out:
            if self.norm:
                self.norm.zero_out()
            else:
                nn.init.constant_(self.conv.weight, 0)
                nn.init.constant_(self.conv.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        if self.custom_padding is not None:
            x = F.pad(x, self.custom_padding, mode=self.custom_padding_mode)

        if (
            self.causal_chunk_length is not None
            and x.shape[2] % self.causal_chunk_length == 0
        ):
            B, C, T, H, W = x.shape
            cl = self.causal_chunk_length
            tp = self.conv.padding[0]
            x = x.reshape(B, C, T // cl, cl, H, W).transpose(1, 2)

            if feature_cache is not None:
                first_left_pad = feature_cache.get(feature_key)
                feature_cache[feature_key] = x[:, -1:, :, -tp:].clone()
            else:
                first_left_pad = None

            if first_left_pad is None:
                first_left_pad = torch.zeros(
                    (B, 1, C, tp, H, W), dtype=x.dtype, device=x.device
                )
            else:
                if first_left_pad.shape[3] < tp:
                    first_left_pad = torch.cat(
                        [
                            torch.zeros(
                                (B, 1, C, tp - first_left_pad.shape[3], H, W),
                                dtype=x.dtype,
                                device=x.device,
                            ),
                            first_left_pad,
                        ],
                        dim=3,
                    )

            left_pad = torch.cat([first_left_pad, x[:, :-1, :, -tp:]], dim=1)
            right_pad = torch.zeros(
                (B, T // cl, C, tp, H, W), dtype=x.dtype, device=x.device
            )
            x = torch.cat([left_pad, x, right_pad], dim=3)
            x = x.reshape(B * (T // cl), C, cl + 2 * tp, H, W)
            x = self.conv(x, (0, self.conv.padding[1], self.conv.padding[2]))
            x = (
                x.reshape(B, T // cl, -1, cl, H, W)
                .transpose(1, 2)
                .reshape(B, -1, T, H, W)
            )
        else:
            x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# =========================================================================== #
#  ResBlock3d
# =========================================================================== #
class ResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple[bool, bool] = False,
        norm: str | tuple[Optional[str], Optional[str]] = ("bn2d", "bn2d"),
        act_func: str | tuple[Optional[str], Optional[str]] = ("relu6", None),
        zero_out: bool = False,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.conv1 = ConvLayer3d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )
        self.conv2 = ConvLayer3d(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            zero_out=zero_out,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv1(
            x, feature_cache, feature_key + "conv1." if feature_key else None
        )
        x = self.conv2(
            x, feature_cache, feature_key + "conv2." if feature_key else None
        )
        return x


# =========================================================================== #
#  Pixel-shuffle / unshuffle helpers
# =========================================================================== #
def pixel_unshuffle_3d(
    x: torch.Tensor, spatial_factor: int, temporal_factor: int
) -> torch.Tensor:
    B, C, T, H, W = x.shape
    x = (
        x.reshape(
            B,
            C,
            T // temporal_factor,
            temporal_factor,
            H // spatial_factor,
            spatial_factor,
            W // spatial_factor,
            spatial_factor,
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(
            B,
            C * temporal_factor * spatial_factor**2,
            T // temporal_factor,
            H // spatial_factor,
            W // spatial_factor,
        )
    )
    return x


def pixel_shuffle_3d(
    x: torch.Tensor, spatial_factor: int, temporal_factor: int
) -> torch.Tensor:
    B, C, T, H, W = x.shape
    x = (
        x.reshape(
            B,
            C // temporal_factor // spatial_factor**2,
            temporal_factor,
            spatial_factor,
            spatial_factor,
            T,
            H,
            W,
        )
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(
            B,
            C // temporal_factor // spatial_factor**2,
            T * temporal_factor,
            H * spatial_factor,
            W * spatial_factor,
        )
    )
    return x


# =========================================================================== #
#  Down-sample / Up-sample layers
# =========================================================================== #
class ConvPixelUnshuffleDownSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        spatial_factor: int,
        temporal_factor: int,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        zero_out: bool = False,
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        out_ratio = spatial_factor**2 * temporal_factor
        self.conv = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            zero_out=zero_out,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv(x, feature_cache, feature_key + "conv." if feature_key else None)
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int,
        temporal_factor: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        self.group_size = (
            in_channels * spatial_factor**2 * temporal_factor // out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        B, C, T, H, W = x.shape
        return x.view(B, self.out_channels, self.group_size, T, H, W).mean(dim=2)


class ConvPixelShuffleUpSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        spatial_factor: int,
        temporal_factor: int,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        zero_out: bool = False,
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        out_ratio = spatial_factor**2 * temporal_factor
        self.conv = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            zero_out=zero_out,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv(x, feature_cache, feature_key + "conv." if feature_key else None)
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class ChannelDuplicatingPixelShuffleUpSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int,
        temporal_factor: int,
    ):
        super().__init__()
        self.repeats = out_channels * spatial_factor**2 * temporal_factor // in_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        return pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)


# =========================================================================== #
#  3D-aware sequential / residual wrappers
# =========================================================================== #
class ResidualBlock3d(ResidualBlock):
    def forward_main(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        fk = feature_key + "main." if feature_key else None
        if self.pre_norm is None:
            return self.main(x, feature_cache, fk)
        return self.main(self.pre_norm(x), feature_cache, fk)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        if self.main is None:
            return x
        if self.shortcut is None:
            return self.forward_main(x, feature_cache, feature_key)
        res = self.forward_main(x, feature_cache, feature_key) + self.shortcut(x)
        if self.post_act:
            res = self.post_act(res)
        return res


class OpSequential3d(OpSequential):
    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        for i, op in enumerate(self.op_list):
            if isinstance(
                op, (ConvLayer3d, ResidualBlock3d, ConvPixelShuffleUpSampleLayer3d)
            ):
                x = op(
                    x,
                    feature_cache,
                    feature_key + f"op_list.{i}." if feature_key else None,
                )
            else:
                x = op(x)
        return x


# =========================================================================== #
#  Dataclass configs
# =========================================================================== #
@dataclass
class DCAEVEncoderConfig:
    in_channels: int = 3
    latent_channels: int = 32
    project_in_block_type: str = "ChunkCausalConvPixelUnshuffle@2@1@3@3@40"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024, 1024)
    depth_list: tuple[int, ...] = (0, 5, 10, 4, 4, 4, 4)
    block_type: Any = "ChunkCausalResBlock3d@3@3@40"
    norm: Any = "rms3d"
    act: str = "silu"
    downsample_block_type: Any = "ChunkCausalConvPixelUnshuffle@2@1@3@3@40"
    downsample_shortcut: Optional[str] = "averaging"
    project_out_block_type: str = "ChunkCausalConvLayer3d@3@3@10"
    zero_out: bool = False


@dataclass
class DCAEVDecoderConfig:
    in_channels: int = 3
    latent_channels: int = 32
    project_in_block_type: str = "ChunkCausalConvLayer3d@3@3@10"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024, 1024)
    depth_list: tuple[int, ...] = (0, 5, 10, 4, 4, 4, 4)
    block_type: Any = "ChunkCausalResBlock3d@3@3@40"
    norm: Any = "rms3d"
    act: Any = "silu"
    upsample_block_type: Any = "ChunkCausalConvPixelShuffle@2@1@3@3@40"
    upsample_shortcut: str = "duplicating"
    project_out_block_type: str = "ChunkCausalConvPixelShuffle@2@1@3@3@40"
    out_norm: str = "rms3d"
    out_act: str = "silu"
    zero_out: bool = False


@dataclass
class DCAEVConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: DCAEVEncoderConfig = field(default_factory=DCAEVEncoderConfig)
    decoder: DCAEVDecoderConfig = field(default_factory=DCAEVDecoderConfig)

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae-v-fsdp"
    pretrained_ema: bool = True
    zero_out: bool = False
    use_feature_cache: bool = True

    encode_temporal_tile_size: Optional[int] = 40
    encode_temporal_tile_latent_size: Optional[int] = 10
    decode_temporal_tile_size: Optional[int] = 40
    decode_temporal_tile_latent_size: Optional[int] = 10
    encode_temporal_tile_overlap_factor: float = 0.0
    decode_temporal_tile_overlap_factor: float = 0.0

    spatial_tile_size: Optional[int] = None
    spatial_tile_overlap_factor: float = 0.25

    num_pad_frames: int = 0
    temporal_remainder: int = 0
    verbose: bool = False


# =========================================================================== #
#  Builder helpers
# =========================================================================== #
def _build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    zero_out: bool = False,
) -> nn.Module:
    cfg = block_type.split("@")
    block_name, spatial_factor, temporal_factor = cfg[0], int(cfg[1]), int(cfg[2])
    if block_name in ("ConvPixelUnshuffle", "CausalConvPixelUnshuffle"):
        kwargs = {}
        if len(cfg) >= 7:
            kwargs["spatial_padding_mode"] = cfg[5]
            kwargs["temporal_padding_mode"] = cfg[6]
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelUnshuffle",
            **kwargs,
        )
    elif block_name == "ChunkCausalConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            causal_chunk_length=int(cfg[5]),
        )
    else:
        raise ValueError(f"downsample block_name {block_name} is not supported")

    if shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    elif shortcut is not None:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def _build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    zero_out: bool = False,
) -> nn.Module:
    cfg = block_type.split("@")
    block_name, spatial_factor, temporal_factor = cfg[0], int(cfg[1]), int(cfg[2])
    if block_name in ("ConvPixelShuffle", "CausalConvPixelShuffle"):
        kwargs = {}
        if len(cfg) >= 7:
            kwargs["spatial_padding_mode"] = cfg[5]
            kwargs["temporal_padding_mode"] = cfg[6]
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelShuffle",
            **kwargs,
        )
    elif block_name == "ChunkCausalConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            zero_out=zero_out,
            causal_chunk_length=int(cfg[5]),
        )
    else:
        raise ValueError(f"upsample block_name {block_name} is not supported")

    if shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    elif shortcut is not None:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def _build_block(
    block_type: str,
    channels: int,
    norm: Optional[str],
    act: Optional[str],
    zero_out: bool,
) -> nn.Module:
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name in ("ResBlock3d", "CausalResBlock3d"):
        kwargs = {}
        if len(cfg) >= 5:
            kwargs["spatial_padding_mode"] = cfg[3]
            kwargs["temporal_padding_mode"] = cfg[4]
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal=block_name == "CausalResBlock3d",
            **kwargs,
        )
        return ResidualBlock3d(main_block, IdentityLayer())
    elif block_name == "ChunkCausalResBlock3d":
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal_chunk_length=int(cfg[3]),
        )
        return ResidualBlock3d(main_block, IdentityLayer())
    raise ValueError(f"block_name {block_name} is not supported")


def _build_stage_main(
    width: int,
    depth: int,
    block_type: str | list[str],
    norm: str,
    act: str,
    zero_out: bool = False,
) -> list[nn.Module]:
    stage = []
    for d in range(depth):
        bt = block_type[d] if isinstance(block_type, list) else block_type
        stage.append(_build_block(bt, width, norm, act, zero_out))
    return stage


def _build_encoder_project_in(block_type: str, in_channels: int, out_channels: int):
    return _build_downsample_block(block_type, in_channels, out_channels, shortcut=None)


def _build_encoder_project_out(block_type: str, in_channels: int, out_channels: int):
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name in ("ConvLayer3d", "CausalConvLayer3d"):
        kwargs = {}
        if len(cfg) >= 5:
            kwargs["spatial_padding_mode"] = cfg[3]
            kwargs["temporal_padding_mode"] = cfg[4]
        return ConvLayer3d(
            in_channels,
            out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name == "ChunkCausalConvLayer3d":
        return ConvLayer3d(
            in_channels,
            out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=int(cfg[3]),
        )
    raise ValueError(f"encoder project out block {block_name} not supported")


def _build_decoder_project_in(block_type: str, in_channels: int, out_channels: int):
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name in ("ConvLayer3d", "CausalConvLayer3d"):
        kwargs = {}
        if len(cfg) >= 5:
            kwargs["spatial_padding_mode"] = cfg[3]
            kwargs["temporal_padding_mode"] = cfg[4]
        return ConvLayer3d(
            in_channels,
            out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name == "ChunkCausalConvLayer3d":
        return ConvLayer3d(
            in_channels,
            out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=int(cfg[3]),
        )
    raise ValueError(f"decoder project in block {block_name} not supported")


def _build_decoder_project_out(
    block_type: str,
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
        _build_upsample_block(block_type, in_channels, out_channels, shortcut=None),
    ]
    return OpSequential3d(layers)


# =========================================================================== #
#  Encoder / Decoder
# =========================================================================== #
class DCAEVEncoder(nn.Module):
    def __init__(self, cfg: DCAEVEncoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)

        self.project_in = _build_encoder_project_in(
            cfg.project_in_block_type,
            cfg.in_channels,
            cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
        )

        self.stages: nn.ModuleList = nn.ModuleList()
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            bt = (
                cfg.block_type[stage_id]
                if isinstance(cfg.block_type, list)
                else cfg.block_type
            )
            nm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            stage = _build_stage_main(width, depth, bt, nm, cfg.act, cfg.zero_out)
            if stage_id < num_stages - 1 and depth > 0:
                dbt = (
                    cfg.downsample_block_type[stage_id]
                    if isinstance(cfg.downsample_block_type, list)
                    else cfg.downsample_block_type
                )
                stage.append(
                    _build_downsample_block(
                        dbt,
                        width,
                        cfg.width_list[stage_id + 1],
                        cfg.downsample_shortcut,
                        cfg.zero_out,
                    )
                )
            self.stages.append(OpSequential3d(stage))

        self.project_out = _build_encoder_project_out(
            cfg.project_out_block_type, cfg.width_list[-1], cfg.latent_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.project_in(
            x, feature_cache, feature_key + "project_in." if feature_key else None
        )
        for sid, stage in enumerate(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(
                x,
                feature_cache,
                feature_key + f"stages.{sid}." if feature_key else None,
            )
        x = self.project_out(
            x, feature_cache, feature_key + "project_out." if feature_key else None
        )
        return x


class DCAEVDecoder(nn.Module):
    def __init__(self, cfg: DCAEVDecoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)

        self.project_in = _build_decoder_project_in(
            cfg.project_in_block_type, cfg.latent_channels, cfg.width_list[-1]
        )

        self.spatial_compression_ratio = 1
        self.temporal_compression_ratio = 1

        stages_list: list[OpSequential3d] = []
        for stage_id, (width, depth) in reversed(
            list(enumerate(zip(cfg.width_list, cfg.depth_list)))
        ):
            stage: list[nn.Module] = []
            if stage_id < num_stages - 1 and depth > 0:
                ubt = (
                    cfg.upsample_block_type[stage_id]
                    if isinstance(cfg.upsample_block_type, list)
                    else cfg.upsample_block_type
                )
                stage.append(
                    _build_upsample_block(
                        ubt,
                        cfg.width_list[stage_id + 1],
                        width,
                        cfg.upsample_shortcut,
                        cfg.zero_out,
                    )
                )
                self.spatial_compression_ratio *= int(ubt.split("@")[1])
                self.temporal_compression_ratio *= int(ubt.split("@")[2])
            bt = (
                cfg.block_type[stage_id]
                if isinstance(cfg.block_type, list)
                else cfg.block_type
            )
            nm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            ac = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(_build_stage_main(width, depth, bt, nm, ac, cfg.zero_out))
            stages_list.insert(0, OpSequential3d(stage))

        self.stages = nn.ModuleList(stages_list)

        self.project_out = _build_decoder_project_out(
            cfg.project_out_block_type,
            cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            cfg.in_channels,
            cfg.out_norm,
            cfg.out_act,
        )
        self.spatial_compression_ratio *= int(cfg.project_out_block_type.split("@")[1])
        self.temporal_compression_ratio *= int(cfg.project_out_block_type.split("@")[2])

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.project_in(
            x, feature_cache, feature_key + "project_in." if feature_key else None
        )
        for sid, stage in reversed(list(enumerate(self.stages))):
            if len(stage.op_list) == 0:
                continue
            x = stage(
                x,
                feature_cache,
                feature_key + f"stages.{sid}." if feature_key else None,
            )
        x = self.project_out(
            x, feature_cache, feature_key + "project_out." if feature_key else None
        )
        return x


# =========================================================================== #
#  Top-level DCAEV model (internal, used by the wrapper)
# =========================================================================== #
class DCAEV(nn.Module):
    def __init__(self, cfg: DCAEVConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = DCAEVEncoder(cfg.encoder)
        self.decoder = DCAEVDecoder(cfg.decoder)

    @property
    def spatial_compression_ratio(self) -> int:
        return self.decoder.spatial_compression_ratio

    @property
    def temporal_compression_ratio(self) -> int:
        return self.decoder.temporal_compression_ratio

    def _blend_t(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            r = x / blend_extent
            b[:, :, x, :, :] = (
                a[:, :, -blend_extent + x, :, :] * (1 - r) + b[:, :, x, :, :] * r
            )
        return b

    def temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        tile_size = self.cfg.encode_temporal_tile_size
        latent_size = self.cfg.encode_temporal_tile_latent_size
        overlap_size = int(
            tile_size * (1 - self.cfg.encode_temporal_tile_overlap_factor)
        )
        blend_extent = int(latent_size * self.cfg.encode_temporal_tile_overlap_factor)
        t_limit = latent_size - blend_extent

        feature_cache = {} if self.cfg.use_feature_cache else None
        row = []
        for i in range(0, x.shape[2], overlap_size):
            tile = x[:, :, i : i + tile_size, :, :]
            tile = self.encoder(
                tile, feature_cache, "encoder." if self.cfg.use_feature_cache else None
            )
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self._blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        return torch.cat(result_row, dim=2)

    def temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        latent_size = self.cfg.decode_temporal_tile_latent_size
        tile_size = self.cfg.decode_temporal_tile_size
        overlap_size = int(
            latent_size * (1 - self.cfg.decode_temporal_tile_overlap_factor)
        )
        blend_extent = int(tile_size * self.cfg.decode_temporal_tile_overlap_factor)
        t_limit = tile_size - blend_extent

        feature_cache = {} if self.cfg.use_feature_cache else None
        row = []
        for i in range(0, z.shape[2], overlap_size):
            tile = z[:, :, i : i + latent_size, :, :]
            decoded = self.decoder(
                tile, feature_cache, "decoder." if self.cfg.use_feature_cache else None
            )
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self._blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        return torch.cat(result_row, dim=2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.num_pad_frames > 0:
            x = F.pad(x, (0, 0, 0, 0, self.cfg.num_pad_frames, 0), mode="replicate")
        if self.cfg.encode_temporal_tile_size is not None:
            return self.temporal_tiled_encode(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.decode_temporal_tile_size is not None:
            out = self.temporal_tiled_decode(z)
        else:
            out = self.decoder(z)
        if self.cfg.num_pad_frames > 0:
            out = out[:, :, self.cfg.num_pad_frames :, :, :]
        return out


class DCAEVStreamingDecodeSession:
    """Stateful exact temporal-tile decoder for incrementally arriving latents.

    Production DCAE-V decodes non-overlapping groups of ten latent frames while
    carrying a causal feature cache between groups. This session preserves that
    global tile grid across arbitrary ``push`` calls, so segment boundaries do
    not reset temporal state or change decoded pixels.
    """

    def __init__(self, model: DCAEV):
        self.model = model
        cfg = model.cfg
        self.tile_latent_frames = cfg.decode_temporal_tile_latent_size
        if cfg.decode_temporal_tile_size is None:
            raise ValueError(
                "Streaming DCAE decode requires temporal tiling to be enabled."
            )
        if self.tile_latent_frames != 10:
            raise ValueError(
                "Streaming DCAE decode requires the production global "
                "10-latent tile configuration, got "
                f"{self.tile_latent_frames!r}."
            )
        if cfg.decode_temporal_tile_overlap_factor != 0.0:
            raise ValueError(
                "Streaming DCAE decode currently requires non-overlapping "
                "temporal tiles (decode_temporal_tile_overlap_factor=0)."
            )
        if int(cfg.num_pad_frames or 0) != 0:
            raise ValueError(
                "Streaming DCAE decode requires num_pad_frames=0 so latent and "
                "pixel cursors remain globally aligned."
            )

        self.feature_cache = {} if cfg.use_feature_cache else None
        self._pending: Optional[torch.Tensor] = None
        self._sample_shape: Optional[Tuple[int, int, int, int]] = None
        self._finished = False
        self.received_latent_frames = 0
        self.decoded_latent_frames = 0
        self.speculative_cache_roundtrips = 0
        self.speculative_cache_offload_bytes = 0
        self.speculative_gpu_cache_previews = 0

    @property
    def pending_latent_frames(self) -> int:
        return 0 if self._pending is None else int(self._pending.shape[2])

    def _validate(self, latents: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("Cannot push latents after streaming decode finish().")
        if latents.ndim != 5:
            raise ValueError(
                "Streaming DCAE decode expects [B,C,T,H,W] latents, "
                f"got shape {tuple(latents.shape)}."
            )
        sample_shape = (
            int(latents.shape[0]),
            int(latents.shape[1]),
            int(latents.shape[3]),
            int(latents.shape[4]),
        )
        if self._sample_shape is None:
            self._sample_shape = sample_shape
        elif sample_shape != self._sample_shape:
            raise ValueError(
                "Streaming DCAE latent shape changed across pushes: "
                f"expected {self._sample_shape}, got {sample_shape}."
            )

    @torch.no_grad()
    def push(
        self,
        latents: torch.Tensor,
        output_callback: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> List[torch.Tensor]:
        """Decode newly completed tiles and retain any incomplete suffix.

        When ``output_callback`` is provided, each tile is handed off
        immediately and the returned list stays empty, bounding decoded GPU
        memory independently of the number of completed tiles.
        """
        self._validate(latents)
        if latents.shape[2] == 0:
            return []

        self.received_latent_frames += int(latents.shape[2])
        buffered = (
            latents
            if self._pending is None
            else torch.cat([self._pending, latents], dim=2)
        )
        outputs: List[torch.Tensor] = []
        cursor = 0
        while buffered.shape[2] - cursor >= self.tile_latent_frames:
            tile = buffered[:, :, cursor : cursor + self.tile_latent_frames, :, :]
            decoded = self.model.decoder(
                tile,
                self.feature_cache,
                "decoder." if self.model.cfg.use_feature_cache else None,
            )
            if output_callback is None:
                outputs.append(decoded)
            else:
                output_callback(decoded)
            cursor += self.tile_latent_frames
            self.decoded_latent_frames += self.tile_latent_frames

        remainder = buffered[:, :, cursor:, :, :]
        # Clone the small suffix so it does not retain storage for the complete
        # segment tensor after the caller moves on to the next DiT segment.
        self._pending = remainder.clone() if remainder.shape[2] else None
        return outputs

    @torch.no_grad()
    def speculative_decode_pending(
        self,
        output_callback: Optional[Callable[[torch.Tensor], None]] = None,
        *,
        offload_canonical_cache: bool = True,
    ) -> Tuple[List[torch.Tensor], int]:
        """Preview the pending suffix with repeat-last padding.

        The canonical feature cache and pending latents are left untouched.
        With CPU offload, preview replaces GPU cache entries in-place and then
        restores canonical state. Without offload, writes land in a disposable
        shallow GPU cache copy. Only real-latent pixel frames are returned.
        """
        if self._finished:
            raise RuntimeError(
                "Cannot speculatively decode after streaming decode finish()."
            )
        valid_latent_frames = self.pending_latent_frames
        if valid_latent_frames == 0:
            return [], 0
        if valid_latent_frames >= self.tile_latent_frames:
            raise RuntimeError(
                "Canonical push() must consume complete tiles before "
                "speculative padding."
            )

        pad_frames = self.tile_latent_frames - valid_latent_frames
        padding = self._pending[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
        padded = torch.cat([self._pending, padding], dim=2)
        canonical_cache_cpu = None
        disposable_gpu_cache = None
        canonical_device = self._pending.device
        if self.feature_cache is not None and offload_canonical_cache:
            # A shallow disposable GPU cache doubles the DCAE cache peak and
            # OOMs while the AR KV cache is resident. Snapshot canonical state
            # to CPU, let preview decode replace entries in-place, then drop
            # preview state before restoring the canonical tensors to GPU.
            canonical_cache_cpu = {
                key: value.to(device="cpu", copy=True)
                for key, value in self.feature_cache.items()
            }
            self.speculative_cache_roundtrips += 1
            self.speculative_cache_offload_bytes += sum(
                value.numel() * value.element_size()
                for value in canonical_cache_cpu.values()
            )
        elif self.feature_cache is not None:
            disposable_gpu_cache = dict(self.feature_cache)
            self.speculative_gpu_cache_previews += 1

        decode_cache = (
            self.feature_cache if offload_canonical_cache else disposable_gpu_cache
        )

        decoded = None
        outputs: List[torch.Tensor] = []
        try:
            decoded = self.model.decoder(
                padded,
                decode_cache,
                "decoder." if self.model.cfg.use_feature_cache else None,
            )
            if decoded.shape[2] % self.tile_latent_frames != 0:
                raise RuntimeError(
                    "DCAE speculative tile output is not divisible by its "
                    "latent tile length: "
                    f"output_frames={decoded.shape[2]}, "
                    f"tile_latents={self.tile_latent_frames}."
                )
            pixel_frames_per_latent = int(decoded.shape[2]) // self.tile_latent_frames
            decoded = decoded[
                :, :, : valid_latent_frames * pixel_frames_per_latent, :, :
            ]

            if output_callback is not None:
                output_callback(decoded)
                decoded = None
            else:
                outputs.append(decoded)
        finally:
            if canonical_cache_cpu is not None:
                preview_cache = self.feature_cache
                self.feature_cache = None
                preview_cache.clear()
                del preview_cache
                if canonical_device.type == "cuda":
                    torch.cuda.empty_cache()
                self.feature_cache = {
                    key: value.to(device=canonical_device)
                    for key, value in canonical_cache_cpu.items()
                }
                canonical_cache_cpu.clear()
            elif disposable_gpu_cache is not None:
                disposable_gpu_cache.clear()

        return outputs, valid_latent_frames

    @torch.no_grad()
    def finish(
        self,
        output_callback: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> List[torch.Tensor]:
        """Decode the final incomplete tile and close the session."""
        if self._finished:
            raise RuntimeError("Streaming DCAE decode finish() called twice.")
        self._finished = True
        if self._pending is None:
            return []

        pending = self._pending
        self._pending = None
        decoded = self.model.decoder(
            pending,
            self.feature_cache,
            "decoder." if self.model.cfg.use_feature_cache else None,
        )
        self.decoded_latent_frames += int(pending.shape[2])
        if output_callback is not None:
            output_callback(decoded)
            return []
        return [decoded]


# =========================================================================== #
#  Config factory for dc-ae-v-1.0-f32t4c32-bf16
# =========================================================================== #
def dc_ae_v_f32t4c32_config() -> DCAEVConfig:
    """Return a ``DCAEVConfig`` for the ``dc-ae-v-1.0-f32t4c32-bf16`` variant."""
    chunk_size = 40
    latent_chunk = chunk_size // 4

    enc_cfg = DCAEVEncoderConfig(
        in_channels=3,
        latent_channels=32,
        project_in_block_type=f"ChunkCausalConvPixelUnshuffle@2@1@3@3@{chunk_size}",
        width_list=(128, 256, 512, 512, 1024, 1024, 1024),
        depth_list=(0, 5, 10, 4, 4, 4, 4),
        block_type=[
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{latent_chunk}",
        ],
        norm="rms3d",
        act="silu",
        downsample_block_type=[
            f"ChunkCausalConvPixelUnshuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelUnshuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelUnshuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelUnshuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelUnshuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelUnshuffle@1@4@3@3@{chunk_size}",
        ],
        downsample_shortcut="averaging",
        project_out_block_type=f"ChunkCausalConvLayer3d@3@3@{latent_chunk}",
        zero_out=False,
    )

    dec_cfg = DCAEVDecoderConfig(
        in_channels=3,
        latent_channels=32,
        project_in_block_type=f"ChunkCausalConvLayer3d@3@3@{latent_chunk}",
        width_list=(128, 256, 512, 512, 1024, 1024, 1024),
        depth_list=(0, 5, 10, 4, 4, 4, 4),
        block_type=[
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{chunk_size}",
            f"ChunkCausalResBlock3d@3@3@{latent_chunk}",
        ],
        norm="rms3d",
        act="silu",
        upsample_block_type=[
            f"ChunkCausalConvPixelShuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelShuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelShuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelShuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelShuffle@2@1@3@3@{chunk_size}",
            f"ChunkCausalConvPixelShuffle@1@4@3@3@{latent_chunk}",
        ],
        upsample_shortcut="duplicating",
        project_out_block_type=f"ChunkCausalConvPixelShuffle@2@1@3@3@{chunk_size}",
        out_norm="rms3d",
        out_act="silu",
        zero_out=False,
    )

    return DCAEVConfig(
        in_channels=3,
        latent_channels=32,
        encoder=enc_cfg,
        decoder=dec_cfg,
        pretrained_source="dc-ae-v-fsdp",
        use_feature_cache=True,
        encode_temporal_tile_size=chunk_size,
        encode_temporal_tile_latent_size=latent_chunk,
        decode_temporal_tile_size=chunk_size,
        decode_temporal_tile_latent_size=latent_chunk,
        spatial_tile_size=None,
    )


# =========================================================================== #
#  Lightweight output containers (mirror diffusers-style API)
# =========================================================================== #
@dataclass
class _LatentDist:
    """Deterministic stand-in for ``DiagonalGaussianDistribution``."""

    _latent: torch.Tensor

    def mode(self) -> torch.Tensor:
        return self._latent

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self._latent


@dataclass
class _EncodeOutput:
    latent_dist: _LatentDist


@dataclass
class _DecodeOutput:
    sample: torch.Tensor


# =========================================================================== #
#  Main wrapper — public API
# =========================================================================== #
class AutoencoderDCAEV(nn.Module):
    """Drop-in replacement backed by NVIDIA DCAE-V (f32t4c32).

    Tiling API stubs are kept so callers that toggle tiling on other VAE
    backends do not crash.  DCAE-V uses chunk-causal temporal convolutions
    internally, so no spatial / temporal tiling is required.
    """

    spatial_compression_ratio: int = 32
    time_compression_ratio: int = 4
    scaling_factor: float = 1.0

    ffactor_spatial: int = 32
    ffactor_temporal: int = 4

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def __init__(self, model: DCAEV):
        super().__init__()
        self.model = model
        self.latent_channels = model.cfg.latent_channels

        from types import SimpleNamespace

        self.config = SimpleNamespace(
            latent_channels=self.latent_channels,
            scaling_factor=self.scaling_factor,
            spatial_compression_ratio=self.spatial_compression_ratio,
            time_compression_ratio=self.time_compression_ratio,
            block_out_channels=[1] * self.spatial_compression_ratio.bit_length(),
        )

        self.use_slicing = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> "AutoencoderDCAEV":
        """Instantiate from a ``dc-ae-v-1.0-f32t4c32-bf16.pt`` file."""
        device = torch.device(device) if isinstance(device, str) else device

        cfg = dc_ae_v_f32t4c32_config()
        inner = DCAEV(cfg)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)

        if "ema_model_state_dict" in ckpt:
            ema_sd = ckpt["ema_model_state_dict"]
            if isinstance(ema_sd, dict) and len(ema_sd) == 1:
                state_dict = next(iter(ema_sd.values()))
            else:
                state_dict = ema_sd

        inner.load_state_dict(state_dict, strict=True)
        inner = inner.to(dtype=dtype, device=device).eval()

        wrapper = cls(model=inner)
        wrapper = wrapper.to(dtype=dtype, device=device)
        return wrapper

    # -- Tiling stubs (no-ops) --
    def enable_tiling(self, use_tiling: bool = True):
        pass

    def disable_tiling(self):
        pass

    def enable_spatial_tiling(self, use_tiling: bool = True):
        pass

    def disable_spatial_tiling(self):
        pass

    def enable_temporal_tiling(self, use_tiling: bool = True):
        pass

    def disable_temporal_tiling(self):
        pass

    def enable_tiling_during_training(self, use_tiling: bool = True):
        pass

    def disable_tiling_during_training(self):
        pass

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def random_reset_tiling(self, x: torch.Tensor):
        pass

    def create_streaming_decode_session(self) -> DCAEVStreamingDecodeSession:
        """Create an exact stateful decoder over the wrapped DCAE-V model."""
        return DCAEVStreamingDecodeSession(self.model)

    # -- encode / decode / forward --
    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[_EncodeOutput, Tuple[_LatentDist]]:
        """Encode ``[B, 3, T, H, W]`` (float, ``[-1, 1]``) to latent."""

        def _encode(x: torch.Tensor) -> torch.Tensor:
            return self.model.encode(x)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(s) for s in x.split(1)]
            latent = torch.cat(encoded_slices)
        else:
            latent = _encode(x)

        dist = _LatentDist(latent)
        if not return_dict:
            return (dist,)
        return _EncodeOutput(latent_dist=dist)

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[_DecodeOutput, Tuple[torch.Tensor]]:
        """Decode latent ``[B, C, T', H', W']`` back to pixel space."""

        def _decode(z: torch.Tensor) -> torch.Tensor:
            return self.model.decode(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(s) for s in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if not return_dict:
            return (decoded,)
        return _DecodeOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """Full round-trip: encode -> decode."""
        enc_out = self.encode(sample)
        z = (
            enc_out.latent_dist.sample(generator)
            if sample_posterior
            else enc_out.latent_dist.mode()
        )
        dec_out = self.decode(z, generator=generator)

        if not return_dict:
            return (
                (dec_out.sample, enc_out.latent_dist)
                if return_posterior
                else (dec_out.sample,)
            )
        return dec_out
