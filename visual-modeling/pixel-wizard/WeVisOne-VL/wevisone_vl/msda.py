"""DINO multi-scale deformable attention (pure PyTorch implementation)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Device-agnostic implementation built on ``F.grid_sample``."""
    b, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(h) * int(w) for h, w in spatial_shapes.tolist()]
    value_levels = value.split(split_sizes, dim=1)
    sampling_grids = sampling_locations.mul(2.0).sub(1.0)
    sampled_per_level = []
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        value_l = value_levels[level]
        value_l = value_l.permute(0, 2, 3, 1).reshape(
            b * num_heads, head_dim, int(height), int(width)
        )
        grid_l = sampling_grids[:, :, :, level]
        grid_l = grid_l.permute(0, 2, 1, 3, 4).reshape(
            b * num_heads, num_queries, num_points, 2
        )
        sampled = F.grid_sample(
            value_l.float(),
            grid_l.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_per_level.append(sampled)
    sampled = torch.stack(sampled_per_level, dim=-2)
    sampled = sampled.flatten(-2)
    weights = attention_weights.permute(0, 2, 1, 3, 4)
    weights = weights.reshape(b * num_heads, 1, num_queries, num_levels * num_points)
    output = (sampled * weights.float()).sum(-1)
    output = output.reshape(b, num_heads * head_dim, num_queries).transpose(1, 2)
    return output.to(dtype=value.dtype)


class MultiScaleDeformableAttention(nn.Module):
    """The MSDeformAttn module used by DINO encoder and decoder layers."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim={embed_dim} not divisible by heads={num_heads}")
        self.embed_dim = int(embed_dim)
        self.num_levels = int(num_levels)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        theta = torch.arange(self.num_heads, dtype=torch.float32)
        theta = theta * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([theta.cos(), theta.sin()], dim=-1)
        grid = grid / grid.abs().amax(dim=-1, keepdim=True)
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        for point in range(self.num_points):
            grid[:, :, point] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, num_queries, _ = query.shape
        b_value, input_len, _ = input_flatten.shape
        if b != b_value:
            raise ValueError(f"query batch {b} != value batch {b_value}")
        expected = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if expected != input_len:
            raise ValueError(f"spatial shapes contain {expected} tokens, got {input_len}")
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(
            b, input_len, self.num_heads, self.embed_dim // self.num_heads
        )
        offsets = self.sampling_offsets(query).view(
            b,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query).view(
            b, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        weights = weights.softmax(-1).view(
            b, num_queries, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack(
                [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1
            ).to(device=query.device, dtype=query.dtype)
            locations = (
                reference_points[:, :, None, :, None]
                + offsets / normalizer[None, None, None, :, None]
            )
        elif reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / float(self.num_points)
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError("reference_points last dimension must be 2 or 4")
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, locations, weights
        )
        return self.output_proj(output.to(dtype=query.dtype))

