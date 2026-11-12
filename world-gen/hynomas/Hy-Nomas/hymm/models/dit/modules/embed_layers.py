from .int8_stored_bf16_linear import linear_dtype
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from hymm.utils.helpers import to_2tuple

FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        is_reshape_temporal_channels=False,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )

        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    @property
    def is_pointwise(self) -> bool:
        """True when patch_size is (1,1,1) — Conv3d is equivalent to Linear."""
        return all(p == 1 for p in self.patch_size)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Project flat token sequences when patch_size is (1,1,1).

        When patch_size is all-ones, Conv3d(in_ch, out_ch, kernel=1, stride=1)
        is mathematically equivalent to Linear(in_ch, out_ch).  This method
        reuses the Conv3d weights to perform the projection on already-flat
        token sequences, avoiding the need to reshape back to 5D.

        Args:
            x: [B, L, C_in] — flat token sequence (L = T*H*W tokens,
               C_in = in_channels of the Conv3d).

        Returns:
            [B, L, C_out] — projected tokens (C_out = embed_dim).

        Raises:
            RuntimeError: if patch_size is not (1,1,1).
        """
        if not self.is_pointwise:
            raise RuntimeError(
                f"forward_flat requires patch_size=(1,1,1), got {self.patch_size}. "
                f"Use forward() with 5D input instead."
            )
        # Conv3d weight shape: [out_ch, in_ch, 1, 1, 1] → squeeze to [out_ch, in_ch]
        weight = self.proj.weight.squeeze(-1).squeeze(-1).squeeze(-1)  # [out_ch, in_ch]
        bias = self.proj.bias  # [out_ch] or None
        x = torch.nn.functional.linear(x, weight, bias)
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VisionProjection(torch.nn.Module):
    """
    Projects vision embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L488
    """

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280)
            )

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class ClipVisionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)

    def forward(self, x):
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): Tensor of arbitrary shape containing timestep indices.
            These may be fractional. Supports any shape like [B], [B, T], etc.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): Tensor of shape (*t.shape, dim) with positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    original_shape = t.shape
    t_flat = t.flatten()  # Flatten to 1D for computation

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t_flat[:, None].float() * freqs[None]  # [N, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N, dim]
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    # Restore original shape with embedding dimension appended
    return embedding.view(*original_shape, dim)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, **factory_kwargs
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

        # nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        # nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(linear_dtype(self.mlp[0]))
        t_emb = self.mlp(t_freq)
        return t_emb
