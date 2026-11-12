"""Discriminator head for DMD+ADV distillation (Nomas DiT).

Structural re-implementation of AppVideo's ``QueryHead``
(``appvideo/distill/wan22_head.py``) adapted to this repo's Nomas DiT, using
standard PyTorch ops (``nn.MultiheadAttention``) instead of the Wan22-specific
``flash_attention`` / ``rope_apply``.

Design (identical idea to the reference):
  * The frozen teacher (real_score) DiT is run with
    ``out_contexts=True, hook_layers=[...]`` and returns a dict
    ``{str(layer_idx): hidden_state[B, L, dim]}`` of per-double-block features.
  * For each hooked layer a ``QueryAttentionBlock`` cross-attends a single
    learnable query against that layer's features (with 6-way time modulation),
    producing a per-layer vector ``[B, dim]``.
  * All per-layer vectors are concatenated ``[B, dim * n_hooks]`` -> LayerNorm ->
    Linear -> scalar logit ``[B, 1]``. The final Linear is zero-initialized so the
    discriminator starts neutral.

For the Nomas-G model: dim=2048, num_heads=16, ffn_dim=8192 (mlp_width_ratio=4),
54 double blocks; default hook layers evenly spaced: [9, 19, 29, 39, 49].
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal timestep embedding (matches AppVideo/Wan22 convention)."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class QueryAttentionBlock(nn.Module):
    """One cross-attention probe over a single layer's features.

    A learnable query token attends (cross-attention) to the hooked feature
    sequence, followed by an FFN, both gated by a 6-way time modulation derived
    from the discriminator timestep. Mirrors AppVideo's ``QueryAttentionBlock``
    but uses ``nn.MultiheadAttention`` (batch_first) for the cross-attn.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        text_cond: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.text_cond = bool(text_cond)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
        )
        if self.text_cond:
            self.norm_txt = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
            self.text_attn = nn.MultiheadAttention(
                dim,
                num_heads,
                batch_first=True,
            )
            self.text_gate = nn.Parameter(torch.zeros(1))
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        # Learnable per-block modulation bias and the query token.
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.emb = nn.Parameter(torch.randn(1, 1, dim))

    def forward(
        self,
        e: torch.Tensor,
        context: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            e: time modulation input, shape [B, 1, 6, dim] (float32).
            context: hooked features, shape [B, L, dim].
        Returns:
            [B, dim] per-layer pooled representation.
        """
        with torch.amp.autocast("cuda", enabled=False):
            # e: [B, 1, 6, dim] + modulation[1,6,dim] -> chunk into 6 x [B,1,1,dim]
            e = (self.modulation.unsqueeze(0) + e.float()).chunk(6, dim=2)

        B = context.shape[0]
        query = self.emb.expand(B, -1, -1).to(dtype=context.dtype)  # [B, 1, dim]

        # cross-attention: query attends to context features
        x = (
            query
            + self.cross_attn(self.norm3(query), context, context, need_weights=False)[
                0
            ]
        )

        if self.text_cond and text_context is not None:
            text_context = text_context.to(dtype=x.dtype)
            x = (
                x
                + self.text_gate.to(dtype=x.dtype)
                * self.text_attn(
                    self.norm_txt(x),
                    text_context,
                    text_context,
                    key_padding_mask=text_padding_mask,
                    need_weights=False,
                )[0]
            )

        # FFN with shift/scale/gate modulation (e[3] shift, e[4] scale, e[5] gate)
        shift = e[3].squeeze(2).to(dtype=x.dtype)  # [B, 1, dim]
        scale = e[4].squeeze(2).to(dtype=x.dtype)
        gate = e[5].squeeze(2).to(dtype=x.dtype)
        y = self.ffn(self.norm2(x.float()).to(dtype=x.dtype) * (1 + scale) + shift)
        x = x + y * gate

        return x[:, 0]  # [B, dim]


class NomasQueryHead(nn.Module):
    """Cross-attention discriminator head over Nomas DiT hooked features.

    Args:
        dim: DiT hidden size (2048 for Nomas-G).
        ffn_dim: FFN hidden size in each probe block (8192 = 4 * dim).
        freq_dim: sinusoidal timestep embedding dim.
        num_heads: attention heads (16 for Nomas-G).
        hook_layers: double-block indices whose features are consumed. MUST match
            the ``hook_layers`` passed to the backbone forward.
    """

    def __init__(
        self,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        num_heads: int = 16,
        eps: float = 1e-6,
        hook_layers: List[int] = (9, 19, 29, 39, 49),
        text_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.hook_layers = list(hook_layers)
        self.text_dim = int(text_dim) if text_dim else None
        num_hook_layers = len(self.hook_layers)

        # timestep embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if self.text_dim is not None:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(self.text_dim, eps=eps),
                nn.Linear(self.text_dim, dim),
            )

        # one probe block per hooked layer, keyed by str(index) to match the
        # backbone's returned feature dict keys.
        self.blocks = nn.ModuleDict(
            {
                str(i): QueryAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    eps,
                    text_cond=self.text_dim is not None,
                )
                for i in self.hook_layers
            }
        )

        # final classifier over concatenated per-layer vectors
        self.norm = nn.LayerNorm(dim * num_hook_layers, eps=eps)
        self.ffn = nn.Linear(dim * num_hook_layers, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        # zero-init output so the discriminator starts neutral (logit == 0)
        nn.init.zeros_(self.ffn.weight)
        nn.init.zeros_(self.ffn.bias)
        for block in self.blocks.values():
            if getattr(block, "text_cond", False):
                nn.init.zeros_(block.text_gate)

    def _prepare_text(
        self,
        text_states: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
    ):
        if self.text_dim is None or text_states is None:
            return None, None
        text = text_states
        mask = encoder_attention_mask
        if text.dim() == 4:
            batch, segments, length, width = text.shape
            text = text.reshape(batch, segments * length, width)
            if mask is not None and mask.dim() == 3:
                mask = mask.reshape(batch, segments * length)
        elif text.dim() != 3:
            raise ValueError(
                "disc text_states must be 3D [B,S,C] or 4D [B,N,S,C], "
                f"got {tuple(text.shape)}"
            )
        if text.shape[-1] != self.text_dim:
            raise ValueError(
                f"disc text_states width {text.shape[-1]} != configured "
                f"text_dim {self.text_dim}"
            )
        text = self.text_proj(text)
        padding = None
        if mask is not None:
            keep = mask.to(dtype=torch.bool)
            if keep.shape[:2] != text.shape[:2]:
                raise ValueError(
                    f"disc text mask shape {tuple(keep.shape)} incompatible "
                    f"with text_states {tuple(text.shape)}"
                )
            padding = ~keep
            all_pad = padding.all(dim=1)
            if bool(all_pad.any()):
                padding = padding.clone()
                padding[all_pad, 0] = False
        return text, padding

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        t: torch.Tensor,
        text_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: dict {str(layer_idx): [B, L, dim]} from the backbone
                (out_contexts=True). Must contain every key in ``hook_layers``.
            t: discriminator timestep. Accepts [B], [B,1], or [B,F] (per-frame,
               uniform across frames) — reduced to one value per sample.
        Returns:
            logits: [B, 1].
        """
        seq_len = 1
        # Reduce to one timestep per sample. The disc conditions per-sample, and
        # the sampler uses uniform_timestep=True (identical across frames), so
        # taking the first column is exact.
        if t.dim() >= 2:
            t = t[:, 0]
        t = t.reshape(-1, 1).expand(t.size(0), seq_len)  # [B, 1]

        with torch.amp.autocast("cuda", enabled=False):
            bt = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat)
                .unflatten(0, (bt, seq_len))
                .float()
            )  # [B, 1, dim]
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))  # [B, 1, 6, dim]

        text_context, text_padding = self._prepare_text(
            text_states,
            encoder_attention_mask,
        )
        logits = []
        for k, block in self.blocks.items():
            feat = features[k]
            logits.append(block(e0, feat, text_context, text_padding))  # [B, dim]
        logits = torch.cat(logits, dim=1)  # [B, dim * n_hooks]

        logits = self.norm(logits)
        logits = self.ffn(logits)  # [B, 1]
        return logits
