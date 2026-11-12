from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn

from hymm.models.dit.modules.activation_layers import get_activation_layer
from hymm.models.dit.modules.attention import attention
from hymm.models.dit.modules.norm_layers import get_norm_layer
from hymm.models.dit.modules.embed_layers import TimestepEmbedder, TextProjection
from hymm.models.dit.modules.mlp_layers import MLP
from hymm.models.dit.modules.modulate_layers import modulate, apply_gate


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        self.self_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = attention(q, k, v, attn_mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        mask = mask.clone().bool()
        # avoid attention weight become NaN
        mask[:, 0] = True
        for block in self.blocks:
            x = block(x, c, mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "torch",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."

        self.input_embedder = nn.Linear(
            in_channels, hidden_size, bias=True, **factory_kwargs
        )

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer, **factory_kwargs
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ):
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.to(x.dtype).unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        x = self.individual_token_refiner(x, c, mask)

        return x


class SingleTokenRefinerPerBlock(nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "torch",
        seg_pos_emb_num: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."

        self.input_embedder = nn.Linear(
            in_channels, hidden_size, bias=True, **factory_kwargs
        )

        # Optional learnable per-segment positional embedding shared across all
        # tokens within a segment. When enabled (seg_pos_emb_num > 0), each
        # segment id n in [0, seg_pos_emb_num) is mapped to a learnable vector
        # in R^hidden_size, which is broadcast-added to all S tokens of that
        # segment after `input_embedder` and before `individual_token_refiner`.
        # Disabled (= 0) preserves legacy behavior exactly.
        self.seg_pos_emb_num = int(seg_pos_emb_num)
        if self.seg_pos_emb_num > 0:
            self.seg_pos_emb = nn.Embedding(
                self.seg_pos_emb_num, hidden_size, **factory_kwargs
            )
            # Zero-init: keeps behavior identical to legacy at step 0; the
            # model learns a non-trivial PE only as training proceeds.
            nn.init.zeros_(self.seg_pos_emb.weight)
        else:
            self.seg_pos_emb = None

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer, **factory_kwargs
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
        seg_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass supporting both standard and pre-chunked text inputs.

        Args:
            x: Text embeddings. Can be:
               - 3D [B, S, D]: standard single text (will be expanded if t is 2D)
               - 4D [B, T, S, D]: pre-chunked text (each chunk has its own text)
            t: Timestep. Can be:
               - 1D [B]: scalar timestep
               - 2D [B, T]: per-chunk timestep
            mask: Attention mask. Can be:
               - 2D [B, S]: single mask (will be expanded if needed)
               - 3D [B, T, S]: per-chunk mask
            seg_ids: Optional explicit per-row segment ids used to look up
               ``seg_pos_emb`` when ``seg_pos_emb_num > 0``. Shape must match
               the row dimension that ``input_embedder`` produces (i.e.
               ``[B*T]`` for the 4D temporal path, or ``[B]`` for a 3D path
               where the caller has already flattened the segment axis into
               batch — e.g. token packing). When provided, this bypasses the
               implicit ``arange(T)`` derived from the temporal dim and lets
               callers inject segment positional embedding even on the 3D
               input path. When ``None``, legacy behavior is preserved.

        Returns:
            - 3D [B, S, C] if t is 1D and x is 3D
            - 4D [B, T, S, C] if t is 2D (regardless of x input dim)
        """
        timestep_aware_representations = self.t_embedder(t)

        # Handle case where t has shape [B, T] -> timestep_aware_representations is [B, T, C]
        # Reshape T dimension into B dimension for parallel processing
        has_temporal_dim = timestep_aware_representations.dim() == 3
        if has_temporal_dim:
            B, T, C = timestep_aware_representations.shape
            # [B, T, C] -> [B*T, C]
            timestep_aware_representations = rearrange(
                timestep_aware_representations, "b t c -> (b t) c"
            )

            if x.dim() == 4:
                # x is already pre-chunked [B, T_text, S, D]
                # T_text should match T (already repeated to num_chunks in Nomas_cpt.py)
                assert x.shape[1] == T, (
                    f"Pre-chunked text has {x.shape[1]} chunks but timestep has {T} chunks. "
                    f"Ensure text_states is repeated to match num_chunks before calling the refiner."
                )
                x = rearrange(x, "b t s d -> (b t) s d")  # [B*T, S, D]
            else:
                # x is [B, S, D], expand same text to all T chunks
                x = x.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, S, D]
                x = rearrange(x, "b t s d -> (b t) s d")  # [B*T, S, D]

            if mask is not None:
                if mask.dim() == 3:
                    # mask is already per-chunk [B, T, S]
                    assert mask.shape[1] == T, (
                        f"Pre-chunked mask has {mask.shape[1]} chunks but timestep has {T} chunks."
                    )
                    mask = rearrange(mask, "b t s -> (b t) s")  # [B*T, S]
                else:
                    # mask is [B, S], expand to all T chunks
                    mask = mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, S]
                    mask = rearrange(mask, "b t s -> (b t) s")  # [B*T, S]

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.to(x.dtype).unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)

        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        # Inject segment-level learnable positional embedding (shared within a
        # segment, distinct across segments). Two code paths:
        #
        # 1. ``seg_ids`` explicitly provided (e.g. from the token-packing
        #    refiner wrapper, where the caller has already flattened segments
        #    into the batch dim of a 3D tensor): use them directly. This works
        #    regardless of ``has_temporal_dim``.
        # 2. ``seg_ids`` is None and ``has_temporal_dim`` is True: derive
        #    ``arange(T)`` and broadcast across B. This preserves the legacy
        #    auto-inject behavior on the per-chunk 4D path.
        #
        # When ``seg_pos_emb`` is None (seg_pos_emb_num == 0) or neither
        # condition matches (legacy 3D single-segment path with no explicit
        # ids), no injection is performed — identical to legacy behavior.
        if self.seg_pos_emb is not None:
            if seg_ids is not None:
                # Explicit path. ``x`` here is whatever ``input_embedder``
                # produced; its leading dim must equal ``seg_ids.numel()``.
                assert seg_ids.dim() == 1, (
                    f"seg_ids must be 1D, got shape {tuple(seg_ids.shape)}"
                )
                assert seg_ids.shape[0] == x.shape[0], (
                    f"seg_ids length ({seg_ids.shape[0]}) must match the row "
                    f"dim of x ({x.shape[0]}). Caller is responsible for "
                    f"flattening segments into batch before calling."
                )
                _max_id = int(seg_ids.max().item()) if seg_ids.numel() > 0 else -1
                assert _max_id < self.seg_pos_emb_num, (
                    f"seg_ids contains id {_max_id} which exceeds "
                    f"seg_pos_emb_num={self.seg_pos_emb_num}. Increase "
                    f"model.text_seg_pos_emb_num."
                )
                seg_emb = self.seg_pos_emb(seg_ids.to(device=x.device))  # [N, D]
                seg_emb = seg_emb.unsqueeze(1)  # [N, 1, D]
                x = x + seg_emb.to(x.dtype)
            elif has_temporal_dim:
                assert T <= self.seg_pos_emb_num, (
                    f"Number of segments T={T} exceeds seg_pos_emb_num="
                    f"{self.seg_pos_emb_num}. Increase model.text_seg_pos_emb_num."
                )
                seg_ids_auto = torch.arange(T, device=x.device)  # [T]
                seg_emb = self.seg_pos_emb(seg_ids_auto)  # [T, D]
                seg_emb = seg_emb.unsqueeze(1)  # [T, 1, D]
                seg_emb = seg_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, 1, D]
                seg_emb = rearrange(seg_emb, "b t one d -> (b t) one d")  # [B*T, 1, D]
                x = x + seg_emb.to(x.dtype)

        x = self.individual_token_refiner(x, c, mask)

        # Reshape back to [B, T, S, C] if we had temporal dimension
        if has_temporal_dim:
            x = rearrange(x, "(b t) s c -> b t s c", b=B, t=T)

        return x
