# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class — Optimized.

Key optimizations over the original nn.MultiheadAttention implementation:
  1. F.scaled_dot_product_attention (FlashAttention / Memory-Efficient kernels)
     replaces nn.MHA, whose slow path (triggered when query != value from
     positional embeddings) causes excessive "copy-to-empty" GPU ops.
  2. Fused QK projection in self-attention — Q and K always receive the same
     input (src+pos or tgt+query_pos), so a single nn.Linear(d, 2d) replaces
     two separate nn.Linear(d, d).  Under bf16 autocast every nn.Linear casts
     its float32 weight+bias+input → bf16 (three aten::to kernels).  Fusing
     QK removes one linear call per self-attention layer, saving ~3 cast
     kernel launches each — 36 fewer kernels per forward pass (6 enc + 6 dec).

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Self-attention with **fused QK projection** + SDPA.

    In DETR self-attention the Q and K inputs are always identical
    (src+pos for encoder, tgt+query_pos for decoder) while V differs.
    Fusing Q and K into one ``nn.Linear(d, 2d)`` halves the projection
    calls for Q/K, and — critically — halves the AMP autocast weight-
    casting overhead that otherwise dominates the profile.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        # Fused Q+K: one matmul → [B, N, 2·D], then split
        self.qk_proj = nn.Linear(d_model, 2 * d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        qk_input: Tensor,
        v_input: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            qk_input: [B, N, D]  — shared input for Q and K (e.g. src + pos)
            v_input:  [B, N, D]  — input for V (e.g. src)
            attn_mask: optional additive float mask  (usually None in DETR)
            key_padding_mask: [B, N] bool, True = padding (ignore)
        Returns:
            output: [B, N, D]
        """
        B, N, _ = qk_input.shape

        # --- projections ---------------------------------------------------
        qk = self.qk_proj(qk_input)                          # [B, N, 2·D]
        qk = qk.view(B, N, 2, self.nhead, self.head_dim)     # split heads
        q, k = qk.unbind(2)            # each [B, N, nhead, head_dim]
        q = q.transpose(1, 2)          # [B, nhead, N, head_dim]
        k = k.transpose(1, 2)

        v = self.v_proj(v_input).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # --- mask ----------------------------------------------------------
        sdpa_mask = _build_sdpa_mask(key_padding_mask, attn_mask, B, N, N,
                                     qk_input.device, qk_input.dtype)

        # --- attention -----------------------------------------------------
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention with separate Q / K / V projections + SDPA.

    In DETR cross-attention the three inputs are all different
    (q=tgt+query_pos, k=memory+pos, v=memory), so projections cannot
    be fused further without adding extra linear layers.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query: [B, N_q, D]
            key:   [B, N_k, D]
            value: [B, N_k, D]
            attn_mask: optional additive float mask  (usually None)
            key_padding_mask: [B, N_k] bool, True = padding
        Returns:
            output: [B, N_q, D]
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        q = self.q_proj(query).view(B, N_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N_k, self.nhead, self.head_dim).transpose(1, 2)

        sdpa_mask = _build_sdpa_mask(key_padding_mask, attn_mask, B, N_q, N_k,
                                     query.device, query.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N_q, self.d_model)
        return self.out_proj(out)


def _build_sdpa_mask(
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    B: int, N_q: int, N_k: int,
    device: torch.device, dtype: torch.dtype,
) -> Optional[Tensor]:
    """Convert nn.MHA-style masks to SDPA-compatible masks.

    Fast path (common): only key_padding_mask → cheap bool inversion.
    Slow path (rare):  combines both masks as float additive mask.
    """
    if key_padding_mask is not None and attn_mask is None:
        # Bool mask: SDPA True = attend, nn.MHA True = ignore → invert
        return ~key_padding_mask[:, None, None, :]          # [B, 1, 1, N_k]
    if attn_mask is not None:
        sdpa_mask = attn_mask
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :].expand(B, 1, N_q, N_k)
            sdpa_mask = sdpa_mask.clone()
            sdpa_mask.masked_fill_(pad, float('-inf'))
        return sdpa_mask
    return None


# ---------------------------------------------------------------------------
# Transformer top-level
# ---------------------------------------------------------------------------

class DetrTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src: [B, N, d_model] torch.FloatTensor.
            mask: [B, N] torch.BoolTensor.
            pos_embed: [B, N, d_model] torch.FloatTensor.
            query_embed: [B, num_queries, d_model] torch.FloatTensor.
        """
        B, num_queries, _ = query_embed.shape

        tgt = torch.zeros_like(query_embed)  # [B, num_queries, d_model]
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs, memory

class DetrWSegTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # seg encoder
        seg_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        seg_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.seg_encoder = TransformerEncoder(seg_encoder_layer, num_encoder_layers, seg_encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src: [B, N, d_model] torch.FloatTensor.
            mask: [B, N] torch.BoolTensor.
            pos_embed: [B, N, d_model] torch.FloatTensor.
            query_embed: [B, num_queries, d_model] torch.FloatTensor.
        """
        B, num_queries, _ = query_embed.shape

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        seg_memory = self.seg_encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        tgt = torch.zeros_like(query_embed)  # [B, num_queries, d_model]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs, memory, seg_memory

class PosTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        """
        Args:
            src: [B, N, d_model] torch.FloatTensor.
            mask: [B, N] torch.BoolTensor.
            pos_embed: [B, N, d_model] torch.FloatTensor.
        """
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory

class DetrWSegWReconTransformer(nn.Module):

    def __init__(self, d_model=512, d_seg_model=None, nhead=8, nhead_seg=None,
                 num_encoder_layers=6,
                 num_seg_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dim_seg_feedforward=None, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 seg_pos_input_first=False):
        super().__init__()

        if d_seg_model is None:
            d_seg_model = d_model

        if dim_seg_feedforward is None:
            dim_seg_feedforward = dim_feedforward

        if nhead_seg is None:
            nhead_seg = nhead

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # seg encoder
        if num_seg_encoder_layers > 0:
            seg_encoder_layer = TransformerEncoderLayer(d_seg_model, nhead_seg, dim_seg_feedforward,
                                                dropout, activation, normalize_before)
            seg_encoder_norm = nn.LayerNorm(d_seg_model) if normalize_before else None
            self.seg_encoder = TransformerEncoder(seg_encoder_layer, num_seg_encoder_layers, seg_encoder_norm, pos_input_first=seg_pos_input_first)
            # print(f"seg_pos_input_first: {seg_pos_input_first}; num_seg_encoder_layers: {num_seg_encoder_layers}")
        else:
            self.seg_encoder = None

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.d_seg_model = d_seg_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, recon_src, recon_mask, recon_pos_embed):
        """
        Args:
            src: [B, N, d_model] torch.FloatTensor.
            mask: [B, N] torch.BoolTensor.
            pos_embed: [B, N, d_model] torch.FloatTensor.
            query_embed: [B, num_queries, d_model] torch.FloatTensor.
            recon_src: [B, N_recon, d_seg_model] torch.FloatTensor.
            recon_mask: [B, N_recon] torch.BoolTensor.
            recon_pos_embed: [B, N_recon, d_seg_model] torch.FloatTensor.
        """
        B, num_queries, _ = query_embed.shape

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        if self.seg_encoder is not None:
            seg_memory = self.seg_encoder(recon_src, src_key_padding_mask=recon_mask, pos=recon_pos_embed)
        else:
            seg_memory = recon_src

        tgt = torch.zeros_like(query_embed)  # [B, num_queries, d_model]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs, memory, seg_memory

# ---------------------------------------------------------------------------
# Encoder / Decoder wrappers
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, pos_input_first=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pos_input_first = pos_input_first

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer_i, layer in enumerate(self.layers):
            if self.pos_input_first:
                if layer_i == 0:
                    output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                else:
                    output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=None)
            else:
                output = layer(output, src_mask=mask,
                            src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        if self.return_intermediate:
            # Pre-allocate to avoid torch.stack copy at the end
            intermediate = torch.empty(
                self.num_layers, *tgt.shape,
                device=tgt.device, dtype=tgt.dtype,
            )
            for i, layer in enumerate(self.layers):
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
                intermediate[i] = self.norm(output)
            output = self.norm(output)
            intermediate[-1] = output
            return intermediate

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


# ---------------------------------------------------------------------------
# Encoder / Decoder layers
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)
        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # Self-attention: Q/K share input (src+pos), V = src
        qk_input = self._with_pos(src, pos)
        src2 = self.self_attn(qk_input, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        qk_input = self._with_pos(src2, pos)
        src2 = self.self_attn(qk_input, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, nhead, dropout=dropout)
        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # --- self-attention (fused QK) ------------------------------------
        qk_input = self._with_pos(tgt, query_pos)
        tgt2 = self.self_attn(qk_input, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # --- cross-attention (separate Q/K/V) -----------------------------
        tgt2 = self.cross_attn(
            query=self._with_pos(tgt, query_pos),
            key=self._with_pos(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # --- FFN ----------------------------------------------------------
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # --- self-attention (fused QK) ------------------------------------
        tgt2 = self.norm1(tgt)
        qk_input = self._with_pos(tgt2, query_pos)
        tgt2 = self.self_attn(qk_input, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # --- cross-attention (separate Q/K/V) -----------------------------
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(
            query=self._with_pos(tgt2, query_pos),
            key=self._with_pos(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)

        # --- FFN ----------------------------------------------------------
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
