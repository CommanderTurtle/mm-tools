# SPDX-License-Identifier: Apache-2.0
"""WeMM audio encoder: Whisper encoder layers + Q-Former + audio_proj.
Matches WeMM modeling_whisper.WhisperEncoder for 1:1 weight loading and
identical forward (conv → layers → layer_norm → unfold → qformer → audio_proj).
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


def _sinusoids(
    length: int, channels: int, max_timescale: float = 10000
) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError(f"channels must be even, got {channels}")
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2))
    t = torch.arange(length).view(-1, 1) * inv.view(1, -1)
    return torch.cat([t.sin(), t.cos()], dim=1)


class WeMMWhisperAttention(nn.Module):
    """WeMM/HF-style: separate q_proj, k_proj, v_proj, out_proj."""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states) * self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(hidden_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask.view(1, -1, 1, 1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)


class WeMMWhisperEncoderLayer(nn.Module):
    """WeMM/HF-style: self_attn (q/k/v/out_proj), fc1, fc2 (no .mlp wrapper)."""

    def __init__(self, config: Any, prefix: str = ""):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WeMMWhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=getattr(config, "attention_dropout", 0.0),
        )
        # self.self_attn = WhisperAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=config.encoder_attention_heads,
        #     attn_type=AttentionType.ENCODER,
        #     prefix=prefix
        # )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = getattr(config, "dropout", 0.0)
        act_fn = getattr(config, "activation_function", "gelu")
        self.activation_fn = getattr(F, act_fn, F.gelu)
        self.activation_dropout = getattr(config, "activation_dropout", 0.0)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return residual + hidden_states


def _load_qformer(qformer_config: Any) -> nn.Module:
    """Build the repository's Q-Former without executing checkpoint Python."""
    from transformers import BertConfig

    from .wemm_bert import BertLMHeadModel

    values = (
        qformer_config.to_dict()
        if hasattr(qformer_config, "to_dict")
        else dict(qformer_config)
    )
    return BertLMHeadModel._from_config(BertConfig.from_dict(values)).bert


class WeMMAudioEncoder(nn.Module):
    """
    WeMM audio encoder: conv → Whisper layers → layer_norm → unfold → Q-Former →
    audio_proj.
    Matches WeMM modeling_whisper.WhisperEncoder for 1:1 weight loading.
    """

    def __init__(
        self,
        *,
        audio_config: Any,  # WhisperConfig directly
        model_path: str,
        qformer_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        from transformers import WhisperConfig

        # audio_config can be WhisperConfig or dict
        if isinstance(audio_config, dict):
            qformer_cfg = qformer_config or audio_config.get("qformer_config")
            ac = WhisperConfig.from_dict(
                {k: v for k, v in audio_config.items() if k != "qformer_config"}
            )
        else:
            qformer_cfg = qformer_config or getattr(
                audio_config, "qformer_config", None
            )
            ac = audio_config

        if qformer_cfg is None:
            raise ValueError(
                "WeMMAudioEncoder requires audio_config.qformer_config or qformer"
                "_config"
            )

        self.config = ac
        embed_dim = ac.d_model
        self.num_mel_bins = ac.num_mel_bins
        self.max_source_positions = ac.max_source_positions
        self.embed_scale = (
            math.sqrt(embed_dim) if getattr(ac, "scale_embedding", False) else 1.0
        )
        self.dropout = getattr(ac, "dropout", 0.0)
        self.layerdrop = getattr(ac, "encoder_layerdrop", 0.0)
        self.second_per_window = getattr(ac, "second_per_window", 0.5)
        self.second_stride = getattr(ac, "second_stride", 0.5)

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        with torch.no_grad():
            self.embed_positions.weight.copy_(
                _sinusoids(self.max_source_positions, embed_dim).to(
                    self.embed_positions.weight.dtype
                )
            )
        self.layers = nn.ModuleList(
            [WeMMWhisperEncoderLayer(ac, prefix) for _ in range(ac.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(ac.d_model)

        self.qformer = _load_qformer(qformer_cfg)
        qfc = (
            qformer_cfg.to_dict()
            if hasattr(qformer_cfg, "to_dict")
            else dict(qformer_cfg)
        )
        query_length = qfc.get("query_length", 1)
        hidden_size = qfc.get("hidden_size", 4104)
        out_hidden_size = qfc.get("out_hidden_size", 4096)
        self.q_tokens = nn.Parameter(
            torch.randn(1, query_length, hidden_size)
            * qfc.get("initializer_range", 0.02)
        )
        self.audio_proj = nn.Linear(hidden_size, out_hidden_size)

    def forward(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        # print("input_features",input_features)
        if isinstance(input_features, list | tuple):
            input_features = torch.cat(
                [
                    x if isinstance(x, torch.Tensor) else x.unsqueeze(0)
                    for x in input_features
                ],
                dim=0,
            )
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states + self.embed_positions.weight[
            : hidden_states.size(1), :
        ].to(hidden_states.dtype)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        speech_embeds = self.layer_norm(hidden_states)
        B, T, C = speech_embeds.shape
        kernel = round(1500 * self.second_per_window / 30.0)
        stride = round(1500 * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(
            speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride
        )
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, (0, 3, 2, 1))
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(
            speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device
        )
        query_tokens = self.q_tokens.expand(speech_embeds.shape[0], -1, -1)
        # WeMM Q-Former is BertLMHeadModel.bert; forward with query_embeds only
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.audio_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(-1)).contiguous()
        # print("speech_embeds",speech_embeds)
        return speech_embeds
