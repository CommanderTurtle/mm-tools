# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ACKNOWLEDGEMENTS.
# Modified for FastFrames: a frame-wise autoregressive Wan2.1 with a persistent
# per-layer KV cache (one latent frame = one 1560-token cache slot).
import functools

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .attention import attention
from .model import (
    Head,
    WanAttentionBlock,
    WanBackbone,
    causal_rope_apply,
    rope_apply,
    sinusoidal_embedding_1d,
)

__all__ = ["CausalWanModel", "FastOps", "allocate_kv_cache", "frame_causal_mask"]

_FLEX_PAD = 128


@functools.cache
def _compiled_flex_attention(dynamic=False):
    """Static-shape kernel for full-clip forwards; ``dynamic`` compiles one autotuned
    kernel for the growing clean prefixes of the DMD rollout (up to 20 lengths)."""
    if dynamic:
        return torch.compile(flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs")
    return torch.compile(flex_attention, dynamic=False, mode="default")


@functools.lru_cache(maxsize=32)
def frame_causal_mask(num_frames, frame_tokens, teacher_forcing=False, device="cuda"):
    """FlexAttention mask for full-sequence training forwards (one frame per block).

    Every frame attends to itself and to all earlier frames. With
    ``teacher_forcing`` the sequence is ``[clean frames, noisy frames]``: clean
    frames are causal among themselves, and noisy frame ``i`` sees the clean
    frames ``< i`` plus itself. The sequence is right-padded to a multiple of
    128 tokens; padding tokens only attend to themselves.
    """
    clean_length = num_frames * frame_tokens
    length = clean_length * (2 if teacher_forcing else 1)
    padded = -(-length // _FLEX_PAD) * _FLEX_PAD
    position = torch.arange(padded, device=device)
    frame = position // frame_tokens
    if teacher_forcing:
        noisy = (position >= clean_length) & (position < length)
        # Clean queries see clean keys up to the end of their frame. Noisy
        # queries see clean keys before their frame and their own noisy frame.
        visible_end = torch.where(noisy, (frame - num_frames) * frame_tokens, (frame + 1) * frame_tokens)
        own_start = torch.where(noisy, frame * frame_tokens, 0)
        own_end = torch.where(noisy, (frame + 1) * frame_tokens, 0)
    else:
        visible_end = (frame + 1) * frame_tokens
        own_start = own_end = torch.zeros_like(position)
    valid = position < length
    visible_end = torch.where(valid, visible_end, 0)

    def mask(b, h, q, kv):
        return (kv < visible_end[q]) | ((kv >= own_start[q]) & (kv < own_end[q])) | (q == kv)

    return create_block_mask(mask, B=None, H=None, Q_LEN=padded, KV_LEN=padded,
                             _compile=False, device=device)


def allocate_kv_cache(model, num_frames, frame_tokens, batch_size=1, dtype=torch.bfloat16, device="cuda"):
    """Per-layer self-attention cache ``{"k", "v"}`` of shape ``[B, F*frame_tokens, H, D]``."""
    shape = (batch_size, num_frames * frame_tokens, model.num_heads, model.dim // model.num_heads)
    return [
        {"k": torch.zeros(shape, dtype=dtype, device=device),
         "v": torch.zeros(shape, dtype=dtype, device=device)}
        for _ in range(model.num_layers)
    ]


def rope_real(x, table):
    """RoPE as FP32 real rotations; ``table`` is ``[L, 1, D/2, 2]`` (cos, sin)."""
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = table[..., 0], table[..., 1]
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).reshape_as(x).to(x.dtype)


class FastOps:
    """Numerically distinct fast paths for per-frame inference.

    * patch embedding as reshape + GEMM instead of Conv3d;
    * one fused QKV GEMM per layer (derived concatenated weights);
    * RoPE from precomputed FP32 (cos, sin) tables instead of complex128 math.

    Built once from a frozen model; rebuild after any weight update.
    """

    def __init__(self, model, height, width, num_frames):
        patch = model.patch_embedding
        self.patch_weight = patch.weight.detach().reshape(model.dim, -1)
        self.patch_bias = patch.bias
        self.qkv = []
        for block in model.blocks:
            sa = block.self_attn
            weight = torch.cat([sa.q.weight, sa.k.weight, sa.v.weight], dim=0).detach().contiguous()
            bias = torch.cat([sa.q.bias, sa.k.bias, sa.v.bias], dim=0).detach().contiguous()
            self.qkv.append((weight, bias))
        h, w = height // model.patch_size[1], width // model.patch_size[2]
        half = model.dim // model.num_heads // 2
        model.freqs = model.freqs.to(patch.weight.device)
        temporal, vertical, horizontal = model.freqs.split(
            [half - 2 * (half // 3), half // 3, half // 3], dim=1)
        self.rope = [
            torch.view_as_real(torch.cat((
                temporal[f:f + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
                vertical[:h].view(1, h, 1, -1).expand(1, h, w, -1),
                horizontal[:w].view(1, 1, w, -1).expand(1, h, w, -1),
            ), dim=-1).reshape(h * w, 1, half)).float().contiguous()
            for f in range(num_frames)
        ]

    def patch_embed(self, latent):
        """``[B, 1, C, H, W]`` -> ``[B, H*W/4, dim]`` in Conv3d channel order."""
        b, _, c, h, w = latent.shape
        x = latent.permute(0, 2, 1, 3, 4).reshape(b, c, 1, h // 2, 2, w // 2, 2)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).reshape(b, h * w // 4, c * 4)
        return F.linear(x, self.patch_weight, self.patch_bias)


class CausalWanAttentionBlock(WanAttentionBlock):
    """Same parameters as the bidirectional block; attends to the cached prefix.

    Called through ``forward`` so that FSDP-wrapped blocks gather their
    parameters: with ``kv_cache`` it denoises one frame against the cache, with
    ``frame`` alone it runs the differentiable one-frame forward, otherwise the
    full-sequence training forward.
    """

    def forward(self, *args, kv_cache=None, frame=None, **kwargs):
        if kv_cache is not None:
            return self.forward_frame(*args, kv_cache=kv_cache, frame=frame, **kwargs)
        if frame is not None:
            return self.forward_frame_train(*args, frame=frame, **kwargs)
        return self.forward_train(*args, **kwargs)

    def forward_train(self, x, e, grid_sizes, freqs, context, block_mask, teacher_forcing=False, return_kv=False):
        """Full sequence ``[B, F*L, C]`` under ``block_mask``; ``e``: ``[B, F, 6, C]``.

        ``return_kv`` also returns the (RoPE'd) keys and values, i.e. the KV cache
        of the whole clip, computed with the dynamic-shape attention kernel.
        """
        num_frames, tokens = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        h = (self.norm1(x).unflatten(1, (num_frames, tokens)) * (1 + e[1]) + e[0]).flatten(1, 2)
        sa = self.self_attn
        q, k, v = sa.qkv(h)
        if teacher_forcing:
            # Clean and noisy copies of a frame share the same RoPE positions.
            q = torch.cat([rope_apply(u, grid_sizes, freqs) for u in q.chunk(2, dim=1)], dim=1)
            k = torch.cat([rope_apply(u, grid_sizes, freqs) for u in k.chunk(2, dim=1)], dim=1)
        else:
            q, k = rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs)
        q, k = q.type_as(v), k.type_as(v)
        length = q.shape[1]
        pad = -(-length // _FLEX_PAD) * _FLEX_PAD - length
        padded = [torch.cat([u, u.new_zeros(u.shape[0], pad, *u.shape[2:])], dim=1) for u in (q, k, v)]
        if return_kv:
            y = _compiled_flex_attention(dynamic=True)(*(u.transpose(2, 1).contiguous() for u in padded),
                                                       block_mask=block_mask,
                                                       kernel_options={"BLOCKS_ARE_CONTIGUOUS": True})
        else:
            y = _compiled_flex_attention()(*(u.transpose(2, 1) for u in padded), block_mask=block_mask)
        y = sa.o(y[:, :, :length].transpose(2, 1).flatten(2))
        x = x + (y.unflatten(1, (num_frames, tokens)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn((self.norm2(x).unflatten(1, (num_frames, tokens)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(1, (num_frames, tokens)) * e[5]).flatten(1, 2)
        return (x, k, v) if return_kv else x

    def forward_frame_train(self, x, e, grid_sizes, freqs, context, context_kv, frame):
        """Differentiable ``forward_frame`` without a cache: attends to itself and to
        ``context_kv = (keys, values)``, lists of ``[B, L_i, H, D]`` chunks of the frames
        ``< frame`` (or ``None``). Returns the block output and this frame's (k, v).
        """
        tokens = x.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        h = (self.norm1(x).unflatten(1, (1, tokens)) * (1 + e[1]) + e[0]).flatten(1, 2)
        sa = self.self_attn
        q, k, v = sa.qkv(h)
        q = causal_rope_apply(q, grid_sizes, freqs, start_frame=frame).type_as(v)
        k = causal_rope_apply(k, grid_sizes, freqs, start_frame=frame).type_as(v)
        keys, values = (k, v) if not context_kv else (torch.cat([*context_kv[0], k], dim=1),
                                                       torch.cat([*context_kv[1], v], dim=1))
        y = sa.o(attention(q, keys, values).flatten(2))
        x = x + (y.unflatten(1, (1, tokens)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn((self.norm2(x).unflatten(1, (1, tokens)) * (1 + e[4]) + e[3]).flatten(1, 2))
        return x + (y.unflatten(1, (1, tokens)) * e[5]).flatten(1, 2), k, v

    def forward_frame(self, x, e, grid_sizes, freqs, cross_kv, kv_cache, frame, fast=None, layer=0):
        """One latent frame. ``e``: ``[B, 1, 6, C]``; writes this frame's K/V into ``kv_cache``."""
        tokens = x.shape[1]
        start, end = frame * tokens, (frame + 1) * tokens
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        h = (self.norm1(x).unflatten(1, (1, tokens)) * (1 + e[1]) + e[0]).flatten(1, 2)
        sa = self.self_attn
        shape = (x.shape[0], tokens, sa.num_heads, sa.head_dim)
        if fast is None:
            q, k, v = sa.qkv(h)
            q = causal_rope_apply(q, grid_sizes, freqs, start_frame=frame).type_as(v)
            k = causal_rope_apply(k, grid_sizes, freqs, start_frame=frame).type_as(v)
        else:
            q, k, v = F.linear(h, *fast.qkv[layer]).chunk(3, dim=-1)
            q = rope_real(sa.norm_q(q).view(shape), fast.rope[frame])
            k = rope_real(sa.norm_k(k).view(shape), fast.rope[frame])
            v = v.view(shape)
        kv_cache["k"][:, start:end].copy_(k)
        kv_cache["v"][:, start:end].copy_(v)
        y = sa.o(attention(q, kv_cache["k"][:, :end], kv_cache["v"][:, :end]).flatten(2))
        x = x + (y.unflatten(1, (1, tokens)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), None, cache=cross_kv)
        y = self.ffn((self.norm2(x).unflatten(1, (1, tokens)) * (1 + e[4]) + e[3]).flatten(1, 2))
        return x + (y.unflatten(1, (1, tokens)) * e[5]).flatten(1, 2)


class CausalHead(Head):

    def forward(self, x, e):
        """x: ``[B, F*L, C]``; e: ``[B, F, 1, C]`` per-frame time embedding."""
        num_frames, frame_tokens = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        return self.head(self.norm(x).unflatten(1, (num_frames, frame_tokens)) * (1 + e[1]) + e[0])


class CausalWanModel(WanBackbone):
    """Frame-wise autoregressive Wan2.1 T2V model."""

    block_class = CausalWanAttentionBlock
    head_class = CausalHead

    def time_conditioning(self, t, dtype):
        """``t``: ``[B, F]`` timesteps -> (block modulation ``[B,F,6,C]``, head ``[B,F,1,C]``)."""
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)
        return e0, e.unflatten(0, t.shape).unsqueeze(2)

    def text_cache(self, prompt_embeds):
        """Project the text context once per prompt into per-layer cross-attention K/V."""
        context = self.embed_text(prompt_embeds)
        caches = []
        for block in self.blocks:
            k, v = block.cross_attn.text_kv(context)
            caches.append({"k": k, "v": v})
        return caches

    def forward(self, *args, frame=None, **kwargs):
        """``model(...)`` (and FSDP's root ``__call__``): ``forward_frame_train`` when a
        ``frame`` index is given, else ``forward_train``."""
        if frame is not None:
            return self.forward_frame_train(*args, frame=frame, **kwargs)
        return self.forward_train(*args, **kwargs)

    def forward_train(self, x, t, prompt_embeds, clean_x=None, clean_t=None, return_kv=False):
        """Frame-causal forward over a whole clip (training; also ``model(...)``).

        x: noisy latents ``[B, F, 16, H, W]``; t: per-frame timesteps ``[B, F]``;
        prompt_embeds: T5 embeddings (list or ``[B, L, 4096]``). With ``clean_x``
        (teacher forcing) noisy frame ``i`` is conditioned on the clean frames
        ``< i``, which are embedded at ``clean_t`` (default 0). Returns the flow
        prediction ``[B, F, 16, H, W]``; with ``return_kv`` also the per-layer
        ``(k, v)`` of the clip (the KV cache of a clean prefix, for DMD rollouts).
        """
        if self.freqs.device != x.device:
            self.freqs = self.freqs.to(x.device)
        b, num_frames, _, height, width = x.shape
        grid_sizes = torch.tensor([[num_frames, height // 2, width // 2]] * b, dtype=torch.long)
        frame_tokens = height * width // 4

        def embed(latent):
            return self.patch_embedding(latent.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2).contiguous()

        tokens = embed(x)
        e0, e_head = self.time_conditioning(t, tokens.dtype)
        teacher_forcing = clean_x is not None
        if teacher_forcing and return_kv:
            raise ValueError("return_kv is not supported with teacher forcing (clean_x)")
        if teacher_forcing:
            tokens = torch.cat([embed(clean_x), tokens], dim=1)
            clean_e0, _ = self.time_conditioning(torch.zeros_like(t) if clean_t is None else clean_t, tokens.dtype)
            e0 = torch.cat([clean_e0, e0], dim=1)
        mask = frame_causal_mask(num_frames, frame_tokens, teacher_forcing, x.device)
        context = self.embed_text(prompt_embeds)
        kv = []
        for block in self.blocks:
            if return_kv:
                tokens, k, v = self._run_block(block, tokens, e0, grid_sizes, self.freqs, context, mask,
                                               return_kv=True)
                kv.append((k, v))
            else:
                tokens = self._run_block(block, tokens, e0, grid_sizes, self.freqs, context, mask, teacher_forcing)
        if teacher_forcing:
            tokens = tokens[:, tokens.shape[1] // 2:]
        flow = torch.stack(self.unpatchify(self.head(tokens, e_head), grid_sizes)).permute(0, 2, 1, 3, 4)
        return (flow, kv) if return_kv else flow

    def forward_frame_train(self, latent, t, prompt_embeds, context_kv=None, frame=0):
        """Differentiable one-frame forward for self-forcing rollouts.

        latent: ``[B, 1, 16, H, W]`` at timesteps ``t`` (``[B, 1]``); ``context_kv``:
        per layer ``(keys, values)`` lists of ``[B, L_i, H, D]`` chunks covering the
        frames ``< frame`` (``None`` for the first frame). Numerically this is
        ``forward_frame`` with the KV cache replaced by ``context_kv``. Returns the
        flow ``[B, 1, 16, H, W]`` and this frame's per-layer ``(k, v)``.
        """
        if self.freqs.device != latent.device:
            self.freqs = self.freqs.to(latent.device)
        b, _, _, height, width = latent.shape
        x = self.patch_embedding(latent.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        grid_sizes = torch.tensor([[1, height // 2, width // 2]] * b, dtype=torch.long)
        e0, e_head = self.time_conditioning(t, x.dtype)
        context = self.embed_text(prompt_embeds)
        kv = []
        for index, block in enumerate(self.blocks):
            x, k, v = self._run_block(block, x, e0, grid_sizes, self.freqs, context,
                                      None if context_kv is None else context_kv[index], frame=frame)
            kv.append((k, v))
        flow = torch.stack(self.unpatchify(self.head(x, e_head), grid_sizes)).permute(0, 2, 1, 3, 4)
        return flow, kv

    def forward_frame(self, latent, e0, e_head, text_cache, kv_cache, frame, fast=None):
        """Denoise one latent frame ``[B, 1, 16, H, W]`` attending to frames ``< frame``.

        Returns the flow prediction ``[B, 1, 16, H, W]`` and the pre-head hidden
        states ``[B, H*W/4, C]`` consumed by the pixel decoder.
        """
        if self.freqs.device != latent.device:
            self.freqs = self.freqs.to(latent.device)
        b, _, _, height, width = latent.shape
        if fast is None:
            x = self.patch_embedding(latent.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        else:
            x = fast.patch_embed(latent)
        grid_sizes = torch.tensor([[1, height // 2, width // 2]] * b, dtype=torch.long)
        for index, block in enumerate(self.blocks):
            x = block(x, e0, grid_sizes, self.freqs, text_cache[index],
                      kv_cache=kv_cache[index], frame=frame, fast=fast, layer=index)
        tokens = self.head(x, e_head)
        flow = torch.stack(self.unpatchify(tokens, grid_sizes)).permute(0, 2, 1, 3, 4)
        return flow, x
