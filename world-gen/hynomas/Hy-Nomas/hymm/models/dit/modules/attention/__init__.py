import torch
import torch.nn.functional as F
import numpy as np

from .flash_attn_no_pad import flash_attn_no_pad_v3, flash_attn_packed_v3

from hymm.distributed.communications import all_gather, all_to_all_4D
from hymm.distributed.parallel_states import get_sequence_parallel_state, nccl_info

from ..sparse_attention import ssta_3d_attention

try:
    from ..sparse_attention import flex_attention, flex_mask_cache
except ImportError:
    pass

from .magi_attn import (
    magi_mp_causal_attention,
    packed_magi_mp_causal_attention,
    MAGI_ATTENTION_AVAILABLE as BIDIRECTIONAL_MAGI_ATTENTION_AVAILABLE,
)
from . import dynamic_ring_attention as DynRingAttn


@torch.compiler.disable
def parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
):

    # Token-packed block-diagonal attention: each packed sample attends
    # only within itself.  Requires cu_seqlens in attn_param.
    if attn_param is not None and "packed_cu_seqlens_img" in attn_param:
        img_q, txt_q = q
        img_k, txt_k = k
        img_v, txt_v = v
        _img_cu = attn_param["packed_cu_seqlens_img"]
        _txt_cu = attn_param["packed_cu_seqlens_txt"]

        # ---- Sequence Parallel support for token packing ----
        # When SP is active, each rank holds L/P img tokens and all H heads.
        # We do all-to-all (scatter head, gather seq) so that each rank holds
        # the full L img tokens but only H/P heads, then run block-diagonal
        # flash attention with cu_seqlens, then reverse all-to-all.
        _is_sp = get_sequence_parallel_state()
        if _is_sp:
            _text_sequence_parallel = bool(
                attn_param.get("text_sequence_parallel", False)
            )
            # img: [B, L_img/P, H, D] -> [B, L_img, H/P, D]
            img_q = all_to_all_4D(
                img_q, nccl_info.sp_group, scatter_dim=2, gather_dim=1
            )
            img_k = all_to_all_4D(
                img_k, nccl_info.sp_group, scatter_dim=2, gather_dim=1
            )
            img_v = all_to_all_4D(
                img_v, nccl_info.sp_group, scatter_dim=2, gather_dim=1
            )

            if _text_sequence_parallel:
                # txt follows the same persistent sequence-sharded layout as
                # img: [B, L_txt/P, H, D] -> [B, L_txt, H/P, D].
                txt_q = all_to_all_4D(
                    txt_q, nccl_info.sp_group, scatter_dim=2, gather_dim=1
                )
                txt_k = all_to_all_4D(
                    txt_k, nccl_info.sp_group, scatter_dim=2, gather_dim=1
                )
                txt_v = all_to_all_4D(
                    txt_v, nccl_info.sp_group, scatter_dim=2, gather_dim=1
                )
            else:
                # Legacy replicated-text layout: every rank has all text
                # tokens, so keep only this rank's attention heads.
                def shrink_head(t, dim):
                    local_heads = t.shape[dim] // nccl_info.sp_size
                    return t.narrow(
                        dim,
                        nccl_info.rank_within_spgroup * local_heads,
                        local_heads,
                    )

                txt_q = shrink_head(txt_q, dim=2)
                txt_k = shrink_head(txt_k, dim=2)
                txt_v = shrink_head(txt_v, dim=2)

        # Determine packed attention mode
        _packed_mode = attn_param.get("packed_attn_mode", "flash3")
        if attn_param.get("attn_mode_override") is not None:
            _packed_mode = attn_param["attn_mode_override"]

        if _packed_mode == "magi_mp_causal":
            _per_sample_params = attn_param.get("packed_per_sample_magi_params", None)
            assert _per_sample_params is not None, (
                "packed_per_sample_magi_params required for packed magi_mp_causal"
            )
            hidden_states = packed_magi_mp_causal_attention(
                img_q,
                img_k,
                img_v,
                txt_q,
                txt_k,
                txt_v,
                img_cu_seqlens=_img_cu,
                txt_cu_seqlens=_txt_cu,
                per_sample_params=_per_sample_params,
                seg_ar_attention_backend=attn_param.get("seg_ar_attention_backend", "magi"),
            )
        else:
            # Default: flash3 block-diagonal via cu_seqlens
            hidden_states = packed_parallel_attention(
                img_q,
                img_k,
                img_v,
                txt_q,
                txt_k,
                txt_v,
                img_cu_seqlens=_img_cu,
                txt_cu_seqlens=_txt_cu,
            )

        if _is_sp:
            # hidden_states: [1, L_img+L_txt, H/P * D]
            # Split back into img and txt, reverse all-to-all on img,
            # all-gather on txt heads, then re-concat.
            L_img = _img_cu[-1].item()
            L_txt = _txt_cu[-1].item()
            head_dim = img_q.shape[-1]
            local_heads = img_q.shape[2]  # H/P after all-to-all

            # Reshape to 4D for all-to-all: [1, L, H/P*D] -> [1, L, H/P, D]
            hidden_states = hidden_states.view(1, L_img + L_txt, local_heads, head_dim)
            img_out, txt_out = hidden_states.split_with_sizes([L_img, L_txt], dim=1)

            # img: [1, L_img, H/P, D] -> [1, L_img/P, H, D]
            img_out = all_to_all_4D(
                img_out, nccl_info.sp_group, scatter_dim=1, gather_dim=2
            )
            if _text_sequence_parallel:
                # txt: [1, L_txt, H/P, D] -> [1, L_txt/P, H, D]
                txt_out = all_to_all_4D(
                    txt_out,
                    nccl_info.sp_group,
                    scatter_dim=1,
                    gather_dim=2,
                )
            else:
                # txt: [1, L_txt, H/P, D] -> [1, L_txt, H, D]
                txt_out = all_gather(
                    txt_out, dim=2, group=nccl_info.sp_group
                ).contiguous()

            img_out = img_out.to(img_q.dtype)
            txt_out = txt_out.to(img_q.dtype)
            hidden_states = torch.cat([img_out, txt_out], dim=1)

            b, s, a, d = hidden_states.shape
            hidden_states = hidden_states.reshape(b, s, -1)

        return hidden_states

    if nccl_info.use_dynamic_ring_attention:
        return dynamic_ring_attention(q, k, v, text_mask)
    else:
        result = sequence_parallel_attention(
            q, k, v, img_q_len, img_kv_len, attn_mode, text_mask, attn_param=attn_param
        )
        return result


def sequence_parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    # Read unpad_zero_tokens flag (default True = current behaviour: exclude pad tokens)
    _unpad_zero_tokens = True
    if attn_param is not None:
        _unpad_zero_tokens = attn_param.get("unpad_zero_tokens", True)

    if get_sequence_parallel_state():
        _text_sequence_parallel = bool(
            attn_param.get("text_sequence_parallel", False)
            if attn_param is not None
            else False
        )
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, nccl_info.sp_group, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, nccl_info.sp_group, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, nccl_info.sp_group, scatter_dim=2, gather_dim=1)

        if _text_sequence_parallel:
            encoder_query = all_to_all_4D(
                encoder_query,
                nccl_info.sp_group,
                scatter_dim=2,
                gather_dim=1,
            )
            encoder_key = all_to_all_4D(
                encoder_key,
                nccl_info.sp_group,
                scatter_dim=2,
                gather_dim=1,
            )
            encoder_value = all_to_all_4D(
                encoder_value,
                nccl_info.sp_group,
                scatter_dim=2,
                gather_dim=1,
            )
        else:

            def shrink_head(encoder_state, dim):
                local_heads = encoder_state.shape[dim] // nccl_info.sp_size
                return encoder_state.narrow(
                    dim,
                    nccl_info.rank_within_spgroup * local_heads,
                    local_heads,
                )

            encoder_query = shrink_head(encoder_query, dim=2)
            encoder_key = shrink_head(encoder_key, dim=2)
            encoder_value = shrink_head(encoder_value, dim=2)

        # [b, s, h, d]
    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    # 短序列用flash3
    # if sequence_length < 20000:
    #    attn_mode = "flash3"

    if attn_mode in ("flash", "flash3"):
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        if _unpad_zero_tokens:
            # Exclude pad tokens: img positions are always valid (True), text uses text_mask
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            # Include pad tokens: all-True mask so unpad_input keeps every position
            attn_mask = torch.ones(
                query.shape[0], query.shape[1], dtype=torch.bool, device=query.device
            )
        hidden_states = flash_attn_no_pad_v3(
            qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
        )

    elif attn_mode == "magi_mp_causal":
        # Chunk-level causal attention with cumulative text per segment.
        # All params (chunk_size_tokens, prompt_video_boundaries_tokens) are
        # pre-computed in CPTHYNomasDiffusionTransformer.forward().
        #
        # diffusion_forcing_magi_mp produces the same attention pattern as
        # magi_mp_causal with chunk_size_tokens = tokens_per_chunk, so both
        # route through magi_mp_causal_attention (which uses coalesced ranges).
        if attn_param.get("seg_ar_attention_backend", "magi") == "magi" and not BIDIRECTIONAL_MAGI_ATTENTION_AVAILABLE:
            raise RuntimeError(
                "magi_mp_causal attention mode requires MagiAttention library. "
                "Please install magi_attention package."
            )

        num_prompts = attn_param.get("num_prompts", None)
        txt_tokens_per_block = attn_param.get("txt_tokens_per_block", None)
        prompt_video_boundaries_tokens = attn_param.get(
            "prompt_video_boundaries_tokens", None
        )
        chunk_size_tokens = attn_param.get("chunk_size_tokens", 0)

        assert prompt_video_boundaries_tokens is not None, (
            "prompt_video_boundaries_tokens required for magi_mp_causal (pre-compute in model forward)"
        )

        if txt_tokens_per_block is None or txt_tokens_per_block <= 0:
            txt_tokens_per_block = encoder_sequence_length // num_prompts

        hidden_states = magi_mp_causal_attention(
            seg_ar_attention_backend=attn_param.get("seg_ar_attention_backend", "magi"),
            query=query,
            key=key,
            value=value,
            encoder_query=encoder_query,
            encoder_key=encoder_key,
            encoder_value=encoder_value,
            num_prompts=num_prompts,
            txt_tokens_per_block=txt_tokens_per_block,
            prompt_video_boundaries=prompt_video_boundaries_tokens,
            chunk_size_tokens=chunk_size_tokens,
            # Intra-segment per-action masking (one_shot). None -> legacy attention.
            txt_action_start_per_block=attn_param.get(
                "txt_action_start_per_block", None
            ),
            txt_action_offsets_per_block=attn_param.get(
                "txt_action_offsets_per_block", None
            ),
            vid_action_boundaries_per_block=attn_param.get(
                "vid_action_boundaries_per_block", None
            ),
            cross_segment_text_head_only=attn_param.get(
                "cross_segment_text_head_only",
                True,
            ),
            # Previous-segment last-frame video->video restriction (segment-level).
            prev_seg_last_frame_only=attn_param.get("prev_seg_last_frame_only", False),
            tokens_per_frame=attn_param.get("tokens_per_frame", 0),
            # When unpad_zero_tokens=False, pass None so magi includes all pad positions
            # (per_sample_txt_valid=None means no valid-length truncation).
            # When True, pass text_mask for per-sample CFG-safe valid-token truncation.
            text_mask=text_mask if _unpad_zero_tokens else None,
        )

    elif attn_mode == "ptm_sparse_attn":
        sparse_type = attn_param["ptm_attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param.get("ssta_lambda", 0.7)
        ssta_sampling_type = attn_param.get("ssta_sampling_type", "importance")
        ssta_adaptive_pool = attn_param.get("ssta_adaptive_pool", None)
        ptm_attn_pad_type = attn_param["ptm_attn_pad_type"]  # repeat/zero
        ptm_attn_use_text_mask = attn_param["ptm_attn_use_text_mask"]
        ptm_attn_mask_share_within_head = attn_param["ptm_attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        if thw[0] == 1:
            if np.prod(tile_size) == 128:
                tile_size = (1, 16, 8)
            elif np.prod(tile_size) == 384:
                tile_size = (1, 16, 24)
            elif np.prod(tile_size) == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(
                    f"Error tile_size {tile_size}, only support in [64, 128, 384]"
                )
            win_size = [1, 1, 1]
        elif thw[0] <= 31:  # 5s x 16 fps / vae 4
            ssta_topk = ssta_topk // 2  # 5s 视频时，topk减半

        query = torch.cat([query, encoder_query], dim=1).permute(
            0, 2, 1, 3
        )  # convert to (B, H, S, D)
        key = torch.cat([key, encoder_key], dim=1).permute(
            0, 2, 1, 3
        )  # convert to (B, H, S, D)
        value = torch.cat([value, encoder_value], dim=1).permute(
            0, 2, 1, 3
        )  # convert to (B, H, S, D)

        assert query.shape[-1] == 128, (
            "The last dimension of query, key and value must be 128 for PTM Sparse Attention."
        )

        result = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=ptm_attn_pad_type,
            text_mask=text_mask if ptm_attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=ptm_attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = result
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    if get_sequence_parallel_state():
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )
        video_len_before_scatter = hidden_states.shape[1]
        hidden_states = all_to_all_4D(
            hidden_states, nccl_info.sp_group, scatter_dim=1, gather_dim=2
        )
        video_len_after_scatter = hidden_states.shape[1]
        if _text_sequence_parallel:
            encoder_hidden_states = all_to_all_4D(
                encoder_hidden_states,
                nccl_info.sp_group,
                scatter_dim=1,
                gather_dim=2,
            )
        else:
            encoder_hidden_states = all_gather(
                encoder_hidden_states, dim=2, group=nccl_info.sp_group
            ).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states


def dynamic_ring_attention(q, k, v, text_mask=None):
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    text_length = text_mask.sum()

    current_rank = nccl_info.rank_within_spgroup
    cp_group = nccl_info.sp_group
    cp_stream = nccl_info.sp_stream
    cp_rank_list = nccl_info.sp_rank_list

    if current_rank == nccl_info.sp_size - 1:
        part_txt_q = encoder_query[:, 0:text_length, ...]
        part_txt_k = encoder_key[:, 0:text_length, ...]
        part_txt_v = encoder_value[:, 0:text_length, ...]
        query = torch.cat([query, part_txt_q], dim=1)
        key = torch.cat([key, part_txt_k], dim=1)
        value = torch.cat([value, part_txt_v], dim=1)

    cu_seqlens_q = DynRingAttn.cache_cu_seqlens_q
    cu_seqlens_k = DynRingAttn.cache_cu_seqlens_k
    max_seqlen_q = DynRingAttn.cache_max_seqlen_q
    max_seqlen_k = DynRingAttn.cache_max_seqlen_k
    sub_attn = DynRingAttn.attn_forward_func(
        is_training=True,
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        cp_group=cp_group,
        cp_global_ranks=cp_rank_list,
        cp_stream=cp_stream,
    )
    b, s, a, d = query.shape
    sub_attn = sub_attn.reshape(b, s, -1)
    if current_rank == nccl_info.sp_size - 1:
        # sub_seq_len = sub_attn.shape[1]
        img_attn, part_txt_attn = torch.split(
            sub_attn, [s - text_length, text_length], dim=1
        )
    else:
        img_attn = sub_attn
        part_txt_attn = torch.zeros(
            [b, text_length, a * d], dtype=sub_attn.dtype, device=sub_attn.device
        )
    # broadcast
    torch.distributed.broadcast(
        part_txt_attn, src=cp_rank_list[nccl_info.sp_size - 1], group=cp_group
    )
    # b,s,h*d
    txt_attn = torch.zeros_like(encoder_query).reshape(b, encoder_query.shape[1], -1)
    txt_attn[:, 0:text_length, ...] = part_txt_attn
    attn = torch.cat([img_attn, txt_attn], dim=1)
    return attn


def preprocess_dynamic_ring_attention(img, text_mask, freqs_cos, freqs_sin):
    global full_img_seq_len
    global sub_seq_len
    full_img_seq_len = img.shape[1]
    text_seq_len = text_mask.sum().item()
    seq_len = full_img_seq_len + text_seq_len
    sp_size = nccl_info.sp_size
    sp_rank = nccl_info.rank_within_spgroup
    sub_seq_len = last_sub_seq_len = int(seq_len / sp_size)
    if seq_len % sp_size != 0:
        last_sub_seq_len += seq_len % sp_size
    assert last_sub_seq_len > text_seq_len
    last_img_sub_seq_len = last_sub_seq_len - text_seq_len
    split_size = [sub_seq_len] * (sp_size - 1) + [last_img_sub_seq_len]
    img = torch.split(img, split_size, dim=1)[sp_rank]
    freqs_cos = torch.split(freqs_cos, split_size, dim=0)[sp_rank]
    freqs_sin = torch.split(freqs_sin, split_size, dim=0)[sp_rank]
    cu_seqlens = (
        torch.Tensor([[0, sub_seq_len]] * (sp_size - 1) + [[0, last_sub_seq_len]])
        .to(img.device)
        .to(torch.int32)
    )
    max_seqlen = [sub_seq_len] * (sp_size - 1) + [last_sub_seq_len]
    DynRingAttn.cache_metadata(cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
    return img, freqs_cos, freqs_sin


def postprocess_dynamic_ring_attention(img):
    sp_size = nccl_info.sp_size
    sp_rank = nccl_info.rank_within_spgroup
    if sp_rank == sp_size - 1:
        last_img = torch.zeros(
            [img.shape[0], sub_seq_len, img.shape[-1]],
            dtype=img.dtype,
            device=img.device,
        )
        last_img[:, : img.shape[1], ...] = img
        img = last_img
    img = all_gather(img, dim=1, group=nccl_info.sp_group)
    img = img[:, :full_img_seq_len, ...]
    return img


def attention(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad_v3(
        qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None
    )

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


@torch.compiler.disable
def packed_parallel_attention(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_cu_seqlens: torch.Tensor,
    txt_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Block-diagonal flash attention for token-packed sequences.

    Each packed sample has its own img and txt tokens.  This function
    concatenates ``(img_i, txt_i)`` for every sample into a single flat
    sequence and uses ``cu_seqlens`` to enforce block-diagonal attention
    (samples do not attend to each other).

    Args:
        img_q/k/v: ``[1, L_img_total, H, D]`` — packed image tokens.
        txt_q/k/v: ``[1, L_txt_total, H, D]`` — packed text tokens.
        img_cu_seqlens: ``[N+1]`` int32 cumulative img token counts per sample.
        txt_cu_seqlens: ``[N+1]`` int32 cumulative txt token counts per sample.

    Returns:
        hidden_states: ``[1, L_img_total + L_txt_total, H*D]`` with the same
        ordering as the input (all img tokens first, then all txt tokens).
    """
    assert img_q.shape[0] == 1, "packed_parallel_attention expects B=1"
    num_samples = img_cu_seqlens.shape[0] - 1

    # Squeeze batch dim: [1, L, H, D] -> [L, H, D]
    iq, ik, iv = img_q.squeeze(0), img_k.squeeze(0), img_v.squeeze(0)
    tq, tk, tv = txt_q.squeeze(0), txt_k.squeeze(0), txt_v.squeeze(0)

    # Build interleaved sequence: [img_0, txt_0, img_1, txt_1, ...]
    q_parts = []
    k_parts = []
    v_parts = []
    combined_lengths = []

    for i in range(num_samples):
        img_start = img_cu_seqlens[i].item()
        img_end = img_cu_seqlens[i + 1].item()
        txt_start = txt_cu_seqlens[i].item()
        txt_end = txt_cu_seqlens[i + 1].item()

        q_parts.append(iq[img_start:img_end])
        q_parts.append(tq[txt_start:txt_end])
        k_parts.append(ik[img_start:img_end])
        k_parts.append(tk[txt_start:txt_end])
        v_parts.append(iv[img_start:img_end])
        v_parts.append(tv[txt_start:txt_end])
        combined_lengths.append((img_end - img_start) + (txt_end - txt_start))

    q_flat = torch.cat(q_parts, dim=0)  # [L_total, H, D]
    k_flat = torch.cat(k_parts, dim=0)
    v_flat = torch.cat(v_parts, dim=0)

    # Build cu_seqlens for the combined sequence
    cu_lens = [0]
    for length in combined_lengths:
        cu_lens.append(cu_lens[-1] + length)
    cu_seqlens_combined = torch.tensor(cu_lens, dtype=torch.int32, device=img_q.device)
    max_seqlen = max(combined_lengths)

    # Run flash attention with block-diagonal masking
    out_flat = flash_attn_packed_v3(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens=cu_seqlens_combined,
        max_seqlen=max_seqlen,
        causal=False,
    )  # [L_total, H, D]

    # De-interleave: split output back into img and txt parts
    img_out_parts = []
    txt_out_parts = []
    offset = 0
    for i in range(num_samples):
        img_len = img_cu_seqlens[i + 1].item() - img_cu_seqlens[i].item()
        txt_len = txt_cu_seqlens[i + 1].item() - txt_cu_seqlens[i].item()
        img_out_parts.append(out_flat[offset : offset + img_len])
        offset += img_len
        txt_out_parts.append(out_flat[offset : offset + txt_len])
        offset += txt_len

    # Reassemble: [all_img, all_txt] ordering, add batch dim
    img_out = torch.cat(img_out_parts, dim=0)  # [L_img_total, H, D]
    txt_out = torch.cat(txt_out_parts, dim=0)  # [L_txt_total, H, D]
    hidden_states = torch.cat([img_out, txt_out], dim=0)  # [L_total, H, D]
    hidden_states = hidden_states.unsqueeze(0)  # [1, L_total, H, D]

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)  # [1, L_total, H*D]
    return hidden_states


def align_prompt_video_boundaries_to_ar_window(
    prompt_video_boundaries,
    chunk_size,
):
    """Snap each prompt_video_boundary down to the nearest multiple of chunk_size.

    The first boundary (0) is kept as-is. Other boundaries are floored to the
    nearest multiple.

    Example:
        boundaries = [0, 7, 15, 25], chunk_size = 3
        -> [0, 6, 15, 24]
    """
    aligned = []
    for i, b in enumerate(prompt_video_boundaries):
        if i == 0:
            aligned.append(0)
        else:
            aligned.append((b // chunk_size) * chunk_size)
    return aligned
