"""
MagiAttention for multi-prompt segment-level causal attention.

Implements segment-level (or chunk-level) causal attention using the
MagiAttention library (flex_flash_attn_func).

Sequence layout: [video_tokens][txt_prompt_0]...[txt_prompt_{P-1}]

For segment j (0-indexed):
  Q_video = video[b_j : b_{j+1})
  Q_text  = text[j*L : (j+1)*L)
  K/V     = video[0 : b_{j+1}) ∪ text[0 : (j+1)*L)

When chunk_size_tokens > 0, chunks are defined globally across the entire
video instead of per-segment.
"""

from typing import Tuple, Optional, List, Dict
import os
import torch

from .backends import magi_available
from .range_attention import range_attention

MAGI_ATTENTION_AVAILABLE = magi_available()

_magi_mask_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_MAGI_DETERMINISTIC = os.environ.get("MAGI_ATTENTION_DETERMINISTIC_MODE", "0") == "1"


def _normalize_prompt_video_boundaries(
    prompt_video_boundaries: Optional[List[int]],
) -> Optional[List[int]]:
    if prompt_video_boundaries is None:
        return None
    if isinstance(prompt_video_boundaries, torch.Tensor):
        prompt_video_boundaries = prompt_video_boundaries.tolist()
    if len(prompt_video_boundaries) > 0 and isinstance(
        prompt_video_boundaries[0], list
    ):
        prompt_video_boundaries = prompt_video_boundaries[0]
    return [int(b) for b in prompt_video_boundaries]


def _validate_prompt_video_boundaries(
    video_len: int,
    num_prompts: int,
    prompt_video_boundaries: List[int],
) -> None:
    assert len(prompt_video_boundaries) == num_prompts + 1, (
        f"prompt_video_boundaries length ({len(prompt_video_boundaries)}) must be "
        f"num_prompts+1 ({num_prompts + 1})"
    )
    assert (
        prompt_video_boundaries[0] == 0 and prompt_video_boundaries[-1] == video_len
    ), (
        f"boundaries must start at 0 and end at {video_len}, got {prompt_video_boundaries}"
    )
    for i in range(num_prompts):
        left = prompt_video_boundaries[i]
        right = prompt_video_boundaries[i + 1]
        assert 0 <= left <= right <= video_len, (
            f"invalid prompt_video_boundaries[{i}:{i + 2}]={left, right} for video_len={video_len}"
        )


def create_magi_mp_causal_ranges(
    video_len: int,
    num_prompts: int,
    txt_tokens_per_block: int,
    prompt_video_boundaries: List[int],
    chunk_size_tokens: int = 0,
    device: torch.device = None,
    txt_valid_per_block: Optional[List[int]] = None,
    txt_action_start_per_block: Optional[List[int]] = None,
    txt_action_offsets_per_block: Optional[List[Optional[List[int]]]] = None,
    vid_action_boundaries_per_block: Optional[List[Optional[List[List[int]]]]] = None,
    cross_segment_text_head_only: bool = True,
    prev_seg_last_frame_only: bool = False,
    tokens_per_frame: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create q_ranges, k_ranges, attn_type_map for causal attention.

    When chunk_size_tokens == 0 (segment-level causal):
      For segment j: video[b_j:b_{j+1}) sees video[0:b_{j+1}) and text[0:(j+1)*L).

    When chunk_size_tokens > 0 (global chunk-wise causal):
      Chunks are defined globally across the entire video.
      Global chunk i sees video[0:chunk_end_i) and text up to
      the segment that contains chunk i's start position.

    If txt_valid_per_block is provided, only the valid (non-padding) tokens
    within each text block participate in attention. txt_valid_per_block[j]
    gives the number of valid tokens in block j (must be <= txt_tokens_per_block).

    If txt_action_start_per_block is provided, block ``tj``'s text tokens are
    split into a HEAD ``[blk_start, blk_start + action_start)`` (the non-action
    part: scene description etc.) and an ACTION TAIL
    ``[blk_start + action_start, blk_valid_end)`` (the ``contained_actions``
    block, forced to the JSON tail). When ``cross_segment_text_head_only`` is
    true, the action tail is visible ONLY to block ``tj``'s OWN segment (its own
    video segment and its own text block); every OTHER querying segment sees
    just the head. ``action_start == -1`` disables the split for that block
    (whole valid range visible as before).

    INTRA-SEGMENT ACTION MASKING (segment-level causal only, chunk_size_tokens==0):
      ``txt_action_offsets_per_block[tj]`` refines the single action tail into
      per-action slices. It is either ``None`` (no split -> legacy tail
      behavior) or a list ``[o_0, o_1, ..., o_A]`` (post-crop token offsets
      into block tj) where the HEAD is ``[blk_start, blk_start + o_0)`` and
      action ``k`` (0-indexed, ``A`` actions total) occupies
      ``[blk_start + o_k, blk_start + o_{k+1})``.

      ``vid_action_boundaries_per_block[j]`` describes how segment j's video
      window ``[vid_start_j, vid_end_j)`` is partitioned per action. It is
      either ``None`` (video sees the WHOLE own tail -> legacy) or a list of
      ``[vstart, vend, action_idx]`` sub-windows (sample-local video token
      coords, contiguous over the segment) where ``action_idx == -1`` marks a
      GAP (sees the head only) and ``action_idx == k`` restricts video to
      action k's text slice.

      With both provided, within a segment video/text of action1 no longer
      attends the text of action2, while cross-segment head-only masking and
      video->video causality are unchanged.

    PREVIOUS-SEGMENT LAST-FRAME VIDEO MASKING (segment-level causal only,
    chunk_size_tokens==0):
      When ``prev_seg_last_frame_only`` is True, segment j's video AND text
      queries keep FULL attention to the OWN segment video window
      ``[vid_start_j, vid_end_j)`` but attend only the FINAL latent frame
      (``tokens_per_frame`` tokens) of EACH previous segment ``i < j`` instead
      of the whole prefix ``[0, vid_end_j)``. This restricts the video K-ranges
      of both video->video and text->video attention; text->text K-ranges are
      unchanged. Requires ``tokens_per_frame > 0``. No effect when
      ``chunk_size_tokens > 0``.
    """
    _validate_prompt_video_boundaries(
        video_len=video_len,
        num_prompts=num_prompts,
        prompt_video_boundaries=prompt_video_boundaries,
    )

    # If txt_valid_per_block is not provided, assume all tokens are valid.
    if txt_valid_per_block is None:
        txt_valid_per_block = [txt_tokens_per_block] * num_prompts
    assert len(txt_valid_per_block) == num_prompts, (
        f"txt_valid_per_block length ({len(txt_valid_per_block)}) must equal "
        f"num_prompts ({num_prompts})"
    )

    if txt_action_start_per_block is not None:
        assert len(txt_action_start_per_block) == num_prompts, (
            f"txt_action_start_per_block length ({len(txt_action_start_per_block)}) "
            f"must equal num_prompts ({num_prompts})"
        )

    if txt_action_offsets_per_block is not None:
        assert len(txt_action_offsets_per_block) == num_prompts, (
            f"txt_action_offsets_per_block length "
            f"({len(txt_action_offsets_per_block)}) must equal num_prompts "
            f"({num_prompts})"
        )
    if vid_action_boundaries_per_block is not None:
        assert len(vid_action_boundaries_per_block) == num_prompts, (
            f"vid_action_boundaries_per_block length "
            f"({len(vid_action_boundaries_per_block)}) must equal num_prompts "
            f"({num_prompts})"
        )

    text_start = video_len

    def _block_head_len(tj: int, valid: int) -> Optional[int]:
        """Head length (non-action prefix) of block tj, or None if unsplit.

        Prefers the per-action offsets (``o_0``) when present; falls back to
        the legacy ``txt_action_start_per_block``. ``None`` -> no split.
        """
        offs = (
            txt_action_offsets_per_block[tj]
            if txt_action_offsets_per_block is not None
            else None
        )
        if offs is not None and len(offs) >= 1:
            return min(int(offs[0]), valid)
        a_start = (
            txt_action_start_per_block[tj]
            if txt_action_start_per_block is not None
            else -1
        )
        if a_start is None or a_start < 0:
            return None
        return min(int(a_start), valid)

    def _text_k_ranges_for_query(
        query_own_block: int,
        max_block: int,
        own_action_indices: Optional[List[int]] = None,
    ):
        """K-ranges for a query that may attend text blocks 0..max_block.

        ``query_own_block`` is the segment index this query "belongs to"; the
        action tail of a block is only emitted when ``tj == query_own_block``.
        ``own_action_indices`` (intra-segment mode) restricts which action
        slices of the OWN block are exposed: ``None`` -> whole tail (legacy),
        ``[]`` -> head only, ``[k, ...]`` -> head + those action slices.
        Returns a list of ``[start, end]`` pairs (padding-trimmed, split).
        """
        out = []
        for tj in range(max_block + 1):
            blk_start = text_start + tj * txt_tokens_per_block
            valid = txt_valid_per_block[tj]
            blk_valid_end = blk_start + valid
            if blk_valid_end <= blk_start:
                continue
            head_len = _block_head_len(tj, valid)

            if tj != query_own_block:
                # Other segment -> optionally expose only the non-action head.
                if not cross_segment_text_head_only or head_len is None:
                    out.append([blk_start, blk_valid_end])
                else:
                    head_end = blk_start + head_len
                    if head_end > blk_start:
                        out.append([blk_start, head_end])
                continue

            # Own block.
            offs = (
                txt_action_offsets_per_block[tj]
                if txt_action_offsets_per_block is not None
                else None
            )
            if own_action_indices is None or offs is None or len(offs) < 2:
                # Legacy: own segment sees head + full action tail.
                out.append([blk_start, blk_valid_end])
                continue

            # Intra-segment: head + only the selected action slices.
            head_end = blk_start + min(int(offs[0]), valid)
            if head_end > blk_start:
                out.append([blk_start, head_end])
            num_actions = len(offs) - 1
            for aidx in own_action_indices:
                if aidx < 0 or aidx >= num_actions:
                    continue
                s = blk_start + min(int(offs[aidx]), valid)
                e = blk_start + min(int(offs[aidx + 1]), valid)
                if e > s:
                    out.append([s, e])
        return out

    q_ranges_list = []
    k_ranges_list = []
    attn_type_list = []

    def _video_kv_ranges_for_segment(seg_start: int, seg_end: int) -> List[List[int]]:
        """Video K-ranges for a query window ``[seg_start, seg_end)``.

        Used by BOTH video queries and text queries of segment j (their causal
        video window is the same: own segment ``[vid_start_j, vid_end_j)`` plus
        the causal prefix).

        Default (``prev_seg_last_frame_only`` off): the whole causal prefix
        ``[0, seg_end)`` as a single range (legacy behavior).

        With ``prev_seg_last_frame_only`` on: the OWN segment window
        ``[seg_start, seg_end)`` in full, plus only the FINAL ``tokens_per_frame``
        tokens (one latent frame) of EACH previous segment ``i`` (boundaries
        ``prompt_video_boundaries[1..]`` that are <= seg_start).
        """
        if not prev_seg_last_frame_only or tokens_per_frame <= 0:
            return [[0, seg_end]]
        ranges: List[List[int]] = []
        # Last latent frame of each previous segment (boundary b_i with b_i <= seg_start).
        for b in prompt_video_boundaries[1:]:
            b = int(b)
            if b <= 0 or b > seg_start:
                continue
            fs = max(0, b - tokens_per_frame)
            if b > fs:
                ranges.append([fs, b])
        # Own segment, in full (keeps within-segment bidirectional/causal context).
        if seg_end > seg_start:
            ranges.append([seg_start, seg_end])
        # Fallback safety: never emit an empty video K set.
        if not ranges:
            ranges.append([0, seg_end])
        return ranges

    if chunk_size_tokens > 0:
        total_video_tokens = video_len
        num_global_chunks = (
            total_video_tokens + chunk_size_tokens - 1
        ) // chunk_size_tokens

        last_chunk_size = (
            total_video_tokens - (num_global_chunks - 1) * chunk_size_tokens
        )
        if num_global_chunks > 1 and last_chunk_size < chunk_size_tokens // 2:
            num_global_chunks -= 1

        for c in range(num_global_chunks):
            chunk_start = c * chunk_size_tokens
            chunk_end = min(chunk_start + chunk_size_tokens, total_video_tokens)

            q_ranges_list.append([chunk_start, chunk_end])
            k_ranges_list.append([0, chunk_end])
            attn_type_list.append(0)

            seg_j = 0
            for j in range(num_prompts):
                if chunk_start >= prompt_video_boundaries[j]:
                    seg_j = j
            # The chunk "belongs to" segment seg_j: only seg_j's action tail is
            # visible to it; earlier blocks contribute head only.
            for blk_start, blk_end in _text_k_ranges_for_query(seg_j, seg_j):
                q_ranges_list.append([chunk_start, chunk_end])
                k_ranges_list.append([blk_start, blk_end])
                attn_type_list.append(0)

        for j in range(num_prompts):
            txt_start_j = text_start + j * txt_tokens_per_block
            txt_valid_end_j = txt_start_j + txt_valid_per_block[j]
            vid_end_j = prompt_video_boundaries[j + 1]

            if txt_valid_end_j > txt_start_j:
                if vid_end_j > 0:
                    q_ranges_list.append([txt_start_j, txt_valid_end_j])
                    k_ranges_list.append([0, vid_end_j])
                    attn_type_list.append(0)

                # Text block j attends to valid tokens in blocks 0..j; only its
                # own block (tj == j) exposes the action tail.
                for blk_start, blk_end in _text_k_ranges_for_query(j, j):
                    q_ranges_list.append([txt_start_j, txt_valid_end_j])
                    k_ranges_list.append([blk_start, blk_end])
                    attn_type_list.append(0)
    else:
        for j in range(num_prompts):
            vid_start_j = prompt_video_boundaries[j]
            vid_end_j = prompt_video_boundaries[j + 1]

            txt_start_j = text_start + j * txt_tokens_per_block
            txt_valid_end_j = txt_start_j + txt_valid_per_block[j]

            cum_vid_end = vid_end_j

            # Intra-segment per-action partitioning of segment j (only when the
            # dataset provided both the per-action video sub-windows AND the
            # per-action text offsets with >=1 action). Otherwise fall back to
            # the legacy whole-tail behavior.
            offs_j = (
                txt_action_offsets_per_block[j]
                if txt_action_offsets_per_block is not None
                else None
            )
            vid_windows_j = (
                vid_action_boundaries_per_block[j]
                if vid_action_boundaries_per_block is not None
                else None
            )
            intra_segment = (
                offs_j is not None
                and len(offs_j) >= 2
                and vid_windows_j is not None
                and len(vid_windows_j) > 0
            )

            if vid_end_j > vid_start_j:
                for _vs, _ve in _video_kv_ranges_for_segment(vid_start_j, cum_vid_end):
                    q_ranges_list.append([vid_start_j, vid_end_j])
                    k_ranges_list.append([_vs, _ve])
                    attn_type_list.append(0)

                if not intra_segment:
                    # Video segment j attends valid text in blocks 0..j; only its
                    # own block (tj == j) exposes the (whole) action tail.
                    for blk_start, blk_end in _text_k_ranges_for_query(j, j):
                        q_ranges_list.append([vid_start_j, vid_end_j])
                        k_ranges_list.append([blk_start, blk_end])
                        attn_type_list.append(0)
                else:
                    # Each action sub-window of segment j's video attends heads
                    # of blocks 0..j plus ONLY its own action's text slice; gap
                    # sub-windows (action_idx == -1) see heads only.
                    for win in vid_windows_j:
                        ws, we, aidx = int(win[0]), int(win[1]), int(win[2])
                        if we <= ws:
                            continue
                        own = [] if aidx < 0 else [aidx]
                        for blk_start, blk_end in _text_k_ranges_for_query(
                            j, j, own_action_indices=own
                        ):
                            q_ranges_list.append([ws, we])
                            k_ranges_list.append([blk_start, blk_end])
                            attn_type_list.append(0)

            if txt_valid_end_j > txt_start_j:
                if cum_vid_end > 0:
                    # Text block j attends video. With prev_seg_last_frame_only
                    # ON, text is restricted exactly like video queries: its own
                    # segment window [vid_start_j, vid_end_j) in full, plus only
                    # the final latent frame of each previous segment (rather
                    # than the whole prefix [0, cum_vid_end)).
                    for _vs, _ve in _video_kv_ranges_for_segment(
                        vid_start_j, cum_vid_end
                    ):
                        q_ranges_list.append([txt_start_j, txt_valid_end_j])
                        k_ranges_list.append([_vs, _ve])
                        attn_type_list.append(0)

                if not intra_segment:
                    # Text block j attends valid text in blocks 0..j; only its own
                    # block (tj == j) exposes the (whole) action tail.
                    for blk_start, blk_end in _text_k_ranges_for_query(j, j):
                        q_ranges_list.append([txt_start_j, txt_valid_end_j])
                        k_ranges_list.append([blk_start, blk_end])
                        attn_type_list.append(0)
                else:
                    # Symmetric text->text masking: split block j's text query
                    # into head + per-action pieces. Head text sees heads 0..j
                    # only; action-k text sees heads 0..j + its own action-k
                    # slice. valid = txt_valid_per_block[j].
                    valid_j = txt_valid_per_block[j]
                    num_actions_j = len(offs_j) - 1
                    # Head-text query piece.
                    head_end_tok = txt_start_j + min(int(offs_j[0]), valid_j)
                    if head_end_tok > txt_start_j:
                        for blk_start, blk_end in _text_k_ranges_for_query(
                            j, j, own_action_indices=[]
                        ):
                            q_ranges_list.append([txt_start_j, head_end_tok])
                            k_ranges_list.append([blk_start, blk_end])
                            attn_type_list.append(0)
                    # Per-action text query pieces.
                    for aidx in range(num_actions_j):
                        qs = txt_start_j + min(int(offs_j[aidx]), valid_j)
                        qe = txt_start_j + min(int(offs_j[aidx + 1]), valid_j)
                        if qe <= qs:
                            continue
                        for blk_start, blk_end in _text_k_ranges_for_query(
                            j, j, own_action_indices=[aidx]
                        ):
                            q_ranges_list.append([qs, qe])
                            k_ranges_list.append([blk_start, blk_end])
                            attn_type_list.append(0)

    assert len(q_ranges_list) > 0, "MagiMP Causal produced empty attention ranges."
    q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
    k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
    attn_type_map = torch.tensor(attn_type_list, dtype=torch.int32, device=device)
    return q_ranges, k_ranges, attn_type_map


def get_cached_magi_mp_causal_ranges(
    video_len: int,
    num_prompts: int,
    txt_tokens_per_block: int,
    prompt_video_boundaries: List[int],
    chunk_size_tokens: int = 0,
    device: torch.device = None,
    txt_valid_per_block: Optional[List[int]] = None,
    txt_action_start_per_block: Optional[List[int]] = None,
    txt_action_offsets_per_block: Optional[List[Optional[List[int]]]] = None,
    vid_action_boundaries_per_block: Optional[List[Optional[List[List[int]]]]] = None,
    cross_segment_text_head_only: bool = True,
    prev_seg_last_frame_only: bool = False,
    tokens_per_frame: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get cached or create MagiMP Causal ranges."""
    boundaries_key = "_".join(str(b) for b in prompt_video_boundaries)
    valid_key = (
        "_".join(str(v) for v in txt_valid_per_block)
        if txt_valid_per_block is not None
        else "none"
    )
    action_key = (
        "_".join(str(a) for a in txt_action_start_per_block)
        if txt_action_start_per_block is not None
        else "none"
    )
    offs_key = (
        "|".join(
            ("n" if o is None else "-".join(str(x) for x in o))
            for o in txt_action_offsets_per_block
        )
        if txt_action_offsets_per_block is not None
        else "none"
    )
    vwin_key = (
        "|".join(
            ("n" if w is None else ";".join(",".join(str(x) for x in win) for win in w))
            for w in vid_action_boundaries_per_block
        )
        if vid_action_boundaries_per_block is not None
        else "none"
    )
    cache_key = (
        f"magi_mp_causal_{video_len}_{num_prompts}_{txt_tokens_per_block}_"
        f"{boundaries_key}_chunk{chunk_size_tokens}_valid{valid_key}_act{action_key}_"
        f"offs{offs_key}_vwin{vwin_key}_crosshead{int(cross_segment_text_head_only)}_"
        f"prevlast{int(bool(prev_seg_last_frame_only))}_tpf{tokens_per_frame}"
    )

    if cache_key not in _magi_mask_cache:
        q_ranges, k_ranges, attn_type_map = create_magi_mp_causal_ranges(
            video_len=video_len,
            num_prompts=num_prompts,
            txt_tokens_per_block=txt_tokens_per_block,
            prompt_video_boundaries=prompt_video_boundaries,
            chunk_size_tokens=chunk_size_tokens,
            device=device,
            txt_valid_per_block=txt_valid_per_block,
            txt_action_start_per_block=txt_action_start_per_block,
            txt_action_offsets_per_block=txt_action_offsets_per_block,
            vid_action_boundaries_per_block=vid_action_boundaries_per_block,
            cross_segment_text_head_only=cross_segment_text_head_only,
            prev_seg_last_frame_only=prev_seg_last_frame_only,
            tokens_per_frame=tokens_per_frame,
        )
        _magi_mask_cache[cache_key] = (q_ranges, k_ranges, attn_type_map)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank == 0:
            print(
                f"==> [rank-{rank}] MagiMP Causal Attn created new mask: {cache_key}, "
                f"video_len={video_len}, num_prompts={num_prompts}, "
                f"txt_tokens_per_block={txt_tokens_per_block}, "
                f"boundaries={prompt_video_boundaries}, num_ranges={len(q_ranges)}"
            )

    cached = _magi_mask_cache[cache_key]
    if device is not None and cached[0].device != device:
        cached = (
            cached[0].to(device),
            cached[1].to(device),
            cached[2].to(device),
        )
        _magi_mask_cache[cache_key] = cached
    return cached


def magi_mp_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    encoder_query: torch.Tensor,
    encoder_key: torch.Tensor,
    encoder_value: torch.Tensor,
    num_prompts: Optional[int] = None,
    txt_tokens_per_block: Optional[int] = None,
    prompt_video_boundaries: Optional[List[int]] = None,
    chunk_size_tokens: int = 0,
    softmax_scale: Optional[float] = None,
    text_mask: Optional[torch.Tensor] = None,
    txt_action_start_per_block: Optional[List[int]] = None,
    txt_action_offsets_per_block: Optional[List[Optional[List[int]]]] = None,
    vid_action_boundaries_per_block: Optional[List[Optional[List[List[int]]]]] = None,
    cross_segment_text_head_only: bool = True,
    prev_seg_last_frame_only: bool = False,
    tokens_per_frame: int = 0,
    seg_ar_attention_backend: str = "magi",
) -> torch.Tensor:
    """Compute segment-level (or chunk-level) causal attention using MagiAttention.

    When chunk_size_tokens == 0: segment-level causal (segment j sees segments 0..j).
    When chunk_size_tokens > 0: global chunk-wise causal (chunk i sees chunks 0..i).
    """
    if seg_ar_attention_backend == "magi" and not MAGI_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "magi_mp_causal_attention requires MagiAttention library. "
            "Please install magi_attention package."
        )

    if seg_ar_attention_backend == "flex" and any(
        tensor.dtype != torch.bfloat16 for tensor in
        (query, key, value, encoder_query, encoder_key, encoder_value)
    ):
        raise ValueError("Flex range adapters require BF16 inputs")
    B, S_video, _, _ = query.shape
    S_text = encoder_query.shape[1]

    prompt_video_boundaries = _normalize_prompt_video_boundaries(
        prompt_video_boundaries
    )
    assert prompt_video_boundaries is not None, (
        "prompt_video_boundaries must be provided for magi_mp_causal_attention."
    )

    inferred_num_prompts = len(prompt_video_boundaries) - 1
    if num_prompts is None:
        num_prompts = inferred_num_prompts
    else:
        assert num_prompts == inferred_num_prompts, (
            f"num_prompts ({num_prompts}) must equal len(prompt_video_boundaries)-1 "
            f"({inferred_num_prompts})"
        )

    assert num_prompts > 0, f"num_prompts must be positive, got {num_prompts}"

    if txt_tokens_per_block is None or txt_tokens_per_block <= 0:
        txt_tokens_per_block = S_text // num_prompts

    _validate_prompt_video_boundaries(
        video_len=S_video,
        num_prompts=num_prompts,
        prompt_video_boundaries=prompt_video_boundaries,
    )

    # Compute per-block valid token counts from text_mask, PER SAMPLE in the batch.
    # text_mask: [B, S_text] bool, True = valid, False = padding.
    # BUG FIX: previously used text_mask[0] (first sample) for the entire batch.
    # Under CFG, batch index 0 is the negative prompt ("") which has very few valid
    # text tokens, causing the positive prompt's attention ranges to be truncated
    # to the negative's valid length — effectively masking out almost all text
    # tokens for the positive sample. Now compute per-sample valid counts and
    # dispatch one flex_flash_attn_func call per sample with its own ranges.
    per_sample_txt_valid = None
    if text_mask is not None:
        per_sample_txt_valid = []
        for b in range(B):
            mask_1d_b = text_mask[b]  # [S_text]
            valid_b = []
            for j in range(num_prompts):
                blk_start = j * txt_tokens_per_block
                blk_end = blk_start + txt_tokens_per_block
                valid_count = int(mask_1d_b[blk_start:blk_end].sum().item())
                valid_b.append(valid_count)
            per_sample_txt_valid.append(valid_b)

    q_total = torch.cat([query, encoder_query], dim=1)
    k_total = torch.cat([key, encoder_key], dim=1)
    v_total = torch.cat([value, encoder_value], dim=1)

    # Flash Attention only supports fp16/bf16/fp8; cast from fp32 when needed.
    _orig_dtype = q_total.dtype
    if _orig_dtype not in (torch.float16, torch.bfloat16):
        q_total = q_total.to(torch.bfloat16)
        k_total = k_total.to(torch.bfloat16)
        v_total = v_total.to(torch.bfloat16)

    if B == 1:
        txt_valid_per_block = (
            per_sample_txt_valid[0] if per_sample_txt_valid is not None else None
        )
        q_ranges, k_ranges, attn_type_map = get_cached_magi_mp_causal_ranges(
            video_len=S_video,
            num_prompts=num_prompts,
            txt_tokens_per_block=txt_tokens_per_block,
            prompt_video_boundaries=prompt_video_boundaries,
            chunk_size_tokens=chunk_size_tokens,
            device=query.device,
            txt_valid_per_block=txt_valid_per_block,
            txt_action_start_per_block=txt_action_start_per_block,
            txt_action_offsets_per_block=txt_action_offsets_per_block,
            vid_action_boundaries_per_block=vid_action_boundaries_per_block,
            cross_segment_text_head_only=cross_segment_text_head_only,
            prev_seg_last_frame_only=prev_seg_last_frame_only,
            tokens_per_frame=tokens_per_frame,
        )
        q_flat = q_total.squeeze(0)
        k_flat = k_total.squeeze(0)
        v_flat = v_total.squeeze(0)

        out_flat, lse = range_attention(
            backend=seg_ar_attention_backend,
            q=q_flat,
            k=k_flat,
            v=v_flat,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            softmax_scale=softmax_scale,
            deterministic=_MAGI_DETERMINISTIC,
        )
        hidden_states = out_flat.unsqueeze(0)
    else:
        outputs = []
        for b in range(B):
            # Compute per-sample ranges so each sample's attention respects its
            # own text-mask valid lengths (critical for CFG where neg/pos differ).
            txt_valid_b = (
                per_sample_txt_valid[b] if per_sample_txt_valid is not None else None
            )
            # Action params describe the POSITIVE sample; under CFG the batch is
            # [neg, pos] (pos == last row). The negative row has an empty caption
            # (no actions) -> None -> legacy attention, matching training dropout.
            _is_pos_row = b == B - 1
            _tas_b = txt_action_start_per_block if _is_pos_row else None
            _off_b = txt_action_offsets_per_block if _is_pos_row else None
            _vwin_b = vid_action_boundaries_per_block if _is_pos_row else None
            q_ranges_b, k_ranges_b, attn_type_map_b = get_cached_magi_mp_causal_ranges(
                video_len=S_video,
                num_prompts=num_prompts,
                txt_tokens_per_block=txt_tokens_per_block,
                prompt_video_boundaries=prompt_video_boundaries,
                chunk_size_tokens=chunk_size_tokens,
                device=query.device,
                txt_valid_per_block=txt_valid_b,
                txt_action_start_per_block=_tas_b,
                txt_action_offsets_per_block=_off_b,
                vid_action_boundaries_per_block=_vwin_b,
                cross_segment_text_head_only=cross_segment_text_head_only,
                prev_seg_last_frame_only=prev_seg_last_frame_only,
                tokens_per_frame=tokens_per_frame,
            )
            out_b, lse = range_attention(
                backend=seg_ar_attention_backend,
                q=q_total[b],
                k=k_total[b],
                v=v_total[b],
                q_ranges=q_ranges_b,
                k_ranges=k_ranges_b,
                attn_type_map=attn_type_map_b,
                softmax_scale=softmax_scale,
                deterministic=_MAGI_DETERMINISTIC,
            )
            outputs.append(out_b)
        hidden_states = torch.stack(outputs, dim=0)

    # Cast output back to the original dtype if we cast the inputs.
    if hidden_states.dtype != _orig_dtype:
        hidden_states = hidden_states.to(_orig_dtype)

    return hidden_states


# ---------------------------------------------------------------------------
# Packed (token-packing) variant of MagiMP causal attention
# ---------------------------------------------------------------------------


def create_packed_magi_ranges(
    per_sample_params: List[Dict],
    img_cu_seqlens: List[int],
    txt_cu_seqlens: List[int],
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build merged magi ranges for a token-packed flat sequence.

    The flat sequence layout is ``[img_0, txt_0, img_1, txt_1, ...]``.
    For each sample *i* we generate the standard magi causal ranges
    (as if it were a standalone sequence ``[video_i | text_i]``) and then
    offset every range by the sample's starting position in the flat
    sequence.

    Args:
        per_sample_params: One dict per packed sample, each containing:
            - ``video_len``: number of video tokens for this sample.
            - ``num_prompts``: number of text prompts.
            - ``txt_tokens_per_block``: text tokens per prompt block.
            - ``prompt_video_boundaries``: list[int] of length num_prompts+1
              in **token** units (relative to this sample's video).
            - ``chunk_size_tokens``: (optional, default 0) chunk size for
              global chunk-wise causal.
            - ``txt_valid_per_block``: (optional) list[int] valid text tokens
              per block.
        img_cu_seqlens: cumulative img token counts ``[0, L_img0, L_img0+L_img1, ...]``.
        txt_cu_seqlens: cumulative txt token counts ``[0, L_txt0, L_txt0+L_txt1, ...]``.
        device: target device.

    Returns:
        (q_ranges, k_ranges, attn_type_map) — merged across all samples,
        ready for a single ``flex_flash_attn_func`` call.
    """
    all_q = []
    all_k = []
    all_at = []

    # Running offset in the flat interleaved sequence
    flat_offset = 0

    for i, params in enumerate(per_sample_params):
        video_len_i = params["video_len"]
        num_prompts_i = params["num_prompts"]
        txt_tokens_per_block_i = params["txt_tokens_per_block"]
        pvb_i = params["prompt_video_boundaries"]
        chunk_size_tokens_i = params.get("chunk_size_tokens", 0)
        txt_valid_i = params.get("txt_valid_per_block", None)
        txt_action_start_i = params.get("txt_action_start_per_block", None)
        txt_action_offsets_i = params.get("txt_action_offsets_per_block", None)
        vid_action_boundaries_i = params.get("vid_action_boundaries_per_block", None)
        cross_segment_text_head_only_i = params.get(
            "cross_segment_text_head_only",
            True,
        )
        prev_seg_last_frame_only_i = params.get("prev_seg_last_frame_only", False)
        tokens_per_frame_i = params.get("tokens_per_frame", 0)

        # Token-packing length consistency: the action-start list (when present)
        # must be 1:1 with this sample's prompt blocks, and every start must lie
        # within the block's valid length so we never split past real tokens.
        if txt_action_start_i is not None:
            assert len(txt_action_start_i) == num_prompts_i, (
                f"[packed] sample {i}: txt_action_start_per_block length "
                f"({len(txt_action_start_i)}) != num_prompts ({num_prompts_i})"
            )
            if txt_valid_i is not None:
                for _j, _a in enumerate(txt_action_start_i):
                    if _a is not None and _a >= 0:
                        assert _a <= txt_valid_i[_j], (
                            f"[packed] sample {i} block {_j}: action_start {_a} > "
                            f"valid_len {txt_valid_i[_j]}"
                        )

        # Generate ranges for this sample as if it were standalone
        q_r, k_r, at_r = create_magi_mp_causal_ranges(
            video_len=video_len_i,
            num_prompts=num_prompts_i,
            txt_tokens_per_block=txt_tokens_per_block_i,
            prompt_video_boundaries=pvb_i,
            chunk_size_tokens=chunk_size_tokens_i,
            device=device,
            txt_valid_per_block=txt_valid_i,
            txt_action_start_per_block=txt_action_start_i,
            txt_action_offsets_per_block=txt_action_offsets_i,
            vid_action_boundaries_per_block=vid_action_boundaries_i,
            cross_segment_text_head_only=cross_segment_text_head_only_i,
            prev_seg_last_frame_only=prev_seg_last_frame_only_i,
            tokens_per_frame=tokens_per_frame_i,
        )

        # Offset ranges into the flat sequence.
        # In the flat layout, sample i occupies [flat_offset, flat_offset + video_len_i + txt_len_i).
        # Within that region, video is at [flat_offset, flat_offset + video_len_i)
        # and text is at [flat_offset + video_len_i, flat_offset + video_len_i + txt_len_i).
        # The standalone ranges already use layout [video | text] with text_start = video_len_i,
        # so we just add flat_offset to all ranges.
        q_r = q_r + flat_offset
        k_r = k_r + flat_offset

        all_q.append(q_r)
        all_k.append(k_r)
        all_at.append(at_r)

        txt_len_i = txt_cu_seqlens[i + 1] - txt_cu_seqlens[i]
        flat_offset += video_len_i + txt_len_i

    return (
        torch.cat(all_q, dim=0),
        torch.cat(all_k, dim=0),
        torch.cat(all_at, dim=0),
    )


def packed_magi_mp_causal_attention(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_cu_seqlens: torch.Tensor,
    txt_cu_seqlens: torch.Tensor,
    per_sample_params: List[Dict],
    softmax_scale: Optional[float] = None,
    seg_ar_attention_backend: str = "magi",
) -> torch.Tensor:
    """Packed MagiMP causal attention for token-packed sequences.

    Interleaves each sample's ``(img_i, txt_i)`` into a single flat
    sequence, builds merged magi ranges with correct offsets, runs one
    ``flex_flash_attn_func`` call, then de-interleaves the output back
    to ``[all_img | all_txt]`` ordering.

    Args:
        img_q/k/v: ``[1, L_img_total, H, D]`` packed image tokens.
        txt_q/k/v: ``[1, L_txt_total, H, D]`` packed text tokens.
        img_cu_seqlens: ``[N+1]`` int32 cumulative img token counts.
        txt_cu_seqlens: ``[N+1]`` int32 cumulative txt token counts.
        per_sample_params: Per-sample magi parameters (see
            :func:`create_packed_magi_ranges`).
        softmax_scale: Optional softmax scaling factor.

    Returns:
        hidden_states: ``[1, L_img_total + L_txt_total, H*D]``
    """
    if seg_ar_attention_backend == "magi" and not MAGI_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "packed_magi_mp_causal_attention requires MagiAttention library."
        )
    if seg_ar_attention_backend == "flex" and any(
        tensor.dtype != torch.bfloat16 for tensor in (img_q, img_k, img_v, txt_q, txt_k, txt_v)
    ):
        raise ValueError("Flex packed adapters require BF16 inputs")
    assert img_q.shape[0] == 1, "packed_magi_mp_causal_attention expects B=1"
    num_samples = img_cu_seqlens.shape[0] - 1

    # Squeeze batch dim
    iq, ik, iv = img_q.squeeze(0), img_k.squeeze(0), img_v.squeeze(0)
    tq, tk, tv = txt_q.squeeze(0), txt_k.squeeze(0), txt_v.squeeze(0)

    # Interleave: [img_0, txt_0, img_1, txt_1, ...]
    q_parts, k_parts, v_parts = [], [], []
    for i in range(num_samples):
        is_ = img_cu_seqlens[i].item()
        ie_ = img_cu_seqlens[i + 1].item()
        ts_ = txt_cu_seqlens[i].item()
        te_ = txt_cu_seqlens[i + 1].item()
        q_parts.extend([iq[is_:ie_], tq[ts_:te_]])
        k_parts.extend([ik[is_:ie_], tk[ts_:te_]])
        v_parts.extend([iv[is_:ie_], tv[ts_:te_]])

    q_flat = torch.cat(q_parts, dim=0)
    k_flat = torch.cat(k_parts, dim=0)
    v_flat = torch.cat(v_parts, dim=0)

    # Flash Attention only supports fp16/bf16/fp8; cast from fp32 when needed.
    _orig_dtype = q_flat.dtype
    if _orig_dtype not in (torch.float16, torch.bfloat16):
        q_flat = q_flat.to(torch.bfloat16)
        k_flat = k_flat.to(torch.bfloat16)
        v_flat = v_flat.to(torch.bfloat16)

    # Build merged ranges
    img_cu_list = img_cu_seqlens.tolist()
    txt_cu_list = txt_cu_seqlens.tolist()
    q_ranges, k_ranges, attn_type_map = create_packed_magi_ranges(
        per_sample_params=per_sample_params,
        img_cu_seqlens=img_cu_list,
        txt_cu_seqlens=txt_cu_list,
        device=img_q.device,
    )

    # Single fused flex_flash_attn_func call
    out_flat, _lse = range_attention(
        backend=seg_ar_attention_backend,
        q=q_flat,
        k=k_flat,
        v=v_flat,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=softmax_scale,
        deterministic=_MAGI_DETERMINISTIC,
    )

    # De-interleave back to [all_img | all_txt]
    img_out_parts, txt_out_parts = [], []
    offset = 0
    for i in range(num_samples):
        il = img_cu_seqlens[i + 1].item() - img_cu_seqlens[i].item()
        tl = txt_cu_seqlens[i + 1].item() - txt_cu_seqlens[i].item()
        img_out_parts.append(out_flat[offset : offset + il])
        offset += il
        txt_out_parts.append(out_flat[offset : offset + tl])
        offset += tl

    img_out = torch.cat(img_out_parts, dim=0)
    txt_out = torch.cat(txt_out_parts, dim=0)
    hidden_states = torch.cat([img_out, txt_out], dim=0).unsqueeze(0)  # [1, L, H, D]

    # Cast output back to the original dtype if we cast the inputs.
    if hidden_states.dtype != _orig_dtype:
        hidden_states = hidden_states.to(_orig_dtype)

    b, s, a, d = hidden_states.shape
    return hidden_states.reshape(b, s, -1)  # [1, L, H*D]
