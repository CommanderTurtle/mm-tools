"""Packed sequence-parallel attention as an ``nn.Module`` (hook-based Ulysses).

This wraps the block-diagonal packed attention kernels in the Ulysses
sequence-parallel collectives. The heavy lifting (the all-to-all / all-gather
that convert between the sequence-sharded ``Shard(1)`` layout the rest of the
model uses and the head-sharded ``Shard(2)`` layout the attention kernel needs)
lives in two forward hooks, so ``forward`` itself is just a call to the chosen
packed kernel. See :class:`SequenceParallelAttention` for the full contract.
"""

import torch

from hymm.distributed.communications import all_gather, all_to_all_4D
from hymm.distributed.parallel_states import nccl_info

from . import packed_parallel_attention
from .magi_attn import packed_magi_mp_causal_attention


def shrink_head(t, dim):
    """Slice ``t`` down to this rank's contiguous head shard along ``dim``.

    Used for the replicated-text branch: every rank starts with all ``H`` heads,
    and this narrows to the ``H / sp_size`` heads owned by the current rank so
    the text lines up with the head-parallel image tokens. A ``narrow`` (view)
    rather than a collective, so its autograd backward is a pad — the input
    gradient stays partial-per-rank instead of being summed ``sp_size`` times.
    """
    local_heads = t.shape[dim] // nccl_info.sp_size
    return t.narrow(
        dim,
        nccl_info.rank_within_spgroup * local_heads,
        local_heads,
    )


class SequenceParallelAttention(torch.nn.Module):
    """Packed sequence-parallel attention as an ``nn.Module``, equivalent to
    ``parallel_attention``'s packed path.

    ``forward`` runs the selected packed attention backend on head-parallel
    shards. Two backends are supported via ``packed_attn_mode``:

    * ``"flash3"`` (default) — ``packed_parallel_attention``, block-diagonal
      flash attention (each packed sample attends only within itself).
    * ``"magi_mp_causal"`` — ``packed_magi_mp_causal_attention``, multi-prompt
      segment/chunk-level causal attention. Requires ``per_sample_params``
      (one dict per packed sample; see ``packed_magi_mp_causal_attention``).

    The backend runs on the full sequence with ``H/P`` heads on every rank, so
    it computes exactly the non-sequence-parallel result restricted to this
    rank's head shard. The Ulysses collectives are carried by two forward hooks
    installed in ``__init__``:

    * **before forward** (``forward_pre_hook``): all-to-all the image q/k/v from
      sequence-parallel ``Shard(1)`` to head-parallel ``Shard(2)`` (scatter head,
      gather seq). Text q/k/v are converted the same way when ``txt_seq_sharded``
      is True; otherwise the replicated text (all ``H`` heads) is sliced to this
      rank's ``H/P`` head shard (``Replicate`` -> ``Shard(2)``) so it lines up with
      the image heads.
    * **after forward** (``forward_hook``): all-to-all the image output back to
      ``Shard(1)``. The text output is all-to-all'd back to ``Shard(1)`` when
      ``txt_seq_sharded`` is True, else all-gathered over heads to ``Replicate``.
      Both branches are reshaped to 3D ``[1, L_x, H*D]`` to match
      ``parallel_attention``.

    No backward hooks are installed: ``all_to_all_4D`` and ``all_gather`` are
    differentiable ``autograd.Function`` s, so the backward pass automatically
    runs their adjoints (reverse all-to-all; reduce_scatter-SUM for the
    replicated-text all-gather). The replicated-text input gradient therefore
    stays partial-per-rank (no x P over-count under FSDP), exactly as in
    ``parallel_attention``.

    Inputs to ``__call__`` are the per-rank local tensors
    ``(img_q, img_k, img_v, txt_q, txt_k, txt_v, img_cu_seqlens, txt_cu_seqlens)``;
    it returns ``(img_out, txt_out)`` local tensors, matching ``parallel_attention``.
    """

    def __init__(
        self,
        txt_seq_sharded: bool = False,
        packed_attn_mode: str = "flash3",
        per_sample_params=None,
        seg_ar_attention_backend="magi",
    ):
        """Install the Ulysses collective hooks and pin the attention backend.

        Args:
            txt_seq_sharded: text placement across the SP group. ``False`` =>
                text is ``Replicate()`` (every rank holds all text tokens and is
                narrowed to its head shard); ``True`` => text is ``Shard(1)``
                like the image (each rank holds ``L_txt / sp_size`` tokens and is
                all-to-all'd). Selects the collective used for text in both hooks.
            packed_attn_mode: ``"flash3"`` (block-diagonal flash attention) or
                ``"magi_mp_causal"`` (multi-prompt causal). The hooks are
                identical for both; only ``forward``'s kernel differs.
            per_sample_params: required for ``"magi_mp_causal"`` — one dict per
                packed sample describing its prompt/video layout (see
                :func:`packed_magi_mp_causal_attention`). Ignored for flash3.
        """
        super().__init__()
        self.txt_seq_sharded = txt_seq_sharded
        self.packed_attn_mode = packed_attn_mode
        if seg_ar_attention_backend not in ("magi", "flex"):
            raise ValueError("Invalid seg_ar_attention_backend")
        self.seg_ar_attention_backend = seg_ar_attention_backend
        self.per_sample_params = per_sample_params
        if packed_attn_mode == "magi_mp_causal":
            assert per_sample_params is not None, (
                "per_sample_params is required when packed_attn_mode='magi_mp_causal'"
            )
        self.register_forward_pre_hook(SequenceParallelAttention._pre_hook)
        self.register_forward_hook(SequenceParallelAttention._post_hook)

    def forward(
        self,
        img_q,
        img_k,
        img_v,
        txt_q,
        txt_k,
        txt_v,
        img_cu_seqlens,
        txt_cu_seqlens,
    ):
        # q/k/v arrive already head-parallel (Shard(2)) via the pre-forward hook,
        # so every rank now holds the FULL packed sequence but only its H/P
        # heads. Both backends therefore run the plain (non-SP) block-diagonal
        # kernel on this rank's head shard; the post-hook re-shards the output.
        if self.packed_attn_mode == "magi_mp_causal":
            return packed_magi_mp_causal_attention(
                img_q,
                img_k,
                img_v,
                txt_q,
                txt_k,
                txt_v,
                img_cu_seqlens=img_cu_seqlens,
                txt_cu_seqlens=txt_cu_seqlens,
                per_sample_params=self.per_sample_params,
                seg_ar_attention_backend=self.seg_ar_attention_backend,
            )
        return packed_parallel_attention(
            img_q,
            img_k,
            img_v,
            txt_q,
            txt_k,
            txt_v,
            img_cu_seqlens=img_cu_seqlens,
            txt_cu_seqlens=txt_cu_seqlens,
        )

    @staticmethod
    def _pre_hook(module, args):
        """Ulysses input collective: sequence-sharded -> head-sharded q/k/v.

        Runs before ``forward``. Rewrites the incoming per-rank locals so the
        kernel sees the full sequence with only this rank's heads. For
        ``all_to_all_4D``, ``scatter_dim=2`` gives away heads and ``gather_dim=1``
        collects the sequence, i.e. ``Shard(1) -> Shard(2)``. The returned tuple
        replaces ``args`` for the wrapped ``forward``.
        """
        (img_q, img_k, img_v, txt_q, txt_k, txt_v, img_cu_seqlens, txt_cu_seqlens) = (
            args
        )
        g = nccl_info.sp_group
        # image: Shard(1) -> Shard(2)  (scatter head, gather seq)
        img_q = all_to_all_4D(img_q, g, scatter_dim=2, gather_dim=1)
        img_k = all_to_all_4D(img_k, g, scatter_dim=2, gather_dim=1)
        img_v = all_to_all_4D(img_v, g, scatter_dim=2, gather_dim=1)
        if module.txt_seq_sharded:
            # text also Shard(1) -> Shard(2), same collective as the image
            txt_q = all_to_all_4D(txt_q, g, scatter_dim=2, gather_dim=1)
            txt_k = all_to_all_4D(txt_k, g, scatter_dim=2, gather_dim=1)
            txt_v = all_to_all_4D(txt_v, g, scatter_dim=2, gather_dim=1)
        else:
            # replicated text (all H heads) -> this rank's H/P head shard.
            # A narrow, so its backward stays partial-per-rank (no x P grad).
            txt_q = shrink_head(txt_q, dim=2)
            txt_k = shrink_head(txt_k, dim=2)
            txt_v = shrink_head(txt_v, dim=2)
        return (
            img_q,
            img_k,
            img_v,
            txt_q,
            txt_k,
            txt_v,
            img_cu_seqlens,
            txt_cu_seqlens,
        )

    @staticmethod
    def _post_hook(module, args, output):
        """Ulysses output collective: head-sharded -> sequence-sharded result.

        Runs after ``forward`` and mirrors ``_pre_hook``. ``args`` here is the
        pre-hook's *return* value (already head-parallel), so ``args[0]`` carries
        the invariant head_dim and ``args[6:8]`` the packed cu_seqlens.
        """
        # The packed backend (flash3 or magi_mp_causal) returns one concatenated
        # [1, L, (H/P)*D] tensor (image tokens first, then text). Split it back
        # into the two head-parallel branches and restore the 4D
        # [1, L_x, H/P, D] layout the output collectives expect. head_dim (D) is
        # invariant to the pre-hook; local_heads (H/P) is recovered from the
        # flattened last dim.
        head_dim = args[0].shape[-1]
        img_cu_seqlens, txt_cu_seqlens = args[6], args[7]
        L_img = int(img_cu_seqlens[-1])
        L_txt = int(txt_cu_seqlens[-1])
        local_heads = output.shape[-1] // head_dim
        output = output.view(1, L_img + L_txt, local_heads, head_dim)
        img_out, txt_out = output.split_with_sizes([L_img, L_txt], dim=1)
        g = nccl_info.sp_group
        # image: Shard(2) -> Shard(1)  (scatter seq, gather head)
        img_out = all_to_all_4D(img_out, g, scatter_dim=1, gather_dim=2)
        if module.txt_seq_sharded:
            # text Shard(2) -> Shard(1), symmetric to the image
            txt_out = all_to_all_4D(txt_out, g, scatter_dim=1, gather_dim=2)
        else:
            # text head-parallel Shard(2) -> Replicate (all-gather over heads).
            # Backward of this all-gather is a reduce_scatter-SUM, which is why
            # the pre-hook narrow (not a collective) keeps the text grad correct.
            txt_out = all_gather(txt_out, dim=2, group=g).contiguous()
        # 4D [1, L_x, H, D] -> 3D [1, L_x, H*D], matching parallel_attention.
        img_out = img_out.reshape(1, img_out.shape[1], -1)
        txt_out = txt_out.reshape(1, txt_out.shape[1], -1)
        return img_out, txt_out
