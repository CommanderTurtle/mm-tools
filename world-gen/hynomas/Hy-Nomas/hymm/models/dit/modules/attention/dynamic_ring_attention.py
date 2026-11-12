import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_forward_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward as _flash_attn_forward,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward as _flash_attn_backward,
)
# from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd

cache_cu_seqlens_q = None
cache_cu_seqlens_k = None
cache_max_seqlen_q = None
cache_max_seqlen_k = None


def cache_metadata(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    global cache_cu_seqlens_q
    global cache_cu_seqlens_k
    global cache_max_seqlen_q
    global cache_max_seqlen_k
    cache_cu_seqlens_q = cu_seqlens_q
    cache_cu_seqlens_k = cu_seqlens_k
    cache_max_seqlen_q = max_seqlen_q
    cache_max_seqlen_k = max_seqlen_k


def get_distributed_world_size(group=None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def get_distributed_rank(group=None) -> int:
    """Return my rank for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    # return torch.distributed.get_rank(group=group)
    return torch.distributed.get_rank()


def flash_attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []
    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


@torch.compile
def flash_attn_fwd_out_correction(
    out, out_per_step, seq_dim, softmax_lse, softmax_lse_per_step
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(
        0, seq_dim
    )
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


@torch.compile
def flash_attn_fwd_softmax_lse_correction(softmax_lse, softmax_lse_per_step):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


class DynamicRingAttention(torch.autograd.Function):
    """
    Attention implementation with dynamic context parallelism.
    Split attention compute into multiple steps, and overlap current-step
    compute with next-step communication.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        cp_group,
        cp_global_ranks,
        cp_stream,
        softmax_scale,
        deterministic,
        use_fused_attention,
    ):
        torch.cuda.synchronize()
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        index = cp_global_ranks.index(rank)
        assert index >= 0, f"rank:{rank} is not in cp_global_ranks:{cp_global_ranks}!"
        assert k.shape[0] == 1  # assert batch_size == 1
        assert k.shape[1] == max_seqlen_k[index]
        send_dst = cp_global_ranks[(index + 1) % cp_size]
        recv_src = cp_global_ranks[(index - 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            cp_size == 2
        )

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        recv_rank_index = (index + cp_size - 1 - i) % cp_size
                        recv_shape = [
                            2,
                            k.shape[0],
                            max_seqlen_k[recv_rank_index],
                            *k.shape[-2:],
                        ]
                        p2p_comm_buffers[i + 1] = torch.empty(
                            recv_shape, dtype=k.dtype, device=k.device
                        )
                        send_recv_reqs[i % 2] = flash_attn_p2p_communicate(
                            rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )

                    kv_inputs[i % 2] = p2p_comm_buffers[i]

                    if use_fused_attention:
                        pass
                    else:
                        # [b, sq, np, hn] -> [b*sq, np, hn]
                        q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                        # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                        kv_inputs[i % 2] = kv_inputs[i % 2].view(2, -1, *k.shape[-2:])
                        current_kv_index = (index + cp_size - i) % cp_size
                        (
                            out_per_step[i],
                            softmax_lse_per_step[i],
                            _,
                            rng_states[i],
                        ) = _flash_attn_forward(
                            q_inputs[i % 2],
                            kv_inputs[i % 2][0],
                            kv_inputs[i % 2][1],
                            cu_seqlens_q[index],
                            cu_seqlens_k[current_kv_index],
                            max_seqlen_q[index],
                            max_seqlen_k[current_kv_index],
                            dropout_p,
                            softmax_scale,
                            window_size_left=-1,
                            window_size_right=-1,
                            causal=False,
                            return_softmax=False,
                            alibi_slopes=None,
                            block_table=None,
                        )
                        softmax_lse_per_step[i] = torch.nan_to_num(
                            softmax_lse_per_step[i], posinf=-float("inf")
                        )
            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(
                        fwd_results_correction_done
                    )

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if i == 1:
                        out = torch.empty_like(q).zero_()
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(
                            torch.double
                        )
                    else:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )

                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(
                        fwd_results_correction_done
                    )
        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        softmax_lse = softmax_lse.to(torch.float)
        seq_dim = 1
        for i in range(cp_size):
            # bshd
            out_per_step[i] = out_per_step[i].view(out.shape[0], -1, *out.shape[-2:])
            flash_attn_fwd_out_correction(
                out.view(*out_per_step[i].shape),
                out_per_step[i],
                seq_dim,
                softmax_lse,
                softmax_lse_per_step[i],
            )

        kv = p2p_comm_buffers[-1]
        out = out.view(-1, *out.shape[-2:])

        ctx.save_for_backward(
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            *rng_states,
        )
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        return out

    @staticmethod
    def backward(ctx, dout):
        (q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k) = ctx.saved_tensors[:6]
        cp_size = get_distributed_world_size(ctx.cp_group)

        rank = get_distributed_rank(ctx.cp_group)
        cp_global_ranks = ctx.cp_global_ranks
        rank_index = cp_global_ranks.index(rank)
        assert rank_index >= 0, (
            f"rank:{rank} is not in cp_global_ranks:{cp_global_ranks}!"
        )
        send_dst = cp_global_ranks[(rank_index + cp_size - 1) % cp_size]
        recv_src = cp_global_ranks[(rank_index + 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            cp_size == 2
        )

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        dq = torch.empty_like(q)

        max_seq_len = max(ctx.max_seqlen_k)
        batch_size = q.shape[0]
        p2p_comm_buffers = [
            torch.empty(
                (2, 2 * batch_size * max_seq_len, *kv.shape[-2:]),
                dtype=kv.dtype,
                device=kv.device,
            ),
            torch.empty(
                (2, 2 * batch_size * max_seq_len, *kv.shape[-2:]),
                dtype=kv.dtype,
                device=kv.device,
            ),
        ]
        kv = kv.view(-1, *kv.shape[-2:])
        p2p_comm_buffers[0][0][: kv.shape[0], ...].copy_(kv)
        send_recv_reqs = []

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]
            send_recv_reqs = flash_attn_p2p_communicate(
                rank,
                send_tensor,
                send_dst,
                recv_tensor,
                recv_src,
                ctx.cp_group,
                batch_p2p_comm,
            )

            kv = p2p_comm_buffers[i % 2][0]
            current_kv_index = (rank_index + 1 + i) % cp_size
            current_kv_seq_len = ctx.max_seqlen_k[current_kv_index]
            kv = kv[0 : 2 * current_kv_seq_len, ...]
            # In reversed order of fwd
            if ctx.use_fused_attention:
                pass
            else:
                # [b, sq, np, hn] -> [b*sq, np, hn]
                q_ = q.view(-1, *q.shape[-2:])
                dq_ = torch.empty_like(q_)
                # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                kv_ = kv.view(2, -1, *kv.shape[-2:])
                dkv_ = torch.empty_like(kv_)
                # [b, sq, np, hn] -> [b*sq, np, hn]
                out_ = out.view(-1, *out.shape[-2:])
                dout_ = dout.view(-1, *dout.shape[-2:])

                _flash_attn_backward(
                    dout_,
                    q_,
                    kv_[0],
                    kv_[1],
                    out_,
                    softmax_lse,
                    dq_,
                    dkv_[0],
                    dkv_[1],
                    cu_seqlens_q[rank_index],
                    cu_seqlens_k[current_kv_index],
                    ctx.max_seqlen_q[rank_index],
                    ctx.max_seqlen_k[current_kv_index],
                    ctx.dropout_p,
                    ctx.softmax_scale,
                    causal=False,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=ctx.deterministic,
                )

            # [b*sq, np, hn] -> [b, sq, np, hn]
            dq_ = dq_.view(*dq.shape)

            if i == 0:
                dq.copy_(dq_)
            else:
                dq.add_(dq_)

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            dkv = p2p_comm_buffers[(i + 1) % 2][1]
            dkv = dkv[: current_kv_seq_len * 2, ...]
            dkv_ = dkv_.view(*dkv.shape)

            if i == 0:
                dkv.copy_(dkv_)
            else:
                dkv.add_(dkv_)
        dkv = dkv.view(2, -1, *dkv.shape[-2:])
        dk, dv = dkv[0].unsqueeze(0), dkv[1].unsqueeze(0)
        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # attn_dbias,
            None,
            None,
        )


def attn_forward_func(
    is_training,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    cp_group,
    cp_global_ranks,
    cp_stream,
    softmax_scale=None,
    deterministic=False,
    use_fused_attention=False,
) -> torch.Tensor:
    """Attention implementation with dynamic context parallelism"""
    out = DynamicRingAttention.apply(
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        cp_group,
        cp_global_ranks,
        cp_stream,
        softmax_scale,
        deterministic,
        use_fused_attention,
    )
    return out
