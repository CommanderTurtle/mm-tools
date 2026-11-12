import os

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
from einops import rearrange

_FLASH_ATTN_DETERMINISTIC = (
    os.environ.get("FLASH_ATTENTION_DETERMINISTIC_MODE", "0") == "1"
)


def flash_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic or _FLASH_ATTN_DETERMINISTIC,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    # Compatibility wrapper: use FlashAttention2 varlen kernels. FA3 deterministic
    # mode is avoided because it is buggy on Hopper in the current environment.
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(  # 复用max_seqlen_q
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    # Flash Attention only supports fp16/bf16/fp8; cast from fp32 when needed.
    _orig_dtype = query_unpad.dtype
    if _orig_dtype not in (torch.float16, torch.bfloat16):
        query_unpad = query_unpad.to(torch.bfloat16)
        key_unpad = key_unpad.to(torch.bfloat16)
        value_unpad = value_unpad.to(torch.bfloat16)

    output_unpad = flash_attn_varlen_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_q,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic or _FLASH_ATTN_DETERMINISTIC,
    )

    # Cast output back to the original dtype if we cast the inputs.
    if output_unpad.dtype != _orig_dtype:
        output_unpad = output_unpad.to(_orig_dtype)

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_packed_v3(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    cu_seqlens: "torch.Tensor",
    max_seqlen: int,
    softmax_scale=None,
    causal: bool = False,
    deterministic: bool = False,
) -> "torch.Tensor":
    """Flash attention for token-packed sequences (no padding).

    All samples are concatenated into a single sequence.  ``cu_seqlens``
    marks the boundaries so that each sample only attends within itself
    (block-diagonal attention).

    Args:
        query:  ``[total_tokens, H, D]``
        key:    ``[total_tokens, H, D]``
        value:  ``[total_tokens, H, D]``
        cu_seqlens: ``[num_samples + 1]`` int32 cumulative sequence lengths
            starting from 0.
        max_seqlen: Maximum individual sequence length.
        softmax_scale: Optional softmax scaling factor.
        causal: Whether to apply causal masking within each sample.
        deterministic: Deterministic backward pass.

    Returns:
        output: ``[total_tokens, H, D]``
    """
    # Compatibility wrapper: use FlashAttention2 varlen kernels. FA3 deterministic
    # mode is avoided because it is buggy on Hopper in the current environment.
    # Flash Attention only supports fp16/bf16/fp8; cast from fp32 when needed.
    _orig_dtype = query.dtype
    if _orig_dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

    output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic or _FLASH_ATTN_DETERMINISTIC,
    )

    # Cast output back to the original dtype if we cast the inputs.
    if output.dtype != _orig_dtype:
        output = output.to(_orig_dtype)

    return output


def test_flash_attn_alignment(
    batch_size: int = 4,
    seq_len: int = 128,
    n_heads: int = 8,
    head_dim: int = 64,
    pct_valid: float = 0.8,  # 有效token比例
    causal: bool = True,
    tol: float = 1e-5,
) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qkv = torch.randn(
        batch_size, seq_len, 3, n_heads, head_dim, device=device, dtype=torch.float16
    )

    key_padding_mask = torch.ones(batch_size, seq_len, device=device)
    for i in range(batch_size):
        valid_len = int(seq_len * pct_valid)
        key_padding_mask[i, valid_len:] = 0  # 设置padding部分为0

    output_v2 = flash_attn_no_pad(
        qkv, key_padding_mask, causal=causal, deterministic=True
    )
    output_v3 = flash_attn_no_pad_v3(
        qkv, key_padding_mask, causal=causal, deterministic=True
    )

    valid_indices = key_padding_mask == 1
    valid_output_v2 = output_v2[valid_indices]
    valid_output_v3 = output_v3[valid_indices]

    abs_diff = (valid_output_v2 - valid_output_v3).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    is_aligned = torch.allclose(valid_output_v2, valid_output_v3, rtol=tol, atol=tol)

    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "is_aligned": bool(is_aligned),
        "output_shape": tuple(output_v2.shape),
        "valid_token_count": int(valid_indices.sum().item()),
        "causal_mode": causal,
        "tolerance": tol,
    }


if __name__ == "__main__":
    base_test = test_flash_attn_alignment(pct_valid=0.7)
    print("基础测试结果:", base_test)

    full_valid_test = test_flash_attn_alignment(pct_valid=1.0)
    print("\n全有效token测试:", full_valid_test)

    high_pad_test = test_flash_attn_alignment(pct_valid=0.3)
    print("\n高padding比例测试:", high_pad_test)

    non_causal_test = test_flash_attn_alignment(causal=False)
    print("\n非因果注意力测试:", non_causal_test)
