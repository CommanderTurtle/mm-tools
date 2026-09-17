"""Native PyTorch SDPA helpers for flattened variable-length sparse batches.

The upstream project historically required FlashAttention or xFormers for
variable-length sparse attention.  Grouping equal-sized sequences lets modern
PyTorch dispatch its own fused SDPA kernels without padding, attention masks,
or a second CUDA attention dependency.  Work is chunked by token count so a
large set of equally-sized voxel windows cannot create an oversized temporary.
"""

from collections import defaultdict
from typing import Iterable

import torch
from torch.nn.functional import scaled_dot_product_attention


def _lengths(values: Iterable[int]) -> list[int]:
    return [int(value) for value in values]


def grouped_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: Iterable[int],
    kv_lengths: Iterable[int],
    *,
    token_budget: int = 65536,
) -> torch.Tensor:
    """Run SDPA over flattened sequences while preserving original ordering."""
    q_lengths = _lengths(q_lengths)
    kv_lengths = _lengths(kv_lengths)
    if len(q_lengths) != len(kv_lengths):
        raise ValueError("q_lengths and kv_lengths must contain the same number of sequences")
    if not q_lengths:
        return q.new_empty((0, q.shape[-2], v.shape[-1]))

    q_offsets = [0]
    kv_offsets = [0]
    for length in q_lengths:
        q_offsets.append(q_offsets[-1] + length)
    for length in kv_lengths:
        kv_offsets.append(kv_offsets[-1] + length)
    if q_offsets[-1] != q.shape[0] or kv_offsets[-1] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("flattened attention features do not match their declared sequence lengths")

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, pair in enumerate(zip(q_lengths, kv_lengths)):
        groups[pair].append(index)

    outputs: list[torch.Tensor | None] = [None] * len(q_lengths)
    for (q_length, kv_length), indices in groups.items():
        if q_length == 0:
            for index in indices:
                outputs[index] = q.new_empty((0, q.shape[-2], v.shape[-1]))
            continue
        if kv_length == 0:
            raise ValueError("SDPA cannot attend to an empty key/value sequence")
        chunk_size = max(1, token_budget // max(q_length, kv_length))
        for begin in range(0, len(indices), chunk_size):
            chunk = indices[begin : begin + chunk_size]
            q_batch = torch.stack(
                [q[q_offsets[i] : q_offsets[i + 1]] for i in chunk], dim=0
            ).permute(0, 2, 1, 3)
            k_batch = torch.stack(
                [k[kv_offsets[i] : kv_offsets[i + 1]] for i in chunk], dim=0
            ).permute(0, 2, 1, 3)
            v_batch = torch.stack(
                [v[kv_offsets[i] : kv_offsets[i + 1]] for i in chunk], dim=0
            ).permute(0, 2, 1, 3)
            batch_out = scaled_dot_product_attention(q_batch, k_batch, v_batch)
            batch_out = batch_out.permute(0, 2, 1, 3)
            for position, index in enumerate(chunk):
                outputs[index] = batch_out[position]

    if any(output is None for output in outputs):
        raise RuntimeError("native SDPA failed to produce an output for every sparse sequence")
    return torch.cat(outputs, dim=0)  # type: ignore[arg-type]
