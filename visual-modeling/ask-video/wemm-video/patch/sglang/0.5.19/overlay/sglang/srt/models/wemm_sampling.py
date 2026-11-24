# SPDX-License-Identifier: Apache-2.0
"""HF/vLLM filtering semantics and request seeds for the WeMM adapter only.

SGLang's joint top-k/top-p filter uses the original distribution for nucleus
selection. The reference evaluator instead applies nucleus selection after
top-k renormalization. Keep the native model kernels and prefix caches; request
seeding does not require enabling global deterministic-kernel mode.
"""

import hashlib

import torch

from sglang.srt.layers.sampler import Sampler, sampling_from_probs_torch


def reference_probabilities(probs, top_ks, top_ps, min_ps=None):
    """Apply top-k, renormalize, then top-p, matching the reference sampler."""
    values, indices = probs.sort(dim=-1)
    vocab_size = values.shape[-1]
    counts = top_ks.long().clamp(min=1, max=vocab_size)
    threshold = values.gather(1, (vocab_size - counts).view(-1, 1))
    values = values.masked_fill(values < threshold, 0)
    values = values / values.sum(dim=-1, keepdim=True)
    remove = values.cumsum(dim=-1) <= (1 - top_ps.view(-1, 1))
    remove[:, -1] = False
    values = values.masked_fill(remove, 0)
    if min_ps is not None:
        values = values.masked_fill(values < values[:, -1:] * min_ps.view(-1, 1), 0)
    values = values / values.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(1, indices, values)


def request_seed_tensor(reqs, device, pin_memory=False):
    """Use explicit seeds, or shared request IDs for unseeded requests.

    Always supply a tensor so seeded/unseeded batches can merge. Request IDs
    differ between calls but agree across TP ranks, avoiding rank-local RNGs.
    """
    seeds = []
    for req in reqs:
        seed = req.sampling_params.sampling_seed
        if seed is None:
            digest = hashlib.blake2b(req.rid.encode(), digest_size=8).digest()
            seed = int.from_bytes(digest, 'little') & ((1 << 63) - 1)
        seeds.append(seed)
    return torch.tensor(seeds, dtype=torch.int64, pin_memory=pin_memory).to(
        device, non_blocking=True
    )


class WeMMReferenceSampler(Sampler):
    def _sample_from_probs(self, probs, sampling_info, positions, simple_sampling_case):
        filtered = reference_probabilities(
            probs, sampling_info.top_ks, sampling_info.top_ps,
            sampling_info.min_ps if sampling_info.need_min_p_sampling else None,
        )
        return sampling_from_probs_torch(
            filtered, sampling_seed=sampling_info.sampling_seed, positions=positions
        )

    def _compute_sampling_mask_from_probs(self, probs, sampling_info):
        filtered = reference_probabilities(
            probs, sampling_info.top_ks, sampling_info.top_ps,
            sampling_info.min_ps if sampling_info.need_min_p_sampling else None,
        )
        values, indices = probs.sort(dim=-1, descending=True)
        keep = filtered.gather(1, indices) > 0
        return indices, values, keep, probs
