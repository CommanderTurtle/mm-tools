#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""torchrun process-group setup, FSDP wrapping and collective error handling."""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass(frozen=True)
class Context:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed(timeout_minutes: int = 30) -> Context:
    """Initialise NCCL from torchrun's environment (or a single process)."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        # Always create a process group (FSDP needs one, also for a single process).
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)
        if "MASTER_ADDR" in os.environ:
            dist.init_process_group("nccl", device_id=device, timeout=timeout)
        else:
            dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1,
                                    device_id=device, timeout=timeout)
    return Context(rank, local_rank, world_size, device)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def raise_if_any_failed(error: BaseException | None, what: str) -> None:
    """Make a rank-local failure collective so that no peer waits at a later barrier."""
    message = None if error is None else repr(error)
    if dist.is_initialized():
        messages = [None] * dist.get_world_size()
        dist.all_gather_object(messages, message)
    else:
        messages = [message]
    failures = [(rank, m) for rank, m in enumerate(messages) if m]
    if failures:
        raise RuntimeError(f"{what} failed on ranks {failures}") from error


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def wrap_fsdp(model, block_types, ctx: Context, *, sharding="full", param_dtype=torch.bfloat16):
    """FSDP with FP32 master shards and BF16 compute; one unit per transformer block.

    ``sharding="hybrid"`` shards within a node and replicates across nodes.
    """
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise RuntimeError("trainable parameters must be FP32 masters before FSDP wrapping")
    strategy = {"full": ShardingStrategy.FULL_SHARD, "hybrid": ShardingStrategy.HYBRID_SHARD,
                "none": ShardingStrategy.NO_SHARD}[sharding]
    if ctx.world_size == 1:
        strategy = ShardingStrategy.NO_SHARD
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=set(block_types)),
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32, cast_forward_inputs=True),
        device_id=ctx.device,
        use_orig_params=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )
