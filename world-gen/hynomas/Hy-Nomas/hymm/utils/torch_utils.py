import contextlib
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class TrainingPrecisionPolicy:
    """Dtypes used at the distinct boundaries of DiT training.

    ``master_dtype`` is the persistent/local parameter dtype updated by the
    optimizer and written to model checkpoints. ``compute_dtype`` is the FSDP2
    mixed-precision parameter dtype materialized for forward/backward.
    ``cache_dtype`` is the KV-cache/attention-kernel dtype.
    """

    master_dtype: torch.dtype
    compute_dtype: torch.dtype
    reduce_dtype: torch.dtype
    cache_dtype: torch.dtype


def resolve_dit_training_precision(master_weight_type) -> TrainingPrecisionPolicy:
    """Resolve persistent, compute, reduction, and cache dtypes for DiT.

    FP32 master weights intentionally still use BF16 forward parameters and KV
    cache. This preserves optimizer/checkpoint precision without feeding FP32
    tensors into FlashAttention. Non-``fp32`` values keep the historical BF16
    fallback used by existing recipes.
    """

    master_dtype = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    return TrainingPrecisionPolicy(
        master_dtype=master_dtype,
        compute_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cache_dtype=torch.bfloat16,
    )


def set_manual_seed(global_seed):
    # Seed the RNG for Python
    random.seed(global_seed)
    # Seed the RNG for Numpy
    np.random.seed(global_seed)
    # Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(global_seed)


def set_reproducibility(enable, global_seed=None):
    if enable:
        # Configure the seed for reproducibility
        set_manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = not enable
    # Use deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(enable)

    # LSTM and RNN networks are not deterministic


class set_worker_seed_builder:
    def __init__(self, global_rank):
        self.global_rank = global_rank

    def __call__(self, worker_id):
        set_manual_seed(torch.initial_seed() % (2**32 - 1))


def profiler_context(enable, exp_dir, worker_name):
    if enable:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=10,
                wait=5,
                warmup=1,
                active=3,
                repeat=2,
            ),
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                exp_dir, worker_name=worker_name
            ),
        )
    else:
        # return empty python context manager
        return contextlib.nullcontext()


def all_gather_sum(running_value, device):
    value = torch.tensor(running_value, device=device, dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.item()
