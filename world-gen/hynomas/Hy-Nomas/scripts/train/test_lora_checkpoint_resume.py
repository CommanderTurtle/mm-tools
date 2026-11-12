#!/usr/bin/env python3
"""Distributed smoke test for deterministic LoRA HSDP checkpoint resume.

Run with, for example:

    torchrun --standalone --nproc_per_node=8 \
      scripts/train/test_lora_checkpoint_resume.py

The default 8-rank layout uses two replicas of a four-rank FSDP shard group.
Only the first replica writes payloads; both replicas must restore from the
same four shard files with rank-local scalar and RNG state preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.optim.lr_scheduler import LambdaLR

from hymm.models.checkpointing.lora_checkpoint import (
    auto_find_latest_lora_checkpoint,
    load_lora_training_checkpoint,
    restore_rng_state,
    save_lora_training_checkpoint,
    shutdown_lora_checkpoint_writer,
    wait_for_pending_lora_checkpoint,
)
from hymm.models.lora import LORA_TARGETS, _adapter_parameters, inject_lora
from hymm.utils.helpers import ScalarStates


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_attn_q = nn.Linear(32, 64)
        self.img_attn_v = nn.Linear(64, 16)
        self.img_mlp = nn.ModuleDict({
            "fc1": nn.Linear(16, 32), "fc2": nn.Linear(32, 16),
        })
        self.img_mod = nn.ModuleDict({"linear": nn.Linear(16, 16)})

    def forward(self, value):
        value = self.img_attn_v(torch.relu(self.img_attn_q(value)))
        return (
            self.img_mlp["fc2"](torch.relu(self.img_mlp["fc1"](value)))
            + self.img_mod["linear"](value)
        )


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([ToyBlock()])

    def forward(self, value):
        return self.double_blocks[0](value)


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        local = value._local_tensor if hasattr(value, "_local_tensor") else value
        return local.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _assert_equal(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        local = actual._local_tensor if hasattr(actual, "_local_tensor") else actual
        if not torch.equal(local.detach().cpu(), expected):
            raise AssertionError(f"tensor mismatch at {path}")
        return
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"key mismatch at {path}")
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"length mismatch at {path}")
        for index, item in enumerate(expected):
            _assert_equal(actual[index], item, f"{path}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"value mismatch at {path}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--shard-size", type=int, default=4)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size % args.shard_size:
        raise ValueError("WORLD_SIZE must be divisible by --shard-size")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    replicas = world_size // args.shard_size
    mesh = init_device_mesh(
        "cuda",
        (replicas, args.shard_size),
        mesh_dim_names=("dp_replicate", "dp_shard_sp"),
    )
    shard_group = mesh["dp_shard_sp"].get_group()
    replica_rank = mesh["dp_replicate"].get_local_rank()

    if rank == 0:
        output_value = args.output or tempfile.mkdtemp(
            prefix="hy_lora_checkpoint_distributed_test_", dir="/tmp"
        )
        output = Path(output_value)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty test output: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
    else:
        output_value = None
    holders = [output_value]
    dist.broadcast_object_list(holders, src=0)
    output = Path(holders[0])
    dist.barrier()
    os.environ["HY_CHECKPOINT_LOCAL_DIR"] = str(output / "local_staging")

    torch.manual_seed(100)
    torch.cuda.manual_seed(200 + rank)
    np.random.seed(300 + rank)
    random.seed(400 + rank)
    model = ToyModel().cuda()
    inject_lora(
        model,
        rank=8,
        alpha=8,
        dropout=0,
    )
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    replicate_group = mesh["dp_replicate"].get_group()
    replicate_source = dist.get_process_group_ranks(replicate_group)[0]
    for parameter in model.parameters():
        local = (
            parameter._local_tensor
            if hasattr(parameter, "_local_tensor")
            else parameter
        )
        dist.broadcast(local, src=replicate_source, group=replicate_group)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
    )
    scheduler = LambdaLR(optimizer, lambda step: 0.95**step)

    for step in range(3):
        torch.manual_seed(1000 + step)
        value = torch.randn(8, 32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        model(value).square().mean().backward()
        optimizer.step()
        scheduler.step()

    from torch.distributed.checkpoint.state_dict import get_state_dict

    saved_model, saved_optimizer = get_state_dict(model, optimizer)
    saved_adapter = _cpu_clone(
        {
            name: tensor
            for name, tensor in saved_model.items()
            if name.endswith(("lora_A.weight", "lora_B.weight"))
        }
    )
    saved_optimizer = _cpu_clone(saved_optimizer)
    saved_scheduler = _cpu_clone(scheduler.state_dict())

    scalar = ScalarStates(train_steps=3, update_steps=3)
    scalar.epoch["video"] = 10 + rank
    scalar.consumed_groups_per_dp["video"] = 100 + rank
    scalar.consumed_samples_per_dp["video"] = 1000 + rank
    save_lora_training_checkpoint(
        model,
        optimizer,
        scheduler,
        output,
        scalar,
        {"rank": 8, "alpha": 8, "dropout": 0},
        dp_replicate_rank=replica_rank,
        shard_group=shard_group,
    )
    expected_rng = (
        torch.rand(4),
        torch.rand(4, device="cuda").cpu(),
        np.random.rand(4),
        [random.random() for _ in range(4)],
    )
    wait_for_pending_lora_checkpoint()

    checkpoint = output / "checkpoints" / "global_step-3"
    persisted = torch.load(
        checkpoint / f"lora_shard_{dist.get_rank(shard_group):04d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    local_mismatch = []
    for name, expected in saved_adapter.items():
        canonical = name.replace("._checkpoint_wrapped_module.", ".").replace(
            "_checkpoint_wrapped_module.", ""
        )
        if not torch.equal(expected, persisted["adapter"][canonical]):
            local_mismatch.append(canonical)
    mismatch_by_rank = [None] * world_size
    dist.all_gather_object(mismatch_by_rank, local_mismatch)
    if rank == 0:
        print(
            f"pre-resume persisted adapter mismatches: {mismatch_by_rank}", flush=True
        )
    selected = auto_find_latest_lora_checkpoint(output)
    if selected != str(checkpoint):
        raise AssertionError(f"auto resume selected {selected}, expected {checkpoint}")
    if rank == 0:
        metadata = json.loads((checkpoint / "adapter_config.json").read_text())
        assert metadata["targets"] == list(LORA_TARGETS)
        shards = sorted(checkpoint.glob("lora_shard_*.pt"))
        if len(shards) != args.shard_size:
            raise AssertionError(
                f"expected {args.shard_size} shards, found {len(shards)}"
            )

    # Recreate the full training objects to exercise the real restart path,
    # including restoration into a freshly constructed AdamW optimizer whose
    # state dict has not seen an optimizer.step() in this process.
    del scheduler, optimizer, model
    torch.cuda.empty_cache()
    torch.manual_seed(9000 + rank)
    model = ToyModel().cuda()
    inject_lora(
        model,
        rank=8,
        alpha=8,
        dropout=0,
    )
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
    )
    scheduler = LambdaLR(optimizer, lambda step: 0.95**step)
    torch.manual_seed(9999 + rank)
    torch.cuda.manual_seed(19999 + rank)
    np.random.seed(29999 + rank)
    random.seed(39999 + rank)

    loaded_scalar, loaded_rng, loaded_scheduler = load_lora_training_checkpoint(
        model,
        optimizer,
        checkpoint,
        shard_group=shard_group,
    )
    scheduler.load_state_dict(loaded_scheduler)
    restore_rng_state(loaded_rng)

    current_model, current_optimizer = get_state_dict(model, optimizer)
    current_adapter = {
        name: tensor
        for name, tensor in current_model.items()
        if name.endswith(("lora_A.weight", "lora_B.weight"))
    }
    _assert_equal(current_adapter, saved_adapter, "adapter")
    _assert_equal(current_optimizer, saved_optimizer, "optimizer")
    _assert_equal(scheduler.state_dict(), saved_scheduler, "scheduler")
    if loaded_scalar["epoch"]["video"] != 10 + rank:
        raise AssertionError("rank-local epoch was not restored")
    if loaded_scalar["consumed_groups_per_dp"]["video"] != 100 + rank:
        raise AssertionError("rank-local packing cursor was not restored")

    actual_rng = (
        torch.rand(4),
        torch.rand(4, device="cuda").cpu(),
        np.random.rand(4),
        [random.random() for _ in range(4)],
    )
    _assert_equal(actual_rng[0], expected_rng[0], "torch_cpu_rng")
    _assert_equal(actual_rng[1], expected_rng[1], "torch_cuda_rng")
    if not np.array_equal(actual_rng[2], expected_rng[2]):
        raise AssertionError("NumPy RNG mismatch")
    _assert_equal(actual_rng[3], expected_rng[3], "python_rng")

    dist.barrier()
    if rank == 0:
        print(
            f"PASS: world_size={world_size} replicas={replicas} "
            f"shard_size={args.shard_size} checkpoint={checkpoint}",
            flush=True,
        )
    shutdown_lora_checkpoint_writer()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
