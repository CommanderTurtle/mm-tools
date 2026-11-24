#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Resumable training state (torch.distributed.checkpoint) and weight export.

A run directory holds ``step_<N>/`` generations written with DCP (sharded, so
FSDP models and optimizers never need a full gather) plus a ``latest`` pointer
that is replaced atomically once every rank has finished writing. Rank-local
state (EMA shards of FSDP parameters, RNG, data position) goes to
``step_<N>/rank_<r>.pt``; resuming therefore requires the same world size.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from .distributed import barrier, raise_if_any_failed


def _state(models, optimizers):
    state = {f"model.{k}": get_model_state_dict(m) for k, m in models.items()}
    for k, (model, optimizer) in optimizers.items():
        state[f"optim.{k}"] = get_optimizer_state_dict(model, optimizer)
    return state


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def save(run_dir, step, models, optimizers=None, rank_state=None, keep=2, is_main=True):
    """Write generation ``step``; ``optimizers`` maps name -> (model, optimizer).

    ``rank_state`` is this rank's picklable local state (EMA shards, RNG,
    counters). Keeps the newest ``keep`` generations.
    """
    run_dir = Path(run_dir)
    target = run_dir / f"step_{step:08d}"
    dcp.save(_state(models, optimizers or {}), checkpoint_id=str(target))
    error = None
    try:
        tmp = target / f".rank_{_rank():05d}.pt.tmp"
        torch.save(rank_state or {}, tmp)
        os.replace(tmp, target / f"rank_{_rank():05d}.pt")
    except Exception as exc:  # noqa: BLE001 - reported collectively
        error = exc
    raise_if_any_failed(error, f"saving rank state for step {step}")
    if is_main:
        tmp = run_dir / "latest.json.tmp"
        tmp.write_text(json.dumps({"step": step, "path": target.name}))
        os.replace(tmp, run_dir / "latest.json")
        generations = sorted(run_dir.glob("step_*"))
        for old in generations[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    barrier()


def latest_step(run_dir):
    pointer = Path(run_dir) / "latest.json"
    return json.loads(pointer.read_text())["step"] if pointer.exists() else None


def load(run_dir, models, optimizers=None):
    """Restore the newest generation in place; returns ``(step, rank_state)`` or ``(None, None)``."""
    step = latest_step(run_dir)
    if step is None:
        return None, None
    target = Path(run_dir) / f"step_{step:08d}"
    state = _state(models, optimizers or {})
    # Keep sharded tensors unflattened: a rank that holds no shard of a small parameter (dim 0 smaller
    # than the world size) would otherwise drop its entries and fail to load the optimizer state.
    dcp.load(state, checkpoint_id=str(target), planner=DefaultLoadPlanner(flatten_sharded_tensors=False))
    for k, m in models.items():
        set_model_state_dict(m, state[f"model.{k}"])
    for k, (model, optimizer) in (optimizers or {}).items():
        set_optimizer_state_dict(model, optimizer, state[f"optim.{k}"])
    rank_state = torch.load(target / f"rank_{_rank():05d}.pt", map_location="cpu", weights_only=True)
    return step, rank_state


def full_state_dict(model):
    """Gather an unsharded CPU state dict on rank 0 (empty dict elsewhere)."""
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_model_state_dict(model, options=options)


def export_weights(path, model, dtype=None, is_main=True):
    """Save the (EMA-swapped, if desired) model weights as one safetensors file."""
    from safetensors.torch import save_file
    state = full_state_dict(model)
    error = None
    if is_main:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {k: (v.to(dtype) if dtype is not None and v.is_floating_point() else v).contiguous()
                     for k, v in state.items()}
            tmp = path.with_suffix(".tmp")
            save_file(state, str(tmp))
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
    raise_if_any_failed(error, f"exporting {path}")
