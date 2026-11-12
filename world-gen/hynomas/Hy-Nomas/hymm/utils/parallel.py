"""Small distributed synchronization helpers used by Hy-Nomas.

These functions are kept local so the public training and inference paths do
not depend on a private parallelism package. Only the two behaviours used by
this repository are implemented: broadcasting nested training objects across
model-parallel groups and synchronizing process RNG state.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _parallel_groups(parallel_dims):
    """Return enabled TP, PP, and SP groups in the historical sync order."""
    if parallel_dims is None:
        from hymm.distributed.parallel_states import nccl_info

        parallel_dims = nccl_info.parallel_dims
    if parallel_dims is None:
        raise ValueError("parallel_dims must be initialized before synchronization")

    mesh = getattr(parallel_dims, "device_mesh", None)
    groups = []
    for name in ("tp", "pp", "sp"):
        if not bool(getattr(parallel_dims, f"{name}_enabled", False)):
            continue
        group = getattr(parallel_dims, f"{name}_group", None)
        if group is None and mesh is not None:
            mesh_names = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
            if name in mesh_names:
                group = mesh.get_group(mesh_dim=name)
        if group is None:
            # Sequence/tensor-parallel inference stores these groups in the
            # repository's communication state rather than ParallelDims.
            from hymm.distributed.parallel_states import nccl_info

            group = getattr(nccl_info, f"{name}_group", None)
        if group is None:
            raise RuntimeError(f"{name.upper()} is enabled but its process group is missing")
        groups.append((name, group))
    return groups


def _group_source(group) -> int:
    return dist.get_process_group_ranks(group)[0]


def _is_source(group, source: int) -> bool:
    return dist.get_rank() == source


def _gather_object(value, group):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _broadcast_object(value, group, source: int):
    holder = [value if _is_source(group, source) else None]
    dist.broadcast_object_list(holder, src=source, group=group)
    return holder[0]


def _broadcast_tensor(value, group, source: int):
    source_rank = _is_source(group, source)
    backend = str(dist.get_backend(group)).lower()
    communication_device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    if source_rank:
        tensor = value.detach().to(communication_device).contiguous()
        metadata = (tuple(tensor.shape), tensor.dtype)
    else:
        tensor = None
        metadata = None
    shape, dtype = _broadcast_object(metadata, group, source)
    if not source_rank:
        tensor = torch.empty(shape, dtype=dtype, device=communication_device)
    dist.broadcast(tensor, src=source, group=group)
    return tensor


def _auto_broadcast(value, group, source: int):
    source_rank = _is_source(group, source)
    value_type = _broadcast_object(type(value) if source_rank else None, group, source)

    if value_type is torch.Tensor:
        return _broadcast_tensor(value, group, source)
    if value_type in (list, tuple):
        length = _broadcast_object(len(value) if source_rank else None, group, source)
        items = value if source_rank else [None] * length
        result = [_auto_broadcast(item, group, source) for item in items]
        # Preserve the historical helper's behaviour: tuples become lists.
        return result
    if value_type is dict:
        keys = _broadcast_object(list(value) if source_rank else None, group, source)
        return {
            key: _auto_broadcast(value[key] if source_rank else None, group, source)
            for key in keys
        }
    if value_type is set:
        # Sets have no stable recursive order, so synchronize them as one
        # pickled object rather than independently broadcasting each member.
        return _broadcast_object(value, group, source)
    return _broadcast_object(value, group, source)


def sync_object_for_parallel_training(
    value,
    inplace: bool = False,
    force_object: bool = False,
    debug_with_check: bool = False,
    trace: str = "arg0",
    parallel_dims=None,
):
    """Broadcast a nested value from each model-parallel group's first rank.

    TP, PP, and SP groups are processed in that order, matching the previous
    private-package helper. ``force_object`` uses PyTorch object collectives;
    otherwise tensors are broadcast directly and containers recursively.
    """
    if not _distributed():
        return value
    if inplace:
        raise NotImplementedError("In-place parallel synchronization is not supported")

    result = value
    for group_name, group in _parallel_groups(parallel_dims):
        source = _group_source(group)
        if debug_with_check:
            types = _gather_object(type(result), group)
            if not all(item == types[0] for item in types):
                raise AssertionError(
                    f"{trace} has different types in {group_name} group: {types}"
                )
            if isinstance(result, torch.Tensor):
                shapes = _gather_object(tuple(result.shape), group)
                dtypes = _gather_object(result.dtype, group)
                if not all(item == shapes[0] for item in shapes):
                    raise AssertionError(
                        f"{trace} has different shapes in {group_name} group: {shapes}"
                    )
                if not all(item == dtypes[0] for item in dtypes):
                    raise AssertionError(
                        f"{trace} has different dtypes in {group_name} group: {dtypes}"
                    )
                gathered = [torch.empty_like(result) for _ in range(dist.get_world_size(group))]
                dist.all_gather(gathered, result, group=group)
                if not all(torch.allclose(item, gathered[0]) for item in gathered[1:]):
                    logger.error(
                        f"{trace} differs across the {group_name} group before synchronization"
                    )
            elif force_object:
                gathered = _gather_object(result, group)
                if not all(item == gathered[0] for item in gathered[1:]):
                    raise AssertionError(
                        f"{trace} differs across the {group_name} group"
                    )

        old_device = result.device if isinstance(result, torch.Tensor) else None
        if force_object:
            result = _broadcast_object(result, group, source)
        else:
            result = _auto_broadcast(result, group, source)
        if old_device is not None:
            result = result.to(old_device)
    return result


def _collect_rng_states() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch.cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_states(states: dict[str, Any]) -> None:
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states.get("torch.cuda") is not None:
        torch.cuda.set_rng_state(states["torch.cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])


def sync_random_states(parallel_dims=None):
    """Synchronize Python, NumPy, Torch CPU, and CUDA RNG across model parallelism."""
    states = sync_object_for_parallel_training(
        _collect_rng_states(),
        parallel_dims=parallel_dims,
        force_object=True,
    )
    _set_rng_states(states)
