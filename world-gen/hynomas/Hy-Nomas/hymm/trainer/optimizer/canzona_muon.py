import heapq
import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
except ImportError:  # pragma: no cover - old torch fallback
    DTensor = None
    Partial = None
    Replicate = None
    Shard = None


@dataclass(frozen=True)
class CanzonaShardSpec:
    full_shape: torch.Size
    local_shape: torch.Size
    shard_dim: int
    device_mesh: object
    placements: tuple
    mesh_dim: int
    group: object
    group_rank: int
    world_size: int


@dataclass
class GatherState:
    full_grads_group: object
    full_moms_group: object
    send_buffer: object = None
    send_split_sizes: object = None
    recv_buffer: object = None
    recv_split_sizes: object = None
    local_numels: object = None


@dataclass
class ScatterState:
    shard_updates_group: object
    shard_moms_group: object
    send_buffer: object = None
    send_split_sizes: object = None
    recv_buffer: object = None
    recv_split_sizes: object = None


def _numel(shape: Sequence[int]) -> int:
    return math.prod(tuple(shape))


def _local_tensor(tensor):
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def _get_group_rank(group) -> int:
    try:
        return dist.get_rank(group)
    except TypeError:
        return group.rank()


def _get_group_world_size(group) -> int:
    try:
        return dist.get_world_size(group)
    except TypeError:
        return group.size()


def _mesh_group(device_mesh, mesh_dim: int):
    try:
        return device_mesh.get_group(mesh_dim=mesh_dim)
    except TypeError:
        return device_mesh.get_group(mesh_dim)


def _dtensor_from_local(local_tensor: torch.Tensor, ref):
    if DTensor is None or not isinstance(ref, DTensor):
        return local_tensor
    return DTensor.from_local(
        local_tensor,
        device_mesh=ref.device_mesh,
        placements=ref.placements,
        run_check=False,
    )


def get_dtensor_shard_spec(param, grad) -> CanzonaShardSpec:
    if DTensor is None or not isinstance(grad, DTensor):
        raise ValueError("Canzona Muon requires DTensor gradients.")

    placements = tuple(grad.placements)
    shard_entries = [
        (idx, placement)
        for idx, placement in enumerate(placements)
        if isinstance(placement, Shard)
    ]
    if len(shard_entries) != 1:
        raise ValueError(
            f"Canzona Muon supports exactly one Shard placement, got {placements}."
        )
    if Partial is not None and any(
        isinstance(placement, Partial) for placement in placements
    ):
        raise ValueError(
            f"Canzona Muon does not support Partial placements, got {placements}."
        )

    mesh_dim, shard = shard_entries[0]
    shard_dim = shard.dim
    full_shape = torch.Size(grad.shape)
    local_shape = torch.Size(grad._local_tensor.shape)
    if shard_dim < 0:
        shard_dim += len(full_shape)
    if len(full_shape) != len(local_shape):
        raise ValueError(
            f"Canzona Muon local shape {tuple(local_shape)} and full shape {tuple(full_shape)} "
            "must have the same rank."
        )

    group = _mesh_group(grad.device_mesh, mesh_dim)
    world_size = _get_group_world_size(group)
    if full_shape[shard_dim] != local_shape[shard_dim] * world_size:
        raise ValueError(
            "Canzona Muon currently requires uniform DTensor sharding: "
            f"full_shape[{shard_dim}]={full_shape[shard_dim]} must equal "
            f"local_shape[{shard_dim}]={local_shape[shard_dim]} * world_size={world_size}."
        )
    for dim, (local_dim, full_dim) in enumerate(zip(local_shape, full_shape)):
        if dim == shard_dim:
            continue
        if local_dim != full_dim:
            raise ValueError(
                f"Canzona Muon local shape {tuple(local_shape)} and full shape "
                f"{tuple(full_shape)} can only differ along shard_dim={shard_dim}."
            )

    if not isinstance(param, DTensor):
        raise ValueError(
            "Canzona Muon expects DTensor parameters when gradients are DTensors."
        )

    return CanzonaShardSpec(
        full_shape=full_shape,
        local_shape=local_shape,
        shard_dim=shard_dim,
        device_mesh=grad.device_mesh,
        placements=placements,
        mesh_dim=mesh_dim,
        group=group,
        group_rank=_get_group_rank(group),
        world_size=world_size,
    )


def ensure_sharded_momentum(state: dict, grad, spec: CanzonaShardSpec):
    local_grad = grad._local_tensor.bfloat16()
    mb = state.get("momentum_buffer")
    if mb is None:
        local_mb = torch.zeros_like(local_grad, dtype=torch.bfloat16)
        mb = DTensor.from_local(
            local_mb,
            device_mesh=spec.device_mesh,
            placements=spec.placements,
            run_check=False,
        )
        state["momentum_buffer"] = mb
        return mb

    if DTensor is not None and isinstance(mb, DTensor):
        local_mb = mb._local_tensor
        if local_mb.dtype != torch.bfloat16:
            mb = DTensor.from_local(
                local_mb.to(torch.bfloat16),
                device_mesh=mb.device_mesh,
                placements=mb.placements,
                run_check=False,
            )
            state["momentum_buffer"] = mb
        return mb

    if isinstance(mb, torch.Tensor):
        # Legacy full-tensor checkpoints cannot be losslessly mapped without
        # knowing the local shard offset. Force an explicit restart path.
        raise RuntimeError(
            "Canzona Muon found a legacy full momentum_buffer. "
            "Resume with muon_comm_mode='legacy' or start Canzona from a fresh optimizer state."
        )

    raise TypeError(f"Unsupported momentum_buffer type for Canzona Muon: {type(mb)!r}")


class CanzonaMuonExecutor:
    def __init__(
        self, balance: str = "global", fused_comm: bool = True, overlap: str = "none"
    ):
        if not fused_comm:
            raise NotImplementedError(
                "Canzona Muon currently requires fused all_to_all_single communication."
            )
        if overlap != "none":
            raise NotImplementedError(
                "Canzona Muon overlap is not implemented in this integration yet."
            )
        self.balance = balance
        self.fused_comm = fused_comm
        self.overlap = overlap
        self._momentum_by_param_id = {}

    def execute(
        self,
        params: Sequence,
        specs: Sequence[CanzonaShardSpec],
        momentum_by_param_id: dict,
        step_fn: Callable,
        update_fn: Callable,
    ):
        if not params:
            return
        self._validate_common_group(specs)
        self._momentum_by_param_id = momentum_by_param_id
        group = specs[0].group
        group_rank = specs[0].group_rank
        world_size = specs[0].world_size
        micro_groups = self._micro_groups(params, specs, world_size)
        for micro_group in micro_groups:
            full_grads_group, full_moms_group = self.gather(
                micro_group, group, group_rank, world_size
            )
            full_updates_group, updated_moms_group = self.compute(
                micro_group,
                full_grads_group,
                full_moms_group,
                group_rank,
                step_fn,
            )
            shard_updates_group, shard_moms_group = self.scatter(
                micro_group,
                full_updates_group,
                updated_moms_group,
                group,
                group_rank,
                world_size,
            )
            self.update(
                micro_group,
                shard_updates_group,
                shard_moms_group,
                group_rank,
                update_fn,
            )
        self._momentum_by_param_id = {}

    def _validate_common_group(self, specs: Sequence[CanzonaShardSpec]):
        first = specs[0]
        for spec in specs[1:]:
            if spec.group != first.group:
                raise ValueError(
                    "Canzona Muon requires all params in one optimizer group to share a shard group."
                )
            if (
                spec.group_rank != first.group_rank
                or spec.world_size != first.world_size
            ):
                raise ValueError(
                    "Canzona Muon shard group metadata is inconsistent across parameters."
                )

    def _micro_groups(
        self, params: Sequence, specs: Sequence[CanzonaShardSpec], world_size: int
    ):
        items = [
            (_numel(spec.full_shape), idx, param, spec)
            for idx, (param, spec) in enumerate(zip(params, specs))
            if param.grad is not None
        ]
        if self.balance == "no":
            groups = []
            for start in range(0, len(items), world_size):
                slots = [[] for _ in range(world_size)]
                for rank, (_, _, param, spec) in enumerate(
                    items[start : start + world_size]
                ):
                    slots[rank].append((param, spec))
                groups.append(list(enumerate(slots)))
            return groups
        if self.balance not in {"single", "slot", "global"}:
            raise ValueError(f"Unknown Canzona Muon balance mode: {self.balance}")

        items.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if self.balance == "single":
            groups = []
            for start in range(0, len(items), world_size):
                slots = [[] for _ in range(world_size)]
                for rank, (_, _, param, spec) in enumerate(
                    items[start : start + world_size]
                ):
                    slots[rank].append((param, spec))
                groups.append(list(enumerate(slots)))
            return groups

        slots = [[] for _ in range(world_size)]
        loads = [0] * world_size
        heap = [(0, rank) for rank in range(world_size)]
        heapq.heapify(heap)
        for cost, _, param, spec in items:
            load, rank = heapq.heappop(heap)
            slots[rank].append((param, spec))
            loads[rank] = load + cost
            heapq.heappush(heap, (loads[rank], rank))
        return [list(enumerate(slots))]

    def gather(self, micro_group, group, group_rank: int, world_size: int):
        state = self._pack_gather(micro_group, group_rank, world_size)
        if state.send_buffer is not None:
            dist.all_to_all_single(
                state.recv_buffer,
                state.send_buffer,
                output_split_sizes=state.recv_split_sizes,
                input_split_sizes=state.send_split_sizes,
                group=group,
            )
        return self._unpack_gather(micro_group, state, group_rank, world_size)

    def _pack_gather(self, micro_group, group_rank: int, world_size: int):
        full_grads_group = [[] for _ in range(world_size)]
        full_moms_group = [[] for _ in range(world_size)]
        ref = self._first_grad(micro_group)
        if ref is None:
            return GatherState(full_grads_group, full_moms_group)

        send_tensors = []
        send_split_sizes = []
        for _, slot in micro_group:
            pieces = []
            for p, spec in slot:
                local_grad = (
                    p.grad._local_tensor.bfloat16().contiguous().view(spec.local_shape)
                )
                local_mom = self._local_momentum(p).contiguous().view(spec.local_shape)
                pieces.append(local_grad.reshape(-1))
                pieces.append(local_mom.reshape(-1))
            if pieces:
                tensor = torch.cat(pieces).contiguous()
            else:
                tensor = torch.empty(0, dtype=torch.bfloat16, device=ref.device)
            send_tensors.append(tensor)
            send_split_sizes.append(tensor.numel())

        _, my_slot = micro_group[group_rank]
        local_numels = [_numel(spec.local_shape) for _, spec in my_slot]
        recv_per_rank = 2 * sum(local_numels)
        recv_split_sizes = [recv_per_rank] * world_size
        send_buffer = (
            torch.cat(send_tensors).contiguous()
            if send_tensors
            else torch.empty(0, dtype=torch.bfloat16, device=ref.device)
        )
        recv_buffer = torch.empty(
            sum(recv_split_sizes), dtype=torch.bfloat16, device=ref.device
        )
        return GatherState(
            full_grads_group=full_grads_group,
            full_moms_group=full_moms_group,
            send_buffer=send_buffer,
            send_split_sizes=send_split_sizes,
            recv_buffer=recv_buffer,
            recv_split_sizes=recv_split_sizes,
            local_numels=local_numels,
        )

    def _unpack_gather(
        self, micro_group, state: GatherState, group_rank: int, world_size: int
    ):
        if state.recv_buffer is None:
            return state.full_grads_group, state.full_moms_group
        _, my_slot = micro_group[group_rank]
        recv_streams = torch.split(state.recv_buffer, state.recv_split_sizes)
        offset = 0
        for (_, spec), shard_numel in zip(my_slot, state.local_numels):
            grad_shards = []
            mom_shards = []
            for src_rank in range(world_size):
                stream = recv_streams[src_rank]
                grad_shards.append(
                    stream[offset : offset + shard_numel]
                    .contiguous()
                    .view(spec.local_shape)
                )
                mom_start = offset + shard_numel
                mom_shards.append(
                    stream[mom_start : mom_start + shard_numel]
                    .contiguous()
                    .view(spec.local_shape)
                )
            state.full_grads_group[group_rank].append(
                torch.cat(grad_shards, dim=spec.shard_dim).view(spec.full_shape)
            )
            state.full_moms_group[group_rank].append(
                torch.cat(mom_shards, dim=spec.shard_dim).view(spec.full_shape)
            )
            offset += 2 * shard_numel
        return state.full_grads_group, state.full_moms_group

    def compute(
        self,
        micro_group,
        full_grads_group,
        full_moms_group,
        group_rank: int,
        step_fn: Callable,
    ):
        full_updates_group = [[] for _ in full_grads_group]
        updated_moms_group = [[] for _ in full_moms_group]
        for host_rank, slot in micro_group:
            for idx, (p, spec) in enumerate(slot):
                if host_rank == group_rank:
                    update, new_mom = step_fn(
                        p,
                        spec,
                        full_grads_group[host_rank][idx],
                        full_moms_group[host_rank][idx],
                    )
                    full_updates_group[host_rank].append(update)
                    updated_moms_group[host_rank].append(new_mom)
                else:
                    full_updates_group[host_rank].append(None)
                    updated_moms_group[host_rank].append(None)
        return full_updates_group, updated_moms_group

    def scatter(
        self,
        micro_group,
        full_updates_group,
        updated_moms_group,
        group,
        group_rank: int,
        world_size: int,
    ):
        state = self._pack_scatter(
            micro_group, full_updates_group, updated_moms_group, group_rank, world_size
        )
        if state.send_buffer is not None:
            dist.all_to_all_single(
                state.recv_buffer,
                state.send_buffer,
                output_split_sizes=state.recv_split_sizes,
                input_split_sizes=state.send_split_sizes,
                group=group,
            )
        return self._unpack_scatter(micro_group, state)

    def _pack_scatter(
        self,
        micro_group,
        full_updates_group,
        updated_moms_group,
        group_rank: int,
        world_size: int,
    ):
        shard_updates_group = [[] for _ in range(world_size)]
        shard_moms_group = [[] for _ in range(world_size)]
        ref = self._first_grad(micro_group)
        if ref is None:
            return ScatterState(shard_updates_group, shard_moms_group)

        send_streams = [[] for _ in range(world_size)]
        for host_rank, slot in micro_group:
            if host_rank != group_rank:
                continue
            for (p, spec), full_update, full_mom in zip(
                slot, full_updates_group[host_rank], updated_moms_group[host_rank]
            ):
                update_shards = torch.chunk(
                    full_update.contiguous().view(spec.full_shape),
                    world_size,
                    dim=spec.shard_dim,
                )
                mom_shards = torch.chunk(
                    full_mom.contiguous().view(spec.full_shape),
                    world_size,
                    dim=spec.shard_dim,
                )
                for target_rank, (update_shard, mom_shard) in enumerate(
                    zip(update_shards, mom_shards)
                ):
                    send_streams[target_rank].append(
                        update_shard.contiguous().reshape(-1)
                    )
                    send_streams[target_rank].append(mom_shard.contiguous().reshape(-1))

        flat_send_tensors = []
        send_split_sizes = []
        for stream in send_streams:
            if stream:
                tensor = torch.cat(stream).contiguous()
            else:
                tensor = torch.empty(0, dtype=torch.bfloat16, device=ref.device)
            flat_send_tensors.append(tensor)
            send_split_sizes.append(tensor.numel())

        recv_split_sizes = [
            2 * sum(_numel(spec.local_shape) for _, spec in slot)
            for _, slot in micro_group
        ]
        send_buffer = (
            torch.cat(flat_send_tensors).contiguous()
            if flat_send_tensors
            else torch.empty(0, dtype=torch.bfloat16, device=ref.device)
        )
        recv_buffer = torch.empty(
            sum(recv_split_sizes), dtype=torch.bfloat16, device=ref.device
        )
        return ScatterState(
            shard_updates_group=shard_updates_group,
            shard_moms_group=shard_moms_group,
            send_buffer=send_buffer,
            send_split_sizes=send_split_sizes,
            recv_buffer=recv_buffer,
            recv_split_sizes=recv_split_sizes,
        )

    def _unpack_scatter(self, micro_group, state: ScatterState):
        if state.recv_buffer is None:
            return state.shard_updates_group, state.shard_moms_group
        recv_streams = torch.split(state.recv_buffer, state.recv_split_sizes)
        for src_rank, (_, slot) in enumerate(micro_group):
            stream = recv_streams[src_rank]
            offset = 0
            for _, spec in slot:
                shard_numel = _numel(spec.local_shape)
                state.shard_updates_group[src_rank].append(
                    stream[offset : offset + shard_numel]
                    .contiguous()
                    .view(spec.local_shape)
                )
                mom_start = offset + shard_numel
                state.shard_moms_group[src_rank].append(
                    stream[mom_start : mom_start + shard_numel]
                    .contiguous()
                    .view(spec.local_shape)
                )
                offset += 2 * shard_numel
        return state.shard_updates_group, state.shard_moms_group

    def update(
        self,
        micro_group,
        shard_updates_group,
        shard_moms_group,
        group_rank: int,
        update_fn: Callable,
    ):
        for (_, slot), shard_updates_slot, shard_moms_slot in zip(
            micro_group, shard_updates_group, shard_moms_group
        ):
            for (p, spec), shard_update, shard_mom in zip(
                slot, shard_updates_slot, shard_moms_slot
            ):
                update_fn(p, spec, shard_update, shard_mom)

    @staticmethod
    def _first_grad(micro_group):
        for _, slot in micro_group:
            if slot:
                return slot[0][0].grad._local_tensor
        return None

    def _local_momentum(self, p):
        mb = self._momentum_by_param_id.get(id(p))
        if mb is None:
            raise RuntimeError(
                "Canzona Muon momentum must be initialized before communication."
            )
        return _local_tensor(mb).bfloat16()
