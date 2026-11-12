"""Optional distributed VAE decode helpers.

This module is intentionally independent from the inference pipelines.  The
default pipeline path still calls ``vae.decode`` directly; callers opt in here
when they want to split VAE decoding across ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TemporalDecodeSplit:
    start: int
    end: int
    context_start: int


def is_dcae_vae(vae: torch.nn.Module) -> bool:
    """Return whether *vae* is the local DCAE-V wrapper."""
    return vae.__class__.__name__ == "AutoencoderDCAEV" and hasattr(vae, "model")


def infer_temporal_ratio(vae: torch.nn.Module, default: int = 4) -> int:
    for attr in (
        "time_compression_ratio",
        "ffactor_temporal",
        "temporal_compression_ratio",
    ):
        value = getattr(vae, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    cfg = getattr(vae, "config", None)
    for attr in (
        "time_compression_ratio",
        "ffactor_temporal",
        "temporal_compression_ratio",
    ):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return default


def infer_spatial_ratio(vae: torch.nn.Module, default: int = 32) -> int:
    for attr in ("spatial_compression_ratio", "ffactor_spatial"):
        value = getattr(vae, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    cfg = getattr(vae, "config", None)
    for attr in ("spatial_compression_ratio", "ffactor_spatial"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return default


def infer_dcae_decode_tile_latent_frames(
    vae: torch.nn.Module, default: int = 10
) -> int:
    cfg = getattr(getattr(vae, "model", None), "cfg", None)
    value = getattr(cfg, "decode_temporal_tile_latent_size", None)
    if isinstance(value, int) and value > 0:
        return value
    return default


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_down(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return (value // multiple) * multiple


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return _ceil_div(value, multiple) * multiple


def make_temporal_splits(
    total_latent_frames: int,
    world_size: int,
    *,
    rank: int,
    context_latent_frames: int,
    split_multiple: int,
) -> TemporalDecodeSplit:
    """Balanced contiguous T splits with optional left context.

    Split boundaries are aligned to ``split_multiple`` by distributing tile
    units.  If there are more ranks than units, high ranks receive an empty
    slice but still participate in the collective gather.
    """
    if total_latent_frames <= 0:
        return TemporalDecodeSplit(0, 0, 0)
    world_size = max(1, world_size)
    rank = max(rank, 0)

    unit = max(1, split_multiple)
    num_units = _ceil_div(total_latent_frames, unit)
    active_ranks = max(1, min(world_size, num_units))
    if rank >= active_ranks:
        return TemporalDecodeSplit(
            total_latent_frames, total_latent_frames, total_latent_frames
        )

    unit_start = rank * num_units // active_ranks
    unit_end = (rank + 1) * num_units // active_ranks
    start = min(unit_start * unit, total_latent_frames)
    end = min(unit_end * unit, total_latent_frames)

    context_latent_frames = max(0, context_latent_frames)
    context_start = max(0, start - context_latent_frames)
    if split_multiple > 1:
        context_start = _align_down(context_start, split_multiple)
    return TemporalDecodeSplit(start, end, context_start)


def _group_rank_to_global(group, group_rank: int, device: torch.device) -> int:
    if group is None:
        return group_rank
    if hasattr(dist, "get_global_rank"):
        return int(dist.get_global_rank(group, group_rank))

    rank_tensor = torch.tensor([dist.get_rank()], device=device, dtype=torch.int64)
    gathered = [
        torch.zeros_like(rank_tensor) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered, rank_tensor, group=group)
    return int(gathered[group_rank].item())


def _gather_temporal_to_rank0(
    local_video: torch.Tensor,
    *,
    group,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Gather local ``[B,C,T,H,W]`` chunks to group rank 0.

    Non-zero group ranks return their local chunk after sending it; the current
    inference code only consumes the returned video on rank 0.
    """
    if world_size <= 1:
        return local_video

    device = local_video.device
    local_t = torch.tensor([local_video.shape[2]], device=device, dtype=torch.int64)
    lengths = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(lengths, local_t, group=group)
    lengths_i = [int(x.item()) for x in lengths]
    max_t = max(lengths_i)

    if local_video.shape[2] < max_t:
        pad_shape = list(local_video.shape)
        pad_shape[2] = max_t - local_video.shape[2]
        padded = torch.cat(
            [
                local_video,
                torch.zeros(pad_shape, dtype=local_video.dtype, device=device),
            ],
            dim=2,
        )
    else:
        padded = local_video

    group_ranks = [
        _group_rank_to_global(group, group_rank, device)
        for group_rank in range(world_size)
    ]
    dst_global = group_ranks[0]

    if rank == 0:
        chunks = []
        for group_rank, global_rank in enumerate(group_ranks):
            if group_rank == 0:
                recv_buf = padded
            else:
                recv_buf = torch.empty_like(padded)
                dist.recv(recv_buf, src=global_rank)
            chunks.append(recv_buf[:, :, : lengths_i[group_rank]].contiguous())
        return torch.cat(chunks, dim=2)

    dist.send(padded.contiguous(), dst=dst_global)
    return local_video


def _gather_temporal_p2p_batch_to_rank0(
    local_video: torch.Tensor,
    *,
    group,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Gather chunks to group rank 0 using batched async P2P ops."""
    if world_size <= 1:
        return local_video

    device = local_video.device
    local_t = torch.tensor([local_video.shape[2]], device=device, dtype=torch.int64)
    lengths = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(lengths, local_t, group=group)
    lengths_i = [int(x.item()) for x in lengths]
    max_t = max(lengths_i)

    if local_video.shape[2] < max_t:
        pad_shape = list(local_video.shape)
        pad_shape[2] = max_t - local_video.shape[2]
        padded = torch.cat(
            [
                local_video,
                torch.zeros(pad_shape, dtype=local_video.dtype, device=device),
            ],
            dim=2,
        ).contiguous()
    else:
        padded = local_video.contiguous()

    group_ranks = [
        _group_rank_to_global(group, group_rank, device)
        for group_rank in range(world_size)
    ]
    dst_global = group_ranks[0]

    if rank == 0:
        recv_bufs = [padded]
        ops = []
        for group_rank, global_rank in enumerate(group_ranks):
            if group_rank == 0:
                continue
            recv_buf = torch.empty_like(padded)
            recv_bufs.append(recv_buf)
            ops.append(dist.P2POp(dist.irecv, recv_buf, global_rank, group=group))
        if ops:
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        chunks = [
            recv_bufs[group_rank][:, :, : lengths_i[group_rank]].contiguous()
            for group_rank in range(world_size)
        ]
        return torch.cat(chunks, dim=2)

    ops = [dist.P2POp(dist.isend, padded, dst_global, group=group)]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    return local_video


def _gather_temporal_allgather_to_rank0(
    local_video: torch.Tensor,
    *,
    group,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Gather chunks with an all-gather collective.

    This is an alternative exact gather path for benchmarking.  It avoids
    rank-0 sequential recvs, at the cost of materializing the padded chunk list
    on every rank.
    """
    if world_size <= 1:
        return local_video

    device = local_video.device
    local_t = torch.tensor([local_video.shape[2]], device=device, dtype=torch.int64)
    lengths = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(lengths, local_t, group=group)
    lengths_i = [int(x.item()) for x in lengths]
    max_t = max(lengths_i)

    if local_video.shape[2] < max_t:
        pad_shape = list(local_video.shape)
        pad_shape[2] = max_t - local_video.shape[2]
        padded = torch.cat(
            [
                local_video,
                torch.zeros(pad_shape, dtype=local_video.dtype, device=device),
            ],
            dim=2,
        )
    else:
        padded = local_video.contiguous()

    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    if rank != 0:
        return local_video

    chunks = [
        gathered[group_rank][:, :, : lengths_i[group_rank]].contiguous()
        for group_rank in range(world_size)
    ]
    return torch.cat(chunks, dim=2)


def _gather_temporal_allgather_tensor_to_rank0(
    local_video: torch.Tensor,
    *,
    group,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Gather chunks with ``all_gather_into_tensor``.

    This keeps the collective in a single contiguous output tensor.  It is
    useful when the backend can optimize the fixed-size gather better than a
    Python list of tensors.
    """
    if world_size <= 1:
        return local_video
    if not hasattr(dist, "all_gather_into_tensor"):
        return _gather_temporal_allgather_to_rank0(
            local_video,
            group=group,
            rank=rank,
            world_size=world_size,
        )

    device = local_video.device
    local_t = torch.tensor([local_video.shape[2]], device=device, dtype=torch.int64)
    lengths = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(lengths, local_t, group=group)
    lengths_i = [int(x.item()) for x in lengths]
    max_t = max(lengths_i)

    if local_video.shape[2] < max_t:
        pad_shape = list(local_video.shape)
        pad_shape[2] = max_t - local_video.shape[2]
        padded = torch.cat(
            [
                local_video,
                torch.zeros(pad_shape, dtype=local_video.dtype, device=device),
            ],
            dim=2,
        )
    else:
        padded = local_video.contiguous()

    batch = padded.shape[0]
    gathered = torch.empty(
        (world_size * batch, *padded.shape[1:]),
        dtype=padded.dtype,
        device=device,
    )
    dist.all_gather_into_tensor(gathered, padded, group=group)
    if rank != 0:
        return local_video

    chunks = []
    for group_rank in range(world_size):
        start = group_rank * batch
        end = start + batch
        chunks.append(gathered[start:end, :, : lengths_i[group_rank]].contiguous())
    return torch.cat(chunks, dim=2)


@torch.no_grad()
def decode_temporal_parallel(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    group=None,
    rank: int = 0,
    world_size: int = 1,
    context_latent_frames: int = 30,
    split_multiple: Optional[int] = None,
    temporal_ratio: Optional[int] = None,
    exact_prefix: bool = False,
    gather_mode: str = "p2p",
    logger: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Decode DCAE-V latents by splitting latent T over distributed ranks.

    The function returns the full decoded video on group rank 0.  Other ranks
    return their local decoded chunk after participating in communication.
    """
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return vae.decode(latents, return_dict=False)[0]

    if not is_dcae_vae(vae):
        if logger is not None and rank == 0:
            logger(
                "parallel_vae_decode currently supports AutoencoderDCAEV only; "
                "falling back to single-rank decode."
            )
        video = (
            vae.decode(latents, return_dict=False)[0]
            if rank == 0
            else latents.new_empty(0)
        )
        dist.barrier(group=group)
        return video

    total_t = int(latents.shape[2])
    temporal_ratio = temporal_ratio or infer_temporal_ratio(vae)
    split_multiple = split_multiple or infer_dcae_decode_tile_latent_frames(vae)
    split = make_temporal_splits(
        total_t,
        world_size,
        rank=rank,
        context_latent_frames=context_latent_frames,
        split_multiple=split_multiple,
    )
    if exact_prefix and split.end > split.start:
        split = TemporalDecodeSplit(split.start, split.end, 0)

    if logger is not None:
        logger(
            f"parallel_vae_decode rank={rank}/{world_size}: "
            f"latent[{split.start}:{split.end}] context_start={split.context_start} "
            f"exact_prefix={exact_prefix}"
        )

    if split.end <= split.start:
        spatial_ratio = infer_spatial_ratio(vae)
        local_video = latents.new_empty(
            (
                latents.shape[0],
                3,
                0,
                latents.shape[3] * spatial_ratio,
                latents.shape[4] * spatial_ratio,
            )
        )
        if gather_mode == "allgather":
            gather = _gather_temporal_allgather_to_rank0
        elif gather_mode == "allgather_tensor":
            gather = _gather_temporal_allgather_tensor_to_rank0
        elif gather_mode == "p2p_batch":
            gather = _gather_temporal_p2p_batch_to_rank0
        else:
            gather = _gather_temporal_to_rank0
        return gather(
            local_video,
            group=group,
            rank=rank,
            world_size=world_size,
        )

    local_latents = latents[:, :, split.context_start : split.end].contiguous()
    decoded = vae.decode(local_latents, return_dict=False)[0]

    prefix_latents = split.start - split.context_start
    crop_start = prefix_latents * temporal_ratio
    keep_frames = (split.end - split.start) * temporal_ratio
    local_video = decoded[:, :, crop_start : crop_start + keep_frames].contiguous()

    if gather_mode == "allgather":
        gather = _gather_temporal_allgather_to_rank0
    elif gather_mode == "allgather_tensor":
        gather = _gather_temporal_allgather_tensor_to_rank0
    elif gather_mode == "p2p_batch":
        gather = _gather_temporal_p2p_batch_to_rank0
    else:
        gather = _gather_temporal_to_rank0
    return gather(
        local_video,
        group=group,
        rank=rank,
        world_size=world_size,
    )
