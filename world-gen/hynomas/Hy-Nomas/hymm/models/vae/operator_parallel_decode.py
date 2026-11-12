"""Experimental operator-level distributed decode for DCAE-V.

This module keeps the DCAE-V implementation untouched.  It exploits the
existing ``feature_cache.get/set`` interface used by chunk-causal conv layers:
the first chunk on rank ``r`` receives its left causal feature from rank
``r - 1``; the last chunk sends its feature cache to rank ``r + 1``.

The default production path remains temporal context splitting in
``parallel_decode.py``.  Use this module only behind an explicit opt-in.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist

from hymm.models.vae.autoencoder_dcae_v import DCAEVStreamingDecodeSession
from hymm.models.vae.parallel_decode import (
    _gather_temporal_to_rank0,
    _group_rank_to_global,
    infer_dcae_decode_tile_latent_frames,
    infer_spatial_ratio,
    is_dcae_vae,
    make_temporal_splits,
)


class DistributedCausalFeatureCache(dict):
    """Feature-cache proxy that exchanges causal halos with neighbor ranks."""

    def __init__(
        self,
        *,
        group,
        rank: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype,
        send_enabled: bool = True,
        async_send: bool = False,
        prev_rank: Optional[int] = None,
        next_rank: Optional[int] = None,
    ):
        super().__init__()
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.dtype = dtype
        self.send_enabled = send_enabled
        self.async_send = async_send
        self._pending_sends: list[
            tuple[torch.Tensor, torch.Tensor, object, object]
        ] = []
        if prev_rank is None:
            prev_rank = rank - 1 if rank > 0 else -1
        if next_rank is None:
            next_rank = rank + 1 if rank + 1 < world_size else -1
        self.prev_global = (
            _group_rank_to_global(group, prev_rank, device) if prev_rank >= 0 else None
        )
        self.next_global = (
            _group_rank_to_global(group, next_rank, device) if next_rank >= 0 else None
        )

    def get(self, key, default=None):  # noqa: D401 - matches dict API
        if key in self:
            return super().get(key, default)
        if self.rank == 0:
            return default

        shape = torch.empty(6, dtype=torch.int64, device=self.device)
        dist.recv(shape, src=self.prev_global)
        shape_i = [int(x) for x in shape.tolist()]
        if any(x <= 0 for x in shape_i):
            return default
        tensor = torch.empty(shape_i, dtype=self.dtype, device=self.device)
        dist.recv(tensor, src=self.prev_global)
        super().__setitem__(key, tensor)
        return tensor

    def __setitem__(self, key, value):
        value = value.contiguous()
        super().__setitem__(key, value)
        if not self.send_enabled or self.next_global is None:
            return

        shape = torch.tensor(value.shape, dtype=torch.int64, device=value.device)
        if self.async_send:
            shape_work = dist.isend(shape, dst=self.next_global)
            value_work = (
                dist.isend(value, dst=self.next_global) if value.numel() > 0 else None
            )
            self._pending_sends.append((shape, value, shape_work, value_work))
            return

        dist.send(shape, dst=self.next_global)
        if value.numel() > 0:
            dist.send(value, dst=self.next_global)

    def wait_pending_sends(self) -> None:
        for _shape, _value, shape_work, value_work in self._pending_sends:
            shape_work.wait()
            if value_work is not None:
                value_work.wait()
        self._pending_sends.clear()


def _make_empty_video_like_latents(
    vae: torch.nn.Module, latents: torch.Tensor
) -> torch.Tensor:
    spatial_ratio = infer_spatial_ratio(vae)
    return latents.new_empty(
        (
            latents.shape[0],
            3,
            0,
            latents.shape[3] * spatial_ratio,
            latents.shape[4] * spatial_ratio,
        )
    )


def _decode_local_operator_once(
    vae: torch.nn.Module,
    local_latents: torch.Tensor,
    *,
    group,
    rank: int,
    active_world_size: int,
) -> torch.Tensor:
    cache = DistributedCausalFeatureCache(
        group=group,
        rank=rank,
        world_size=active_world_size,
        device=local_latents.device,
        dtype=local_latents.dtype,
        send_enabled=True,
    )
    out = vae.model.decoder(local_latents, cache, "decoder.")
    cache.wait_pending_sends()
    return out


def _decode_local_operator_tiled(
    vae: torch.nn.Module,
    local_latents: torch.Tensor,
    *,
    group,
    rank: int,
    active_world_size: int,
    tile_latent_frames: int,
    async_send: bool = False,
) -> torch.Tensor:
    rows = []
    cache: Optional[DistributedCausalFeatureCache] = None
    for start in range(0, local_latents.shape[2], tile_latent_frames):
        end = min(start + tile_latent_frames, local_latents.shape[2])
        is_last_tile = end >= local_latents.shape[2]
        if cache is None:
            cache = DistributedCausalFeatureCache(
                group=group,
                rank=rank,
                world_size=active_world_size,
                device=local_latents.device,
                dtype=local_latents.dtype,
                send_enabled=is_last_tile,
                async_send=async_send,
            )
        else:
            cache.send_enabled = is_last_tile
        rows.append(
            vae.model.decoder(local_latents[:, :, start:end], cache, "decoder.")
        )
    if cache is not None:
        cache.wait_pending_sends()
    return torch.cat(rows, dim=2)


def _gather_indexed_temporal_to_rank0(
    local_chunks: list[torch.Tensor],
    local_indices: list[int],
    *,
    vae: torch.nn.Module,
    latents: torch.Tensor,
    group,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Gather round-robin temporal tiles and restore global tile order."""
    if local_chunks:
        local_video = torch.cat(local_chunks, dim=2).contiguous()
        local_lengths = [int(chunk.shape[2]) for chunk in local_chunks]
    else:
        local_video = _make_empty_video_like_latents(vae, latents)
        local_lengths = []

    metadata = {
        "indices": [int(index) for index in local_indices],
        "lengths": local_lengths,
    }
    all_metadata = [None for _ in range(world_size)]
    dist.all_gather_object(all_metadata, metadata, group=group)

    local_t = int(local_video.shape[2])
    max_t = max(sum(item["lengths"]) for item in all_metadata)
    if local_t < max_t:
        pad_shape = list(local_video.shape)
        pad_shape[2] = max_t - local_t
        local_video = torch.cat(
            [
                local_video,
                torch.zeros(
                    pad_shape,
                    dtype=local_video.dtype,
                    device=local_video.device,
                ),
            ],
            dim=2,
        )

    group_ranks = [
        _group_rank_to_global(group, group_rank, latents.device)
        for group_rank in range(world_size)
    ]
    if rank != 0:
        dist.send(local_video.contiguous(), dst=group_ranks[0])
        return torch.cat(local_chunks, dim=2) if local_chunks else local_video[:, :, :0]

    indexed_chunks: list[tuple[int, torch.Tensor]] = []
    for group_rank, global_rank in enumerate(group_ranks):
        if group_rank == 0:
            received = local_video
        else:
            received = torch.empty_like(local_video)
            dist.recv(received, src=global_rank)
        offset = 0
        rank_meta = all_metadata[group_rank]
        for tile_index, tile_length in zip(rank_meta["indices"], rank_meta["lengths"]):
            indexed_chunks.append(
                (
                    int(tile_index),
                    received[:, :, offset : offset + int(tile_length)].contiguous(),
                )
            )
            offset += int(tile_length)

    indexed_chunks.sort(key=lambda item: item[0])
    return torch.cat([chunk for _, chunk in indexed_chunks], dim=2)


def _decode_operator_round_robin_tiles(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    group,
    rank: int,
    world_size: int,
    tile_latent_frames: int,
) -> torch.Tensor:
    """Pipeline exact causal tiles in waves of at most ``world_size`` tiles.

    Within one wave, rank ``r`` owns the r-th consecutive tile and layer-local
    feature caches flow left-to-right.  At the wave boundary, the last active
    rank transfers its completed cache to rank 0 in one batched P2P exchange.
    This avoids a cyclic NCCL dependency while preserving the exact reference
    tile order and keeping each decoder invocation memory-bounded.
    """
    total_t = int(latents.shape[2])
    num_full_tiles = total_t // tile_latent_frames
    remainder_t = total_t - num_full_tiles * tile_latent_frames
    local_chunks: list[torch.Tensor] = []
    local_indices: list[int] = []
    carry_cache: Optional[dict[str, torch.Tensor]] = None
    num_waves = (num_full_tiles + world_size - 1) // world_size
    rank0_global = _group_rank_to_global(group, 0, latents.device)

    for wave in range(num_waves):
        wave_start = wave * world_size
        active_ranks = min(world_size, num_full_tiles - wave_start)
        local_cache: Optional[DistributedCausalFeatureCache] = None

        if rank < active_ranks:
            tile_index = wave_start + rank
            start = tile_index * tile_latent_frames
            end = start + tile_latent_frames
            local_cache = DistributedCausalFeatureCache(
                group=group,
                rank=rank,
                world_size=active_ranks,
                device=latents.device,
                dtype=latents.dtype,
                send_enabled=rank + 1 < active_ranks,
                async_send=True,
                prev_rank=rank - 1 if rank > 0 else -1,
                next_rank=rank + 1 if rank + 1 < active_ranks else -1,
            )
            if rank == 0 and carry_cache:
                local_cache.update(carry_cache)
            decoded = vae.model.decoder(
                latents[:, :, start:end].contiguous(),
                local_cache,
                "decoder.",
            )
            local_cache.wait_pending_sends()
            local_chunks.append(decoded)
            local_indices.append(tile_index)

        # Move the final causal state of this wave back to rank 0.  All ranks
        # participate in the barrier so the next wave cannot overtake this
        # cache handoff.
        last_rank = active_ranks - 1
        if active_ranks > 1:
            last_global = _group_rank_to_global(group, last_rank, latents.device)
            if rank == 0:
                assert local_cache is not None
                keys = list(local_cache.keys())
                received = {key: torch.empty_like(local_cache[key]) for key in keys}
                ops = [
                    dist.P2POp(
                        dist.irecv,
                        received[key],
                        last_global,
                        group=group,
                    )
                    for key in keys
                ]
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
                carry_cache = received
            elif rank == last_rank:
                assert local_cache is not None
                keys = list(local_cache.keys())
                ops = [
                    dist.P2POp(
                        dist.isend,
                        local_cache[key].contiguous(),
                        rank0_global,
                        group=group,
                    )
                    for key in keys
                ]
                for work in dist.batch_isend_irecv(ops):
                    work.wait()
        elif rank == 0 and local_cache is not None:
            carry_cache = {key: value for key, value in local_cache.items()}
        dist.barrier(group=group)

    if remainder_t > 0:
        remainder_owner = 0
        if rank == remainder_owner:
            remainder = vae.model.decoder(
                latents[:, :, num_full_tiles * tile_latent_frames :].contiguous(),
                None,
                None,
            )
            local_chunks.append(remainder)
            local_indices.append(num_full_tiles)

    return _gather_indexed_temporal_to_rank0(
        local_chunks,
        local_indices,
        vae=vae,
        latents=latents,
        group=group,
        rank=rank,
        world_size=world_size,
    )


@torch.no_grad()
def decode_operator_temporal_parallel(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    group=None,
    rank: int = 0,
    world_size: int = 1,
    split_multiple: Optional[int] = None,
    tile_loop: bool = False,
    async_cache_send: bool = False,
    logger=None,
) -> torch.Tensor:
    """Decode by sharding latent T and exchanging layer-local causal halos.

    Returns the full decoded video on group rank 0; other ranks return their
    local decoded chunk after participating in communication.
    """
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return vae.decode(latents, return_dict=False)[0]
    if not is_dcae_vae(vae):
        if logger is not None and rank == 0:
            logger(
                "operator VAE parallel only supports AutoencoderDCAEV; falling back."
            )
        video = (
            vae.decode(latents, return_dict=False)[0]
            if rank == 0
            else latents.new_empty(0)
        )
        dist.barrier(group=group)
        return video

    total_t = int(latents.shape[2])
    split_multiple = split_multiple or infer_dcae_decode_tile_latent_frames(vae)
    if tile_loop:
        if logger is not None and rank == 0:
            logger(
                "operator_parallel_vae_decode: exact round-robin tile pipeline "
                f"world_size={world_size}, tile_latent={split_multiple}, total_t={total_t}"
            )
        return _decode_operator_round_robin_tiles(
            vae,
            latents,
            group=group,
            rank=rank,
            world_size=world_size,
            tile_latent_frames=split_multiple,
        )

    # DCAE-V's reference temporal_tiled_decode processes complete
    # ``split_multiple`` chunks with the causal feature cache, then decodes a
    # short tail (if any) as one ordinary non-cache decoder call.  Including
    # that short tail in an operator-parallel shard is both incorrect and can
    # deadlock: ConvLayer3d does not call feature_cache.get/set when the local
    # temporal length is not divisible by its causal chunk length.
    #
    # Keep only complete tiles in the distributed causal pipeline and append
    # the reference-style short-tail decode on the last active rank.
    parallel_t = (total_t // split_multiple) * split_multiple
    remainder_t = total_t - parallel_t
    if parallel_t == 0:
        video = (
            vae.decode(latents, return_dict=False)[0]
            if rank == 0
            else _make_empty_video_like_latents(vae, latents)
        )
        return _gather_temporal_to_rank0(
            video,
            group=group,
            rank=rank,
            world_size=world_size,
        )

    split = make_temporal_splits(
        parallel_t,
        world_size,
        rank=rank,
        context_latent_frames=0,
        split_multiple=split_multiple,
    )
    active_world_size = max(1, min(world_size, parallel_t // split_multiple))

    if logger is not None:
        logger(
            f"operator_parallel_vae_decode rank={rank}/{world_size}: "
            f"active={active_world_size}, latent[{split.start}:{split.end}], "
            f"remainder={remainder_t}, "
            f"tile_loop={tile_loop}, async_cache_send={async_cache_send}"
        )

    if rank >= active_world_size or split.end <= split.start:
        return _gather_temporal_to_rank0(
            _make_empty_video_like_latents(vae, latents),
            group=group,
            rank=rank,
            world_size=world_size,
        )

    local_latents = latents[:, :, split.start : split.end].contiguous()
    local_video = _decode_local_operator_once(
        vae,
        local_latents,
        group=group,
        rank=rank,
        active_world_size=active_world_size,
    )

    if remainder_t > 0 and rank == active_world_size - 1:
        remainder_latents = latents[:, :, parallel_t:].contiguous()
        # Match DCAEV.temporal_tiled_decode: the short final tile is passed to
        # decoder with the feature-cache object, but every chunk-causal conv
        # takes its non-cache path because T is not divisible by chunk length.
        # Passing no cache is therefore equivalent and avoids unmatched P2P.
        remainder_video = vae.model.decoder(remainder_latents, None, None)
        local_video = torch.cat([local_video, remainder_video], dim=2)

    return _gather_temporal_to_rank0(
        local_video.contiguous(),
        group=group,
        rank=rank,
        world_size=world_size,
    )


class DistributedDCAEVStreamingDecodeSession(DCAEVStreamingDecodeSession):
    """Incremental DCAE decode with temporal tiles pipelined over an SP group.

    Rank 0 owns the canonical feature cache between ``push`` calls. Complete
    newly-arrived tiles are assigned to consecutive ranks and exchange their
    layer-local causal cache through ``DistributedCausalFeatureCache``. The
    final active rank returns the updated canonical cache to rank 0, allowing
    the next segment to continue without re-decoding history.

    A short speculative/final tail is inherently smaller than one temporal
    tile, so rank 0 decodes it while all ranks synchronize at the call boundary.
    """

    def __init__(
        self,
        model,
        *,
        group,
        rank: int,
        world_size: int,
    ):
        super().__init__(model)
        self.group = group
        self.rank = int(rank)
        self.world_size = int(world_size)

    def _decode_complete_tiles(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        num_tiles = int(latents.shape[2]) // self.tile_latent_frames
        local_chunks: list[torch.Tensor] = []
        local_indices: list[int] = []
        carry_cache = self.feature_cache if self.rank == 0 else None
        rank0_global = _group_rank_to_global(self.group, 0, latents.device)

        for wave_start in range(0, num_tiles, self.world_size):
            active_ranks = min(self.world_size, num_tiles - wave_start)
            local_cache: Optional[DistributedCausalFeatureCache] = None

            if self.rank < active_ranks:
                tile_index = wave_start + self.rank
                start = tile_index * self.tile_latent_frames
                end = start + self.tile_latent_frames
                local_cache = DistributedCausalFeatureCache(
                    group=self.group,
                    rank=self.rank,
                    world_size=active_ranks,
                    device=latents.device,
                    dtype=latents.dtype,
                    send_enabled=self.rank + 1 < active_ranks,
                    async_send=True,
                )
                if self.rank == 0 and carry_cache:
                    local_cache.update(carry_cache)
                decoded = self.model.decoder(
                    latents[:, :, start:end].contiguous(),
                    local_cache,
                    "decoder." if self.model.cfg.use_feature_cache else None,
                )
                local_cache.wait_pending_sends()
                local_chunks.append(decoded)
                local_indices.append(tile_index)

            last_rank = active_ranks - 1
            if active_ranks > 1:
                last_global = _group_rank_to_global(
                    self.group, last_rank, latents.device
                )
                if self.rank == 0:
                    assert local_cache is not None
                    received = {
                        key: torch.empty_like(value)
                        for key, value in local_cache.items()
                    }
                    ops = [
                        dist.P2POp(
                            dist.irecv,
                            received[key],
                            last_global,
                            group=self.group,
                        )
                        for key in received
                    ]
                    for work in dist.batch_isend_irecv(ops):
                        work.wait()
                    carry_cache = received
                elif self.rank == last_rank:
                    assert local_cache is not None
                    ops = [
                        dist.P2POp(
                            dist.isend,
                            value.contiguous(),
                            rank0_global,
                            group=self.group,
                        )
                        for value in local_cache.values()
                    ]
                    for work in dist.batch_isend_irecv(ops):
                        work.wait()
            elif self.rank == 0:
                assert local_cache is not None
                carry_cache = dict(local_cache)
            dist.barrier(group=self.group)

        if self.rank == 0:
            self.feature_cache = carry_cache
        return _gather_indexed_temporal_to_rank0(
            local_chunks,
            local_indices,
            vae=self.model,
            latents=latents,
            group=self.group,
            rank=self.rank,
            world_size=self.world_size,
        )

    @torch.no_grad()
    def push(
        self,
        latents: torch.Tensor,
        output_callback: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> list[torch.Tensor]:
        self._validate(latents)
        if latents.shape[2] == 0:
            return []

        self.received_latent_frames += int(latents.shape[2])
        buffered = (
            latents
            if self._pending is None
            else torch.cat([self._pending, latents], dim=2)
        )
        complete_frames = (
            int(buffered.shape[2]) // self.tile_latent_frames
        ) * self.tile_latent_frames
        complete = buffered[:, :, :complete_frames]
        remainder = buffered[:, :, complete_frames:]
        self._pending = remainder.clone() if remainder.shape[2] else None
        if complete_frames == 0:
            return []

        decoded = self._decode_complete_tiles(complete)
        outputs: list[torch.Tensor] = []
        if self.rank == 0:
            pixel_frames_per_tile = int(decoded.shape[2]) // (
                complete_frames // self.tile_latent_frames
            )
            for start in range(0, int(decoded.shape[2]), pixel_frames_per_tile):
                tile = decoded[:, :, start : start + pixel_frames_per_tile]
                if output_callback is None:
                    outputs.append(tile)
                else:
                    output_callback(tile)
                self.decoded_latent_frames += self.tile_latent_frames
        else:
            self.decoded_latent_frames += complete_frames
        return outputs

    @torch.no_grad()
    def speculative_decode_pending(
        self,
        output_callback: Optional[Callable[[torch.Tensor], None]] = None,
        *,
        offload_canonical_cache: bool = True,
    ):
        if self.rank == 0:
            result = super().speculative_decode_pending(
                output_callback=output_callback,
                offload_canonical_cache=offload_canonical_cache,
            )
        else:
            result = ([], 0)
        return result

    @torch.no_grad()
    def finish(
        self,
        output_callback: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> list[torch.Tensor]:
        if self.rank == 0:
            outputs = super().finish(output_callback=output_callback)
        else:
            if self._finished:
                raise RuntimeError("Streaming DCAE decode finish() called twice.")
            self._finished = True
            pending_frames = self.pending_latent_frames
            self._pending = None
            self.decoded_latent_frames += pending_frames
            outputs = []
        return outputs
