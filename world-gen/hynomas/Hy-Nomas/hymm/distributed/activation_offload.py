"""Activation offloading to CPU (pinned) with async DMA + double-buffered prefetch.

Offloading attention activations reduces peak GPU memory. Asynchronous
prefetch can overlap transfers with computation, subject to host memory
capacity and transfer bandwidth.

Design: each gradient-bearing top-level model invocation receives a unique
forward id. Logical handles lease physical pinned slots until their own
backward restores them, allowing several segment forwards before one shared
backward. Released slots are pooled for later steps. D2H/H2D may use dedicated
streams, but restore lookup is always handle-specific.

Attention `out` is offloaded for Magi and FA3 selective-checkpoint paths;
small auxiliary outputs stay on GPU.
"""

from __future__ import annotations

import os
import json
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


_RESID_OFFLOAD_MIN_BYTES = 1 * 1024 * 1024


@dataclass
class OffloadHandle:
    """Lightweight stand-in stored where the GPU tensor used to live."""

    handle_id: int
    forward_id: int
    key: str
    numel: int
    shape: Tuple[int, ...]
    dtype: torch.dtype


class ActivationOffloadManager:
    """Manages D2H offload (forward) and H2D prefetch (backward) of activations.

    When ``enabled=False`` every method is a cheap no-op / passthrough so the
    training path is byte-identical to not using it.
    """

    def __init__(
        self,
        enabled: bool,
        device: torch.device,
        num_prefetch_buffers: int = 3,
        async_enabled: Optional[bool] = None,
    ):
        self.enabled = bool(enabled)
        self.device = device
        self._lifecycle_path = os.environ.get(
            "HY_DEBUG_OFFLOAD_LIFECYCLE_PATH", ""
        ).strip()
        self._lifecycle_enabled = bool(
            self._lifecycle_path and int(os.environ.get("LOCAL_RANK", "0")) == 0
        )
        self._lifecycle_lock = threading.Lock()
        # Synchronous by default: correct + saves the memory (offload happens
        # either way; only the transfer overlap is lost, ~+6% step time).
        # Production training passes ``parallelism.offload_activations_async``
        # through async_enabled. ``None`` preserves the legacy OFFLOAD_ASYNC
        # environment-variable interface used by standalone benchmarks.
        # Per-direction async toggles allow isolating the D2H and H2D paths
        # during bring-up/verification. Explicit OFFLOAD_ASYNC_D2H/H2D values
        # remain the highest-priority debugging overrides.
        #   OFFLOAD_ASYNC_D2H=1 -> async offload (forward D2H on a side stream)
        #   OFFLOAD_ASYNC_H2D=1 -> async prefetch (backward H2D on a side stream)
        _async = (
            os.environ.get("OFFLOAD_ASYNC", "0") == "1"
            if async_enabled is None
            else bool(async_enabled)
        )
        self._async_d2h = (
            os.environ.get("OFFLOAD_ASYNC_D2H", "1" if _async else "0") == "1"
        )
        self._async_h2d = (
            os.environ.get("OFFLOAD_ASYNC_H2D", "1" if _async else "0") == "1"
        )
        self._sync = not (
            self._async_d2h or self._async_h2d
        )  # retained for external reads
        # Only block-INPUT tensors this big (img/txt residuals; ~90-370MB @360K)
        # are offloaded by resid_offload_hooks; tiny saved tensors (e.g. the
        # checkpoint dummy) pass through untouched.
        # Fixed implementation policy, not a training hyperparameter. Keeping
        # this threshold in code ensures every activation-offload config covers
        # the same checkpoint residual tensors.
        self._resid_min_bytes = _RESID_OFFLOAD_MIN_BYTES
        # Active leases.  A key is invocation-unique, so two gradient-bearing
        # forwards may coexist until their shared backward consumes both.
        self._cpu: Dict[str, torch.Tensor] = {}
        # Reusable physical pinned slots.  Slots move here only after a complete
        # backward/step boundary; they are never reused while an OffloadHandle
        # can still reference them.
        self._free_cpu: List[torch.Tensor] = []
        self._offload_ev: Dict[str, torch.cuda.Event] = {}  # key -> D2H done event
        self._handles: Dict[
            str, "OffloadHandle"
        ] = {}  # key -> handle (for backward prefetch lookup)
        self._prefetched: Dict[
            str, Tuple[torch.Tensor, torch.cuda.Event]
        ] = {}  # key -> (gpu buf, H2D event)
        # key -> event marking the H2D READ of the persistent pinned slot is done.
        # Guards the cross-step WAR hazard: next step's forward D2H (on _offload_stream)
        # must wait for the previous step's prefetch H2D read (on _prefetch_stream)
        # before overwriting the same reused slot. MUST survive begin_forward().
        self._prefetch_read_ev: Dict[str, torch.cuda.Event] = {}
        self._fwd_counters: Dict[str, int] = {}
        self._next_forward_id = 0
        self._next_handle_id = 0
        self._active_forward_id: Optional[int] = None
        self._active_forward_grad_enabled = False
        self._consumed_handles: set[str] = set()
        self._peak_live_handles = 0
        self._peak_pinned_bytes = 0
        self._peak_active_pinned_bytes = 0
        self._slot_reuse_count = 0
        self._sync_fallback_count = 0
        self._offloaded_bytes_total = 0
        self._restored_bytes_total = 0
        self._offloaded_bytes_by_kind: Dict[str, int] = {}
        self._handles_by_forward: Dict[int, int] = {}
        if self.enabled:
            self._offload_stream = torch.cuda.Stream(device=device)
            self._prefetch_stream = torch.cuda.Stream(device=device)
        self._trace_lifecycle(
            "manager_init",
            async_d2h=self._async_d2h,
            async_h2d=self._async_h2d,
            resid_min_bytes=self._resid_min_bytes,
        )

    @staticmethod
    def _graph_task_id() -> int:
        try:
            return int(torch._C._current_graph_task_id())
        except (AttributeError, RuntimeError):
            return -1

    def _trace_lifecycle(self, event: str, **fields) -> None:
        if not self._lifecycle_enabled:
            return
        record = {
            "ts_ns": time.time_ns(),
            "event": event,
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "graph_task_id": self._graph_task_id(),
            "grad_enabled": bool(torch.is_grad_enabled()),
            **fields,
        }
        os.makedirs(os.path.dirname(self._lifecycle_path), exist_ok=True)
        with self._lifecycle_lock:
            with open(self._lifecycle_path, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    def _trace_restored_finalize(
        self, key: str, gpu_data_ptr: int, nbytes: int
    ) -> None:
        self._trace_lifecycle(
            "restore_tensor_finalize",
            key=key,
            gpu_data_ptr=gpu_data_ptr,
            nbytes=nbytes,
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
            reserved_bytes=int(torch.cuda.memory_reserved(self.device)),
        )

    def _trace_offload_source_finalize(
        self, key: str, gpu_data_ptr: int, nbytes: int
    ) -> None:
        self._trace_lifecycle(
            "offload_source_finalize",
            key=key,
            gpu_data_ptr=gpu_data_ptr,
            nbytes=nbytes,
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
            reserved_bytes=int(torch.cuda.memory_reserved(self.device)),
        )

    def _track_restored_tensor(
        self, tensor: torch.Tensor, handle: OffloadHandle
    ) -> torch.Tensor:
        weakref.finalize(
            tensor,
            self._trace_restored_finalize,
            handle.key,
            int(tensor.data_ptr()),
            handle.numel * tensor.element_size(),
        )
        return tensor

    # ---- forward ----------------------------------------------------------- #
    def begin_forward(self, grad_enabled: Optional[bool] = None):
        """Start one top-level model invocation.

        Segment self-forcing executes several model forwards before one shared
        backward.  Therefore every gradient-bearing invocation receives a
        distinct ``forward_id``; layer-only keys would let later segments
        overwrite activations still owned by earlier autograd graphs.

        ``torch.no_grad()`` cache-write forwards are intentionally ineligible
        for persistent offload because they have no backward consumer.
        """
        if grad_enabled is None:
            grad_enabled = torch.is_grad_enabled()
        self._fwd_counters.clear()
        self._active_forward_grad_enabled = bool(grad_enabled)
        if self.enabled and self._active_forward_grad_enabled:
            self._active_forward_id = self._next_forward_id
            self._next_forward_id += 1
        else:
            self._active_forward_id = None
        self._trace_lifecycle(
            "begin_forward",
            forward_id=self._active_forward_id,
            forward_grad_enabled=self._active_forward_grad_enabled,
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
        )
        return self._active_forward_id

    def should_offload(self) -> bool:
        return bool(
            self.enabled
            and self._active_forward_grad_enabled
            and self._active_forward_id is not None
        )

    def current_forward_id(self) -> Optional[int]:
        return self._active_forward_id

    def next_slot_key(self, name: str) -> str:
        """Return an invocation-unique logical activation key."""
        if not self.should_offload():
            raise RuntimeError(
                "Activation offload requested outside a gradient-bearing model forward"
            )
        idx = self._fwd_counters.get(name, 0)
        self._fwd_counters[name] = idx + 1
        return f"{name}:f{self._active_forward_id}:{idx}"

    def make_resid_key(self, forward_id: int, block_idx: int, tensor_idx: int) -> str:
        return f"resid:f{forward_id}:b{block_idx}:{tensor_idx}"

    def _acquire_cpu_slot(
        self, key: str, numel: int, dtype: torch.dtype
    ) -> torch.Tensor:
        best_idx = None
        best_numel = None
        for idx, candidate in enumerate(self._free_cpu):
            if candidate.dtype != dtype or candidate.numel() < numel:
                continue
            if best_numel is None or candidate.numel() < best_numel:
                best_idx = idx
                best_numel = candidate.numel()
        reused = best_idx is not None
        if best_idx is None:
            slot = torch.empty(numel, dtype=dtype, pin_memory=True)
        else:
            slot = self._free_cpu.pop(best_idx)
            self._slot_reuse_count += 1
        self._cpu[key] = slot
        active_pinned_bytes = sum(
            t.numel() * t.element_size() for t in self._cpu.values()
        )
        self._peak_active_pinned_bytes = max(
            self._peak_active_pinned_bytes, active_pinned_bytes
        )
        pinned_bytes = sum(
            t.numel() * t.element_size() for t in (*self._cpu.values(), *self._free_cpu)
        )
        self._peak_pinned_bytes = max(self._peak_pinned_bytes, pinned_bytes)
        self._trace_lifecycle(
            "cpu_slot_acquire",
            key=key,
            cpu_data_ptr=int(slot.data_ptr()),
            requested_numel=numel,
            capacity_numel=slot.numel(),
            dtype=str(dtype),
            reused=reused,
            active_pinned_bytes=active_pinned_bytes,
            total_pinned_bytes=pinned_bytes,
        )
        return slot

    def offload_async(self, key: str, gpu_t: torch.Tensor) -> OffloadHandle:
        """Async D2H copy `gpu_t` -> its pinned CPU slot, freeing the GPU tensor
        for reuse once the copy completes. Returns a handle to store in place of
        the GPU tensor."""
        numel = gpu_t.numel()
        if key in self._cpu or key in self._handles:
            raise RuntimeError(f"Duplicate live activation offload key: {key}")
        slot = self._acquire_cpu_slot(key, numel, gpu_t.dtype)
        src = gpu_t.reshape(-1)  # attention out is contiguous; reshape is a view
        self._trace_lifecycle(
            "offload_begin",
            key=key,
            gpu_data_ptr=int(gpu_t.data_ptr()),
            cpu_data_ptr=int(slot.data_ptr()),
            nbytes=numel * gpu_t.element_size(),
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
        )
        weakref.finalize(
            gpu_t,
            self._trace_offload_source_finalize,
            key,
            int(gpu_t.data_ptr()),
            numel * gpu_t.element_size(),
        )
        if not self._async_d2h:
            slot[:numel].copy_(src)  # synchronous D2H on the current stream
            self._trace_lifecycle(
                "offload_complete",
                key=key,
                cpu_data_ptr=int(slot.data_ptr()),
                nbytes=numel * gpu_t.element_size(),
            )
        else:
            self._offload_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._offload_stream):
                # WAR guard: the persistent slot is reused across steps. Wait for the
                # previous step's prefetch H2D READ of this slot (on _prefetch_stream)
                # to finish before overwriting it. pinned-CPU reuse is NOT protected
                # by record_stream, so this event is the only thing ordering them.
                prev_read = self._prefetch_read_ev.get(key)
                if prev_read is not None:
                    self._offload_stream.wait_event(prev_read)
                slot[:numel].copy_(src, non_blocking=True)
            # keep gpu_t alive until the D2H (on offload_stream) is done, then reuse
            gpu_t.record_stream(self._offload_stream)
            ev = self._offload_ev.get(key) or torch.cuda.Event()
            ev.record(self._offload_stream)
            self._offload_ev[key] = ev
            self._trace_lifecycle(
                "offload_enqueued",
                key=key,
                cpu_data_ptr=int(slot.data_ptr()),
                nbytes=numel * gpu_t.element_size(),
            )
        forward_id = self._active_forward_id
        if forward_id is None:
            raise RuntimeError(f"Offload key {key} has no active forward_id")
        handle = OffloadHandle(
            handle_id=self._next_handle_id,
            forward_id=forward_id,
            key=key,
            numel=numel,
            shape=tuple(gpu_t.shape),
            dtype=gpu_t.dtype,
        )
        self._next_handle_id += 1
        self._handles[key] = handle
        nbytes = numel * gpu_t.element_size()
        kind = key.split(":", 1)[0]
        self._offloaded_bytes_total += nbytes
        self._offloaded_bytes_by_kind[kind] = (
            self._offloaded_bytes_by_kind.get(kind, 0) + nbytes
        )
        self._handles_by_forward[forward_id] = (
            self._handles_by_forward.get(forward_id, 0) + 1
        )
        self._peak_live_handles = max(self._peak_live_handles, len(self._handles))
        return handle

    # ---- backward ---------------------------------------------------------- #
    def prefetch_layer(self, name: str, i: int):
        """Prefetch layer i's offloaded tensor by (name, index). Used by the
        per-block backward pre-hook. No-op if that slot wasn't offloaded."""
        # A module-level backward hook cannot identify which of several
        # outstanding segment forwards it belongs to.  Multi-forward-safe
        # restoration is handle-driven in wait_and_get(); keep this legacy API
        # as a correctness-preserving no-op.
        return

    def prefetch_resid_layer(self, block_idx: int):
        """Prefetch ALL offloaded resid (img/txt) tensors of a block.

        resid tensors are keyed ``resid:{block_idx}:{i}`` for i=0,1,... (assigned
        in forward order by ``resid_offload_hooks``); iterate until the key is
        absent. Each becomes a prefetch-hit in ``wait_and_get`` during the block's
        checkpoint recompute. No-op if the block offloaded no resid tensors."""
        return

    def prefetch_async(self, handle: OffloadHandle):
        """Async H2D copy CPU slot -> a fresh GPU tensor (one layer ahead).

        A fresh allocator tensor (not a manually-rotated pool) is used so the
        caching allocator handles cross-stream reuse safely via record_stream —
        avoids the race where a pooled buffer is overwritten by the next
        prefetch while the compute stream is still reading it."""
        if handle.key in self._prefetched:
            return
        if not self._async_h2d:
            return  # sync mode: wait_and_get does an on-demand H2D
        slot = self._cpu[handle.key]
        cur = torch.cuda.current_stream()
        # Allocate on the COMPUTE stream (NOT inside the _prefetch_stream context):
        # the caching allocator tags the block's creation stream as compute, so its
        # consumer read (out_t.copy_(cached_out) on the compute stream) is safe.
        buf = torch.empty(handle.numel, dtype=handle.dtype, device=self.device)
        off_ev = self._offload_ev.get(handle.key)
        with torch.cuda.stream(self._prefetch_stream):
            if off_ev is not None:
                self._prefetch_stream.wait_event(off_ev)  # forward D2H landed in slot
            # PRIMARY FIX: wait for the compute stream so the buf allocation is done
            # AND the compute-stream op that freed this block has completed — else the
            # H2D writes into memory the compute stream is still using -> NaN grads.
            self._prefetch_stream.wait_stream(cur)
            buf.copy_(slot[: handle.numel], non_blocking=True)
        buf.record_stream(self._prefetch_stream)  # writer stream keeps buf alive
        ev = torch.cuda.Event()
        ev.record(self._prefetch_stream)
        self._prefetched[handle.key] = (buf, ev)
        # `ev` marks completion of the slot READ; reuse it as the WAR guard event.
        self._prefetch_read_ev[handle.key] = ev

    def wait_and_get(self, handle: OffloadHandle) -> torch.Tensor:
        """Return the prefetched GPU tensor (waiting on its H2D). Falls back to a
        synchronous H2D if the key was not prefetched ahead of time."""
        entry = self._prefetched.pop(handle.key, None)
        self._trace_lifecycle(
            "restore_begin",
            key=handle.key,
            cpu_data_ptr=int(self._cpu[handle.key].data_ptr()),
            nbytes=handle.numel * handle.dtype.itemsize,
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
        )
        if entry is None:
            # Handle-driven fallback.  With async H2D enabled, issue the copy on
            # the dedicated stream and wait on its event.  This is invocation
            # safe; later work may add look-ahead without returning to
            # block-index-only lookup.
            slot = self._cpu[handle.key]
            off_ev = self._offload_ev.get(handle.key)
            if self._async_h2d:
                self.prefetch_async(handle)
                entry = self._prefetched.pop(handle.key)
                buf, ev = entry
                torch.cuda.current_stream().wait_event(ev)
            else:
                if off_ev is not None:
                    torch.cuda.current_stream().wait_event(off_ev)
                buf = torch.empty(handle.numel, dtype=handle.dtype, device=self.device)
                buf.copy_(slot[: handle.numel])
                self._prefetch_read_ev[handle.key] = (
                    torch.cuda.current_stream().record_event()
                )
                self._sync_fallback_count += 1
            self._consumed_handles.add(handle.key)
            self._restored_bytes_total += handle.numel * handle.dtype.itemsize
            restored = buf.view(handle.shape)
            self._trace_lifecycle(
                "restore_complete",
                key=handle.key,
                gpu_data_ptr=int(restored.data_ptr()),
                nbytes=handle.numel * handle.dtype.itemsize,
                allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
            )
            return self._track_restored_tensor(restored, handle)
        buf, ev = entry
        torch.cuda.current_stream().wait_event(ev)
        self._consumed_handles.add(handle.key)
        self._restored_bytes_total += handle.numel * handle.dtype.itemsize
        restored = buf.view(handle.shape)
        self._trace_lifecycle(
            "restore_complete",
            key=handle.key,
            gpu_data_ptr=int(restored.data_ptr()),
            nbytes=handle.numel * handle.dtype.itemsize,
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
        )
        return self._track_restored_tensor(restored, handle)

    # ---- block-input (img/txt) offload via saved_tensors_hooks -------------- #
    def resid_offload_hooks(self, block_idx: int, forward_id: Optional[int] = None):
        """Context manager: offload the checkpoint block INPUTS (img/txt) that are
        saved via ``save_for_backward`` to pinned CPU, and bring them back on
        unpack (backward recompute).

        This is Phase 2 of activation offload. It must wrap ONLY the
        ``torch.utils.checkpoint.checkpoint(...)`` call of a single block, with
        img/txt passed POSITIONALLY so torch's non-reentrant checkpoint saves them
        through ``_NoopSaveInputs.save_for_backward`` (hookable here). Tensors
        passed as kwargs (vec/freqs_cis/text_mask — small and shared across all
        blocks) go through the recompute closure instead and are left on GPU.

        Only the block's own forward-compute activations are NOT touched: those
        are saved under checkpoint's inner ``_checkpoint_hook`` (which shadows
        these hooks during the forward), and are recomputed in backward anyway.

        No-op passthrough when disabled. D2H offload and H2D restore follow the
        async toggles (see OFFLOAD_ASYNC); when async prefetch hooks are registered
        for resid, ``wait_and_get`` gets a prefetch-hit instead of the sync H2D."""
        mgr = self
        ctr = {"i": 0}
        if forward_id is None:
            forward_id = self.current_forward_id()

        def pack(t):
            if (
                (forward_id is None)
                or (not mgr.enabled)
                or (not t.is_cuda)
                or t.numel() * t.element_size() < mgr._resid_min_bytes
            ):
                return ("keep", t)
            key = mgr.make_resid_key(forward_id, block_idx, ctr["i"])
            ctr["i"] += 1
            return ("off", mgr.offload_async(key, t))

        def unpack(p):
            kind, v = p
            return mgr.wait_and_get(v) if kind == "off" else v

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def finish_step(self, assert_all_consumed: bool = True) -> dict:
        """Finalize one backward/optimizer step and recycle pinned leases."""
        live = set(self._handles)
        unconsumed = sorted(live - self._consumed_handles)
        stats = self.stats()
        self._trace_lifecycle(
            "finish_step_begin",
            active_cpu_slots=len(self._cpu),
            free_cpu_slots=len(self._free_cpu),
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
            **stats,
        )
        if assert_all_consumed and unconsumed:
            raise RuntimeError(
                "Activation offload step ended with unconsumed handles: "
                f"count={len(unconsumed)} examples={unconsumed[:8]}"
            )
        # Correctness-first pool boundary: async H2D may have completed only in
        # CUDA stream order when backward returns.  Synchronize before making
        # its pinned source slots writable by the next step.  Intra-step D2H/H2D
        # remains asynchronous; removing this boundary requires carrying each
        # slot's read event into the free pool.
        if self.enabled and (self._async_d2h or self._async_h2d):
            torch.cuda.synchronize(self.device)
        self._free_cpu.extend(self._cpu.values())
        self._cpu.clear()
        self._handles.clear()
        self._consumed_handles.clear()
        self._offload_ev.clear()
        self._prefetched.clear()
        self._prefetch_read_ev.clear()
        self._fwd_counters.clear()
        self._active_forward_id = None
        self._active_forward_grad_enabled = False
        self._peak_active_pinned_bytes = 0
        self._peak_live_handles = 0
        self._offloaded_bytes_total = 0
        self._restored_bytes_total = 0
        self._offloaded_bytes_by_kind.clear()
        self._handles_by_forward.clear()
        self._trace_lifecycle(
            "finish_step_complete",
            free_cpu_slots=len(self._free_cpu),
            allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
        )
        return stats

    def stats(self) -> dict:
        active_bytes = sum(t.numel() * t.element_size() for t in self._cpu.values())
        free_bytes = sum(t.numel() * t.element_size() for t in self._free_cpu)
        return {
            "live_handles": len(self._handles),
            "consumed_handles": len(self._consumed_handles),
            "active_pinned_bytes": active_bytes,
            "pooled_pinned_bytes": free_bytes,
            "peak_pinned_bytes": self._peak_pinned_bytes,
            "peak_active_pinned_bytes": self._peak_active_pinned_bytes,
            "peak_live_handles": self._peak_live_handles,
            "slot_reuse_count": self._slot_reuse_count,
            "sync_fallback_count": self._sync_fallback_count,
            "offloaded_bytes_total": self._offloaded_bytes_total,
            "restored_bytes_total": self._restored_bytes_total,
            "offloaded_bytes_by_kind": dict(self._offloaded_bytes_by_kind),
            "handles_by_forward": dict(self._handles_by_forward),
        }


# --------------------------------------------------------------------------- #
# process-global handle (the SAC dispatch modes + backward hooks are created in
# places that don't easily receive the manager by argument; train.py sets it
# once after building the manager). None => offloading disabled.
# --------------------------------------------------------------------------- #
_MANAGER: Optional[ActivationOffloadManager] = None


def set_offload_manager(m: Optional[ActivationOffloadManager]):
    global _MANAGER
    _MANAGER = m


def get_offload_manager() -> Optional[ActivationOffloadManager]:
    return _MANAGER


def register_prefetch_hooks(
    blocks, name: str = "attn_out", prefetch_resid: bool = True
):
    """Register a backward pre-hook on each block that prefetches its offloaded
    activations one layer ahead (backward runs blocks in reverse, so block i's
    hook readies block i and kicks off block i-1). No-op when disabled.

    Prefetches both the SAC attention `out` (single ``{name}:{i}`` slot) and, when
    ``prefetch_resid`` is set, the block's img/txt residual inputs (multiple
    ``resid:{i}:{k}`` slots). `blocks` should be the (checkpoint-wrapped) modules
    whose backward includes the recompute that consumes the offloaded tensors.
    """
    mgr = get_offload_manager()
    if mgr is None or not mgr.enabled:
        return

    def make_hook(i):
        def hook(module, grad_output):
            mgr.prefetch_layer(name, i)  # ensure current layer ready (no-op if done)
            if prefetch_resid:
                mgr.prefetch_resid_layer(i)
            if i - 1 >= 0:
                mgr.prefetch_layer(name, i - 1)  # one layer ahead (reverse order)
                if prefetch_resid:
                    mgr.prefetch_resid_layer(i - 1)

        return hook

    for i, b in enumerate(blocks):
        b.register_full_backward_pre_hook(make_hook(i))
