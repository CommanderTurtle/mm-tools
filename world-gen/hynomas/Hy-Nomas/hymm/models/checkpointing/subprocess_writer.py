"""Process-isolated asynchronous checkpoint shard writer.

Training stages CPU tensors in shared memory, then a forked child performs the
actual local-disk write and optional copy to the final shared filesystem.  This
keeps filesystem page-cache and copy traffic out of the training process.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


SHARD_MANIFEST_FILE = "shard_manifest.json"


def _share_tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.is_cuda:
            tensor = tensor.to("cpu", copy=True)
        elif not tensor.is_shared():
            tensor = tensor.clone()
        try:
            tensor.share_memory_()
        except RuntimeError:
            pass
        return tensor
    if isinstance(value, dict):
        return {key: _share_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_share_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_share_tensors(item) for item in value)
    return value


def _worker_loop(task_queue, done_queue, rank: int, log_path: str):
    import torch

    if log_path:
        try:
            stream = open(log_path, "a", buffering=1)
            sys.stdout = stream
            sys.stderr = stream
        except OSError:
            pass
    print(f"[ckpt_writer rank={rank}] started pid={os.getpid()}", flush=True)
    while True:
        try:
            task = task_queue.get()
        except (EOFError, KeyboardInterrupt):
            return
        if task is None:
            return
        payload, local_path, remote_path = task
        try:
            local = Path(local_path)
            remote = Path(remote_path) if remote_path else local
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_name(local.name + ".tmp")
            started = time.perf_counter()
            torch.save(payload, temporary)
            os.replace(temporary, local)
            save_seconds = time.perf_counter() - started
            copy_seconds = 0.0
            if local.resolve() != remote.resolve():
                remote.parent.mkdir(parents=True, exist_ok=True)
                remote_temporary = remote.with_name(remote.name + ".tmp")
                copy_started = time.perf_counter()
                shutil.copyfile(local, remote_temporary)
                os.replace(remote_temporary, remote)
                local.unlink(missing_ok=True)
                copy_seconds = time.perf_counter() - copy_started
            done_path = remote.with_name(remote.name + ".done")
            done_temporary = done_path.with_name(done_path.name + ".tmp")
            done_temporary.write_text("complete\n", encoding="utf-8")
            os.replace(done_temporary, done_path)
            done_queue.put((str(remote), None))
            print(
                f"[ckpt_writer rank={rank}] wrote {remote} "
                f"save={save_seconds:.2f}s copy={copy_seconds:.2f}s",
                flush=True,
            )
        except Exception as exc:
            done_queue.put((None, f"{type(exc).__name__}: {exc}"))
            print(traceback.format_exc(), flush=True)


class SubprocessCheckpointWriter:
    """One lazily-created checkpoint writer subprocess per participating rank."""

    _instance = None

    @classmethod
    def get_or_create(cls, rank: int, log_dir: str | None = None):
        if cls._instance is None:
            cls._instance = cls(rank, log_dir)
        return cls._instance

    @classmethod
    def maybe_instance(cls):
        return cls._instance

    @classmethod
    def shutdown_instance(cls, timeout: float = 60.0):
        instance = cls._instance
        if instance is None:
            return
        try:
            instance.shutdown(timeout=timeout)
        finally:
            cls._instance = None

    def __init__(self, rank: int, log_dir: str | None):
        context = mp.get_context("fork")
        self.rank = int(rank)
        self.task_queue = context.Queue()
        self.done_queue = context.Queue()
        self.inflight = 0
        self.errors: list[str] = []
        directory = Path(
            log_dir or os.getenv("HY_CHECKPOINT_WRITER_LOG_DIR", "/tmp/hy_ckpt_writer")
        )
        directory.mkdir(parents=True, exist_ok=True)
        self.process = context.Process(
            target=_worker_loop,
            args=(
                self.task_queue,
                self.done_queue,
                self.rank,
                str(directory / f"rank_{rank:04d}.log"),
            ),
            daemon=True,
            name=f"checkpoint_writer_rank_{rank}",
        )
        self.process.start()

    def submit(self, payload, local_path: str, remote_path: str | None = None):
        self.task_queue.put((_share_tensors(payload), local_path, remote_path))
        self.inflight += 1

    def _record(self, result):
        _path, error = result
        self.inflight -= 1
        if error:
            self.errors.append(error)

    def _raise_if_failed(self):
        if self.errors:
            errors = "; ".join(self.errors)
            self.errors.clear()
            raise RuntimeError(f"checkpoint writer failed: {errors}")
        if self.inflight and not self.process.is_alive():
            raise RuntimeError(
                f"checkpoint writer rank {self.rank} exited with "
                f"code {self.process.exitcode} while work was pending"
            )

    def poll(self) -> bool:
        import queue

        while True:
            try:
                self._record(self.done_queue.get_nowait())
            except queue.Empty:
                break
        self._raise_if_failed()
        return self.inflight == 0

    def wait_all(self, timeout: float = 3600.0) -> bool:
        import queue

        deadline = time.monotonic() + timeout
        while self.inflight:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for {self.inflight} checkpoint shard(s)"
                )
            try:
                self._record(self.done_queue.get(timeout=remaining))
            except queue.Empty as exc:
                raise TimeoutError("checkpoint writer timed out") from exc
        self._raise_if_failed()
        return True

    def shutdown(self, timeout: float = 60.0):
        """Drain writes and close the child plus queue feeder threads."""
        self.wait_all(timeout=timeout)
        if self.process.is_alive():
            self.task_queue.put(None)
            self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        if self.process.is_alive():
            raise RuntimeError(
                f"checkpoint writer rank {self.rank} did not exit after shutdown"
            )
        for queue in (self.task_queue, self.done_queue):
            queue.close()
            queue.join_thread()


def extract_local_state(value):
    """Recursively replace DTensors with their rank-local tensor payloads."""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value._local_tensor if hasattr(value, "_local_tensor") else value
        return tensor.detach()
    if isinstance(value, dict):
        return {key: extract_local_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [extract_local_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(extract_local_state(item) for item in value)
    return value


def restore_local_state(live, loaded) -> int:
    """Copy a nested local checkpoint payload into live DTensor/tensor leaves."""
    import torch

    restored = 0
    if isinstance(live, dict) and isinstance(loaded, dict):
        for key, source in loaded.items():
            if key not in live:
                continue
            target = live[key]
            if isinstance(target, dict) and isinstance(source, dict):
                restored += restore_local_state(target, source)
            elif isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor):
                local = (
                    target._local_tensor if hasattr(target, "_local_tensor") else target
                )
                local.copy_(source.to(device=local.device, dtype=local.dtype))
                restored += 1
            elif not isinstance(target, torch.Tensor) and not isinstance(
                source, torch.Tensor
            ):
                live[key] = source
                restored += 1
    return restored
