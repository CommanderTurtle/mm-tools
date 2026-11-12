"""Inference stage timing and allocator peaks without retaining model tensors."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import time

import torch

_ACTIVE = ContextVar("inference_metrics", default=None)


def tensor_storage_bytes(module):
    """Scalar storage totals, deduplicated without retaining tensor/view references."""
    seen = set()
    result = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        if tensor.is_meta:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            result[key[0]] = result.get(key[0], 0) + storage.nbytes()
    return result


class InferenceMetrics:
    def __init__(self, device, *, detailed=False, jsonl=None):
        self.device = torch.device(device)
        self.detailed = detailed
        self.jsonl = Path(jsonl) if jsonl else None
        self.stack = []
        self.rows = []
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.peak_device_used = 0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _snapshot(self):
        if self.device.type != "cuda":
            return dict(allocated=0, reserved=0, peak_allocated=0, peak_reserved=0, device_used=0)
        free, total = torch.cuda.mem_get_info(self.device)
        return dict(
            allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            peak_allocated=torch.cuda.max_memory_allocated(self.device),
            peak_reserved=torch.cuda.max_memory_reserved(self.device),
            device_used=total-free,
        )

    def _boundary(self):
        self._sync()
        state = self._snapshot()
        self.peak_allocated = max(self.peak_allocated, state["peak_allocated"])
        self.peak_reserved = max(self.peak_reserved, state["peak_reserved"])
        self.peak_device_used = max(self.peak_device_used, state["device_used"])
        for row in self.stack:
            row["peak_allocated_bytes"] = max(row["peak_allocated_bytes"], state["peak_allocated"])
            row["peak_reserved_bytes"] = max(row["peak_reserved_bytes"], state["peak_reserved"])
            row["device_used_boundary_peak_bytes"] = max(row["device_used_boundary_peak_bytes"], state["device_used"])
        # Every peak is collected into all active scopes before a nested reset.
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return state

    def __enter__(self):
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self._boundary()
            self.elapsed = time.perf_counter() - self.started
            self.error = error_type.__name__ if error_type else None
            self._write(dict(stage="sample_end", wall_seconds=self.elapsed, error=self.error,
                             peak_allocated_bytes=self.peak_allocated, peak_reserved_bytes=self.peak_reserved))
        finally:
            _ACTIVE.reset(self.token)

    def _write(self, row):
        if self.jsonl:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            # Runtime logs should use a POSIX/local path; caller copies to COS after completion.
            with self.jsonl.open("a") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    @contextmanager
    def stage(self, name):
        if not self.detailed:
            yield
            return
        start = self._boundary()
        row = dict(stage=name, depth=len(self.stack), start_allocated_bytes=start["allocated"],
                   start_reserved_bytes=start["reserved"], peak_allocated_bytes=start["allocated"],
                   peak_reserved_bytes=start["reserved"], device_used_boundary_peak_bytes=start["device_used"])
        self.stack.append(row)
        began = time.perf_counter()
        error_name = None
        try:
            yield
        except BaseException as error:
            error_name = type(error).__name__
            raise
        finally:
            end = self._boundary()
            row.update(wall_seconds=time.perf_counter()-began, end_allocated_bytes=end["allocated"],
                       end_reserved_bytes=end["reserved"], error=error_name)
            row["allocated_delta_bytes"] = row["end_allocated_bytes"] - row["start_allocated_bytes"]
            self.stack.pop()
            self.rows.append(row)
            self._write(row)

    def summary(self):
        stages = {}
        for row in self.rows:
            item = stages.setdefault(row["stage"], dict(wall_seconds=0., calls=0,
                                    peak_allocated_bytes=0, peak_reserved_bytes=0))
            item["wall_seconds"] += row["wall_seconds"]
            item["calls"] += 1
            item["peak_allocated_bytes"] = max(item["peak_allocated_bytes"], row["peak_allocated_bytes"])
            item["peak_reserved_bytes"] = max(item["peak_reserved_bytes"], row["peak_reserved_bytes"])
        return dict(total_seconds=self.elapsed, peak_allocated_gib=self.peak_allocated/2**30,
                    peak_reserved_gib=self.peak_reserved/2**30,
                    device_used_boundary_peak_gib=self.peak_device_used/2**30,
                    timings=stages, error=self.error)


@contextmanager
def stage(name):
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
    else:
        with recorder.stage(name):
            yield


def profiled(name):
    def decorate(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            with stage(name):
                return method(*args, **kwargs)
        return wrapped
    return decorate
