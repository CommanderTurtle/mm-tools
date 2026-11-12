"""Lightweight node telemetry for the multi-user Demo control plane."""

from __future__ import annotations

import csv
import io
import subprocess
import threading
import time
from typing import Any, Callable, Dict


NVIDIA_QUERY = (
    "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,power.limit,clocks.current.sm,pstate"
)


class ResourceMonitor:
    """Cache nvidia-smi samples and anonymous browser-presence heartbeats."""

    def __init__(
        self,
        *,
        sample_interval: float = 2.0,
        presence_ttl: float = 30.0,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = (
            subprocess.run
        ),
    ) -> None:
        self.sample_interval = max(float(sample_interval), 0.5)
        self.presence_ttl = max(float(presence_ttl), 5.0)
        self.command_runner = command_runner
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._clients: Dict[str, float] = {}
        self._sample: Dict[str, Any] = {
            "available": False,
            "sampled_at": None,
            "gpus": [],
        }
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="demo-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval * 2 + 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.sample_interval)

    @staticmethod
    def _number(value: str) -> float:
        clean = value.strip()
        if clean in {"", "N/A", "[N/A]"}:
            return 0.0
        return float(clean)

    def sample_once(self) -> Dict[str, Any]:
        sampled_at = time.time()
        try:
            completed = self.command_runner(
                [
                    "nvidia-smi",
                    f"--query-gpu={NVIDIA_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(1.0, self.sample_interval),
            )
            rows = csv.reader(io.StringIO(completed.stdout))
            gpus = []
            for row in rows:
                if len(row) != 10:
                    continue
                (
                    index,
                    name,
                    gpu_util,
                    memory_util,
                    memory_used,
                    memory_total,
                    power_draw,
                    power_limit,
                    sm_clock,
                    pstate,
                ) = row
                limit = self._number(power_limit)
                draw = self._number(power_draw)
                gpus.append(
                    {
                        "index": int(index.strip()),
                        "name": name.strip(),
                        "gpu_utilization_percent": self._number(gpu_util),
                        "memory_utilization_percent": self._number(memory_util),
                        "memory_used_mib": self._number(memory_used),
                        "memory_total_mib": self._number(memory_total),
                        "power_draw_watts": draw,
                        "power_limit_watts": limit,
                        "power_ratio": draw / limit if limit > 0 else 0.0,
                        "sm_clock_mhz": self._number(sm_clock),
                        "pstate": pstate.strip(),
                    }
                )
            sample = {
                "available": bool(gpus),
                "sampled_at": sampled_at,
                "gpus": gpus,
            }
        except Exception as exc:  # noqa: BLE001
            sample = {
                "available": False,
                "sampled_at": sampled_at,
                "gpus": [],
                "error": str(exc),
            }
        with self._lock:
            self._sample = sample
        return sample

    def touch(self, client_id: str) -> int:
        clean = client_id.strip()[:128]
        if not clean:
            return self.online_count()
        now = time.time()
        with self._lock:
            self._prune_clients(now)
            self._clients[clean] = now
            return len(self._clients)

    def remove(self, client_id: str) -> int:
        with self._lock:
            self._clients.pop(client_id.strip()[:128], None)
            self._prune_clients(time.time())
            return len(self._clients)

    def _prune_clients(self, now: float) -> None:
        cutoff = now - self.presence_ttl
        self._clients = {
            client_id: heartbeat_at
            for client_id, heartbeat_at in self._clients.items()
            if heartbeat_at >= cutoff
        }

    def online_count(self) -> int:
        with self._lock:
            self._prune_clients(time.time())
            return len(self._clients)

    def snapshot(self, *, running_jobs: int = 0) -> Dict[str, Any]:
        with self._lock:
            sample = {
                **self._sample,
                "gpus": [dict(gpu) for gpu in self._sample.get("gpus", [])],
            }
            self._prune_clients(time.time())
            online_users = len(self._clients)
        gpus = sample["gpus"]
        total_power = sum(gpu["power_draw_watts"] for gpu in gpus)
        total_power_limit = sum(gpu["power_limit_watts"] for gpu in gpus)
        avg_gpu_util = (
            sum(gpu["gpu_utilization_percent"] for gpu in gpus) / len(gpus)
            if gpus
            else 0.0
        )
        avg_power_ratio = (
            total_power / total_power_limit if total_power_limit > 0 else 0.0
        )
        total_memory_used = sum(gpu["memory_used_mib"] for gpu in gpus)
        total_memory = sum(gpu["memory_total_mib"] for gpu in gpus)
        max_memory_ratio = max(
            (
                gpu["memory_used_mib"] / gpu["memory_total_mib"]
                for gpu in gpus
                if gpu["memory_total_mib"] > 0
            ),
            default=0.0,
        )
        # nvidia-smi utilization alone can remain high because of sampling,
        # communication, or a synthetic occupancy process. The UI therefore
        # calls the node actively computing only when a Demo job is running and
        # either power or SM activity is materially above idle.
        compute_active = bool(
            running_jobs > 0 and avg_power_ratio >= 0.25 and avg_gpu_util >= 5.0
        )
        return {
            **sample,
            "online_users": online_users,
            "presence_ttl_seconds": self.presence_ttl,
            "total_power_draw_watts": total_power,
            "total_power_limit_watts": total_power_limit,
            "average_power_ratio": avg_power_ratio,
            "average_gpu_utilization_percent": avg_gpu_util,
            "memory_used_mib": total_memory_used,
            "memory_total_mib": total_memory,
            "max_gpu_memory_ratio": max_memory_ratio,
            "memory_pressure": (
                "critical"
                if max_memory_ratio >= 0.92
                else ("warning" if max_memory_ratio >= 0.80 else "normal")
            ),
            "compute_active": compute_active,
            "compute_state": (
                "active"
                if compute_active
                else ("scheduled" if running_jobs > 0 else "idle")
            ),
        }
