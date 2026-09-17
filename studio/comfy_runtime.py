from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from studio.runtime import JobCancelled, StudioContext


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class ComfyRuntime:
    """Job-scoped, loopback-only Comfy runtime for native mm-tools graphs.

    A private subprocess gives quantized Comfy models their intended loader and
    scheduler without exposing Comfy's editor or API to the LAN. The outer
    Studio remains the sole product/API surface and owns queueing/cancellation.
    """

    def __init__(
        self,
        *,
        python: Path,
        comfy_root: Path,
        runtime_root: Path,
        context: StudioContext,
        extra_env: dict[str, str] | None = None,
        custom_node_allowlist: list[str] | None = None,
    ) -> None:
        self.python = python
        self.comfy_root = comfy_root
        self.runtime_root = runtime_root
        self.context = context
        self.port = _free_port()
        self.client_id = f"mm-tools-{uuid.uuid4().hex}"
        self.input_dir = runtime_root / "comfy-input" / context.job_id
        self.output_dir = runtime_root / "comfy-output" / context.job_id
        self.user_dir = runtime_root / "comfy-user" / context.job_id
        self.log_path = runtime_root / "comfy-logs" / f"{context.job_id}.log"
        for directory in (self.input_dir, self.output_dir, self.user_dir, self.log_path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        if extra_env:
            env.update(extra_env)
        command = [
            str(python), str(comfy_root / "main.py"),
            "--listen", "127.0.0.1", "--port", str(self.port),
            "--input-directory", str(self.input_dir),
            "--output-directory", str(self.output_dir),
            "--user-directory", str(self.user_dir),
            "--disable-auto-launch", "--disable-metadata", "--disable-api-nodes",
            "--disable-all-custom-nodes", "--gpu-only",
        ]
        # Comfy's whitelist is folder-name based. Keeping the global disable
        # flag in place means a runtime-generated or user-installed node can
        # never execute merely because it exists in custom_nodes/. Individual
        # mm-tools products opt in only their audited, repo-pinned node pack.
        allowlist = [str(name) for name in (custom_node_allowlist or []) if str(name).strip()]
        if allowlist:
            command.extend(["--whitelist-custom-nodes", *allowlist])
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=comfy_root,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log_offset = 0
        self._wait_ready()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _request(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            self.context.check_cancelled()
            if self.process.poll() is not None:
                self._flush_log()
                raise RuntimeError(f"The private Comfy runtime exited during startup ({self.process.returncode}).")
            try:
                self._request("/system_stats", timeout=2)
                self.context.log(f"Private GPU-only Comfy runtime ready on ephemeral loopback port {self.port}.")
                return
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(0.5)
        self._flush_log()
        raise TimeoutError("The private Comfy runtime did not become ready within five minutes.")

    def add_input(self, source: Path, name: str) -> str:
        safe = "".join(character if character.isalnum() or character in "-_." else "-" for character in name)
        destination = self.input_dir / safe
        shutil.copy2(source, destination)
        return destination.name

    def execute(self, prompt: dict[str, Any], stage: str = "Running native Comfy graph") -> dict[str, Any]:
        result = self._request("/prompt", {"prompt": prompt, "client_id": self.client_id}, timeout=30)
        prompt_id = str(result.get("prompt_id", ""))
        if not prompt_id:
            raise RuntimeError(f"Comfy rejected the graph: {result}")
        self.context.log(f"Queued native Comfy prompt {prompt_id}.")
        started = time.monotonic()
        while True:
            if self.context.cancel_event.is_set():
                try:
                    self._request("/interrupt", {}, timeout=5)
                finally:
                    raise JobCancelled("Cancelled by user")
            if self.process.poll() is not None:
                self._flush_log()
                raise RuntimeError(f"The private Comfy runtime exited with code {self.process.returncode}.")
            history = self._request(f"/history/{prompt_id}", timeout=10)
            record = history.get(prompt_id)
            if record:
                self._flush_log()
                status = record.get("status") or {}
                if status.get("status_str") == "error" or not status.get("completed", True):
                    raise RuntimeError(f"Native Comfy graph failed: {status}")
                return record
            elapsed = time.monotonic() - started
            # Diffusion progress is deliberately asymptotic while Comfy owns the
            # inner scheduler; queue state remains exact and never jumps backward.
            progress = min(0.92, 0.12 + elapsed / (elapsed + 240.0) * 0.78)
            self.context.update(stage, progress)
            self._flush_log(limit=40)
            time.sleep(1.0)

    def output_files(self) -> list[Path]:
        return sorted(path for path in self.output_dir.rglob("*") if path.is_file())

    def _flush_log(self, limit: int = 200) -> None:
        self._log.flush()
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as reader:
                reader.seek(self._log_offset)
                lines = reader.readlines()
                self._log_offset = reader.tell()
        except OSError:
            return
        for line in lines[-limit:]:
            clean = line.rstrip()
            if clean:
                self.context.log(f"[comfy] {clean}")

    def close(self) -> None:
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            try:
                self._request("/interrupt", {}, timeout=2)
            except Exception:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if getattr(self, "_log", None) is not None:
            self._log.close()

    def cleanup(self) -> None:
        self.close()
        for directory in (self.input_dir, self.output_dir, self.user_dir):
            shutil.rmtree(directory, ignore_errors=True)

    def __enter__(self) -> "ComfyRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()
