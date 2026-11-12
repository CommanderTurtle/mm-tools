from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from .diffusers_runtime import JobCancelled, runtime


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "outputs" / "workbench"
ALLOWED_INPUTS = {".png", ".jpg", ".jpeg", ".webp"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    id: str
    name: str
    input_path: str
    output_dir: str
    created_at: str
    command: list[str]
    parameters: dict[str, int | float]
    status: str = "queued"
    finished_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=5000), repr=False)
    log_cursor: int = field(default=0, repr=False)

    def append_log(self, line: str) -> None:
        self.log_cursor += 1
        self.logs.append((self.log_cursor, line))

    def public(self) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "command": self.command,
            "parameters": self.parameters,
            "status": self.status,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
        }
        data["artifacts"] = artifacts_for(self)
        return data


def artifacts_for(job: Job) -> list[dict]:
    root = Path(job.output_dir)
    if not root.is_dir():
        return []
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".json", ".log", ".svg", ".zip"}
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed and not path.is_symlink():
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "name": path.name,
                    "kind": (
                        "image"
                        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
                        else path.suffix[1:]
                    ),
                    "bytes": path.stat().st_size,
                }
            )
    return artifacts


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        source: BinaryIO,
        filename: str,
        *,
        layers: int,
        steps: int,
        resolution: int,
        cfg: float,
        seed: int,
        max_bytes: int,
    ) -> Job:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_INPUTS:
            raise ValueError("Input must be PNG, JPEG, or WebP.")
        if not runtime.status()["ready"]:
            raise ValueError("Load the local Diffusers model first.")
        if any(job.status in {"queued", "running", "stopping"} for job in self.jobs.values()):
            raise ValueError("A decomposition is already running on this workstation.")
        layers = max(2, min(int(layers), 8))
        steps = max(1, min(int(steps), 100))
        if resolution not in {640, 1024}:
            raise ValueError("Resolution must be 640 or 1024.")
        cfg = max(0.0, min(float(cfg), 20.0))
        seed = max(0, min(int(seed), 2**63 - 1))

        job_id = uuid.uuid4().hex[:12]
        job_root = OUTPUT_ROOT / job_id
        input_dir = job_root / "input"
        output_dir = job_root / "run"
        input_dir.mkdir(parents=True)
        target = input_dir / f"design{suffix}"
        total = 0
        with target.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Image exceeds {max_bytes // (1024 * 1024)} MiB limit.")
                output.write(chunk)

        parameters: dict[str, int | float] = {
            "layers": layers,
            "steps": steps,
            "resolution": resolution,
            "cfg": cfg,
            "seed": seed,
        }
        command = ["diffusers", "QwenImageLayeredPipeline", str(runtime.model_path)]
        job = Job(
            job_id,
            Path(filename).stem,
            str(target),
            str(output_dir),
            utc_now(),
            command,
            parameters,
        )
        with self.lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), name=f"redesign-job-{job_id}", daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        log_path = OUTPUT_ROOT / job.id / "workbench.log"
        try:
            job.status = "running"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            def log(line: str) -> None:
                job.append_log(line)
                with log_path.open("a", encoding="utf-8") as output:
                    output.write(line + "\n")

            log("Starting direct local Qwen-Image-Layered Diffusers inference.")
            runtime.decompose(
                job_id=job.id,
                image_path=Path(job.input_path),
                output_dir=Path(job.output_dir),
                layers=int(job.parameters["layers"]),
                steps=int(job.parameters["steps"]),
                resolution=int(job.parameters["resolution"]),
                cfg=float(job.parameters["cfg"]),
                seed=int(job.parameters["seed"]),
                log=log,
            )
            job.returncode = 0
            job.status = "complete"
            log("Layer decomposition complete. The Diffusers model remains resident.")
        except JobCancelled as exc:
            job.status = "stopped"
            job.returncode = 130
            job.append_log(str(exc))
        except Exception as exc:
            job.status = "failed"
            job.returncode = 1
            job.error = str(exc)
            job.append_log(f"Diffusers error: {exc}")
        finally:
            job.finished_at = utc_now()
            metadata = {key: value for key, value in job.public().items() if key != "artifacts"}
            (OUTPUT_ROOT / job.id / "job.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

    def stop(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running" and runtime.cancel(job_id):
            job.status = "stopping"
            job.append_log("Stopping after the current diffusion step…")
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError("Unknown or expired job.") from exc


manager = JobManager()
