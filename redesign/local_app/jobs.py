from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "outputs" / "workbench"
ALLOWED_INPUTS = {".png", ".jpg", ".jpeg", ".webp"}
QWEN_ROOT = ROOT.parent / "models" / "qwen"
DEFAULT_NATIVE_FP8 = QWEN_ROOT / "T5B--qwen-image-layered-fp8" / "qwen_image_layered_fp8_e4m3fn.safetensors"
DEFAULT_NATIVE_COMPONENTS = QWEN_ROOT / "diffusers--hfstaff--Qwen-Image-Layered-modular"
DEFAULT_NATIVE_TEXT_ENCODER = QWEN_ROOT / "suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4" / "text_encoder"
DEFAULT_COMFY_SOURCES = {
    "REDESIGN_QWEN_INT8_PATH": QWEN_ROOT / "appmana--diffusion--qwen-image-layered-int8convrot" / "qwen_image_layered_int8convrot.safetensors",
    "REDESIGN_QWEN_TEXT_ENCODER_PATH": QWEN_ROOT / "comfy-org--text--qwen_2.5_vl_7b_fp8_scaled.safetensors" / "split_files" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "REDESIGN_QWEN_VAE_PATH": QWEN_ROOT / "comfy-org--vae--qwen_image_layered_vae.safetensors" / "split_files" / "vae" / "qwen_image_layered_vae.safetensors",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def endpoint_is_local(value: str) -> bool:
    from ipaddress import ip_address

    host = urlparse(value).hostname
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ip_address(host)
        return address.is_private or address.is_loopback
    except ValueError:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }
            return bool(addresses) and all(
                ip_address(item).is_private or ip_address(item).is_loopback
                for item in addresses
            )
        except (OSError, ValueError):
            return False


@dataclass
class Job:
    id: str
    name: str
    input_path: str
    output_dir: str
    created_at: str
    command: list[str]
    status: str = "queued"
    finished_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=5000), repr=False)
    log_cursor: int = field(default=0, repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

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
            artifacts.append({
                "path": path.relative_to(root).as_posix(),
                "name": path.name,
                "kind": "image" if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"} else path.suffix[1:],
                "bytes": path.stat().st_size,
            })
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
        controller_url: str,
        controller_model: str,
        qwen_backend: str,
        comfy_url: str,
        qwen_model: str,
        qwen_gpus: str,
        qwen_pair_size: int,
        tool_gpus: str,
        workers: int,
        cpu_offload: bool,
        max_bytes: int,
    ) -> Job:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_INPUTS:
            raise ValueError("Input must be PNG, JPEG, or WebP.")
        if not endpoint_is_local(controller_url):
            raise ValueError("Controller URL must be loopback, a private-LAN IP, or a local hostname.")
        if not controller_model.strip():
            raise ValueError("Controller model name is required.")
        backend = qwen_backend.strip().lower()
        if backend not in {"native", "native-fp8", "fp8", "comfy", "comfyui", "diffusers"}:
            raise ValueError("Qwen backend must be native-fp8, comfyui, or diffusers.")
        qwen_path: Path | None = None
        if backend in {"comfy", "comfyui"}:
            if not endpoint_is_local(comfy_url):
                raise ValueError("ComfyUI URL must be loopback, a private-LAN IP, or a local hostname.")
            source_variables = (
                "REDESIGN_QWEN_INT8_PATH",
                "REDESIGN_QWEN_TEXT_ENCODER_PATH",
                "REDESIGN_QWEN_VAE_PATH",
            )
            missing = []
            for variable in source_variables:
                value = os.path.expandvars(os.environ.get(variable, str(DEFAULT_COMFY_SOURCES[variable])))
                if not value or not Path(value).expanduser().is_file():
                    missing.append(variable)
            if missing:
                raise ValueError("Missing local Comfy model files: " + ", ".join(missing))
        elif backend == "diffusers":
            if not qwen_model.strip():
                raise ValueError("Set the complete Diffusers Qwen-Image-Layered checkpoint path.")
            qwen_path = Path(qwen_model).expanduser().resolve()
            if not qwen_path.is_dir() or not (qwen_path / "model_index.json").is_file():
                raise ValueError(f"Complete Diffusers checkpoint not found: {qwen_path}")
        else:
            qwen_path = Path(qwen_model or os.environ.get("REDESIGN_QWEN_MODEL", str(DEFAULT_NATIVE_FP8))).expanduser().resolve()
            components = Path(os.environ.get("REDESIGN_QWEN_COMPONENTS", str(DEFAULT_NATIVE_COMPONENTS))).expanduser().resolve()
            text_encoder = Path(os.environ.get("REDESIGN_QWEN_TEXT_ENCODER_COMPONENTS", str(DEFAULT_NATIVE_TEXT_ENCODER))).expanduser().resolve()
            missing = []
            if not qwen_path.is_file():
                missing.append(str(qwen_path))
            if not (components / "modular_model_index.json").is_file():
                missing.append(str(components))
            if not (text_encoder / "config.json").is_file():
                missing.append(str(text_encoder))
            if missing:
                raise ValueError("Missing native FP8 components: " + ", ".join(missing))
        python = ROOT / ".venv" / "bin" / "python"
        if not python.is_file():
            raise ValueError("The isolated runtime is missing. Run ./uvsetup.sh first.")
        if any(job.status in {"queued", "running"} for job in self.jobs.values()):
            raise ValueError("A decomposition is already running on this workstation.")

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

        command = [
            str(python), "-m", "ReDesign.run_single_image",
            "--image", str(target),
            "--output_dir", str(output_dir),
            "--qwen_gpus", qwen_gpus,
            "--qwen_pair_size", str(qwen_pair_size),
            "--tool_gpus", tool_gpus,
            "--workers", str(workers),
        ]
        job = Job(job_id, Path(filename).stem, str(target), str(output_dir), utc_now(), command)
        with self.lock:
            self.jobs[job_id] = job

        environment = os.environ.copy()
        environment.update({
            "OPENAI_BASE_URL": controller_url.rstrip("/"),
            "OPENAI_API_KEY": environment.get("OPENAI_API_KEY", "local-vllm"),
            "VLM_MODEL": controller_model,
            "REDESIGN_QWEN_BACKEND": (
                "comfyui" if backend in {"comfy", "comfyui"}
                else "diffusers" if backend == "diffusers"
                else "native-fp8"
            ),
            "REDESIGN_COMFY_URL": comfy_url.rstrip("/"),
            "REDESIGN_QWEN_MODEL": str(qwen_path) if qwen_path else qwen_model,
            "REDESIGN_ALLOW_CPU_OFFLOAD": "1" if cpu_offload else "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        })
        thread = threading.Thread(target=self._run, args=(job, environment), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, environment: dict[str, str]) -> None:
        log_path = OUTPUT_ROOT / job.id / "workbench.log"
        try:
            job.status = "running"
            process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            job.process = process
            with log_path.open("w", encoding="utf-8") as log:
                assert process.stdout is not None
                for line in process.stdout:
                    clean = line.rstrip("\n")
                    job.append_log(clean)
                    log.write(line)
                    log.flush()
            job.returncode = process.wait()
            job.status = "complete" if job.returncode == 0 else "failed"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.append_log(f"Workbench error: {exc}")
        finally:
            job.process = None
            job.finished_at = utc_now()
            metadata = {key: value for key, value in job.public().items() if key != "artifacts"}
            (OUTPUT_ROOT / job.id / "job.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def stop(self, job_id: str) -> Job:
        job = self.get(job_id)
        process = job.process
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            job.status = "stopped"
            job.append_log("Stopped by user.")
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError("Unknown or expired job.") from exc


manager = JobManager()
