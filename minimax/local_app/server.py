from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
RUNTIME = ROOT / "runtime"


def _path_from_env(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


MODEL_ROOT = _path_from_env(
    "MINIMAX_MODEL_DIR", ROOT.parent / "models" / "Comfy-Org--Minimax-Music-3"
)
OUTPUT_ROOT = _path_from_env("MINIMAX_OUTPUT_DIR", ROOT / "outputs")
STATE_ROOT = ROOT / ".runtime"
ENGINE_HOST = os.getenv("MINIMAX_ENGINE_HOST", "127.0.0.1")
ENGINE_PORT = int(os.getenv("MINIMAX_ENGINE_PORT", "8264"))
ENGINE_URL = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
ENGINE_START_TIMEOUT = float(os.getenv("MINIMAX_ENGINE_START_TIMEOUT", "120"))
JOB_TIMEOUT = float(os.getenv("MINIMAX_JOB_TIMEOUT", "7200"))

UNET_NAME = "minimax_music3_dit_fp16.safetensors"
CLIP_NAME = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
VAE_NAME = "minimax_music3_dav.safetensors"
MODEL_FILES = {
    "diffusion": MODEL_ROOT / "diffusion_models" / UNET_NAME,
    "text_encoder": MODEL_ROOT / "text_encoders" / CLIP_NAME,
    "audio_vae": MODEL_ROOT / "vae" / VAE_NAME,
}

SAMPLERS = {
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "uni_pc",
}
SCHEDULERS = {"simple", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform", "beta"}
FORMATS = {"flac", "mp3", "opus"}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class GenerationRequest(BaseModel):
    global_metadata: str = Field(min_length=1, max_length=8000)
    vocal_details: str = Field(default="", max_length=8000)
    arrangement: str = Field(default="", max_length=12000)
    lyrics: str = Field(default="[Instrumental]", max_length=30000)
    duration: float = Field(default=120.0, ge=0.04, le=300.0)
    seed: int = Field(default=0, ge=0, le=0xFFFFFFFFFFFFFFFF)
    steps: int = Field(default=30, ge=1, le=100)
    cfg: float = Field(default=1.7, ge=0.0, le=100.0)
    top_k: int = Field(default=50, ge=1, le=1024)
    batch: int = Field(default=1, ge=1, le=8)
    sampler: str = "euler"
    scheduler: str = "simple"
    tiled_decode: bool = False
    tile_size: int = Field(default=1536, ge=32, le=8192)
    tile_overlap: int = Field(default=64, ge=0, le=1024)
    output_format: str = "flac"
    quality: str = "V0"

    def caption(self) -> str:
        sections = [("Global Metadata", self.global_metadata.strip())]
        if self.vocal_details.strip():
            sections.append(("Vocal Details", self.vocal_details.strip()))
        if self.arrangement.strip():
            sections.append(("Arrangement", self.arrangement.strip()))
        return "\n\n".join(f"{name}: {value}" for name, value in sections)


class EngineManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.model_loaded = False
        self._client = httpx.AsyncClient(base_url=ENGINE_URL, timeout=30.0)
        self._lifecycle_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def _probe(self) -> bool:
        try:
            response = await self._client.get("/system_stats", timeout=2.0)
            return response.is_success
        except httpx.HTTPError:
            return False

    def _write_runtime_config(self) -> Path:
        for path in (
            STATE_ROOT,
            STATE_ROOT / "input",
            STATE_ROOT / "temp",
            STATE_ROOT / "user",
            OUTPUT_ROOT,
        ):
            path.mkdir(parents=True, exist_ok=True)
        config = STATE_ROOT / "extra_model_paths.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "mm_tools_minimax": {
                        "base_path": str(MODEL_ROOT),
                        "diffusion_models": "diffusion_models",
                        "text_encoders": "text_encoders",
                        "vae": "vae",
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return config

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self.running and await self._probe():
                return
            if await self._probe():
                raise RuntimeError(
                    f"Port {ENGINE_PORT} already hosts another inference engine; stop it before starting this studio"
                )
            if not (RUNTIME / "main.py").is_file():
                raise RuntimeError("The bundled MiniMax inference runtime is incomplete")
            config = self._write_runtime_config()
            env = os.environ.copy()
            env.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "DO_NOT_TRACK": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            command = [
                sys.executable,
                str(RUNTIME / "main.py"),
                "--listen",
                ENGINE_HOST,
                "--port",
                str(ENGINE_PORT),
                "--extra-model-paths-config",
                str(config),
                "--output-directory",
                str(OUTPUT_ROOT),
                "--input-directory",
                str(STATE_ROOT / "input"),
                "--temp-directory",
                str(STATE_ROOT / "temp"),
                "--user-directory",
                str(STATE_ROOT / "user"),
                "--database-url",
                f"sqlite:///{STATE_ROOT / 'user' / 'comfyui.db'}",
                "--disable-auto-launch",
                "--disable-api-nodes",
                "--disable-metadata",
                "--disable-all-custom-nodes",
                "--whitelist-custom-nodes",
                "mmtools_minimax",
            ]
            self.process = subprocess.Popen(
                command,
                cwd=RUNTIME,
                env=env,
                start_new_session=True,
            )
            deadline = time.monotonic() + ENGINE_START_TIMEOUT
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    code = self.process.returncode
                    self.process = None
                    raise RuntimeError(f"MiniMax inference engine exited during startup (code {code})")
                if await self._probe():
                    return
                await asyncio.sleep(0.5)
            await self._stop_process()
            raise RuntimeError(f"MiniMax inference engine did not become ready in {ENGINE_START_TIMEOUT:g} seconds")

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if not await self._probe():
            await self.start()
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response

    async def queue_prompt(self, graph: dict[str, Any]) -> str:
        response = await self.request(
            "POST",
            "/prompt",
            json={"prompt": graph, "client_id": f"mm-tools-{uuid.uuid4().hex}"},
            timeout=60.0,
        )
        payload = response.json()
        errors = payload.get("node_errors") or {}
        if errors:
            raise RuntimeError(f"Inference graph validation failed: {json.dumps(errors, ensure_ascii=False)}")
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("Inference engine accepted no prompt id")
        return str(prompt_id)

    async def wait_for_history(self, prompt_id: str, timeout: float = JOB_TIMEOUT) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = await self.request("GET", f"/history/{prompt_id}", timeout=30.0)
            payload = response.json()
            if prompt_id in payload:
                entry = payload[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"MiniMax inference failed: {json.dumps(messages, ensure_ascii=False)}")
                return entry
            await asyncio.sleep(0.75)
        raise TimeoutError(f"MiniMax inference exceeded {timeout:g} seconds")

    async def load_models(self) -> None:
        async with self._model_lock:
            prompt_id = await self.queue_prompt(_load_graph())
            await self.wait_for_history(prompt_id, timeout=1800.0)
            self.model_loaded = True

    async def unload_models(self) -> None:
        async with self._model_lock:
            if await self._probe():
                try:
                    await self._client.post("/interrupt", timeout=10.0)
                except httpx.HTTPError:
                    pass
                response = await self._client.post(
                    "/free",
                    json={"unload_models": True, "free_memory": True},
                    timeout=120.0,
                )
                response.raise_for_status()
            self.model_loaded = False

    async def system_stats(self) -> dict[str, Any]:
        if not await self._probe():
            return {}
        try:
            return (await self._client.get("/system_stats", timeout=5.0)).json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def _stop_process(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.send_signal(signal.SIGINT)
            await asyncio.to_thread(process.wait, 20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 10)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)

    async def stop(self) -> None:
        try:
            await self.unload_models()
        except Exception:
            self.model_loaded = False
        await self._stop_process()
        await self._client.aclose()


def _load_graph() -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {
            "class_type": "MMToolsMiniMaxPreload",
            "inputs": {"model": ["1", 0], "clip": ["2", 0], "vae": ["3", 0]},
        },
    }


def _save_node(output_format: str, filename_prefix: str, quality: str) -> dict[str, Any]:
    if output_format == "mp3":
        mp3_quality = quality if quality in {"V0", "128k", "320k"} else "V0"
        return {
            "class_type": "SaveAudioMP3",
            "inputs": {"audio": ["9", 0], "filename_prefix": filename_prefix, "quality": mp3_quality},
        }
    if output_format == "opus":
        opus_quality = quality if quality in {"64k", "96k", "128k", "192k", "320k"} else "192k"
        return {
            "class_type": "SaveAudioOpus",
            "inputs": {"audio": ["9", 0], "filename_prefix": filename_prefix, "quality": opus_quality},
        }
    return {
        "class_type": "SaveAudio",
        "inputs": {"audio": ["9", 0], "filename_prefix": filename_prefix},
    }


def _generation_graph(request: GenerationRequest, job_id: str) -> dict[str, Any]:
    decode_inputs: dict[str, Any] = {"samples": ["7", 0], "vae": ["3", 0]}
    decode_type = "VAEDecodeAudio"
    if request.tiled_decode:
        decode_type = "VAEDecodeAudioTiled"
        decode_inputs.update({"tile_size": request.tile_size, "overlap": request.tile_overlap})
    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {
            "class_type": "MiniMaxMusic3TextEncode",
            "inputs": {
                "clip": ["2", 0],
                "caption": request.caption(),
                "lyrics": request.lyrics,
                "seed": request.seed,
                "max_duration": request.duration,
                "cfg_scale": request.cfg,
                "top_k": request.top_k,
            },
        },
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {
            "class_type": "EmptyMiniMaxMusic3LatentAudio",
            "inputs": {"seconds": ["4", 1], "batch_size": request.batch},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": request.seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": request.sampler,
                "scheduler": request.scheduler,
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "denoise": 1.0,
            },
        },
        "9": {"class_type": decode_type, "inputs": decode_inputs},
    }
    graph["10"] = _save_node(request.output_format, f"minimax/{job_id}", request.quality)
    return graph


def _resolve_output(subfolder: str, filename: str) -> Path:
    candidate = (OUTPUT_ROOT / subfolder / filename).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise RuntimeError("Inference engine returned an output outside the configured directory") from exc
    if not candidate.is_file():
        raise RuntimeError(f"Inference output is missing: {candidate.name}")
    return candidate


def _history_audio_paths(history: dict[str, Any], job_id: str) -> list[Path]:
    paths: list[Path] = []
    for node_output in history.get("outputs", {}).values():
        for item in node_output.get("audio", []):
            if isinstance(item, dict) and item.get("filename"):
                paths.append(_resolve_output(str(item.get("subfolder", "")), str(item["filename"])))
    if not paths:
        paths = sorted(path.resolve() for path in (OUTPUT_ROOT / "minimax").glob(f"{job_id}*.*") if path.is_file())
    if not paths:
        raise RuntimeError("MiniMax completed without returning an audio file")
    return paths


engine = EngineManager()
jobs: dict[str, dict[str, Any]] = {}
job_tasks: set[asyncio.Task[None]] = set()
generation_gate = asyncio.Lock()


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if key not in {"paths", "task"}}
    if job.get("paths"):
        result["audio"] = [f"/api/jobs/{job['id']}/audio/{index}" for index, _ in enumerate(job["paths"])]
    return result


async def _run_generation(job: dict[str, Any], request: GenerationRequest) -> None:
    job["status"] = "waiting"
    try:
        async with generation_gate:
            job["status"] = "generating"
            job["started_at"] = time.time()
            prompt_id = await engine.queue_prompt(_generation_graph(request, job["id"]))
            job["prompt_id"] = prompt_id
            history = await engine.wait_for_history(prompt_id)
            job["paths"] = _history_audio_paths(history, job["id"])
            job["status"] = "complete"
            job["completed_at"] = time.time()
            engine.model_loaded = True
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["completed_at"] = time.time()
        raise
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["completed_at"] = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await engine.start()
    try:
        yield
    finally:
        for task in list(job_tasks):
            task.cancel()
        if job_tasks:
            await asyncio.gather(*job_tasks, return_exceptions=True)
        await engine.stop()


app = FastAPI(title="MiniMax Music Studio", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    stats = await engine.system_stats()
    return {
        "ok": True,
        "cloud": False,
        "engine": engine.running and bool(stats),
        "loaded": engine.model_loaded,
        "model_dir": str(MODEL_ROOT),
        "models": {name: path.is_file() for name, path in MODEL_FILES.items()},
        "devices": stats.get("devices", []),
    }


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "samplers": sorted(SAMPLERS),
        "schedulers": sorted(SCHEDULERS),
        "formats": sorted(FORMATS),
        "defaults": {
            "duration": 120,
            "steps": 30,
            "cfg": 1.7,
            "top_k": 50,
            "sampler": "euler",
            "scheduler": "simple",
            "tile_size": 1536,
            "tile_overlap": 64,
        },
    }


@app.post("/api/load")
async def load_models() -> dict[str, Any]:
    missing = [str(path) for path in MODEL_FILES.values() if not path.is_file()]
    if missing:
        raise HTTPException(400, f"Missing model artifacts: {', '.join(missing)}")
    try:
        await engine.load_models()
        return {"ok": True, "loaded": True}
    except Exception as exc:
        raise HTTPException(500, f"Model load failed: {exc}") from exc


@app.post("/api/unload")
async def unload_models() -> dict[str, Any]:
    for task in list(job_tasks):
        if not task.done():
            task.cancel()
    try:
        await engine.unload_models()
        return {"ok": True, "loaded": False}
    except Exception as exc:
        raise HTTPException(500, f"Model unload failed: {exc}") from exc


@app.post("/api/interrupt")
async def interrupt() -> dict[str, Any]:
    for task in list(job_tasks):
        if not task.done():
            task.cancel()
    try:
        await engine.request("POST", "/interrupt", timeout=10.0)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(500, f"Interrupt failed: {exc}") from exc


@app.post("/api/generate", status_code=202)
async def generate(request: GenerationRequest) -> dict[str, Any]:
    if request.sampler not in SAMPLERS:
        raise HTTPException(400, f"Unsupported sampler: {request.sampler}")
    if request.scheduler not in SCHEDULERS:
        raise HTTPException(400, f"Unsupported scheduler: {request.scheduler}")
    if request.output_format not in FORMATS:
        raise HTTPException(400, f"Unsupported output format: {request.output_format}")
    if request.tile_overlap >= request.tile_size:
        raise HTTPException(400, "Tile overlap must be smaller than tile size")
    if any(not path.is_file() for path in MODEL_FILES.values()):
        raise HTTPException(400, "MiniMax model artifacts are incomplete; run models/download_models.py")

    job_id = uuid.uuid4().hex[:16]
    job: dict[str, Any] = {
        "id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "request": request.model_dump(),
        "paths": [],
    }
    jobs[job_id] = job
    task = asyncio.create_task(_run_generation(job, request), name=f"minimax-{job_id}")
    job["task"] = task
    job_tasks.add(task)
    task.add_done_callback(job_tasks.discard)
    return _public_job(job)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown generation job")
    return _public_job(job)


@app.get("/api/jobs/{job_id}/audio/{index}")
def job_audio(job_id: str, index: int) -> FileResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown generation job")
    paths: list[Path] = job.get("paths", [])
    if index < 0 or index >= len(paths):
        raise HTTPException(404, "Unknown audio output")
    return FileResponse(paths[index], filename=paths[index].name)
