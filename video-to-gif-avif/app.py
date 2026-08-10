from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine import AnimationOptions, available_encoders, convert, probe


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
WORKSPACE = ROOT / "workspace"
INPUTS = WORKSPACE / "inputs"
OUTPUTS = WORKSPACE / "outputs"
for directory in (INPUTS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    id: str
    name: str
    input_path: Path
    output_path: Path
    options: AnimationOptions
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    before: dict | None = None
    after: dict | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=300), repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "status": self.status,
            "progress": self.progress, "error": self.error, "created_at": self.created_at,
            "before": self.before, "after": self.after, "options": asdict(self.options),
            "logs": list(self.logs),
            "output_ready": self.status == "complete" and self.output_path.is_file(),
        }


class Jobs:
    def __init__(self) -> None:
        self.items: dict[str, Job] = {}
        self.lock = threading.RLock()

    def add(self, job: Job) -> None:
        with self.lock:
            if any(item.status in {"queued", "running"} for item in self.items.values()):
                raise ValueError("An animation job is already active.")
            self.items[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def get(self, job_id: str) -> Job:
        try:
            return self.items[job_id]
        except KeyError as exc:
            raise KeyError("Unknown or expired job") from exc

    def _run(self, job: Job) -> None:
        job.status = "running"
        try:
            job.before = asdict(probe(job.input_path))

            def update(progress: float, line: str) -> None:
                if progress >= 0:
                    job.progress = progress
                elif line and not line.startswith(("frame=", "fps=", "bitrate=", "speed=")):
                    job.logs.append(line)

            job.after = asdict(
                convert(
                    job.input_path, job.output_path, job.options,
                    on_progress=update,
                    on_process=lambda process: setattr(job, "process", process),
                    cancel=job.cancel,
                )
            )
            job.status = "complete"
            job.progress = 1.0
        except InterruptedError:
            job.status = "stopped"
            job.error = "Stopped by user."
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.logs.append(str(exc))
        finally:
            job.process = None
            (OUTPUTS / f"{job.id}.json").write_text(json.dumps(job.public(), indent=2) + "\n", encoding="utf-8")


jobs = Jobs()
app = FastAPI(title="Video to GIF / AVIF", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/capabilities")
def capabilities() -> dict:
    encoders = available_encoders()
    return {
        "ffmpeg": shutil.which("ffmpeg"),
        "gif": "gif" in encoders,
        "avif_software": "libaom-av1" in encoders,
        "avif_nvenc": "av1_nvenc" in encoders,
        "unlimited_upload": True,
    }


async def save_upload(upload: UploadFile, target: Path) -> None:
    minimum = float(os.getenv("ANIMATOR_MIN_FREE_GIB", "2")) * 1024**3
    total = 0
    try:
        with target.open("xb") as destination:
            while chunk := await upload.read(8 * 1024 * 1024):
                if minimum and shutil.disk_usage(WORKSPACE).free - len(chunk) < minimum:
                    raise ValueError("Local disk free-space reserve reached while streaming the input.")
                destination.write(chunk)
                total += len(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if not total:
        target.unlink(missing_ok=True)
        raise ValueError("The uploaded file is empty.")


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    format: str = Form("gif"),
    start: float = Form(0), end: float = Form(0), width: int = Form(640),
    fps: float = Form(15), speed: float = Form(1),
    crop_x: int = Form(0), crop_y: int = Form(0), crop_width: int = Form(0), crop_height: int = Form(0),
    rotate: str = Form("none"), flip: str = Form("none"), loop: int = Form(0),
    gif_colors: int = Form(256), gif_dither: str = Form("sierra2_4a"),
    gif_palette: str = Form("diff"), gif_alpha_threshold: int = Form(128),
    avif_quality: int = Form(78), avif_effort: int = Form(6),
    avif_engine: str = Form("software"), avif_10bit: bool = Form(False),
) -> dict:
    options = AnimationOptions(
        format, start, end, width, fps, speed, crop_x, crop_y, crop_width, crop_height,
        rotate, flip, loop, gif_colors, gif_dither, gif_palette, gif_alpha_threshold,
        avif_quality, avif_effort, avif_engine, avif_10bit,
    )
    try:
        options.validate()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".video"
    input_path = INPUTS / f"{job_id}{suffix}"
    try:
        await save_upload(video, input_path)
        info = probe(input_path)
    except (ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read the video: {exc}") from exc
    job = Job(job_id, Path(video.filename or "video").stem, input_path, OUTPUTS / f"{job_id}.{format}", options, before=asdict(info))
    try:
        jobs.add(job)
    except ValueError as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        return jobs.get(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    job.cancel.set()
    if job.process and job.process.poll() is None:
        job.process.terminate()
    return job.public()


@app.get("/api/jobs/{job_id}/output")
def output_file(job_id: str, download: bool = False) -> FileResponse:
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.status != "complete" or not job.output_path.is_file():
        raise HTTPException(404, "Output is not ready.")
    media_type = "image/gif" if job.output_path.suffix == ".gif" else "image/avif"
    filename = f"{job.name}.{job.options.format}"
    return FileResponse(job.output_path, media_type=media_type, filename=filename if download else None)
