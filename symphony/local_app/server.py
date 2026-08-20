from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
MODEL_ROOT = Path(os.getenv("SYMPHONY_MODEL_DIR", ROOT.parent / "models" / "SymphonyGen--SymphonyGen")).expanduser().resolve()
OUTPUT_ROOT = Path(os.getenv("SYMPHONY_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
MAX_UPLOAD = int(os.getenv("SYMPHONY_MAX_UPLOAD_MB", "32")) * 1024 * 1024

MODELS = {
    "stage-two": MODEL_ROOT / "stage_two_pretrained.pt",
    "grpo-clamp": MODEL_ROOT / "grpo_clamp_epoch_10.pt",
    "grpo-clamp-track": MODEL_ROOT / "grpo_clamp+track_epoch_6.pt",
}
HARMONY_MODEL = MODEL_ROOT / "stage_one_pretrained.pt"
JOB_LOCK = threading.Lock()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SymphonyGen Local Studio")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_ROOT), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status() -> dict:
    expected = {"harmony": HARMONY_MODEL, **MODELS}
    return {
        "ok": True,
        "cloud": False,
        "busy": JOB_LOCK.locked(),
        "model_dir": str(MODEL_ROOT),
        "models": {name: path.is_file() for name, path in expected.items()},
    }


def _copy_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as destination:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise ValueError(f"Upload exceeds {MAX_UPLOAD // (1024 * 1024)} MiB")
            destination.write(chunk)


def _run(command: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "unknown generator failure")[-6000:]
        raise RuntimeError(detail.strip())
    return (completed.stdout + "\n" + completed.stderr).strip()[-6000:]


def _artifact_response(run_dir: Path, log: str) -> dict:
    artifacts = []
    for path in sorted(run_dir.rglob("*.mid")):
        relative = path.relative_to(OUTPUT_ROOT).as_posix()
        artifacts.append({"name": path.name, "url": f"/outputs/{quote(relative)}"})
    if not artifacts:
        raise RuntimeError("The generator completed without writing a MIDI file")
    return {"ok": True, "artifacts": artifacts, "log": log}


def _new_run(prefix: str) -> Path:
    run = OUTPUT_ROOT / f"{prefix}-{uuid.uuid4().hex[:10]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


@app.post("/api/generate-harmony")
async def generate_harmony(count: int = Form(4)) -> dict:
    if not 1 <= count <= 16:
        raise HTTPException(400, "Harmony count must be between 1 and 16")
    if not HARMONY_MODEL.is_file():
        raise HTTPException(400, f"Harmony checkpoint is missing: {HARMONY_MODEL}")
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another SymphonyGen job is already running")
    run_dir: Path | None = None
    try:
        run_dir = _new_run("harmony")
        command = [
            sys.executable,
            "-m", "arch.harmo.generator",
            str(HARMONY_MODEL),
            "--num_batches", "1",
            "--batch_size", str(count),
            "--save_dir", str(run_dir),
        ]
        log = await run_in_threadpool(_run, command)
        return _artifact_response(run_dir, log)
    except Exception as exc:
        if run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(500, f"Harmony generation failed: {exc}") from exc
    finally:
        JOB_LOCK.release()


@app.post("/api/orchestrate")
async def orchestrate(
    midi: UploadFile = File(...),
    model: str = Form("grpo-clamp-track"),
    group_size: int = Form(2),
    analyze_harmony: bool = Form(True),
    forbid_piano: bool = Form(False),
    dissonance_averse: bool = Form(True),
    hn_weight: float = Form(10.0),
    nn_weight: float = Form(10.0),
    register_decay: int = Form(1),
) -> dict:
    if Path(midi.filename or "").suffix.lower() not in {".mid", ".midi"}:
        raise HTTPException(400, "Upload a MIDI file")
    if model not in MODELS:
        raise HTTPException(400, "Unknown orchestration model")
    model_path = MODELS[model]
    if not model_path.is_file():
        raise HTTPException(400, f"Orchestration checkpoint is missing: {model_path}")
    if not 1 <= group_size <= 8:
        raise HTTPException(400, "Variations must be between 1 and 8")
    if not 0 <= hn_weight <= 100 or not 0 <= nn_weight <= 100:
        raise HTTPException(400, "Dissonance weights must be between 0 and 100")
    if register_decay not in {0, 1}:
        raise HTTPException(400, "Register decay must be 0 or 1")
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another SymphonyGen job is already running")

    work: Path | None = None
    run_dir: Path | None = None
    try:
        work = Path(tempfile.mkdtemp(prefix="symphony-upload-"))
        run_dir = _new_run("orchestration")
        source = work / "condition.mid"
        await run_in_threadpool(_copy_upload, midi, source)
        command = [
            sys.executable,
            "-m", "arch.symph.generator",
            str(model_path),
            str(source),
            "--group_size", str(group_size),
            "--save_dir", str(run_dir),
            "--hn_weight", str(hn_weight),
            "--nn_weight", str(nn_weight),
            "--register_decay", str(register_decay),
        ]
        if analyze_harmony:
            command.append("--analyze_harmo")
        if forbid_piano:
            command.append("--forbid_piano")
        if not dissonance_averse:
            command.append("--disable_dissonance_averse")
        log = await run_in_threadpool(_run, command)
        return _artifact_response(run_dir, log)
    except ValueError as exc:
        if run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        if run_dir is not None:
            shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(500, f"Orchestration failed: {exc}") from exc
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        await midi.close()
        JOB_LOCK.release()
