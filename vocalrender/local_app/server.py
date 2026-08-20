from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from scripts.infer_vocalrender_svs_single import (
    build_svs_prompt_from_entry,
    encode_prompt_audio,
    load_svs_model,
)
from vocalrender.evaluation.audio_utils import normalize_audio
from vocalrender.model.utils import get_out_sample_rate

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
MODEL_ROOT = Path(os.getenv("VOCALRENDER_MODEL_DIR", ROOT.parent / "models" / "pymaster--VocalRender")).expanduser().resolve()
OUTPUT_ROOT = Path(os.getenv("VOCALRENDER_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
MAX_UPLOAD = int(os.getenv("VOCALRENDER_MAX_UPLOAD_MB", "128")) * 1024 * 1024
VARIANTS = {"VocalRender", "VocalRender-Pro"}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class ModelManager:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._variant: str | None = None
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @property
    def device(self) -> str:
        configured = os.getenv("VOCALRENDER_DEVICE", "auto").strip().lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def variant(self) -> str | None:
        return self._variant

    def variant_path(self, variant: str) -> Path:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown model variant: {variant}")
        return MODEL_ROOT / variant

    def load(self, variant: str) -> Any:
        with self._inference_lock:
            with self._state_lock:
                if self._model is not None and self._variant == variant:
                    return self._model
                self._release_unlocked()
                checkpoint = self.variant_path(variant)
                for name in ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json"):
                    if not (checkpoint / name).is_file():
                        raise FileNotFoundError(f"Incomplete {variant} checkpoint: {checkpoint / name}")
                self._model = load_svs_model(str(checkpoint), device=self.device)
                self._variant = variant
                return self._model

    def _release_unlocked(self) -> None:
        self._model = None
        self._variant = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        with self._inference_lock:
            with self._state_lock:
                self._release_unlocked()

    def synthesize(
        self,
        *,
        variant: str,
        entry: dict,
        prompt_audio: Path,
        output: Path,
        lyrics_only: bool,
        prompt_max_frames: int,
        cfg_value: float,
        inference_timesteps: int,
        max_len: int,
        temperature: float,
        fsq_temperature: float,
    ) -> float:
        model = self.load(variant)
        with self._inference_lock:
            svs_prompt = build_svs_prompt_from_entry(entry, model, force_lyrics_only=lyrics_only)
            prompt_feats = encode_prompt_audio(str(prompt_audio), model, max_frames=prompt_max_frames)
            if prompt_feats is None or prompt_feats.numel() == 0:
                raise RuntimeError("Prompt-audio encoding returned no features")
            with torch.inference_mode():
                generated = model.generate_batch(
                    target_texts=[svs_prompt],
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_len,
                    verbose=True,
                    temperature=temperature,
                    fsq_temperature=fsq_temperature,
                    prompt_audio_feats=[prompt_feats],
                )
            if not generated or generated[0] is None or generated[0].numel() == 0:
                raise RuntimeError("VocalRender returned no audio")
            audio = normalize_audio(generated[0].detach().float().cpu().numpy().flatten())
            sample_rate = get_out_sample_rate(model)
            sf.write(output, audio, sample_rate)
            return len(audio) / sample_rate


manager = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _truthy("VOCALRENDER_AUTOLOAD"):
        variant = os.getenv("VOCALRENDER_DEFAULT_VARIANT", "VocalRender-Pro")
        await run_in_threadpool(manager.load, variant)
    try:
        yield
    finally:
        await run_in_threadpool(manager.unload)


app = FastAPI(title="VocalRender Local Studio", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_ROOT), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {
        "ok": True,
        "cloud": False,
        "loaded": manager.loaded,
        "variant": manager.variant,
        "device": manager.device,
        "model_dir": str(MODEL_ROOT),
        "models": {
            variant: all((MODEL_ROOT / variant / name).is_file() for name in ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json"))
            for variant in sorted(VARIANTS)
        },
    }


@app.post("/api/load")
async def load(variant: str = Form("VocalRender-Pro")) -> dict:
    try:
        await run_in_threadpool(manager.load, variant)
        return {"ok": True, "loaded": True, "variant": manager.variant}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Model load failed: {exc}") from exc


@app.post("/api/unload")
async def unload() -> dict:
    await run_in_threadpool(manager.unload)
    return {"ok": True, "loaded": False}


def _copy_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as destination:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise ValueError(f"Upload exceeds {MAX_UPLOAD // (1024 * 1024)} MiB")
            destination.write(chunk)


def _select_entry(raw: str, item_name: str) -> dict:
    data = json.loads(raw)
    entries = data if isinstance(data, list) else [data]
    if not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Entry JSON must be an object or list of objects")
    if item_name:
        for entry in entries:
            if str(entry.get("item_name", "")) == item_name:
                return entry
        raise ValueError(f"item_name '{item_name}' was not found in the JSON")
    if len(entries) != 1:
        raise ValueError("Provide item_name when the JSON contains multiple entries")
    return entries[0]


@app.post("/api/generate")
async def generate(
    prompt_audio: UploadFile = File(...),
    entry_json: str = Form(...),
    item_name: str = Form(""),
    variant: str = Form("VocalRender-Pro"),
    lyrics_only: bool = Form(False),
    prompt_max_frames: int = Form(50),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    max_len: int = Form(2000),
    temperature: float = Form(1.0),
    fsq_temperature: float = Form(0.0),
) -> dict:
    suffix = Path(prompt_audio.filename or "prompt.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg"}:
        raise HTTPException(400, "Prompt audio must be WAV, FLAC, or OGG")
    if not 1 <= prompt_max_frames <= 200:
        raise HTTPException(400, "Prompt frames must be between 1 and 200")
    if not 1 <= inference_timesteps <= 100:
        raise HTTPException(400, "Inference steps must be between 1 and 100")
    if not 64 <= max_len <= 8000:
        raise HTTPException(400, "Maximum length must be between 64 and 8000")
    if not 0.0 <= temperature <= 4.0 or not 0.0 <= fsq_temperature <= 4.0:
        raise HTTPException(400, "Temperatures must be between 0 and 4")
    if not 0.0 <= cfg_value <= 10.0:
        raise HTTPException(400, "CFG must be between 0 and 10")

    work = Path(tempfile.mkdtemp(prefix="vocalrender-upload-"))
    output = OUTPUT_ROOT / f"vocal-{uuid.uuid4().hex[:12]}.wav"
    try:
        entry = _select_entry(entry_json, item_name.strip())
        source = work / f"prompt{suffix}"
        await run_in_threadpool(_copy_upload, prompt_audio, source)
        duration = await run_in_threadpool(
            manager.synthesize,
            variant=variant,
            entry=entry,
            prompt_audio=source,
            output=output,
            lyrics_only=lyrics_only,
            prompt_max_frames=prompt_max_frames,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            temperature=temperature,
            fsq_temperature=fsq_temperature,
        )
        relative = output.relative_to(OUTPUT_ROOT).as_posix()
        return {"ok": True, "name": output.name, "url": f"/outputs/{quote(relative)}", "duration": duration}
    except (ValueError, json.JSONDecodeError) as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(500, f"Singing generation failed: {exc}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
        await prompt_audio.close()
