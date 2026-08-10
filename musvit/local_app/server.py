from __future__ import annotations

import gc
import os
import shutil
import tempfile
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from sheet_to_midi import (
    IMAGE_SUFFIXES,
    SMTModelForCausalLM,
    _decode_bekern,
    _fit_page,
    _score_pages,
    _write_derivatives,
)

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = int(os.getenv("MUSVIT_MAX_UPLOAD_MB", "256")) * 1024 * 1024


class ModelManager:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._lock = threading.RLock()
        self._inference = threading.Lock()

    @property
    def model_path(self) -> Path:
        return Path(os.getenv("SMT_MODEL_PATH", "")).expanduser().resolve()

    @property
    def device(self) -> str:
        return os.getenv("MUSVIT_DEVICE", "cuda")

    @property
    def dtype_name(self) -> str:
        return os.getenv("MUSVIT_DTYPE", "float32")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            model_path = self.model_path
            if not (model_path / "model.safetensors").is_file():
                raise FileNotFoundError(f"SMT checkpoint is incomplete: {model_path}")
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            dtype = getattr(torch, self.dtype_name)
            if self.device == "cpu" and dtype != torch.float32:
                raise RuntimeError("CPU inference requires MUSVIT_DTYPE=float32")
            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
            self._model = SMTModelForCausalLM.from_pretrained(
                str(model_path), local_files_only=True, use_safetensors=True
            ).to(device=self.device, dtype=dtype)
            self._model.eval()
            return self._model

    def unload(self) -> None:
        with self._inference:
            with self._lock:
                self._model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def convert(
        self,
        source: Path,
        output_dir: Path,
        *,
        page: int | None,
        pdf_dpi: int,
        max_tokens: int | None,
        write_svg: bool,
    ) -> list[Path]:
        with self._inference:
            model = self.load()
            pages_dir = output_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            pages = _score_pages(source, pages_dir, pdf_dpi)
            selected = [(index, path) for index, path in enumerate(pages, start=1)]
            if page is not None:
                selected = [item for item in selected if item[0] == page]
                if not selected:
                    raise ValueError(f"PDF has no page {page}")

            original_maxlen = model.maxlen
            if max_tokens:
                model.maxlen = min(original_maxlen, max_tokens)
            created: list[Path] = []
            try:
                for page_number, page_path in selected:
                    stem = source.stem if len(selected) == 1 else f"page-{page_number:03d}"
                    midi_path = output_dir / f"{stem}.mid"
                    image = _fit_page(page_path, model.config.maxh, model.config.maxw).to(
                        device=self.device, dtype=getattr(torch, self.dtype_name)
                    )
                    with torch.inference_mode():
                        tokens, _ = model.predict(image, convert_to_str=True)
                    kern = _decode_bekern(tokens)
                    _write_derivatives(kern, midi_path, write_svg=write_svg)
                    created.extend([midi_path, midi_path.with_suffix(".krn")])
                    svg = midi_path.with_suffix(".svg")
                    if svg.is_file():
                        created.append(svg)
            finally:
                model.maxlen = original_maxlen
            return created


manager = ModelManager()


def _autoload() -> bool:
    return os.getenv("MUSVIT_AUTOLOAD", "0").strip().lower() not in {"0", "false", "no"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _autoload():
        await run_in_threadpool(manager.load)
    try:
        yield
    finally:
        await run_in_threadpool(manager.unload)


app = FastAPI(title="MuSViT Local Studio", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "loaded": manager.loaded,
        "model_path": str(manager.model_path),
        "model_present": (manager.model_path / "model.safetensors").is_file(),
        "device": manager.device,
        "dtype": manager.dtype_name,
        "cloud": False,
    }


@app.post("/api/load")
def load() -> dict:
    manager.load()
    return {"ok": True, "loaded": True}


@app.post("/api/unload")
def unload() -> dict:
    manager.unload()
    return {"ok": True, "loaded": False}


def _copy_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise ValueError(f"Upload exceeds {MAX_UPLOAD // (1024 * 1024)} MiB")
            output.write(chunk)


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    page: int | None = Form(None),
    pdf_dpi: int = Form(220),
    max_tokens: int | None = Form(None),
    write_svg: bool = Form(True),
) -> FileResponse:
    if page is not None and page < 1:
        raise HTTPException(400, "Page is 1-based")
    if not 72 <= pdf_dpi <= 600:
        raise HTTPException(400, "PDF DPI must be between 72 and 600")
    if max_tokens is not None and not 64 <= max_tokens <= 8192:
        raise HTTPException(400, "Maximum tokens must be between 64 and 8192")
    suffix = Path(file.filename or "score").suffix.lower()
    if suffix != ".pdf" and suffix not in IMAGE_SUFFIXES:
        raise HTTPException(400, f"Unsupported score format: {suffix or '(none)'}")

    work = Path(tempfile.mkdtemp(prefix="musvit-web-"))
    try:
        source = work / f"score{suffix}"
        await run_in_threadpool(_copy_upload, file, source)
        outputs = work / "outputs"
        outputs.mkdir()
        created = await run_in_threadpool(
            manager.convert,
            source,
            outputs,
            page=page,
            pdf_dpi=pdf_dpi,
            max_tokens=max_tokens,
            write_svg=write_svg,
        )
        archive = work / f"{Path(file.filename or 'score').stem}-omr.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for artifact in created:
                bundle.write(artifact, artifact.name)
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=archive.name,
            background=BackgroundTask(shutil.rmtree, work, ignore_errors=True),
        )
    except (ValueError, FileNotFoundError, SystemExit) as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Score conversion failed: {exc}") from exc
    finally:
        await file.close()
