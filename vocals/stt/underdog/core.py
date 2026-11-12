from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SOURCE = json.loads((REPO / "models/manifests/underdog.json").read_text())
VARIANTS = SOURCE["variants"]
MODELS = Path(os.getenv("BARK_MODELS_DIR", str(ROOT / "models"))).expanduser().resolve()
DEFAULT_MODEL = os.getenv("BARK_MODEL", "pro")
DEVICE = os.getenv("BARK_DEVICE", "cuda")
PRO_DTYPE = os.getenv("BARK_PRO_DTYPE", "bfloat16")
MAX_SECONDS = int(os.getenv("BARK_MAX_AUDIO_SECONDS", "600"))
if DEFAULT_MODEL not in VARIANTS:
    raise ValueError("BARK_MODEL must be base, plus, clean or pro")
if DEVICE not in {"cuda", "cpu", "mps"}:
    raise ValueError("BARK_DEVICE must be cuda, cpu or mps")
if PRO_DTYPE not in {"bfloat16", "float16", "float32"}:
    raise ValueError("BARK_PRO_DTYPE must be bfloat16, float16 or float32")
if not 1 <= MAX_SECONDS <= 86400:
    raise ValueError("BARK_MAX_AUDIO_SECONDS must be between 1 and 86400")


def missing_files(variant: str) -> list[str]:
    files = [
        MODELS / variant / name
        for name in VARIANTS[variant]["files"]
        if name not in {"LICENSE", "NOTICE", "README.md"}
    ]
    if variant != "pro":
        files.extend(
            MODELS / "processor" / name
            for name in SOURCE["processor"]["files"]
            if name != "README.md"
        )
    missing = []
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
        elif path.stat().st_size < 256:
            if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec"):
                missing.append(str(path))
    return missing


def require_model(variant: str) -> None:
    absent = missing_files(variant)
    if absent:
        raise FileNotFoundError(
            f"Complete the underdog-{variant} downloader bundle. Missing: {absent[0]}"
        )


_decoder = None


def stop_decoder():
    decoder = _decoder
    if decoder is not None and decoder.poll() is None:
        try:
            decoder.terminate()
        except ProcessLookupError:
            pass


def normalize_audio(source: Path, destination: Path, cancel: threading.Event) -> None:
    global _decoder
    # A local upload cannot ask FFmpeg to open a network playlist. Decode only
    # one bounded audio stream; the extra second detects oversize recordings.
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-t",
        str(MAX_SECONDS + 1),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        str(destination),
    ]
    decoder = None
    try:
        if cancel.is_set():
            raise Cancelled("Transcription cancelled")
        decoder = subprocess.Popen(
            [sys.executable, str(ROOT / "media_exec.py"), str(os.getpid()), *command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _decoder = decoder
        if cancel.is_set():
            stop_decoder()
        _, error = decoder.communicate(timeout=180)
        if cancel.is_set():
            raise Cancelled("Transcription cancelled")
        if decoder.returncode:
            raise ValueError(error.strip()[-2000:] or "Could not decode audio")
    except FileNotFoundError as exc:
        raise ValueError("Install ffmpeg before transcribing media") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio conversion exceeded 180 seconds") from exc
    finally:
        if decoder is not None:
            if decoder.poll() is None:
                try:
                    decoder.kill()
                except ProcessLookupError:
                    pass
            decoder.communicate()
        _decoder = None


class Cancelled(Exception):
    pass


class ModelManager:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.stopping = False
        self.progress = {"phase": "idle", "variant": None, "done": 0, "total": 0}

    def status(self):
        return {
            "backend": "transformers",
            "device": DEVICE,
            "models": [
                {
                    "id": key,
                    "name": row["name"],
                    "download": row["download"],
                    "operation": row["operation"],
                    "loaded": key in self.models,
                    "model_present": not missing_files(key),
                    "compute_type": PRO_DTYPE if key == "pro" else "float32",
                }
                for key, row in VARIANTS.items()
            ],
            "progress": self.progress.copy(),
        }

    def load(self, variant):
        with self.lock:
            if variant in self.models:
                return
            require_model(variant)
            self.progress = {
                "phase": "loading",
                "variant": variant,
                "done": 0,
                "total": 0,
            }
            model = processor = state = None
            try:
                import torch
                from transformers import (
                    AutoProcessor,
                    GenerationConfig,
                    ParakeetForTDT,
                    ParakeetTDTConfig,
                )

                if DEVICE == "cuda" and not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA unavailable; select BARK_DEVICE=cpu explicitly for CPU execution"
                    )
                if DEVICE == "mps" and not torch.backends.mps.is_available():
                    raise ValueError("MPS unavailable on this machine")
                dtype = getattr(torch, PRO_DTYPE) if variant == "pro" else torch.float32
                if DEVICE == "cpu" and dtype != torch.float32:
                    raise ValueError("Set BARK_PRO_DTYPE=float32 for CPU execution")
                if (
                    DEVICE == "cuda"
                    and dtype == torch.bfloat16
                    and not torch.cuda.is_bf16_supported()
                ):
                    raise ValueError(
                        "GPU lacks BF16 support; choose BARK_PRO_DTYPE explicitly"
                    )
                source = MODELS / variant
                if variant == "pro":
                    processor = AutoProcessor.from_pretrained(
                        str(source), local_files_only=True
                    )
                    model = ParakeetForTDT.from_pretrained(
                        str(source), dtype=dtype, local_files_only=True
                    )
                else:
                    # Keep Conway's exact packed-weight decoder. Construct the
                    # stock model here solely to replace its implicit Hub calls
                    # with the pinned processor/config already downloaded.
                    loader_path = source / "bark_loader.py"
                    code = loader_path.read_bytes().replace(b"\r\n", b"\n")
                    if (
                        hashlib.sha256(code).hexdigest()
                        != VARIANTS[variant]["loader_sha256"]
                    ):
                        raise ValueError(
                            f"Restore the pinned underdog-{variant} bark_loader.py"
                        )
                    spec = importlib.util.spec_from_file_location(
                        f"underdog_{variant}", loader_path
                    )
                    loader = importlib.util.module_from_spec(spec)
                    exec(compile(code, str(loader_path), "exec"), loader.__dict__)
                    state, config = loader.load_state_dict(source / "model.bark")
                    model = ParakeetForTDT(ParakeetTDTConfig.from_dict(config))
                    missing, unexpected = model.load_state_dict(
                        {k: v.float() for k, v in state.items()}, strict=False
                    )
                    if unexpected or any(
                        "num_batches_tracked" not in k for k in missing
                    ):
                        raise ValueError(
                            f"Bark checkpoint mismatch: {missing[:5]}, {unexpected[:5]}"
                        )
                    del state
                    state = None
                    processor_path = str(MODELS / "processor")
                    model.generation_config = GenerationConfig.from_pretrained(
                        processor_path, local_files_only=True
                    )
                    processor = AutoProcessor.from_pretrained(
                        processor_path, local_files_only=True
                    )
                model = model.eval().to(device=DEVICE, dtype=dtype)
                self.processors[variant] = processor
                self.models[variant] = model
            finally:
                model = processor = state = None
                gc.collect()
                self._empty_cache()
                self.progress = {
                    "phase": "idle",
                    "variant": None,
                    "done": 0,
                    "total": 0,
                }

    @staticmethod
    def _empty_cache():
        torch = sys.modules.get("torch")
        if torch is not None:
            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif DEVICE == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def unload(self, variant):
        with self.lock:
            self.models.pop(variant, None)
            self.processors.pop(variant, None)
            gc.collect()
            self._empty_cache()

    def close(self):
        self.cancel.set()
        with self.lock:
            self.models.clear()
            self.processors.clear()
            gc.collect()
            self._empty_cache()

    def transcribe(self, path: Path, variant: str, chunk_seconds: int):
        with self.lock:
            import numpy as np
            import soundfile as sf
            import torch

            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError("Expected normalized mono 16 kHz audio")
            duration = info.frames / 16000
            if not 0 < duration <= MAX_SECONDS:
                raise ValueError(
                    f"Audio must be nonempty and at most {MAX_SECONDS} seconds"
                )
            self.load(variant)
            if self.cancel.is_set():
                raise Cancelled("Transcription cancelled")
            model, processor = self.models[variant], self.processors[variant]
            dtype = next(model.parameters()).dtype
            total = (info.frames + chunk_seconds * 16000 - 1) // (chunk_seconds * 16000)
            started = time.monotonic()
            segments = []
            try:
                with sf.SoundFile(path) as audio:
                    for index in range(total):
                        if self.cancel.is_set():
                            raise Cancelled("Transcription cancelled")
                        self.progress = {
                            "phase": "transcribing",
                            "variant": variant,
                            "done": index,
                            "total": total,
                        }
                        wave = audio.read(chunk_seconds * 16000, dtype="float32")
                        end = audio.tell() / 16000
                        wave = np.concatenate(
                            [
                                np.zeros(
                                    int(VARIANTS[variant]["padding_seconds"] * 16000),
                                    np.float32,
                                ),
                                wave,
                            ]
                        )
                        inputs = processor(
                            [wave],
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True,
                        )
                        inputs = {
                            key: value.to(
                                device=DEVICE,
                                dtype=dtype
                                if value.dtype.is_floating_point
                                else value.dtype,
                            )
                            for key, value in inputs.items()
                        }
                        with torch.inference_mode():
                            output = model.generate(**inputs)
                        ids = (
                            output.sequences if hasattr(output, "sequences") else output
                        )
                        text = processor.batch_decode(ids, skip_special_tokens=True)[
                            0
                        ].strip()
                        segments.append(
                            {"start": index * chunk_seconds, "end": end, "text": text}
                        )
                        del inputs, output, ids
                if self.cancel.is_set():
                    raise Cancelled("Transcription cancelled")
                return {
                    "text": " ".join(s["text"] for s in segments if s["text"]),
                    "duration": round(duration, 3),
                    "processing_time": round(time.monotonic() - started, 3),
                    "language": "en",
                    "variant": variant,
                    "words": [],
                    "segments": segments,
                    "timing": "chunk",
                    "revision": VARIANTS[variant]["revision"],
                }
            finally:
                self.progress = {
                    "phase": "idle",
                    "variant": None,
                    "done": 0,
                    "total": 0,
                }
