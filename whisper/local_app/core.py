from __future__ import annotations

import dataclasses
import gc
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path.home() / "multimedia/models/nyralabs--CrisperWhisper2.0_large"


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path
    backend: str
    device: str
    compute_type: str
    draft_model: str | None
    cache_dir: Path

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        draft = os.getenv("CW2_DRAFT_MODEL", "").strip() or None
        return cls(
            model_path=Path(os.getenv("CW2_MODEL_PATH", str(DEFAULT_MODEL))).expanduser(),
            backend=os.getenv("CW2_BACKEND", "transformers").strip(),
            device=os.getenv("CW2_DEVICE", "cuda").strip(),
            compute_type=os.getenv("CW2_COMPUTE_TYPE", "float16").strip(),
            draft_model=draft,
            cache_dir=Path(
                os.getenv("CW2_CACHE_DIR", str(Path.home() / ".cache/crisperwhisper"))
            ).expanduser(),
        )


def normalize_audio(source: Path, destination: Path) -> Path:
    """Decode any ffmpeg-supported input into the model's canonical PCM WAV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        message = completed.stderr.strip() or "ffmpeg could not decode this media file"
        raise ValueError(message)
    return destination


def result_to_dict(result: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unexpected CrisperWhisper result: {type(result).__name__}")


class ModelManager:
    """One lazy model per server, guarded against concurrent first loads."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._config: RuntimeConfig | None = None
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def config(self) -> RuntimeConfig:
        return self._config or RuntimeConfig.from_env()

    def load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            cfg = RuntimeConfig.from_env()
            if not cfg.model_path.is_dir():
                raise FileNotFoundError(
                    f"CrisperWhisper checkpoint not found: {cfg.model_path}. "
                    "Set CW2_MODEL_PATH in .env."
                )
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            from crisperwhisper import CrisperWhisperModel

            self._model = CrisperWhisperModel(
                str(cfg.model_path),
                backend=cfg.backend,
                compute_type=cfg.compute_type,
                device=cfg.device,
                draft_model=cfg.draft_model,
                cache_dir=cfg.cache_dir,
            )
            self._config = cfg
            return self._model

    def unload(self) -> None:
        # Use the same lock order as inference -> load so unloading cannot race
        # a live transcription or invert the locks.
        with self._inference_lock:
            with self._lock:
                self._model = None
                self._config = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    def run(
        self,
        audio: Path,
        *,
        operation: str,
        language: str,
        transcript: str,
        word_timestamps: bool,
        strategy: str,
        chunk_duration: float,
        stride: float,
        context_words: int,
        max_new_tokens: int,
        hotwords: list[str],
    ) -> dict[str, Any]:
        with self._inference_lock:
            return self._run_locked(
                audio,
                operation=operation,
                language=language,
                transcript=transcript,
                word_timestamps=word_timestamps,
                strategy=strategy,
                chunk_duration=chunk_duration,
                stride=stride,
                context_words=context_words,
                max_new_tokens=max_new_tokens,
                hotwords=hotwords,
            )

    def detect_language(self, audio: Path) -> dict[str, Any]:
        """Reuse the resident model for a single Whisper language-ID pass."""
        with self._inference_lock:
            model = self.load()
            language, confidence = model.detect_language(audio)
            return {"language": language, "confidence": confidence}

    def _run_locked(
        self,
        audio: Path,
        *,
        operation: str,
        language: str,
        transcript: str,
        word_timestamps: bool,
        strategy: str,
        chunk_duration: float,
        stride: float,
        context_words: int,
        max_new_tokens: int,
        hotwords: list[str],
    ) -> dict[str, Any]:
        model = self.load()
        common = {
            "language": language,
            "max_new_tokens": max_new_tokens,
            "word_timestamps": word_timestamps,
        }
        transcribe = {
            **common,
            "hotwords": hotwords or None,
            "longform_strategy": strategy,
            "chunk_duration": chunk_duration,
            "stride": stride,
            "context_words": context_words,
        }

        if operation in {"verbatim", "intended"}:
            return {operation: result_to_dict(model.transcribe(audio, mode=operation, **transcribe))}
        if operation == "both":
            if self.config.backend == "ct2":
                results = model.transcribe_dual(
                    audio,
                    modes=("verbatim", "intended"),
                    **transcribe,
                )
            else:
                results = (
                    model.transcribe(audio, mode="verbatim", **transcribe),
                    model.transcribe(audio, mode="intended", **transcribe),
                )
            return {
                "verbatim": result_to_dict(results[0]),
                "intended": result_to_dict(results[1]),
            }
        if operation == "verbatimize":
            if not transcript.strip():
                raise ValueError("Verbatimize requires an intended transcript.")
            return {
                "verbatim": result_to_dict(
                    model.verbatimize(audio, transcript, **common)
                )
            }
        if operation == "align":
            if not transcript.strip():
                raise ValueError("Forced alignment requires a reference transcript.")
            return {
                "alignment": result_to_dict(
                    model.forced_align(
                        audio,
                        transcript,
                        language=language,
                        longform_strategy=strategy,
                    )
                )
            }
        raise ValueError(f"Unknown operation: {operation}")


manager = ModelManager()
