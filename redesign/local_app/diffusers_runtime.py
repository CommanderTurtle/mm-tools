from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
QWEN_ROOT = ROOT.parent / "models" / "qwen"
DEFAULT_TRANSFORMER = (
    QWEN_ROOT
    / "T5B--qwen-image-layered-fp8"
    / "qwen_image_layered_fp8_e4m3fn.safetensors"
)
DEFAULT_COMPONENTS = QWEN_ROOT / "diffusers--hfstaff--Qwen-Image-Layered-modular"
DEFAULT_TEXT_ENCODER = (
    QWEN_ROOT
    / "suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4"
    / "text_encoder"
)


class JobCancelled(RuntimeError):
    pass


class DiffusersRuntime:
    """One resident, local-only Qwen-Image-Layered Diffusers pipeline."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._operation = threading.Lock()
        self._pipeline: Any = None
        self._offload_dir: str | None = None
        self._state = "unloaded"
        self._error: str | None = None
        self._active_job: str | None = None
        self._cancel = threading.Event()

    @property
    def model_path(self) -> Path:
        value = os.getenv("REDESIGN_DIFFUSERS_TRANSFORMER", "").strip()
        return Path(value).expanduser().resolve() if value else DEFAULT_TRANSFORMER.resolve()

    @property
    def components_path(self) -> Path:
        value = os.getenv("REDESIGN_DIFFUSERS_COMPONENTS", "").strip()
        return Path(value).expanduser().resolve() if value else DEFAULT_COMPONENTS.resolve()

    @property
    def text_encoder_path(self) -> Path:
        value = os.getenv("REDESIGN_DIFFUSERS_TEXT_ENCODER", "").strip()
        return Path(value).expanduser().resolve() if value else DEFAULT_TEXT_ENCODER.resolve()

    def model_present(self) -> bool:
        return bool(
            self.model_path.is_file()
            and (self.components_path / "modular_model_index.json").is_file()
            and (self.text_encoder_path / "config.json").is_file()
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "ready": self._state == "ready",
                "model": str(self.model_path),
                "components": str(self.components_path),
                "text_encoder": str(self.text_encoder_path),
                "model_present": self.model_present(),
                "pipeline": type(self._pipeline).__name__ if self._pipeline is not None else None,
                "active_job": self._active_job,
                "error": self._error,
                "backend": "diffusers",
                "offload": os.getenv("REDESIGN_DIFFUSERS_OFFLOAD", "group"),
            }

    def load_async(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"loading", "ready", "running"}:
                return self.status()
            if not self.model_present():
                raise ValueError(
                    "The local FP8 transformer, Diffusers component tree, or Qwen-VL "
                    "text encoder is missing. Check REDESIGN_DIFFUSERS_* in .env."
                )
            self._state = "loading"
            self._error = None
        threading.Thread(target=self._load, name="redesign-diffusers-load", daemon=True).start()
        return self.status()

    def _load(self) -> None:
        with self._operation:
            offload_dir = tempfile.mkdtemp(prefix="redesign_diffusers_offload_")
            try:
                # This WebUI deliberately ignores the upstream agent/controller backend.
                # It constructs one QwenImageLayeredPipeline from local Diffusers
                # components and keeps only the active module on the GPU.
                os.environ["REDESIGN_QWEN_BACKEND"] = "native-fp8"
                os.environ["REDESIGN_QWEN_MODEL"] = str(self.model_path)
                os.environ["REDESIGN_QWEN_COMPONENTS"] = str(self.components_path)
                os.environ["REDESIGN_QWEN_TEXT_ENCODER_COMPONENTS"] = str(
                    self.text_encoder_path
                )
                os.environ["REDESIGN_QWEN_DTYPE"] = os.getenv(
                    "REDESIGN_DIFFUSERS_DTYPE", "bfloat16"
                )
                os.environ["REDESIGN_NATIVE_OFFLOAD"] = os.getenv(
                    "REDESIGN_DIFFUSERS_OFFLOAD", "group"
                )
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                os.environ.setdefault("DO_NOT_TRACK", "1")
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                from ReDesign.qwen_worker import _load_native_fp8_pipeline

                pipeline = _load_native_fp8_pipeline()
                with self._lock:
                    self._pipeline = pipeline
                    self._offload_dir = offload_dir
                    self._state = "ready"
                    self._error = None
            except Exception as exc:
                shutil.rmtree(offload_dir, ignore_errors=True)
                with self._lock:
                    self._pipeline = None
                    self._offload_dir = None
                    self._state = "error"
                    self._error = str(exc)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def unload_async(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"loading", "running"}:
                raise ValueError(f"Cannot unload while the model is {self._state}.")
            if self._state in {"unloaded", "unloading"}:
                return self.status()
            self._state = "unloading"
        threading.Thread(target=self._unload, name="redesign-diffusers-unload", daemon=True).start()
        return self.status()

    def _unload(self) -> None:
        with self._operation:
            with self._lock:
                pipeline = self._pipeline
                offload_dir = self._offload_dir
                self._pipeline = None
                self._offload_dir = None
            if pipeline is not None:
                try:
                    pipeline.remove_all_hooks()
                except (AttributeError, RuntimeError):
                    pass
                del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            if offload_dir:
                shutil.rmtree(offload_dir, ignore_errors=True)
            with self._lock:
                self._state = "unloaded"
                self._error = None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if self._active_job != job_id or self._state != "running":
                return False
            self._cancel.set()
            return True

    def decompose(
        self,
        *,
        job_id: str,
        image_path: Path,
        output_dir: Path,
        layers: int,
        steps: int,
        resolution: int,
        cfg: float,
        seed: int,
        log: Callable[[str], None],
    ) -> list[str]:
        with self._operation:
            with self._lock:
                if self._state != "ready" or self._pipeline is None:
                    raise RuntimeError("Load the local Diffusers model before decomposing an image.")
                pipeline = self._pipeline
                self._state = "running"
                self._active_job = job_id
                self._cancel.clear()

            try:
                with Image.open(image_path) as opened:
                    source = opened.convert("RGBA")
                original_size = source.size
                log(
                    f"Direct Diffusers run: {layers} layers, {steps} steps, "
                    f"{resolution}px, CFG {cfg:g}, seed {seed}."
                )

                def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    if self._cancel.is_set():
                        raise JobCancelled("Stopped by user.")
                    completed = step_index + 1
                    if completed == 1 or completed == steps or completed % 5 == 0:
                        log(f"Diffusion step {completed}/{steps}")
                    return callback_kwargs

                with torch.inference_mode():
                    result = pipeline(
                        image=source,
                        negative_prompt=" ",
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                        num_inference_steps=steps,
                        layers=layers,
                        resolution=resolution,
                        true_cfg_scale=cfg,
                        cfg_normalize=True,
                        use_en_prompt=True,
                        callback_on_step_end=on_step_end,
                    )
                generated = result.images[0]
                if not generated:
                    raise RuntimeError("The Diffusers pipeline returned no layers.")
                return write_layer_artifacts(
                    output_dir=output_dir,
                    source_size=original_size,
                    generated=generated,
                    model_path=self.model_path,
                    settings={
                        "layers": layers,
                        "steps": steps,
                        "resolution": resolution,
                        "true_cfg_scale": cfg,
                        "seed": seed,
                    },
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with self._lock:
                    self._active_job = None
                    self._cancel.clear()
                    if self._pipeline is not None:
                        self._state = "ready"


def write_layer_artifacts(
    *,
    output_dir: Path,
    source_size: tuple[int, int],
    generated: list[Image.Image],
    model_path: Path,
    settings: dict[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    composite = Image.new("RGBA", source_size, (0, 0, 0, 0))
    elements: list[dict[str, Any]] = []
    paths: list[str] = []
    bordered_layers: list[tuple[Image.Image, tuple[int, int, int, int] | None]] = []

    for index, layer in enumerate(generated):
        rgba = layer.convert("RGBA")
        if rgba.size != source_size:
            rgba = rgba.resize(source_size, Image.Resampling.LANCZOS)
        path = layer_dir / f"layer_{index:02d}.png"
        rgba.save(path)
        paths.append(str(path))
        composite = Image.alpha_composite(composite, rgba)
        bbox = rgba.getchannel("A").getbbox()
        bordered_layers.append((rgba, bbox))
        elements.append(
            {
                "id": f"qwen-layer-{index + 1}",
                "name": "Background" if index == 0 else f"Layer {index + 1}",
                "type": "image",
                "bbox": [0, 0, source_size[0], source_size[1]],
                "rendered_image_path": path.relative_to(output_dir).as_posix(),
                "alpha_bbox": list(bbox) if bbox else None,
                "z": index,
            }
        )

    composite.save(output_dir / "reconstructed.png")
    bordered = composite.copy()
    draw = ImageDraw.Draw(bordered)
    colors = ("#7dd3fc", "#a78bfa", "#fb7185", "#34d399", "#fbbf24", "#f472b6")
    for index, (_layer, bbox) in enumerate(bordered_layers):
        if bbox:
            draw.rectangle(bbox, outline=colors[index % len(colors)], width=max(2, min(source_size) // 300))
    bordered.save(output_dir / "reconstructed_bordered.png")
    (output_dir / "parse.json").write_text(
        json.dumps({"elements": elements}, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "layer_manifest.json").write_text(
        json.dumps(
            {
                "backend": "diffusers",
                "pipeline": "QwenImageLayeredPipeline",
                "model": str(model_path),
                "settings": settings,
                "layers": [Path(path).relative_to(output_dir).as_posix() for path in paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


runtime = DiffusersRuntime()
