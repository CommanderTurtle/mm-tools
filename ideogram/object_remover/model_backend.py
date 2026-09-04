from __future__ import annotations

import gc
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.getenv("MULTIMEDIA_MODELS", "~/multimedia/models")).expanduser()
OBJECTCLEAR_SOURCE = Path(
    os.getenv("OBJECTCLEAR_SOURCE", str(ROOT / "vendor" / "ObjectClear"))
).expanduser()
OBJECTCLEAR_MODEL = Path(
    os.getenv("OBJECTCLEAR_MODEL", str(MODELS / "jixin0101--ObjectClear"))
).expanduser()
BIREFNET_MODEL = Path(
    os.getenv("BIREFNET_MODEL", str(MODELS / "ZhengPeng7--BiRefNet"))
).expanduser()


class ModelUnavailable(RuntimeError):
    pass


def _has_objectclear_model() -> bool:
    return (
        (OBJECTCLEAR_MODEL / "model_index.json").is_file()
        and (OBJECTCLEAR_MODEL / "unet" / "config.json").is_file()
        and any((OBJECTCLEAR_MODEL / "unet").glob("*.safetensors"))
    )


def _has_birefnet_model() -> bool:
    return (
        (BIREFNET_MODEL / "config.json").is_file()
        and any(BIREFNET_MODEL.glob("*.safetensors"))
    )


class EditingModels:
    """Lazy, process-local lifecycle for private editing models."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._objectclear: Any | None = None
        self._birefnet: Any | None = None
        self._device: str | None = None
        self._ideogram = None

    def ideogram(self):
        if self._ideogram is None:
            from .ideogram_edit import IdeogramEditing
            self._ideogram = IdeogramEditing()
        return self._ideogram

    def _torch(self):
        import torch

        if self._device is None:
            requested = os.getenv("OBJECT_REMOVER_DEVICE", "auto").strip().lower()
            if requested == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            elif requested in {"cuda", "cpu", "mps"}:
                self._device = requested
            else:
                raise ValueError("OBJECT_REMOVER_DEVICE must be auto, cuda, cpu, or mps")
        return torch

    def status(self) -> dict[str, Any]:
        with self._lock:
            torch = None
            free_vram = total_vram = None
            try:
                torch = self._torch()
                if self._device == "cuda" and torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    free_vram = round(free / 1024**3, 2)
                    total_vram = round(total / 1024**3, 2)
            except Exception:
                pass
            return {
                "device": self._device or "unknown",
                "vram_free_gib": free_vram,
                "vram_total_gib": total_vram,
                "engines": {
                    "ideogram": self.ideogram().engine.status(),
                    "objectclear": {
                        "loaded": self._objectclear is not None,
                        "available": _has_objectclear_model() and OBJECTCLEAR_SOURCE.is_dir(),
                        "model_path": str(OBJECTCLEAR_MODEL),
                        "source_path": str(OBJECTCLEAR_SOURCE),
                    },
                    "birefnet": {
                        "loaded": self._birefnet is not None,
                        "available": _has_birefnet_model(),
                        "model_path": str(BIREFNET_MODEL),
                    },
                },
            }

    def load(self, engine: str) -> dict[str, Any]:
        with self._lock:
            if engine == "ideogram":
                self.unload("all")
                self.ideogram().engine.start()
            elif engine == "objectclear":
                if self._ideogram:
                    self._ideogram.engine.stop()
                self._load_objectclear()
            elif engine == "birefnet":
                if self._ideogram:
                    self._ideogram.engine.stop()
                self._load_birefnet()
            else:
                raise ValueError(f"Unknown editing engine: {engine}")
            return self.status()

    def _load_objectclear(self) -> None:
        if self._objectclear is not None:
            return
        if not OBJECTCLEAR_SOURCE.is_dir():
            raise ModelUnavailable(
                f"ObjectClear source is missing at {OBJECTCLEAR_SOURCE}. Run ./setup-editing-sources.sh."
            )
        if not _has_objectclear_model():
            raise ModelUnavailable(
                f"ObjectClear FP16 weights are missing at {OBJECTCLEAR_MODEL}."
            )

        source = str(OBJECTCLEAR_SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)
        torch = self._torch()
        from objectclear.pipelines import ObjectClearPipeline

        fp16 = (OBJECTCLEAR_MODEL / "unet" / "diffusion_pytorch_model.fp16.safetensors").is_file()
        dtype = torch.float16 if self._device == "cuda" and fp16 else torch.float32
        variant = "fp16" if fp16 else None
        pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            str(OBJECTCLEAR_MODEL),
            torch_dtype=dtype,
            variant=variant,
            local_files_only=True,
            apply_attention_guided_fusion=os.getenv("OBJECTCLEAR_AGF", "1") != "0",
        )
        pipe.set_progress_bar_config(disable=True)
        if self._device == "cuda" and os.getenv("OBJECT_REMOVER_CPU_OFFLOAD", "0") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)
        self._objectclear = pipe

    def _load_birefnet(self) -> None:
        if self._birefnet is not None:
            return
        if not _has_birefnet_model():
            raise ModelUnavailable(f"BiRefNet weights are missing at {BIREFNET_MODEL}.")

        torch = self._torch()
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            str(BIREFNET_MODEL),
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to(self._device).eval()
        self._birefnet = model

    def unload(self, engine: str = "all") -> dict[str, Any]:
        with self._lock:
            if engine in {"ideogram", "all"} and self._ideogram:
                self._ideogram.engine.stop()
            if engine in {"objectclear", "all"}:
                self._objectclear = None
            if engine in {"birefnet", "all"}:
                self._birefnet = None
            if engine not in {"objectclear", "birefnet", "ideogram", "all"}:
                raise ValueError(f"Unknown editing engine: {engine}")
            gc.collect()
            try:
                torch = self._torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            return self.status()

    def edit_ideogram(self, image_bytes, mask_bytes, **options):
        with self._lock:
            self.unload("all")
            return self.ideogram().edit(image_bytes, mask_bytes, **options)

    def caption_ideogram(self, image_bytes, mask_bytes, *, instruction, caption_seed=None, **region_options):
        from .regions import prepare_region
        with self._lock:
            region = prepare_region(image_bytes, mask_bytes, **region_options)
            self.unload("all")
            return self.ideogram().caption(region, instruction, seed=caption_seed)

    def remove_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        *,
        steps: int,
        guidance: float,
        seed: int,
    ) -> bytes:
        with self._lock:
            self._load_objectclear()
            torch = self._torch()
            from objectclear.utils import resize_by_short_side

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
            original_size = image.size
            image = resize_by_short_side(image, 512, resample=Image.Resampling.BICUBIC)
            mask = resize_by_short_side(mask, 512, resample=Image.Resampling.NEAREST)
            generator = torch.Generator(device=self._device).manual_seed(seed)
            result = self._objectclear(
                prompt="remove the instance of object",
                image=image,
                mask_image=mask,
                generator=generator,
                num_inference_steps=max(4, min(steps, 60)),
                guidance_scale=max(0.0, min(guidance, 10.0)),
                height=image.height,
                width=image.width,
                return_attn_map=False,
            ).images[0]
            result = result.resize(original_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            result.save(output, format="PNG", optimize=True)
            return output.getvalue()

    def remove_background(self, image_bytes: bytes) -> bytes:
        with self._lock:
            self._load_birefnet()
            torch = self._torch()
            from torchvision import transforms
            from torchvision.transforms.functional import to_pil_image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = image.size
            transform = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            model_dtype = next(self._birefnet.parameters()).dtype
            input_tensor = transform(image).unsqueeze(0).to(
                device=self._device,
                dtype=model_dtype,
            )
            with torch.inference_mode():
                prediction = self._birefnet(input_tensor)[-1].sigmoid().cpu()[0].squeeze()
            alpha = to_pil_image(prediction).resize(original_size, Image.Resampling.LANCZOS)
            result = image.convert("RGBA")
            result.putalpha(alpha)
            output = io.BytesIO()
            result.save(output, format="PNG", optimize=True)
            return output.getvalue()

    def foreground_mask(self, image_bytes: bytes) -> bytes:
        """Optional alpha producer for the removal workbench; preserve the old route."""
        from .matte import refine_matte
        from .regions import check_size, png
        from PIL import ImageOps
        # Normalize EXIF once before the existing matting implementation sees it.
        with Image.open(io.BytesIO(image_bytes)) as image:
            check_size(image)
            normalized = png(ImageOps.exif_transpose(image).convert("RGB"))
        with self._lock:
            self.unload("all")
            try:
                alpha = self.remove_background(normalized)
                return refine_matte(normalized, alpha)
            finally:
                self.unload("birefnet")


models = EditingModels()
