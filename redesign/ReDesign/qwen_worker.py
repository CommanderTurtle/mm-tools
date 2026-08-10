# ReDesign/qwen_worker.py
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import signal
import socket
import tempfile
import time
import traceback
import uuid
from ipaddress import ip_address
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode, urlparse

import requests
import torch
from PIL import Image


def _detect_gpu_memory() -> Dict[str, Any]:
    """Return real device VRAM plus conservative CPU-offload limits."""
    gpu_count = torch.cuda.device_count()
    limits: dict[Any, str] = {}
    total_gb = 0.0
    gpu_headroom = float(os.environ.get("REDESIGN_GPU_HEADROOM_GIB", "3"))
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / (1024**3)
        total_gb += gib
        limits[index] = f"{max(1, int(gib - gpu_headroom))}GiB"

    if os.environ.get("REDESIGN_CPU_OFFLOAD_GIB"):
        cpu_gb = int(os.environ["REDESIGN_CPU_OFFLOAD_GIB"])
    else:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_gb = available_pages * page_size / (1024**3)
        reserve_gb = float(os.environ.get("REDESIGN_CPU_HEADROOM_GIB", "8"))
        cpu_gb = max(1, int(available_gb - reserve_gb))
    limits["cpu"] = f"{cpu_gb}GiB"
    return {"gpu_count": gpu_count, "total_gb": total_gb, "limit_mem_dict": limits}


def _local_endpoint(value: str) -> bool:
    host = urlparse(value).hostname
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ip_address(host).is_private or ip_address(host).is_loopback
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


def _checkpoint_quantization_methods(model_path: Path) -> set[str]:
    """Return quantizers declared by complete pipeline components."""
    methods: set[str] = set()
    for component in ("transformer", "text_encoder"):
        config_path = model_path / component / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        quantization = config.get("quantization_config")
        if isinstance(quantization, dict) and quantization.get("quant_method"):
            methods.add(str(quantization["quant_method"]).lower())
    return methods


def _load_diffusers_pipeline(offload_dir: str):
    """Load a complete Diffusers checkpoint (the supplied INT4 tree is valid)."""
    from diffusers import QwenImageLayeredPipeline

    gpu_info = _detect_gpu_memory()
    model_value = os.environ.get("REDESIGN_QWEN_MODEL", "").strip()
    if not model_value:
        raise RuntimeError("REDESIGN_QWEN_MODEL is unset for the Diffusers backend.")
    model_path = Path(model_value).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "model_index.json").is_file():
        raise RuntimeError(f"Complete local Diffusers checkpoint not found: {model_path}")

    quantization_methods = _checkpoint_quantization_methods(model_path)
    quantized = bool(quantization_methods)
    if "sdnq" in quantization_methods:
        try:
            import sdnq  # noqa: F401 - registers SDNQ with Diffusers/Transformers
        except ImportError as exc:
            raise RuntimeError(
                "This checkpoint is SDNQ-quantized, but sdnq is missing. "
                "Run ./uvsetup.sh in the ReDesign directory."
            ) from exc
    default_dtype = "float16" if "sdnq" in quantization_methods else "bfloat16"
    dtype_name = os.environ.get("REDESIGN_QWEN_DTYPE", default_dtype).strip().lower()
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtypes:
        raise RuntimeError(
            "REDESIGN_QWEN_DTYPE must be float16, bfloat16, or float32 "
            f"(got {dtype_name!r})."
        )
    compute_dtype = dtypes[dtype_name]
    allow_cpu_offload = os.environ.get("REDESIGN_ALLOW_CPU_OFFLOAD", "0") == "1"
    if not quantized and gpu_info["total_gb"] < 52 and not allow_cpu_offload:
        raise RuntimeError(
            f"The unquantized pipeline needs about 55 GiB; only {gpu_info['total_gb']:.1f} "
            "GiB of physical VRAM is visible. Use the Comfy INT8 backend, a complete "
            "quantized Diffusers tree, or explicitly permit CPU offload."
        )

    profiles = []
    if quantized:
        profiles.append(
            {
                "name": "/".join(sorted(quantization_methods)) + " GPU-resident",
                "kwargs": {"device_map": "cuda"},
            }
        )
    else:
        profiles.append(
            {
                "name": "balanced BF16",
                "kwargs": {"device_map": "balanced", "max_memory": None},
            }
        )
    if allow_cpu_offload:
        profiles.append(
            {
                "name": "GPU/RAM hybrid",
                "kwargs": {
                    "device_map": "balanced",
                    "max_memory": gpu_info["limit_mem_dict"],
                },
            }
        )

    last_error: Exception | None = None
    for profile in profiles:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            print(f"[QwenWorker] Loading Diffusers profile: {profile['name']}")
            print(f"[QwenWorker] Explicit compute dtype: {compute_dtype}")
            load_kwargs: dict[str, Any] = {
                "dtype": compute_dtype,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                **profile["kwargs"],
            }
            if "max_memory" in profile["kwargs"]:
                load_kwargs.update(
                    offload_folder=offload_dir,
                    offload_state_dict=True,
                )
            pipeline = QwenImageLayeredPipeline.from_pretrained(str(model_path), **load_kwargs)
            return pipeline
        except Exception as exc:  # keep the next explicitly enabled profile available
            last_error = exc
            print(f"[QwenWorker] {profile['name']} failed: {exc}")
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"All Diffusers loading profiles failed. Last error: {last_error}")


def _load_native_fp8_pipeline():
    """Load the mixed FP8/BF16 transformer without duplicating it in host RAM."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageLayeredPipeline,
        QwenImageTransformer2DModel,
    )
    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope
    from safetensors import safe_open
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

    weight_value = os.environ.get("REDESIGN_QWEN_MODEL", "").strip()
    component_value = os.environ.get("REDESIGN_QWEN_COMPONENTS", "").strip()
    text_encoder_value = os.environ.get("REDESIGN_QWEN_TEXT_ENCODER_COMPONENTS", "").strip()
    if not weight_value:
        raise RuntimeError("REDESIGN_QWEN_MODEL must point to the local FP8 safetensors file.")
    if not component_value:
        raise RuntimeError(
            "REDESIGN_QWEN_COMPONENTS must point to the complete local Diffusers component tree."
        )
    weights_path = Path(weight_value).expanduser().resolve()
    components_path = Path(component_value).expanduser().resolve()
    text_encoder_path = (
        Path(text_encoder_value).expanduser().resolve()
        if text_encoder_value
        else components_path / "text_encoder"
    )
    transformer_config = components_path / "transformer" / "config.json"
    if not weights_path.is_file() or weights_path.suffix != ".safetensors":
        raise RuntimeError(f"Native FP8 transformer not found: {weights_path}")
    component_index = components_path / "modular_model_index.json"
    if not component_index.is_file() or not transformer_config.is_file():
        raise RuntimeError(f"Complete local component tree not found: {components_path}")
    if not (text_encoder_path / "config.json").is_file():
        raise RuntimeError(f"Local Qwen-VL text encoder not found: {text_encoder_path}")

    dtype_name = os.environ.get("REDESIGN_QWEN_DTYPE", "bfloat16").strip().lower()
    compute_dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in compute_dtypes:
        raise RuntimeError("REDESIGN_QWEN_DTYPE must be float16 or bfloat16.")
    compute_dtype = compute_dtypes[dtype_name]

    config = json.loads(transformer_config.read_text(encoding="utf-8"))
    # The companion tree's transformer happens to be NF4. Its architecture is
    # canonical, but the downloaded checkpoint already contains real FP8/BF16
    # tensors and must not inherit a BitsAndBytes quantizer.
    config.pop("quantization_config", None)
    print("[QwenWorker] Constructing the native transformer on the meta device.")
    with init_empty_weights(include_buffers=True):
        transformer = QwenImageTransformer2DModel.from_config(config)
    # Rotary frequencies are ordinary tensor attributes rather than state-dict
    # buffers. Recreate that tiny table outside the meta-device context.
    transformer.pos_embed = QwenEmbedRope(
        theta=transformer.pos_embed.theta,
        axes_dim=transformer.pos_embed.axes_dim,
        scale_rope=transformer.pos_embed.scale_rope,
    )

    expected = set(transformer.state_dict())
    print(f"[QwenWorker] Streaming mixed FP8/BF16 weights from {weights_path.name}.")
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        actual = set(checkpoint.keys())
        if expected != actual:
            missing = sorted(expected - actual)[:8]
            extra = sorted(actual - expected)[:8]
            raise RuntimeError(
                "FP8 transformer architecture does not match its component config; "
                f"missing={missing}, extra={extra}"
            )
        for name in checkpoint.keys():
            value = checkpoint.get_tensor(name)
            # The native checkpoint intentionally mixes FP8 and BF16. Keep FP8
            # weights compressed and align its sensitive weights with the
            # selected execution dtype one tensor at a time.
            if value.dtype == torch.bfloat16 and compute_dtype == torch.float16:
                value = value.to(torch.float16)
            set_module_tensor_to_device(
                transformer,
                name,
                "cpu",
                value=value,
                dtype=value.dtype,
            )

    precision_patterns = ("pos_embed", "patch_embed", "norm", r"^proj_in$", r"^proj_out$")
    for module_name, module in transformer.named_modules():
        if any(re.search(pattern, module_name) for pattern in precision_patterns):
            module.to(dtype=compute_dtype)
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=compute_dtype,
    )
    print("[QwenWorker] Loading the remaining local pipeline components.")
    processor = Qwen2VLProcessor.from_pretrained(
        str(components_path / "processor"), local_files_only=True
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(components_path / "tokenizer"), local_files_only=True
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(components_path / "scheduler"), local_files_only=True
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        str(components_path / "vae"),
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_encoder_config = json.loads(
        (text_encoder_path / "config.json").read_text(encoding="utf-8")
    )
    text_quantizer = str(
        text_encoder_config.get("quantization_config", {}).get("quant_method", "")
    ).lower()
    text_dtype = torch.float16 if text_quantizer == "sdnq" else compute_dtype
    if text_quantizer == "sdnq":
        import sdnq  # noqa: F401 - registers the local INT4 text encoder
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(text_encoder_path),
        dtype=text_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    class NativeFP8LayeredPipeline(QwenImageLayeredPipeline):
        def get_image_caption(self, prompt_image, use_en_prompt=True, device=None):
            prompt = self.image_caption_prompt_en if use_en_prompt else self.image_caption_prompt_cn
            model_inputs = self.vl_processor(
                text=prompt,
                images=prompt_image,
                padding=True,
                return_tensors="pt",
            ).to(device=device, dtype=text_dtype)
            generated_ids = self.text_encoder.generate(**model_inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                output[len(source) :]
                for source, output in zip(model_inputs.input_ids, generated_ids)
            ]
            return self.vl_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

    if text_dtype != compute_dtype:
        encoder_class = type(text_encoder)

        class ComputeDtypeTextEncoder(encoder_class):
            @property
            def dtype(self):
                # Qwen's pipeline uses this property for every downstream
                # tensor. Captioning is explicitly cast back to text_dtype in
                # the override above.
                return compute_dtype

        text_encoder.__class__ = ComputeDtypeTextEncoder

    pipeline = NativeFP8LayeredPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    offload = os.environ.get("REDESIGN_NATIVE_OFFLOAD", "model").strip().lower()
    if offload == "model":
        pipeline.enable_model_cpu_offload()
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif offload != "none":
        raise RuntimeError("REDESIGN_NATIVE_OFFLOAD must be model, sequential, or none.")
    return pipeline


class ComfyLayeredPipeline:
    """Small local-only adapter for ComfyUI's official Qwen layered graph."""

    REQUIRED_NODES = {
        "LoadImage",
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "VAEEncode",
        "ReferenceLatent",
        "ModelSamplingAuraFlow",
        "EmptyQwenImageLayeredLatentImage",
        "KSampler",
        "LatentCutToBatch",
        "VAEDecode",
        "SaveImage",
    }

    def __init__(self) -> None:
        self.base_url = os.environ.get("REDESIGN_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
        if not _local_endpoint(self.base_url):
            raise RuntimeError("REDESIGN_COMFY_URL must resolve to loopback or a private-LAN host.")
        self.diffusion = os.environ.get(
            "REDESIGN_QWEN_DIFFUSION", "qwen_image_layered_int8convrot.safetensors"
        )
        self.text_encoder = os.environ.get(
            "REDESIGN_QWEN_TEXT_ENCODER", "qwen_2.5_vl_7b_fp8_scaled.safetensors"
        )
        self.vae = os.environ.get("REDESIGN_QWEN_VAE", "qwen_image_layered_vae.safetensors")
        self.timeout = float(os.environ.get("REDESIGN_COMFY_TIMEOUT_SECONDS", "3600"))
        self.session = requests.Session()
        self._preflight()

    def _preflight(self) -> None:
        try:
            response = self.session.get(f"{self.base_url}/object_info", timeout=8)
            response.raise_for_status()
            objects = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"ComfyUI is not reachable at {self.base_url}. Start the isolated backend first."
            ) from exc
        missing = sorted(self.REQUIRED_NODES.difference(objects))
        if missing:
            raise RuntimeError(
                "ComfyUI is too old for the official Qwen layered graph; missing: "
                + ", ".join(missing)
            )

        for node, field, expected in (
            ("UNETLoader", "unet_name", self.diffusion),
            ("CLIPLoader", "clip_name", self.text_encoder),
            ("VAELoader", "vae_name", self.vae),
        ):
            choices = (
                objects.get(node, {})
                .get("input", {})
                .get("required", {})
                .get(field, [[]])[0]
            )
            if isinstance(choices, list) and expected not in choices:
                raise RuntimeError(
                    f"ComfyUI cannot see {expected} for {node}. Run install-comfy-backend.sh "
                    "against this checkout, then restart ComfyUI."
                )

    @staticmethod
    def _fit_image(image: Image.Image, resolution: int) -> Image.Image:
        source = image.convert("RGBA")
        scale = min(1.0, resolution / max(source.size))
        width = max(16, int(source.width * scale) // 16 * 16)
        height = max(16, int(source.height * scale) // 16 * 16)
        if (width, height) == source.size:
            return source
        return source.resize((width, height), Image.Resampling.LANCZOS)

    def _upload(self, image: Image.Image) -> str:
        temporary = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
        image.save(temporary, format="PNG")
        temporary.seek(0)
        name = f"redesign-{uuid.uuid4().hex}.png"
        response = self.session.post(
            f"{self.base_url}/upload/image",
            files={"image": (name, temporary, "image/png")},
            data={"type": "input", "subfolder": "redesign", "overwrite": "true"},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        subfolder = str(result.get("subfolder", "")).strip("/")
        return f"{subfolder}/{result.get('name', name)}" if subfolder else str(result.get("name", name))

    def _graph(
        self,
        image_name: str,
        *,
        width: int,
        height: int,
        layers: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> dict[str, dict[str, Any]]:
        prompt = os.environ.get("REDESIGN_QWEN_PROMPT", "")
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "2": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": self.diffusion, "weight_dtype": "default"},
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": self.text_encoder,
                    "type": "qwen_image",
                    "device": "default",
                },
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": self.vae}},
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["3", 0]},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["3", 0]},
            },
            "7": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["1", 0], "vae": ["4", 0]},
            },
            "8": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["5", 0], "latent": ["7", 0]},
            },
            "9": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["6", 0], "latent": ["7", 0]},
            },
            "10": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"model": ["2", 0], "shift": 1.0},
            },
            "11": {
                "class_type": "EmptyQwenImageLayeredLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "layers": max(1, layers - 1),
                    "batch_size": 1,
                },
            },
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["10", 0],
                    "positive": ["8", 0],
                    "negative": ["9", 0],
                    "latent_image": ["11", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "13": {
                "class_type": "LatentCutToBatch",
                "inputs": {"samples": ["12", 0], "dim": "t", "slice_size": 1},
            },
            "14": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["13", 0], "vae": ["4", 0]},
            },
            "15": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["14", 0],
                    "filename_prefix": f"redesign/{uuid.uuid4().hex}",
                },
            },
        }

    def _wait(self, prompt_id: str) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            history = response.json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    messages = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI layered workflow failed: {messages}")
                images: list[dict[str, Any]] = []
                for output in history.get("outputs", {}).values():
                    images.extend(output.get("images", []))
                if images:
                    return images
            time.sleep(0.75)
        self.interrupt()
        raise TimeoutError(f"ComfyUI layered workflow exceeded {self.timeout:.0f} seconds")

    def _download(self, descriptor: dict[str, Any]) -> Image.Image:
        query = urlencode(
            {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            }
        )
        response = self.session.get(f"{self.base_url}/view?{query}", timeout=120)
        response.raise_for_status()
        temporary = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
        temporary.write(response.content)
        temporary.seek(0)
        return Image.open(temporary).convert("RGBA").copy()

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        prepared = self._fit_image(inputs["image"], int(inputs.get("resolution", 640)))
        uploaded = self._upload(prepared)
        graph = self._graph(
            uploaded,
            width=prepared.width,
            height=prepared.height,
            layers=int(inputs.get("layers", 4)),
            steps=int(inputs.get("num_inference_steps", 50)),
            cfg=float(inputs.get("true_cfg_scale", 4.0)),
            seed=int(inputs.get("seed", 777)),
        )
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": graph, "client_id": f"redesign-{uuid.uuid4().hex}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected the layered graph: {payload['node_errors']}")
        descriptors = self._wait(str(payload["prompt_id"]))
        return SimpleNamespace(images=[[self._download(item) for item in descriptors]])

    def interrupt(self) -> None:
        try:
            self.session.post(f"{self.base_url}/interrupt", timeout=5)
        except requests.RequestException:
            pass


def _load_pipeline(offload_dir: str):
    backend = os.environ.get("REDESIGN_QWEN_BACKEND", "native-fp8").strip().lower()
    if backend in {"native", "native-fp8", "fp8"}:
        return _load_native_fp8_pipeline(), "native Diffusers FP8"
    if backend in {"comfy", "comfyui"}:
        print("[QwenWorker] Using local ComfyUI TensorWise/ConvRot backend.")
        return ComfyLayeredPipeline(), "ComfyUI INT8"
    if backend == "diffusers":
        return _load_diffusers_pipeline(offload_dir), "Diffusers"
    raise RuntimeError("REDESIGN_QWEN_BACKEND must be native-fp8, comfyui, or diffusers.")


def worker_main(worker_id: str, physical_pair: Tuple[int, ...], in_q, out_q):
    """Dedicated layered-inference worker used by the upstream process pool."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_pair))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    offload_dir = tempfile.mkdtemp(prefix=f"qwen_offload_{worker_id}_")

    pipeline: Any = None

    def stop_worker(_signum, _frame):
        if hasattr(pipeline, "interrupt"):
            pipeline.interrupt()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)

    try:
        pipeline, backend_name = _load_pipeline(offload_dir)
        print(f"[{worker_id}] Qwen layered worker ready ({backend_name}).")
    except Exception as exc:
        out_q.put(
            {
                "worker_id": worker_id,
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )
        return

    try:
        while True:
            job = in_q.get()
            if job is None:
                break
            job_id = job["job_id"]
            try:
                image = Image.open(job["image_path"]).convert("RGBA")
                original_size = image.size
                seed = int(job.get("seed", 777))
                inputs = {
                    "image": image,
                    "negative_prompt": " ",
                    "generator": torch.Generator(device="cpu").manual_seed(seed),
                    "num_inference_steps": int(job.get("num_inference_steps", 50)),
                    "layers": int(job.get("num_layers", 4)),
                    "resolution": int(job.get("resolution", 640)),
                    "true_cfg_scale": float(job.get("true_cfg_scale", 4.0)),
                    "cfg_normalize": True,
                    "use_en_prompt": True,
                }
                if isinstance(pipeline, ComfyLayeredPipeline):
                    inputs["seed"] = seed
                with torch.inference_mode():
                    output_images = pipeline(**inputs).images[0]

                output_dir = Path(job["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                layer_paths: List[str] = []
                for index, layer in enumerate(output_images):
                    layer = layer.convert("RGBA")
                    if layer.size != original_size:
                        layer = layer.resize(original_size, Image.Resampling.LANCZOS)
                    path = output_dir / f"layer_{index:02d}.png"
                    layer.save(path)
                    layer_paths.append(str(path))
                out_q.put(
                    {
                        "worker_id": worker_id,
                        "job_id": job_id,
                        "ok": True,
                        "data": {"layer_images": layer_paths},
                    }
                )
            except Exception as exc:
                out_q.put(
                    {
                        "worker_id": worker_id,
                        "job_id": job_id,
                        "ok": False,
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    }
                )
            finally:
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    finally:
        if os.path.exists(offload_dir):
            shutil.rmtree(offload_dir)
