"""Single-model Transformers inference with optional alpha-matte decoding."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from .config import Qwen3VLSegConfig
from .prompts import build_prompt

try:
    import torch
    import torch.nn.functional as F

    from .decoder import build_mask_decoder_from_qwen
    from .feature_bridge import (
        LastHiddenCapture,
        VisualFeatureCapture,
        assert_cnn_rgb,
        extract_t_seg,
        per_sample_premerge_feature_maps,
        premerge_grid_hw,
    )

    _inference_mode = torch.inference_mode
except ImportError:  # pragma: no cover - allows lightweight imports (prompts/visualization/mock demo)
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    def _inference_mode():  # type: ignore[misc]
        return lambda fn: fn


SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL_ID = "tencent/WeVisOne-VL-4B"
BOX_RE = re.compile(r"<box>(.*?)(?:</box>|$)", re.DOTALL | re.IGNORECASE)
COORD_RE = re.compile(r"<(\d{1,4})>")
SEG_RE = re.compile(r"<OBJ>", re.IGNORECASE)


@dataclass
class InferenceResult:
    response: str
    prompt: str
    boxes: list[list[float]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    latency_s: float = 0.0


def parse_seg_boxes(text: str) -> list[list[float]]:
    """Extract normalized xyxy boxes immediately followed by ``<OBJ>``."""
    text = (text or "").split("<|im_end|>", 1)[0]
    boxes: list[list[float]] = []
    for body in BOX_RE.findall(text):
        for prefix in SEG_RE.split(body)[:-1]:
            numbers = [int(value) for value in COORD_RE.findall(prefix)]
            if len(numbers) < 4:
                continue
            x1, y1, x2, y2 = numbers[-4:]
            if all(0 <= value <= 1000 for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                boxes.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return boxes


def _pil_to_float_rgb(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resolve_sidecar(model_name_or_path: str, revision: Optional[str]) -> str:
    local = Path(model_name_or_path).expanduser()
    if local.is_dir():
        sidecar = local / "mask_decoder.pt"
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing alpha decoder sidecar: {sidecar}")
        return str(sidecar)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for a remote model id") from exc
    return hf_hub_download(
        repo_id=model_name_or_path,
        filename="mask_decoder.pt",
        revision=revision,
    )


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


class WeVisOneVL:
    """Load the released VLM and matting decoder as one inference engine."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_ID,
        *,
        backend: str = "transformers",
        device: str = "auto",
        dtype: str = "bfloat16",
        revision: Optional[str] = None,
        min_pixels: int = 16 * 32 * 32,
        max_pixels: int = 2025 * 32 * 32,
        attention_implementation: str = "sdpa",
        decoder_batch_size: int = 16,
        gpu_memory_utilization: float = 0.45,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        tokenizer_mode: str = "slow",
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name_or_path = model_name_or_path
        self.backend = backend.strip().lower()
        if self.backend not in {"transformers", "vllm"}:
            raise ValueError(f"unsupported backend {backend!r}; choose 'transformers' or 'vllm'")
        self.device = _auto_device() if device == "auto" else torch.device(device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.decoder_batch_size = max(1, int(decoder_batch_size))

        sidecar_path = _resolve_sidecar(model_name_or_path, revision)
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        if not isinstance(sidecar, dict) or not isinstance(sidecar.get("state_dict"), dict):
            raise ValueError(f"invalid mask decoder sidecar: {sidecar_path}")
        self.cfg = Qwen3VLSegConfig.from_dict(sidecar.get("cfg"))

        self.vllm = None
        self._sampling_params_cls = None
        if self.backend == "vllm":
            from vllm import LLM, SamplingParams

            self._sampling_params_cls = SamplingParams
            self.vllm = LLM(
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                tokenizer_mode=tokenizer_mode,
                limit_mm_per_prompt={"image": 10, "video": 10},
                max_model_len=int(max_model_len),
                gpu_memory_utilization=float(gpu_memory_utilization),
                tensor_parallel_size=int(tensor_parallel_size),
                enable_prefix_caching=False,
                trust_remote_code=True,
                dtype=dtype,
            )

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            revision=revision,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if "<OBJ>" not in self.tokenizer.get_vocab():
            raise RuntimeError("released tokenizer does not contain the <OBJ> token")
        self.seg_token_id = int(self.tokenizer.convert_tokens_to_ids("<OBJ>"))
        saved_seg_id = sidecar.get("seg_token_id")
        if saved_seg_id is not None and int(saved_seg_id) != self.seg_token_id:
            raise RuntimeError(
                f"<OBJ> token mismatch: tokenizer={self.seg_token_id}, sidecar={saved_seg_id}"
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            revision=revision,
            dtype=self.dtype,
            attn_implementation=attention_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.mask_decoder = build_mask_decoder_from_qwen(self.model, self.cfg)
        incompatible = self.mask_decoder.load_state_dict(sidecar["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"mask decoder state mismatch: {incompatible}")
        self.mask_decoder.to(device=self.device, dtype=self.dtype).eval()
        self.spatial_merge_size = int(
            sidecar.get("spatial_merge_size")
            or self.model.config.vision_config.spatial_merge_size
        )
        self.cnn_size = int(getattr(self.cfg, "cnn_image_size", 1024) or 1024)

    def _messages(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def _inputs(self, messages: list[dict[str, Any]], image: Image.Image, generation: bool):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
        )
        return self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    @_inference_mode()
    def generate_text(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> str:
        if self.backend == "vllm" and self.vllm is not None:
            from qwen_vl_utils import process_vision_info

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            chat_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(messages)
            sampling = self._sampling_params_cls(
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                skip_special_tokens=False,
                stop=["<|im_end|>"],
            )
            outputs = self.vllm.generate(
                [{"prompt": chat_prompt, "multi_modal_data": {"image": image_inputs}}],
                sampling_params=sampling,
            )
            raw = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            return raw.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()

        inputs = self._inputs(self._messages(image, prompt), image, generation=True)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "use_cache": True,
            "do_sample": float(temperature) > 0,
        }
        if float(temperature) > 0:
            kwargs.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        sequences = self.model.generate(**inputs, **kwargs)
        new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()

    @_inference_mode()
    def generate_text_batch(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> list[str]:
        """Generate responses for an image batch.

        This is intentionally separate from :meth:`generate_text`: callers such
        as evaluation can batch the expensive VLM generation while the existing
        single-image API remains unchanged.
        """
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        if not images:
            return []
        rgb_images = [image.convert("RGB") for image in images]

        if self.backend == "vllm" and self.vllm is not None:
            from qwen_vl_utils import process_vision_info

            requests: list[dict[str, Any]] = []
            for image, prompt in zip(rgb_images, prompts):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "min_pixels": self.min_pixels,
                                "max_pixels": self.max_pixels,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                chat_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, _ = process_vision_info(messages)
                requests.append(
                    {"prompt": chat_prompt, "multi_modal_data": {"image": image_inputs}}
                )
            sampling = self._sampling_params_cls(
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                skip_special_tokens=False,
                stop=["<|im_end|>"],
            )
            outputs = self.vllm.generate(requests, sampling_params=sampling)
            return [
                (output.outputs[0].text if output.outputs else "")
                .split("<|im_end|>", 1)[0]
                .split("<|endoftext|>", 1)[0]
                .strip()
                for output in outputs
            ]

        messages = [self._messages(image, prompt) for image, prompt in zip(rgb_images, prompts)]
        texts = [
            self.processor.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in messages
        ]
        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.processor(
                text=texts,
                images=rgb_images,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = previous_padding_side
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "use_cache": True,
            "do_sample": float(temperature) > 0,
        }
        if float(temperature) > 0:
            kwargs.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        sequences = self.model.generate(**inputs, **kwargs)
        new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return [
            response.split("<|im_end|>", 1)[0]
            .split("<|endoftext|>", 1)[0]
            .strip()
            for response in responses
        ]

    def _masks_from_logits(self, logits: torch.Tensor, height: int, width: int) -> list[np.ndarray]:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.ndim == 2:
            logits = logits[None]
        if logits.ndim != 3:
            raise ValueError(f"unexpected mask logit shape: {tuple(logits.shape)}")
        probabilities = F.interpolate(
            logits.float().sigmoid()[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        return [row.cpu().numpy().astype(np.float32) for row in probabilities]

    @_inference_mode()
    def decode_masks(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
        boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> tuple[list[list[float]], list[np.ndarray]]:
        """Teacher-force ``response`` and decode one probability map per ``<OBJ>``.

        ``boxes`` (normalized xyxy) overrides the boxes parsed from ``response``,
        e.g. to condition the decoder on unrounded ground-truth boxes.
        """
        if boxes is None:
            boxes = parse_seg_boxes(response)
        else:
            boxes = [[float(v) for v in box] for box in boxes]
        if not boxes:
            return [], []
        messages = self._messages(image, prompt) + [{"role": "assistant", "content": response}]
        teacher = self._inputs(messages, image, generation=False)
        n_seg = int((teacher["input_ids"] == self.seg_token_id).sum().item())
        if n_seg <= 0:
            raise RuntimeError("the teacher-forced response contains no <OBJ> token")

        decoder_image = image.convert("RGB").resize(
            (self.cnn_size, self.cnn_size),
            resample=Image.Resampling.BICUBIC,
        )
        cnn_image = assert_cnn_rgb(
            _pil_to_float_rgb(decoder_image).to(self.device, dtype=self.dtype),
            cnn_image_size=self.cnn_size,
        )

        old_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
        try:
            with VisualFeatureCapture(
                self.model, feature_layers=self.cfg.vision_feature_layers
            ) as visual_capture:
                with LastHiddenCapture(self.model) as hidden_capture:
                    backbone_output = self.model(
                        **teacher,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
        finally:
            self.model.config.use_cache = old_use_cache

        hidden = hidden_capture.hidden
        if hidden is None:
            hidden = getattr(backbone_output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("model returned no last hidden state for mask decoding")
        t_seg = extract_t_seg(hidden, teacher["input_ids"], self.seg_token_id)[0]
        grid_thw = teacher.get("image_grid_thw")
        if grid_thw is None:
            raise RuntimeError("processor did not return image_grid_thw")
        grid_hw = premerge_grid_hw(grid_thw)[0]
        visual_maps = per_sample_premerge_feature_maps(
            visual_capture.premerge_features,
            self.cfg.vision_feature_layers,
            grid_thw,
            self.spatial_merge_size,
        )[0]
        visual_maps = [feature.to(self.device, dtype=self.dtype) for feature in visual_maps]

        count = min(len(boxes), int(t_seg.shape[0]))
        boxes_tensor = torch.tensor(boxes[:count], device=self.device, dtype=self.dtype)
        t_seg = t_seg[:count].to(self.device, dtype=self.dtype)
        masks: list[np.ndarray] = []
        for start in range(0, count, self.decoder_batch_size):
            end = min(count, start + self.decoder_batch_size)
            output = self.mask_decoder(
                image=cnn_image,
                visual_features_2d=visual_maps,
                t_seg=t_seg[start:end],
                boxes01=boxes_tensor[start:end],
                grid_hw=grid_hw,
            )
            masks.extend(self._masks_from_logits(output["mask_logits"], image.height, image.width))
        return boxes[:count], masks

    def predict(
        self,
        image: Image.Image | str | os.PathLike[str],
        *,
        task: str = "freeform",
        query: str | Sequence[str] = "",
        prompt: Optional[str] = None,
        reference_boxes: Optional[Sequence[Sequence[float]]] = None,
        decode_matting: bool = True,
        **generation_kwargs: Any,
    ) -> InferenceResult:
        """Run one image request and decode mattes whenever ``<OBJ>`` is emitted."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        actual_prompt = prompt or build_prompt(
            task,
            query,
            reference_boxes=reference_boxes,
            image_size=image.size,
        )
        started = time.perf_counter()
        response = self.generate_text(image, actual_prompt, **generation_kwargs)
        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        if decode_matting and SEG_RE.search(response):
            boxes, masks = self.decode_masks(image, actual_prompt, response)
        return InferenceResult(
            response=response,
            prompt=actual_prompt,
            boxes=boxes,
            masks=masks,
            latency_s=time.perf_counter() - started,
        )

    def predict_batch(
        self,
        images: Sequence[Image.Image | str | os.PathLike[str]],
        *,
        task: str = "freeform",
        query: str | Sequence[str] = "",
        prompt: Optional[str | Sequence[str]] = None,
        decode_matting: bool = True,
        **generation_kwargs: Any,
    ) -> list[InferenceResult]:
        """Run batched VLM generation and return one result per image.

        Dense masks are decoded per image after batched text generation because
        their number and spatial shape depend on each generated response.
        """
        loaded: list[Image.Image] = []
        for image in images:
            if isinstance(image, Image.Image):
                loaded.append(image.convert("RGB"))
            else:
                with Image.open(image) as source:
                    loaded.append(source.convert("RGB"))
        if not loaded:
            return []

        if prompt is None:
            actual_prompts = [
                build_prompt(task, query, image_size=image.size) for image in loaded
            ]
        elif isinstance(prompt, str):
            actual_prompts = [prompt] * len(loaded)
        else:
            actual_prompts = [str(item) for item in prompt]
            if len(actual_prompts) != len(loaded):
                raise ValueError("prompt sequence must have the same length as images")

        started = time.perf_counter()
        responses = self.generate_text_batch(
            loaded,
            actual_prompts,
            **generation_kwargs,
        )
        batch_latency = time.perf_counter() - started
        results: list[InferenceResult] = []
        for image, actual_prompt, response in zip(loaded, actual_prompts, responses):
            boxes: list[list[float]] = []
            masks: list[np.ndarray] = []
            if decode_matting and SEG_RE.search(response):
                boxes, masks = self.decode_masks(image, actual_prompt, response)
            results.append(
                InferenceResult(
                    response=response,
                    prompt=actual_prompt,
                    boxes=boxes,
                    masks=masks,
                    latency_s=batch_latency / len(loaded),
                )
            )
        return results
