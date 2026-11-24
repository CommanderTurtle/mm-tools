# SPDX-License-Identifier: Apache-2.0
"""Adapt the release AutoProcessor to SGLang's multimodal scheduler.

Raw media decoding, sampling, Mel extraction, and token expansion use the
checkpoint's WeMMProcessor. Image/video/audio offsets and visual-only MRoPE
are built here without importing APEX hooks or changing standard Qwen models.
"""

import asyncio
import base64
import copy
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from transformers import BatchFeature

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.wemm_omni import (
    WeMMVideoForConditionalGeneration,
    WeMMVideoMoeForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

PLACEHOLDERS = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "video": "<|vision_start|><|video_pad|><|vision_end|>",
    "audio": "<|audio_start|><|audio_pad|><|audio_end|>",
}
MEDIA_PATTERN = re.compile(
    "|".join(re.escape(value) for value in PLACEHOLDERS.values())
)


class SingleVideoPreprocessCache:
    """Optional one-video CPU cache; reuse media tensors, retokenize every prompt."""

    def __init__(self, processor, max_bytes=0):
        self.processor = processor
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.key = self.value = None

    def _key(self, routes, images_kwargs):
        if len(routes) != 1 or routes[0]["type"] not in ("video", "video_audio"):
            return None
        identities = []
        for name in ("video", "audio"):
            value = routes[0].get(name)
            if value is None:
                continue
            if not isinstance(value, (str, Path)):
                return None
            value = str(value)
            if value.startswith("file://"):
                value = unquote(urlparse(value).path)
            elif "://" in value:
                return None
            stat = Path(value).stat()
            identities.append((value, stat.st_dev, stat.st_ino, stat.st_size,
                               stat.st_mtime_ns, stat.st_ctime_ns))
        return json.dumps([routes, images_kwargs, identities], sort_keys=True, default=str)

    def __call__(self, prompt, routes, images_kwargs=None):
        def original(text):
            return self.processor(text=text, text_is_prompt=True, media=routes,
                                  images_kwargs=images_kwargs)

        key = self._key(routes, images_kwargs) if self.max_bytes > 0 else None
        matches = list(MEDIA_PATTERN.finditer(prompt))
        if key is None or len(matches) != 1 or matches[0].group() != PLACEHOLDERS["video"]:
            return original(prompt)
        with self.lock:
            if self.key != key:
                self.key = self.value = None
                value = original(PLACEHOLDERS["video"])
                size = sum(item.numel() * item.element_size() for item in value.values()
                           if isinstance(item, torch.Tensor))
                if size <= self.max_bytes:
                    self.key, self.value = key, value
            else:
                value = self.value
            source = self.processor.tokenizer.encode(prompt, add_special_tokens=False)
            pattern = self.processor.tokenizer.encode(PLACEHOLDERS["video"], add_special_tokens=False)
            starts = [i for i in range(len(source)) if source[i:i + len(pattern)] == pattern]
            if len(starts) != 1:
                raise ValueError("Expected one tokenized video placeholder")
            start = starts[0]
            ids = source[:start] + value["input_ids"][0].tolist() + source[start + len(pattern):]
            result = BatchFeature({name: tensor.clone() for name, tensor in value.items()
                                   if name not in ("input_ids", "attention_mask")})
            result["input_ids"] = torch.tensor([ids], dtype=torch.long)
            result["attention_mask"] = torch.ones(1, len(ids), dtype=torch.long)
            result.media_metadata = copy.deepcopy(value.media_metadata)
            result.prompt_text = prompt
            return result


def token_runs(ids, token):
    offsets = []
    index = 0
    while index < len(ids):
        if ids[index] != token:
            index += 1
            continue
        end = index + 1
        while end < len(ids) and ids[end] == token:
            end += 1
        offsets.append((index, end - 1))
        index = end
    return offsets


def mrope_positions(ids, items, config):
    """Only image/video pads use a spatial grid; bundled audio remains 1D."""
    merge = config.vision_config.spatial_merge_size
    grids = {config.image_token_id: [], config.video_token_id: []}
    for item in items:
        if item.is_image():
            for t, h, w in item.image_grid_thw.tolist():
                grids[config.image_token_id].append((t, h // merge, w // merge))
        elif item.is_video():
            for t, h, w in item.video_grid_thw.tolist():
                grids[config.video_token_id].extend([(1, h // merge, w // merge)] * t)
    cursors = dict.fromkeys(grids, 0)
    segments = []
    consumed = index = base = 0
    while index < len(ids):
        token = ids[index]
        if token not in grids:
            index += 1
            continue
        grid_index = cursors[token]
        if grid_index >= len(grids[token]):
            raise ValueError("Visual placeholders exceed the supplied grids")
        shape = grids[token][grid_index]
        count = int(np.prod(shape))
        if ids[index : index + count] != [token] * count:
            raise ValueError("Visual token count does not match its grid")
        text_length = index - consumed
        if text_length:
            segments.append(
                np.broadcast_to(np.arange(text_length), (3, text_length)) + base
            )
            base += text_length
        grid = np.indices(shape).reshape(3, -1)
        segments.append(grid + base)
        base += int(grid.max()) + 1
        index += count
        consumed = index
        cursors[token] += 1
    if any(cursors[token] != len(grid) for token, grid in grids.items()):
        raise ValueError("Supplied visual grids exceed the prompt placeholders")
    if consumed < len(ids):
        length = len(ids) - consumed
        segments.append(np.broadcast_to(np.arange(length), (3, length)) + base)
        base += length
    positions = (
        np.concatenate(segments, axis=1)
        if segments
        else np.empty((3, 0), dtype=np.int64)
    )
    return torch.from_numpy(positions.copy()).long(), torch.tensor(
        [[base - len(ids)]], dtype=torch.long
    )


def media_source(value):
    if hasattr(value, "url"):
        value = value.url
    if isinstance(value, dict) and "url" in value:
        value = value["url"]
    if isinstance(value, str) and value.startswith("data:"):
        header, payload = value.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("Media data URLs must use base64 encoding")
        return base64.b64decode(payload, validate=True)
    return value


def media_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def split_offsets(offsets, sizes):
    """Partition noncontiguous placeholder spans without dropping tail tokens."""
    cursor = 0
    current = offsets[0][0] if offsets else 0
    groups = []
    for size in sizes:
        group = []
        remaining = int(size)
        while remaining:
            if cursor >= len(offsets):
                raise ValueError("Media features exceed their placeholder offsets")
            start, end = offsets[cursor]
            start = max(start, current)
            take = min(remaining, end - start + 1)
            group.append((start, start + take - 1))
            remaining -= take
            current = start + take
            if current > end:
                cursor += 1
        groups.append(group)
    if cursor != len(offsets):
        raise ValueError("Media placeholders exceed their feature counts")
    return groups


class WeMMMultimodalProcessor(BaseMultimodalProcessor):
    models = [WeMMVideoForConditionalGeneration, WeMMVideoMoeForConditionalGeneration]

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        if processor.__class__.__name__ != "WeMMProcessor":
            raise ValueError(
                "Use the release checkpoint with AutoProcessor=processing_wemm.WeMMProcessor"
            )
        self.preprocess_cache = SingleVideoPreprocessCache(
            self._processor, int(os.environ.get("WEMM_PREPROCESS_CACHE_BYTES", "0"))
        )

    def compute_mrope_positions(self, input_ids, mm_items):
        return mrope_positions(list(input_ids), mm_items, self.hf_config)

    def prepare(self, image_data, audio_data, input_text, request_obj):
        if not isinstance(input_text, str):
            input_text = self._tokenizer.decode(
                input_text,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        inputs = {
            "image": media_list(image_data),
            "video": media_list(request_obj.video_data),
            "audio": media_list(audio_data),
        }
        cursors = dict.fromkeys(inputs, 0)
        reverse = {value: key for key, value in PLACEHOLDERS.items()}
        routes = []
        for match in MEDIA_PATTERN.finditer(input_text):
            kind = reverse[match.group()]
            if cursors[kind] >= len(inputs[kind]):
                raise ValueError(f"Missing {kind} data for a prompt placeholder")
            value = inputs[kind][cursors[kind]]
            cursors[kind] += 1
            route = {"type": kind, kind: media_source(value)}
            if kind == "audio" and isinstance(value, dict):
                if value.get("timestamp_interval_sec") is not None:
                    route["timestamp_interval_sec"] = value["timestamp_interval_sec"]
            if kind == "video":
                route.update(self.video_config)
                if request_obj.use_audio_in_video:
                    route["type"] = "video_audio"
                if isinstance(value, dict):
                    route.update(
                        {
                            key: value[key]
                            for key in (
                                "fps",
                                "num_frames",
                                "nframes",
                                "min_frames",
                                "max_frames",
                                "min_pixels",
                                "max_pixels",
                                "total_pixels",
                            )
                            if key in value
                            and value[key] is not None
                        }
                    )
                    if value.get("use_audio"):
                        route["type"] = "video_audio"
                    if "audio" in value:
                        route["type"] = "video_audio"
                        route["audio"] = media_source(value["audio"])
                    if value.get("uuid") is not None:
                        route["uuid"] = str(value["uuid"])
                # Match vLLM's ZIP deployment limit, independent of a client's
                # overridable raw-video max_frames default.
                route["wemm_zip_max_frames"] = min(
                    int(self.video_config.get("max_frames") or self._processor.max_frames),
                    int(self.video_config.get("max_allowed_frames") or
                        self.video_config.get("max_frames") or self._processor.max_frames))
                # Service hard limits are separate from request defaults.
                for key in ("max_allowed_frames", "max_allowed_total_pixels"):
                    if key in self.video_config:
                        route[key] = self.video_config[key]
            routes.append(route)
        if any(cursors[kind] != len(values) for kind, values in inputs.items()):
            raise ValueError("Media data and prompt placeholders differ")
        output = self.preprocess_cache(input_text, routes, self.image_config or None)
        ids = output["input_ids"][0].tolist()
        items = []
        for modality, feature, grid, token in (
            (
                Modality.IMAGE,
                "pixel_values",
                "image_grid_thw",
                self.hf_config.image_token_id,
            ),
            (
                Modality.VIDEO,
                "pixel_values_videos",
                "video_grid_thw",
                self.hf_config.video_token_id,
            ),
            (Modality.AUDIO, "audio_feature", None, self.hf_config.audio_token_id),
        ):
            offsets = token_runs(ids, token)
            if feature not in output:
                if offsets:
                    raise ValueError(f"Missing {feature} for expanded media tokens")
                continue
            if not offsets:
                raise ValueError(f"Missing media token offsets for {feature}")
            # Keep one cache identity per source media item, allowing image or
            # audio reuse when another media item in the request changes.
            if grid:
                grids = output[grid].cpu()
                feature_sizes = grids.prod(-1).tolist()
                token_sizes = [
                    size // self.hf_config.vision_config.spatial_merge_size**2
                    for size in feature_sizes
                ]
                metadata = [{grid: value.unsqueeze(0)} for value in grids]
            else:
                audio_metadata = [
                    item
                    for item in output.media_metadata
                    if item.get("audio_tokens", 0)
                ]
                feature_sizes = [item["segments"] for item in audio_metadata]
                token_sizes = [item["audio_tokens"] for item in audio_metadata]
                metadata = [{"wemm_media_metadata": item} for item in audio_metadata]
            spans = split_offsets(offsets, token_sizes)
            features = output[feature].cpu().split(feature_sizes, dim=0)
            for value, spans_for_item, data in zip(features, spans, metadata):
                items.append(
                    MultimodalDataItem(
                        modality=modality,
                        feature=value,
                        offsets=spans_for_item,
                        model_specific_data=data,
                    )
                )
        positions, delta = self.compute_mrope_positions(ids, items)
        return MultimodalProcessorOutput(
            mm_items=items,
            input_ids=ids,
            im_token_id=self.hf_config.image_token_id,
            im_start_id=self.hf_config.vision_start_token_id,
            im_end_id=self.hf_config.vision_end_token_id,
            video_token_id=self.hf_config.video_token_id,
            audio_token_id=self.hf_config.audio_token_id,
            mrope_positions=positions,
            mrope_position_delta=delta,
        )

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, audio_data=None, **kwargs
    ):
        # Decode and extract features outside the HTTP event loop. The model's
        # processor returns request-local tensors and never mutates global sampling.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.io_executor,
            self.prepare,
            image_data,
            audio_data,
            input_text,
            request_obj,
        )
