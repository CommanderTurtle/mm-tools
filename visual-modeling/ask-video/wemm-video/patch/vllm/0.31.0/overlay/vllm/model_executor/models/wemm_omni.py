# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WeMM Video 1.5: Qwen3.5 language/vision, Whisper audio, and Q-Former.

Contains the model, multimodal processing information, dummy input builder,
and image/video/audio processor. Token-layout helpers live in wemm_processor.
"""

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, WhisperConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput

# Qwen3VL config / video resize (for ProcessingInfo._get_vision_info)
from transformers.processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging as hf_logging
from transformers.video_utils import VideoInput

from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLProcessingInfo as _Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import AudioProcessorItems, ImageSize, MultiModalDataItems, MultiModalDataParser
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.multimodal.wemm_timestamps import resolve_timestamp_interval, timestamp_interval_option
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.utils.collection_utils import is_list_of

from .wemm_whisper import WeMMAudioEncoder

logger = init_logger(__name__)
_hf_logger = hf_logging.get_logger(__name__)


def merge_multimodal_embeddings(input_ids, inputs_embeds, embeddings, token_ids):
    """Use v0.31's mask-based merge with WeMM's modality token selection."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in token_ids:
        if token_id is not None:
            mask |= input_ids == token_id
    return _merge_multimodal_embeddings(inputs_embeds, embeddings, mask)


# ---------- In-file Qwen3VL Processor (adapted from omni processing_qwen3_vl)
# ----------
class Qwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    # oncehack
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": False},
    }


def _split_into_groups_3vl(counts, groups, second_per_grid_ts=None):
    """Split counts (audio token counts) into groups by temporal steps."""
    result = []
    if second_per_grid_ts is None:
        for count, g in zip(counts, groups):
            g = int(g)
            base = count // g
            remainder = count % g
            if remainder == 0:
                group_list = [base] * g
            else:
                group_list = [base] * g
                step = g / remainder
                for i in range(1, remainder + 1):
                    position = i * step
                    index = math.floor(position) - 1
                    if index >= g:
                        index = g - 1
                    group_list[index] += 1
            result.append(group_list)
    else:
        for count, g, second in zip(counts, groups, second_per_grid_ts):
            g = int(g)
            frame_idx = (torch.arange(g) * second * 2).long()
            per_grid_t = torch.diff(frame_idx)
            group_list = per_grid_t.tolist()
            group_list.append(count - sum(group_list))
            assert sum(group_list) == count
            result.append(group_list)
    return result


def _processor_calculate_timestamps(
    indices: list[int] | np.ndarray,
    video_fps: float,
    merge_size: int = 2,
):
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
    timestamps = [idx / video_fps for idx in indices]
    timestamps = [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2
        for i in range(0, len(timestamps), merge_size)
    ]
    return timestamps


class Qwen3VLProcessor(ProcessorMixin):
    """Qwen3VL processor with image/video/audio support. In-file implementation
    (no import from transformers.models.qwen3_vl). Load via from_pretrained
    which uses AutoProcessor to get subcomponents then builds this class."""

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = (
            "<|image_pad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.audio_token = (
            "<|audio_pad|>"
            if not hasattr(tokenizer, "audiotoken")
            else tokenizer.audio_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_token)
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.vision_start_token = (
            "<|vision_start|>"
            if not hasattr(tokenizer, "vision_start_token")
            else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>"
            if not hasattr(tokenizer, "vision_end_token")
            else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        """Load subcomponents via AutoProcessor then build this processor."""
        base = AutoProcessor.from_pretrained(model_name_or_path, *args, **kwargs)
        return cls(
            image_processor=base.image_processor,
            tokenizer=base.tokenizer,
            video_processor=base.video_processor,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audios=None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        import time

        per_video_sizes = kwargs.pop("wemm_video_sizes", None)
        per_video_metadata = kwargs.pop("wemm_video_metadata", None)
        start_time = time.time()
        use_audio, only_audio = False, False
        if audios is not None:
            use_audio = True
            if videos is None:
                only_audio = True

        # vLLM merges service defaults again at the HF call boundary. Resolve
        # shared defaults before Transformers rejects duplicate nested kwargs.
        if "images_kwargs" in kwargs or "videos_kwargs" in kwargs:
            image_options = dict(kwargs.get("images_kwargs", {}))
            video_options = dict(kwargs.get("videos_kwargs", {}))
            for key in ("size", "do_resize"):
                if key in kwargs:
                    value = kwargs.pop(key)
                    image_options.setdefault(key, value)
                    video_options.setdefault(key, value)
            for key in (
                "fps",
                "do_sample_frames",
                "num_frames",
                "min_frames",
                "max_frames",
                "return_metadata",
            ):
                if key in kwargs and key in video_options:
                    kwargs.pop(key)
            kwargs["images_kwargs"] = image_options
            kwargs["videos_kwargs"] = video_options

        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # print("output_kwargs",output_kwargs)
        # print("kwargs",kwargs)
        if images is not None:
            image_inputs = self.image_processor(
                images=images, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            if per_video_sizes is None:
                videos_inputs = self.video_processor(
                    videos=videos, **output_kwargs["videos_kwargs"]
                )
            else:
                if len(per_video_sizes) != len(videos):
                    raise ValueError("Video size count differs from video inputs")
                chunks = [self.video_processor(videos=[video], **{
                    **output_kwargs["videos_kwargs"], "size": size,
                }) for video, size in zip(videos, per_video_sizes)]
                videos_inputs = {}
                for key in chunks[0]:
                    values = [chunk[key] for chunk in chunks]
                    videos_inputs[key] = (torch.cat(values) if isinstance(values[0], torch.Tensor)
                                          else sum(values, []))
            video_grid_thw = videos_inputs["video_grid_thw"]  # [B, T', H', W']
            video_kwargs = output_kwargs["videos_kwargs"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * num_image_tokens, 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0  # Iterate over the videos.
            for i in range(len(text)):
                while self.video_token in text[i]:
                    if "fps" not in video_kwargs:
                        logger.warning_once(
                            "Missing video FPS; using 24 for frame timestamps."
                        )
                        video_kwargs["fps"] = 24

                    # if timestamps are not provided, calculate them
                    _fi = video_kwargs.get("frames_indices") or kwargs.get(
                        "frames_indices"
                    )
                    current_fps = video_kwargs["fps"]
                    if per_video_metadata is not None:
                        item_metadata = per_video_metadata[index]
                        _fi = item_metadata.get("frames_indices", _fi)
                        current_fps = item_metadata.get("fps", current_fps)
                    if _fi is None:
                        # Fallback: generate sequential indices
                        n_frames = (
                            int(video_grid_thw[index][0])
                            * self.video_processor.merge_size
                        )
                        _fi = list(range(n_frames))
                    curr_timestamp = self._calculate_timestamps(
                        _fi,
                        current_fps,
                        self.video_processor.merge_size,
                    )

                    # Build the video placeholder, starting with a timestamp such as <0.0.
                    # seconds> followed by vision_start, repeated video_pad, vision_end.
                    # * T'
                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token
                            + "<|placeholder|>" * frame_seqlen
                            + self.vision_end_token
                        )
                    if (
                        f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                        in text[i]
                    ):
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}",
                            video_placeholder,
                            1,
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(
                            self.video_token, video_placeholder, 1
                        )
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", None
        )
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        # @xzj add:

        audio_inputs = {}
        # Insert audio placeholders into video prompts.
        decoded_text = self.tokenizer.decode(
            text_inputs["input_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )  # Decode the text while preserving special tokens and whitespace.
        # print(f"generate_video: decoded_text=\n{decoded_text}")

        if use_audio:
            audio_lengths_list, audio_durations_list = (
                kwargs["audio_lengths_list"],
                kwargs["audio_durations_list"],
            )
            audio_list = audios
            if only_audio:
                # print(f"\n only audio: audio_lengths_list: {audio_lengths_list}")
                audio_length = audio_lengths_list[0]
                replacement = (
                    "<tool_call>" + "<|audio_pad|>" * audio_length + "<|vision_end|>"
                )
                decoded_text = replacement + "\n" + decoded_text

                input_ids = self.tokenizer(decoded_text)["input_ids"]
                input_ids = torch.tensor(input_ids).unsqueeze(0).long()

                text_inputs["input_ids"] = input_ids
                text_inputs["attention_mask"] = input_ids.ne(
                    self.tokenizer.pad_token_id
                ).long()
                audio_list = [
                    x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
                    for x in audio_list
                ]
                audio_inputs["audio_feature"] = torch.cat(audio_list, dim=0)
            else:
                tot_video_temper_cnt = 0  # Count the video segments.
                # Read the spatial merge size from the image processor.
                merge_size = getattr(self.image_processor, "merge_size", 2)
                video_grid_thw_list = (
                    video_grid_thw  # tensor([[t=nframes/2, h=height/32, w=width/32]])
                )

                second_per_grid_ts_list = []  # Store the duration of each video grid.
                for audio_duration in audio_durations_list:
                    # Get video metadata - try video_kwargs first, then kwargs
                    _total_num_frames = video_kwargs.get(
                        "total_num_frames"
                    ) or kwargs.get("total_num_frames", 64)
                    _frames_indices = video_kwargs.get("frames_indices") or kwargs.get(
                        "frames_indices"
                    )
                    _total_num_frames / video_kwargs["fps"]
                    # frame sampling fps (using audio duration to calculate)
                    if _frames_indices is not None:
                        sample_fps = len(_frames_indices) / audio_duration
                    else:
                        # Fallback: use total_num_frames / duration
                        sample_fps = _total_num_frames / audio_duration
                    second_per_grid_ts_list.append(
                        [self.image_processor.temporal_patch_size / sample_fps]
                    )

                # Process one video grid at a time.
                for grid_thw_video, audio_lengths, second_per_grid_ts in zip(
                    video_grid_thw_list, audio_lengths_list, second_per_grid_ts_list
                ):
                    # Split audio by duration and video grid using split_into_groups_3vl.
                    per_timestep_audio_len = _split_into_groups_3vl(
                        [audio_lengths], [grid_thw_video[0]], second_per_grid_ts
                    )[0]
                    assert sum(per_timestep_audio_len) % 60 == 0, (
                        f"{len(per_timestep_audio_len)}, f{text}"
                    )  # Check the audio length.
                    origin_string = (
                        "<|vision_start|>"
                        + "<|video_pad|>"
                        * (grid_thw_video[1] * grid_thw_video[2] // merge_size**2)
                        + "<|vision_end|>"
                    )  # Build the original video placeholder.
                    tot_video_temper_cnt += int(
                        grid_thw_video[0]
                    )  # 32 temporal patches for 64 frames.
                    for timestep in range(grid_thw_video[0]):  # t, h, w
                        # Interleave audio and video placeholders in the replacement string.
                        replacement = (
                            "<|vision_start|>"
                            + "<|video_pad|>"
                            * (grid_thw_video[1] * grid_thw_video[2] // merge_size**2)
                            + "<|audio_pad|>" * per_timestep_audio_len[timestep]
                            + "<|vision_end|>"
                        )
                        # print("decoded_text:", decoded_text)
                        assert origin_string in decoded_text, (
                            decoded_text
                        )  # Check that the original placeholder is in the decoded text.
                        decoded_text = decoded_text.replace(
                            origin_string, replacement, 1
                        )  # Replace the original placeholder.

                # Some inputs include four additional images.
                assert tot_video_temper_cnt == decoded_text.count(
                    "<|vision_start|><|video_pad|>"
                ), "Video temporal grid count does not match the prompt"

                # Encode the updated text.
                input_ids = self.tokenizer(decoded_text)["input_ids"]
                input_ids = (
                    torch.tensor(input_ids).unsqueeze(0).long()
                )  # Convert to a tensor and add a batch dimension.

                # Update the model inputs.
                text_inputs["input_ids"] = input_ids
                text_inputs["attention_mask"] = input_ids.ne(
                    self.tokenizer.pad_token_id
                ).long()  # Build the attention mask.
                audio_list = [
                    x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
                    for x in audio_list
                ]
                audio_inputs["audio_feature"] = torch.cat(
                    audio_list, dim=0
                )  # Concatenate the audio features.

            # Inspect the input IDs.
            debug_input_ids = text_inputs["input_ids"]
            audio_token_number = (
                debug_input_ids == self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
            ).sum()  # Count the audio placeholders.
            assert audio_token_number % 60 == 0, (
                f"Weird data: {audio_token_number}, {text}, {video_grid_thw}"
            )  # Check that the audio placeholder count is a multiple of 60.
        # Insert audio placeholders into video prompts.
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            mm_token_type_ids[array_ids == self.audio_token_id] = 3
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        # print("text_inputs",text_inputs)
        # print("videos_inputs",videos_inputs)
        # print("audio_inputs",audio_inputs)
        endtime = time.time()
        logger.debug("Processing took %.4f seconds", endtime - start_time)
        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the
        given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per
            each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Qwen3VLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = (
                images_kwargs.get("merge_size", None) or self.image_processor.merge_size
            )

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(
                    *image_size, images_kwargs
                )
                for image_size in image_sizes
            ]
            num_image_tokens = [
                (num_patches // merge_size**2) for num_patches in num_image_patches
            ]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        if video_sizes is not None:
            videos_kwargs = Qwen3VLProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(
                    *video_size, videos_kwargs
                )
                for video_size in video_sizes
            ]
            num_video_tokens = [
                (num_patches // merge_size**2) for num_patches in num_video_patches
            ]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to
                be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed
                to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to
                the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode
                method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _calculate_timestamps(
        self, indices: list[int] | np.ndarray, video_fps: float, merge_size: int = 2
    ):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        # @JJJYmmm frames are merged by self.merge_size, \
        # so we need to average the timestamps between the first/last frame within the
        # temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


# --- ProcessingInfo ---
class WeMMAudioProcessorItems(AudioProcessorItems):
    """Keep per-audio options in item hashes and processor-cache miss batches."""

    def get_audio_length(self, item_idx):
        return len(self.get(item_idx)["wemm_audio"])


class WeMMDataParser(Qwen2VLMultiModalDataParser):
    def _parse_audio_data(self, data):
        rows = data if isinstance(data, list) else [data]
        if not any(isinstance(row, dict) and "wemm_audio" in row for row in rows):
            return super()._parse_audio_data(data)
        sources = [row["wemm_audio"] if isinstance(row, dict) and "wemm_audio" in row else row for row in rows]
        parsed = super()._parse_audio_data(sources)
        return WeMMAudioProcessorItems([
            {"wemm_audio": value, "timestamp_interval_sec": timestamp_interval_option(row)
             if isinstance(row, dict) else None}
            for value, row in zip(parsed.get_all(), rows)
        ])


class WeMMProcessingInfo(_Qwen3VLProcessingInfo):
    """WeMM processing information with a local vision/audio processor."""

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        limits = super().get_mm_max_tokens_per_item(seq_len, mm_counts)
        # Complete video/audio can exceed the small startup profiling input.
        for modality in ("video", "audio"):
            if mm_counts.get(modality, 0):
                limits[modality] = max(limits.get(modality, 0), seq_len)
        return limits

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """Include audio so video-with-audio prompts are allowed (audio from video)."""
        return {**super().get_supported_mm_limits(), "audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        """Override parent's get_data_parser to include target_sr for audio resampling.
        video_needs_metadata=True is required so that video metadata (fps,
        frames_indices, audio_bytes) is preserved through the pipeline.
        """
        return WeMMDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            target_sr=16000,
        )

    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_video_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).video_processor

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_image_size_with_most_features(self, **kwargs) -> ImageSize:
        """Cap profiling image size while respecting parent's max_pixels.

        WeMM-Video-8B's preprocessor_config has longest_edge=16,777,216
        (16M pixels). Without capping, profiling creates 4096x4096 dummy
        images that OOM the ViT.

        When the parent (e.g. get_max_video_tokens) passes max_pixels,
        we forward to super() which handles it correctly. Only when called
        without max_pixels (profiling dummy path) do we inject our cap.
        """
        if "max_pixels" in kwargs:
            # Parent already provides a reasonable cap — delegate to it
            return super().get_image_size_with_most_features(**kwargs)
        # Profiling path: no max_pixels provided, inject our cap
        _PROFILING_CAP = 672 * 672  # ~450K pixels, safe for ViT profiling
        return super().get_image_size_with_most_features(
            max_pixels=_PROFILING_CAP, **kwargs
        )

    def get_second_per_window(self) -> float:
        """Read audio token window duration from model audio_config."""
        config = self.get_hf_config()
        audio_config = getattr(config, "audio_config", None)
        if audio_config is None:
            return 0.5
        if isinstance(audio_config, dict):
            spw = audio_config.get("second_per_window", 0.5)
        else:
            spw = getattr(audio_config, "second_per_window", 0.5)
        try:
            spw = float(spw)
        except (TypeError, ValueError):
            spw = 0.5
        return spw if spw > 0 else 0.5

    def get_audio_tokens_per_segment(self) -> int:
        """Get audio tokens per 30s segment from model audio_config.

        Dynamically computes the token count based on the WhisperEncoder's
        second_per_window/second_stride/query_length config. This replaces
        the hardcoded *60 formula which only works for second_per_window=0.5.

        Returns:
            Tokens per 30s segment (150 for spw=0.2, 60 for spw=0.5).
        """
        from .wemm_processor import compute_audio_tokens_per_segment

        config = self.get_hf_config()
        audio_config = getattr(config, "audio_config", None)
        if audio_config is None:
            return 60  # fallback for models without audio config
        spw = self.get_second_per_window()
        if isinstance(audio_config, dict):
            ss = audio_config.get("second_stride", spw)
            qf_cfg = audio_config.get("qformer_config", {})
            ql = qf_cfg.get("query_length", 1) if isinstance(qf_cfg, dict) else 1
        else:
            ss = getattr(audio_config, "second_stride", spw)
            qf_cfg = getattr(audio_config, "qformer_config", None)
            if qf_cfg is not None:
                if isinstance(qf_cfg, dict):
                    ql = qf_cfg.get("query_length", 1)
                else:
                    ql = getattr(qf_cfg, "query_length", 1)
            else:
                ql = 1
        return compute_audio_tokens_per_segment(spw, ss, ql)

    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int], **kwargs
    ) -> int:
        """Safety cap on max frames for profiling.

        Standard Qwen3VL parent handles profiling correctly with the model's
        preprocessor config. We just add a safety cap at 4096 frames to prevent
        edge cases where the parent returns an extremely large number.
        """
        num_frames = super().get_num_frames_with_most_features(
            seq_len, mm_counts, **kwargs
        )
        return min(num_frames, 64)

    def _calculate_timestamps(
        self, indices: list, video_fps: float, merge_size: int
    ) -> list:
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def _get_video_second_idx(
        self,
        metadata: dict,
        out_item: MultiModalKwargsItem,
        do_sample_frames: bool | None = None,
        sampled_fps: float | None = None,
    ) -> list:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)
        if do_sample_frames:
            video_fps = sampled_fps or video_processor.fps
            total_num_frames = metadata["total_num_frames"]
            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
            num_frames = min(
                min(
                    max(num_frames, video_processor.min_frames),
                    video_processor.max_frames,
                ),
                total_num_frames,
            )
            indices = (
                np.linspace(0, total_num_frames - 1, num_frames)
                .round()
                .astype(int)
                .tolist()
            )
        return self._calculate_timestamps(indices, video_fps, merge_size)


# --- DummyInputsBuilder ---
class WeMMDummyInputsBuilder(BaseDummyInputsBuilder[WeMMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"
        audio_token = "<|audio_start|><|audio_pad|><|audio_end|>"
        return image_token * num_images + video_token * num_videos + audio_token * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], *args, **kwargs
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )
        target_video_size, _ = self.info._get_vision_info(
            image_width=target_width,
            image_height=target_height,
            num_frames=target_num_frames,
            image_processor=self.info.get_video_processor(),
            mm_kwargs={},
        )
        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
            "video": self._get_dummy_videos(
                width=target_video_size.width,
                height=target_video_size.height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides=None,
    ) -> list[VideoItem]:
        num_frames = max(num_frames, 2)
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        return [
            (
                video.copy(),
                {
                    "fps": 2.0,
                    "duration": num_frames / 2.0,
                    "total_num_frames": num_frames,
                    "frames_indices": list(range(num_frames)),
                    "video_backend": "opencv",
                    "do_sample_frames": False,
                },
            )
            for _ in range(num_videos)
        ]

# --- MultiModalProcessor ---
class WeMMMultiModalProcessor(BaseMultiModalProcessor[WeMMProcessingInfo]):
    def __init__(self, info, dummy_inputs):
        super().__init__(info, dummy_inputs)
        self.processor = None

    def _get_hf_mm_text(self, mm_counts):
        # v0.31 encodes the real prompt in the renderer and only sends this
        # modality-matched dummy text through the HF media processor.
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def get_dummy_inputs(self, seq_len, mm_counts, mm_options):
        inputs = super().get_dummy_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {
            **inputs.hf_processor_mm_kwargs, "do_resize": False,
        }
        return inputs

    def apply(self, inputs, timing_ctx):
        # A processor is reused across requests, including all-cache-hit calls.
        # Do not let a previous fullbypass request alter a later normal request.
        self._full_bypass = any(
            isinstance(item, (list, tuple)) and len(item) == 2
            and isinstance(item[1], dict) and item[1].get("__full_bypass__", False)
            for items in inputs.mm_data_items.values()
            for item in items.get_all_items_for_hash()
        )
        return super().apply(inputs, timing_ctx)

    def _compute_audio_groups_from_tokens(self, input_ids, tokenizer, video_grid_thw):
        """Compute _audio_groups by scanning expanded token IDs.

        For each temporal frame (vision_start...vision_end region), count the
        number of audio_pad tokens. This gives us the per-frame audio distribution
        that matches the user's expanded token layout.

        For multi-video: uses -1 as separator between videos' groups.
        """
        hf_config = self.info.get_hf_config()
        vision_start_id = hf_config.vision_start_token_id
        vision_end_id = getattr(hf_config, "vision_end_token_id", vision_start_id)
        audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.info.get_image_processor().merge_size ** 2

        if video_grid_thw is None:
            return None

        # Compute expected video tokens per video
        if video_grid_thw.dim() == 1:
            video_grid_thw = video_grid_thw.unsqueeze(0)
        num_videos = video_grid_thw.shape[0]

        # Scan tokens: find vision_start/vision_end regions and count audio_pad per
        # region
        # Each "frame" is one vision_start...vision_end block
        frame_audio_counts = []  # audio_pad count per frame
        in_region = False
        frame_audio = 0
        frame_video = 0

        for tid in input_ids:
            if tid == vision_start_id:
                in_region = True
                frame_audio = 0
                frame_video = 0
            elif tid == vision_end_id:
                if in_region and frame_video:
                    frame_audio_counts.append(frame_audio)
                in_region = False
            elif tid == audio_pad_id and in_region:
                frame_audio += 1
            elif tid == hf_config.video_token_id and in_region:
                frame_video += 1

        if not frame_audio_counts:
            return None
        if len(frame_audio_counts) != int(video_grid_thw[:, 0].sum()):
            raise ValueError("Expanded video frame count differs from per-video grids")

        # Group frames by video using video_grid_thw
        # Each video has grid_t temporal frames
        all_groups = []
        frame_offset = 0
        for vid_idx in range(num_videos):
            grid_t = int(video_grid_thw[vid_idx, 0].item())
            vid_frames = frame_audio_counts[frame_offset : frame_offset + grid_t]
            frame_offset += grid_t
            all_groups.append(vid_frames)

        # Build the _audio_groups tensor
        if len(all_groups) == 1:
            groups = all_groups[0]
            if sum(groups) == 0:
                return None  # No audio tokens found
            return torch.tensor(groups, dtype=torch.long)
        else:
            # Multi-video: use -1 separator
            combined = []
            has_any_audio = False
            for i, groups in enumerate(all_groups):
                if i > 0:
                    combined.append(-1)
                combined.extend(groups)
                if sum(groups) > 0:
                    has_any_audio = True
            if not has_any_audio:
                return None
            return torch.tensor(combined, dtype=torch.long)

    def _build_preprocessed_result(
        self,
        videos_raw: list,
        prompt: str = "",
        tok_kwargs: Mapping = None,
    ) -> "BatchFeature":
        """Construct BatchFeature directly from pre-computed tensors (bypass mode).

        Supports ALL modality combinations via metadata keys:
            Video:  pixel_values_videos + video_grid_thw
            Image:  pixel_values + image_grid_thw
            Audio:  audio_feature (+ _audio_groups if bundled with video)

        These can be combined in a single request (e.g., video_audio_product).
        Multiple items in videos_raw are aggregated (multi-video, multi-image).
        """
        all_video_pv = []
        all_video_gt = []
        all_image_pv = []
        all_image_gt = []
        all_audio_features = []
        all_audio_groups = []
        video_audio_specs = []
        tokens_per_segment = self.info.get_audio_tokens_per_segment()

        for i, item in enumerate(videos_raw):
            _, meta = item

            # ---- Video tensors ----
            pv_vid = meta.get("pixel_values_videos")
            if pv_vid is not None:
                if isinstance(pv_vid, np.ndarray):
                    pv_vid = torch.from_numpy(pv_vid)
                all_video_pv.append(pv_vid.float())

                gt_vid = meta["video_grid_thw"]
                if isinstance(gt_vid, np.ndarray):
                    gt_vid = torch.from_numpy(gt_vid)
                if gt_vid.dim() == 2:
                    gt_vid = gt_vid.squeeze(0)
                all_video_gt.append(gt_vid.long())
                rows = gt_vid.unsqueeze(0) if gt_vid.dim() == 1 else gt_vid
                raw_groups = meta.get("_audio_groups")
                parts = _split_video_audio_groups(raw_groups, len(rows), rows[:, 0].tolist()) if raw_groups is not None else None
                feature = meta.get("audio_feature")
                if feature is None:
                    counts = [0] * len(rows)
                elif len(rows) == 1:
                    shape = feature.shape
                    counts = [1 if len(shape) == 2 else int(shape[1] if len(shape) == 4 else shape[0])]
                elif parts is not None:
                    if any(sum(part) % tokens_per_segment for part in parts):
                        raise ValueError("Precomputed video audio groups do not describe whole segments")
                    counts = [sum(part) // tokens_per_segment for part in parts]
                else:
                    raise ValueError("Multiple precomputed videos require explicit per-video audio groups")
                video_audio_specs.extend((int(row[0]), count, part) for row, count, part in zip(
                    rows, counts, parts if parts is not None else [None] * len(rows)))

            # ---- Image tensors ----
            pv_img = meta.get("pixel_values")
            if pv_img is not None:
                if isinstance(pv_img, np.ndarray):
                    pv_img = torch.from_numpy(pv_img)
                all_image_pv.append(pv_img.float())

                gt_img = meta["image_grid_thw"]
                if isinstance(gt_img, np.ndarray):
                    gt_img = torch.from_numpy(gt_img)
                # image_grid_thw can be (N, 3) for multi-image or (3,) for single
                if gt_img.dim() == 1:
                    gt_img = gt_img.unsqueeze(0)
                all_image_gt.append(gt_img.long())

            # ---- Audio tensors ----
            af = meta.get("audio_feature")
            if af is not None:
                if isinstance(af, np.ndarray):
                    af = torch.from_numpy(af)
                if af.dim() == 2:
                    af = af.unsqueeze(0)
                if af.dim() == 4 and af.shape[0] == 1:
                    af = af.squeeze(0)
                all_audio_features.append(af.float())

            ag = meta.get("_audio_groups")
            if ag is not None:
                if isinstance(ag, np.ndarray):
                    ag = torch.from_numpy(ag)
                all_audio_groups.append(ag.long())

        # ---- Assemble BatchFeature ----
        data = {}

        # Video
        if all_video_pv:
            data["pixel_values_videos"] = torch.cat(all_video_pv, dim=0)
            # Ensure all grid_thw are 2D before concatenating
            gt_2d = []
            for gt in all_video_gt:
                if gt.dim() == 1:
                    gt_2d.append(gt.unsqueeze(0))
                else:
                    gt_2d.append(gt)
            data["video_grid_thw"] = torch.cat(gt_2d, dim=0)
            # Shape validation
            expected = data["video_grid_thw"].prod(dim=-1).sum().item()
            actual = data["pixel_values_videos"].shape[0]
            if expected != actual:
                raise ValueError(
                    f"Bypass: pixel_values_videos.shape[0]={actual} != "
                    f"video_grid_thw.prod()={expected}"
                )

        # Store for _build_placeholders_from_expanded_tokens (avoid relying on mm_kwargs
        # items)
        self._bypass_video_grid_thw = data.get("video_grid_thw")
        self._bypass_num_audio_segments = (
            len(all_audio_features) if all_audio_features else 0
        )
        tokens_per_segment = self.info.get_audio_tokens_per_segment()
        self._bypass_audio_length_per_segment = (
            all_audio_features[0].shape[0] * tokens_per_segment
            if all_audio_features
            else 0
        )

        # Image
        if all_image_pv:
            data["pixel_values"] = torch.cat(all_image_pv, dim=0)
            data["image_grid_thw"] = torch.cat(all_image_gt, dim=0)
            # Shape validation
            expected = data["image_grid_thw"].prod(dim=-1).sum().item()
            actual = data["pixel_values"].shape[0]
            if expected != actual:
                raise ValueError(
                    f"Bypass: pixel_values.shape[0]={actual} != "
                    f"image_grid_thw.prod()={expected}"
                )

        # Audio
        second_per_window = self.info.get_second_per_window()
        if all_audio_features:
            data["audio_feature"] = torch.cat(all_audio_features, dim=0)
            total_audio_length = data["audio_feature"].shape[0] * tokens_per_segment
            data["_audio_lengths_list"] = [total_audio_length]
            data["_audio_durations_list"] = torch.tensor(
                [total_audio_length * second_per_window], dtype=torch.float32
            )

        if video_audio_specs and all_audio_features:
            from .wemm_processor import _split_audio_length_for_video

            # Keep the exact source-item segment mapping. A proportional split
            # by video duration cannot establish ownership of encoded audio.
            data["_video_audio_num_segments"] = torch.tensor(
                [count for _, count, _ in video_audio_specs], dtype=torch.long
            )
            parts = [part if part is not None else _split_audio_length_for_video(
                audio_length=count * tokens_per_segment, grid_t=grid_t,
                video_fps=1.0, temporal_patch_size=2, second_per_window=second_per_window,
            ) for grid_t, count, part in video_audio_specs]
            data["_audio_groups"] = torch.tensor(
                [value for part in parts for value in (*part, -1)], dtype=torch.long
            )
        elif all_audio_groups:
            if len(all_audio_groups) == 1:
                data["_audio_groups"] = all_audio_groups[0]
            else:
                combined = []
                for i, ag in enumerate(all_audio_groups):
                    if i > 0:
                        combined.append(torch.tensor([-1], dtype=torch.long))
                    combined.append(ag)
                data["_audio_groups"] = torch.cat(combined)

        logger.debug(
            "WeMM BYPASS result: video=%s, image=%s, audio=%s, _audio_groups=%s",
            data.get("pixel_values_videos", torch.empty(0)).shape
            if "pixel_values_videos" in data
            else None,
            data.get("pixel_values", torch.empty(0)).shape
            if "pixel_values" in data
            else None,
            data.get("audio_feature", torch.empty(0)).shape
            if "audio_feature" in data
            else None,
            data.get("_audio_groups", torch.empty(0)).shape
            if "_audio_groups" in data
            else None,
        )

        # Generate input_ids from prompt text (required by vLLM framework).
        if self.processor is None:
            self.processor = self.info.get_hf_processor()
        tokenizer = self.processor.tokenizer
        _tok_kwargs = dict(tok_kwargs) if tok_kwargs else {}
        _tok_kwargs.setdefault("add_special_tokens", False)
        input_ids = tokenizer(prompt, **_tok_kwargs)["input_ids"]
        data["input_ids"] = torch.tensor([input_ids], dtype=torch.long)

        return BatchFeature(data=data)

    # ====== FULL BYPASS: skip token expansion ======
    def _maybe_apply_prompt_updates(
        self, mm_items, mm_res
    ):
        """Override base class to support full bypass mode.

        When __full_bypass__=True, the user provides fully-expanded token IDs.
        We skip vLLM's pattern matching and directly scan for embedding positions.
        """
        prompt_ids = mm_res.prompt_ids
        mm_kwargs = mm_res.kwargs
        mm_prompt_updates = mm_res.prompt_updates
        full_bypass = getattr(self, "_full_bypass", False)
        if not full_bypass:
            # Secondary detection: check if prompt_ids are already expanded.
            # In compact mode: exactly 1 video_pad between each vision_start/end pair.
            # In expanded mode: many consecutive video_pad tokens (or audio_pad tokens).
            # Only trigger if there are MORE than 1 video_pad per video item.
            hf_config = self.info.get_hf_config()
            video_token_id = hf_config.video_token_id
            vision_start_id = hf_config.vision_start_token_id
            getattr(hf_config, "vision_end_token_id", vision_start_id)
            # Count video_pad tokens. Compact has exactly 1 per video item.
            video_pad_count = sum(1 for t in prompt_ids if t == video_token_id)
            # Count video items (number of vision_start tokens)
            vision_start_count = sum(1 for t in prompt_ids if t == vision_start_id)
            # Full bypass: more video_pad tokens than vision_start tokens
            # (expanded = many pads per frame), compact = exactly 1 pad per item
            if video_pad_count > vision_start_count and vision_start_count > 0:
                full_bypass = True
                logger.debug(
                    "WeMM FULL BYPASS: detected via expanded token count "
                    "(video_pads=%d > vision_starts=%d)",
                    video_pad_count,
                    vision_start_count,
                )
            elif video_pad_count == 0 and vision_start_count == 0:
                # Check image: expanded image has many image_pad tokens
                image_token_id = getattr(hf_config, "image_token_id", None)
                if image_token_id:
                    image_pad_count = sum(1 for t in prompt_ids if t == image_token_id)
                    if image_pad_count > 1:
                        full_bypass = True
                        logger.debug(
                            "WeMM FULL BYPASS: detected via image_pad count=%d",
                            image_pad_count,
                        )

        if full_bypass:
            self._full_bypass = (
                False  # Reset to avoid contaminating subsequent requests
            )
            logger.debug(
                "WeMM FULL BYPASS: scanning expanded tokens for placeholder positions"
            )

            # ----- Recompute _audio_groups from the user's actual expanded
            # prompt token layout. _build_preprocessed_result has already put
            # a provisional fps=1.0 estimate in `_audio_groups`, but it's
            # wrong whenever the user built the prompt with non-default fps.
            # Scanning prompt_ids directly is fps-agnostic and matches what
            # the model will see: each vision_start..vision_end region's
            # audio_pad count IS the per-frame audio_groups[t]. -----
            try:
                tokenizer = self.info.get_tokenizer()
                # Aggregate all video grid_thw across video items in mm_kwargs.
                _video_items = [copy(item) if item is not None else None
                                for item in mm_kwargs.get("video", [])]
                mm_kwargs["video"] = _video_items
                _grid_rows = []
                for _item in _video_items:
                    if _item is None:
                        continue
                    _gt_field = (
                        _item.get("video_grid_thw") if hasattr(_item, "get") else None
                    )
                    if _gt_field is None:
                        continue
                    _gt = _gt_field.data if hasattr(_gt_field, "data") else _gt_field
                    if isinstance(_gt, torch.Tensor):
                        if _gt.dim() == 1:
                            _grid_rows.append(_gt)
                        else:
                            for _row in _gt:
                                _grid_rows.append(_row)
                if _grid_rows:
                    _grid_stacked = torch.stack(_grid_rows, dim=0)
                    _scanned = self._compute_audio_groups_from_tokens(
                        prompt_ids,
                        tokenizer,
                        _grid_stacked,
                    )
                    if _scanned is not None:
                        _valid_items = [item for item in _video_items if item is not None]
                        _parts = _split_video_audio_groups(
                            _scanned, len(_valid_items), _grid_stacked[:, 0].tolist())
                        for _item, _part in zip(_valid_items, _parts):
                            _ag_field = _item.get("_audio_groups")
                            if _ag_field is None:
                                if sum(_part):
                                    raise ValueError("Expanded prompt references missing video audio")
                                continue
                            _counts = _item.get("_video_audio_num_segments")
                            if _counts is not None and sum(_part) != int(_counts.data) * self.info.get_audio_tokens_per_segment():
                                raise ValueError("Expanded prompt audio count differs from this video's feature slice")
                            # Do not mutate a field also held by the processor cache.
                            _new_field = copy(_ag_field)
                            _new_field.data = torch.tensor([*_part, -1], dtype=torch.long)
                            _item["_audio_groups"] = _new_field
                    else:
                        logger.debug(
                            "WeMM FULL BYPASS: scan returned None (no audio_pad in "
                            "vision regions?) — keeping provisional _audio_groups"
                        )
            except Exception as _e:
                raise ValueError(f"Invalid expanded video/audio layout: {_e}") from _e

            placeholders = self._build_placeholders_from_expanded_tokens(
                prompt_ids, mm_kwargs, mm_prompt_updates
            )
            # Log PlaceholderRange details for debugging consistency
            for modality, ph_list in placeholders.items():
                for ph in ph_list:
                    embed_count = (
                        int(ph.is_embed.sum()) if ph.is_embed is not None else "N/A"
                    )
                    logger.debug(
                        "WeMM FULL BYPASS PlaceholderRange: modality=%s, item=%d, "
                        "start_idx=%d, length=%d, embed_count=%s",
                        modality,
                        ph.item_idx,
                        ph.start_idx,
                        len(ph.tokens),
                        embed_count,
                    )
            return prompt_ids, placeholders
        result_ids, placeholders = super()._maybe_apply_prompt_updates(
            mm_items, mm_res
        )
        # Log PlaceholderRange for normal path too (for comparison)
        if placeholders:
            for modality, ph_list in placeholders.items():
                for ph in ph_list:
                    embed_count = (
                        int(ph.is_embed.sum()) if ph.is_embed is not None else "N/A"
                    )
                    logger.debug(
                        "WeMM NORMAL PlaceholderRange: modality=%s, item=%d, "
                        "start_idx=%d, length=%d, embed_count=%s",
                        modality,
                        ph.item_idx,
                        ph.start_idx,
                        len(ph.tokens),
                        embed_count,
                    )
        return result_ids, placeholders

    def _build_placeholders_from_expanded_tokens(
        self, prompt_ids, mm_kwargs, mm_prompt_updates
    ):
        """Scan pre-expanded token IDs to locate embedding regions.

        For video+audio: creates SEPARATE PlaceholderRanges for video and audio
        because they have different embedding dimensions (video: 1536, audio: 384).
        Both cover the same region but with different is_embed masks.
        """

        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_id
        image_token_id = getattr(hf_config, "image_token_id", None)
        vision_start_id = hf_config.vision_start_token_id
        vision_end_id = getattr(hf_config, "vision_end_token_id", vision_start_id)
        tokenizer = self.info.get_tokenizer()
        audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

        image_processor = self.info.get_image_processor()
        merge_length = image_processor.merge_size**2

        result = {}

        # --- Video placeholders (video_token_id only) ---
        if "video" in mm_prompt_updates:
            num_videos = len(mm_prompt_updates["video"])

            expected_video_embeds = []
            expected_audio_embeds = []
            video_items = list(mm_kwargs.get("video", []))
            for item in video_items:
                if item is None or "video_grid_thw" not in item:
                    raise ValueError("Missing per-video grid metadata for expanded prompt")
                expected_video_embeds.append(int(item["video_grid_thw"].data.prod()) // merge_length)
                groups = item.get("_audio_groups")
                expected_audio_embeds.append(
                    sum(_split_video_audio_groups(groups.data, 1)[0]) if groups is not None else 0)
            if len(expected_video_embeds) != num_videos:
                raise ValueError("Expanded video item count differs from cached metadata")
            has_bundled_audio = any(expected_audio_embeds)

            # When audio is bundled with video, model's embed_multimodal returns
            # ONE interleaved tensor per video containing both video+audio embeddings.
            # So the PlaceholderRange must include BOTH video_pad AND audio_pad tokens.
            if has_bundled_audio:
                # Count audio_pad tokens per video from expanded prompt
                audio_per_video = self._count_audio_per_video(
                    prompt_ids,
                    num_videos,
                    video_token_id,
                    audio_pad_id,
                    vision_start_id,
                    vision_end_id,
                    expected_video_embeds,
                )
                if audio_per_video != expected_audio_embeds:
                    raise ValueError(f"Expanded prompt audio placeholders {audio_per_video} differ from per-video features {expected_audio_embeds}")
                # Add audio tokens to expected embeds
                expected_combined = [
                    v + a for v, a in zip(expected_video_embeds, audio_per_video)
                ]
                video_embed_ids = {video_token_id, audio_pad_id}
                logger.debug(
                    "WeMM FULL BYPASS _build_placeholders: num_videos=%d, "
                    "video_embeds=%s, audio_per_video=%s, combined=%s",
                    num_videos,
                    expected_video_embeds,
                    audio_per_video,
                    expected_combined,
                )
            else:
                expected_combined = expected_video_embeds
                video_embed_ids = {video_token_id}
                logger.debug(
                    "WeMM FULL BYPASS _build_placeholders: num_videos=%d, "
                    "expected_video_embeds=%s, has_audio=%s",
                    num_videos,
                    expected_video_embeds,
                    has_bundled_audio,
                )

            video_placeholders = self._scan_modality_placeholders(
                prompt_ids,
                num_videos,
                video_embed_ids,
                expected_combined,
                "video",
                region_start_id=vision_start_id,
                region_end_id=vision_end_id,
            )
            if video_placeholders:
                result["video"] = video_placeholders
            logger.debug(
                "WeMM FULL BYPASS: video placeholders=%d, expected_embeds=%s",
                len(video_placeholders) if video_placeholders else 0,
                expected_combined,
            )

        # --- Image placeholders ---
        if "image" in mm_prompt_updates and image_token_id is not None:
            num_images = len(mm_prompt_updates["image"])
            image_items = list(mm_kwargs.get("image", []))

            expected_embeds_per_image = []
            for item in image_items:
                grid_thw = item["image_grid_thw"].data
                n_image = int(grid_thw.prod()) // merge_length
                expected_embeds_per_image.append(n_image)

            image_placeholders = self._scan_modality_placeholders(
                prompt_ids,
                num_images,
                {image_token_id},
                expected_embeds_per_image,
                "image",
                region_start_id=vision_start_id,
                region_end_id=vision_end_id,
            )
            if image_placeholders:
                result["image"] = image_placeholders
            logger.debug(
                "WeMM FULL BYPASS: image placeholders=%d, expected=%s",
                len(image_placeholders) if image_placeholders else 0,
                expected_embeds_per_image,
            )

        # --- Standalone audio placeholders (not bundled with video) ---
        if "audio" in mm_prompt_updates and "audio" not in result:
            num_audios = len(mm_prompt_updates["audio"])
            audio_placeholders = self._scan_modality_placeholders(
                prompt_ids,
                num_audios,
                {audio_pad_id},
                None,
                "audio",
                region_start_id=audio_start_id,
                region_end_id=audio_end_id,
            )
            if audio_placeholders:
                result["audio"] = audio_placeholders
            logger.debug(
                "WeMM FULL BYPASS: standalone audio placeholders=%d",
                len(audio_placeholders) if audio_placeholders else 0,
            )

        return result

    def _scan_modality_placeholders(
        self,
        prompt_ids,
        num_items,
        embed_token_ids,
        expected_embeds_per_item,
        modality,
        region_start_id,
        region_end_id,
    ):
        """Scan token IDs to find placeholder regions for a single modality.

        Strategy:
        - Find regions bounded by region_start_id ... region_end_id
        - Within each region, count embed tokens (tokens in embed_token_ids)
        - Group regions into items based on expected_embeds_per_item counts
        - Handle cases where one item spans multiple start/end regions (timestamps)

        For video: each item's region contains multiple <vision_start>...<vision_end>
        sub-regions (one per frame), all grouped into a single PlaceholderFeaturesInfo.
        The full region starts from the first relevant token (timestamp before first
        vision_start) to the last vision_end.
        """
        from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo

        placeholders = []
        n = len(prompt_ids)

        # Find all embed token positions
        embed_positions = []
        for i, tid in enumerate(prompt_ids):
            if tid in embed_token_ids:
                embed_positions.append(i)

        if not embed_positions:
            return placeholders

        if expected_embeds_per_item is None:
            # Auto-split: divide all embed tokens equally among items
            total_embeds = len(embed_positions)
            per_item = total_embeds // max(num_items, 1)
            expected_embeds_per_item = [per_item] * num_items
            # Adjust last item for remainder
            if num_items > 0:
                expected_embeds_per_item[-1] = total_embeds - per_item * (num_items - 1)

        # Group embed positions by item using expected counts
        pos_idx = 0
        for item_idx in range(num_items):
            if pos_idx >= len(embed_positions):
                break
            expected = expected_embeds_per_item[item_idx]
            if expected <= 0:
                continue

            # Collect embed positions for this item
            item_embed_start = pos_idx
            item_embed_end = min(pos_idx + expected, len(embed_positions))
            item_positions = embed_positions[item_embed_start:item_embed_end]

            if not item_positions:
                continue

            # Determine the full region boundaries
            # Expand backward from first embed token to find region start
            first_embed_pos = item_positions[0]
            last_embed_pos = item_positions[-1]

            # Find region start: scan backward for region_start_id or a timestamp
            region_start = first_embed_pos
            for j in range(first_embed_pos - 1, -1, -1):
                if prompt_ids[j] == region_start_id:
                    # Continue scanning backward to include timestamps before
                    # vision_start
                    region_start = j
                    # Check if there's a timestamp token before this
                    # (timestamps are usually 3-5 tokens before vision_start)
                    break
                elif j < first_embed_pos - 50:
                    # Don't scan too far back
                    break
                else:
                    region_start = j + 1

            # For video/image: include the timestamp tokens before vision_start.
            # In the expanded format: <timestamp><vision_start><pads><vision_end>
            # The timestamp is typically 3-6 tokens before vision_start.
            # Strategy: find the vision_start token closest before/at region_start,
            # then scan backward from there to include timestamp tokens.
            # Stop at: newline, im_start, im_end, another vision_end, or another embed
            # token.
            if modality in ("video", "image"):
                # Find the first vision_start at or before current region_start
                vs_pos = region_start
                for j in range(region_start, max(region_start - 5, -1), -1):
                    if prompt_ids[j] == region_start_id:
                        vs_pos = j
                        break

                # Scan backward from vision_start to find timestamp start
                # Timestamps are "0.5 seconds " → ~3-5 tokens before vision_start
                # Stop at boundary tokens
                tokenizer = self.info.get_tokenizer()
                im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                stop_tokens = {region_end_id, im_start_id, im_end_id}
                stop_tokens.update(embed_token_ids)
                # Also stop at newline — but get the correct newline token
                # For Qwen tokenizer: try both "\n" and the encoded version
                for nl_candidate in ["\n", "\r\n"]:
                    nl_ids = tokenizer.encode(nl_candidate, add_special_tokens=False)
                    for nid in nl_ids:
                        stop_tokens.add(nid)

                new_start = vs_pos
                for j in range(vs_pos - 1, max(vs_pos - 15, -1), -1):
                    tid = prompt_ids[j]
                    if tid in stop_tokens:
                        break
                    new_start = j
                region_start = new_start

            # Find region end: scan forward from last embed token for region_end_id
            region_end = last_embed_pos + 1
            for j in range(last_embed_pos + 1, min(n, last_embed_pos + 10)):
                region_end = j + 1
                if prompt_ids[j] == region_end_id:
                    break

            # Log boundary tokens for debugging
            logger.debug(
                "WeMM FULL BYPASS region: start=%d, end=%d, "
                "first_embed=%d, last_embed=%d, "
                "tokens_at_start=%s, tokens_at_end=%s",
                region_start,
                region_end,
                first_embed_pos,
                last_embed_pos,
                prompt_ids[region_start : min(region_start + 5, n)],
                prompt_ids[max(0, region_end - 5) : region_end],
            )

            # Extract region tokens and build is_embed mask
            region_tokens = prompt_ids[region_start:region_end]
            is_embed = torch.tensor(
                [t in embed_token_ids for t in region_tokens], dtype=torch.bool
            )

            placeholders.append(
                PlaceholderFeaturesInfo(
                    modality=modality,
                    item_idx=item_idx,
                    start_idx=region_start,
                    tokens=region_tokens,
                    is_embed=is_embed,
                )
            )

            pos_idx = item_embed_end

        return placeholders

    def _count_audio_per_video(
        self,
        prompt_ids,
        num_videos,
        video_token_id,
        audio_pad_id,
        vision_start_id,
        vision_end_id,
        expected_video_embeds,
    ):
        """Count audio_pad tokens per video in expanded token sequence.

        The expanded format interleaves video_pad and audio_pad within
        vision_start/vision_end regions. We use the cumulative video_token
        count to determine which video each audio_pad belongs to.
        """
        audio_per_video = [0] * num_videos
        current_video = 0
        video_count_so_far = 0
        in_region = False
        frame_video = frame_audio = 0

        for tid in prompt_ids:
            if tid == vision_start_id:
                in_region = True
                frame_video = frame_audio = 0
            elif tid == vision_end_id:
                if in_region and frame_video:
                    if current_video >= num_videos:
                        raise ValueError("Too many video regions in expanded prompt")
                    video_count_so_far += frame_video
                    audio_per_video[current_video] += frame_audio
                    if video_count_so_far > expected_video_embeds[current_video]:
                        raise ValueError("Expanded video region exceeds its item's visual tokens")
                    if video_count_so_far == expected_video_embeds[current_video]:
                        video_count_so_far = 0
                        current_video += 1
                in_region = False
            elif in_region and tid == video_token_id:
                frame_video += 1
            elif in_region and tid == audio_pad_id:
                frame_audio += 1
        if current_video != num_videos or video_count_so_far:
            raise ValueError("Expanded prompt has incomplete per-video visual tokens")

        logger.debug("WeMM FULL BYPASS _count_audio_per_video: %s", audio_per_video)
        return audio_per_video

    # ====== END FULL BYPASS ======

    def _call_hf_processor(self, hf_data, hf_kwargs):
        mm_data = dict(hf_data)
        prompt = mm_data.pop("text", "")
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise ValueError("WeMM expects one processor prompt")
            prompt = prompt[0]
        return self._call_wemm_hf_processor(prompt, mm_data, hf_kwargs, {})

    def _call_wemm_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if self.processor is None:
            self.processor = self.info.get_hf_processor()

        # vLLM 0.17+ passes videos as list of (array, metadata) tuples
        # We need to extract the video arrays and metadata for our processor
        mm_data = dict(mm_data)

        # ====== PRE-PROCESSED BYPASS ======
        # Check video channel for bypass sentinel
        _bypass_videos = mm_data.get("videos") or mm_data.get("video")
        if _bypass_videos and len(_bypass_videos) > 0:
            _first_item = _bypass_videos[0]
            if isinstance(_first_item, list | tuple) and len(_first_item) == 2:
                _, _first_meta = _first_item
                if isinstance(_first_meta, dict) and _first_meta.get(
                    "__preprocessed__"
                ):
                    self._full_bypass = bool(_first_meta.get("__full_bypass__", False))
                    logger.debug(
                        (
                            "WeMM _call_hf_processor: BYPASS MODE (pre-processed tens"
                            "ors, full_bypass=%s)"
                        ),
                        self._full_bypass,
                    )
                    return self._build_preprocessed_result(
                        _bypass_videos, prompt=prompt, tok_kwargs=tok_kwargs
                    )
        # Check image channel for bypass sentinel
        _bypass_images = mm_data.get("images") or mm_data.get("image")
        if _bypass_images and len(_bypass_images) > 0:
            _first_item = _bypass_images[0]
            if isinstance(_first_item, list | tuple) and len(_first_item) == 2:
                _, _first_meta = _first_item
                if isinstance(_first_meta, dict) and _first_meta.get(
                    "__preprocessed__"
                ):
                    self._full_bypass = bool(_first_meta.get("__full_bypass__", False))
                    logger.debug(
                        (
                            "WeMM _call_hf_processor: BYPASS MODE (image/audio pre-pr"
                            "ocessed, full_bypass=%s)"
                        ),
                        self._full_bypass,
                    )
                    return self._build_preprocessed_result(
                        _bypass_images, prompt=prompt, tok_kwargs=tok_kwargs
                    )
        # Check audio channel for bypass sentinel
        _bypass_audios = mm_data.get("audios") or mm_data.get("audio")
        if _bypass_audios and len(_bypass_audios) > 0:
            _first_item = _bypass_audios[0]
            if isinstance(_first_item, list | tuple) and len(_first_item) == 2:
                _, _first_meta = _first_item
                if isinstance(_first_meta, dict) and _first_meta.get(
                    "__preprocessed__"
                ):
                    self._full_bypass = bool(_first_meta.get("__full_bypass__", False))
                    logger.debug(
                        (
                            "WeMM _call_hf_processor: BYPASS MODE (audio pre-processe"
                            "d, full_bypass=%s)"
                        ),
                        self._full_bypass,
                    )
                    return self._build_preprocessed_result(
                        _bypass_audios, prompt=prompt, tok_kwargs=tok_kwargs
                    )
        # ====== END BYPASS ======

        logger.debug("WeMM _call_hf_processor: mm_data keys=%s", list(mm_data.keys()))
        videos_raw = mm_data.pop("videos", None)
        # vLLM may use "audios" (plural) or "audio" (singular) depending on version
        audios_raw = mm_data.pop("audios", None) or mm_data.pop("audio", None)
        logger.debug(
            "WeMM _call_hf_processor: videos_raw=%s (len=%s), audios_raw=%s",
            type(videos_raw).__name__ if videos_raw else None,
            len(videos_raw) if videos_raw else 0,
            type(audios_raw).__name__ if audios_raw else None,
        )

        hf_processor_kwargs = {}
        video_arrays = None
        # Per-video metadata: list of dicts for multi-video support
        per_video_metadata = []
        # Per-video bundled audio bytes (None if video has no audio)
        per_video_audio_bytes = []
        # Keep references to original metadata dicts so audio duration computed
        # here is visible to _get_prompt_updates_normal via mm_items.
        original_video_metadata_refs = []

        if videos_raw is not None and len(videos_raw) > 0:
            video_arrays = []
            for i, item in enumerate(videos_raw):
                if isinstance(item, list | tuple) and len(item) == 2:
                    video_array, metadata = item
                    video_arrays.append(video_array)
                    per_video_metadata.append(dict(metadata))
                    original_video_metadata_refs.append(
                        metadata if isinstance(metadata, dict) else None
                    )
                    # Consume audio bytes only from the existing local copy;
                    # callers may reuse the original metadata for another request.
                    # Keep the original reference for audio duration writeback.
                    metadata = per_video_metadata[-1]
                    # Extract audio_bytes from each video's metadata
                    per_video_audio_bytes.append(metadata.pop("audio_bytes", None))
                    # Use first video's metadata for HF processor kwargs
                    if i == 0:
                        hf_processor_kwargs.update(metadata)
                        _fi_dbg = metadata.get("frames_indices")
                        if _fi_dbg is not None:
                            try:
                                _fi_list = list(_fi_dbg)
                                _fi_str = f"frames={len(_fi_list)}"
                            except Exception:
                                _fi_str = str(_fi_dbg)[:200]
                        else:
                            _fi_str = "None"
                        logger.debug(
                            (
                                "WeMM: Video %d metadata keys=%s, fps=%s, backend=%s,"
                                " audio_bytes=%s, frames_indices=%s"
                            ),
                            i,
                            list(metadata.keys()),
                            metadata.get("fps"),
                            metadata.get("video_backend"),
                            len(per_video_audio_bytes[-1])
                            if per_video_audio_bytes[-1]
                            else None,
                            _fi_str,
                        )
                else:
                    video_arrays.append(item)
                    per_video_metadata.append({})
                    original_video_metadata_refs.append(None)
                    per_video_audio_bytes.append(None)

            # Remove audio_bytes from hf_processor_kwargs if it ended up there
            hf_processor_kwargs.pop("audio_bytes", None)

        # Process raw audio data from vLLM pipeline (standalone audio modality)
        # vLLM passes audio as list of (numpy_array, sample_rate) tuples
        audio_feature_tensors = None
        audio_lengths_list = None
        audio_durations_list = None
        # For standalone multi_audios: how many whisper segments each user-level
        # audio item produced. Used by _get_mm_fields_config to slice
        # audio_feature per audio item via flat_from_sizes.
        standalone_audio_num_segments = None
        standalone_audio_durations = None
        standalone_audio_intervals = None
        if audios_raw is not None and len(audios_raw) > 0:
            import os
            from io import BytesIO

            import soundfile

            from vllm.multimodal.wemm_audio import process_audio

            audio_feature_size = int(os.environ.get("AUDIO_FEATURE_SIZE", "128"))
            all_features = []
            all_lengths = []
            all_durations = []
            per_audio_segments = []  # len == len(audios_raw)
            per_audio_intervals = []

            for audio_item in audios_raw:
                option = None
                if isinstance(audio_item, dict) and "wemm_audio" in audio_item:
                    option = timestamp_interval_option(audio_item)
                    audio_item = audio_item["wemm_audio"]
                if option is None:
                    option = timestamp_interval_option(mm_kwargs)
                per_audio_intervals.append(resolve_timestamp_interval(option))
                # audio_item is (numpy_array, sample_rate) from AudioMediaIO
                if isinstance(audio_item, list | tuple) and len(audio_item) == 2:
                    audio_np, sr = audio_item
                else:
                    audio_np = audio_item
                    sr = 16000

                # Convert numpy array to WAV bytes for process_audio
                wav_buffer = BytesIO()
                soundfile.write(wav_buffer, audio_np, int(sr), format="WAV")
                wav_bytes = wav_buffer.getvalue()

                # Process through WhisperFeatureExtractor
                features, lengths, durations = process_audio(
                    BytesIO(wav_bytes),
                    audio_feature_size=audio_feature_size,
                )

                seg_count = (
                    sum(int(feature.shape[0]) for feature in features)
                    if features
                    else 0
                )
                per_audio_segments.append(seg_count)
                if features:
                    all_features.extend(features)
                    all_lengths.extend(lengths)
                    all_durations.extend(durations)

            if all_features:
                audio_feature_tensors = all_features
                audio_lengths_list = all_lengths
                audio_durations_list = all_durations
                standalone_audio_num_segments = per_audio_segments
                standalone_audio_durations = all_durations
                standalone_audio_intervals = per_audio_intervals

        # Process bundled audio from video ZIPs (video_audio cases)
        # Each video may independently have bundled audio
        per_video_audio_info = None
        bundled_audio_feature_tensors = None
        bundled_audio_lengths = None
        bundled_audio_durations = None
        if any(ab is not None for ab in per_video_audio_bytes):
            import os
            from io import BytesIO

            from vllm.multimodal.wemm_audio import process_audio

            audio_feature_size = int(os.environ.get("AUDIO_FEATURE_SIZE", "128"))
            all_features = []
            all_lengths = []
            all_durations = []
            # Track which segments belong to which video
            # per_video_audio_info[i] = {"segment_start": s, "segment_count": c,
            # "audio_length": l}
            per_video_audio_info = []
            segment_offset = 0

            for vid_idx, audio_bytes in enumerate(per_video_audio_bytes):
                if audio_bytes is not None:
                    features, lengths, durations = process_audio(
                        BytesIO(audio_bytes),
                        audio_feature_size=audio_feature_size,
                        preserve_float32=True,
                    )
                    if features:
                        seg_count = sum(int(feature.shape[0]) for feature in features)
                        audio_info = {
                            "segment_start": segment_offset,
                            "segment_count": seg_count,
                            "audio_length": lengths[0] if lengths else 0,
                            "audio_duration": durations[0] if durations else 0,
                        }
                        per_video_audio_info.append(audio_info)
                        if vid_idx < len(per_video_metadata):
                            per_video_metadata[vid_idx].update(audio_info)
                        if vid_idx < len(original_video_metadata_refs):
                            meta_ref = original_video_metadata_refs[vid_idx]
                            if isinstance(meta_ref, dict):
                                meta_ref.update(audio_info)
                        all_features.extend(features)
                        all_lengths.extend(lengths)
                        all_durations.extend(durations)
                        segment_offset += seg_count
                        logger.debug(
                            "WeMM: Video %d bundled audio: %d segments, length=%s",
                            vid_idx,
                            seg_count,
                            lengths,
                        )
                    else:
                        per_video_audio_info.append(None)
                else:
                    per_video_audio_info.append(None)

            if all_features:
                bundled_audio_feature_tensors = all_features
                bundled_audio_lengths = all_lengths
                bundled_audio_durations = all_durations

        self._wemm_last_per_video_audio_info = per_video_audio_info
        self._wemm_last_audio_durations_list = audio_durations_list

        # NEVER pass audio to the HF processor. The HF processor's audio
        # handling modifies input_ids directly which conflicts with vLLM's
        # PromptReplacement mechanism. Instead, we:
        # 1. Process video only through the HF processor
        # 2. Store audio features in the result for the model encoder
        # 3. Let _get_prompt_updates handle audio token expansion
        call_data = dict(text=prompt, **mm_data)
        if video_arrays is not None:
            call_data["videos"] = video_arrays
        call_kwargs = self.info.ctx.get_merged_mm_kwargs(
            dict(**mm_kwargs, **tok_kwargs, **hf_processor_kwargs)
        )
        if video_arrays is not None:
            # Video decoder budgets must not change image features in a mixed
            # request: image cache keys are independent of video cache misses.
            video_options = dict(call_kwargs.get("videos_kwargs", {}))
            for key in (
                "fps",
                "do_sample_frames",
                "num_frames",
                "min_frames",
                "max_frames",
                "return_metadata",
            ):
                if key in call_kwargs:
                    video_options.setdefault(key, call_kwargs.pop(key))
            shared_size = call_kwargs.pop("size", None)
            image_options = dict(call_kwargs.get("images_kwargs", {}))
            if shared_size is not None:
                image_options.setdefault("size", shared_size)
            shared_resize = call_kwargs.pop("do_resize", None)
            if shared_resize is not None:
                image_options.setdefault("do_resize", shared_resize)
                video_options.setdefault("do_resize", shared_resize)
            if image_options:
                call_kwargs["images_kwargs"] = image_options
            metadata = per_video_metadata[0]
            if bool(metadata.get("wemm_pre_resized", True)):
                video_options.setdefault("do_resize", False)
            else:
                # A checkpoint size is a default, not an implicit request cap.
                # Match WeMMProcessor: explicit processor kwargs win, otherwise
                # use this video's complete pixel budget for both size bounds.
                size = video_options.get("size", shared_size)
                explicit_size = size is not None
                if size is None:
                    size = metadata.get("wemm_video_size")
                if size is None and "wemm_max_pixels" in metadata:
                    frames = len(video_arrays[0])
                    size = {"longest_edge": int(metadata["wemm_max_pixels"]) * frames,
                            "shortest_edge": int(metadata.get("wemm_min_pixels", 4096)) * frames}
                size = dict(size or self.processor.video_processor.size)
                video_options["size"] = size
                if not explicit_size and len(video_arrays) > 1 and all(
                    not item.get("wemm_pre_resized", True) for item in per_video_metadata
                ):
                    call_kwargs["wemm_video_sizes"] = [
                        item.get("wemm_video_size") or {
                            "longest_edge": int(item["wemm_max_pixels"]) * len(array),
                            "shortest_edge": int(item.get("wemm_min_pixels", 4096)) * len(array),
                        } for item, array in zip(per_video_metadata, video_arrays)
                    ]
            call_kwargs["videos_kwargs"] = video_options
            for key in (
                "wemm_pre_resized",
                "wemm_max_pixels",
                "wemm_min_pixels",
                "wemm_video_size",
                "video_bytes_sha256",
                "audio_bytes_sha256",
                "wemm_decode_kwargs",
            ):
                call_kwargs.pop(key, None)

        # For multi-video: remove per-video metadata that doesn't apply to all
        # videos. The HF processor uses these for timestamp generation, but
        # frames_indices from video 0 would be wrong for video 1, 2, etc.
        if video_arrays is not None and len(video_arrays) > 1:
            call_kwargs["wemm_video_metadata"] = per_video_metadata
            call_kwargs.pop("frames_indices", None)
            call_kwargs.pop("total_num_frames", None)

        result = self.info.ctx.call_hf_processor(
            self.processor,
            call_data,
            call_kwargs,
        )

        # Standalone audio and audio bound to video are separate field families.
        # This also handles a paired video plus additional independent audio items.
        if bundled_audio_feature_tensors is not None:
            if standalone_audio_num_segments is not None:
                result["audio_feature"] = torch.cat(audio_feature_tensors, dim=0)
                result["audio_num_segments"] = torch.tensor(
                    standalone_audio_num_segments, dtype=torch.long
                )
            audio_feature_tensors = bundled_audio_feature_tensors
            audio_lengths_list = bundled_audio_lengths
            audio_durations_list = bundled_audio_durations
            standalone_audio_num_segments = None

        if standalone_audio_durations is not None:
            result["audio_duration_seconds"] = torch.tensor(standalone_audio_durations, dtype=torch.float64)
            result["timestamp_interval_sec"] = torch.tensor(standalone_audio_intervals, dtype=torch.float64)

        # Manually add audio features to the output for the model
        if audio_feature_tensors is not None:
            audio_field = (
                "video_audio_feature"
                if bundled_audio_feature_tensors is not None
                else "audio_feature"
            )
            result[audio_field] = torch.cat(audio_feature_tensors, dim=0)
            # Store lengths/durations for _get_prompt_updates to use
            result["_audio_lengths_list"] = audio_lengths_list
            result["_audio_durations_list"] = torch.tensor(
                audio_durations_list, dtype=torch.float32
            )

            # For standalone multi_audios, publish per-item segment counts so
            # _get_mm_fields_config can slice audio_feature back to per-item
            # tensors via MultiModalFieldConfig.flat_from_sizes. Without this
            # the field defaults to shared(batch_size=1) which crashes vLLM's
            # _merge_mm_kwargs with IndexError when the request contains >1
            # audio_url in the same user turn.
            if standalone_audio_num_segments is not None:
                result["audio_num_segments"] = torch.tensor(
                    standalone_audio_num_segments, dtype=torch.long
                )

            # Compute per-timestep audio groups for embed_multimodal interleaving
            video_grid_thw = result.get("video_grid_thw")
            if (
                bundled_audio_feature_tensors is not None
                and video_grid_thw is not None
                and len(video_grid_thw) > 0
            ):
                from .wemm_processor import _split_audio_length_for_video

                image_processor = self.info.get_image_processor()
                temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
                second_per_window = self.info.get_second_per_window()

                num_videos = len(video_grid_thw)
                has_per_video_info = (
                    per_video_audio_info is not None
                    and len(per_video_audio_info) == num_videos
                )

                def _sampled_fps_for_audio_groups(
                    vid_idx: int, grid_t: int, audio_duration=None
                ):
                    vid_meta = (
                        per_video_metadata[vid_idx]
                        if vid_idx < len(per_video_metadata)
                        else {}
                    )
                    frame_indices = (
                        vid_meta.get("frames_indices")
                        if isinstance(vid_meta, dict)
                        else None
                    )
                    nframes_sampled = (
                        len(frame_indices)
                        if frame_indices is not None
                        else grid_t * temporal_patch_size
                    )
                    if audio_duration is not None and audio_duration > 0:
                        return float(nframes_sampled) / float(audio_duration)
                    duration = (
                        vid_meta.get("duration") if isinstance(vid_meta, dict) else None
                    )
                    if duration is None and isinstance(vid_meta, dict):
                        total_frames = vid_meta.get("total_num_frames")
                        origin_fps = vid_meta.get("fps")
                        if (
                            total_frames is not None
                            and origin_fps is not None
                            and origin_fps > 0
                        ):
                            duration = float(total_frames) / float(origin_fps)
                    if duration is not None and duration > 0:
                        return float(nframes_sampled) / float(duration)
                    return 1.0

                if num_videos == 1:
                    # Single video: original logic
                    grid_thw = video_grid_thw[0]
                    grid_t = int(grid_thw[0].item())
                    # Convert process_audio's segments*60 to actual model tokens
                    tokens_per_segment = self.info.get_audio_tokens_per_segment()
                    if audio_lengths_list:
                        num_seg = audio_lengths_list[0] // 60
                        audio_length = num_seg * tokens_per_segment
                    else:
                        audio_length = 0
                    audio_duration = (
                        audio_durations_list[0] if audio_durations_list else None
                    )
                    video_fps = _sampled_fps_for_audio_groups(0, grid_t, audio_duration)
                    audio_groups = _split_audio_length_for_video(
                        audio_length=audio_length,
                        grid_t=grid_t,
                        video_fps=video_fps,
                        temporal_patch_size=temporal_patch_size,
                        second_per_window=second_per_window,
                    )
                    result["_audio_groups"] = torch.tensor(
                        audio_groups, dtype=torch.long
                    )
                else:
                    # Multi-video: compute audio_groups per video, concatenate
                    # with -1 separator
                    tokens_per_segment = self.info.get_audio_tokens_per_segment()
                    all_audio_groups = []
                    for vid_idx in range(num_videos):
                        grid_thw = video_grid_thw[vid_idx]
                        grid_t = int(grid_thw[0].item())
                        # Get this video's audio_length (convert from *60 to model
                        # tokens)
                        audio_duration = None
                        if (
                            has_per_video_info
                            and per_video_audio_info[vid_idx] is not None
                        ):
                            old_length = per_video_audio_info[vid_idx]["audio_length"]
                            num_seg = old_length // 60
                            audio_length = num_seg * tokens_per_segment
                            audio_duration = per_video_audio_info[vid_idx].get(
                                "audio_duration"
                            )
                        else:
                            audio_length = 0
                        video_fps = _sampled_fps_for_audio_groups(
                            vid_idx, grid_t, audio_duration
                        )

                        if audio_length > 0:
                            audio_groups = _split_audio_length_for_video(
                                audio_length=audio_length,
                                grid_t=grid_t,
                                video_fps=video_fps,
                                temporal_patch_size=temporal_patch_size,
                                second_per_window=second_per_window,
                            )
                        else:
                            # No audio for this video: all zeros
                            audio_groups = [0] * grid_t

                        if vid_idx > 0:
                            all_audio_groups.append(-1)  # separator
                        all_audio_groups.extend(audio_groups)

                    result["_audio_groups"] = torch.tensor(
                        all_audio_groups, dtype=torch.long
                    )
                    logger.debug(
                        "WeMM: Multi-video audio_groups (len=%d): %s",
                        len(all_audio_groups),
                        all_audio_groups[:20],
                    )

        tokenizer = self.info.get_tokenizer()
        token_options = dict(tok_kwargs)
        token_options.setdefault("add_special_tokens", False)
        compact_ids = tokenizer(prompt, **token_options)["input_ids"]
        if isinstance(compact_ids, torch.Tensor):
            compact_ids = compact_ids.reshape(-1).tolist()
        elif compact_ids and isinstance(compact_ids[0], list):
            compact_ids = compact_ids[0]
        result["input_ids"] = torch.tensor([compact_ids], dtype=torch.long)
        if "attention_mask" in result:
            result["attention_mask"] = torch.ones_like(result["input_ids"])
        return result

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)
        video_audio_field = _prepare_video_audio_fields(
            hf_inputs, self.info.get_audio_tokens_per_segment()
        ) if len(video_grid_thw) else None
        logger.debug(
            "WeMM _get_mm_fields_config: hf_inputs keys=%s, has_audio=%s",
            list(hf_inputs.keys()),
            "audio_feature" in hf_inputs,
        )
        fields = dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            # vLLM retains CPU metadata when prefix caching covers the media.
            # M-RoPE still needs the original grid even if encoding is skipped.
            image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )
        # Audio feature: declare under "video" modality when video is present
        # (video_audio bundled case), otherwise under "audio" modality (standalone)
        if "audio_feature" in hf_inputs:
            has_video = video_grid_thw.shape[0] > 0
            if video_audio_field == "audio_feature":
                fields["audio_feature"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", hf_inputs["_video_audio_num_segments"]
                )
            else:
                # Standalone audio. If audio_num_segments is present (published
                # by _call_hf_processor for multi_audios), slice audio_feature
                # per user audio item via flat_from_sizes so vLLM can address N
                # audio items separately. Fall back to shared(batch_size=1)
                # only when the hint is missing (older callers).
                audio_num_segments = hf_inputs.get("audio_num_segments")
                if (
                    audio_num_segments is not None
                    and hasattr(audio_num_segments, "numel")
                    and audio_num_segments.numel() > 0
                ):
                    fields["audio_feature"] = MultiModalFieldConfig.flat_from_sizes(
                        "audio", audio_num_segments
                    )
                    fields["audio_num_segments"] = MultiModalFieldConfig.batched(
                        "audio"
                    )
                else:
                    fields["audio_feature"] = MultiModalFieldConfig.shared(
                        "audio", batch_size=1
                    )
        if "video_audio_feature" in hf_inputs:
            fields["video_audio_feature"] = MultiModalFieldConfig.flat_from_sizes(
                "video", hf_inputs["_video_audio_num_segments"]
            )
        if "_video_audio_num_segments" in hf_inputs:
            fields["_video_audio_num_segments"] = MultiModalFieldConfig.batched("video")
        for name in ("audio_duration_seconds", "timestamp_interval_sec"):
            if name in hf_inputs:
                fields[name] = MultiModalFieldConfig.batched("audio")
        # _audio_groups: internal metadata for embed_multimodal interleaving
        if "_audio_groups" in hf_inputs:
            has_video = video_grid_thw.shape[0] > 0
            if has_video:
                fields["_audio_groups"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", video_grid_thw[:, 0].long() + 1
                )
            else:
                fields["_audio_groups"] = MultiModalFieldConfig.shared(
                    "audio", batch_size=1
                )
        # Standalone audio duration/options are per-item fields above; they must
        # survive preprocessing cache hits and mixed standalone/video requests.
        return fields

    # once check here
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # In full bypass mode, return dummy replacements that won't access metadata.
        # The actual placeholder scanning is done in _maybe_apply_prompt_updates.
        # Detection: self._full_bypass flag (set by _call_hf_processor) OR
        # fallback when normal path fails (cache hit scenario).
        is_full_bypass = getattr(self, "_full_bypass", False)

        if not is_full_bypass:
            # Try normal path; if it fails with KeyError/IndexError (missing metadata
            # because _call_hf_processor was cached), fall back to full bypass.
            try:
                return self._get_prompt_updates_normal(
                    mm_items, hf_processor_mm_kwargs, out_mm_kwargs
                )
            except (KeyError, IndexError, TypeError, AssertionError) as e:
                # Check if this looks like a full bypass request (has video data but
                # missing normal metadata like frames_indices)
                has_video = "video" in out_mm_kwargs and len(out_mm_kwargs["video"]) > 0
                if has_video:
                    logger.debug(
                        "WeMM _get_prompt_updates: normal path failed (%s: %s), "
                        "falling back to full bypass dummy replacements",
                        type(e).__name__,
                        e,
                    )
                    is_full_bypass = True
                else:
                    raise

        if is_full_bypass:
            hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
            tokenizer = self.info.get_tokenizer()
            replacements = []
            has_video = "video" in out_mm_kwargs and len(out_mm_kwargs["video"]) > 0
            has_image = "image" in out_mm_kwargs and len(out_mm_kwargs["image"]) > 0
            has_audio = "audio" in out_mm_kwargs and len(out_mm_kwargs["audio"]) > 0
            # Check for bundled audio in video items
            has_bundled_audio = False
            if has_video:
                video_item_0 = (
                    out_mm_kwargs["video"][0] if len(out_mm_kwargs["video"]) > 0 else {}
                )
                has_bundled_audio = (
                    "video_audio_feature" in video_item_0
                    or "audio_feature" in video_item_0
                )
            if has_image:
                replacements.append(
                    PromptReplacement(
                        modality="image",
                        target=tokenizer.encode(hf_processor.image_token, add_special_tokens=False),
                        replacement=lambda item_idx: [0],  # dummy, won't be used
                    )
                )
            if has_video:
                replacements.append(
                    PromptReplacement(
                        modality="video",
                        target=tokenizer.encode("<|vision_start|><|video_pad|><|vision_end|>", add_special_tokens=False),
                        replacement=lambda item_idx: [0],  # dummy, won't be used
                    )
                )
            # Audio: only add replacement for standalone audio (audio_start/end in
            # prompt)
            # For bundled audio (within video region), don't create audio replacement
            # because mm_hashes only has "video" key, not "audio"
            if "audio" in out_mm_kwargs and len(out_mm_kwargs["audio"]) > 0:
                # Standalone audio
                replacements.append(
                    PromptReplacement(
                        modality="audio",
                        target=tokenizer.encode("<|audio_start|><|audio_pad|><|audio_end|>", add_special_tokens=False),
                        replacement=lambda item_idx: [0],  # dummy, won't be used
                    )
                )
            logger.debug(
                "WeMM FULL BYPASS _get_prompt_updates: returning dummy replacements "
                "(video=%s, image=%s, audio=%s, bundled_audio=%s)",
                has_video,
                has_image,
                has_audio,
                has_bundled_audio,
            )
            return replacements

        return self._get_prompt_updates_normal(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        )

    def _get_prompt_updates_normal(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Normal (non-bypass) prompt updates with token expansion."""
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = getattr(
            hf_config, "vision_end_token_id", vision_start_token_id
        )
        merge_length = image_processor.merge_size**2
        second_per_window = self.info.get_second_per_window()

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            # Use image_pad token ID, consistent with video_pad and audio_pad
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            return [image_pad_id] * num_tokens

        # Determine if this is a video+audio request
        has_audio = "audio" in out_mm_kwargs and len(out_mm_kwargs["audio"]) > 0
        has_video = "video" in out_mm_kwargs and len(out_mm_kwargs["video"]) > 0

        # For bundled video_audio: audio_feature is under "video" modality
        # Check if the video item contains audio_feature (bundled case)
        has_bundled_audio = False
        if has_video:
            video_item_0 = (
                out_mm_kwargs["video"][0] if len(out_mm_kwargs["video"]) > 0 else {}
            )
            if "video_audio_feature" in video_item_0 or "audio_feature" in video_item_0:
                has_bundled_audio = True
        # has_audio covers both standalone and bundled audio for video replacement logic
        has_audio = has_audio or has_bundled_audio

        # Each item owns its audio and groups, including after a processor cache hit.
        num_videos = len(out_mm_kwargs.get("video", []))
        per_video_audio_lengths = None  # Will be computed if multi-video with audio
        if has_bundled_audio and num_videos > 0:
            per_video_audio_lengths = [
                sum(_split_video_audio_groups(item["_audio_groups"].data, 1)[0])
                for item in out_mm_kwargs["video"]
            ]

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            # Get video metadata for fps
            video_item = mm_items["video"][item_idx]
            if isinstance(video_item, list | tuple) and len(video_item) == 2:
                video_array, metadata = video_item
            else:
                metadata = {}

            # Determine if THIS specific video has audio
            this_video_has_audio = False
            this_audio_length = 0
            tokens_per_segment = self.info.get_audio_tokens_per_segment()
            if has_bundled_audio:
                if per_video_audio_lengths is not None:
                    # Multi-video: check this video's audio allocation
                    if item_idx < len(per_video_audio_lengths):
                        this_audio_length = per_video_audio_lengths[item_idx]
                        this_video_has_audio = this_audio_length > 0
                elif num_videos == 1:
                    # Single video with audio
                    this_video_has_audio = True
                    if has_bundled_audio:
                        audio_feat = out_item.get(
                            "video_audio_feature", out_item.get("audio_feature")
                        )
                        if audio_feat is not None:
                            audio_feat_data = (
                                audio_feat.data
                                if hasattr(audio_feat, "data")
                                else audio_feat
                            )
                            if isinstance(audio_feat_data, torch.Tensor):
                                this_audio_length = (
                                    audio_feat_data.shape[0] * tokens_per_segment
                                )
                            else:
                                this_audio_length = tokens_per_segment
                    else:
                        out_audio = out_mm_kwargs["audio"][0]
                        audio_feat = out_audio["audio_feature"].data
                        if isinstance(audio_feat, torch.Tensor):
                            this_audio_length = audio_feat.shape[0] * tokens_per_segment
                        else:
                            this_audio_length = tokens_per_segment

            if this_video_has_audio and this_audio_length > 0:
                from .wemm_processor import build_video_audio_input_ids

                # Get sampled FPS for audio split, and original frame metadata for
                # timestamp generation. To match lmms-eval exactly:
                #   audio split fps = len(metadata["frames_indices"]) / audio_duration
                #   timestamps = average(original frame indices) / origin_video_fps
                frame_indices = (
                    metadata.get("frames_indices")
                    if isinstance(metadata, dict)
                    else None
                )
                origin_fps = metadata.get("fps") if isinstance(metadata, dict) else None
                try:
                    nframes_actual = (
                        len(frame_indices)
                        if frame_indices is not None
                        else int(grid_thw[0].item())
                        * int(image_processor.temporal_patch_size)
                    )
                except Exception:
                    nframes_actual = int(grid_thw[0].item()) * int(
                        image_processor.temporal_patch_size
                    )

                audio_duration = (
                    metadata.get("audio_duration")
                    if isinstance(metadata, dict)
                    else None
                )
                if audio_duration is None:
                    last_info = getattr(self, "_wemm_last_per_video_audio_info", None)
                    if (
                        isinstance(last_info, list)
                        and item_idx < len(last_info)
                        and last_info[item_idx] is not None
                    ):
                        audio_duration = last_info[item_idx].get("audio_duration")
                if audio_duration is None:
                    last_durations = getattr(
                        self, "_wemm_last_audio_durations_list", None
                    )
                    if (
                        isinstance(last_durations, list | tuple)
                        and len(last_durations) > 0
                    ):
                        audio_duration = last_durations[
                            min(item_idx, len(last_durations) - 1)
                        ]

                duration = (
                    metadata.get("duration") if isinstance(metadata, dict) else None
                )
                if duration is None and isinstance(metadata, dict):
                    total_frames = metadata.get("total_num_frames")
                    if (
                        total_frames is not None
                        and origin_fps is not None
                        and origin_fps > 0
                    ):
                        duration = float(total_frames) / float(origin_fps)

                if audio_duration is not None and audio_duration > 0:
                    video_fps = float(nframes_actual) / float(audio_duration)
                elif duration is not None and duration > 0:
                    video_fps = float(nframes_actual) / float(duration)
                else:
                    video_fps = 1.0

                logger.debug(
                    "WeMM get_video_replacement: item_idx=%d, grid_thw=%s, "
                    "audio_length=%d, video_fps=%s (actual_sampled, "
                    "duration=%s, nframes_actual=%s), has_bundled=%s",
                    item_idx,
                    grid_thw.tolist(),
                    this_audio_length,
                    video_fps,
                    audio_duration if audio_duration is not None else duration,
                    nframes_actual,
                    has_bundled_audio,
                )

                # Generate interleaved video+audio token sequence
                audio_token_ids = build_video_audio_input_ids(
                    tokenizer,
                    video_grid_thw=grid_thw,
                    audio_length=this_audio_length,
                    video_fps=video_fps,
                    temporal_patch_size=image_processor.temporal_patch_size,
                    merge_size=image_processor.merge_size,
                    second_per_window=second_per_window,
                    frame_indices=frame_indices,
                    origin_video_fps=origin_fps,
                )

                audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
                num_video_tokens = sum(
                    1 for t in audio_token_ids if t == video_token_id
                )
                num_audio_tokens = sum(1 for t in audio_token_ids if t == audio_pad_id)
                logger.debug(
                    "WeMM get_video_replacement: total_tokens=%d, video_tokens=%d, "
                    "audio_tokens=%d, audio_pad_id=%d",
                    len(audio_token_ids),
                    num_video_tokens,
                    num_audio_tokens,
                    audio_pad_id,
                )

                # === WeMM DEBUG DUMP === (compare with HF demo dump for diff)
                try:
                    _vsta = tokenizer.convert_tokens_to_ids("<|vision_start|>")
                    _vend = tokenizer.convert_tokens_to_ids("<|vision_end|>")
                    _ttok = audio_token_ids
                    # Build per-step audio_pad counts (audio groups distribution
                    # as actually inserted into the sequence)
                    per_step_audio = []
                    cur = 0
                    in_block = False
                    for t in _ttok:
                        if t == _vsta:
                            in_block = True
                            cur = 0
                        elif t == _vend:
                            if in_block:
                                per_step_audio.append(cur)
                            in_block = False
                            cur = 0
                        elif in_block and t == audio_pad_id:
                            cur += 1
                    # First/last 30 token ids as compact summary
                    head = list(_ttok[:40])
                    tail = list(_ttok[-30:])
                    # Also decode the head as text to see what bytes were
                    # actually generated (timestamps, etc.)
                    try:
                        head_text = tokenizer.decode(head, skip_special_tokens=False)
                    except Exception:
                        head_text = "(decode failed)"
                    logger.debug(
                        "WeMM DUMP audio_groups (per-step audio counts, first 32): %s",
                        per_step_audio[:32],
                    )
                    logger.debug(
                        "WeMM DUMP audio_groups sum=%s len=%s",
                        sum(per_step_audio),
                        len(per_step_audio),
                    )
                    logger.debug("WeMM DUMP head 40 ids: %s", head)
                    logger.debug("WeMM DUMP head decoded: %r", head_text)
                    logger.debug("WeMM DUMP tail 30 ids: %s", tail)
                except Exception as _dbg_e:
                    logger.warning("WeMM DUMP failed: %s", _dbg_e)

                return PromptUpdateDetails.select_token_ids(
                    audio_token_ids, [video_token_id, audio_pad_id]
                )
            else:
                # Video-only: use timestamps from metadata
                do_sample_frames = hf_processor_mm_kwargs.get("do_sample_frames")
                sampled_fps = hf_processor_mm_kwargs.get("fps")
                if is_list_of(sampled_fps, float):
                    sampled_fps = sampled_fps[item_idx]
                timestamps = self.info._get_video_second_idx(
                    metadata, out_item, do_sample_frames, sampled_fps
                )
                assert len(timestamps) == grid_thw[0], (
                    f"The timestamps length({len(timestamps)}) should be equal "
                    f"video length ({grid_thw[0]})."
                )
                frames_idx_token = [
                    tokenizer.encode(
                        f"<{curr_time:.1f} seconds>", add_special_tokens=False
                    )
                    for curr_time in timestamps
                ]
                num_tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
                placeholder = []
                for frame_idx in frames_idx_token:
                    placeholder.extend(frame_idx)
                    placeholder.extend(
                        [vision_start_token_id]
                        + [video_token_id] * num_tokens_per_frame
                        + [vision_end_token_id]
                    )
                return PromptUpdateDetails.select_token_id(placeholder, video_token_id)

        # num_tokens = self.info.get_num_audio_tokens()

        def get_audio_replacement(item_idx: int):
            """Generate audio token sequence for standalone audio.

            Independent audio uses its own placeholder even when video is present.
            """
            from .wemm_processor import build_audio_input_ids

            # Standalone audio: generate audio token sequence
            # Get audio feature for THIS audio item. When multi_audios is in
            # use, _get_mm_fields_config declares audio_feature via
            # flat_from_sizes → each mm item carries its own segment slice,
            # so we index by item_idx (was hard-coded to 0, which mis-sized
            # every audio item to the first one and required an equal-split
            # workaround below that could not handle unequal audio lengths).
            out_item = out_mm_kwargs["audio"][item_idx]
            audio_feat = out_item["audio_feature"].data

            # Compute audio_length from feature shape using model config
            if isinstance(audio_feat, torch.Tensor):
                num_segments = audio_feat.shape[0]
            else:
                num_segments = 1

            tokens_per_segment = self.info.get_audio_tokens_per_segment()
            audio_length = num_segments * tokens_per_segment

            if "audio_duration_seconds" not in out_item:
                raise ValueError("Missing true audio duration in WeMM processor output")
            actual_duration = float(out_item["audio_duration_seconds"].data.item())
            interval = float(out_item["timestamp_interval_sec"].data.item())

            # Generate the full token sequence using build_audio_input_ids
            audio_token_ids = build_audio_input_ids(
                tokenizer,
                audio_length=audio_length,
                timestamp_interval_sec=interval,
                second_per_window=second_per_window,
                enable_timestamps=True,
                actual_duration_seconds=actual_duration,
            )

            # Return just the token IDs (PromptReplacement handles the rest)
            audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
            return PromptUpdateDetails.select_token_id(audio_token_ids, audio_pad_id)

        replacements = [
            PromptReplacement(
                modality="image",
                target=tokenizer.encode(hf_processor.image_token, add_special_tokens=False),
                replacement=get_image_replacement_qwen3vl,
            ),
            PromptReplacement(
                modality="video",
                target=tokenizer.encode("<|vision_start|><|video_pad|><|vision_end|>", add_special_tokens=False),
                replacement=get_video_replacement_qwen3vl,
            ),
        ]

        # Add audio replacement only for standalone audio (not video_audio).
        # Paired audio belongs to video; independent audio keeps its own replacement.
        if "audio" in out_mm_kwargs and len(out_mm_kwargs["audio"]) > 0:
            replacements.append(
                PromptReplacement(
                    modality="audio",
                    target=tokenizer.encode("<|audio_start|><|audio_pad|><|audio_end|>", add_special_tokens=False),
                    replacement=get_audio_replacement,
                ),
            )

        return replacements


# --- Model: Qwen3VLOmniForConditionalGeneration ---


def _split_video_audio_groups(groups, num_videos, grid_t=None):
    """Read complete per-video groups; a trailing separator belongs to its item."""
    values = groups.tolist() if isinstance(groups, torch.Tensor) else list(groups)
    parts, current = [], []
    for value in values:
        if isinstance(value, bool) or int(value) != value or value < -1:
            raise ValueError("Video audio groups must contain integer counts or -1")
        value = int(value)
        if value == -1:
            if not current:
                raise ValueError("Empty video audio group; item mapping is invalid")
            parts.append(current)
            current = []
        else:
            current.append(value)
    if current:
        parts.append(current)
    if len(parts) != num_videos:
        raise ValueError(f"Video audio groups describe {len(parts)} items, expected {num_videos}")
    if grid_t is not None and [len(part) for part in parts] != [int(t) for t in grid_t]:
        raise ValueError("Video audio group timesteps differ from video_grid_thw")
    return parts


def _prepare_video_audio_fields(inputs, tokens_per_segment):
    """Give every cached/scheduled video its own feature slice and group slice."""
    grid = inputs["video_grid_thw"]
    field = "video_audio_feature" if "video_audio_feature" in inputs else (
        "audio_feature" if "audio_feature" in inputs and "audio_num_segments" not in inputs else None
    )
    if "_audio_groups" not in inputs:
        if field is not None:
            raise ValueError("Video-bound audio requires per-video audio groups")
        return None
    parts = _split_video_audio_groups(inputs["_audio_groups"], len(grid), grid[:, 0].tolist())
    totals = [sum(part) for part in parts]
    if tokens_per_segment <= 0 or any(total % tokens_per_segment for total in totals):
        raise ValueError("Video audio token counts do not match complete encoder segments")
    counts = [total // tokens_per_segment for total in totals]
    previous = inputs.get("_video_audio_num_segments")
    if previous is not None and torch.as_tensor(previous).reshape(-1).tolist() != counts:
        raise ValueError("Video audio groups disagree with per-video feature segment counts")
    if field is not None:
        feature = inputs[field]
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        if feature.ndim == 4 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        if feature.ndim != 3 or feature.shape[0] != sum(counts):
            raise ValueError("Video audio features and per-video segment counts differ")
        inputs[field] = feature
    elif sum(counts):
        raise ValueError("Video audio groups reference missing audio features")
    inputs["_video_audio_num_segments"] = torch.tensor(counts, dtype=torch.long)
    # A terminal separator is included in each flat slice, so arbitrary subsets
    # and batches from different requests retain the same unambiguous format.
    inputs["_audio_groups"] = torch.tensor(
        [value for part in parts for value in (*part, -1)], dtype=torch.long
    )
    return field


def _split_audio_embeds_by_items(
    audio_flat: torch.Tensor,
    audio_num_segments: Any,
) -> list[torch.Tensor]:
    """Split a flat (total_audio_tokens, hidden) tensor into one tensor per
    user-level audio item.

    vLLM's embed_multimodal contract requires len(mm_embeds) == num audio
    items. When multi_audios (audio_url * N in one turn) is used, the model
    encoder produces a single concatenated tensor; without splitting, vLLM
    asserts a mismatch.

    Each user audio item contributed `seg_i` whisper segments to
    audio_feature; the encoder outputs `tokens_per_seg = total_tokens /
    total_segs` tokens per segment (uniform across all segments). We slice
    the flat tensor into N chunks of `seg_i * tokens_per_seg` rows.
    """
    if audio_flat is None:
        return []
    if audio_num_segments is None:
        return [audio_flat]

    if isinstance(audio_num_segments, torch.Tensor):
        seg_counts = audio_num_segments.tolist()
    else:
        seg_counts = list(audio_num_segments)

    # Flatten nested lists (batched fields may show up as [[c0], [c1], ...]).
    flat_counts: list[int] = []
    for c in seg_counts:
        if isinstance(c, list | tuple):
            flat_counts.extend(int(x) for x in c)
        else:
            flat_counts.append(int(c))

    if len(flat_counts) <= 1:
        return [audio_flat]

    total_segs = sum(flat_counts)
    total_tokens = audio_flat.size(0)
    if any(count <= 0 for count in flat_counts) or total_tokens % total_segs != 0:
        raise ValueError(
            f"Invalid audio segment counts {flat_counts} for {total_tokens} tokens"
        )
    tokens_per_seg = total_tokens // total_segs

    parts: list[torch.Tensor] = []
    offset = 0
    for count in flat_counts:
        n = int(count) * tokens_per_seg
        parts.append(audio_flat[offset : offset + n])
        offset += n
    return parts


class _WeMMVideoMixin:
    """Shared WeMM media, audio, weight mapping and MRoPE for both backbones."""

    def _process_image_input(self, image_input):
        # v0.18's inherited ViT uses numpy for grid metadata. The runner
        # places multimodal fields on CUDA, so keep this small tensor on CPU.
        image_input = copy(image_input)
        image_input.image_grid_thw = image_input["image_grid_thw"].cpu()
        return super()._process_image_input(image_input)

    def _process_video_input(self, video_input):
        video_input = copy(video_input)
        video_input.video_grid_thw = video_input["video_grid_thw"].cpu()
        return super()._process_video_input(video_input)

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # HF WeMM stores Q-Former as model.audio.qformer.bert.*
            # We flatten to audio.qformer.*
            ".qformer.bert.": ".qformer.",
            # Old BERT naming convention (beta/gamma) -> new (bias/weight)
            "LayerNorm.beta": "LayerNorm.bias",
            "LayerNorm.gamma": "LayerNorm.weight",
        },
        orig_to_new_prefix={
            # Skip BERT embeddings.position_ids (buffer, not parameter)
            # Must appear before "model.audio." to take priority
            "model.audio.qformer.embeddings.position_ids": None,
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.audio.": "audio.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Override parent to support audio modality in addition to image/video."""
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError(
            f"Only image, video, or audio modality is supported, got: {modality}"
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        config = vllm_config.model_config.hf_config
        if config.text_config.model_type != self._wemm_text_model_type:
            raise ValueError(
                f"{type(self).__name__} requires text_config.model_type="
                f"{self._wemm_text_model_type!r}, got {config.text_config.model_type!r}"
            )
        if self._wemm_text_model_type == "qwen3_5_moe_text" and getattr(
            config.text_config, "num_experts", 0
        ) <= 0:
            raise ValueError("A WeMM MoE checkpoint requires positive num_experts")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Audio encoder (optional - only created if audio_config exists)
        self.audio = None
        self.audio_token_id = getattr(config, "audio_token_id", None)

        if hasattr(config, "audio_config") and config.audio_config is not None:
            raw_audio = config.audio_config
            if isinstance(raw_audio, dict):
                qformer_config = raw_audio.get("qformer_config")
                audio_config = WhisperConfig.from_dict(
                    {k: v for k, v in raw_audio.items() if k != "qformer_config"}
                )
            else:
                qformer_config = getattr(raw_audio, "qformer_config", None)
                audio_config = raw_audio

            # vLLM registry expects architectures to be an iterable
            if getattr(audio_config, "architectures", None) is None:
                ac_dict = dict(audio_config.to_dict())
                ac_dict["architectures"] = ["WhisperForConditionalGeneration"]
                audio_config = WhisperConfig.from_dict(ac_dict)

            model_path = vllm_config.model_config.model
            self.audio = WeMMAudioEncoder(
                audio_config=audio_config,
                model_path=model_path,
                qformer_config=qformer_config,
                prefix=maybe_prefix(prefix, "audio"),
            )
            logger.info(
                "WeMM-Video: audio encoder loaded (audio_token_id=%s)",
                self.audio_token_id,
            )

    def _process_audio_input(self, audio_feature: Any) -> torch.Tensor | None:
        """Process audio features through Whisper + Q-Former encoder."""
        if self.audio is None or audio_feature is None:
            return None

        if isinstance(audio_feature, list | tuple):
            if len(audio_feature) == 0:
                return None
            audio_feature = torch.cat(
                [
                    x if isinstance(x, torch.Tensor) else x.unsqueeze(0)
                    for x in audio_feature
                ],
                dim=0,
            )

        if not isinstance(audio_feature, torch.Tensor):
            return None

        # Check for empty tensor
        if audio_feature.numel() == 0:
            return None

        # Cast to model dtype (bfloat16) and move to correct device
        model_dtype = next(self.audio.parameters()).dtype
        model_device = next(self.audio.parameters()).device
        audio_feature = audio_feature.to(dtype=model_dtype, device=model_device)

        # Ensure audio_feature has shape (num_segments, feature_size, time)
        # It may come as (num_segments, 128, 3000) or (1, num_segments, 128, 3000)
        if audio_feature.dim() == 4:
            # Extra batch dim - squeeze it
            audio_feature = audio_feature.squeeze(0)
        elif audio_feature.dim() == 2:
            # Missing segment dim - add it: (128, 3000) → (1, 128, 3000)
            audio_feature = audio_feature.unsqueeze(0)

        # Now audio_feature should be (num_segments, feature_size, time)
        # Process in batches to avoid OOM on long audio
        bz = 32
        audio_embeds_list = []
        for x in range(0, int(audio_feature.size(0)), bz):
            batch_audio_embeds = self.audio(audio_feature[x : x + bz])
            audio_embeds_list.append(batch_audio_embeds)

        return torch.cat(audio_embeds_list, dim=0)

    # ------------------------------------------------------------------
    # Multimodal embedding: override parent's methods for audio support
    # ------------------------------------------------------------------

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        """Parse multimodal inputs, extracting audio_feature if present."""
        mm_input_by_modality = super()._parse_and_validate_multimodal_inputs(**kwargs)
        audio_feature = kwargs.pop("audio_feature", None)
        if audio_feature is not None and "audio" not in mm_input_by_modality:
            mm_input_by_modality["audio"] = audio_feature
        # Extract _audio_groups for video_audio interleaving
        audio_groups = kwargs.pop("_audio_groups", None)
        if audio_groups is not None:
            mm_input_by_modality["_audio_groups"] = audio_groups
        # Extract per-item audio segment counts (standalone multi_audios).
        # Set by wemm_omni._call_hf_processor and declared as batched("audio")
        # in _get_mm_fields_config. Used below to split audio embeddings back
        # into one tensor per user audio item (vLLM asserts len(mm_embeds) ==
        # num_items).
        audio_num_segments = kwargs.pop("audio_num_segments", None)
        if audio_num_segments is not None:
            mm_input_by_modality["_audio_num_segments"] = audio_num_segments
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object):
        """Override parent's embed_multimodal to support audio modality.

        The parent only handles image and video. This adds audio embedding
        support for standalone audio and video+audio inputs.

        For video_audio (bundled): produces interleaved tensors containing
        both video and audio embeddings in the order they appear in the token
        sequence (video_pad*N, audio_pad*M per timestep).

        Supports multi-video: each video independently has or doesn't have audio.
        _audio_groups uses -1 as separator between videos.
        """
        video_audio = kwargs.pop("video_audio_feature", None)
        if video_audio is not None:
            # Video-bound features travel under the video modality. Reuse the
            # interleaver with only that audio, then encode independent items.
            standalone = kwargs.get("audio_feature")
            paired = dict(kwargs, audio_feature=video_audio)
            paired.pop("audio_num_segments", None)
            embeddings = list(self.embed_multimodal(**paired) or [])
            if standalone is not None:
                encoded = self._process_audio_input(standalone)
                if encoded is not None:
                    embeddings.extend(
                        _split_audio_embeds_by_items(
                            encoded.reshape(-1, encoded.shape[-1]),
                            kwargs.get("audio_num_segments"),
                        )
                    )
            return embeddings

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: list = []

        has_video = "video" in mm_input_by_modality
        has_audio = "audio" in mm_input_by_modality
        audio_groups = mm_input_by_modality.pop("_audio_groups", None)
        # Convert from tensor back to list if needed
        if isinstance(audio_groups, torch.Tensor):
            audio_groups = audio_groups.tolist()

        if has_video and has_audio and audio_groups is not None:
            # Video+audio bundled case: produce interleaved tensors.
            video_input = mm_input_by_modality["video"]
            audio_input = mm_input_by_modality["audio"]

            # Get video embeddings: tuple of tensors (one per video)
            video_embeddings = self._process_video_input(video_input)

            # Get audio embeddings
            audio_embeds = self._process_audio_input(audio_input)
            if audio_embeds is not None:
                audio_flat = audio_embeds.view(-1, audio_embeds.size(-1))
            else:
                audio_flat = torch.zeros(
                    0,
                    video_embeddings[0].size(-1) if video_embeddings else 0,
                    dtype=video_embeddings[0].dtype
                    if video_embeddings
                    else torch.float32,
                    device=video_embeddings[0].device if video_embeddings else "cpu",
                )

            per_video_groups = _split_video_audio_groups(audio_groups, len(video_embeddings))
            required_audio = sum(map(sum, per_video_groups))
            if required_audio != audio_flat.shape[0]:
                raise ValueError(
                    f"Video audio embeddings mismatch: groups require {required_audio} tokens, "
                    f"encoder returned {audio_flat.shape[0]}"
                )

            # Interleave each video with its audio portion
            audio_offset = 0
            for vid_idx, video_flat in enumerate(video_embeddings):
                vid_audio_groups = per_video_groups[vid_idx]
                total_audio_for_vid = sum(vid_audio_groups)

                if total_audio_for_vid > 0 and len(vid_audio_groups) > 0:
                    # Interleave video and audio for this video
                    total_video_tokens = video_flat.size(0)
                    grid_t = len(vid_audio_groups)
                    if total_video_tokens % grid_t:
                        raise ValueError("Video embedding length is not divisible by audio group timesteps")
                    tokens_per_frame = (
                        total_video_tokens // grid_t
                        if grid_t > 0
                        else total_video_tokens
                    )

                    interleaved_parts = []
                    for t_idx in range(grid_t):
                        # Video tokens for this timestep
                        v_start = t_idx * tokens_per_frame
                        v_end = v_start + tokens_per_frame
                        interleaved_parts.append(video_flat[v_start:v_end])

                        # Audio tokens for this timestep
                        a_count = vid_audio_groups[t_idx]
                        if a_count > 0:
                            a_end = audio_offset + a_count
                            interleaved_parts.append(audio_flat[audio_offset:a_end])
                            audio_offset = a_end

                    combined = torch.cat(interleaved_parts, dim=0)
                    multimodal_embeddings.append(combined)
                else:
                    # Pure video (no audio for this one)
                    multimodal_embeddings.append(video_flat)

            if audio_offset != audio_flat.size(0):
                raise ValueError("Video audio interleaving did not consume its exact feature slice")

        else:
            # Standard case: process each modality independently
            audio_num_segments = mm_input_by_modality.get("_audio_num_segments")
            for modality in mm_input_by_modality:
                if modality.startswith("_"):
                    continue  # Skip internal metadata
                multimodal_input = mm_input_by_modality[modality]
                if modality == "image":
                    image_embeddings = self._process_image_input(multimodal_input)
                    multimodal_embeddings.extend(image_embeddings)
                elif modality == "video":
                    video_embeddings = self._process_video_input(multimodal_input)
                    multimodal_embeddings.extend(video_embeddings)
                elif modality == "audio":
                    audio_embeds = self._process_audio_input(multimodal_input)
                    if audio_embeds is not None:
                        # Flatten to 2D: (total_audio_tokens, hidden_dim)
                        audio_flat = audio_embeds.view(-1, audio_embeds.size(-1))
                        # Split into one tensor per user audio item so vLLM's
                        # embed_multimodal contract (len(mm_embeds) == num
                        # items) is satisfied under multi_audios. Without
                        # splitting a single flat tensor is returned and
                        # vLLM asserts a mismatch.
                        parts = _split_audio_embeds_by_items(
                            audio_flat, audio_num_segments
                        )
                        multimodal_embeddings.extend(parts)

        if not multimodal_embeddings:
            return None

        return multimodal_embeddings

    def get_multimodal_embeddings(self, **kwargs: object):
        return self.embed_multimodal(**kwargs)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: tuple[torch.Tensor, ...] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge multimodal embeddings into text embeddings.

        Replaces placeholder tokens (image_pad, video_pad, audio_pad) with
        their corresponding computed embeddings from the encoders.

        When both vision (image/video) and audio tokens are present, we
        separate the embedding tensors by their hidden dimension size:
        - Vision embeddings have the full LLM hidden_size (e.g., 3584)
        - Audio embeddings have a smaller dimension (e.g., 896 from audio_proj)

        This allows us to correctly route each embedding tensor to its
        corresponding placeholder token positions.
        """
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None:
            return inputs_embeds

        # Determine which token types are present in input_ids
        has_vision_token = (
            (input_ids == self.config.video_token_id)
            | (input_ids == self.config.image_token_id)
        ).any()
        has_audio_token = (
            self.audio_token_id is not None and (input_ids == self.audio_token_id).any()
        )

        if has_vision_token and has_audio_token:
            # Both vision and audio present - separate by embedding dimension
            embedding_shapes = [emb.shape[-1] for emb in multimodal_embeddings]
            unique_shapes = sorted(set(embedding_shapes))

            if len(unique_shapes) == 2:
                # Two distinct dims: smaller = audio, larger = vision
                audio_dim = unique_shapes[0]
                vision_dim = unique_shapes[1]

                audio_embeddings = tuple(
                    emb for emb in multimodal_embeddings if emb.shape[-1] == audio_dim
                )
                vision_embeddings = tuple(
                    emb for emb in multimodal_embeddings if emb.shape[-1] == vision_dim
                )

                # Merge vision embeddings at image_pad and video_pad positions
                if vision_embeddings:
                    inputs_embeds = merge_multimodal_embeddings(
                        input_ids,
                        inputs_embeds,
                        vision_embeddings,
                        [self.config.image_token_id, self.config.video_token_id],
                    )

                # Merge audio embeddings at audio_pad positions
                if audio_embeddings:
                    inputs_embeds = merge_multimodal_embeddings(
                        input_ids,
                        inputs_embeds,
                        audio_embeddings,
                        [self.audio_token_id],
                    )
            elif len(unique_shapes) == 1:
                # Same dimension (audio_proj outputs LLM hidden_size)
                # Merge all at once using all placeholder token IDs
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    multimodal_embeddings,
                    [
                        self.config.image_token_id,
                        self.config.video_token_id,
                        self.audio_token_id,
                    ],
                )
            else:
                # More than 2 unique shapes - unexpected but handle gracefully
                logger.warning(
                    "Unexpected number of unique embedding shapes: %s. "
                    "Falling back to combined merge.",
                    unique_shapes,
                )
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    multimodal_embeddings,
                    [
                        self.config.image_token_id,
                        self.config.video_token_id,
                        self.audio_token_id,
                    ],
                )
        elif has_vision_token:
            # Only vision tokens (no audio)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id],
            )
        elif has_audio_token:
            # Only audio tokens (standalone audio, no vision)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.audio_token_id],
            )
        else:
            # No multimodal tokens found but embeddings provided - log warning
            logger.warning(
                "Multimodal embeddings provided but no placeholder tokens "
                "found in input_ids. Skipping embedding merge."
            )

        return inputs_embeds

    # Keep the explicit embedding entrypoint used by the bypass regression.
    get_input_embeddings = embed_input_ids

    def get_mrope_input_positions(self, input_tokens: list, mm_features: list):
        """Use spatial positions only for visual tokens in paired video/audio."""
        from .wemm_processor import get_wemm_mrope_input_positions

        return get_wemm_mrope_input_positions(input_tokens, mm_features, self.config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["mtp."]
        if self.audio is None:
            skip_prefixes.append("audio.")
        # vLLM 0.31 removed skip_prefixes. Filter after mapping so disabled
        # audio and MTP tensors are skipped even if a matching module exists.
        mapped = self.hf_to_vllm_mapper.apply(weights)
        selected = ((name, data) for name, data in mapped
                    if not name.startswith(tuple(skip_prefixes)))
        return AutoWeightsLoader(self).load_weights(selected)


@MULTIMODAL_REGISTRY.register_processor(
    WeMMMultiModalProcessor,
    info=WeMMProcessingInfo,
    dummy_inputs=WeMMDummyInputsBuilder,
)
class WeMMVideoForConditionalGeneration(
    _WeMMVideoMixin, Qwen3_5ForConditionalGeneration
):
    """WeMM audio/video with the upstream Qwen3.5 dense backbone."""

    _wemm_text_model_type = "qwen3_5_text"


@MULTIMODAL_REGISTRY.register_processor(
    WeMMMultiModalProcessor,
    info=WeMMProcessingInfo,
    dummy_inputs=WeMMDummyInputsBuilder,
)
class WeMMVideoMoeForConditionalGeneration(
    _WeMMVideoMixin, Qwen3_5MoeForConditionalGeneration
):
    """WeMM audio/video with the upstream Qwen3.5 MoE backbone."""

    _wemm_text_model_type = "qwen3_5_moe_text"
