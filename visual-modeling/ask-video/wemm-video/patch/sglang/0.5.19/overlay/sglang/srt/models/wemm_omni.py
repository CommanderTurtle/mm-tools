# SPDX-License-Identifier: Apache-2.0
"""WeMM-Video 1.5 on the native SGLang Qwen3.5 language and vision paths.

The audio implementation originates in the APEX 0.5.16 reference image.
The public architecture matches the existing HF/vLLM release checkpoint.
"""

import logging
import os

import torch

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.wemm_whisper import WeMMAudioEncoder

logger = logging.getLogger(__name__)


class _WeMMVideoMixin:
    def __init__(self, config, quant_config=None, prefix=""):
        # Transformers may add expert defaults to dense text configs as well.
        if config.text_config.model_type != self._wemm_text_model_type:
            raise ValueError(
                f"{type(self).__name__} requires text_config.model_type="
                f"{self._wemm_text_model_type!r}, got {config.text_config.model_type!r}"
            )
        super().__init__(config, quant_config, prefix)
        if config.audio_config is None:
            raise ValueError("WeMM requires audio_config from the release checkpoint")
        self.audio_token_id = config.audio_token_id
        self.audio_segment_batch_size = int(
            os.environ.get("WEMM_AUDIO_SEGMENT_BATCH_SIZE", "8")
        )
        if self.audio_segment_batch_size <= 0:
            raise ValueError("WEMM_AUDIO_SEGMENT_BATCH_SIZE must be positive")
        self.audio = WeMMAudioEncoder(audio_config=config.audio_config).to(
            dtype=next(self.parameters()).dtype
        )
        logger.info(
            "WeMM native model=%s, language_model=%s, text_model_type=%s, experts=%s",
            type(self).__name__, type(self.model).__name__, config.text_config.model_type,
            config.text_config.num_experts if self._wemm_text_model_type == "qwen3_5_moe_text" else 0,
        )

    def get_audio_feature(self, items):
        parameter = next(self.audio.parameters())
        embeddings = []
        for item in items:
            item_embeddings = []
            feature = torch.as_tensor(item.feature)
            if feature.ndim == 4 and feature.shape[0] == 1:
                feature = feature.squeeze(0)
            if feature.ndim == 2:
                feature = feature.unsqueeze(0)
            if feature.ndim != 3 or feature.shape[0] == 0:
                raise ValueError(
                    f"Expected nonempty [segments, mel, time], got {feature.shape}"
                )
            for chunk in feature.split(self.audio_segment_batch_size):
                encoded = self.audio(chunk.to(parameter.device, parameter.dtype))
                item_embeddings.append(encoded.flatten(0, 1))
            actual = sum(value.shape[0] for value in item_embeddings)
            offsets = getattr(item, "offsets", None)
            if offsets is not None:
                expected = sum(end - start + 1 for start, end in offsets)
                if actual != expected:
                    raise ValueError(
                        f"Audio embeddings and this item's placeholders differ: "
                        f"encoder={actual}, placeholders={expected}"
                    )
            embeddings.extend(item_embeddings)
        return torch.cat(embeddings)

    def load_weights(self, weights):
        audio_state = {}

        def backbone_weights():
            for name, weight in weights:
                if not name.startswith("model.audio."):
                    yield name, weight
                    continue
                name = name.removeprefix("model.audio.")
                if name.endswith("embeddings.position_ids"):
                    continue
                name = name.replace("qformer.bert.", "qformer.")
                name = name.replace("LayerNorm.gamma", "LayerNorm.weight")
                name = name.replace("LayerNorm.beta", "LayerNorm.bias")
                if name in audio_state:
                    raise ValueError(f"Duplicate audio checkpoint parameter: {name}")
                audio_state[name] = weight

        loaded = super().load_weights(backbone_weights())
        missing, unexpected = self.audio.load_state_dict(audio_state, strict=False)
        # Older checkpoints omit the deterministic sinusoidal position table.
        missing = [
            name
            for name in missing
            if name not in ("embed_positions.weight", "qformer.embeddings.position_ids")
        ]
        if missing or unexpected:
            raise ValueError(
                f"Incomplete WeMM audio weights: missing={missing}, unexpected={unexpected}"
            )
        logger.info(
            "WeMM audio weights loaded: %d tensors, missing=0, unexpected=0",
            len(audio_state),
        )
        loaded.update("audio." + name for name in audio_state)
        return loaded


class WeMMVideoForConditionalGeneration(_WeMMVideoMixin, Qwen3_5ForConditionalGeneration):
    _wemm_text_model_type = "qwen3_5_text"


class WeMMVideoMoeForConditionalGeneration(_WeMMVideoMixin, Qwen3_5MoeForConditionalGeneration):
    _wemm_text_model_type = "qwen3_5_moe_text"


EntryClass = [WeMMVideoForConditionalGeneration, WeMMVideoMoeForConditionalGeneration]
