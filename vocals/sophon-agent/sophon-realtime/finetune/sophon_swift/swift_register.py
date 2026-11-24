# Copyright 2026 Adventists AI. Apache-2.0.
"""ms-swift support for Sophon-Realtime v1 (fine-tune the thinker's LLM with LoRA; audio encoder and
projector stay frozen).

Use with ms-swift >= 4.6:
    swift sft --custom_register_path /path/to/sophon_swift/swift_register.py --model <Sophon-Realtime-v1 dir> ...

* model_type `sophon_realtime`, template `sophon_realtime`.
* `--model` may point at the repo root (weights are read from `base/`) or at `base/` itself.
* Audio: put `<audio>` in a message (user or assistant turn) and list the files in `audios`.
  Audio inside an assistant turn is context only (never a training target).
"""
import os
import sys
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sophon_swift.modeling_duplexjev import (DuplexJevConfig, DuplexJevForConditionalGeneration,  # noqa: E402
                                             register_auto)
from sophon_swift.processing import SAMPLING_RATE, SophonRealtimeProcessor  # noqa: E402

from swift.model import (Model, ModelGroup, ModelLoader, ModelMeta, MultiModelKeys,  # noqa: E402
                         register_model, register_model_arch)
from swift.template import Template, TemplateMeta, register_template  # noqa: E402
from swift.template.vision_utils import load_audio  # noqa: E402

MODEL_TYPE = TEMPLATE_TYPE = ARCH = "sophon_realtime"
AUDIO_TOKEN_INDEX = 151669
REPO_IDS = ("AdventistsAI/Sophon-Realtime-v1", "adventists-ai/Sophon-Realtime-v1")

register_auto()


# ---------------------------------------------------------------------------------------------- repo layout
def _thinker_dir(model_dir: str) -> str:
    """The v1 repo keeps the shared 32B weights under base/; accept either the root or base/."""
    if model_dir and not os.path.isfile(os.path.join(model_dir, "config.json")) and os.path.isfile(
            os.path.join(model_dir, "base", "config.json")):
        return os.path.join(model_dir, "base")
    return model_dir


def _patch_snapshot_download():
    import swift.model.model_meta as mm
    if getattr(mm.safe_snapshot_download, "_sophon_patched", False):
        return
    orig = mm.safe_snapshot_download

    def wrapped(*args, **kwargs):
        return _thinker_dir(orig(*args, **kwargs))

    wrapped._sophon_patched = True
    mm.safe_snapshot_download = wrapped


_patch_snapshot_download()


# ---------------------------------------------------------------------------------------------- model
class SophonRealtimeLoader(ModelLoader):

    def get_config(self, model_dir: str):
        return DuplexJevConfig.from_pretrained(_thinker_dir(model_dir))

    def get_processor(self, model_dir: str, config):
        return SophonRealtimeProcessor.from_pretrained(_thinker_dir(model_dir), config)

    def get_model(self, model_dir: str, *args, **kwargs):
        self.auto_model_cls = self.auto_model_cls or DuplexJevForConditionalGeneration
        return super().get_model(_thinker_dir(model_dir), *args, **kwargs)


register_model_arch(
    MultiModelKeys(ARCH, language_model="language_model", aligner="multi_modal_projector",
                   vision_tower="audio_tower"),
    exist_ok=True)

register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([Model(REPO_IDS[0], REPO_IDS[1])],
                    ignore_patterns=["talker/*", "e2/*", "engine/*", "voices/*", "assets/*"])],
        SophonRealtimeLoader,
        template=TEMPLATE_TYPE,
        model_arch=ARCH,
        architectures=["DuplexJevForConditionalGeneration"],
        is_multimodal=True,
        tags=["audio"],
        requires=["transformers>=4.51", "librosa"],
    ),
    exist_ok=True)


# ---------------------------------------------------------------------------------------------- template
class SophonRealtimeTemplate(Template):
    """ChatML exactly as the released tokenizer renders it (no think block, no default system prompt).

    Each `<audio>` becomes N copies of `<|audio|>` (6.25 per second of audio); the model swaps them for the
    projected audio embeddings. Audio positions never carry labels.
    """
    use_model = False
    support_padding_free = False

    def replace_tag(self, media_type: Literal["image", "video", "audio"], index: int, inputs) -> List:
        assert media_type == "audio", f"Sophon-Realtime only takes audio, got {media_type}"
        return [[AUDIO_TOKEN_INDEX]]

    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale", None)
        idx_list = [i for i, t in enumerate(input_ids) if t == AUDIO_TOKEN_INDEX]
        if inputs.audios:
            if len(idx_list) != len(inputs.audios):
                raise ValueError(f"{len(idx_list)} <audio> tags but {len(inputs.audios)} audio files")
            sr = inputs.chat_template_kwargs.get("sampling_rate") or SAMPLING_RATE
            wavs = [load_audio(a, sampling_rate=sr) for a in inputs.audios]
            if sr != SAMPLING_RATE:
                raise ValueError("Sophon-Realtime expects 16 kHz audio")
            feats = self.processor.audio_features(wavs)
            n_tok = feats.pop("tokens_per_clip")
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda i: [AUDIO_TOKEN_INDEX] * n_tok[i])
            encoded.update(feats)
        elif idx_list:
            raise ValueError("<audio> tag without an audio file")
        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        if loss_scale is not None:
            encoded["loss_scale"] = loss_scale
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        vals = [b["audio_values"] for b in batch if b.get("audio_values") is not None]
        if vals:
            T = max(v.shape[-1] for v in vals)
            res["audio_values"] = torch.cat([F.pad(v, (0, T - v.shape[-1])) for v in vals])
            res["audio_lens"] = torch.cat([b["audio_lens"] for b in batch if b.get("audio_lens") is not None])
            res["audio_token_len"] = torch.cat(
                [b["audio_token_len"] for b in batch if b.get("audio_token_len") is not None])
        return res


register_template(
    TemplateMeta(
        TEMPLATE_TYPE,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        default_system=None,
        auto_add_bos=False,
        stop_words=["<|endoftext|>"],
        agent_template="hermes",
        template_cls=SophonRealtimeTemplate,
    ),
    exist_ok=True)
