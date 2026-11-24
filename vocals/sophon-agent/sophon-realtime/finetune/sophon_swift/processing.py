# Copyright 2026 Adventists AI. Apache-2.0.
"""Audio preprocessing for the Sophon-Realtime thinker (same maths as `duplexjev_vllm.processing`).

Whisper-style 128-bin log-mel (16 kHz, hop 160), every clip right-padded with silence to a whole number of
audio tokens, then cut into 30 s chunks. One LLM token per `encoder_ds_factor * stack_factor` = 16 mel frames
(6.25 tokens / s).
"""
import os
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, WhisperFeatureExtractor

HOP = 160
SAMPLING_RATE = 16000


class SophonRealtimeProcessor:
    """Holds the tokenizer and the feature extractor; `audio_features()` turns waveforms into model inputs."""

    def __init__(self, tokenizer, feature_extractor, *, stack_factor: int = 2, encoder_ds_factor: int = 8,
                 audio_context_size: int = 3000, audio_token_index: int = 151669):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.audio_processor = feature_extractor
        self.stack_factor = stack_factor
        self.encoder_ds_factor = encoder_ds_factor
        self.audio_context_size = audio_context_size
        self.audio_token_index = audio_token_index
        self.sampling_rate = SAMPLING_RATE

    @classmethod
    def from_pretrained(cls, model_dir: str, config=None, **kwargs):
        tok = AutoTokenizer.from_pretrained(model_dir)
        fe = WhisperFeatureExtractor.from_pretrained(model_dir)
        kw = {}
        if config is not None:
            kw = dict(stack_factor=config.stack_factor, audio_token_index=config.audio_token_index,
                      encoder_ds_factor=int(getattr(config.audio_config, "encoder_ds_factor", 8)))
        return cls(tok, fe, **kw)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)

    # tokenizer passthrough (some trainers call processor.decode / batch_decode / pad_token_id ...)
    def __getattr__(self, name):
        if name in ("tokenizer", "feature_extractor", "audio_processor"):
            raise AttributeError(name)
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def _align(self, a: np.ndarray) -> np.ndarray:
        step = HOP * self.encoder_ds_factor * self.stack_factor
        n = max(len(a), 2 * HOP)
        n = -(-n // step) * step
        return np.pad(a, (0, n - len(a))) if n > len(a) else a

    def audio_features(self, audios: Sequence[np.ndarray]) -> dict:
        """Waveforms (16 kHz mono float) -> audio_values / audio_lens / audio_token_len + tokens per clip."""
        audios = [self._align(np.asarray(a, dtype=np.float32)) for a in audios]
        x = self.feature_extractor(audios, sampling_rate=SAMPLING_RATE, padding="longest", pad_to_multiple_of=HOP,
                                   truncation=False, return_attention_mask=True, return_tensors="pt")
        values = torch.as_tensor(x.input_features)
        lens = torch.as_tensor(x.attention_mask).sum(-1)
        ctx = self.audio_context_size or values.shape[-1]
        chunks: List[torch.Tensor] = []
        clens: List[int] = []
        per_clip: List[int] = []
        div = self.encoder_ds_factor * self.stack_factor
        for i in range(values.shape[0]):
            L = int(lens[i])
            n_tok = 0
            for off in range(0, L, ctx):
                c = values[i, :, off:off + ctx]
                if off > 0 and c.shape[-1] < ctx:
                    c = F.pad(c, (0, ctx - c.shape[-1]))
                chunks.append(c)
                cl = min(L - off, ctx)
                clens.append(cl)
                n_tok += -(-cl // div)
            per_clip.append(n_tok)
        T = max(c.shape[-1] for c in chunks)
        chunks = [F.pad(c, (0, T - c.shape[-1])) for c in chunks]
        audio_lens = torch.tensor(clens, dtype=torch.int64)
        return {
            "audio_values": torch.stack(chunks, 0),
            "audio_lens": audio_lens,
            "audio_token_len": torch.ceil(audio_lens / div).to(torch.int64),
            "tokens_per_clip": per_clip,
        }
