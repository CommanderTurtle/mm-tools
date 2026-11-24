# Copyright 2026 Adventists AI. Apache-2.0.
"""Transformers implementation of the Sophon-Realtime thinker (DuplexJev checkpoint, `model_type: ultravox`).

Qwen3-ASR audio encoder -> Ultravox-style projector (stack 2, RMSNorm, SwiGLU) -> Qwen3 LLM.
Same maths as the vLLM plugin (`duplexjev_vllm`) and the training code used for the tag LoRA.
Parameter names match the released safetensors exactly, so `from_pretrained` loads the checkpoint as is.

Audio enters as `audio_values` (mel chunks, <=30 s each), `audio_lens` (valid mel frames per chunk) and
`audio_token_len` (LLM tokens per chunk); the prompt holds `audio_token_index` once per audio token,
in chunk order. Those positions are overwritten with the projected audio embeddings.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, GenerationMixin, PretrainedConfig, PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

from .configuration_qwen3_asr_encoder import Qwen3ASREncoderConfig
from .modeling_qwen3_asr_encoder import Qwen3ASREncoder, _get_feat_extract_output_lengths


class DuplexJevConfig(PretrainedConfig):
    model_type = "ultravox"
    sub_configs = {"text_config": Qwen3Config, "audio_config": Qwen3ASREncoderConfig}

    def __init__(self, text_config=None, audio_config=None, audio_token_index=151669, stack_factor=2,
                 hidden_size=4096, projector_act="swiglu", projector_ln_mid=True, norm_init=0.4,
                 audio_full_length=True, ignore_index=-100, **kwargs):
        if isinstance(text_config, dict):
            text_config = {k: v for k, v in text_config.items() if k not in ("architectures",)}
            text_config.pop("model_type", None)
            text_config = Qwen3Config(**text_config)
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASREncoderConfig(**{k: v for k, v in audio_config.items() if k != "model_type"})
        self.text_config = text_config if text_config is not None else Qwen3Config()
        self.audio_config = audio_config if audio_config is not None else Qwen3ASREncoderConfig()
        self.audio_token_index = audio_token_index
        self.stack_factor = stack_factor
        self.hidden_size = hidden_size  # projector width (before SwiGLU halves it), not the LLM width
        self.projector_act = projector_act
        self.projector_ln_mid = projector_ln_mid
        self.norm_init = norm_init
        self.audio_full_length = audio_full_length
        self.ignore_index = ignore_index
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    # trainers / ms-swift read these from the top-level config
    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @vocab_size.setter
    def vocab_size(self, v):
        pass

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers


class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


class DuplexJevProjector(nn.Module):
    def __init__(self, config: DuplexJevConfig):
        super().__init__()
        self.stack = config.stack_factor
        din = config.audio_config.d_model * self.stack
        hid = config.hidden_size
        self.ln_pre = _RMSNorm(din)
        self.linear_1 = nn.Linear(din, hid, bias=False)
        self.ln_mid = _RMSNorm(hid // 2)
        self.linear_2 = nn.Linear(hid // 2, config.text_config.hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        Tp = -(-T // self.stack) * self.stack
        x = F.pad(x, (0, 0, 0, Tp - T)).reshape(B, Tp // self.stack, C * self.stack)
        h = self.linear_1(self.ln_pre(x))
        a, g = h.chunk(2, -1)  # vLLM MulAndSilu: x[:d] * silu(x[d:])
        return self.linear_2(self.ln_mid(a * F.silu(g)))


class DuplexJevPreTrainedModel(PreTrainedModel):
    config_class = DuplexJevConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer", "Qwen3ASRAudioEncoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        # Only the projector norms are ours; everything is loaded from the checkpoint.
        if isinstance(module, _RMSNorm):
            with torch.no_grad():
                module.weight.fill_(1.0)


class DuplexJevForConditionalGeneration(DuplexJevPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}

    def __init__(self, config: DuplexJevConfig):
        super().__init__(config)
        self.audio_tower = Qwen3ASREncoder(config.audio_config)
        self.multi_modal_projector = DuplexJevProjector(config)
        self.language_model = Qwen3ForCausalLM(config.text_config)
        self.post_init()

    # --- plumbing ---------------------------------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _set_gradient_checkpointing(self, *args, **kwargs):  # transformers < 5
        return self.language_model._set_gradient_checkpointing(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # The audio tower is frozen in every supported recipe; checkpoint the LLM only.
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return self.language_model.is_gradient_checkpointing

    # --- audio ---------------------------------------------------------------------------------------------------
    def get_audio_embeds(self, audio_values: torch.Tensor, audio_lens: torch.Tensor,
                         audio_token_len: torch.Tensor) -> torch.Tensor:
        """(chunks, mel, T) -> (sum(audio_token_len), llm_hidden), in chunk order."""
        tower = self.audio_tower
        x = audio_values.to(device=self.multi_modal_projector.linear_1.weight.device,
                            dtype=self.multi_modal_projector.linear_1.weight.dtype)
        h = tower(x, audio_len=audio_lens.to(x.device)).last_hidden_state
        # pad every call to the length implied by the padded mel length (same as the vLLM tower)
        n = int(_get_feat_extract_output_lengths(torch.tensor(x.shape[-1]), tower.n_window))
        if h.shape[1] < n:
            h = F.pad(h, (0, 0, 0, n - h.shape[1]))
        h = self.multi_modal_projector(h[:, :n])
        tl = audio_token_len.to(h.device)
        mask = torch.arange(h.shape[1], device=h.device)[None, :] < tl[:, None]
        return h[mask]

    def _merge_audio(self, input_ids, inputs_embeds, audio_values, audio_lens, audio_token_len):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if audio_values is None or input_ids is None:
            return inputs_embeds
        pos = input_ids == self.config.audio_token_index
        n_pos = int(pos.sum())
        if n_pos == 0:
            return inputs_embeds
        a = self.get_audio_embeds(audio_values, audio_lens, audio_token_len)
        if a.shape[0] != n_pos:
            raise ValueError(f"audio tokens in prompt ({n_pos}) != audio embeddings ({a.shape[0]})")
        a = a.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(pos.to(inputs_embeds.device)[..., None].expand_as(inputs_embeds), a)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                audio_values: Optional[torch.Tensor] = None, audio_lens: Optional[torch.Tensor] = None,
                audio_token_len: Optional[torch.Tensor] = None, **kwargs):
        if input_ids is not None and (inputs_embeds is None or audio_values is not None):
            inputs_embeds = self._merge_audio(input_ids, inputs_embeds, audio_values, audio_lens, audio_token_len)
        return self.language_model(input_ids=None, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                   use_cache=use_cache, **kwargs)

    def prepare_inputs_for_generation(self, *args, audio_values=None, audio_lens=None, audio_token_len=None,
                                      is_first_iteration=False, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, is_first_iteration=is_first_iteration, **kwargs)
        if is_first_iteration or not kwargs.get("use_cache", True):
            inputs.update(audio_values=audio_values, audio_lens=audio_lens, audio_token_len=audio_token_len)
        return inputs


def register_auto():
    """Make `AutoConfig` / `AutoModelForCausalLM` resolve the released checkpoint."""
    from transformers import AutoModelForCausalLM
    try:
        AutoConfig.register("ultravox", DuplexJevConfig, exist_ok=True)
    except TypeError:  # very old transformers
        try:
            AutoConfig.register("ultravox", DuplexJevConfig)
        except ValueError:
            pass
    try:
        AutoConfig.register("qwen3_asr_encoder_portable", Qwen3ASREncoderConfig, exist_ok=True)
    except Exception:
        pass
    try:
        AutoModelForCausalLM.register(DuplexJevConfig, DuplexJevForConditionalGeneration, exist_ok=True)
    except Exception:
        pass
