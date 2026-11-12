# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""vLLM model-config updates for the composed SpeechLM architecture."""

from vllm.model_executor.models.config import NemotronHForCausalLMConfig, VerifyAndUpdateConfig


class NeMoSpeechLMForConditionalGenerationConfig(VerifyAndUpdateConfig):
    """Delegate hybrid-backbone cache defaults through the Speech wrapper."""

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if not getattr(hf_config, "is_hybrid", False):
            return

        NemotronHForCausalLMConfig.update_mamba_ssm_cache_dtype(
            cache_config=vllm_config.cache_config,
            hf_config=hf_config.text_config,
        )
