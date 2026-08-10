"""AudioDiT — Conditional Flow Matching TTS with DiT backbone.

Auto-registers AudioDiTConfig and AudioDiTModel with HuggingFace Transformers
so that ``AutoConfig.from_pretrained`` and ``AutoModel.from_pretrained`` work
out of the box.

Usage:
    import audiodit  # triggers auto-registration
    from audiodit import AudioDiTConfig, AudioDiTModel

    model = AudioDiTModel.from_pretrained("path/to/hf_audiodit_1b")
"""

from .configuration_audiodit import AudioDiTConfig, AudioDiTVaeConfig
from .modeling_audiodit import (
    AudioDiTModel,
    AudioDiTPreTrainedModel,
    AudioDiTTransformer,
    AudioDiTVae,
    AudioDiTOutput,
)

# Auto-register with transformers so AutoConfig/AutoModel work
from transformers import AutoConfig, AutoModel

AutoConfig.register("audiodit", AudioDiTConfig, exist_ok=True)
AutoModel.register(AudioDiTConfig, AudioDiTModel, exist_ok=True)

__all__ = [
    "AudioDiTConfig",
    "AudioDiTVaeConfig",
    "AudioDiTModel",
    "AudioDiTPreTrainedModel",
    "AudioDiTTransformer",
    "AudioDiTVae",
    "AudioDiTOutput",
]
