# Copyright 2026 Adventists AI. Apache-2.0.
"""Sophon-Realtime v1 thinker for transformers / ms-swift (LoRA fine-tuning of the LLM backbone)."""
__version__ = "0.1.0"

from .modeling_duplexjev import DuplexJevConfig, DuplexJevForConditionalGeneration, register_auto  # noqa: F401
from .processing import SophonRealtimeProcessor  # noqa: F401
