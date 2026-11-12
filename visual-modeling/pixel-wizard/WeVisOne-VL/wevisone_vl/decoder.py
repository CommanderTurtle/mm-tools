"""Construction helpers for the inference-only alpha-matte decoder."""

from __future__ import annotations

from typing import List, Optional

from .config import ARCHITECTURE_VERSION, Qwen3VLSegConfig
from .mask_decoder import WeVisOneMaskDecoder


def build_mask_decoder_from_qwen(
    qwen,
    cfg: Optional[Qwen3VLSegConfig] = None,
    visual_in_dims: Optional[List[int]] = None,
) -> WeVisOneMaskDecoder:
    """Build the decoder using dimensions from a loaded Qwen3-VL model."""
    cfg = cfg or Qwen3VLSegConfig()
    if cfg.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError(f"unsupported decoder architecture: {cfg.architecture_version}")
    config = qwen.config
    llm_dim = int(config.text_config.hidden_size)
    if visual_in_dims is None:
        visual_dim = int(config.vision_config.hidden_size)
        visual_in_dims = [visual_dim] * len(cfg.vision_feature_layers)
    return WeVisOneMaskDecoder(
        visual_in_dims=visual_in_dims,
        llm_dim=llm_dim,
        cfg=cfg,
    )
