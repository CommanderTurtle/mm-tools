"""Final decoder configuration for the WeVisOne-VL v1 checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


ARCHITECTURE_VERSION = "wevisone_vl_v1"


@dataclass
class Qwen3VLSegConfig:
    """Only fields that instantiate the released 9.6M-parameter decoder."""

    architecture_version: str = ARCHITECTURE_VERSION
    vision_feature_layers: Tuple[int, ...] = (5, 11, 17, 23)

    hidden_dim: int = 192
    mask_feature_dim: int = 64
    num_fourier_freqs: int = 32

    msda_encoder_layers: int = 3
    msda_num_levels: int = 4
    msda_num_points: int = 4

    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_ffn_dim: int = 768
    transformer_dropout: float = 0.0

    group_norm_groups: int = 32
    stem_mid_dim: int = 128
    alpha_cascade_dim: int = 48
    alpha_cascade_blocks: int = 2
    cnn_image_size: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "Qwen3VLSegConfig":
        if not values:
            return cls()
        known = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in values.items() if key in known})


SEG_SPECIAL_TOKENS = ("<OBJ>",)
