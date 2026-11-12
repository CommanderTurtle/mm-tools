"""
Utilities package for Hy-Nomas training and inference.

This package contains various utility modules for training stability monitoring,
data processing, distributed training, and other helper functions.
"""

try:
    from hymm.models.dit.modules.attention import dynamic_ring_attention
    from hymm.models.dit.modules.attention.dynamic_ring_attention import (
        attn_forward_func,
    )
except ModuleNotFoundError as exc:
    if exc.name != "flash_attn":
        raise
    dynamic_ring_attention = None
    attn_forward_func = None
