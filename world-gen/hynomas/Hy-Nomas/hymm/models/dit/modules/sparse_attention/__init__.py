from .ssta_attention import ssta_3d_attention

try:
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    flex_mask_cache = {}
