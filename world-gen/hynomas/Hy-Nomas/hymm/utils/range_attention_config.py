"""Configuration and runtime capabilities for offline Flex range attention."""

import os

from hymm.models.dit.modules.uniform_history_cache import HISTORY_SAMPLINGS


def validate_range_attention_config(args, *, mode="inference"):
    from hymm.models.dit.modules.kv_cache_storage import validate_kv_cache_config

    validate_kv_cache_config(args)
    inf = args.inference
    cfg = getattr(inf, "sparse_attention", {})
    if type(cfg.get("memory_efficient", False)) is not bool:
        raise ValueError("sparse_attention.memory_efficient must be boolean")
    if cfg.get("memory_cache_device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("sparse_attention.memory_cache_device must be cpu or cuda")
    if type(cfg.get("memory_attention_heads", 4)) is not int or cfg.get("memory_attention_heads", 4) <= 0:
        raise ValueError("sparse_attention.memory_attention_heads must be a positive integer")
    if type(cfg.get("memory_adaptive_heads", False)) is not bool:
        raise ValueError("sparse_attention.memory_adaptive_heads must be boolean")
    refresh = cfg.get("route_refresh_steps", [])
    if cfg.get("enabled", False) and refresh:
        values = list(refresh)
        if (any(type(x) is not int or x <= 0 for x in values)
                or values != sorted(set(values))):
            raise ValueError("route_refresh_steps must be sorted unique positive 1-based steps")
        if values[0] != int(cfg.get("dense_first_n_steps", 0)) + 1:
            raise ValueError("First route refresh must be the first Sparse denoising step")
        if values[-1] > int(inf.infer_steps_video):
            raise ValueError("route_refresh_steps exceeds infer_steps_video")
        if (inf.rollout_type != "seg_ar"
                or inf.kv_cache_quantization not in ("none", "int8")
                or (inf.sliding_window.enabled and
                    (inf.sliding_window.sampling != "recent" or inf.cfg_scale_video != 1.0
                     or inf.camera_cfg_scale != 1.0 or inf.max_cache_segments is not None))):
            raise ValueError("Route reuse requires seg_ar Sparse, none/int8 KV and recent SWA with CFG=1")
    backend = getattr(args.dit, "seg_ar_attention_backend", "magi")
    if backend not in ("magi", "flex"):
        raise ValueError("model.dit.seg_ar_attention_backend must be magi or flex")
    if backend == "magi":
        return
    if mode != "inference":
        raise ValueError("Flex range attention supports offline inference only")
    if args.dit.master_weight_type != "bf16":
        raise ValueError("Flex range attention requires BF16 compute weights")
    if args.dit.attn_mode != "magi_mp_causal":
        raise ValueError("Flex keeps attn_mode=magi_mp_causal for segment masking semantics")
    inf = args.inference
    if getattr(inf, "rollout_type", None) not in ("seg_ar", "chunk_ar", "one_shot"):
        raise ValueError("Flex supports offline seg_ar/chunk_ar/one_shot only")
    if getattr(args.parallelism, "use_dynamic_ring_attention", False):
        raise ValueError("Flex does not support dynamic ring attention")
    if os.getenv("HY_DIT_COMPILE_PREPOST", "0") != "0":
        raise ValueError("Use inference.torch_compile instead of HY_DIT_COMPILE_PREPOST")
    # Public W8/KV quantization is storage-only. Both reach attention as BF16;
    # public pre/post compilation keeps attention/cache mutation outside graphs.
    if os.getenv("HY_AR_KV_INT8_STORAGE", "0") != "0":
        raise ValueError("P3 requires HY_AR_KV_INT8_STORAGE unset/0")
    window = getattr(inf, "sliding_window", {})
    if window.get("enabled", False) and window.get("sampling", "recent") not in HISTORY_SAMPLINGS:
        raise ValueError(f"Flex SWA sampling must be one of {HISTORY_SAMPLINGS}")
    # Segment VAE/PixelUMM decode policy is validated by the causal pipeline and
    # does not select the DiT attention backend.
    for name in ("interactive_kv_cache_layerwise_cpu_offload", "preserve_kv_cache"):
        if getattr(inf, name, False):
            raise ValueError(f"Flex offline attention cannot be combined with {name}")


def validate_range_attention_device(args, device):
    """Separate software selection from the device capability check."""
    import torch
    validate_range_attention_config(args)
    device = torch.device(device)
    if device.type != "cuda":
        if getattr(args.dit, "seg_ar_attention_backend", "magi") == "flex":
            raise ValueError("Flex range attention requires CUDA")
        return
    capability = torch.cuda.get_device_capability(device)
    if getattr(args.dit, "seg_ar_attention_backend", "magi") == "flex":
        from torch.nn.attention.flex_attention import flex_attention
        if not callable(flex_attention):
            raise RuntimeError("This PyTorch build does not provide FlexAttention")
        if capability < (8, 0):
            raise ValueError("Flex range attention requires an SM80-or-newer GPU")
    else:
        from hymm.models.dit.modules.attention.backends import magi_kernel
        if capability < (9, 0):
            raise ValueError("The Magi backend requires Hopper/SM90 or newer; select flex on SM89")
        magi_kernel()
