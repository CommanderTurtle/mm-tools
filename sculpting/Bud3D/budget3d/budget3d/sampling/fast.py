"""Optimised inference for the Budget3D-Idol sampler.

    from budget3d.sampling.fast import optimize_for_inference
    optimize_for_inference(student)            # after loading the weights, before sampling

What it switches on (measured in scripts/bench_infer.py; the token layouts and the weights are unchanged):

  * bf16 matmuls inside the transformer blocks, refiners and context/noise refiners -- the precision the
    model was trained in (Idol's own sampler runs these blocks in reduced precision too); the residual
    stream, norms, RoPE angles and output heads stay fp32.
  * TF32 for the remaining fp32 matmuls (input/output layers, adaLN, mixers).
  * fused, tiled attention (FlashAttention-style SDPA kernels; cuDNN's on Hopper): attention is computed
    block by block in on-chip memory, never materialising the full (N + 4102)^2 score matrix.
  * torch.compile of Idol's transformer blocks and RePo layers (and, with ``compile_adapters``, of the
    Budget3D refiners and mixers).  Shapes are static per (batch, token budget), so each budget compiles
    once on its first call and reuses the graph afterwards; the first call per shape is slow.
"""
import torch

ATTN_DEFAULT = "sdpa_cudnn"


def _compile_blocks(student, mode=None):
    """torch.compile Idol's transformer blocks and RePo layers (the same as Budget3DModel.compile_blocks, with a mode).
    RePo is switched to real (cos, sin) rotations first: Inductor cannot compile the complex form."""
    import types
    from ..upstream import idol_layers
    kinds = (idol_layers.UnifiedTransformerBlock, idol_layers.RePo3DRotaryEmbedding)
    compiled = {}
    for mod in student.modules():
        if isinstance(mod, idol_layers.RePo3DRotaryEmbedding):
            mod.real_valued = True
        if isinstance(mod, kinds):
            key = type(mod)
            if key not in compiled:
                compiled[key] = torch.compile(key.forward, dynamic=False, mode=mode)
            mod.forward = types.MethodType(compiled[key], mod)


def optimize_for_inference(student, attention=ATTN_DEFAULT, compile_blocks=True, compile_adapters=True,
                           cache_size_limit=256, mode=None):
    from ..upstream import idol_layers
    for m in student.modules():
        if hasattr(m, "bf16_blocks"):
            m.bf16_blocks = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    idol_layers.ATTN_BACKEND = attention
    if compile_blocks or compile_adapters:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, cache_size_limit)
        if hasattr(dynamo.config, "accumulated_cache_size_limit"):
            dynamo.config.accumulated_cache_size_limit = max(dynamo.config.accumulated_cache_size_limit,
                                                             4 * cache_size_limit)
    if compile_blocks and not getattr(student, "_compiled_blocks", False):
        _compile_blocks(student, mode)
        student._compiled_blocks = True
    if compile_adapters and not getattr(student, "_compiled_adapters", False):
        # the mixers run on the fixed 16^3 grid; the refiners are NOT compiled: their row count (tokens per scale)
        # changes with every new layout, so a static-shape compile would recompile on every relayout (the 5-step
        # turbo sampler relayouts every step), and compiling them gained < 1 % in the H100 ablation anyway
        for m in student.bp_mixers.values():
            m.forward = torch.compile(m.forward, dynamic=False, mode=mode)
        student._compiled_adapters = True
    return student


def describe(attention=ATTN_DEFAULT, compile_blocks=True, compile_adapters=True):
    parts = ["bf16 transformer blocks", "TF32 elsewhere",
             "cuDNN fused attention" if attention == "sdpa_cudnn" else "fused SDPA attention"]
    if compile_blocks:
        parts.append("torch.compile")
    return ", ".join(parts)
