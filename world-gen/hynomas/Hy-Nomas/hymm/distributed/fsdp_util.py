import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
)
from torch.utils.checkpoint import (
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
    checkpoint as torch_checkpoint,
    noop_context_fn,
)

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:
    TorchDispatchMode = None
from peft.utils.other import fsdp_auto_wrap_policy

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from hymm.models.checkpointing.load import get_no_split_modules
from hymm.distributed.activation_offload import get_offload_manager

from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import functools


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    # checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT,
)

# Lazy cache for the FA3 forward op reference (populated on first use)
_FA3_FWD_OP = None
_FA3_FWD_OP_INITIALIZED = False

# Lazy cache for the MagiAttention forward op reference (populated on first use)
_MAGI_FWD_OP = None
_MAGI_FWD_OP_INITIALIZED = False


def _get_fa3_fwd_op():
    """Return torch.ops.flash_attn_3._flash_attn_forward.default, or None if unavailable."""
    global _FA3_FWD_OP, _FA3_FWD_OP_INITIALIZED
    if _FA3_FWD_OP_INITIALIZED:
        return _FA3_FWD_OP
    _FA3_FWD_OP_INITIALIZED = True
    try:
        # flash_attn_3._C registers the low-level fwd/bwd kernels.
        # flash_attn_interface registers the high-level custom op
        # `flash_attn_3::_flash_attn_forward`; try to ensure it is imported.
        try:
            import flash_attn_3._C  # noqa: F401
        except ImportError:
            pass
        try:
            import flash_attn_interface  # noqa: F401
        except ImportError:
            pass
        _FA3_FWD_OP = torch.ops.flash_attn_3._flash_attn_forward.default
    except AttributeError:
        _FA3_FWD_OP = None
    return _FA3_FWD_OP


def _get_magi_fwd_op():
    """Return torch.ops.flex_flash_attn._flex_flash_attn_forward_compilable.default, or None.

    magi_attention wraps its CUDA kernel in a torch.library.custom_op named
    ``flex_flash_attn::_flex_flash_attn_forward_compilable``.  The outer
    ``FlexFlashAttnFunc`` is a ``torch.autograd.Function`` (not interceptable
    by TorchDispatchMode), but the inner compilable op IS interceptable.
    Importing ``magi_attention.functional`` triggers op registration.
    """
    global _MAGI_FWD_OP, _MAGI_FWD_OP_INITIALIZED
    if _MAGI_FWD_OP_INITIALIZED:
        return _MAGI_FWD_OP
    _MAGI_FWD_OP_INITIALIZED = True
    try:
        try:
            import magi_attention.functional  # noqa: F401  triggers custom_op registration
        except ImportError:
            pass
        _MAGI_FWD_OP = (
            torch.ops.flex_flash_attn._flex_flash_attn_forward_compilable.default
        )
    except AttributeError:
        _MAGI_FWD_OP = None
    return _MAGI_FWD_OP


# Indices of out_ and lse in _flex_flash_attn_forward_compilable positional args:
# (q[0], k[1], v[2], sink[3], sink_layout[4:str], out_[5], lse[6], q_ranges[7], ...)
_MAGI_COMPILABLE_OUT_IDX = 5
_MAGI_COMPILABLE_LSE_IDX = 6


def _get_out_lse_from_magi_args(args, kwargs):
    """Get out_ and lse tensor refs from magi compilable op (args, kwargs)."""
    kwargs = kwargs or {}
    out_t = kwargs.get("out_")
    lse_t = kwargs.get("lse")
    if out_t is None and len(args) > _MAGI_COMPILABLE_OUT_IDX:
        out_t = args[_MAGI_COMPILABLE_OUT_IDX]
    if lse_t is None and len(args) > _MAGI_COMPILABLE_LSE_IDX:
        lse_t = args[_MAGI_COMPILABLE_LSE_IDX]
    return out_t, lse_t


def _is_fa3_op(func, fa3_fwd_op):
    return (fa3_fwd_op is not None and func is fa3_fwd_op) or (
        "flash_attn_3" in str(func) and "_flash_attn_forward" in str(func)
    )


def _is_magi_op(func, magi_fwd_op):
    return (magi_fwd_op is not None and func is magi_fwd_op) or (
        "flex_flash_attn" in str(func)
        and "_flex_flash_attn_forward_compilable" in str(func)
    )


def _build_attn_skip_context_fn():
    """
    Build a context_fn for selective checkpointing that skips attention forward
    recomputation for both FA3 (flash3_mp_causal) and Magi (magi_mp_causal).

    FA3: the op returns (out, lse); we cache that and replay it on recompute.

    Magi: the compilable op uses mutates_args(out_, lse) and returns None.
    We run the op on forward, then cache (out_.clone(), lse.clone()). On
    recompute we copy the cached tensors into the passed-out_/lse and return
    None, so the kernel is skipped but the buffers are correctly filled.

    Supported ops
    -------------
    - FA3:  flash_attn_3::_flash_attn_forward
    - Magi: flex_flash_attn::_flex_flash_attn_forward_compilable
    """
    fa3_fwd_op = _get_fa3_fwd_op()
    magi_fwd_op = _get_magi_fwd_op()
    print(f"[SAC] FA3  op resolved : {fa3_fwd_op}")
    print(f"[SAC] Magi op resolved : {magi_fwd_op}")

    if TorchDispatchMode is None:
        # Fallback: FA3 only (standard SAC; magi causes NaN)
        def policy_fn(ctx, func, *args, **kwargs):
            if not _is_fa3_op(func, fa3_fwd_op):
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE

        return functools.partial(create_selective_checkpoint_contexts, policy_fn)

    # Custom modes: support both FA3 and Magi (cache mutates_args for magi)
    def context_fn():
        storage = {}

        class _AttnSkipCachingMode(TorchDispatchMode):
            def __init__(self, st):
                super().__init__()
                self._st = st

            def __torch_dispatch__(self, func, types, args, kwargs):
                kwargs = kwargs or {}
                args = args or ()
                if _is_magi_op(func, magi_fwd_op):
                    func(*args, **kwargs)
                    out_t, lse_t = _get_out_lse_from_magi_args(args, kwargs)
                    if out_t is not None and lse_t is not None:
                        mgr = get_offload_manager()
                        if mgr is not None and mgr.should_offload():
                            # offload the big `out` to CPU (freed from GPU peak);
                            # keep the tiny `lse` clone on GPU.
                            # NOTE: clone first (a correct snapshot on the compute
                            # stream) — magi's out_ buffer may be a reused
                            # workspace that the next block overwrites, which
                            # would race an async D2H reading out_ directly.
                            # The clone is transient: freed once its D2H completes
                            # (~1-2 alive at once = GPU double buffer), not the
                            # persistent 54-layer hold.
                            key = mgr.next_slot_key("attn_out")
                            handle = mgr.offload_async(key, out_t.clone())
                            self._st[func] = ("magi_offload", handle, lse_t.clone())
                        else:
                            self._st[func] = ("magi", out_t.clone(), lse_t.clone())
                    return None
                if _is_fa3_op(func, fa3_fwd_op):
                    result = func(*args, **kwargs)
                    mgr = get_offload_manager()
                    if (
                        mgr is not None
                        and mgr.should_offload()
                        and isinstance(result, (tuple, list))
                        and result
                        and isinstance(result[0], torch.Tensor)
                    ):
                        key = mgr.next_slot_key("attn_out")
                        handle = mgr.offload_async(key, result[0].clone())
                        self._st[func] = (
                            "fa3_offload",
                            handle,
                            tuple(result[1:]),
                            type(result),
                        )
                    else:
                        self._st[func] = ("fa3", result)
                    return result
                return func(*args, **kwargs)

        class _AttnSkipCachedMode(TorchDispatchMode):
            def __init__(self, st):
                super().__init__()
                self._st = st

            def __torch_dispatch__(self, func, types, args, kwargs):
                kwargs = kwargs or {}
                args = args or ()
                if func not in self._st:
                    return func(*args, **kwargs)
                kind, *rest = self._st.pop(func)
                if kind == "magi":
                    cached_out, cached_lse = rest
                    out_t, lse_t = _get_out_lse_from_magi_args(args, kwargs)
                    if out_t is not None and lse_t is not None:
                        out_t.copy_(cached_out)
                        lse_t.copy_(cached_lse)
                    return None
                if kind == "magi_offload":
                    handle, cached_lse = rest
                    cached_out = get_offload_manager().wait_and_get(handle)
                    out_t, lse_t = _get_out_lse_from_magi_args(args, kwargs)
                    if out_t is not None and lse_t is not None:
                        out_t.copy_(cached_out)
                        lse_t.copy_(cached_lse)
                    return None
                if kind == "fa3_offload":
                    handle, tail, result_type = rest
                    cached_out = get_offload_manager().wait_and_get(handle)
                    values = (cached_out, *tail)
                    return list(values) if result_type is list else values
                return rest[0]

        return (_AttnSkipCachingMode(storage), _AttnSkipCachedMode(storage))

    return context_fn


# Keep old name as alias for backwards compatibility
_build_fa3_skip_context_fn = _build_attn_skip_context_fn


class _OffloadCheckpointWrapper(CheckpointWrapper):
    """Activation-checkpoint wrapper that ALSO offloads the block's img/txt
    INPUTS to pinned CPU during forward and restores them before backward
    recompute (Phase 2 activation offload).

    img/txt are forwarded POSITIONALLY to ``torch.utils.checkpoint.checkpoint`` so
    torch's non-reentrant checkpoint saves them via ``_NoopSaveInputs`` /
    ``save_for_backward`` — which the manager's ``resid_offload_hooks`` can
    intercept. Everything else (vec/vec_txt/freqs_cis/text_mask, ...) is passed as
    kwargs; those go through checkpoint's recompute closure and stay on GPU (they
    are small and shared across all blocks, so not worth offloading).

    The attention-out SAC skip + offload (Phase 1) is preserved via ``context_fn``.

    Subclasses torch's ``CheckpointWrapper`` so it is **state-dict transparent**:
    the wrapped module is stored as ``_checkpoint_wrapped_module`` and the inherited
    state_dict post-hook / load pre-hook strip that prefix, so parameter FQNs are
    identical to an unwrapped model (checkpoint save/resume works unchanged). We
    only override ``forward``; ``CheckpointWrapper``'s own forward is unused.
    """

    def __init__(self, block, block_idx, context_fn, mgr):
        super().__init__(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        self._offload_block_idx = block_idx
        self._offload_context_fn = context_fn
        self._offload_mgr = mgr

    def forward(self, img, txt, **kwargs):
        forward_id = self._offload_mgr.current_forward_id()
        with self._offload_mgr.resid_offload_hooks(
            self._offload_block_idx, forward_id=forward_id
        ):
            return torch_checkpoint(
                self._checkpoint_wrapped_module,
                img,
                txt,
                use_reentrant=False,
                context_fn=self._offload_context_fn,
                **kwargs,
            )


def apply_fsdp_checkpointing(model, no_split_modules, p=1, skip_attn_recompute=False):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """Apply activation checkpointing to model.
    Returns None as model is updated directly.

    Args:
        p: Fraction of blocks to apply checkpointing to.
        skip_attn_recompute: If True, use selective checkpointing to skip the
            attention forward recomputation during backward.  Supported for
            both flash3_mp_causal (FA3) and magi_mp_causal (MagiAttention).

            The attention outputs (out + softmax_lse) are saved during the
            original forward pass and returned directly when the backward pass
            would otherwise re-launch the attention forward kernel.

            For magi_mp_causal: 1 attention call per block is saved.
            For flash3_mp_causal: ~5 FA3 calls per block are saved.

            Memory cost: ~+N_attn_calls × (out + lse) per checkpointed block
              (≈ N_attn_calls × block-input size with SP).
            Compute saving: eliminates all attention forward CUDA kernels from
              backward recomputation (attention ≈ 40% of per-block compute
              at long seqlen with SP=8).
    """
    p = eval(p) if isinstance(p, str) else p

    mgr = get_offload_manager()
    offload_on = mgr is not None and mgr.enabled

    if skip_attn_recompute:
        print(
            f"--> applying fsdp activation checkpointing (selective p={p}, skip_attn_recompute=True)..."
        )
        context_fn = _build_attn_skip_context_fn()
    else:
        print(f"--> applying fsdp activation checkpointing (selective p={p})...")
        context_fn = noop_context_fn

    if offload_on:
        # Phase 2: for double-stream blocks use the custom wrapper that offloads
        # img/txt inputs to CPU; other block types fall back to normal
        # checkpointing (they don't take the img/txt kwargs). The attention-out
        # SAC skip/offload (Phase 1) is preserved via context_fn either way.
        from hymm.models.dit.nomas_cpt import MMDoubleStreamBlock

        print("-->   activation offload ON: double-block img/txt inputs -> pinned CPU")
        _blk = {"i": -1}

        def wrapper_fn(submodule):
            if isinstance(submodule, MMDoubleStreamBlock):
                _blk["i"] += 1
                return _OffloadCheckpointWrapper(submodule, _blk["i"], context_fn, mgr)
            if skip_attn_recompute:
                return checkpoint_wrapper(
                    submodule,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    context_fn=context_fn,
                )
            return non_reentrant_wrapper(submodule)
    elif skip_attn_recompute:
        wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=context_fn,
        )
    else:
        wrapper_fn = non_reentrant_wrapper

    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=selective_checkpointing,
    )


def apply_camera_module_checkpointing(model, p=1.0):
    """Apply activation checkpointing to camera ActionModule layers."""
    from hymm.models.camera.action_module import ActionModule

    p = eval(p) if isinstance(p, str) else p
    if p <= 0:
        return

    print(f"--> applying camera ActionModule activation checkpointing (p={p})...")

    module_idx = 0
    cut_off = 1 / 2

    def selective_checkpointing(submodule):
        nonlocal module_idx
        nonlocal cut_off

        if isinstance(submodule, ActionModule):
            module_idx += 1
            if module_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=True,
    )
    return mixed_precision


def resolve_fsdp_param_dtype(args) -> torch.dtype:
    """Resolve FSDP forward/all-gather dtype.

    By default this preserves the historical behavior of tying FSDP param dtype
    to ``dit.master_weight_type``. Experiments that need fp32 master weights but
    bf16 forward compute can set ``parallelism.fsdp_param_dtype: bf16``.
    """
    parallelism = getattr(args, "parallelism", None)
    configured = getattr(parallelism, "fsdp_param_dtype", None)
    if configured is None or str(configured).lower() in ("", "auto", "none"):
        return (
            torch.float32 if args.dit.master_weight_type == "fp32" else torch.bfloat16
        )

    dtype_name = str(configured).lower()
    if dtype_name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half", "torch.float16"):
        return torch.float16
    if dtype_name in ("fp32", "float32", "torch.float32"):
        return torch.float32
    raise ValueError(
        "parallelism.fsdp_param_dtype must be one of auto/bf16/fp16/fp32, "
        f"got {configured!r}"
    )


def get_dit_fsdp_kwargs_v2(
    transformer,
    parallel_dims,
    world_mesh,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    # Safety check for None parallel_dims
    if parallel_dims is None:
        raise ValueError(
            "parallel_dims cannot be None. Please ensure parallel_dims and world_mesh are properly initialized."
        )

    no_split_modules = get_no_split_modules(transformer)
    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=no_split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
        assert not parallel_dims.dp_replicate_enabled
    elif sharding_strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if parallel_dims.dp_shard_enabled or parallel_dims.sp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD  # zero3 within each node
            # sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2 # zero2 within each node
        else:
            dp_mesh_dim_names = ("dp_shard_sp",)
            sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        dp_mesh_dim_names = ("dp_replicate",)
        sharding_strategy = ShardingStrategy.NO_SHARD  # DDP

    device_mesh = world_mesh[tuple(dp_mesh_dim_names)]
    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        # "device_id": torch.cuda.current_device(),
        "device_mesh": device_mesh,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "use_orig_params": True,
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update(
            {
                "use_orig_params": False,  # Required for LoRA memory savings
                "sync_module_states": True,
            }
        )

    return fsdp_kwargs, no_split_modules


def apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    num_forward_prefetch: int = 2,
    num_backward_prefetch: int = 2,
    overlap: bool = True,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    all_blocks = list(model.double_blocks) + list(model.single_blocks)
    num_blocks = len(all_blocks)

    for layer_id, transformer_block in enumerate(all_blocks):
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = layer_id < num_blocks - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)

    if overlap:
        backward_reshard_enabled = (
            reshard_after_forward_policy not in ("never",) and not pp_enabled
        )

        for i, layer in enumerate(all_blocks):
            fwd_prefetch_end = min(i + num_forward_prefetch, num_blocks - 1)
            if i < fwd_prefetch_end:
                layers_to_fwd_prefetch = [
                    all_blocks[i + j] for j in range(1, fwd_prefetch_end - i + 1)
                ]
                layer.set_modules_to_forward_prefetch(layers_to_fwd_prefetch)

            if backward_reshard_enabled and i >= num_backward_prefetch:
                layers_to_bwd_prefetch = [
                    all_blocks[i - j] for j in range(1, num_backward_prefetch + 1)
                ]
                layer.set_modules_to_backward_prefetch(layers_to_bwd_prefetch)


def _find_decoder_layers(hf_model: nn.Module):
    """Locate the ``nn.ModuleList`` of transformer decoder layers inside a
    HuggingFace text-encoder model, robust to the common nesting variants
    (``model.layers`` for Qwen2.5-VL language model, ``model.model.layers`` for
    decoder-style wrappers). Returns ``(layers, parent_module)`` or
    ``(None, None)`` if not found."""
    candidates = [
        hf_model,
        getattr(hf_model, "model", None),
        getattr(getattr(hf_model, "model", None), "model", None),
    ]
    for parent in candidates:
        if parent is None:
            continue
        layers = getattr(parent, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers, parent
    return None, None


def _fully_shard_with_hf_output_capture(module, **kwargs):
    """Preserve Transformers' class-keyed output capture after FSDP2 wrapping."""
    # Transformers 4.57/5 look up hidden-state capture by the exact class.
    # FSDP2 creates a dynamic subclass; preserve its inherited capture spec.
    try:
        from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    except ImportError:
        try:
            from transformers.utils.generic import _CAN_RECORD_REGISTRY
        except ImportError:
            registry = None
        else:
            registry = _CAN_RECORD_REGISTRY
    else:
        registry = _CAN_RECORD_REGISTRY
    capture_spec = registry.get(str(type(module))) if registry is not None else None
    fully_shard(module, **kwargs)
    if capture_spec:
        registry.setdefault(str(type(module)), capture_spec)


def apply_fsdp2_text_encoder(
    text_encoder,
    sp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    logger=None,
    cpu_offload: bool = False,
):
    """Shard a frozen, inference-only text encoder with FSDP2 over the Ulysses
    sequence-parallel mesh.

    The text encoder is replicated across every rank of an SP group (all ranks
    in the group encode the *same* prompt), so its weights are pure redundancy.
    Sharding over ``sp_mesh`` reduces the resident weight footprint by a factor
    of ``sp_mesh.size()`` (e.g. 13.2GB -> ~1.65GB at sp=8).

    Notes
    -----
    * Parameters are frozen (``requires_grad=False``) and the forward runs under
      ``torch.no_grad()``, so there is no reduce-scatter / gradient traffic — the
      added communication is parameter all-gather before each wrapped forward.
    * ``reshard_after_forward=True`` everywhere so the gathered full weights are
      released immediately after each layer's forward, keeping the transient
      peak low (only one layer is materialized at a time during the gather).
    * ``reduce_dtype`` is irrelevant (no gradients); we still pass a policy so
      the all-gather happens in ``param_dtype`` (the encoder's fp16 weights).
    * With ``cpu_offload``, FSDP2 owns pinned CPU shards, stages them to GPU
      before all-gather, and frees gathered weights after each forward.
      Frozen weights need no copy-back: the CPU shard remains authoritative.
    """
    # The actual HF module lives at ``text_encoder.model`` (see TextEncoder).
    hf_model = getattr(text_encoder, "model", text_encoder)

    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=param_dtype)
    fsdp_config = {"mesh": sp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

    if hf_model.training or any(p.requires_grad for p in hf_model.parameters()):
        raise ValueError("Inference text-encoder sharding requires frozen evaluation weights")
    # TextEncoder.encode may call this norm after the HF root forward returns.
    # Give it its own FSDP hooks so it can safely all-gather again at that point.
    if getattr(text_encoder, "apply_final_norm", False):
        final_norm = getattr(hf_model, "final_layer_norm", None)
        if not isinstance(final_norm, nn.Module):
            raise ValueError("Sharded text encoder requires a module for final_layer_norm")
        fully_shard(final_norm, **fsdp_config, reshard_after_forward=True)

    layers, _parent = _find_decoder_layers(hf_model)
    if layers is None:
        # Fall back to a single flat shard of the whole module. Still halves the
        # footprint, just with a coarser (single) communication unit.
        if logger is not None:
            logger.warning(
                "[TE-FSDP] Could not locate decoder layers on the text encoder; "
                "applying a single flat fully_shard over the whole module."
            )
        _fully_shard_with_hf_output_capture(hf_model, **fsdp_config, reshard_after_forward=True)
        return

    num_layers = len(layers)
    for layer in layers:
        fully_shard(layer, **fsdp_config, reshard_after_forward=True)
    # Shard the root to cover embeddings / final norm / lm_head-adjacent params.
    _fully_shard_with_hf_output_capture(hf_model, **fsdp_config, reshard_after_forward=True)

    if logger is not None:
        logger.info(
            f"[TE-FSDP] Sharded text encoder over SP mesh "
            f"(shard_size={sp_mesh.size()}, decoder_layers={num_layers}, "
            f"param_dtype={param_dtype}, cpu_offload={cpu_offload})."
        )
