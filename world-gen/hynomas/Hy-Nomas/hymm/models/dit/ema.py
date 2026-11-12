"""
EMA (Exponential Moving Average) utilities for FSDP2-sharded models.

Maintains shadow copies of model parameters with optional multiple decay rates.
EMA state is stored as per-rank shards matching the model's FSDP2 sharding;
full state dict is gathered only when saving to checkpoint.

All shadows are stored and updated in float32 — see tests/test_ema_distributed.py for why.
The caller (train.py) must skip update_ema when grad_norm is NaN/Inf/0.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from loguru import logger as _loguru_logger

# Param names to skip for EMA (e.g. pos_embed in some architectures)
DEFAULT_SKIP_NAMES: Set[str] = frozenset({"pos_embed"})

# Sentinel rate value used as the dict key for schedule-mode EMA (single shadow,
# step-dependent rate).
EMA_SCHEDULED_RATE: float = -1.0


def _rate_to_safetensors_name(rate: float) -> str:
    """Convert a rate value to the EMA safetensors filename."""
    if rate == EMA_SCHEDULED_RATE:
        return "diffusion_pytorch_model_ema_scheduled.safetensors"
    rate_str = str(rate).replace(".", "_")
    return f"diffusion_pytorch_model_ema_{rate_str}.safetensors"


def _safe_print(message: str, level: str = "info") -> None:
    getattr(_loguru_logger, level)(message)


def _get_local_tensor(param: torch.Tensor) -> torch.Tensor:
    """Get the local shard of a parameter (for DTensor) or the param itself."""
    if hasattr(param, "_local_tensor"):
        return param._local_tensor
    return param.data


def _canonical_state_dict_key(name: str) -> str:
    """Map activation-checkpoint wrapper names to model state dict names."""
    return name.replace("._checkpoint_wrapped_module.", ".").replace(
        "_checkpoint_wrapped_module.", ""
    )


def build_ema_states(
    model: torch.nn.Module,
    ema_rates: List[float],
    skip_names: Optional[Set[str]] = None,
) -> Dict[float, Dict[str, torch.Tensor]]:
    """
    Build EMA shadow state for each decay rate. Only includes trainable parameters;
    shadow tensors have the same shape as each rank's local parameter shard.

    Args:
        model: The FSDP2-wrapped model (after apply_fsdp2).
        ema_rates: List of decay rates (e.g. [0.9999, 0.999, 0.99]).
        skip_names: Optional set of parameter names to skip (e.g. {"pos_embed"}).

    Returns:
        ema_states[decay_rate][param_name] = shadow tensor (local shard, same device as param).
    """
    skip = skip_names or set()
    ema_states: Dict[float, Dict[str, torch.Tensor]] = {rate: {} for rate in ema_rates}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in skip or name.split(".")[-1] in DEFAULT_SKIP_NAMES:
                continue
            local = _get_local_tensor(param)
            for rate in ema_rates:
                ema_states[rate][name] = local.detach().clone().float()

    return ema_states


@torch.no_grad()
def update_ema(
    ema_states: Dict[float, Dict[str, torch.Tensor]],
    model: torch.nn.Module,
    decay_rates: List[float],
    skip_names: Optional[Set[str]] = None,
) -> None:
    """
    Update EMA shadows: ema = decay * ema + (1 - decay) * param (post-optimizer step).

    Call this after optimizer.step() and optimizer.zero_grad() on update steps only.
    """
    param_dict = dict(model.named_parameters())
    for rate in decay_rates:
        if rate not in ema_states:
            continue
        list_ema: List[torch.Tensor] = []
        list_local: List[torch.Tensor] = []
        mismatched = []
        for name, ema_shadow in ema_states[rate].items():
            param = param_dict.get(name)
            if param is None or not param.requires_grad:
                continue
            local = _get_local_tensor(param).float()
            # A shadow built from a differently-sharded (or unsharded) view of
            # the param makes _foreach_add_ fail with a bare "size of tensor a
            # (X) must match tensor b (Y)" that names no parameter, which is
            # essentially undebuggable. Name the offenders instead.
            if local.shape != ema_shadow.shape:
                mismatched.append((name, tuple(ema_shadow.shape), tuple(local.shape)))
                continue
            list_ema.append(ema_shadow)
            list_local.append(local)
        if mismatched:
            preview = "; ".join(
                f"{n}: shadow{s} vs param{p}" for n, s, p in mismatched[:8]
            )
            raise RuntimeError(
                f"update_ema: {len(mismatched)} EMA shadow(s) do not match the live "
                f"parameter shard shape. The shadows were allocated from a different "
                f"sharding/wrapping of the model than the one being trained "
                f"(build_ema_states must run AFTER the final FSDP wrap of every "
                f"module whose params land in ema_states). {preview}"
            )
        if not list_ema:
            continue
        torch._foreach_mul_(list_ema, rate)
        torch._foreach_add_(list_ema, list_local, alpha=1.0 - rate)


@torch.no_grad()
def update_ema_with_rate(
    ema_states: Dict[float, Dict[str, torch.Tensor]],
    model: torch.nn.Module,
    state_key: float,
    decay_rate: float,
    skip_names: Optional[Set[str]] = None,
) -> None:
    """
    Update a single EMA shadow with an explicit decay rate value.

    This is used for step-dependent EMA schedules where the decay rate changes
    over time but only one EMA shadow is maintained.

    Args:
        ema_states: EMA state dict (keyed by the canonical rate used at init).
        model: The model whose parameters to track.
        state_key: The key in ema_states to update (the canonical rate from init).
        decay_rate: The actual decay rate to use for this step.
        skip_names: Optional set of parameter names to skip.
    """
    if state_key not in ema_states:
        return
    param_dict = dict(model.named_parameters())
    list_ema: List[torch.Tensor] = []
    list_local: List[torch.Tensor] = []
    for name, ema_shadow in ema_states[state_key].items():
        param = param_dict.get(name)
        if param is None or not param.requires_grad:
            continue
        list_ema.append(ema_shadow)
        list_local.append(_get_local_tensor(param).float())
    if not list_ema:
        return
    torch._foreach_mul_(list_ema, decay_rate)
    torch._foreach_add_(list_ema, list_local, alpha=1.0 - decay_rate)


@torch.no_grad()
def sync_ema_from_model(
    ema_states: Dict[float, Dict[str, torch.Tensor]],
    model: torch.nn.Module,
) -> None:
    """Overwrite every EMA shadow with the model's current parameters.

    This preserves the already-allocated float32 local shards, avoiding the
    temporary memory spike that rebuilding all EMA states would cause.
    """
    for state_key in ema_states:
        update_ema_with_rate(
            ema_states=ema_states,
            model=model,
            state_key=state_key,
            decay_rate=0.0,
        )


def _consensus_key_list(local_keys: List[str], tag: str) -> List[str]:
    """Return the canonical sorted key list, verified identical across every rank.

    Uses ``all_gather_object`` (not broadcast) so that *every* rank sees every
    other rank's key set and makes the same pass/raise decision. This is
    critical: if only rank 0 diverged, a broadcast-from-rank-0 scheme would
    silently accept rank 0's bad set and have it enter the subsequent
    ``full_tensor()`` collectives alone while the other ranks raised — that
    recreates the exact deadlock we are trying to avoid.

    With all_gather_object, either *all* ranks raise or *all* ranks proceed,
    turning the #1 silent-hang failure mode (rank-dependent iteration set) into
    an immediate, loud error *before* any NCCL collective is issued.
    """
    local_sorted = sorted(local_keys)
    if not (dist.is_available() and dist.is_initialized()):
        return local_sorted

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gathered: List[Optional[List[str]]] = [None] * world_size
    dist.all_gather_object(gathered, local_sorted)

    # Every rank computes the same boolean: "do all ranks have identical sets?"
    # If ANY pair diverges, ALL ranks raise. Otherwise ALL ranks proceed with
    # the shared canonical list. No "one rank passes while peers raise" split.
    sets = [set(g or []) for g in gathered]  # type: ignore[arg-type]
    all_equal = all(s == sets[0] for s in sets)
    if not all_equal:
        union: Set[str] = set()
        for s in sets:
            union |= s
        local_set = sets[rank]
        missing = sorted(union - local_set)
        extra = sorted(local_set - union)
        divergent_ranks = [i for i, s in enumerate(sets) if s != sets[0]]
        raise RuntimeError(
            f"[EMA:{tag}] rank {rank}: key set diverged across ranks "
            f"(divergent_ranks_vs_rank0={divergent_ranks}). "
            f"On this rank: missing={missing}  extra={extra}. "
            f"This would deadlock subsequent collectives, so failing fast."
        )
    return sorted(sets[0])


def _gather_base_state_full_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Gather base model state_dict to full CPU tensors (collective).

    Deterministic key order + consensus-verified across ranks so the per-key
    ``full_tensor()`` collectives are guaranteed to line up.
    """
    sd = model.state_dict()
    keys = _consensus_key_list(list(sd.keys()), tag="base_state_dict")
    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k in keys:
            v = sd[k]
            # Canonicalize: strip the activation-checkpoint wrapper infix so the
            # saved safetensors keys match a model built without checkpointing
            # (e.g. inference).
            ck = _canonical_state_dict_key(k)
            if hasattr(v, "full_tensor"):
                out[ck] = v.full_tensor().detach().cpu().clone()
            else:
                out[ck] = v.detach().cpu().clone()
    return out


def _gather_ema_shadows_full_cpu(
    ema_states_for_rate: Dict[str, torch.Tensor],
    param_dict: Dict[str, torch.Tensor],
    rate_tag: str,
) -> Dict[str, torch.Tensor]:
    """Gather all EMA shadows for a single rate to full CPU float32 tensors.

    Determinism guarantees (to prevent the per-param ``full_tensor()`` hang):

    1. Consensus key list (sorted + ``all_gather_object`` + full set-equality
       check), so every rank issues exactly the same sequence of collectives,
       or all ranks raise symmetrically before any NCCL op.
    2. No ``continue`` in the collective loop — a missing param raises
       RuntimeError. Since the keys are already known-equal across ranks, any
       missing param means the model structure itself changed between
       ``build_ema_states()`` and this save, and all ranks will hit the same
       raise at the same key.
    3. float32 is preserved because we wrap the fp32 shadow directly in
       ``DTensor.from_local`` and gather that (not the bf16 model param).
    """
    from torch.distributed.tensor import DTensor

    keys = _consensus_key_list(list(ema_states_for_rate.keys()), tag=f"rate={rate_tag}")

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in keys:
            ema_shadow = ema_states_for_rate[name]
            param = param_dict.get(name)
            if param is None:
                raise RuntimeError(
                    f"[EMA:rate={rate_tag}] param '{name}' not found in model. "
                    f"Model structure changed between build_ema_states() and save."
                )
            if hasattr(param, "_local_tensor"):
                ema_dt = DTensor.from_local(
                    ema_shadow,
                    device_mesh=param.device_mesh,
                    placements=param.placements,
                    run_check=False,
                )
                out[_canonical_state_dict_key(name)] = (
                    ema_dt.full_tensor().detach().cpu().clone()
                )
            else:
                out[_canonical_state_dict_key(name)] = ema_shadow.detach().cpu().clone()
    return out


def save_ema_checkpoint(
    ema_states: Dict[float, Dict[str, torch.Tensor]],
    model: torch.nn.Module,
    rates: List[float],
    save_dir: str,
    rank: int,
) -> None:
    """
    Save each EMA rate's full state dict to ``diffusion_pytorch_model_ema_<rate>.safetensors``.

    Both float32 precision AND deadlock-free gather:

    - base model state is gathered once via :func:`_gather_base_state_full_cpu`.
    - EMA shadows are gathered via :func:`_gather_ema_shadows_full_cpu`, which
      wraps the float32 shadow directly in a DTensor (preserving fp32 precision)
      and uses a consensus key list to guarantee collective alignment across
      all ranks.

    Key hang mitigations vs. the previous version:

    * Consensus-verified, sorted iteration keys (no rank-dependent ordering).
    * Fail-fast RuntimeError on any key/param divergence instead of silent hang.
    * Pre-gather CUDA synchronization before issuing NCCL collectives.
    """
    active_rates = [r for r in rates if r in ema_states and ema_states[r]]

    # Consensus on the rate set FIRST — before any other collective or any
    # early-return branch. If ranks disagree (e.g. one rank has empty ema_states),
    # all ranks raise here symmetrically instead of later diverging into
    # "some return, some hit barrier alone".
    rate_tags = [str(r) for r in active_rates]
    _consensus_key_list(rate_tags, tag="active_rates")

    if set(active_rates) != set(rates):
        raise ValueError("EMA checkpoint requested for a rate without shadow weights")

    if not active_rates:
        return

    param_dict = dict(model.named_parameters())

    # Drain pending CUDA work before issuing new collectives.
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()

    base_cpu = _gather_base_state_full_cpu(model)

    for rate, rate_tag in zip(active_rates, rate_tags):
        ema_full = _gather_ema_shadows_full_cpu(
            ema_states[rate], param_dict, rate_tag=rate_tag
        )
        cpu_state = dict(base_cpu)
        cpu_state.update(ema_full)

        from hymm.models.checkpointing.checkpoint import _rank0_write

        def write_ema():
            fname = _rate_to_safetensors_name(rate)
            os.makedirs(save_dir, exist_ok=True)
            save_file(cpu_state, os.path.join(save_dir, fname))

        _rank0_write(write_ema)


def load_ema_states_from_checkpoint(
    ema_states: Dict[float, Dict[str, torch.Tensor]],
    checkpoint_dir: str,
    model: torch.nn.Module,
    rates: List[float],
) -> None:
    """
    Load EMA state from checkpoint into ema_states. For each rate, load
    diffusion_pytorch_model_ema_<rate>.safetensors and distribute into local shards
    (same logic as loading into FSDP2 model).

    Raises FileNotFoundError if the EMA checkpoint file for any rate is missing.
    """
    from torch.distributed.tensor import DTensor, Replicate

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    model_state = {
        _canonical_state_dict_key(key): value
        for key, value in model.state_dict().items()
    }

    for rate in rates:
        if rate not in ema_states:
            continue
        fname = _rate_to_safetensors_name(rate)
        path = os.path.join(checkpoint_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[EMA] EMA checkpoint not found for rate {rate} at {path}. "
                f"When ema_rates is set, the corresponding EMA checkpoint files must exist "
                f"in the resume directory. Either provide the EMA checkpoint or remove "
                f"the rate from ema_rates."
            )
        data = load_file(path)
        loaded = {k: v for k, v in data.items() if k != "__metadata__"}

        # Track which keys ended up applied vs. skipped/extra so we can warn
        # at the end instead of silently dropping mismatches.
        loaded_keys = set(loaded.keys())
        ema_keys = {_canonical_state_dict_key(key): key for key in ema_states[rate]}
        ema_tracked_keys = set(ema_keys)
        applied_keys: set = set()
        skipped_not_in_model: List[str] = []
        skipped_not_in_ema: List[str] = []

        with torch.no_grad():
            for key, loaded_tensor in loaded.items():
                if key not in model_state:
                    skipped_not_in_model.append(key)
                    continue
                if key not in ema_keys:
                    skipped_not_in_ema.append(key)
                    continue
                model_param = model_state[key]
                ema_shadow = ema_states[rate][ema_keys[key]]
                if hasattr(model_param, "_local_tensor"):
                    local_tensor = model_param._local_tensor
                    device_mesh = model_param.device_mesh
                    placements = model_param.placements
                    loaded_tensor = loaded_tensor.to(
                        dtype=ema_shadow.dtype, device=local_tensor.device
                    )
                    replicate_placements = tuple(Replicate() for _ in placements)
                    replicated_dtensor = DTensor.from_local(
                        loaded_tensor,
                        device_mesh=device_mesh,
                        placements=replicate_placements,
                        run_check=False,
                    )
                    redistributed = replicated_dtensor.redistribute(
                        device_mesh, placements
                    )
                    ema_shadow.copy_(redistributed._local_tensor)
                else:
                    ema_shadow.copy_(
                        loaded_tensor.to(
                            device=ema_shadow.device, dtype=ema_shadow.dtype
                        )
                    )
                applied_keys.add(key)

        # Keys this model expects but the EMA file did not provide.
        ema_missing_in_ckpt = sorted(ema_tracked_keys - loaded_keys)

        if rank == 0:
            print(
                f"[EMA] Loaded EMA state for rate {rate} from {path} "
                f"(applied={len(applied_keys)}, "
                f"missing_in_ckpt={len(ema_missing_in_ckpt)}, "
                f"unexpected_in_ckpt={len(skipped_not_in_model) + len(skipped_not_in_ema)})"
            )
            if ema_missing_in_ckpt:
                _safe_print(
                    f"[EMA] WARNING: {len(ema_missing_in_ckpt)} EMA-tracked param(s) "
                    f"are missing from checkpoint {path}; they will keep their "
                    f"current EMA shadow values (likely freshly initialized). Listing "
                    f"first {min(20, len(ema_missing_in_ckpt))}:",
                    level="warning",
                )
                for k in ema_missing_in_ckpt[:20]:
                    _safe_print(f"[EMA]   - {k}", level="warning")
                if len(ema_missing_in_ckpt) > 20:
                    _safe_print(
                        f"[EMA]   ... and {len(ema_missing_in_ckpt) - 20} more",
                        level="warning",
                    )
            if skipped_not_in_model:
                _safe_print(
                    f"[EMA] WARNING: {len(skipped_not_in_model)} key(s) in checkpoint "
                    f"are not present in current model. Listing first "
                    f"{min(10, len(skipped_not_in_model))}:",
                    level="warning",
                )
                for k in skipped_not_in_model[:10]:
                    _safe_print(f"[EMA]   - {k}", level="warning")
                if len(skipped_not_in_model) > 10:
                    _safe_print(
                        f"[EMA]   ... and {len(skipped_not_in_model) - 10} more",
                        level="warning",
                    )
            if skipped_not_in_ema:
                _safe_print(
                    f"[EMA] INFO: {len(skipped_not_in_ema)} key(s) in checkpoint "
                    f"are not EMA-tracked (e.g. non-trainable buffers). Skipped.",
                    level="info",
                )
