"""Production HSDP-aware LoRA training checkpoints.

Only the first HSDP replica writes the ``dp_shard_sp`` local shards.  Adapter
parameters and optimizer state are replicated across the ``dp_replicate`` mesh, so
writing the other replicas would duplicate identical bytes.  Rank-specific RNG
and scalar/data-loader state are gathered into one metadata file so exact
same-world-size resume remains deterministic.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from hymm.models.lora import LORA_TARGETS, _adapter_parameters, _canonical_adapter_name
from .subprocess_writer import (
    SubprocessCheckpointWriter,
    extract_local_state,
    restore_local_state,
)


FORMAT = "lora_hsdp_async_v1"
COMPLETE_FILE = "checkpoint.json"
METADATA_FILE = "training_metadata.pt"
ADAPTER_CONFIG_FILE = "adapter_config.json"
SHARD_TEMPLATE = "lora_shard_{shard_rank:04d}.pt"

_PENDING: dict | None = None


def _optimizer_kind(optimizer):
    if type(optimizer).__name__ == "Dist_Muon":
        return "muon"
    if isinstance(optimizer, torch.optim.AdamW):
        return "adamw"
    raise ValueError(f"Unsupported LoRA optimizer: {type(optimizer).__name__}")


def _optimizer_kind_from_state(state):
    """Identify old checkpoints that predate the optimizer_type manifest field."""
    groups = state.get("param_groups", [])
    if groups and "wd" in groups[0] and "adamw_wd" in groups[0]:
        return "muon"
    if groups and "betas" in groups[0] and "weight_decay" in groups[0]:
        return "adamw"
    return None


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _distributed() else 0


def _world_size():
    return dist.get_world_size() if _distributed() else 1


def _rng_state():
    return {
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state, logger=None):
    if not state:
        return False
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if logger is not None:
        logger.info("Restored rank-local RNG state from LoRA training checkpoint")
    return True


def _canonicalize_optimizer_state(optimizer_state):
    result = dict(optimizer_state)
    result["state"] = {
        _canonical_adapter_name(name): value
        for name, value in optimizer_state.get("state", {}).items()
    }
    groups = []
    for group in optimizer_state.get("param_groups", []):
        copied = dict(group)
        copied["params"] = [
            _canonical_adapter_name(name) for name in group.get("params", [])
        ]
        groups.append(copied)
    result["param_groups"] = groups
    return result


def _local_adapter_state(model_state):
    result = {}
    metadata = {}
    for name, tensor in model_state.items():
        canonical = _canonical_adapter_name(name)
        if not canonical.endswith(("lora_A.weight", "lora_B.weight")):
            continue
        local = tensor._local_tensor if hasattr(tensor, "_local_tensor") else tensor
        shard_dim = None
        if hasattr(tensor, "placements"):
            for placement in tensor.placements:
                if hasattr(placement, "dim"):
                    shard_dim = int(placement.dim)
                    break
        result[canonical] = local.detach()
        metadata[canonical] = {
            "global_shape": list(tensor.shape),
            "shard_dim": shard_dim,
        }
    if not result:
        raise ValueError("No LoRA tensors found in the FSDP model state")
    return result, metadata


def _staging_paths(directory: Path, filename: str):
    remote = directory / filename
    local_root = Path(
        os.getenv("HY_CHECKPOINT_LOCAL_DIR", "/tmp/hy_lora_checkpoint_staging")
    )
    local = local_root / str(directory).lstrip("/") / filename
    return str(local), str(remote)


def _clear_incomplete_markers(directory: Path, shard_count: int):
    """Remove stale completion markers before retrying an interrupted step."""
    filenames = [METADATA_FILE]
    filenames.extend(
        SHARD_TEMPLATE.format(shard_rank=index) for index in range(shard_count)
    )
    for name in (COMPLETE_FILE, COMPLETE_FILE + ".tmp"):
        (directory / name).unlink(missing_ok=True)
    for filename in filenames:
        (directory / (filename + ".done")).unlink(missing_ok=True)
        (directory / (filename + ".done.tmp")).unlink(missing_ok=True)


def _manifest(directory: Path):
    try:
        value = json.loads((directory / COMPLETE_FILE).read_text(encoding="utf-8"))
        if value.get("format") != FORMAT or value.get("complete") is not True:
            return None
        shard_count = int(value["shard_count"])
        required = [ADAPTER_CONFIG_FILE, METADATA_FILE]
        required.extend(
            SHARD_TEMPLATE.format(shard_rank=index) for index in range(shard_count)
        )
        return (
            value
            if all(
                (directory / name).is_file() and (directory / name).stat().st_size > 0
                for name in required
            )
            else None
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def is_complete_lora_checkpoint(directory):
    return _manifest(Path(directory)) is not None


def _publish_pending():
    global _PENDING
    if _PENDING is None:
        return
    if _rank() == 0:
        if not _pending_files_complete():
            raise RuntimeError(
                f"Cannot publish incomplete LoRA checkpoint: {_PENDING['directory']}"
            )
        directory = Path(_PENDING["directory"])
        temporary = directory / (COMPLETE_FILE + ".tmp")
        temporary.write_text(
            json.dumps(_PENDING["manifest"], indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, directory / COMPLETE_FILE)
    _PENDING = None


def _pending_files_complete():
    if _PENDING is None:
        return True
    directory = Path(_PENDING["directory"])
    filenames = [METADATA_FILE]
    filenames.extend(
        SHARD_TEMPLATE.format(shard_rank=index)
        for index in range(int(_PENDING["manifest"]["shard_count"]))
    )
    return all(
        (directory / filename).is_file()
        and (directory / (filename + ".done")).is_file()
        for filename in filenames
    )


def poll_pending_lora_checkpoint():
    """Publish a pending checkpoint without adding a training collective."""
    if _PENDING is None:
        return True
    writer = SubprocessCheckpointWriter.maybe_instance()
    if writer is not None:
        writer.poll()
    if _rank() == 0 and _pending_files_complete():
        directory = Path(_PENDING["directory"])
        temporary = directory / (COMPLETE_FILE + ".tmp")
        temporary.write_text(
            json.dumps(_PENDING["manifest"], indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, directory / COMPLETE_FILE)
    # Do not clear _PENDING here. Polling is intentionally non-collective, so
    # ranks can observe the marker at different iterations. The next
    # collective wait clears it consistently on every rank.
    return (Path(_PENDING["directory"]) / COMPLETE_FILE).is_file()


def wait_for_pending_lora_checkpoint():
    global _PENDING
    local_pending = _PENDING is not None
    if _distributed():
        pending_flags = [None] * _world_size()
        dist.all_gather_object(pending_flags, local_pending)
        if any(bool(flag) != local_pending for flag in pending_flags):
            raise RuntimeError(
                "LoRA checkpoint pending state diverged across distributed ranks"
            )
    if not local_pending:
        return True

    writer = SubprocessCheckpointWriter.maybe_instance()
    local_error = None
    try:
        if writer is not None:
            writer.wait_all()
    except Exception as exc:  # synchronize failures before raising
        local_error = f"rank {_rank()}: {type(exc).__name__}: {exc}"

    if _distributed():
        errors = [None] * _world_size()
        dist.all_gather_object(errors, local_error)
        failures = [error for error in errors if error]
    else:
        failures = [local_error] if local_error else []
    if failures:
        raise RuntimeError("LoRA checkpoint writer failed: " + "; ".join(failures))

    if _distributed():
        dist.barrier()
    _publish_pending()
    if _distributed():
        dist.barrier()
    return True


def shutdown_lora_checkpoint_writer():
    """Close the rank-local async writer after the final checkpoint."""
    wait_for_pending_lora_checkpoint()
    SubprocessCheckpointWriter.shutdown_instance()


def save_lora_training_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    output_dir,
    scalar_states,
    lora_config,
    *,
    dp_replicate_rank=0,
    shard_group=None,
    logger=None,
):
    """Stage a complete LoRA+optimizer+cursor+RNG checkpoint asynchronously."""
    global _PENDING
    wait_for_pending_lora_checkpoint()

    rank = _rank()
    world_size = _world_size()
    shard_count = (
        dist.get_world_size(shard_group) if shard_group is not None else world_size
    )
    shard_rank = dist.get_rank(shard_group) if shard_group is not None else rank
    step = int(scalar_states.update_steps)
    directory = Path(output_dir) / "checkpoints" / f"global_step-{step}"

    if rank == 0 and _manifest(directory) is not None:
        exists = True
    else:
        exists = False
    if _distributed():
        flags = [exists]
        dist.broadcast_object_list(flags, src=0)
        exists = bool(flags[0])
    if exists:
        if logger is not None and rank == 0:
            logger.info(
                f"[checkpoint] complete LoRA checkpoint already exists: {directory}"
            )
        return directory

    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        _clear_incomplete_markers(directory, shard_count)
        (directory / ADAPTER_CONFIG_FILE).write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "rank": int(lora_config["rank"]),
                    "alpha": float(lora_config["alpha"]),
                    "dropout": float(lora_config["dropout"]),
                    "targets": list(LORA_TARGETS),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if _distributed():
        dist.barrier()

    # Preserve every rank's data cursor and RNG even though model/optimizer
    # payloads are deduplicated across the HSDP replicate dimension.
    local_metadata = {
        "scalar_state": scalar_states.to_dict(),
        "rng_state": _rng_state(),
    }
    gathered = [None] * world_size if rank == 0 else None
    if _distributed():
        dist.gather_object(local_metadata, gathered, dst=0)
    else:
        gathered = [local_metadata]

    model_state, optimizer_state = get_state_dict(model, optimizer)
    adapter_state, param_metadata = _local_adapter_state(model_state)
    optimizer_state = _canonicalize_optimizer_state(optimizer_state)
    local_optimizer = extract_local_state(optimizer_state)

    if int(dp_replicate_rank) == 0:
        payload = {
            "format_version": 1,
            "shard_rank": int(shard_rank),
            "shard_count": int(shard_count),
            "adapter": adapter_state,
            "optimizer": local_optimizer,
            "param_metadata": param_metadata,
        }
        filename = SHARD_TEMPLATE.format(shard_rank=shard_rank)
        local_path, remote_path = _staging_paths(directory, filename)
        SubprocessCheckpointWriter.get_or_create(rank).submit(
            payload, local_path, remote_path
        )

    if rank == 0:
        metadata_payload = {
            "format_version": 1,
            "world_size": world_size,
            "states": gathered,
            "lr_scheduler": lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
        }
        local_path, remote_path = _staging_paths(directory, METADATA_FILE)
        SubprocessCheckpointWriter.get_or_create(rank).submit(
            metadata_payload, local_path, remote_path
        )

    _PENDING = {
        "directory": str(directory),
        "manifest": {
            "format": FORMAT,
            "format_version": 1,
            "complete": True,
            "step": step,
            "world_size": world_size,
            "shard_count": int(shard_count),
            "hsdp_replicas": world_size // int(shard_count),
            "saves_optimizer": True,
            "optimizer_type": _optimizer_kind(optimizer),
            "saves_scheduler": lr_scheduler is not None,
            "saves_rng": True,
        },
    }
    if _distributed():
        dist.barrier()
    if logger is not None and rank == 0:
        logger.info(
            f"[checkpoint] staged async LoRA HSDP checkpoint at step {step}: "
            f"{shard_count} real shards for {world_size} ranks"
        )
    return directory


def _copy_optimizer_state(live_optimizer, loaded_optimizer):
    live_state = live_optimizer.get("state", {})
    loaded_state = loaded_optimizer.get("state", {})
    # AdamW creates state lazily: unused targets can be in param_groups without
    # any moments (e.g. the last block's text output projection/FFN).
    actual_keys = {
        _canonical_adapter_name(key): key
        for group in live_optimizer.get("param_groups", [])
        for key in group.get("params", [])
    }
    normalized_loaded = {
        _canonical_adapter_name(key): value for key, value in loaded_state.items()
    }
    unexpected = sorted(set(normalized_loaded) - set(actual_keys))
    if unexpected:
        raise ValueError(
            "Optimizer parameter keys do not match the live LoRA optimizer: "
            f"unexpected={unexpected[:5]}"
        )
    translated = {actual_keys[key]: value for key, value in normalized_loaded.items()}
    for key, source in translated.items():
        target = live_state.get(key)
        if not isinstance(target, dict) or not isinstance(source, dict):
            raise ValueError(f"Invalid optimizer state structure for {key}")
        if set(target) != set(source):
            raise ValueError(
                f"Optimizer state fields do not match for {key}: "
                f"live={sorted(target)} checkpoint={sorted(source)}"
            )
    restored = restore_local_state(live_state, translated)
    if loaded_optimizer.get("param_groups"):
        if len(loaded_optimizer["param_groups"]) != len(
            live_optimizer.get("param_groups", [])
        ):
            raise ValueError("Optimizer param-group count does not match checkpoint")
        loaded_groups = []
        for group, live_group in zip(
            loaded_optimizer["param_groups"], live_optimizer["param_groups"]
        ):
            copied = dict(group)
            copied["params"] = [
                actual_keys.get(_canonical_adapter_name(name), name)
                for name in group.get("params", [])
            ]
            if set(copied["params"]) != set(live_group.get("params", [])):
                raise ValueError(
                    "Optimizer param group keys do not match the live optimizer"
                )
            loaded_groups.append(copied)
        live_optimizer["param_groups"] = loaded_groups
    # A fresh optimizer may have dummy states initialized by get_state_dict().
    # Preserve the checkpoint's lazy-state semantics for never-used targets.
    for key in set(live_state) - set(translated):
        del live_state[key]
    return restored


def load_lora_training_checkpoint(
    model,
    optimizer,
    checkpoint_dir,
    *,
    shard_group=None,
    logger=None,
):
    """Restore LoRA weights, optimizer state, scheduler, cursor and RNG handle."""
    directory = Path(checkpoint_dir)
    manifest = _manifest(directory)
    if manifest is None:
        raise FileNotFoundError(f"Incomplete LoRA training checkpoint: {directory}")
    expected_optimizer = _optimizer_kind(optimizer)
    recorded_optimizer = manifest.get("optimizer_type")
    if recorded_optimizer is not None and recorded_optimizer != expected_optimizer:
        raise ValueError(
            f"LoRA checkpoint uses {recorded_optimizer}, but the current optimizer "
            f"is {expected_optimizer}; start a new run or use a matching checkpoint"
        )
    world_size = _world_size()
    if int(manifest["world_size"]) != world_size:
        raise ValueError(
            f"LoRA checkpoint world_size={manifest['world_size']} differs from {world_size}"
        )
    shard_count = (
        dist.get_world_size(shard_group) if shard_group is not None else world_size
    )
    shard_rank = dist.get_rank(shard_group) if shard_group is not None else _rank()
    if int(manifest["shard_count"]) != shard_count:
        raise ValueError(
            f"LoRA checkpoint shard_count={manifest['shard_count']} differs from {shard_count}"
        )

    payload = torch.load(
        directory / SHARD_TEMPLATE.format(shard_rank=shard_rank),
        map_location="cpu",
        weights_only=False,
    )
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("Unsupported LoRA checkpoint shard format")
    if int(payload.get("shard_rank", -1)) != shard_rank:
        raise ValueError("LoRA checkpoint shard rank does not match the live rank")
    if int(payload.get("shard_count", -1)) != shard_count:
        raise ValueError("LoRA checkpoint shard count is inconsistent")
    recorded_optimizer = recorded_optimizer or _optimizer_kind_from_state(
        payload["optimizer"]
    )
    if recorded_optimizer != expected_optimizer:
        raise ValueError(
            f"LoRA checkpoint uses {recorded_optimizer or 'an unknown optimizer'}, "
            f"but the current optimizer is {expected_optimizer}"
        )
    live_model, live_optimizer = get_state_dict(model, optimizer)
    actual_model = {
        _canonical_adapter_name(name): tensor
        for name, tensor in live_model.items()
        if _canonical_adapter_name(name).endswith(("lora_A.weight", "lora_B.weight"))
    }
    if set(actual_model) != set(payload["adapter"]):
        raise ValueError("LoRA shard keys do not match the live model")
    restored_model = restore_local_state(actual_model, payload["adapter"])
    restored_optimizer = _copy_optimizer_state(live_optimizer, payload["optimizer"])
    set_state_dict(
        model,
        optimizer,
        model_state_dict={},
        optim_state_dict=live_optimizer,
        options=StateDictOptions(strict=False),
    )

    metadata = None
    if _rank() == 0:
        metadata = torch.load(
            directory / METADATA_FILE, map_location="cpu", weights_only=False
        )
    if _distributed():
        holders = [metadata]
        dist.broadcast_object_list(holders, src=0)
        metadata = holders[0]
    if not isinstance(metadata, dict) or int(metadata.get("format_version", -1)) != 1:
        raise ValueError("Unsupported LoRA checkpoint metadata format")
    if int(metadata.get("world_size", -1)) != world_size:
        raise ValueError("LoRA checkpoint metadata world size is inconsistent")
    states = metadata.get("states")
    if not isinstance(states, list) or len(states) != world_size:
        raise ValueError("LoRA checkpoint does not contain one state per rank")
    state_for_rank = states[_rank()]
    if logger is not None and _rank() == 0:
        logger.info(
            f"Restored LoRA HSDP checkpoint {directory}: "
            f"adapter_tensors={restored_model}, optimizer_entries={restored_optimizer}"
        )
    return (
        state_for_rank["scalar_state"],
        state_for_rank["rng_state"],
        metadata.get("lr_scheduler"),
    )


def auto_find_latest_lora_checkpoint(output_dir, logger=None):
    root = Path(output_dir) / "checkpoints"
    candidates = []
    if root.is_dir():
        for directory in root.iterdir():
            match = re.fullmatch(r"global_step-(\d+)", directory.name)
            if match and directory.is_dir() and _manifest(directory) is not None:
                candidates.append((int(match.group(1)), directory))
    if not candidates:
        if logger is not None:
            logger.info(f"[auto-resume] No complete LoRA HSDP checkpoint in {root}")
        return None
    step, directory = max(candidates)
    if logger is not None:
        logger.info(f"[auto-resume] Selected LoRA checkpoint step {step}: {directory}")
    return str(directory)
