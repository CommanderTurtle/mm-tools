"""Synchronous safetensors checkpoints for public SFT.

Every rank participates in gathering FSDP2 weights. Rank 0 writes one model
safetensors file, then optional optimizer/EMA state. A completion marker is
published only after every rank has saved its RNG state. Training returns from
save_checkpoint only after all files are durable from the process's view.
"""

from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

MODEL_FILE = "diffusion_pytorch_model.safetensors"
SCALAR_FILE = "scalar_states.pkl"
OPTIMIZER_FILE = "optimizer.pt"
COMPLETE_FILE = "checkpoint.json"


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _world_size() -> int:
    return dist.get_world_size() if _distributed() else 1


def _complete_metadata(directory: Path) -> dict | None:
    """Read the completion marker only when all declared files are present."""
    marker = directory / COMPLETE_FILE
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
            return None
        world_size = int(metadata["world_size"])
        if world_size < 1:
            return None
        required = [MODEL_FILE, SCALAR_FILE, "config.json"]
        if metadata.get("has_optimizer"):
            required.append(OPTIMIZER_FILE)
        required.extend(f"rng_state_rank{rank}.pt" for rank in range(world_size))
        from hymm.models.dit.ema import _rate_to_safetensors_name

        required.extend(
            _rate_to_safetensors_name(rate) for rate in metadata.get("ema_rates", [])
        )
        return (
            metadata if all((directory / name).is_file() for name in required) else None
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _rank0_write(action) -> None:
    """Run a write on rank 0 and report its failure to every rank."""
    error = None
    if _rank() == 0:
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if _distributed():
        messages = [error]
        dist.broadcast_object_list(messages, src=0)
        error = messages[0]
    if error:
        raise RuntimeError(f"Checkpoint write failed: {error}")


def _all_rank_write(action) -> None:
    """Run a per-rank write and fail together when any rank cannot write."""
    error = None
    try:
        action()
    except Exception as exc:
        error = f"rank {_rank()}: {type(exc).__name__}: {exc}"
    if _distributed():
        errors = [None] * _world_size()
        dist.all_gather_object(errors, error)
        error = next((item for item in errors if item is not None), None)
    if error:
        raise RuntimeError(f"Checkpoint write failed: {error}")


def _model_state_on_rank0(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Gather each FSDP2 tensor on all ranks, retaining CPU weights on rank 0."""
    state = model.state_dict()
    keys = sorted(state)
    if _distributed():
        key_lists = [None] * _world_size()
        dist.all_gather_object(key_lists, keys)
        if any(other != keys for other in key_lists):
            raise RuntimeError("Model state keys differ across ranks")
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in keys:
            tensor = state[key]
            full = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
            if _rank() == 0:
                result[key] = full.detach().cpu().contiguous()
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _save_rng_state(directory: Path) -> None:
    state = {
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    torch.save(state, directory / f"rng_state_rank{_rank()}.pt")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    scalar_states: Any,
    *,
    checkpoint_step: int | None = None,
    save_optimizer_state: bool = False,
    ema_states: dict | None = None,
    ema_rates: list[float] | None = None,
    logger=None,
) -> Path:
    """Save a complete checkpoint before allowing the next training step."""
    step = int(
        scalar_states.train_steps if checkpoint_step is None else checkpoint_step
    )
    directory = Path(output_dir) / "checkpoints" / f"global_step-{step}"
    marker = directory / COMPLETE_FILE
    rates = list(ema_rates or [])

    if _rank() == 0:
        previous = _complete_metadata(directory)
        exists = bool(
            previous
            and previous.get("step") == step
            and previous.get("world_size") == _world_size()
            and (not save_optimizer_state or previous.get("has_optimizer"))
            and all(rate in previous.get("ema_rates", []) for rate in rates)
        )
    else:
        exists = None
    if _distributed():
        flags = [exists]
        dist.broadcast_object_list(flags, src=0)
        exists = flags[0]
    if exists:
        if logger is not None and _rank() == 0:
            logger.info(f"[checkpoint] step {step} already saved at {directory}")
        return directory

    _rank0_write(lambda: directory.mkdir(parents=True, exist_ok=True))

    model_state = _model_state_on_rank0(model)
    _rank0_write(lambda: save_file(model_state, str(directory / MODEL_FILE)))
    model_state = None

    if save_optimizer_state:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        optimizer_state = get_optimizer_state_dict(model, optimizer, options=options)
        _rank0_write(lambda: torch.save(optimizer_state, directory / OPTIMIZER_FILE))
        optimizer_state = None

    scalar = scalar_states.to_dict()
    scalar_states_list = [None] * _world_size()
    if _distributed():
        dist.all_gather_object(scalar_states_list, scalar)
    else:
        scalar_states_list[0] = scalar

    def write_metadata() -> None:
        with (directory / SCALAR_FILE).open("wb") as stream:
            pickle.dump(scalar_states_list, stream)
        config = dict(getattr(model, "config", {}))
        config.pop("dtype", None)
        (directory / "config.json").write_text(
            json.dumps(_jsonable(config), indent=2) + "\n", encoding="utf-8"
        )

    _rank0_write(write_metadata)
    _all_rank_write(lambda: _save_rng_state(directory))

    if rates:
        from hymm.models.dit.ema import save_ema_checkpoint

        save_ema_checkpoint(ema_states or {}, model, rates, str(directory), _rank())

    def write_complete() -> None:
        temporary = directory / f"{COMPLETE_FILE}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "step": step,
                    "world_size": _world_size(),
                    "has_optimizer": bool(save_optimizer_state),
                    "ema_rates": rates,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(marker)

    _rank0_write(write_complete)
    if logger is not None and _rank() == 0:
        logger.info(f"[checkpoint] saved synchronously: {directory}")
    return directory


def _load_model_weights(model: torch.nn.Module, weight_path: Path) -> None:
    """Load full safetensors into regular or FSDP2 DTensor parameters."""
    weights = load_file(str(weight_path))
    current = model.state_dict()
    if set(weights) != set(current):
        missing = sorted(set(current) - set(weights))
        extra = sorted(set(weights) - set(current))
        raise RuntimeError(
            f"Checkpoint model keys differ: missing={missing[:10]}, extra={extra[:10]}"
        )

    with torch.no_grad():
        for key, target in current.items():
            source = weights[key]
            if hasattr(target, "_local_tensor"):
                from torch.distributed.tensor import DTensor, Replicate

                local = target._local_tensor
                replicated = DTensor.from_local(
                    source.to(device=local.device, dtype=local.dtype),
                    device_mesh=target.device_mesh,
                    placements=tuple(Replicate() for _ in target.placements),
                    run_check=False,
                )
                shard = replicated.redistribute(
                    target.device_mesh, target.placements
                )._local_tensor
                local.copy_(shard)
            else:
                target.copy_(source.to(device=target.device, dtype=target.dtype))
    if _distributed():
        dist.barrier()


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    *,
    load_optimizer_state: bool = False,
) -> tuple[torch.nn.Module, list[dict], bool]:
    """Restore model and training cursor; optionally restore optimizer state."""
    directory = Path(checkpoint_dir)
    weight_path = directory / MODEL_FILE
    scalar_path = directory / SCALAR_FILE
    metadata = _complete_metadata(directory)
    if metadata is None:
        raise FileNotFoundError(
            f"Expected completed checkpoint with {MODEL_FILE} and {SCALAR_FILE} in {directory}"
        )
    if metadata["world_size"] != _world_size():
        raise ValueError(
            f"Checkpoint world size {metadata['world_size']} differs from current {_world_size()}"
        )
    _load_model_weights(model, weight_path)
    with scalar_path.open("rb") as stream:
        scalar_states_list = pickle.load(stream)

    optimizer_loaded = False
    optimizer_path = directory / OPTIMIZER_FILE
    if load_optimizer_state:
        if not optimizer_path.is_file():
            raise FileNotFoundError(
                f"Optimizer state not found at {optimizer_path}; "
                "set training.optimizer.save_optimizer_state=false for a weights-only resume"
            )
        optimizer_state = torch.load(
            optimizer_path, map_location="cpu", weights_only=False
        )
        set_optimizer_state_dict(
            model,
            optimizer,
            optimizer_state,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        optimizer_loaded = True
    if _distributed():
        dist.barrier()
    return model, scalar_states_list, optimizer_loaded


def load_rng_states(ckpt_dir: str, rank: int, logger=None) -> bool:
    """Restore this rank's RNG state after dataloader and trackers initialize."""
    path = Path(ckpt_dir) / f"rng_state_rank{rank}.pt"
    if not path.is_file():
        if logger is not None:
            logger.warning(f"RNG state file not found: {path}")
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if logger is not None:
        logger.info(f"Restored RNG state from {path}")
    return True


def auto_find_latest_checkpoint(
    output_dir: str,
    logger=None,
    required_ema_rates: list[float] | None = None,
    require_optimizer: bool = False,
) -> str | None:
    """Choose the newest fully written public checkpoint."""
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return None
    required_rates = list(required_ema_rates or [])
    candidates = []
    for directory in root.iterdir():
        match = re.fullmatch(r"global_step-(\d+)", directory.name)
        if not match or not directory.is_dir():
            continue
        metadata = _complete_metadata(directory)
        if (
            metadata
            and metadata.get("step") == int(match.group(1))
            and metadata.get("world_size") == _world_size()
            and (not require_optimizer or metadata.get("has_optimizer"))
            and all(rate in metadata.get("ema_rates", []) for rate in required_rates)
        ):
            candidates.append((int(match.group(1)), directory))
    if not candidates:
        if logger is not None:
            logger.info(f"[auto-resume] No complete checkpoint in {root}")
        return None
    _, latest = max(candidates)
    if logger is not None:
        logger.info(f"[auto-resume] Selected {latest}")
    return str(latest)
