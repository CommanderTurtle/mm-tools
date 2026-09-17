"""Shared LC64 PBR-flow helpers for external reconstruction benchmarks.

The benchmark runners already compute authoritative DINO/AnyUp features for
SS/Shape inference.  PBR training uses the same DINO features plus a learned
RGB-cell residual.  Reusing the former and evaluating only the latter avoids a
second DINO copy without changing the PBR conditioning numerically.
"""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from eval.reconstruction.lc64_shape_pbr_decode import atomic_npz, load_sparse_pair


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PBR_RUN = (
    REPO_ROOT
    / "checkpoints/Fire3D/reconstruction/flows/pbr"
)
DEFAULT_PBR_VAE = REPO_ROOT / "checkpoints/Fire3D/reconstruction/vae/pbr"


def resolve_checkpoint(run_dir: Path, checkpoint: str | Path) -> Path:
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        direct = run_dir / candidate
        candidate = direct if direct.exists() else REPO_ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate.resolve()


def _checkpoint_state(payload: dict[str, Any], use_ema: bool) -> tuple[str, dict[str, Any]]:
    key = "ema_model" if use_ema and "ema_model" in payload else "model"
    state = payload[key]
    if isinstance(state, dict) and "shadow" in state:
        state = state["shadow"]
    return key, state


def dino_signature(config: dict[str, Any]) -> dict[str, Any]:
    dino = config.get("dino") or {}
    return {
        key: dino.get(key)
        for key in ("repo_dir", "model_name", "model_path", "downsample", "upsample")
    }


def align_shared_dino_config(
    model_config: dict[str, Any], shared_dino_config: dict[str, Any]
) -> dict[str, Any]:
    """Use the already-loaded SS DINO contract for shared PBR features."""

    dino = model_config.setdefault("dino", {})
    shared = shared_dino_config.get("dino") or {}
    for key in (
        "repo_dir",
        "model_name",
        "model_path",
        "downsample",
        "upsample",
        "anyup_repo",
        "anyup_source",
    ):
        if key in shared:
            dino[key] = shared[key]
    return model_config


def load_pbr_flow_model(
    *,
    run_dir: Path,
    checkpoint: str | Path,
    device: torch.device,
    max_cond_len: int,
    dino_upsample: int,
    anyup_frame_batch_size: int,
    use_ema: bool = False,
    model_family: str = "pbr_x2_offline",
    shared_dino_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load PBR flow and release its redundant DINO encoder when it is shared."""
    if model_family != "pbr_x2_offline":
        raise ValueError(f"Unsupported PBR model family: {model_family}")
    from models.object_gen_pbr_x2_offline_with_dino import ObjectGen

    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    model_config = copy.deepcopy(config["model"])
    model_config["max_cond_len"] = int(max_cond_len)
    dino = model_config.get("dino") or {}
    dino["upsample"] = int(dino_upsample)
    dino["anyup_frame_batch_size"] = int(anyup_frame_batch_size)
    model_config["dino"] = dino
    stats_root = run_dir.parents[1] / "stats"
    model_config["shape_x2"]["x2_latent_stats_path"] = str(
        stats_root / "shape.json"
    )
    model_config["pbr_x2"]["x2_latent_stats_path"] = str(
        stats_root / "pbr.json"
    )
    if shared_dino_config is not None:
        model_config = align_shared_dino_config(model_config, shared_dino_config)
        expected = dino_signature(model_config)
        actual = dino_signature(shared_dino_config)
        if expected != actual:
            raise ValueError(
                "SS/Shape and PBR DINO contracts differ; shared features are unsafe: "
                f"pbr={expected}, shared={actual}"
            )

    model = ObjectGen(model_config)
    checkpoint_path = resolve_checkpoint(run_dir, checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_key, state = _checkpoint_state(payload, use_ema)
    model.load_state_dict(state, strict=True)
    del payload, state

    # The caller supplies the already computed DINO cells. Keep only the small
    # RGB residual encoder from PBR flow and prevent accidental recomputation.
    model.dino_model = None
    model.__dict__["_anyup_upsampler"] = None
    gc.collect()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "state_key": state_key,
        "model_family": model_family,
        "use_ema": bool(use_ema),
        "appearance_mode": model.appearance_mode,
        "max_cond_len": int(max_cond_len),
        "dino_upsample": int(dino_upsample),
        "shared_dino_features": True,
    }
    return model, metadata


@torch.no_grad()
def appearance_features_from_shared_dino(
    model: Any,
    rgbs: torch.Tensor,
    dino_features: torch.Tensor,
    grid_keep_mask: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Reproduce ``ObjectGen._appearance_feats`` from shared DINO cells."""
    mode = str(model.appearance_mode)
    dino_features = dino_features.float()
    if mode == "dino":
        return dino_features
    rgb_cells = model.get_rgb_cell_feats(rgbs).to(dino_features.device)
    if grid_keep_mask is not None:
        keep = torch.as_tensor(
            grid_keep_mask, device=rgb_cells.device, dtype=torch.bool
        ).reshape(-1)
        if rgb_cells.shape[0] != keep.numel():
            raise ValueError(
                f"RGB cell count {rgb_cells.shape[0]} != depth validity grid "
                f"{keep.numel()}"
            )
        rgb_cells = rgb_cells[keep]
    if rgb_cells.shape[0] != dino_features.shape[0]:
        raise ValueError(
            f"RGB/DINO cell mismatch: rgb={rgb_cells.shape[0]} dino={dino_features.shape[0]}"
        )
    rgb_features = model.rgb_mlp(rgb_cells * 2.0 - 1.0).float()
    if mode == "rgb":
        return rgb_features
    if mode == "dino_rgb":
        if rgb_features.shape != dino_features.shape:
            raise ValueError(
                f"RGB residual {tuple(rgb_features.shape)} != DINO {tuple(dino_features.shape)}"
            )
        return dino_features + rgb_features
    if mode == "dino_rgb_concat":
        return torch.cat([dino_features, rgb_features], dim=-1)
    raise ValueError(f"Unsupported PBR appearance mode: {mode}")


@torch.no_grad()
def predict_pbr_sparse(
    *,
    model: Any,
    common: dict[str, Any],
    shape_features: torch.Tensor,
    coords: torch.Tensor,
    selected_indices: torch.Tensor,
    sample_method: str,
    inference_num_steps: int,
    device: torch.device,
    guidance_strength: float | None = None,
    prune_out_of_range: bool = True,
) -> torch.Tensor:
    if shape_features.ndim != 2 or shape_features.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Shape/coordinate mismatch: shape={tuple(shape_features.shape)} coords={tuple(coords.shape)}"
        )
    original_mode = model.appearance_mode
    model.appearance_mode = "dino"  # ``common['colors']`` is already dino+rgb.
    model_inputs = dict(common)
    model_inputs["selected_indices"] = selected_indices.to(device)
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            result = model(
                **model_inputs,
                object_shape_x2_feats=shape_features.to(device),
                object_coords=coords.to(device),
                return_denoised_latents=True,
                sample_method=sample_method,
                inference_num_steps=int(inference_num_steps),
                guidance_strength=guidance_strength,
                prune_out_of_range=prune_out_of_range,
                inference_only=True,
            )
    finally:
        model.appearance_mode = original_mode
    predicted = result["pbr_x2_pred_feats_raw"].float()
    if predicted.shape != shape_features.shape:
        raise ValueError(
            f"PBR prediction {tuple(predicted.shape)} != Shape-X2 {tuple(shape_features.shape)}"
        )
    return predicted


def concatenate_sparse_object_pairs(
    records: Sequence[dict[str, Any]],
    *,
    shape_output: Path,
    pbr_output: Path,
) -> dict[str, Any]:
    """Combine one-object sparse pairs into a decoder batch with stable IDs."""
    shape_parts: list[torch.Tensor] = []
    pbr_parts: list[torch.Tensor] = []
    coord_parts: list[torch.Tensor] = []
    objects = []
    for position, record in enumerate(records):
        shape_path = Path(record["shape_x2_path"])
        pbr_path = Path(record["pbr_x2_path"])
        shape, pbr, coords = load_sparse_pair(shape_path, pbr_path)
        present = torch.unique(coords[:, 0]).tolist()
        if present != [0]:
            raise ValueError(f"Expected local sparse batch 0 in {shape_path}, got {present}")
        coords = coords.clone()
        coords[:, 0] = int(position)
        shape_parts.append(shape)
        pbr_parts.append(pbr)
        coord_parts.append(coords)
        objects.append(
            {
                "object_position": position,
                "object_id": record.get("object_id"),
                "sample_id": record.get("sample_id"),
                "tokens": int(shape.shape[0]),
            }
        )
    if not shape_parts:
        raise ValueError("Cannot concatenate an empty appearance record list")
    shape = torch.cat(shape_parts)
    pbr = torch.cat(pbr_parts)
    coords = torch.cat(coord_parts)
    atomic_npz(shape_output, feats=shape.numpy(), coords=coords.numpy())
    atomic_npz(pbr_output, feats=pbr.numpy(), coords=coords.numpy())
    return {
        "shape_x2": str(shape_output),
        "pbr_x2": str(pbr_output),
        "object_count": len(objects),
        "tokens": int(shape.shape[0]),
        "coordinates_equal": True,
        "objects": objects,
    }
