"""Model loading helpers for the public Fire3D reconstruction runtime.

These helpers are intentionally independent of baseline integrations. The
validated research runner historically imported them from the ShapeR adapter;
the release keeps the behavior while giving the core pipeline clear ownership.
"""

from __future__ import annotations

import copy
import gc
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


DEFAULT_SHAPE_DECODER = (
    "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
)


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        direct = run_dir / checkpoint
        candidate = direct if direct.exists() else candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {candidate}")
    return candidate.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_state(
    checkpoint: dict[str, Any], use_ema: bool
) -> tuple[str, dict[str, Any]]:
    key = "ema_model" if use_ema and "ema_model" in checkpoint else "model"
    state = checkpoint[key]
    if isinstance(state, dict) and "shadow" in state:
        state = state["shadow"]
    return key, state


def conditioning_overrides(
    config: dict[str, Any], args: Any
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    model = config.setdefault("model", {})
    model["max_cond_len"] = int(args.max_cond_len)
    dino = model.get("dino")
    if dino is None:
        raise ValueError("Fire3D reconstruction requires model.dino in the config")
    dino["upsample"] = int(args.dino_upsample)
    dino["anyup_frame_batch_size"] = int(args.anyup_frame_batch_size)
    dino["repo_dir"] = str(args.dino_repo_dir)
    dino["model_path"] = str(args.dino_model_path)
    dino["anyup_repo"] = str(args.anyup_repo_dir)
    dino["anyup_source"] = "local"
    stats_root = Path(args.ss_run_dir).parents[1] / "stats"
    if "ss_x2" in model:
        model["ss_x2"]["latent_stats_path"] = str(
            stats_root / "ss.json"
        )
    if "shape_x2" in model:
        model["shape_x2"]["x2_latent_stats_path"] = str(
            stats_root / "shape.json"
        )
    return config


def load_flow_models(
    args: Any, device: torch.device
) -> tuple[Any, Any, dict[str, Any]]:
    if args.shape_model_family != "shape_x2_offline":
        raise ValueError(f"Unsupported release Shape model: {args.shape_model_family}")
    if args.ss_model_family != "ss_x2_offline":
        raise ValueError(f"Unsupported release SS model: {args.ss_model_family}")
    from models.object_gen_shape_x2_offline_with_dino import ObjectGen as ShapeObjectGen
    from models.object_gen_ss_x2_offline_with_dino import ObjectGen as SSObjectGen

    ss_config = conditioning_overrides(load_yaml(args.ss_run_dir / "config.yaml"), args)
    shape_config = conditioning_overrides(
        load_yaml(args.shape_run_dir / "config.yaml"), args
    )
    ss_checkpoint_path = resolve_checkpoint(args.ss_run_dir, args.ss_checkpoint)
    shape_checkpoint_path = resolve_checkpoint(args.shape_run_dir, args.shape_checkpoint)

    ss_checkpoint = torch.load(
        ss_checkpoint_path, map_location="cpu", weights_only=False
    )
    ss_state_key, ss_state = checkpoint_state(ss_checkpoint, args.ss_use_ema)
    ss_model = SSObjectGen(ss_config["model"])
    ss_model.load_state_dict(ss_state, strict=True)
    ss_model.to(device).eval()

    shape_checkpoint = torch.load(
        shape_checkpoint_path, map_location="cpu", weights_only=False
    )
    shape_state_key, shape_state = checkpoint_state(
        shape_checkpoint, args.shape_use_ema
    )
    shape_model = ShapeObjectGen(shape_config["model"])
    shape_model.load_state_dict(shape_state, strict=True)
    shape_model.to(device).eval()

    if getattr(ss_model, "dino_model", None) is None:
        raise ValueError("The SS model did not initialize its DINO encoder")
    shape_model.dino_model = None
    shape_model.__dict__["_anyup_upsampler"] = None
    del ss_checkpoint, ss_state, shape_checkpoint, shape_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for module in (ss_model, shape_model):
        for parameter in module.parameters():
            parameter.requires_grad = False

    metadata = {
        "ss_config": str(args.ss_run_dir / "config.yaml"),
        "ss_checkpoint": str(ss_checkpoint_path),
        "ss_state_key": ss_state_key,
        "ss_use_ema": bool(args.ss_use_ema),
        "ss_model_family": args.ss_model_family,
        "shape_config": str(args.shape_run_dir / "config.yaml"),
        "shape_checkpoint": str(shape_checkpoint_path),
        "shape_state_key": shape_state_key,
        "shape_use_ema": bool(args.shape_use_ema),
        "shape_model_family": args.shape_model_family,
        "dino_downsample": int(ss_config["model"]["dino"].get("downsample", 16)),
        "dino_upsample": int(ss_config["model"]["dino"].get("upsample", 1)),
        "max_cond_len": int(ss_config["model"].get("max_cond_len", 512)),
    }
    if metadata["dino_downsample"] % metadata["dino_upsample"] != 0:
        raise ValueError("DINO downsample must be divisible by the AnyUp factor")
    return ss_model, shape_model, metadata


def resolve_vae_decoder_checkpoint(
    root: Path, *, checkpoint: str | Path
) -> tuple[dict[str, Any], Path]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = root / "ckpts" / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing decoder checkpoint: {candidate}")
    return config, candidate.resolve()


def load_x2_decoder(
    root: Path,
    *,
    kind: str,
    use_ema: bool,
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load one release Shape/PBR-X2 VAE decoder and its normalization."""

    from trellis2 import models

    config, checkpoint_path = resolve_vae_decoder_checkpoint(
        root, checkpoint=checkpoint
    )
    model_config = config["models"]["decoder"]
    decoder = getattr(models, model_config["name"])(**model_config.get("args", {}))
    decoder.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    normalization_key = (
        "normalization" if kind == "shape" else "pbr_slat_normalization"
    )
    normalization = config["dataset"]["args"][normalization_key]
    mean = torch.as_tensor(
        normalization["mean"], dtype=torch.float32, device=device
    )
    std = torch.as_tensor(
        normalization["std"], dtype=torch.float32, device=device
    )
    return decoder, mean, std, {
        "checkpoint": str(checkpoint_path),
        "weights": "ema" if use_ema else "normal",
        "normalization_key": normalization_key,
    }


def load_decoders(
    args: Any, device: torch.device
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from trellis2 import models as trellis_models

    from eval.reconstruction.shape_decode import (
        load_shape_vae_x2_decoder,
        load_trellis_shape_decoder,
    )

    ss_config, ss_path = resolve_vae_decoder_checkpoint(
        args.ss_vae_root, checkpoint=args.ss_vae_checkpoint
    )
    ss_decoder_config = ss_config["models"]["decoder"]
    ss_decoder = getattr(trellis_models, ss_decoder_config["name"])(
        **ss_decoder_config.get("args", {})
    )
    ss_decoder.load_state_dict(
        torch.load(ss_path, map_location="cpu", weights_only=True)
    )
    ss_decoder.to(device).eval()

    x2_decoder, x2_path, input_mean, input_std = (
        load_shape_vae_x2_decoder(
            ckpt_root=args.shape_vae_root,
            checkpoint=args.shape_vae_checkpoint,
            use_ema=args.shape_vae_use_ema,
            device=device,
        )
    )
    shape_decoder = load_trellis_shape_decoder(
        args.shape_decoder_pretrained, device=device
    )
    for module in (ss_decoder, x2_decoder, shape_decoder):
        for parameter in module.parameters():
            parameter.requires_grad = False
    metadata = {
        "representation": "lc64_x2",
        "ss_vae_root": str(args.ss_vae_root),
        "ss_vae_decoder": str(ss_path),
        "shape_vae_root": str(args.shape_vae_root),
        "shape_vae_decoder": x2_path,
        "shape_decoder_pretrained": args.shape_decoder_pretrained,
    }
    return ss_decoder, x2_decoder, shape_decoder, (input_mean, input_std), metadata


def call_model_inference(model: Any, **kwargs: Any) -> dict[str, Any]:
    if "inference_only" in inspect.signature(model.forward).parameters:
        kwargs["inference_only"] = True
    return model(**kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(base: int, sample_id: str) -> int:
    import hashlib

    digest = hashlib.blake2b(f"{base}:{sample_id}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def write_point_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for point in points:
            handle.write(f"{float(point[0])} {float(point[1])} {float(point[2])}\n")
