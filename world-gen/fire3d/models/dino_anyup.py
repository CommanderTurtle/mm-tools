"""Helpers for optional AnyUp DINO feature upsampling."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch


DINO_PATCH_DOWNSAMPLE = 16
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANYUP_REPO = REPO_ROOT / "third_party/anyup"


def validate_dino_upsample(upsample: int, dino_downsample: int = DINO_PATCH_DOWNSAMPLE) -> int:
    upsample = int(upsample)
    dino_downsample = int(dino_downsample)
    if upsample < 1:
        raise ValueError(f"dino upsample must be >= 1, got {upsample}")
    if dino_downsample % upsample != 0:
        raise ValueError(
            f"dino upsample must divide dino downsample {dino_downsample}, got {upsample}"
        )
    return upsample


def dino_feature_downsample(upsample: int, dino_downsample: int = DINO_PATCH_DOWNSAMPLE) -> int:
    return int(dino_downsample) // validate_dino_upsample(upsample, dino_downsample)


def load_anyup_upsampler(dino_config: dict, device: torch.device | str):
    local_repo = DEFAULT_ANYUP_REPO
    repo = dino_config.get(
        "anyup_repo", str(local_repo) if local_repo.is_dir() else "wimmerth/anyup"
    )
    model_name = dino_config.get("anyup_model", "anyup_multi_backbone")
    source = dino_config.get(
        "anyup_source", "local" if Path(repo).is_dir() else "github"
    )
    use_natten = bool(dino_config.get("anyup_use_natten", True))
    upsampler = torch.hub.load(
        repo,
        model_name,
        source=source,
        use_natten=use_natten,
    )
    upsampler = upsampler.to(device).eval()
    for param in upsampler.parameters():
        param.requires_grad = False
    return upsampler


def move_module_to_device(module, device: torch.device | str):
    try:
        param = next(module.parameters())
    except StopIteration:
        param = None
    if param is None or param.device != torch.device(device):
        module.to(device)
    module.eval()
    return module


def get_anyup_tile_grid(image_downsample: int, anyup_tile_grid: int) -> int:
    if anyup_tile_grid > 0:
        return int(anyup_tile_grid)
    if image_downsample == 1:
        return 4
    if image_downsample == 2:
        return 2
    return 1


def split_evenly(length: int, num_splits: int) -> list[int]:
    edges = np.linspace(0, length, num_splits + 1)
    return np.rint(edges).astype(np.int64).tolist()


def is_anyup_batch_split_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return (
        "canUse32BitIndexMath" in msg
        or "CUDA out of memory" in msg
        or "CUBLAS_STATUS_ALLOC_FAILED" in msg
        or "out of memory" in msg.lower()
    )


def run_anyup_upsampler_batched(anyup_upsampler, image, features, output_size, anyup_kwargs):
    try:
        return anyup_upsampler(
            image,
            features,
            output_size=output_size,
            **anyup_kwargs,
        )
    except RuntimeError as exc:
        if image.shape[0] <= 1 or not is_anyup_batch_split_error(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        mid = image.shape[0] // 2
        left = run_anyup_upsampler_batched(
            anyup_upsampler,
            image[:mid].contiguous(),
            features[:mid].contiguous(),
            output_size,
            anyup_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        right = run_anyup_upsampler_batched(
            anyup_upsampler,
            image[mid:].contiguous(),
            features[mid:].contiguous(),
            output_size,
            anyup_kwargs,
        )
        return torch.cat([left, right], dim=0)


@torch.no_grad()
def run_dino_anyup_features(
    *,
    dino_model,
    anyup_upsampler,
    rgbs_tensor: torch.Tensor,
    dino_downsample: int,
    dino_upsample: int,
    anyup_q_chunk_size: int | None = None,
    anyup_tile_grid: int = 0,
    anyup_frame_batch_size: int = 1,
) -> torch.Tensor:
    validate_dino_upsample(dino_upsample, dino_downsample)
    image_downsample = dino_feature_downsample(dino_upsample, dino_downsample)
    n_total, _, height, width = rgbs_tensor.shape
    output_h = height // image_downsample
    output_w = width // image_downsample
    patch_h = height // dino_downsample
    patch_w = width // dino_downsample
    if output_h <= 0 or output_w <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError(
            f"Invalid DINO/AnyUp shapes: rgbs={tuple(rgbs_tensor.shape)}, "
            f"dino_downsample={dino_downsample}, dino_upsample={dino_upsample}"
        )

    frame_batch_size = max(1, int(anyup_frame_batch_size))
    tile_grid = get_anyup_tile_grid(image_downsample, int(anyup_tile_grid))
    tile_grid = max(1, min(tile_grid, patch_h, patch_w))
    anyup_kwargs = {}
    if anyup_q_chunk_size is not None:
        anyup_kwargs["q_chunk_size"] = int(anyup_q_chunk_size)

    frame_features = []
    for frame_start in range(0, n_total, frame_batch_size):
        frame_end = min(frame_start + frame_batch_size, n_total)
        rgbs_frame = rgbs_tensor[frame_start:frame_end].contiguous()
        n = rgbs_frame.shape[0]
        outputs = dino_model(rgbs_frame, is_training=True)
        patch_tokens = outputs["x_norm_patchtokens"]
        expected_tokens = patch_h * patch_w
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"DINO patch token count mismatch: got {patch_tokens.shape[1]}, "
                f"expected {expected_tokens} from input shape {(height, width)}."
            )
        lr_features = patch_tokens.reshape(n, patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()

        if tile_grid == 1:
            hr_features = run_anyup_upsampler_batched(
                anyup_upsampler,
                rgbs_frame,
                lr_features,
                (output_h, output_w),
                anyup_kwargs,
            )
        else:
            y_edges = split_evenly(patch_h, tile_grid)
            x_edges = split_evenly(patch_w, tile_grid)
            tile_rows = []
            for y_tile in range(tile_grid):
                tile_cols = []
                patch_y0, patch_y1 = y_edges[y_tile], y_edges[y_tile + 1]
                image_y0 = patch_y0 * dino_downsample
                image_y1 = height if y_tile == tile_grid - 1 else patch_y1 * dino_downsample
                out_y0 = image_y0 // image_downsample
                out_y1 = output_h if y_tile == tile_grid - 1 else image_y1 // image_downsample
                for x_tile in range(tile_grid):
                    patch_x0, patch_x1 = x_edges[x_tile], x_edges[x_tile + 1]
                    image_x0 = patch_x0 * dino_downsample
                    image_x1 = width if x_tile == tile_grid - 1 else patch_x1 * dino_downsample
                    out_x0 = image_x0 // image_downsample
                    out_x1 = output_w if x_tile == tile_grid - 1 else image_x1 // image_downsample
                    image_tile = rgbs_frame[:, :, image_y0:image_y1, image_x0:image_x1].contiguous()
                    lr_tile = lr_features[:, :, patch_y0:patch_y1, patch_x0:patch_x1].contiguous()
                    hr_tile = run_anyup_upsampler_batched(
                        anyup_upsampler,
                        image_tile,
                        lr_tile,
                        (out_y1 - out_y0, out_x1 - out_x0),
                        anyup_kwargs,
                    )
                    tile_cols.append(hr_tile)
                tile_rows.append(torch.cat(tile_cols, dim=-1))
            hr_features = torch.cat(tile_rows, dim=-2)

        frame_features.append(hr_features.permute(0, 2, 3, 1).reshape(n * output_h * output_w, -1))
        del outputs, patch_tokens, lr_features, hr_features

    return torch.cat(frame_features, dim=0)
