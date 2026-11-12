#!/usr/bin/env python3
"""Encode original local videos into one self-contained SFT Arrow, without shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import sys
import tempfile
import time
import traceback
from urllib.parse import unquote, urlsplit

import av
import numpy as np
import pyarrow as pa
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hymm.dataset.single_file_sft import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SingleFileSFTDataset,
)
from hymm.models.vae import load_vae


@dataclass
class PreparedVideo:
    pixels: np.ndarray
    source_frames: int
    source_fps: float
    sampled_frames: int
    output_fps: float
    source_height: int
    source_width: int
    sampled_height: int = 704
    sampled_width: int = 1280


def read_arrow(path: Path) -> tuple[pa.Schema, list[dict]]:
    if path.suffix != ".arrow" or not path.is_file():
        raise ValueError("--input-arrow must be one existing .arrow file")
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        schema = reader.schema
        missing = (REQUIRED_COLUMNS | {"video_path"}) - set(schema.names)
        extra = set(schema.names) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS
        if missing or extra:
            raise ValueError(f"Invalid SFT schema: missing={sorted(missing)}, extra={sorted(extra)}")
        expected = {
            "latent": pa.binary(), "latent_shape": pa.list_(pa.int32()),
            "prompts": pa.list_(pa.string()), "segment_boundaries": pa.list_(pa.int32()),
            "camera_extrinsics": pa.binary(), "camera_intrinsics": pa.binary(),
            "video_path": pa.string(), "has_camera": pa.bool_(), "sample_id": pa.string(),
        }
        for field in schema:
            if field.type != expected[field.name]:
                raise ValueError(f"Unexpected type for {field.name}: {field.type}")
        rows = reader.read_all().to_pylist()
    if not rows:
        raise ValueError("Input Arrow has no rows")
    return schema, rows


def select_rows(raw: str, count: int) -> list[int]:
    if raw == "all":
        return list(range(count))
    try:
        values = [int(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError("--rows must be 'all' or comma-separated row indices") from exc
    if not values or len(values) != len(set(values)) or min(values) < 0 or max(values) >= count:
        raise ValueError("--rows contains duplicates or out-of-range indices")
    return sorted(values)


def video_path(value: str, input_arrow: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Row must contain a nonempty video_path")
    parsed = urlsplit(value)
    if parsed.scheme not in {"", "file"} or parsed.netloc not in {"", "localhost"}:
        raise ValueError("video_path must be a local path; downloading is intentionally unsupported")
    path = Path(unquote(parsed.path) if parsed.scheme else value).expanduser()
    if not path.is_absolute():
        path = input_arrow.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def crop_window(source_height: int, source_width: int, height: int, width: int):
    if min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError("Crop dimensions must be positive multiples of 32")
    if source_height < height or source_width < width:
        raise ValueError(
            f"Source {source_height}x{source_width} is smaller than crop {height}x{width}; "
            "resizing and padding are intentionally disabled"
        )
    return (source_height - height) // 2, (source_width - width) // 2


def sample_indices(source_frames: int, source_fps: float, fps: float) -> np.ndarray:
    """Nearest source frames on an exact target-FPS grid, including low-FPS repeats."""
    if source_frames < 4 or source_fps <= 0 or fps <= 0:
        raise ValueError("Need at least four frames and positive FPS")
    count = math.floor((source_frames - 1) / source_fps * fps + 1e-8) + 1
    indices = np.rint(np.arange(count, dtype=np.float64) * source_fps / fps).astype(np.int64)
    indices = np.clip(indices, 0, source_frames - 1)
    if len(indices) < 4:
        raise ValueError("Video is too short after FPS sampling")
    return indices


def prepare_video(row: dict, args) -> PreparedVideo:
    path = video_path(row["video_path"], args.input_arrow)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        source_fps = float(stream.average_rate or stream.guessed_rate or 0)
        source_frames = int(stream.frames)
        if source_frames <= 0:
            source_frames = sum(1 for _ in container.decode(stream))
    indices = sample_indices(source_frames, source_fps, args.fps)
    sampled_frames = len(indices)
    encoded_frames = sampled_frames // 4 * 4
    indices = indices[:encoded_frames]
    pixels = None
    kept = decoded = 0
    last_time = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            decoded += 1
            if frame.time is not None:
                if last_time is not None and frame.time <= last_time:
                    raise ValueError(f"{path}: frame timestamps are not increasing")
                last_time = frame.time
            if pixels is None:
                source_height, source_width = frame.height, frame.width
                rh, rw = args.height, args.width
                top, left = crop_window(source_height, source_width, rh, rw)
                pixels = np.empty((encoded_frames, rh, rw, 3), dtype=np.uint8)
            if frame.height != source_height or frame.width != source_width:
                raise ValueError(f"{path}: video dimensions change mid-stream")
            if kept >= len(indices) or index != indices[kept]:
                continue
            image = frame.to_ndarray(format="rgb24")
            image = image[top:top + rh, left:left + rw]
            while kept < len(indices) and index == indices[kept]:
                pixels[kept] = image
                kept += 1
    if kept != len(indices) or decoded != source_frames:
        raise ValueError(f"{path}: metadata says {source_frames} frames, decoded {decoded}, kept {kept}")
    return PreparedVideo(
        pixels, source_frames, source_fps, sampled_frames, args.fps,
        source_height, source_width, args.height, args.width,
    )


def normalize_pixels(pixels: np.ndarray, device: torch.device, dtype: torch.dtype):
    """Bound normalization intermediates, without dividing VAE encoding into segments."""
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError("Expected [T,H,W,3] uint8 RGB pixels")
    frames, height, width, _ = pixels.shape
    tensor = torch.empty((1, 3, frames, height, width), device=device, dtype=dtype)
    chunk_frames = max(1, min(frames, (256 * 1024 * 1024) // (height * width * 3 * 4)))
    for start in range(0, frames, chunk_frames):
        end = min(frames, start + chunk_frames)
        chunk = torch.from_numpy(pixels[start:end]).to(device=device, dtype=torch.float32)
        chunk.mul_(1.0 / 127.5).sub_(1.0)
        tensor[0, :, start:end].copy_(chunk.permute(3, 0, 1, 2))
    return tensor


def update_boundaries(row: dict, latent_frames: int, output_fps: float, boundary_fps: float):
    original = row["segment_boundaries"]
    prompts = row["prompts"]
    if (
        not isinstance(prompts, list) or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
        or not isinstance(original, list) or len(original) != len(prompts) + 1
        or original[0] != 0
        or any(not isinstance(value, int) for value in original)
        or any(a >= b for a, b in zip(original, original[1:]))
    ):
        raise ValueError("prompts and segment_boundaries must define a nonempty ordered partition")
    for prompt in prompts:
        parsed = json.loads(prompt)
        if (
            not isinstance(parsed, dict) or not isinstance(parsed.get("segments"), list)
            or len(parsed["segments"]) != 1 or not isinstance(parsed["segments"][0], dict)
        ):
            raise ValueError("Each JSON prompt must contain exactly one segment")
    if abs(output_fps - boundary_fps) > 1e-6:
        raise ValueError(
            "Encoding FPS differs from the prompt boundary FPS; regenerate boundaries and "
            "segment-local action timestamps together before encoding"
        )
    boundaries = [*original[:-1], latent_frames]
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Actual latent T is too short for the existing prompt segments")
    return boundaries


def serialize_latent(latent: torch.Tensor) -> tuple[bytes, list[int]]:
    latent = latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if latent.ndim != 4 or latent.shape[0] != 32 or min(latent.shape) <= 0:
        raise ValueError(f"Unexpected DCAE latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("VAE produced NaN or Inf values; refusing to write the row")
    payload = latent.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    return payload, list(latent.shape)


@torch.inference_mode()
def encode_row(row: dict, prepared: PreparedVideo, vae, args, device: torch.device):
    if row.get("has_camera", True) or row["camera_extrinsics"] is not None or row["camera_intrinsics"] is not None:
        raise ValueError(
            "Pre-VAE input must have has_camera=false and null camera columns. "
            "Calibrated camera tensors require an explicit matching preprocessing transform."
        )
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.vae_precision]
    boundaries = update_boundaries(
        row, len(prepared.pixels) // 4, prepared.output_fps, args.boundary_fps
    )
    started = time.monotonic()
    tensor = normalize_pixels(prepared.pixels, device, dtype)
    latent = vae.encode(tensor).latent_dist.mode()[0]
    expected = [32, len(prepared.pixels) // 4, tensor.shape[-2] // 32, tensor.shape[-1] // 32]
    if list(latent.shape) != expected:
        raise ValueError(f"DCAE-V returned {list(latent.shape)}, expected {expected}")
    payload, shape = serialize_latent(latent)
    output = dict(row)
    output["latent"] = payload
    output["latent_shape"] = shape
    output["segment_boundaries"] = boundaries
    SingleFileSFTDataset._decode_row(output, 0)
    stats = {
        "sample_id": row.get("sample_id"),
        "source_frames": prepared.source_frames,
        "source_fps": prepared.source_fps,
        "sampled_frames": prepared.sampled_frames,
        "encoded_frames": len(prepared.pixels),
        "sampling_fps": prepared.output_fps,
        "sampled_resolution": [prepared.sampled_height, prepared.sampled_width],
        "vae_input_resolution": list(prepared.pixels.shape[1:3]),
        "center_crop_origin": [
            (prepared.source_height - prepared.sampled_height) // 2,
            (prepared.source_width - prepared.sampled_width) // 2,
        ],
        "spatial_padding": [0, 0],
        "temporal_padding_frames": 0,
        "temporal_alignment_dropped_frames": prepared.sampled_frames - len(prepared.pixels),
        "latent_shape": shape,
        "latent_bytes": len(payload),
        "encode_seconds": round(time.monotonic() - started, 3),
    }
    del tensor, latent
    if device.type == "cuda":
        stats["peak_gpu_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        torch.cuda.empty_cache()
    return output, stats


def worker_main(device_name: str, tasks: list[tuple[int, dict]], args, results):
    try:
        torch.set_num_threads(1)
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        vae, _, _, _ = load_vae(
            vae_precision=args.vae_precision, vae_path=str(args.vae_checkpoint), device=device
        )
        print(f"[{device}] VAE loaded, {len(tasks)} rows assigned", flush=True)
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            pending = prefetch.submit(prepare_video, tasks[0][1], args) if args.prefetch else None
            for position, (index, row) in enumerate(tasks):
                prepared = pending.result() if pending is not None else prepare_video(row, args)
                pending = (
                    prefetch.submit(prepare_video, tasks[position + 1][1], args)
                    if args.prefetch and position + 1 < len(tasks) else None
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                output, stats = encode_row(row, prepared, vae, args, device)
                stats["device"] = device_name
                results.put(("row", index, output, stats))
                del output, prepared
        results.put(("done", device_name, None, None))
    except BaseException:
        results.put(("error", device_name, None, traceback.format_exc()))


def encode_to_arrow(rows: list[dict], schema: pa.Schema, args, devices: list[str]):
    """GPU workers return rows to a single ordered, atomic Arrow writer."""
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=max(2, len(devices) * 2))
    workers = []
    buffer = {}
    tasks = []
    for index, row in enumerate(rows):
        if row["latent"] is not None or row["latent_shape"] is not None:
            SingleFileSFTDataset._decode_row(row, index)
            buffer[index] = row
        else:
            tasks.append((index, row))
    temporary = None
    started = time.monotonic()
    try:
        if tasks:
            if not args.vae_checkpoint.is_file():
                raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
            for offset, device in enumerate(devices[:len(tasks)]):
                assignments = tasks[offset::min(len(devices), len(tasks))]
                worker = context.Process(target=worker_main, args=(device, assignments, args, results))
                worker.start()
                workers.append(worker)
        with tempfile.NamedTemporaryFile(
            prefix=f".{args.output_arrow.name}.", suffix=".tmp", dir=args.output_arrow.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        next_index = completed_workers = received = 0
        with pa.ipc.new_file(str(temporary), schema) as writer:
            while next_index < len(rows) or completed_workers < len(workers):
                while next_index in buffer:
                    writer.write_table(pa.Table.from_pylist([buffer.pop(next_index)], schema=schema))
                    next_index += 1
                if next_index == len(rows) and completed_workers == len(workers):
                    break
                if not workers:
                    raise RuntimeError("No worker can produce the missing output row")
                try:
                    kind, index, row, detail = results.get(timeout=1)
                except Empty:
                    failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError(f"Encoding worker exited unexpectedly: {[w.exitcode for w in failed]}")
                    if all(worker.exitcode is not None for worker in workers):
                        raise RuntimeError("Workers exited before all output rows were received")
                    continue
                if kind == "error":
                    raise RuntimeError(f"Encoding failed on {index}:\n{detail}")
                if kind == "done":
                    completed_workers += 1
                    continue
                if index in buffer or index < next_index:
                    raise RuntimeError(f"Duplicate output row: {index}")
                buffer[index] = row
                received += 1
                print(f"[{received}/{len(tasks)}] " + json.dumps(detail), flush=True)
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive() or worker.exitcode != 0:
                raise RuntimeError(f"Worker did not exit cleanly: {worker.pid}, {worker.exitcode}")
        with pa.memory_map(str(temporary), "r") as source:
            reader = pa.ipc.open_file(source)
            if not reader.schema.equals(schema, check_metadata=True):
                raise ValueError("Output schema changed")
            validated = 0
            for batch_index in range(reader.num_record_batches):
                for row in reader.get_batch(batch_index).to_pylist():
                    SingleFileSFTDataset._decode_row(row, validated)
                    validated += 1
            if validated != len(rows):
                raise ValueError(f"Expected {len(rows)} rows, got {validated}")
        # A hard link publishes without overwriting an existing output, even if
        # another invocation raced the initial existence check.
        os.link(temporary, args.output_arrow)
        print(json.dumps({
            "output_arrow": str(args.output_arrow), "rows": len(rows),
            "encoded_rows": len(tasks), "preserved_rows": len(rows) - len(tasks),
            "schema_preserved": True, "all_rows_sft_decoded": True,
            "output_size_bytes": args.output_arrow.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }, indent=2), flush=True)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join()
        results.close()
        results.join_thread()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-arrow", type=Path, required=True)
    parser.add_argument("--output-arrow", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu-ids", default="", help="Logical CUDA IDs within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--fps", type=int, choices=(24,), default=24)
    parser.add_argument("--boundary-fps", type=float, default=24, help="FPS used by the input prompt boundaries")
    parser.add_argument("--height", type=int, default=704, help="Center-crop height, a multiple of 32")
    parser.add_argument("--width", type=int, default=1280, help="Center-crop width, a multiple of 32")
    parser.add_argument("--prefetch", type=int, choices=(0, 1), default=1, help="Prepare the next row on CPU during GPU encoding")
    parser.add_argument("--rows", default="all", help="all (default), or explicit smoke-test row indices")
    args = parser.parse_args()
    if (
        args.num_gpus < 1 or min(args.height, args.width) < 32
        or args.height % 32 or args.width % 32
        or not math.isfinite(args.fps) or args.fps <= 0
        or not math.isfinite(args.boundary_fps) or args.boundary_fps <= 0
    ):
        parser.error("GPU count/FPS must be positive; height and width must be multiples of 32")
    if sys.byteorder != "little":
        parser.error("BF16 SFT storage requires a little-endian host")
    args.input_arrow = args.input_arrow.expanduser().resolve()
    args.output_arrow = args.output_arrow.expanduser().resolve()
    args.vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.output_arrow.suffix != ".arrow":
        parser.error("--output-arrow must name one .arrow file")
    if args.output_arrow == args.input_arrow or args.output_arrow.exists():
        parser.error("Refusing to overwrite the input or an existing output")
    schema, rows = read_arrow(args.input_arrow)
    indices = select_rows(args.rows, len(rows))
    rows = [rows[index] for index in indices]
    if args.device == "cpu":
        devices = ["cpu"]
    else:
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; choose --device cpu explicitly")
        try:
            ids = [int(part) for part in args.gpu_ids.split(",")] if args.gpu_ids else list(
                range(min(args.num_gpus, torch.cuda.device_count()))
            )
        except ValueError:
            parser.error("--gpu-ids must be comma-separated integers")
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= torch.cuda.device_count():
            parser.error("--gpu-ids contains duplicate or unavailable devices")
        devices = [f"cuda:{index}" for index in ids]
    args.output_arrow.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "input_arrow": str(args.input_arrow), "output_arrow": str(args.output_arrow),
        "selected_input_rows": indices, "devices": devices, "vae": str(args.vae_checkpoint),
        "fps": args.fps, "sampled_resolution": [args.height, args.width],
        "vae_input_resolution": [args.height, args.width],
        "spatial_processing": "center_crop_no_resize_no_padding",
        "prefetch": args.prefetch,
    }, indent=2), flush=True)
    encode_to_arrow(rows, schema, args, devices)


if __name__ == "__main__":
    main()
