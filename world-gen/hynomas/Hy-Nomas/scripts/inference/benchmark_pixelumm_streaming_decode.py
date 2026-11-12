#!/usr/bin/env python3
"""PixelUMM per-segment (stable-prefix) streaming decode vs full-clip decode.

Input: saved DiT latents ``<name>.npy`` ([32,T,H,W]) with their ``<name>.json`` (pixel
``frame_boundaries``), as written by ``inference.save_latents=true``. For every sample:

* reference: ``decode`` of the whole clip (what ``seg_ar_segment_vae_decode=false`` does);
* streaming: one ``push_preview`` per segment with the pipeline's own latent/pixel boundary
  arithmetic, assembled by in-place replacing the provisional tail (what
  ``seg_ar_segment_vae_decode=true model.vae.stable_prefix_streaming=true`` does).

Reports numeric difference and per-push timings. Run under torchrun (see the ``.sh`` wrapper).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder

TEMPORAL_RATIO = 4


def save_clip_mp4(
    clip_tchw_01: torch.Tensor,
    path: str,
    fps: int = 8,
) -> Optional[str]:
    """Save ``[T, 3, H, W]`` in [0, 1] -> mp4, matching master_nomas_1.5 eval_video.py.

    Returns the actual mp4/GIF path, or None if imageio is missing.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio  # type: ignore
        except ImportError:
            return None
    clip = clip_tchw_01.detach().float().cpu().clamp(0, 1)
    arr = (clip.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if str(path).lower().endswith(".gif"):
        imageio.mimsave(path, list(arr), duration=1000.0 / max(1, fps))
        return path
    try:
        imageio.mimsave(path, list(arr), fps=fps, codec="libx264", quality=8)
        return path
    except Exception:
        gif_path = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(
            gif_path,
            list(arr),
            duration=1000.0 / max(1, fps),
        )
        return gif_path


def latent_boundaries(pixel_boundaries, latent_frames):
    """Pixel -> latent segment boundaries, exactly as ``causal_ar.py`` converts them."""
    result = []
    for b in pixel_boundaries:
        if b == 0:
            result.append(0)
        else:
            result.append(max(result[-1] + 1, min(b // TEMPORAL_RATIO, latent_frames)))
    result[0] = 0
    result[-1] = latent_frames
    return result


def stream_decode(model, latent, pixel_boundaries, *, sync=None):
    """Segment-by-segment ``push_preview`` decode, as the pipeline drives it.

    Returns ``(video, pushes)``; ``video`` is ``None`` on ranks that do not hold the output.
    """
    num_frames = int(pixel_boundaries[-1])
    latent_frames = int(latent.shape[2])
    if (num_frames - 1) // TEMPORAL_RATIO + 1 != latent_frames:
        raise ValueError(f"{num_frames} pixel frames do not match {latent_frames} latent frames")
    bounds = latent_boundaries(pixel_boundaries, latent_frames)
    session = model.create_streaming_decode_session(to_cpu=True, retain_history=False)
    video = None
    pushes = []
    for index in range(len(bounds) - 1):
        end = bounds[index + 1]
        target = num_frames if index == len(bounds) - 2 else min(num_frames, end * TEMPORAL_RATIO)
        segment_pixels = target - session.available_pixel_frames
        if sync:
            sync()
        started = time.perf_counter()
        session.final_segment = index == len(bounds) - 2  # nothing follows to correct a padded tail
        update = session.push_preview(latent[:, :, bounds[index] : end], pixel_frames=segment_pixels)
        if sync:
            sync()
        elapsed = time.perf_counter() - started
        if update.video.numel() > 0:
            kept = update.pixel_start
            video = update.video if video is None else torch.cat([video[:, :, :kept], update.video], dim=2)
        pushes.append(
            {
                "segment": index,
                "latent_frames": end - bounds[index],
                "pixel_frames": segment_pixels,
                "replaced_frames": update.replace_pixel_frames,
                "stable_pixel_frames": update.stable_pixel_frames,
                "provisional_pixel_frames": update.provisional_pixel_frames,
                "seconds": elapsed,
            }
        )
    return video, pushes


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    total, sum_sq, max_abs, unequal, u8_unequal = 0, 0.0, 0.0, 0, 0
    for start in range(0, reference.shape[2], 4):
        ref = reference[:, :, start : start + 4].float()
        cand = candidate[:, :, start : start + 4].float()
        delta = cand - ref
        total += delta.numel()
        sum_sq += float(delta.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        unequal += int((delta != 0).sum())
        to_u8 = lambda x: ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        u8_unequal += int((to_u8(ref) != to_u8(cand)).sum())
    mse = sum_sq / max(total, 1)
    return {
        "shape_equal": True,
        "torch_equal": unequal == 0,
        "unequal_element_count": unequal,
        "uint8_unequal_element_count": u8_unequal,
        "element_count": total,
        "mse": mse,
        "max_abs": max_abs,
        "psnr_db_for_range_minus1_plus1": math.inf if mse == 0.0 else 10.0 * math.log10(4.0 / mse),
    }


def measure_offload(model, device, sync):
    """Time the decoder GPU->CPU->GPU move the pipeline does around every segment.

    ``causal_ar.py`` only does it with ``inference.interactive_kv_cache_cpu_mode=phase_swap``
    (``_offload_interactive_pixelumm`` / ``_restore_interactive_pixelumm``): ``self.vae.to("cpu")`` after a
    segment's decode, ``self.vae.to(device)`` before the next one. Same calls here.
    """
    sync()
    started = time.perf_counter()
    model.to(device="cpu")
    sync()
    to_cpu = time.perf_counter() - started
    started = time.perf_counter()
    model.to(device=device)
    sync()
    to_gpu = time.perf_counter() - started
    gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    return {"to_cpu_seconds": to_cpu, "to_gpu_seconds": to_gpu, "parameter_gib": gib}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-dir", default=None, help="if set, save each streamed video as <name>.mp4 here")
    parser.add_argument("--save-fps", type=int, default=24)
    parser.add_argument("--names", nargs="*", default=None, help="sample stems; default: first --max-samples")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-window-latents", type=int, default=10)
    parser.add_argument("--temporal-overlap-latents", type=int, default=5)
    parser.add_argument("--tile-pixels", type=int, default=0)
    parser.add_argument("--tile-overlap-pixels", type=int, default=64)
    parser.add_argument("--local-attention-tile-pixels", type=int, default=128)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    model = PixelUMMDCAELatentDecoder.from_checkpoint(
        args.checkpoint,
        dtype=torch.bfloat16,
        device=device,
        local_attention_tile_pixels=args.local_attention_tile_pixels,
        num_steps=1,
        seed=args.seed,
        temporal_window_latents=args.temporal_window_latents,
        temporal_overlap_latents=args.temporal_overlap_latents,
        decoder_tile_pixels=args.tile_pixels,
        decoder_tile_overlap_pixels=args.tile_overlap_pixels,
    )
    if world_size > 1:
        model.configure_distributed_decode("window", process_group=dist.group.WORLD, output_rank=0)

    latent_dir = Path(args.latent_dir)
    names = args.names or [p.stem for p in sorted(latent_dir.glob("*.npy"))[: args.max_samples]]
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    def sync():
        torch.cuda.synchronize(device)
        dist.barrier()

    reports = []
    warmed = False
    for name in names:
        array = np.load(latent_dir / f"{name}.npy")
        if array.ndim != 4 or array.shape[0] != 32:
            raise ValueError(f"{name}: expected [32,T,H,W], got {array.shape}")
        meta = json.loads((latent_dir / f"{name}.json").read_text())
        pixel_boundaries = [int(v) for v in meta["frame_boundaries"]]
        latent = torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        frames = pixel_boundaries[-1]

        with torch.inference_mode():
            if not warmed:  # first call pays kernel / FlexAttention compilation
                model.decode(latent[:, :, : args.temporal_window_latents], return_dict=False)
                sync()
                warmed = True
                offload = measure_offload(model, device, sync)
                if rank == 0:
                    print(f"decoder offload round trip: {offload}", flush=True)

            sync()
            started = time.perf_counter()
            reference = model.decode(latent, return_dict=False, pixel_frames=frames)[0]
            sync()
            reference_seconds = time.perf_counter() - started
            reference = reference.cpu() if rank == 0 else None
            torch.cuda.empty_cache()

            streamed, pushes = stream_decode(model, latent, pixel_boundaries, sync=sync)
            torch.cuda.empty_cache()

        if rank == 0 and args.video_dir:
            # [1, 3, T, H, W] in [-1, 1] -> [T, 3, H, W] in [0, 1], as decode_latents.py does.
            clip = ((streamed.float() + 1.0) * 0.5).clamp(0.0, 1.0)[0].permute(1, 0, 2, 3)
            # Hidden partial name, renamed when complete (same as decode_latents.py).
            partial = os.path.join(args.video_dir, "." + name + ".partial.mp4")
            written = save_clip_mp4(clip.cpu(), partial, fps=args.save_fps)
            if not written:
                raise RuntimeError(f"{name}: imageio missing, cannot save mp4")
            os.replace(written, os.path.join(args.video_dir, name + os.path.splitext(written)[1]))

        if rank == 0:
            report = {
                "name": name,
                "latent_shape": list(latent.shape),
                "pixel_frames": frames,
                "segments": len(pushes),
                "world_size": world_size,
                "full_clip_decode_seconds": reference_seconds,
                "streaming_total_seconds": sum(p["seconds"] for p in pushes),
                "streaming_first_push_seconds": pushes[0]["seconds"],
                "streaming_max_push_seconds": max(p["seconds"] for p in pushes),
                "decoder_offload_round_trip": offload,
                "streaming_total_with_offload_seconds": sum(p["seconds"] for p in pushes)
                + len(pushes) * (offload["to_cpu_seconds"] + offload["to_gpu_seconds"]),
                "pushes": pushes,
                "numeric": compare(reference, streamed.cpu()),
            }
            (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
            reports.append(report)
            print(
                f"[{name}] T={latent.shape[2]} segs={len(pushes)} "
                f"full={reference_seconds:.2f}s stream_total={report['streaming_total_seconds']:.2f}s "
                f"first_push={report['streaming_first_push_seconds']:.2f}s "
                f"max_push={report['streaming_max_push_seconds']:.2f}s "
                f"torch_equal={report['numeric'].get('torch_equal')} "
                f"max_abs={report['numeric'].get('max_abs')} "
                f"psnr={report['numeric'].get('psnr_db_for_range_minus1_plus1')}",
                flush=True,
            )
        del reference, streamed

    failed = False
    if rank == 0:
        failed = any(not r["numeric"].get("torch_equal") for r in reports)
        summary = {
            "checkpoint": args.checkpoint,
            "samples": [r["name"] for r in reports],
            "all_torch_equal": not failed,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
