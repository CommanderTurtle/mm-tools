"""CPU handoff for video saving; no torch views escape into the entry loop."""

import torch


def video_to_uint8_numpy(video: torch.Tensor, chunk_frames: int | None = None):
    """Convert [C,T,H,W] display-range video with the original pixel formula.

    None preserves the whole-video conversion. A positive frame count bounds
    GPU temporaries while copying each converted block directly into the final
    CPU uint8 buffer. Copies complete before NumPy exposes that buffer.
    """
    if chunk_frames is not None and (type(chunk_frames) is not int or chunk_frames <= 0):
        raise ValueError("chunk_frames must be None or a positive integer")
    frames = video.permute(1, 2, 3, 0)
    if chunk_frames is None:
        return (frames * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    output = torch.empty(tuple(frames.shape), dtype=torch.uint8, device="cpu")
    for start in range(0, frames.shape[0], chunk_frames):
        stop = min(start + chunk_frames, frames.shape[0])
        block = (frames[start:stop] * 255).clamp(0, 255).to(torch.uint8)
        output[start:stop].copy_(block, non_blocking=False)
        del block
    return output.numpy()
