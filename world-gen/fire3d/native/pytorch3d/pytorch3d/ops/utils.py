import torch


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if len(idx) != len(points):
        raise ValueError("points and indices must have the same batch dimension")

    dimensions = points.shape[-1]
    if idx.ndim == 3:
        neighbors = idx.shape[2]
        expanded = idx[..., None].expand(-1, -1, -1, dimensions)
        points = points[:, :, None, :].expand(-1, -1, neighbors, -1)
    elif idx.ndim == 2:
        expanded = idx[..., None].expand(-1, -1, dimensions)
    else:
        raise ValueError(f"unsupported index shape: {tuple(idx.shape)}")

    padding = expanded.eq(-1)
    expanded = expanded.masked_fill(padding, 0)
    selected = points.gather(dim=1, index=expanded)
    return selected.masked_fill(padding, 0.0)
