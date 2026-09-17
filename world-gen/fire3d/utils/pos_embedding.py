import torch

def fourier_encode_vector(vec, output_dim, sample_rate=60):
    """Fourier encode a vector.

    Args:
        vec: [B, N, D] torch.FloatTensor.
        output_dim: int. Target output dimension (will pad/truncate to match exactly).
        sample_rate: int.

    Returns:
        [B, N, output_dim] torch.FloatTensor.
    """
    b, n, d = vec.shape
    # Calculate num_bands to get closest match: output = (2 * num_bands + 1) * d
    num_bands = round((output_dim / d - 1) / 2)
    num_bands = max(1, num_bands)  # Ensure at least 1 band

    samples = torch.linspace(1, sample_rate / 2, num_bands).to(vec.device) * torch.pi
    sines = torch.sin(samples[None, None, :, None] * vec[:, :, None, :])
    cosines = torch.cos(samples[None, None, :, None] * vec[:, :, None, :])

    encoding = torch.stack([sines, cosines], dim=3).reshape(b, n, 2 * num_bands, d)
    encoding = torch.cat([vec[:, :, None, :], encoding], dim=2)
    encoding = encoding.flatten(2)

    # Pad or truncate to exact output_dim
    current_dim = encoding.shape[-1]
    if current_dim < output_dim:
        padding = torch.zeros(b, n, output_dim - current_dim, device=vec.device, dtype=vec.dtype)
        encoding = torch.cat([encoding, padding], dim=-1)
    elif current_dim > output_dim:
        encoding = encoding[..., :output_dim]

    return encoding
