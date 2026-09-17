import torch
import torch.nn as nn

def fourier_encode_vector(vec, output_dim, sample_rate=60):
    """Fourier encode a vector.

    Uses sin/cos(freq * vec); no clipping. Bounded input (e.g. [-1, 1] or [0, 1]) is
    recommended so that freq * vec stays in a reasonable range for meaningful features.

    Args:
        vec: [B, N, D] torch.FloatTensor. Typically in [-1, 1] or [0, 1] per dimension.
        output_dim: int. Target output dimension (will pad/truncate to match exactly).
        sample_rate: int. Controls frequency range (linspace 1 to sample_rate/2, then * pi).

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


def prune_out_of_range_points(coords, mask):
    """
    Prune points outside [-0.5, 0.5]^3 by setting mask to False for those points.

    Args:
        coords: [..., N, 3] float tensor.
        mask: [..., N] bool or float (True/1 = valid).

    Returns:
        mask: same shape and dtype as input; 0/False where coords are out of range.
    """
    in_range = (coords >= -0.5) & (coords <= 0.5)  # [..., N, 3]
    all_in_range = in_range.all(dim=-1)  # [..., N]
    if mask.dtype == torch.bool:
        return mask & all_in_range
    return mask * all_in_range.to(mask.dtype)

def filter_points_in_object_unit_box(coords, feats, instance_ids, object_transforms, max_num_objects):
    """Drop conditioning points outside their assigned object's [-0.5, 0.5]^3 box."""
    import os

    if os.environ.get("FF_DISABLE_UNIT_BOX_FILTER") == "1":
        # Diagnostic: this function is an uncommitted addition (HEAD lacks it).
        # Bypassing it restores the pre-addition conditioning coverage so the
        # filter's effect on PBR albedo can be measured.
        return coords, feats, instance_ids
    if coords.numel() == 0 or instance_ids.numel() == 0:
        return coords, feats, instance_ids

    if torch.is_tensor(max_num_objects):
        max_num_objects = int(max_num_objects.item())
    max_valid_objects = min(int(max_num_objects), int(object_transforms.shape[0]))
    valid_id_mask = (instance_ids >= 0) & (instance_ids < max_valid_objects)
    if not valid_id_mask.any():
        return coords[:0], feats[:0], instance_ids[:0]

    valid_coords = coords[valid_id_mask]
    valid_feats = feats[valid_id_mask]
    valid_ids = instance_ids[valid_id_mask].long()
    transforms = object_transforms[valid_ids]
    ones = torch.ones((valid_coords.shape[0], 1), device=valid_coords.device, dtype=valid_coords.dtype)
    valid_coords_h = torch.cat([valid_coords, ones], dim=-1)
    local_coords = (valid_coords_h[:, None, :] @ transforms.transpose(-2, -1))[:, 0, :3]
    in_unit_box = ((local_coords >= -0.5) & (local_coords <= 0.5)).all(dim=-1)
    return valid_coords[in_unit_box], valid_feats[in_unit_box], valid_ids[in_unit_box].to(instance_ids.dtype)

def fix_empty_context(context, context_mask):
    # context: (B, N, C), context_mask: (B, N)
    # if any item in the batch has zero valid context points (e.g. context_mask[i].any() == False),
    # set the first point to all zeros and mark it as valid (context[i] = 0, context_mask[i, 0] = True).
    empty_rows = ~context_mask.any(dim=1)  # (B,)
    if not empty_rows.any():
        return context, context_mask

    context[:, 0] = torch.where(
        empty_rows.unsqueeze(-1),
        torch.zeros_like(context[:, 0]),
        context[:, 0],
    )
    context_mask[:, 0] = context_mask[:, 0] | empty_rows
    return context, context_mask

class GeometricContextEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 32,
        fourier_dim: int = 64,
        fourier_sample_rate: int = 60,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.fourier_dim = fourier_dim
        self.fourier_sample_rate = fourier_sample_rate

        self.fourier_to_bias = nn.Linear(fourier_dim, num_heads)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, coords, mask=None):
        """
        Args:
            x: [B, N, C]
            coords: [B, N, 3] (already normalized to [-0.5, 0.5]^3)
            mask: [B, N], bool, False means ignore
        Returns:
            out: [B, N, C]
        """
        B, N, C = x.shape
        residual = x
        x = self.norm(x)

        # --- 1. Compute Geometric Bias from Fourier-encoded coordinates ---
        # Use pairwise diff (not absolute coords): bias is per (query_i, key_j), and relative
        # geometry is translation-invariant. diff in [-1, 1]^3 (coords in [-0.5, 0.5]^3).

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 3]
        B, N, _, _ = diff.shape
        diff_flat = diff.reshape(B, N * N, 3)
        fourier_enc = fourier_encode_vector(
            diff_flat, output_dim=self.fourier_dim, sample_rate=self.fourier_sample_rate
        )  # [B, N*N, fourier_dim]
        fourier_enc = fourier_enc.reshape(B, N, N, self.fourier_dim)
        attn_bias = self.fourier_to_bias(fourier_enc).permute(0, 3, 1, 2)  # [B, num_heads, N, N]

        # --- 2. Geometric Self-Attention ---
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (C // self.num_heads) ** -0.5
        attn_logits = (q @ k.transpose(-2, -1)) * scale

        attn_logits = attn_logits + attn_bias

        if mask is not None:
            # Mask Keys
            mask_expanded = mask.view(B, 1, 1, N).float()
            attn_logits = attn_logits.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # --- FIX 2: Cleaner NaN handling ---
        # While nan_to_num works, checking the mask prevents 'dirty' gradients.
        # But nan_to_num is acceptable here provided gradients aren't exploding.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Mask Attention Weights (Queries)
        # Ensure padded queries don't aggregate information (optimization)
        if mask is not None:
            attn_weights = attn_weights * mask.view(B, 1, N, 1).float()

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)

        # return self.to_out(out) + residual
        # FIX 2: Mask the output to remove Linear bias from padded tokens
        final_out = self.to_out(out) + residual
        if mask is not None:
            final_out = final_out * mask.unsqueeze(-1).float()

        return final_out


class FourierAddContextEncoder(nn.Module):
    """Context encoder that injects position via direct addition: x + fourier(coords).

    Fourier encoding output dim is channels (no extra linear). No self-attention.
    Invalid positions (mask=False) get zero position encoding and zero output.
    Same (x, coords, mask) interface so it can be swapped in config.
    """

    def __init__(
        self,
        channels: int,
        fourier_sample_rate: int = 60,
    ):
        super().__init__()
        self.channels = channels
        self.fourier_sample_rate = fourier_sample_rate

    def forward(self, x, coords, mask=None):
        """
        Args:
            x: [B, N, C]
            coords: [B, N, 3] (normalized to [-0.5, 0.5]^3)
            mask: [B, N], bool, False means ignore
        Returns:
            out: [B, N, C]
        """
        normalized_coords = coords + 0.5 # normalize to [0, 1]^3
        pos_enc = fourier_encode_vector(
            normalized_coords, output_dim=self.channels, sample_rate=self.fourier_sample_rate
        )  # [B, N, C]
        x = x + pos_enc
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


def get_context_encoder(config: dict) -> nn.Module:
    """Build a context encoder from config. Use 'type': 'fourier_add' for direct-add encoding."""
    cfg = dict(config)
    encoder_type = cfg.pop("type", "fourier_add")
    print(f"Using context encoder type: {encoder_type}")
    if encoder_type == "geometric":
        return GeometricContextEncoder(**cfg)
    if encoder_type == "fourier_add":
        return FourierAddContextEncoder(**cfg)
    raise ValueError(f"Unknown context_encoder type: {encoder_type}")
