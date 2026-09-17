import torch
import spconv.pytorch as spconv
from einops import repeat

from torch import nn
from torch.nn import functional as F

from pytorch3d.ops import sample_farthest_points

class SparseGroupNorm(nn.Module):
    """GroupNorm wrapper for use in spconv.SparseSequential.

    Note: spconv.SparseSequential automatically extracts .features from
    sparse tensors for non-sparse modules and calls replace_feature() on
    the result. So this module receives plain tensors, not SparseConvTensors.
    """
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SparseReLU(nn.Module):
    """ReLU wrapper for use in spconv.SparseSequential.

    Note: spconv.SparseSequential automatically extracts .features from
    sparse tensors for non-sparse modules and calls replace_feature() on
    the result. So this module receives plain tensors, not SparseConvTensors.
    """
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


def make_conv3d_sparse(
    channels_in,
    channels_out,
    kernel_size=3,
    num_groups=8,
):
    num_groups = min(num_groups, channels_out)
    block = spconv.SparseSequential(
        spconv.SubMConv3d(channels_in, channels_out, kernel_size=kernel_size, bias=True),
        SparseGroupNorm(num_groups, channels_out),
        SparseReLU(inplace=True),
    )
    return block


# def make_conv3d_downscale_sparse(
#     channels_in,
#     channels_out,
#     num_groups=8,
# ):
#     num_groups = min(num_groups, channels_out)
#     block = spconv.SparseSequential(
#         spconv.SparseConv3d(channels_in, channels_out, kernel_size=2, stride=2, bias=True),
#         SparseGroupNorm(num_groups, channels_out),
#         SparseReLU(inplace=True),
#     )
#     return block

def make_conv3d_downscale_sparse(
    channels_in,
    channels_out,
    indice_key, # NEW: Added to track the downsampling coordinate map
    num_groups=8,
):
    num_groups = min(num_groups, channels_out)
    block = spconv.SparseSequential(
        spconv.SparseConv3d(
            channels_in, channels_out, kernel_size=2, stride=2, bias=True, indice_key=indice_key
        ),
        SparseGroupNorm(num_groups, channels_out),
        SparseReLU(inplace=True),
    )
    return block

def make_conv3d_upscale_sparse(
    channels_in,
    channels_out,
    indice_key, # NEW: Reuses the map from the encoder
    num_groups=8,
):
    num_groups = min(num_groups, channels_out)
    block = spconv.SparseSequential(
        # SparseInverseConv3d is the magic here. It perfectly reverses the downsampling step
        # that shares the same indice_key. No stride needed, as it looks up the hash map.
        spconv.SparseInverseConv3d(
            channels_in, channels_out, kernel_size=2, bias=True, indice_key=indice_key
        ),
        SparseGroupNorm(num_groups, channels_out),
        SparseReLU(inplace=True),
    )
    return block

def cat_sparse(x: spconv.SparseConvTensor, y: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    """Helper to concatenate two sparse tensors that share the same indices."""
    new_feats = torch.cat([x.features, y.features], dim=1)
    return x.replace_feature(new_feats)


class ResBlockSparse(spconv.SparseModule):
    """Residual block for sparse tensors.

    Inherits from spconv.SparseModule so that spconv.SparseSequential
    recognizes it as a sparse module and passes SparseConvTensor directly.
    """
    def __init__(
        self,
        channels,
        num_groups=8,
    ):
        super().__init__()

        self.block0 = make_conv3d_sparse(
            channels, channels, num_groups=num_groups
        )
        self.block1 = make_conv3d_sparse(
            channels, channels, num_groups=num_groups
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x
        out = self.block0(x)
        out = self.block1(out)
        # Add residual: combine features and keep the same sparse structure
        out_feats = identity.features + out.features
        return out.replace_feature(out_feats)


class SparseBottleneck(nn.Module):
    """Bottleneck that applies linear layers to sparse tensor features."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_channels, 2 * in_channels),
            nn.GroupNorm(8, 2 * in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * in_channels, out_channels),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        feats = self.bottleneck(x.features)
        return x.replace_feature(feats)


# class ResNet3DSparse(nn.Module):
#     def __init__(self, dim_in, dim_out, layers):
#         super().__init__()

#         self.stem = spconv.SparseSequential(
#             spconv.SubMConv3d(dim_in, layers[0], kernel_size=7, bias=True),
#             SparseGroupNorm(min(8, layers[0]), layers[0]),
#             SparseReLU(inplace=True),
#             ResBlockSparse(layers[0]),
#         )

#         # Number of down-convs is len(layers) - 1
#         blocks = []
#         for i in range(len(layers) - 1):
#             blocks.append(
#                 spconv.SparseSequential(
#                     spconv.SparseConv3d(layers[i], layers[i + 1], kernel_size=2, stride=2, bias=True),
#                     SparseGroupNorm(min(8, layers[i + 1]), layers[i + 1]),
#                     SparseReLU(inplace=True),
#                     ResBlockSparse(layers[i + 1]),
#                     ResBlockSparse(layers[i + 1]),
#                 )
#             )
#         self.blocks = spconv.SparseSequential(*blocks)

#         self.bottleneck = SparseBottleneck(layers[-1], dim_out)

#     def forward(self, x):
#         out = self.stem(x)
#         out = self.blocks(out)
#         out = self.bottleneck(out)
#         return out

class SparseEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, layers):
        super().__init__()

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(dim_in, layers[0], kernel_size=7, bias=True),
            SparseGroupNorm(min(8, layers[0]), layers[0]),
            SparseReLU(inplace=True),
            ResBlockSparse(layers[0]),
        )

        self.blocks = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.blocks.append(
                spconv.SparseSequential(
                    # Tag this layer with a unique ID
                    make_conv3d_downscale_sparse(layers[i], layers[i + 1], indice_key=f"down_{i}"),
                    ResBlockSparse(layers[i + 1]),
                    ResBlockSparse(layers[i + 1]),
                )
            )

        self.bottleneck = SparseBottleneck(layers[-1], dim_out)

    def forward(self, x):
        skips = []

        out = self.stem(x)
        skips.append(out) # Skip 0 matches original voxel resolution

        for block in self.blocks:
            out = block(out)
            skips.append(out) # Skips 1 to N match progressively smaller resolutions

        out = self.bottleneck(out)

        # We drop the last skip from the list because it's what goes into the bottleneck anyway
        return out, skips[:-1]

class SparseDecoder(nn.Module):
    def __init__(self, dim_in, layers):
        super().__init__()

        # Reverse the bottleneck
        self.bottleneck_reverse = SparseBottleneck(dim_in, layers[-1])

        self.up_blocks = nn.ModuleList()
        # Iterate backwards through the layers to construct the decoder
        for i in reversed(range(len(layers) - 1)):
            in_ch = layers[i + 1]
            out_ch = layers[i]

            # Upscale using the exact same key from the encoder
            up_conv = make_conv3d_upscale_sparse(in_ch, out_ch, indice_key=f"down_{i}")

            # Process the concatenated features (upscaled + skip connection)
            process = spconv.SparseSequential(
                make_conv3d_sparse(out_ch * 2, out_ch), # *2 because of concatenation
                ResBlockSparse(out_ch),
                ResBlockSparse(out_ch)
            )

            self.up_blocks.append(nn.ModuleDict({
                'up': up_conv,
                'process': process
            }))

    def forward(self, x, skips):
        out = self.bottleneck_reverse(x)

        # Iterate backwards through the skips
        for i, up_block in enumerate(self.up_blocks):
            # Pop the corresponding skip connection
            skip = skips[-(i + 1)]

            # 1. Inverse Sparse Convolution (projects back to skip connection's coordinates)
            out = up_block['up'](out)

            # 2. Concatenate features
            out = cat_sparse(out, skip)

            # 3. Process to smooth out the concatenation
            out = up_block['process'](out)

        return out

class SparseUNet(nn.Module):
    def __init__(self, dim_in, dim_latent, layers=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = SparseEncoder(dim_in, dim_latent, layers)
        self.decoder = SparseDecoder(dim_latent, layers)


    def forward(self, x):
        latent, skips = self.encoder(x)
        decoded = self.decoder(latent, skips)
        return latent, decoded

def index_batched_sparse_tensor(sparse_tensor: spconv.SparseConvTensor, index: int):
    """Index into a batched spconv.SparseConvTensor.

    Args:
        sparse_tensor: a spconv.SparseConvTensor with batched data.
        index: int.

    Returns:
        spconv.SparseConvTensor with single batch.
    """
    batch_mask = sparse_tensor.indices[:, 0] == index
    coords = sparse_tensor.indices[batch_mask]  # Keep batch dim for consistency
    coords = coords.clone()
    coords[:, 0] = 0  # Reset batch index to 0
    feats = sparse_tensor.features[batch_mask]
    return spconv.SparseConvTensor(
        features=feats,
        indices=coords,
        spatial_shape=sparse_tensor.spatial_shape,
        batch_size=1,
    )


def sparse_uncollate(sparse_tensor: spconv.SparseConvTensor):
    """Un-Collate a batched spconv.SparseConvTensor.

    Args:
        sparse_tensor: a batched spconv.SparseConvTensor.

    Returns:
        List[spconv.SparseConvTensor].
    """
    batch_size = sparse_tensor.batch_size

    sparse_tensor_list = []
    for b in range(batch_size):
        sparse_tensor_list.append(index_batched_sparse_tensor(sparse_tensor, b))
    return sparse_tensor_list


def vox_to_sequence(sparse_tensor: spconv.SparseConvTensor, max_len_output: int = 4096):
    """Compute sequence from sparse point cloud.
    Fully vectorized implementation using scatter operations.
    Output is padded/truncated to max_len_output.

    Args:
        sparse_tensor: spconv.SparseConvTensor.
        max_len_output: Fixed output sequence length. Sequences longer than this will be truncated,
                        shorter sequences will be padded.

    Returns:
        Dict with the following keys:
            seq: [B, max_len_output, C] torch.FloatTensor.
            coords: [B, max_len_output, 3] torch.IntTensor. To be used with embeddings for Transformers.
            mask: [B, max_len_output] torch.BoolTensor. To be used with Transformers.
                  True indicates padding/truncated positions, False indicates valid positions.
    """
    batch_size = sparse_tensor.batch_size
    indices = sparse_tensor.indices  # [N_total, 4] - (batch, x, y, z)
    features = sparse_tensor.features  # [N_total, C]

    device = features.device
    channels = features.shape[-1]

    # Get batch indices and spatial coords
    batch_indices = indices[:, 0]  # [N_total]
    spatial_coords = indices[:, 1:]  # [N_total, 3]

    # Count points per batch using bincount (fully vectorized)
    counts_per_batch = torch.bincount(batch_indices, minlength=batch_size)  # [B]

    # Compute position within each batch using cumsum trick
    # For each point, compute its position index within its batch
    ones = torch.ones(indices.shape[0], device=device, dtype=torch.long)
    cumsum = torch.zeros(indices.shape[0] + 1, device=device, dtype=torch.long)
    cumsum[1:] = torch.cumsum(ones, dim=0)

    # Get start index for each batch
    batch_start_indices = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
    batch_start_indices[1:] = torch.cumsum(counts_per_batch, dim=0)

    # Position within batch = global_position - batch_start
    position_in_batch = cumsum[:-1] - batch_start_indices[batch_indices]

    # Filter out points that exceed max_len_output (truncation)
    valid_mask = position_in_batch < max_len_output
    # sample_farthest_points()
    batch_indices = batch_indices[valid_mask]
    spatial_coords = spatial_coords[valid_mask]
    features = features[valid_mask]
    position_in_batch = position_in_batch[valid_mask]

    # # Initialize output tensors with fixed length max_len_output
    # seq = torch.zeros(batch_size, max_len_output, channels, dtype=features.dtype, device=device)
    # coords = torch.zeros(batch_size, max_len_output, 3, dtype=indices.dtype, device=device)
    # mask = torch.ones(batch_size, max_len_output, dtype=torch.bool, device=device)

    # # Scatter features and coords to their positions
    # # Use advanced indexing for direct assignment (more efficient than scatter for this case)
    # seq[batch_indices, position_in_batch] = features
    # coords[batch_indices, position_in_batch] = spatial_coords
    # mask[batch_indices, position_in_batch] = False

    # Initialize output tensors with fixed length max_len_output
    # seq does not need requires_grad=True initially, because index_put will create a new tensor with history
    seq_base = torch.zeros(batch_size, max_len_output, channels, dtype=features.dtype, device=device)
    coords = torch.zeros(batch_size, max_len_output, 3, dtype=indices.dtype, device=device)
    mask = torch.ones(batch_size, max_len_output, dtype=torch.bool, device=device)

    # Scatter features out-of-place using index_put
    # This creates a new 'seq' tensor that tracks gradients correctly
    seq = seq_base.index_put((batch_indices, position_in_batch), features)

    # Coords and mask don't require gradients, so in-place assignment is perfectly fine
    coords[batch_indices, position_in_batch] = spatial_coords
    mask[batch_indices, position_in_batch] = False

    return {
        "seq": seq,
        "coords": coords,
        "mask": mask,
    }

def fps(spatial_coords: torch.Tensor, features: torch.Tensor, max_len: int):
    # print(f"fps")
    N_total = spatial_coords.shape[0]
    device = spatial_coords.device

    lengths = torch.full((1,), N_total, dtype=torch.int64, device=device)
    spatial_coords_unsqueezed = spatial_coords.unsqueeze(0)
    # debug
    # print(f"spatial_coords_unsqueezed shape: {spatial_coords_unsqueezed.shape}", f"lengths shape: {lengths.shape}", f"max_len: {max_len}")
    _, sampled_indices = sample_farthest_points(spatial_coords_unsqueezed, lengths, K=max_len)
    # print(f"sampled_indices shape: {sampled_indices.shape}, {sampled_indices.dtype}, {sampled_indices.device}, {sampled_indices.min()}, {sampled_indices.max()}")
    sampled_indices = sampled_indices.squeeze(0)

    features = features[sampled_indices]
    spatial_coords = spatial_coords[sampled_indices]

    return spatial_coords, features

def uniform_sampling(spatial_coords: torch.Tensor, features: torch.Tensor, max_len: int):
    # print(f"uniform_sampling")
    N_total = spatial_coords.shape[0]
    # print(f"N_total: {N_total}", f"max_len: {max_len}")
    # print(f"spatial_coords shape: {spatial_coords.shape}, {spatial_coords.dtype}, {spatial_coords.device}, {spatial_coords.min()}, {spatial_coords.max()}")
    # print(f"features shape: {features.shape}, {features.dtype}, {features.device}, {features.min()}, {features.max()}")
    sampled_indices = torch.linspace(0, N_total - 1, max_len, device=spatial_coords.device, dtype=torch.long)
    # print(f"sampled_indices shape: {sampled_indices.shape}, {sampled_indices.dtype}, {sampled_indices.device}, {sampled_indices.min()}, {sampled_indices.max()}")
    spatial_coords = spatial_coords[sampled_indices]
    features = features[sampled_indices]
    return spatial_coords, features

def vox_to_sequence_one_batch(sparse_tensor: spconv.SparseConvTensor, max_len_output: int = None):
    """Compute sequence from sparse point cloud.
    Fully vectorized implementation using scatter operations.
    Output is padded/truncated to max_len_output.

    Args:
        sparse_tensor: spconv.SparseConvTensor.
        max_len_output: Fixed output sequence length. Sequences longer than this will be truncated,
                        shorter sequences will be padded.

    Returns:
        Dict with the following keys:
            B = 1

            seq: [B, max_len_output, C] torch.FloatTensor.
            coords: [B, max_len_output, 3] torch.IntTensor. To be used with embeddings for Transformers.
            mask: [B, max_len_output] torch.BoolTensor. To be used with Transformers.
                  True indicates padding/truncated positions, False indicates valid positions.
    """
    batch_size = sparse_tensor.batch_size
    assert batch_size == 1, "vox_to_sequence_one_batch expects a batch size of exactly 1."

    indices = sparse_tensor.indices  # [N_total, 4] - (batch, x, y, z)
    features = sparse_tensor.features  # [N_total, C]

    device = features.device
    channels = features.shape[-1]

    # Get spatial coords
    spatial_coords = indices[:, 1:]  # [N_total, 3]
    N_total = indices.shape[0]
    # print(f"num_points: {N_total}; max_len_output: {max_len_output};")

    # Truncation via Farthest Point Sampling (Assuming fps returns tensors with grad intact)
    if max_len_output is not None and N_total > max_len_output:
        spatial_coords, features = fps(spatial_coords, features, max_len_output)

    num_points = features.shape[0]

    # # --- THE FIX: Out-of-place padding for sequence ---
    # pad_len = max_len_output - num_points

    # # F.pad format for 2D tensors is (pad_left, pad_right, pad_top, pad_bottom)
    # # We pad the points dimension (bottom) by pad_len, leaving channels (left/right) untouched
    # if pad_len > 0:
    #     seq_2d = F.pad(features, (0, 0, 0, pad_len))
    # else:
    #     seq_2d = features

    # # Expand to [1, max_len_output, C]
    # seq = seq_2d.unsqueeze(0)

    # # --- In-place assignments for non-grad tensors are perfectly fine ---
    # coords = torch.zeros(batch_size, max_len_output, 3, dtype=indices.dtype, device=device)
    # mask = torch.ones(batch_size, max_len_output, dtype=torch.bool, device=device)

    # coords[:, :num_points] = spatial_coords
    # mask[:, :num_points] = False

    coords = spatial_coords.unsqueeze(0)
    mask = torch.zeros(1, num_points, dtype=torch.bool, device=device)
    seq = features.unsqueeze(0)

    return {
        "seq": seq,
        "coords": coords,
        "mask": mask,
    }


def vox_to_sequence_batched(
    sparse_tensor: spconv.SparseConvTensor,
    max_len_output: int = None,
):
    """Convert one batched sparse tensor into padded transformer sequences.

    Sparse convolution remains fully batched. This small formatting loop only
    separates the final occupied features by scene because each scene has a
    different sparse length. Feature padding uses differentiable F.pad so
    gradients from every transformer sequence flow back into the shared spconv
    U-Net.
    """

    batch_size = int(sparse_tensor.batch_size)
    indices = sparse_tensor.indices
    features = sparse_tensor.features
    scene_features, scene_coords, scene_lengths = [], [], []
    for batch_index in range(batch_size):
        selected = indices[:, 0] == batch_index
        coords = indices[selected, 1:]
        feats = features[selected]
        if coords.numel() == 0:
            raise ValueError(f"sparse batch item {batch_index} has no occupied voxels")
        if max_len_output is not None and len(coords) > max_len_output:
            coords, feats = fps(coords, feats, max_len_output)
        scene_features.append(feats)
        scene_coords.append(coords)
        scene_lengths.append(int(len(coords)))

    padded_length = max(scene_lengths)
    padded_features, padded_coords = [], []
    for feats, coords, length in zip(
        scene_features, scene_coords, scene_lengths
    ):
        pad = padded_length - length
        padded_features.append(F.pad(feats, (0, 0, 0, pad)))
        padded_coords.append(F.pad(coords, (0, 0, 0, pad)))
    sequence = torch.stack(padded_features, dim=0)
    coords = torch.stack(padded_coords, dim=0)
    lengths = torch.as_tensor(
        scene_lengths, dtype=torch.long, device=features.device
    )
    padding_mask = (
        torch.arange(padded_length, device=features.device)[None]
        >= lengths[:, None]
    )
    return {
        "seq": sequence,
        "coords": coords,
        "mask": padding_mask,
    }

def fourier_encode_vector(vec, output_dim, sample_rate: int = 60):
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



def create_sparse_tensor(coords: torch.Tensor, feats: torch.Tensor, spatial_shape=None):
    """Create a spconv.SparseConvTensor from coords and features.

    Args:
        coords: [N, 4] torch.IntTensor where first column is batch index.
        feats: [N, C] torch.FloatTensor.
        spatial_shape: Optional list/tuple of spatial dimensions [D, H, W].

    Returns:
        spconv.SparseConvTensor.
    """
    if spatial_shape is None:
        # Compute spatial shape from coords
        spatial_shape = (coords[:, 1:].max(dim=0)[0] + 1).tolist()
    batch_size = int(coords[:, 0].max().item()) + 1
    return spconv.SparseConvTensor(
        features=feats,
        indices=coords.int(),
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )


class PointCloudUNet(nn.Module):
    def __init__(
        self,
        input_channels,
        d_model,
        conv_layers,
        resolution,
        max_len_input,
        max_len_output,
        sample_rate=60
    ):
        """Point Cloud Encoder.

        Args:
            input_channels: int.
            d_model: int.
            conv_layers: List[int].
            resolution: int.
            max_len_input: int.
            max_len_output: int.
        """

        super().__init__()

        self.sparse_resnet_unet = SparseUNet(
            dim_in=input_channels,
            dim_latent=d_model,
            layers=conv_layers,
        )

        downconvs = len(conv_layers) - 1
        res_reduction = 2**downconvs  # voxel resolution reduction
        self.reduced_grid_size = int(resolution / res_reduction)
        self.resolution = resolution
        self.d_model = d_model
        self.sample_rate = sample_rate

        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

        # print(f"sample_rate for fourier encoding: {self.sample_rate}")

    def process_outputs(self, output_sp_tensor: spconv.SparseConvTensor, max_len_output: int = None, grid_size: int = None):

        outputs = vox_to_sequence_batched(
            output_sp_tensor, max_len_output=max_len_output
        )
        context = outputs["seq"]
        context_mask = outputs["mask"]
        coords = outputs["coords"]
        coords_normalised = coords / (grid_size - 1)
        encoded_coords = fourier_encode_vector(coords_normalised, output_dim=context.shape[-1], sample_rate=self.sample_rate)

        return (
            context,
            context_mask,
            encoded_coords,
            coords_normalised,
            coords,
        )

    def forward(self, voxel_coords: torch.Tensor, voxel_feats: torch.Tensor, B: int):
        """Forward function.

        Args:
            voxel_coords: [N, 4] torch.IntTensor where first column is batch index.
            voxel_feats: [N, C] torch.FloatTensor.
            B: int.

        Returns: a Dict with the following keys:
            context: [B, max_len_output, d_model] torch.FloatTensor.
            context_mask: [B, max_len_output] torch.BoolTensor. True means ignore.
            coords: [B, max_len_output, d_model] torch.FloatTensor.
            recon_voxel_coords: [N_total, 4] torch.IntTensor.
            recon_voxel_feats: [N_total, C] torch.FloatTensor.
        """
        input_len = voxel_coords.shape[0]
        if B < 1:
            raise ValueError("PointCloudUNet batch size must be positive")
        if voxel_coords.ndim != 2 or voxel_coords.shape[-1] != 4:
            raise ValueError(
                f"expected sparse coordinates [N,4], got {tuple(voxel_coords.shape)}"
            )
        if int(voxel_coords[:, 0].min()) < 0 or int(voxel_coords[:, 0].max()) >= B:
            raise ValueError("sparse coordinate batch indices are outside [0,B)")
        if False and input_len > self.max_len_input: # skip uniform sampling for now
            voxel_coords, voxel_feats = uniform_sampling(voxel_coords, voxel_feats, self.max_len_input)

        # Build Sparse Tensor for SpConv
        point_cloud = spconv.SparseConvTensor(
            features=voxel_feats.detach(),
            indices=voxel_coords.detach(),  # [b, x, y, z]
            spatial_shape=[self.resolution] * 3,
            batch_size=B
        )

        # All trainable perception modules intentionally remain FP32. DINO is
        # the only component with an FP16 autocast region.
        encoded_context, recon_context = self.sparse_resnet_unet(point_cloud)

        (
            context,
            context_mask,
            context_coords,
            context_coords_normalised,
            context_voxel_coords,
        ) = self.process_outputs(
            encoded_context,
            max_len_output=self.max_len_output,
            grid_size=self.reduced_grid_size
        )
        (
            recon_context,
            recon_context_mask,
            recon_context_coords,
            recon_context_coords_normalised,
            recon_voxel_coords,
        ) = self.process_outputs(
            recon_context,
            max_len_output=None,
            grid_size=self.resolution
        )

        return {
            "context": context,
            "context_mask": context_mask,
            "context_coords": context_coords,
            "context_coords_normalised": context_coords_normalised,
            "context_voxel_coords": context_voxel_coords,
            "recon_context": recon_context,
            "recon_context_mask": recon_context_mask,
            "recon_context_coords": recon_context_coords,
            "recon_context_coords_normalised": recon_context_coords_normalised,
            "recon_voxel_coords": recon_voxel_coords,
        }
