"""
GPU-accelerated voxelization for point clouds.
Highly parallelized implementation using PyTorch operations.
"""
import torch
from utils.read_frames import GLOBAL_MEAN, GLOBAL_STD
import numpy as np
from torch import nn
# Pre-convert to contiguous arrays for numba
_GLOBAL_MEAN = torch.from_numpy(np.ascontiguousarray(GLOBAL_MEAN, dtype=np.float32))
_GLOBAL_STD = torch.from_numpy(np.ascontiguousarray(GLOBAL_STD, dtype=np.float32))

def voxel_aggregation() -> str:
    """How per-voxel features are pooled: "mean" (shipped) or "first".

    The shipped voxelizer averages every point that lands in a voxel. At a
    9.38 cm voxel a stride-4 cloud puts ~21 samples in each one, so the average
    is doing real smoothing; "first" keeps a single representative sample
    instead, which isolates that smoothing from the extra coverage. Env-gated
    because the model is constructed from a frozen checkpoint config.
    """

    import os

    mode = os.environ.get("FF_VOXEL_AGGREGATION", "mean").strip().lower()
    if mode not in {"mean", "first"}:
        raise ValueError(f"FF_VOXEL_AGGREGATION must be mean or first, got {mode!r}")
    return mode


class Voxelize(nn.Module):
    def __init__(self, voxel_size, resolution, apply_standardization):
        super().__init__()
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.apply_standardization = apply_standardization
        if self.apply_standardization:
            self.global_mean = _GLOBAL_MEAN
            self.global_std = _GLOBAL_STD
        else:
            self.global_mean = None
            self.global_std = None


    def forward(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        return_inverse_indices: bool = False,
    ):
        """
        GPU-accelerated voxelization of batched point clouds.
        """

        voxel_size = self.voxel_size
        resolution = self.resolution
        if self.apply_standardization:
            global_mean = self.global_mean.to(points.device)
            global_std = self.global_std.to(points.device)
        else:
            global_mean = None
            global_std = None

        device = points.device
        B, N, C = colors.shape

        # Precompute inverse voxel size for multiplication (faster than division)
        inv_voxel_size = 1.0 / voxel_size

        # Precompute resolution constants
        res_sq = resolution * resolution
        max_linear_idx = resolution ** 3

        # =========================================================================
        # Step 1: Compute voxel coordinates for all points in parallel
        # =========================================================================
        # Use multiply + int() directly - faster than floor().long() for positive values
        # Shape: (B, N, 3) -> integer voxel coordinates clamped to valid range
        voxel_coords = (points * inv_voxel_size).int().clamp(0, resolution - 1)

        # =========================================================================
        # Step 2: Compute global linear indices (batch-aware) directly
        # =========================================================================
        # Flatten coordinates for vectorized computation
        flat_coords = voxel_coords.view(-1, 3)  # (B*N, 3)
        flat_colors = colors.view(-1, C).float()  # (B*N, C)

        # Create batch indices: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
        batch_indices = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)

        # Global linear index = batch * max_idx + x * res^2 + y * res + z
        global_linear_indices = (
            batch_indices * max_linear_idx +
            flat_coords[:, 0].long() * res_sq +
            flat_coords[:, 1].long() * resolution +
            flat_coords[:, 2].long()
        )

        # =========================================================================
        # Step 3: Find unique voxels with counts in one operation
        # =========================================================================
        # return_counts=True eliminates need for separate scatter counting
        unique_global, inverse_indices, counts = torch.unique(
            global_linear_indices, return_inverse=True, return_counts=True, sorted=True
        )
        num_unique = unique_global.shape[0]

        # =========================================================================
        # Step 4: Aggregate colors using scatter_add (float32 is sufficient)
        # =========================================================================
        color_sums = torch.zeros(num_unique, C, device=device, dtype=torch.float32)
        expanded_inverse = inverse_indices.unsqueeze(1).expand(-1, C)
        color_sums.scatter_add_(0, expanded_inverse, flat_colors)

        # =========================================================================
        # Step 5: Compute per-voxel colors -- averaged, or one representative
        # =========================================================================
        if voxel_aggregation() == "first":
            # One real sample per voxel, never a blend: stable-sort the points by
            # their voxel, then take the lowest-index point of each run.
            order = torch.argsort(inverse_indices, stable=True)
            sorted_inverse = inverse_indices[order]
            boundary = torch.ones(
                inverse_indices.shape[0], dtype=torch.bool, device=device
            )
            boundary[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
            first_positions = torch.zeros(num_unique, dtype=torch.long, device=device)
            first_positions[sorted_inverse[boundary]] = order[boundary]
            voxel_colors_all = flat_colors[first_positions]
        else:
            voxel_colors_all = color_sums / counts.float().unsqueeze(1)

        # =========================================================================
        # Step 6: Standardize colors if mean/std provided
        # =========================================================================
        if global_mean is not None and global_std is not None:
            # Convert to tensor if needed, using contiguous memory
            if not isinstance(global_mean, torch.Tensor):
                global_mean = torch.tensor(global_mean, device=device, dtype=torch.float32)
            else:
                global_mean = global_mean.to(device=device, dtype=torch.float32)

            if not isinstance(global_std, torch.Tensor):
                global_std = torch.tensor(global_std, device=device, dtype=torch.float32)
            else:
                global_std = global_std.to(device=device, dtype=torch.float32)

            # Fused subtract and divide
            voxel_colors_all = (voxel_colors_all - global_mean) / global_std

        # =========================================================================
        # Step 7: Recover batch indices and 3D coordinates from global linear indices
        # =========================================================================
        batch_ids = unique_global // max_linear_idx
        local_linear = unique_global % max_linear_idx

        # Convert local linear indices back to 3D coordinates using integer division
        coord_x = local_linear // res_sq
        remainder = local_linear % res_sq
        coord_y = remainder // resolution
        coord_z = remainder % resolution

        # =========================================================================
        # Step 8: Return results in collate-compatible format
        # =========================================================================
        # Stack all coordinates at once (faster than cat with unsqueeze)
        voxel_coords_with_batch = torch.stack([
            batch_ids, coord_x, coord_y, coord_z
        ], dim=1).int()

        if return_inverse_indices:
            return voxel_coords_with_batch, voxel_colors_all, inverse_indices
        else:
            return voxel_coords_with_batch, voxel_colors_all


class InstanceVoxelize(nn.Module):
    def __init__(self, voxel_size, resolution, apply_standardization):
        super().__init__()
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.apply_standardization = apply_standardization

        # Determine global stats based on standardization flag
        if self.apply_standardization:
            # Assuming _GLOBAL_MEAN and _GLOBAL_STD are available in the scope
            # as implied by the reference code.
            self.global_mean = _GLOBAL_MEAN
            self.global_std = _GLOBAL_STD
        else:
            self.global_mean = None
            self.global_std = None

    def forward(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        instance_ids: torch.Tensor,
        return_inverse_indices: bool = False,
        valid_masks: torch.Tensor = None,
    ):
        """
        GPU-accelerated voxelization of batched point clouds with instance IDs.
        """
        voxel_size = self.voxel_size
        resolution = self.resolution

        # Prepare standardization tensors
        if self.apply_standardization:
            global_mean = self.global_mean.to(points.device)
            global_std = self.global_std.to(points.device)
        else:
            global_mean = None
            global_std = None

        device = points.device
        B, N, C = colors.shape

        # Precompute inverse voxel size for multiplication
        inv_voxel_size = 1.0 / voxel_size

        # Precompute resolution constants
        res_sq = resolution * resolution
        max_linear_idx = resolution ** 3

        # =========================================================================
        # Step 1: Compute voxel coordinates for all points in parallel
        # =========================================================================
        # Shape: (B, N, 3) -> integer voxel coordinates clamped to valid range
        voxel_coords = (points * inv_voxel_size).int().clamp(0, resolution - 1)

        # =========================================================================
        # Step 2: Compute global linear indices (batch-aware)
        # =========================================================================
        # Flatten input tensors
        flat_coords = voxel_coords.view(-1, 3)          # (B*N, 3)
        flat_colors = colors.view(-1, C).float()        # (B*N, C)
        flat_instance_ids = instance_ids.view(-1)       # (B*N)

        # Create batch indices: [0...0, 1...1, ..., B-1...B-1]
        batch_indices = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
        if valid_masks is not None:
            flat_valid_masks = valid_masks.reshape(-1).bool()
            if flat_valid_masks.numel() != flat_coords.shape[0]:
                raise ValueError(
                    "valid mask length does not match flattened points: "
                    f"{flat_valid_masks.numel()} vs {flat_coords.shape[0]}"
                )
            flat_coords = flat_coords[flat_valid_masks]
            flat_colors = flat_colors[flat_valid_masks]
            flat_instance_ids = flat_instance_ids[flat_valid_masks]
            batch_indices = batch_indices[flat_valid_masks]
            if flat_coords.shape[0] == 0:
                raise ValueError("voxelization received no valid points")

        # Global linear index = batch * max_idx + x * res^2 + y * res + z
        # This creates a unique 1D hash for every voxel in the batch
        global_linear_indices = (
            batch_indices * max_linear_idx +
            flat_coords[:, 0].long() * res_sq +
            flat_coords[:, 1].long() * resolution +
            flat_coords[:, 2].long()
        ) # [B*N]

        # =========================================================================
        # Step 3: Find unique voxels and the mapping from points to voxels
        # =========================================================================
        # unique_global: The sorted unique voxel indices
        # inverse_indices: Map from original points to the index in unique_global
        # counts: Number of points falling into each voxel
        unique_global, inverse_indices, counts = torch.unique(
            global_linear_indices, return_inverse=True, return_counts=True, sorted=True
        )
        num_unique = unique_global.shape[0]

        # =========================================================================
        # Step 4: Aggregate colors (Average)
        # =========================================================================
        # Sum colors falling into the same voxel
        color_sums = torch.zeros(num_unique, C, device=device, dtype=torch.float32)
        expanded_inverse = inverse_indices.unsqueeze(1).expand(-1, C) # [B*N, C]
        color_sums.scatter_add_(0, expanded_inverse, flat_colors)

        # Compute mean
        voxel_feats = color_sums / counts.float().unsqueeze(1)

        # =========================================================================
        # Step 5: Aggregate Instance IDs
        # =========================================================================
        # We need to assign an instance ID to each unique voxel.
        # Since 'scatter_' overwrites values when multiple indices point to the same location,
        # this effectively acts as a sampling mechanism (last point processed wins).
        # This is highly efficient and avoids explicit loops.
        voxel_instance_ids = torch.zeros(num_unique, dtype=torch.long, device=device)
        voxel_instance_ids.scatter_(0, inverse_indices, flat_instance_ids)

        # =========================================================================
        # Step 6: Standardize colors if needed
        # =========================================================================
        if global_mean is not None and global_std is not None:
            # Ensure tensor format and dtypes match
            if not isinstance(global_mean, torch.Tensor):
                global_mean = torch.tensor(global_mean, device=device, dtype=torch.float32)
            else:
                global_mean = global_mean.to(device=device, dtype=torch.float32)

            if not isinstance(global_std, torch.Tensor):
                global_std = torch.tensor(global_std, device=device, dtype=torch.float32)
            else:
                global_std = global_std.to(device=device, dtype=torch.float32)

            voxel_feats = (voxel_feats - global_mean) / global_std

        # =========================================================================
        # Step 7: Recover batch indices and 3D coordinates
        # =========================================================================
        batch_ids = unique_global // max_linear_idx
        local_linear = unique_global % max_linear_idx

        # Decode 1D indices back to 3D
        coord_x = local_linear // res_sq
        remainder = local_linear % res_sq
        coord_y = remainder // resolution
        coord_z = remainder % resolution

        # =========================================================================
        # Step 8: Return results
        # =========================================================================
        voxel_coords = torch.stack([
            batch_ids, coord_x, coord_y, coord_z
        ], dim=1).int()

        if return_inverse_indices:
            return voxel_coords, voxel_feats, voxel_instance_ids, inverse_indices
        else:
            return voxel_coords, voxel_feats, voxel_instance_ids


class InstanceNormalVoxelize(nn.Module):
    def __init__(self, voxel_size, resolution, apply_standardization):
        super().__init__()
        self.voxel_size = voxel_size
        self.resolution = resolution
        self.apply_standardization = apply_standardization

        # Determine global stats based on standardization flag
        if self.apply_standardization:
            # Assuming _GLOBAL_MEAN and _GLOBAL_STD are available in the scope
            # as implied by the reference code.
            self.global_mean = _GLOBAL_MEAN
            self.global_std = _GLOBAL_STD
        else:
            self.global_mean = None
            self.global_std = None

    def forward(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        instance_ids: torch.Tensor,
        normals: torch.Tensor,
        return_inverse_indices: bool = False,
    ):
        """
        GPU-accelerated voxelization of batched point clouds with instance IDs.
        """
        voxel_size = self.voxel_size
        resolution = self.resolution

        # Prepare standardization tensors
        if self.apply_standardization:
            global_mean = self.global_mean.to(points.device)
            global_std = self.global_std.to(points.device)
        else:
            global_mean = None
            global_std = None

        device = points.device
        B, N, C = colors.shape

        # Precompute inverse voxel size for multiplication
        inv_voxel_size = 1.0 / voxel_size

        # Precompute resolution constants
        res_sq = resolution * resolution
        max_linear_idx = resolution ** 3

        # =========================================================================
        # Step 1: Compute voxel coordinates for all points in parallel
        # =========================================================================
        # Shape: (B, N, 3) -> integer voxel coordinates clamped to valid range
        voxel_coords = (points * inv_voxel_size).int().clamp(0, resolution - 1)

        # =========================================================================
        # Step 2: Compute global linear indices (batch-aware)
        # =========================================================================
        # Flatten input tensors
        flat_coords = voxel_coords.view(-1, 3)          # (B*N, 3)
        flat_colors = colors.view(-1, C).float()        # (B*N, C)
        flat_instance_ids = instance_ids.view(-1)       # (B*N)
        flat_normals = normals.view(-1, 3).float()      # (B*N, 3)

        # Create batch indices: [0...0, 1...1, ..., B-1...B-1]
        batch_indices = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)

        # Global linear index = batch * max_idx + x * res^2 + y * res + z
        # This creates a unique 1D hash for every voxel in the batch
        global_linear_indices = (
            batch_indices * max_linear_idx +
            flat_coords[:, 0].long() * res_sq +
            flat_coords[:, 1].long() * resolution +
            flat_coords[:, 2].long()
        ) # [B*N]

        # =========================================================================
        # Step 3: Find unique voxels and the mapping from points to voxels
        # =========================================================================
        # unique_global: The sorted unique voxel indices
        # inverse_indices: Map from original points to the index in unique_global
        # counts: Number of points falling into each voxel
        unique_global, inverse_indices, counts = torch.unique(
            global_linear_indices, return_inverse=True, return_counts=True, sorted=True
        )
        num_unique = unique_global.shape[0]

        # =========================================================================
        # Step 4: Aggregate colors (Average)
        # =========================================================================
        # Sum colors falling into the same voxel
        color_sums = torch.zeros(num_unique, C, device=device, dtype=torch.float32)
        expanded_inverse = inverse_indices.unsqueeze(1).expand(-1, C) # [B*N, C]
        color_sums.scatter_add_(0, expanded_inverse, flat_colors)

        # Compute mean
        voxel_feats = color_sums / counts.float().unsqueeze(1)

        # =========================================================================
        # Step 5: Aggregate Instance IDs
        # =========================================================================
        # We need to assign an instance ID to each unique voxel.
        # Since 'scatter_' overwrites values when multiple indices point to the same location,
        # this effectively acts as a sampling mechanism (last point processed wins).
        # This is highly efficient and avoids explicit loops.
        voxel_instance_ids = torch.zeros(num_unique, dtype=torch.long, device=device)
        voxel_instance_ids.scatter_(0, inverse_indices, flat_instance_ids)

        voxel_normals = torch.zeros(num_unique, 3, device=device)
        expanded_inverse_normals = inverse_indices.unsqueeze(1).expand(-1, 3) # [B*N, 3]
        voxel_normals.scatter_(0, expanded_inverse_normals, flat_normals)

        # =========================================================================
        # Step 6: Standardize colors if needed
        # =========================================================================
        if global_mean is not None and global_std is not None:
            # Ensure tensor format and dtypes match
            if not isinstance(global_mean, torch.Tensor):
                global_mean = torch.tensor(global_mean, device=device, dtype=torch.float32)
            else:
                global_mean = global_mean.to(device=device, dtype=torch.float32)

            if not isinstance(global_std, torch.Tensor):
                global_std = torch.tensor(global_std, device=device, dtype=torch.float32)
            else:
                global_std = global_std.to(device=device, dtype=torch.float32)

            voxel_feats = (voxel_feats - global_mean) / global_std

        # =========================================================================
        # Step 7: Recover batch indices and 3D coordinates
        # =========================================================================
        batch_ids = unique_global // max_linear_idx
        local_linear = unique_global % max_linear_idx

        # Decode 1D indices back to 3D
        coord_x = local_linear // res_sq
        remainder = local_linear % res_sq
        coord_y = remainder // resolution
        coord_z = remainder % resolution

        # =========================================================================
        # Step 8: Return results
        # =========================================================================
        voxel_coords = torch.stack([
            batch_ids, coord_x, coord_y, coord_z
        ], dim=1).int()

        if return_inverse_indices:
            return voxel_coords, voxel_feats, voxel_instance_ids, voxel_normals, inverse_indices
        else:
            return voxel_coords, voxel_feats, voxel_instance_ids, voxel_normals
