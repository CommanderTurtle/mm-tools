import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points


def _valid_points_mask(counts, max_p):
    return torch.arange(max_p, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)


def _normalize_coords_per_object(batched_coords, valid_mask):
    coords_f = batched_coords.float()
    valid_3d = valid_mask.unsqueeze(-1)
    coord_min = coords_f.masked_fill(~valid_3d, float("inf")).min(dim=1).values
    coord_max = coords_f.masked_fill(~valid_3d, -float("inf")).max(dim=1).values
    span = (coord_max - coord_min).clamp_min(1e-6)
    coords01 = (coords_f - coord_min.unsqueeze(1)) / span.unsqueeze(1)
    return coords01.clamp(0.0, 1.0).masked_fill(~valid_3d, 0.0)


def _robust_fps_indices(batched_coords, counts, max_feats_len, grid_size=8):
    device = batched_coords.device
    num_rows, max_p, _ = batched_coords.shape
    valid_points = _valid_points_mask(counts, max_p)
    K_tensor = torch.clamp(counts, max=max_feats_len)

    if max_p == 0:
        return (
            torch.empty((num_rows, 0), dtype=torch.long, device=device),
            torch.empty((num_rows, 0), dtype=torch.bool, device=device),
        )

    coords01 = _normalize_coords_per_object(batched_coords, valid_points)
    bin_coords = (coords01 * grid_size).long().clamp(0, grid_size - 1)
    flat_bins = (
        bin_coords[..., 0] * grid_size * grid_size
        + bin_coords[..., 1] * grid_size
        + bin_coords[..., 2]
    ).masked_fill(~valid_points, 0)

    num_bins = grid_size ** 3
    bin_counts = torch.zeros((num_rows, num_bins), dtype=torch.float32, device=device)
    bin_counts.scatter_add_(1, flat_bins, valid_points.float())
    bin_counts_3d = bin_counts.view(num_rows, 1, grid_size, grid_size, grid_size)
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
    neighbor_counts = F.conv3d(bin_counts_3d, kernel, padding=1).view(num_rows, num_bins)
    point_support = neighbor_counts.gather(1, flat_bins)
    candidate_mask = valid_points & (point_support > 1.0)

    candidate_counts = candidate_mask.sum(dim=1)
    candidate_mask = torch.where(
        (candidate_counts >= K_tensor).unsqueeze(1),
        candidate_mask,
        valid_points,
    )
    candidate_counts = candidate_mask.sum(dim=1)
    max_candidates = int(candidate_counts.max().item())

    candidate_coords = torch.zeros((num_rows, max_candidates, 3), dtype=batched_coords.dtype, device=device)
    candidate_to_original = torch.zeros((num_rows, max_candidates), dtype=torch.long, device=device)
    for row in range(num_rows):
        row_indices = torch.nonzero(candidate_mask[row], as_tuple=False).flatten()
        row_count = row_indices.numel()
        if row_count == 0:
            continue
        candidate_coords[row, :row_count] = batched_coords[row, row_indices]
        candidate_to_original[row, :row_count] = row_indices

    K_tensor = torch.minimum(K_tensor, candidate_counts)
    _, candidate_indices = sample_farthest_points(
        candidate_coords.float(), lengths=candidate_counts, K=K_tensor, random_start_point=True
    )
    k_arange = torch.arange(candidate_indices.shape[1], device=device).unsqueeze(0)
    valid = (k_arange < K_tensor.unsqueeze(1)) & (candidate_indices != -1)

    safe_candidate_indices = candidate_indices.clone()
    safe_candidate_indices[~valid] = 0
    original_indices = torch.gather(candidate_to_original, 1, safe_candidate_indices)
    return original_indices, valid


def sample_object_feats(
    coords: torch.Tensor,
    feats: torch.Tensor,
    instance_ids: torch.Tensor,
    max_num_objects: int,
    max_feats_len: int,
    sample_method: str = "random",
):
    """
    Args:
        coords: (Total_Voxels, 3)
        feats: (Total_Voxels, C)
        instance_ids: (Total_Voxels,): long, 0 to max_num_objects-1
        max_num_objects: int, number of objects in the scene
        max_feats_len: int
        sample_method: str, one of ["random", "fps", "robust_fps", "hybrid"].
    Returns:
        feats_per_object: (max_num_objects, max_feats_len, C)
        coords_per_object: (max_num_objects, max_feats_len, 3)
        masks_per_object: (max_num_objects, max_feats_len), bool, True means valid
    """
    if sample_method not in ["random", "fps", "robust_fps", "hybrid"]:
        raise ValueError(f"Unknown sample_method: {sample_method}")

    device = coords.device
    C = feats.shape[-1]

    # 1. Filter out invalid IDs (-1 background, or out of bounds)
    valid_id_mask = (instance_ids >= 0) & (instance_ids < max_num_objects)
    v_coords = coords[valid_id_mask]
    v_feats = feats[valid_id_mask]
    v_ids = instance_ids[valid_id_mask]

    # Handle empty scene case
    if v_coords.shape[0] == 0:
        return (
            torch.zeros((max_num_objects, max_feats_len, C), device=device, dtype=feats.dtype),
            torch.zeros((max_num_objects, max_feats_len, 3), device=device, dtype=coords.dtype),
            torch.zeros((max_num_objects, max_feats_len), device=device, dtype=torch.bool)
        )

    # Pre-allocate Final Outputs
    out_feats = torch.zeros((max_num_objects, max_feats_len, C), dtype=feats.dtype, device=device)
    out_coords = torch.zeros((max_num_objects, max_feats_len, 3), dtype=coords.dtype, device=device)
    out_masks = torch.zeros((max_num_objects, max_feats_len), dtype=torch.bool, device=device)
    if max_feats_len <= 0:
        return out_feats, out_coords, out_masks

    # ---------------------------------------------------------------------
    # PATH A: PURE RANDOM SAMPLING
    # ---------------------------------------------------------------------
    if sample_method == "random":
        # Use one sort to both group by object id and randomize order within
        # each object. Since rand_scores are in [0, 1), ids remain grouped.
        rand_scores = torch.rand(v_ids.shape[0], device=device, dtype=torch.float32)
        sort_keys = v_ids.to(torch.float32) + rand_scores
        sort_indices = torch.argsort(sort_keys)
        sorted_ids = v_ids[sort_indices]
        sorted_coords = v_coords[sort_indices]
        sorted_feats = v_feats[sort_indices]

        _, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        run_starts = torch.zeros_like(counts)
        run_starts[1:] = torch.cumsum(counts[:-1], dim=0)
        repeat_starts = torch.repeat_interleave(run_starts, counts)
        intra_group_indices = torch.arange(sorted_ids.shape[0], device=device) - repeat_starts

        keep_mask = intra_group_indices < max_feats_len
        final_ids = sorted_ids[keep_mask]
        final_indices = intra_group_indices[keep_mask]

        out_feats[final_ids, final_indices] = sorted_feats[keep_mask]
        out_coords[final_ids, final_indices] = sorted_coords[keep_mask]
        out_masks[final_ids, final_indices] = True

        return out_feats, out_coords, out_masks

    # ---------------------------------------------------------------------
    # PATH B: FPS or HYBRID SAMPLING
    # ---------------------------------------------------------------------
    # No initial shuffle needed since we rely on FPS and scored random selection
    sorted_ids, sort_indices = torch.sort(v_ids, stable=True)
    sorted_coords = v_coords[sort_indices]
    sorted_feats = v_feats[sort_indices]

    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    run_starts = torch.zeros_like(counts)
    run_starts[1:] = torch.cumsum(counts[:-1], dim=0)
    repeat_starts = torch.repeat_interleave(run_starts, counts)
    intra_group_indices = torch.arange(sorted_ids.shape[0], device=device) - repeat_starts

    N_active = len(unique_ids)
    max_p = counts.max().item()

    # Natively batched coordinates ONLY for active objects: [N_active, max_p, 3].
    # Do not pad the feature tensor to [N_active, max_p, C]: for DINO features
    # (C=1024) that temporary can require tens of GiB on large scenes.  The
    # selected features are gathered from sorted_feats below with the same
    # indices, so this changes memory use without changing sampled outputs.
    batched_coords = torch.zeros((N_active, max_p, 3), device=device, dtype=coords.dtype)

    active_batch_idx = torch.repeat_interleave(torch.arange(N_active, device=device), counts)
    batched_coords[active_batch_idx, intra_group_indices] = sorted_coords

    if sample_method == "fps":
        K_tensor = torch.clamp(counts, max=max_feats_len)
        _, combined_indices = sample_farthest_points(
            batched_coords.float(), lengths=counts, K=K_tensor, random_start_point=True
        )
        k_arange = torch.arange(combined_indices.shape[1], device=device).unsqueeze(0)
        combined_valid = (k_arange < K_tensor.unsqueeze(1)) & (combined_indices != -1)

    elif sample_method == "robust_fps":
        combined_indices, combined_valid = _robust_fps_indices(
            batched_coords, counts, max_feats_len
        )

    else:  # sample_method == "hybrid"
        K_fps_max = max_feats_len // 2
        K_rand_max = max_feats_len - K_fps_max

        K_tensor_fps = torch.clamp(counts, max=K_fps_max)
        K_tensor_rand = torch.clamp(counts - K_tensor_fps, max=K_rand_max)

        # print(
        #     f"[FPS-debug] N_active={N_active} max_p={max_p} "
        #     f"K_fps_max={K_fps_max} max_feats_len={max_feats_len}\n"
        #     f"  counts.dtype={counts.dtype} counts.device={counts.device}\n"
        #     f"  counts(min/max/sum)={int(counts.min())}/{int(counts.max())}/{int(counts.sum())}\n"
        #     f"  K_tensor_fps(min/max)={int(K_tensor_fps.min())}/{int(K_tensor_fps.max())}\n"
        #     f"  batched_coords.shape={tuple(batched_coords.shape)} "
        #     f"finite={torch.isfinite(batched_coords).all().item()}",
        #     flush=True,
        # )

        # 1. FPS Pass
        _, sampled_indices_fps = sample_farthest_points(
            batched_coords.float(), lengths=counts, K=K_tensor_fps, random_start_point=True
        )
        k_fps_arange = torch.arange(sampled_indices_fps.shape[1], device=device).unsqueeze(0)
        valid_fps_mask = (k_fps_arange < K_tensor_fps.unsqueeze(1)) & (sampled_indices_fps != -1)

        # 2. Random Pass (via Top-K scoring)
        rand_scores = torch.rand((N_active, max_p), device=device)

        # Mask out padding points from being selected
        seq_arange = torch.arange(max_p, device=device).unsqueeze(0)
        valid_points_mask = seq_arange < counts.unsqueeze(1)
        rand_scores[~valid_points_mask] = -float('inf')

        # Mask out points already selected by FPS
        safe_fps_indices = sampled_indices_fps.clone()
        safe_fps_indices[~valid_fps_mask] = 0
        selected_mask = torch.zeros((N_active, max_p), dtype=torch.bool, device=device)
        selected_rows = torch.arange(N_active, device=device).unsqueeze(1).expand_as(safe_fps_indices)
        selected_mask[selected_rows[valid_fps_mask], safe_fps_indices[valid_fps_mask]] = True
        rand_scores[selected_mask] = -float('inf')

        # Select remaining random points
        actual_k_rand = K_tensor_rand.max().item()
        if actual_k_rand > 0:
            rand_values, sampled_indices_rand = torch.topk(rand_scores, actual_k_rand, dim=1)
            k_rand_arange = torch.arange(actual_k_rand, device=device).unsqueeze(0)
            valid_rand_mask = (k_rand_arange < K_tensor_rand.unsqueeze(1)) & torch.isfinite(rand_values)
        else:
            sampled_indices_rand = torch.empty((N_active, 0), dtype=torch.long, device=device)
            valid_rand_mask = torch.empty((N_active, 0), dtype=torch.bool, device=device)

        # Concatenate FPS and Random selections
        combined_indices = torch.cat([sampled_indices_fps, sampled_indices_rand], dim=1)
        combined_valid = torch.cat([valid_fps_mask, valid_rand_mask], dim=1)

    # ---------------------------------------------------------------------
    # GATHER AND TIGHTLY PACK (Shared by FPS and Hybrid)
    # ---------------------------------------------------------------------
    safe_combined_indices = combined_indices.clone()
    safe_combined_indices[~combined_valid] = 0

    sampled_coords_final = torch.gather(batched_coords, 1, safe_combined_indices.unsqueeze(-1).expand(-1, -1, 3))
    sampled_feat_indices = run_starts.unsqueeze(1) + safe_combined_indices
    sampled_feats_final = sorted_feats[sampled_feat_indices]

    # Calculate tightly packed indices (e.g., [True, False, True] -> [0, 0, 1])
    pack_idx = torch.cumsum(combined_valid, dim=1) - 1

    # Expand unique_ids to match the combined sequence length
    global_batch_idx = unique_ids.unsqueeze(1).expand(-1, combined_valid.shape[1])

    # Scatter tightly packed data directly into final buffers
    out_coords[global_batch_idx[combined_valid], pack_idx[combined_valid]] = sampled_coords_final[combined_valid]
    out_feats[global_batch_idx[combined_valid], pack_idx[combined_valid]] = sampled_feats_final[combined_valid]
    out_masks[global_batch_idx[combined_valid], pack_idx[combined_valid]] = True

    return out_feats, out_coords, out_masks
