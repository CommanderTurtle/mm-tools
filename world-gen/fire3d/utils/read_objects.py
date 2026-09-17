import numpy as np
from utils.transforms import transform_6d_from_transform_batch, _rotation_matrix_to_euler_sxyz_batch
from utils.discrete import discrete_transform_batch, continue_transform_batch
from utils.transforms import get_transform_matrix_batch


ss_feat_latents_stats = {
    "mean": np.array([
        0.08575359048634054, -0.33352827951900493, 0.5064794349490173, -0.08820589650647523, 0.2667975235528481, 0.08530554103358035, 0.29587603870710266, 0.38937252411594336]
    ),
    "std": np.array([
        0.7068617301850365, 0.8329831140293174, 0.8226509359339738, 0.9953821938345329, 0.8455223003672696, 0.9058781526775858, 0.8137887726677032, 0.7753252318750902]
    )
}
shape_feat_latent_stats = {
    "mean": np.array([
        -0.6639263211833489, 2.6967951372682646, 1.3185684374779638, -2.1939826327439302, -0.8664432463957266, -1.3173132235425002, 1.2021398071883562, 1.1973052742873478, -0.5754005182212506, -1.4831937152808503, 0.26452581174125456, 1.421314974817294, -0.4971240231950078, -0.12693748621158812, -3.8639453207941874, 0.5998774811009531]
    ),
    "std": np.array([
        3.8101859396724884, 4.0269099075884895, 3.9429651443738245, 3.831794349595539, 3.6694766218873616, 3.698207444139863, 3.706346343580214, 3.447708235837477, 3.8665936540297823, 3.952923053230185, 3.5748495081270097, 3.4748260032421663, 3.844190583570717, 3.4619456525222425, 3.8716267693407573, 3.8261404935630954]
    )
}
pbr_feat_latent_stats = {
    "mean": np.array([
        -2.6683799304840115, 1.3803240657068419, 0.11194998193100203, 0.9395922456293251, -1.7111049214727336, -0.9442168346218943, -1.2833437538319057, -1.2032354525352202, 0.19729180423113046, 0.48238977103666003, -1.0851221985809403, -1.949575776685697, -1.4617330151319974, -1.660627172086573, -0.7235096834179949, -1.6495963482382683],
    ),
    "std": np.array([
        3.3074655297301208, 3.8889619992391493, 3.4158773181902102, 2.992162836265447, 3.154662163088432, 2.5865420189293142, 2.9402682960800486, 3.6771381926763222, 3.278606226877671, 2.869220110863408, 2.909808481347111, 2.7424427603103845, 3.1517746746089674, 2.900364973784404, 2.6398456285509373, 2.893324199070787]
    )
}



def read_objects_to_tokens_with_instance_ids_occupancy(
    ss_latents, objects_transforms, augment_info, existing_indices, instance_ids,
):
    """
    Combined and optimized version of read_objects_info + to_object_tokens.
    Directly returns the object tokens dict without intermediate dict creation.

    Args:
        latents: dict of latent_id -> {'feats': array, 'coords': array}
        objects_transforms: dict of obj_id -> transform dict
        augment_info: dict with 'augment_transform' and 'norm_transform'
        existing_indices: set/list of existing object indices
        instance_ids: (num_points,) np.array, original instance ids, need to be mapped to the new instance ids

    Returns:
        object_tokens_dict: {
            "feats": [num_feats_i, dim_feats] len=N list of float32,
            "coords": [num_feats_i, 3] len=N list of int32,
            "translations": (N, 3) int32,
            "angles": (N, 3) int32,
            "scales": (N, 1) int32,
            "instance_ids": (num_points,) int32
        }
    """

    ss_feat_latent_mean = ss_feat_latents_stats["mean"]
    ss_feat_latent_std = ss_feat_latents_stats["std"]

    augment_transform = augment_info['augment_transform']
    norm_transform = augment_info['norm_transform']

    # Precompute combined transform
    combined_transform = np.asarray(norm_transform) @ np.asarray(augment_transform)

    # Convert to set for O(1) membership check
    existing_indices_set = set(existing_indices) if not isinstance(existing_indices, set) else existing_indices

    # Separate background from objects
    obj_ids = []
    bg_id = None
    for obj_id in objects_transforms:
        if not obj_id.startswith("layout_"):
            obj_ids.append(obj_id)
        else:
            bg_id = obj_id
    assert bg_id is not None, "Background ID is not found"

    obj_ids = sorted(obj_ids)
    # --- MODIFICATION START ---
    # We need to track the original index (1-based) alongside the object ID
    # to create the mapping later.
    obj_ids_existing = []
    # Initialize with [0] for the background (assuming bg is index 0 in original instance_ids)
    original_indices_filtered = [0]

    for i, obj_id in enumerate(obj_ids):
        # Original logic: checks if i+1 is in existing_indices
        if i + 1 in existing_indices_set:
            obj_ids_existing.append(obj_id)
            original_indices_filtered.append(i + 1)
    # --- MODIFICATION END ---

    # Build list: [bg_id] + existing objects
    obj_ids_to_process = [bg_id] + obj_ids_existing
    N = len(obj_ids_to_process)

    # Gather all transform data into arrays for batched processing
    scales_in = np.empty(N, dtype=np.float64)
    angles_in = np.empty((N, 3), dtype=np.float64)
    trans_in = np.empty((N, 3), dtype=np.float64)
    latent_ids = []

    for i, obj_id in enumerate(obj_ids_to_process):
        transform_dict = objects_transforms[obj_id]
        scales_in[i] = transform_dict["scale"]
        angles_in[i] = transform_dict["angles"]
        trans_in[i] = transform_dict["trans"]
        latent_ids.append(transform_dict["latent"])

    # Batched transform and discretization
    new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
        scales_in, angles_in, trans_in, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)

    # Sort non-background objects by d_trans (z, y, x priority)
    # Background stays at index 0, sort indices 1:N
    if N > 1:
        # Get sort order for objects (excluding background at index 0)
        obj_trans = d_trans[1:]  # (N-1, 3)
        # lexsort sorts by last key first, so order is (x, y, z) to get priority z, y, x
        sort_keys = (obj_trans[:, 0], obj_trans[:, 1], obj_trans[:, 2])
        sort_order = np.lexsort(sort_keys)  # indices into obj_trans

        # Build final order: [0] + [1 + sort_order]
        final_order = np.concatenate([[0], 1 + sort_order])
    else:
        final_order = np.array([0])

    # Reorder discretized transforms
    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]

    new_scales_r, new_angles_r, new_trans_r = continue_transform_batch(d_scales, d_angles, d_trans)
    new_transforms_r = get_transform_matrix_batch(new_scales_r, new_angles_r, new_trans_r) # [N, 4, 4] from [-0.5, 0.5]^3 to world coordinates
    new_transforms_inv = np.linalg.inv(new_transforms_r) # [N, 4, 4] from world coordinates to [-0.5, 0.5]^3

    # Reorder latent_ids to match
    latent_ids = [latent_ids[i] for i in final_order]

    # --- MODIFICATION START: Instance ID Mapping ---
    # 1. Reorder the filtered original indices using the same spatial sort (final_order)
    # This gives us: ordered_original_indices[new_index] = original_index
    ordered_original_indices = [original_indices_filtered[i] for i in final_order]

    # 2. Create a dense mapping array
    # Determine size for mapping array (max id + 1)
    max_original_id = instance_ids.max() if instance_ids.size > 0 else len(obj_ids)
    # Initialize with 0.
    # Note: Any original ID not present in existing_indices will map to 0 (background).
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)

    # 3. Populate map: id_map[original_index] = new_index
    for new_idx, old_idx in enumerate(ordered_original_indices):
        if old_idx < id_map.shape[0]:
            id_map[old_idx] = new_idx

    # 4. Vectorized lookup to generate new instance ids
    out_instance_ids = id_map[instance_ids]
    # --- MODIFICATION END ---

    # Get unique latent_ids (preserving first occurrence order) and build mapping
    unique_latent_ids = list(dict.fromkeys(latent_ids))
    latent_id_to_idx = {lid: idx for idx, lid in enumerate(unique_latent_ids)}

    # Gather all feats and coords for unique latents
    # feats_list = [latents[lid]['feats'] for lid in unique_latent_ids]
    # coords_list = [latents[lid]['coords'] for lid in unique_latent_ids]
    feats_list = [ss_latents[lid]['ss_latent'].reshape(8, 2*2*2).transpose(1, 0) for lid in unique_latent_ids]

    # Batch normalize: concatenate, normalize, then split back
    all_feats = np.stack(feats_list, axis=0) # [N, 2*2*2, 8]
    all_feats_normalized = (all_feats - ss_feat_latent_mean) / ss_feat_latent_std
    # debug: check the mean and std of the normalized feats
    # print(f"Normalized ss feats mean: {all_feats_normalized.reshape(-1, 8).mean(axis=0).tolist()}, std: {all_feats_normalized.reshape(-1, 8).std(axis=0).tolist()}")

    # # Batch sparse to dense
    # dense_feats_batch, dense_occ_batch = sparse_to_dense_batch(
    #     feats_list_normalized, coords_list, resolution=8
    # )

    # # Build output indices and assign using vectorized indexing
    output_indices = np.array([latent_id_to_idx[lid] for lid in latent_ids])
    # out_latents = dense_feats_batch[output_indices].reshape(N, -1).astype(np.float32, copy=False)
    # out_occupancies = dense_occ_batch[output_indices].reshape(N, -1).astype(np.float32, copy=False)

    # feats_list_normalized = [feats_list_normalized[idx] for idx in output_indices]
    all_feats_normalized = all_feats_normalized[output_indices].reshape(N, -1).astype(np.float32, copy=False)
    # print(f"all_feats_normalized.shape: {all_feats_normalized.shape}")

    # Build output dict with correct dtypes
    return {
        "feats": all_feats_normalized,
        "transforms": new_transforms_inv.astype(np.float32),
        "instance_ids": out_instance_ids.astype(np.int32)
    }



def read_objects_to_tokens_with_instance_ids_feats(
    latents, objects_transforms, augment_info, existing_indices, instance_ids,
):
    """
    Combined and optimized version of read_objects_info + to_object_tokens.
    Directly returns the object tokens dict without intermediate dict creation.

    Args:
        latents: dict of latent_id -> {'feats': array, 'coords': array}
        objects_transforms: dict of obj_id -> transform dict
        augment_info: dict with 'augment_transform' and 'norm_transform'
        existing_indices: set/list of existing object indices
        instance_ids: (num_points,) np.array, original instance ids, need to be mapped to the new instance ids

    Returns:
        object_tokens_dict: {
            "feats": list of [N, C] len=N float32,
            "coords": list of [N, 3] len=N int32,
            "translations": (N, 3) int32,
            "angles": (N, 3) int32,
            "scales": (N, 1) int32,
            "instance_ids": (num_points,) int32
        }
    """
    feat_latent_mean = shape_feat_latent_stats["mean"]
    feat_latent_std = shape_feat_latent_stats["std"]

    augment_transform = augment_info['augment_transform']
    norm_transform = augment_info['norm_transform']

    # Precompute combined transform
    combined_transform = np.asarray(norm_transform) @ np.asarray(augment_transform)

    # Convert to set for O(1) membership check
    existing_indices_set = set(existing_indices) if not isinstance(existing_indices, set) else existing_indices

    # Separate background from objects
    obj_ids = []
    bg_id = None
    for obj_id in objects_transforms:
        if not obj_id.startswith("layout_"):
            obj_ids.append(obj_id)
        else:
            bg_id = obj_id
    assert bg_id is not None, "Background ID is not found"

    obj_ids = sorted(obj_ids)
    # --- MODIFICATION START ---
    # We need to track the original index (1-based) alongside the object ID
    # to create the mapping later.
    obj_ids_existing = []
    # Initialize with [0] for the background (assuming bg is index 0 in original instance_ids)
    original_indices_filtered = [0]

    for i, obj_id in enumerate(obj_ids):
        # Original logic: checks if i+1 is in existing_indices
        if i + 1 in existing_indices_set:
            obj_ids_existing.append(obj_id)
            original_indices_filtered.append(i + 1)
    # --- MODIFICATION END ---

    # Build list: [bg_id] + existing objects
    obj_ids_to_process = [bg_id] + obj_ids_existing
    N = len(obj_ids_to_process)

    # Gather all transform data into arrays for batched processing
    scales_in = np.empty(N, dtype=np.float64)
    angles_in = np.empty((N, 3), dtype=np.float64)
    trans_in = np.empty((N, 3), dtype=np.float64)
    latent_ids = []

    for i, obj_id in enumerate(obj_ids_to_process):
        transform_dict = objects_transforms[obj_id]
        scales_in[i] = transform_dict["scale"]
        angles_in[i] = transform_dict["angles"]
        trans_in[i] = transform_dict["trans"]
        latent_ids.append(transform_dict["latent"])

    # Batched transform and discretization
    new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
        scales_in, angles_in, trans_in, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)

    # Sort non-background objects by d_trans (z, y, x priority)
    # Background stays at index 0, sort indices 1:N
    if N > 1:
        # Get sort order for objects (excluding background at index 0)
        obj_trans = d_trans[1:]  # (N-1, 3)
        # lexsort sorts by last key first, so order is (x, y, z) to get priority z, y, x
        sort_keys = (obj_trans[:, 0], obj_trans[:, 1], obj_trans[:, 2])
        sort_order = np.lexsort(sort_keys)  # indices into obj_trans

        # Build final order: [0] + [1 + sort_order]
        final_order = np.concatenate([[0], 1 + sort_order])
    else:
        final_order = np.array([0])

    # Reorder discretized transforms
    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]

    new_scales_r, new_angles_r, new_trans_r = continue_transform_batch(d_scales, d_angles, d_trans)
    new_transforms_r = get_transform_matrix_batch(new_scales_r, new_angles_r, new_trans_r) # [N, 4, 4] from [-0.5, 0.5]^3 to world coordinates
    new_transforms_inv = np.linalg.inv(new_transforms_r) # [N, 4, 4] from world coordinates to [-0.5, 0.5]^3

    # Reorder latent_ids to match
    latent_ids = [latent_ids[i] for i in final_order]

    # --- MODIFICATION START: Instance ID Mapping ---
    # 1. Reorder the filtered original indices using the same spatial sort (final_order)
    # This gives us: ordered_original_indices[new_index] = original_index
    ordered_original_indices = [original_indices_filtered[i] for i in final_order]

    # 2. Create a dense mapping array
    # Determine size for mapping array (max id + 1)
    max_original_id = instance_ids.max() if instance_ids.size > 0 else len(obj_ids)
    # Initialize with 0.
    # Note: Any original ID not present in existing_indices will map to 0 (background).
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)

    # 3. Populate map: id_map[original_index] = new_index
    for new_idx, old_idx in enumerate(ordered_original_indices):
        if old_idx < id_map.shape[0]:
            id_map[old_idx] = new_idx

    # 4. Vectorized lookup to generate new instance ids
    out_instance_ids = id_map[instance_ids]
    # --- MODIFICATION END ---

    # Get unique latent_ids (preserving first occurrence order) and build mapping
    unique_latent_ids = list(dict.fromkeys(latent_ids))
    latent_id_to_idx = {lid: idx for idx, lid in enumerate(unique_latent_ids)}

    # Gather all feats and coords for unique latents
    feats_list = [latents[lid]['feats'] for lid in unique_latent_ids]
    coords_list = [latents[lid]['coords'] for lid in unique_latent_ids]

    # Batch normalize: concatenate, normalize, then split back
    split_sizes = [f.shape[0] for f in feats_list]
    all_feats = np.concatenate(feats_list, axis=0)
    all_feats_normalized = (all_feats - feat_latent_mean) / feat_latent_std
    # debug: check the mean and std of the normalized feats
    # print(f"Normalized feats mean: {all_feats_normalized.reshape(-1, 16).mean(axis=0).tolist()}, std: {all_feats_normalized.reshape(-1, 16).std(axis=0).tolist()}")
    split_indices = np.cumsum(split_sizes[:-1])
    feats_list_normalized = np.split(all_feats_normalized, split_indices)

    # # Batch sparse to dense
    # dense_feats_batch, dense_occ_batch = sparse_to_dense_batch(
    #     feats_list_normalized, coords_list, resolution=8
    # )

    # Build output indices and assign using vectorized indexing
    output_indices = np.array([latent_id_to_idx[lid] for lid in latent_ids])
    # out_latents = dense_feats_batch[output_indices].reshape(N, -1).astype(np.float32, copy=False)
    # out_occupancies = dense_occ_batch[output_indices].reshape(N, -1).astype(np.float32, copy=False)
    feats_list_normalized = [feats_list_normalized[i] for i in output_indices]
    coords_list = [coords_list[i] for i in output_indices]

    # Build output dict with correct dtypes
    return {
        "feats": feats_list_normalized,
        "coords": coords_list,
        "transforms": new_transforms_inv.astype(np.float32),
        "instance_ids": out_instance_ids.astype(np.int32)
    }



def _safe_latent_stem(value):
    from urllib.parse import quote
    cleaned = str(value).replace("/", "_").replace("|", "_").replace(" ", "_")
    return quote(cleaned, safe="")


def _load_raw_shape_latent_from_dir(latent_root, local_name, latent_rotation=None):
    from pathlib import Path
    root = Path(latent_root)
    safe_name = _safe_latent_stem(local_name)
    if latent_rotation is None:
        candidates = [
            root / f"{local_name}.npz",
            root / f"{safe_name}.npz",
        ]
    else:
        rotation = str(latent_rotation).strip().lower().replace("rot", "")
        candidates = [
            root / f"{local_name}__rot{rotation}.npz",
            root / f"{safe_name}__rot{rotation}.npz",
        ]
    for path in candidates:
        if path.exists():
            with np.load(path) as data:
                feats = data["feats"].astype(np.float32, copy=False)
                coords = data["coords"].astype(np.int32, copy=False)
            if feats.ndim != 2 or feats.shape[1] != 32:
                raise ValueError(f"Expected raw TRELLIS2 shape latent with 32 channels, got {feats.shape} at {path}")
            return {"feats": feats, "coords": coords}
    raise FileNotFoundError(f"Missing raw TRELLIS2 shape latent for {local_name!r} under {latent_root}")


def _normalize_latent_rotation_degrees(latent_rotation):
    if latent_rotation is None:
        return None
    label = str(latent_rotation).strip().lower().replace("rot", "")
    if label == "":
        return None
    degrees = int(label) % 360
    if degrees not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported latent rotation {latent_rotation!r}; expected 000/090/180/270")
    return degrees


def _apply_latent_rotation_to_pose(scales, angles, trans, latent_rotation):
    degrees = _normalize_latent_rotation_degrees(latent_rotation)
    if degrees is None or degrees == 0:
        return scales, angles, trans

    local_to_world = get_transform_matrix_batch(scales, angles, trans)
    theta = -np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    canonical_to_augmented = np.eye(4, dtype=np.float64)
    canonical_to_augmented[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adjusted = np.einsum("nij,jk->nik", local_to_world, canonical_to_augmented)
    linear = adjusted[:, :3, :3]
    adjusted_scales = np.linalg.norm(linear, axis=1).mean(axis=1)
    rotations = linear / np.maximum(adjusted_scales[:, None, None], 1e-8)
    adjusted_angles = _rotation_matrix_to_euler_sxyz_batch(rotations)
    adjusted_trans = adjusted[:, :3, 3]
    return adjusted_scales, adjusted_angles, adjusted_trans


def _split_bg_and_object_ids(objects_transforms):
    keys = list(objects_transforms.keys())
    bg_candidates = [key for key in keys if str(key).startswith("layout_")]
    if not bg_candidates and "0000" in objects_transforms:
        bg_candidates = ["0000"]
    if not bg_candidates:
        # Keep this as a fallback for datasets that preserve bg as the first transform entry.
        bg_candidates = [keys[0]]
    bg_id = bg_candidates[0]
    obj_ids = sorted([key for key in keys if key != bg_id])
    return bg_id, obj_ids


def read_objects_to_tokens_with_instance_ids_raw_shape(
    latent_root, objects_transforms, augment_info, existing_indices, instance_ids, latent_rotation=None,
):
    """Return raw 32-channel TRELLIS2 shape slats for scene objects.

    Unlike ``read_objects_to_tokens_with_instance_ids_feats``, this function
    intentionally does not normalize features.  The online Shape VAE X2 model
    applies the TRELLIS2-shape normalization immediately before the frozen x2
    encoder and applies the 512-channel x2 normalization after encoding.
    """

    augment_transform = augment_info['augment_transform']
    norm_transform = augment_info['norm_transform']
    combined_transform = np.asarray(norm_transform) @ np.asarray(augment_transform)
    existing_indices_set = set(existing_indices) if not isinstance(existing_indices, set) else existing_indices

    bg_id, obj_ids = _split_bg_and_object_ids(objects_transforms)
    obj_ids_existing = []
    original_indices_filtered = [0]
    for i, obj_id in enumerate(obj_ids):
        if i + 1 in existing_indices_set:
            obj_ids_existing.append(obj_id)
            original_indices_filtered.append(i + 1)

    obj_ids_to_process = [bg_id] + obj_ids_existing
    N = len(obj_ids_to_process)

    scales_in = np.empty(N, dtype=np.float64)
    angles_in = np.empty((N, 3), dtype=np.float64)
    trans_in = np.empty((N, 3), dtype=np.float64)
    local_names = []
    for i, obj_id in enumerate(obj_ids_to_process):
        transform_dict = objects_transforms[obj_id]
        scales_in[i] = transform_dict["scale"]
        angles_in[i] = transform_dict["angles"]
        trans_in[i] = transform_dict["trans"]
        local_names.append(obj_id)

    scales_in, angles_in, trans_in = _apply_latent_rotation_to_pose(
        scales_in, angles_in, trans_in, latent_rotation
    )

    new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
        scales_in, angles_in, trans_in, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)

    if N > 1:
        obj_trans = d_trans[1:]
        sort_keys = (obj_trans[:, 0], obj_trans[:, 1], obj_trans[:, 2])
        sort_order = np.lexsort(sort_keys)
        final_order = np.concatenate([[0], 1 + sort_order])
    else:
        final_order = np.array([0])

    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]
    new_scales_r, new_angles_r, new_trans_r = continue_transform_batch(d_scales, d_angles, d_trans)
    new_transforms_r = get_transform_matrix_batch(new_scales_r, new_angles_r, new_trans_r)
    new_transforms_inv = np.linalg.inv(new_transforms_r)

    local_names = [local_names[i] for i in final_order]
    ordered_original_indices = [original_indices_filtered[i] for i in final_order]
    max_original_id = instance_ids.max() if instance_ids.size > 0 else len(obj_ids)
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(ordered_original_indices):
        if old_idx < id_map.shape[0]:
            id_map[old_idx] = new_idx
    out_instance_ids = id_map[instance_ids]

    unique_local_names = list(dict.fromkeys(local_names))
    latent_by_name = {
        name: _load_raw_shape_latent_from_dir(latent_root, name, latent_rotation=latent_rotation)
        for name in unique_local_names
    }
    feats_list = [latent_by_name[name]["feats"] for name in local_names]
    coords_list = [latent_by_name[name]["coords"] for name in local_names]

    return {
        "feats": feats_list,
        "coords": coords_list,
        "transforms": new_transforms_inv.astype(np.float32),
        "instance_ids": out_instance_ids.astype(np.int32),
        # Preserve the authoritative filtered/spatially-sorted identity order.
        # Evaluation code must resolve frozen object IDs through this list
        # instead of assuming transform-file or stale manifest positions.
        "local_names": local_names,
    }


def _load_shape_x2_latent_from_dir(
    latent_root, local_name, latent_rotation=None, latent_channels=512
):
    from pathlib import Path
    root = Path(latent_root)
    safe_name = _safe_latent_stem(local_name)
    if latent_rotation is None:
        candidates = [
            root / f"{local_name}.npz",
            root / f"{safe_name}.npz",
        ]
    else:
        rotation = str(latent_rotation).strip().lower().replace("rot", "")
        candidates = [
            root / f"{local_name}__rot{rotation}.npz",
            root / f"{safe_name}__rot{rotation}.npz",
        ]
    for path in candidates:
        if path.exists():
            with np.load(path) as data:
                feats = data["feats"].astype(np.float32, copy=False)
                coords = data["coords"].astype(np.int32, copy=False)
            if feats.ndim != 2 or feats.shape[1] != int(latent_channels):
                raise ValueError(
                    f"Expected Shape VAE X2 latent with {int(latent_channels)} channels, "
                    f"got {feats.shape} at {path}"
                )
            return {"feats": feats, "coords": coords}
    raise FileNotFoundError(f"Missing Shape VAE X2 latent for {local_name!r} under {latent_root}")


def read_objects_to_tokens_with_instance_ids_shape_x2(
    latent_root,
    objects_transforms,
    augment_info,
    existing_indices,
    instance_ids,
    latent_rotation=None,
    latent_channels=512,
):
    """Return raw Shape VAE X2 slats for scene objects."""

    augment_transform = augment_info['augment_transform']
    norm_transform = augment_info['norm_transform']
    combined_transform = np.asarray(norm_transform) @ np.asarray(augment_transform)
    existing_indices_set = set(existing_indices) if not isinstance(existing_indices, set) else existing_indices

    bg_id, obj_ids = _split_bg_and_object_ids(objects_transforms)
    obj_ids_existing = []
    original_indices_filtered = [0]
    for i, obj_id in enumerate(obj_ids):
        if i + 1 in existing_indices_set:
            obj_ids_existing.append(obj_id)
            original_indices_filtered.append(i + 1)

    obj_ids_to_process = [bg_id] + obj_ids_existing
    N = len(obj_ids_to_process)

    scales_in = np.empty(N, dtype=np.float64)
    angles_in = np.empty((N, 3), dtype=np.float64)
    trans_in = np.empty((N, 3), dtype=np.float64)
    local_names = []
    for i, obj_id in enumerate(obj_ids_to_process):
        transform_dict = objects_transforms[obj_id]
        scales_in[i] = transform_dict["scale"]
        angles_in[i] = transform_dict["angles"]
        trans_in[i] = transform_dict["trans"]
        local_names.append(obj_id)

    scales_in, angles_in, trans_in = _apply_latent_rotation_to_pose(
        scales_in, angles_in, trans_in, latent_rotation
    )

    new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
        scales_in, angles_in, trans_in, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)

    if N > 1:
        obj_trans = d_trans[1:]
        sort_keys = (obj_trans[:, 0], obj_trans[:, 1], obj_trans[:, 2])
        sort_order = np.lexsort(sort_keys)
        final_order = np.concatenate([[0], 1 + sort_order])
    else:
        final_order = np.array([0])

    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]
    new_scales_r, new_angles_r, new_trans_r = continue_transform_batch(d_scales, d_angles, d_trans)
    new_transforms_r = get_transform_matrix_batch(new_scales_r, new_angles_r, new_trans_r)
    new_transforms_inv = np.linalg.inv(new_transforms_r)

    local_names = [local_names[i] for i in final_order]
    ordered_original_indices = [original_indices_filtered[i] for i in final_order]
    max_original_id = instance_ids.max() if instance_ids.size > 0 else len(obj_ids)
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(ordered_original_indices):
        if old_idx < id_map.shape[0]:
            id_map[old_idx] = new_idx
    out_instance_ids = id_map[instance_ids]

    unique_local_names = list(dict.fromkeys(local_names))
    latent_by_name = {
        name: _load_shape_x2_latent_from_dir(
            latent_root,
            name,
            latent_rotation=latent_rotation,
            latent_channels=latent_channels,
        )
        for name in unique_local_names
    }
    feats_list = [latent_by_name[name]["feats"] for name in local_names]
    coords_list = [latent_by_name[name]["coords"] for name in local_names]

    return {
        "feats": feats_list,
        "coords": coords_list,
        "transforms": new_transforms_inv.astype(np.float32),
        "instance_ids": out_instance_ids.astype(np.int32),
        "local_names": local_names,
    }


def read_objects_to_tokens_with_instance_ids_feats_pbr(
    latents, pbr_latents, objects_transforms, augment_info, existing_indices, instance_ids,
):
    """
    Combined and optimized version of read_objects_info + to_object_tokens.
    Directly returns the object tokens dict without intermediate dict creation.

    Args:
        latents: dict of latent_id -> {'feats': array, 'coords': array}
        pbr_latents: dict of latent_id -> {'feats': array, 'coords': array}
        objects_transforms: dict of obj_id -> transform dict
        augment_info: dict with 'augment_transform' and 'norm_transform'
        existing_indices: set/list of existing object indices
        instance_ids: (num_points,) np.array, original instance ids, need to be mapped to the new instance ids

    Returns:
        object_tokens_dict: {
            "feats": list of [N, C] len=N float32,
            "pbr_feats": list of [N, C] len=N float32,
            "coords": list of [N, 3] len=N int32,
            "translations": (N, 3) int32,
            "angles": (N, 3) int32,
            "scales": (N, 1) int32,
            "instance_ids": (num_points,) int32
        }
    """
    feat_latent_mean = shape_feat_latent_stats["mean"]
    feat_latent_std = shape_feat_latent_stats["std"]

    pbr_feat_latent_mean = pbr_feat_latent_stats["mean"]
    pbr_feat_latent_std = pbr_feat_latent_stats["std"]

    augment_transform = augment_info['augment_transform']
    norm_transform = augment_info['norm_transform']

    # Precompute combined transform
    combined_transform = np.asarray(norm_transform) @ np.asarray(augment_transform)

    # Convert to set for O(1) membership check
    existing_indices_set = set(existing_indices) if not isinstance(existing_indices, set) else existing_indices

    # Separate background from objects
    obj_ids = []
    bg_id = None
    for obj_id in objects_transforms:
        if not obj_id.startswith("layout_"):
            obj_ids.append(obj_id)
        else:
            bg_id = obj_id
    assert bg_id is not None, "Background ID is not found"

    obj_ids = sorted(obj_ids)
    # --- MODIFICATION START ---
    # We need to track the original index (1-based) alongside the object ID
    # to create the mapping later.
    obj_ids_existing = []
    # Initialize with [0] for the background (assuming bg is index 0 in original instance_ids)
    original_indices_filtered = [0]

    for i, obj_id in enumerate(obj_ids):
        # Original logic: checks if i+1 is in existing_indices
        if i + 1 in existing_indices_set:
            obj_ids_existing.append(obj_id)
            original_indices_filtered.append(i + 1)
    # --- MODIFICATION END ---

    # Build list: [bg_id] + existing objects
    obj_ids_to_process = [bg_id] + obj_ids_existing
    N = len(obj_ids_to_process)

    # Gather all transform data into arrays for batched processing
    scales_in = np.empty(N, dtype=np.float64)
    angles_in = np.empty((N, 3), dtype=np.float64)
    trans_in = np.empty((N, 3), dtype=np.float64)
    latent_ids = []

    for i, obj_id in enumerate(obj_ids_to_process):
        transform_dict = objects_transforms[obj_id]
        scales_in[i] = transform_dict["scale"]
        angles_in[i] = transform_dict["angles"]
        trans_in[i] = transform_dict["trans"]
        latent_ids.append(transform_dict["latent"])

    # Batched transform and discretization
    new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
        scales_in, angles_in, trans_in, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)

    # Sort non-background objects by d_trans (z, y, x priority)
    # Background stays at index 0, sort indices 1:N
    if N > 1:
        # Get sort order for objects (excluding background at index 0)
        obj_trans = d_trans[1:]  # (N-1, 3)
        # lexsort sorts by last key first, so order is (x, y, z) to get priority z, y, x
        sort_keys = (obj_trans[:, 0], obj_trans[:, 1], obj_trans[:, 2])
        sort_order = np.lexsort(sort_keys)  # indices into obj_trans

        # Build final order: [0] + [1 + sort_order]
        final_order = np.concatenate([[0], 1 + sort_order])
    else:
        final_order = np.array([0])

    # Reorder discretized transforms
    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]

    new_scales_r, new_angles_r, new_trans_r = continue_transform_batch(d_scales, d_angles, d_trans)
    new_transforms_r = get_transform_matrix_batch(new_scales_r, new_angles_r, new_trans_r) # [N, 4, 4] from [-0.5, 0.5]^3 to world coordinates
    new_transforms_inv = np.linalg.inv(new_transforms_r) # [N, 4, 4] from world coordinates to [-0.5, 0.5]^3

    # Reorder latent_ids to match
    latent_ids = [latent_ids[i] for i in final_order]

    # --- MODIFICATION START: Instance ID Mapping ---
    # 1. Reorder the filtered original indices using the same spatial sort (final_order)
    # This gives us: ordered_original_indices[new_index] = original_index
    ordered_original_indices = [original_indices_filtered[i] for i in final_order]

    # 2. Create a dense mapping array
    # Determine size for mapping array (max id + 1)
    max_original_id = instance_ids.max() if instance_ids.size > 0 else len(obj_ids)
    # Initialize with 0.
    # Note: Any original ID not present in existing_indices will map to 0 (background).
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)

    # 3. Populate map: id_map[original_index] = new_index
    for new_idx, old_idx in enumerate(ordered_original_indices):
        if old_idx < id_map.shape[0]:
            id_map[old_idx] = new_idx

    # 4. Vectorized lookup to generate new instance ids
    out_instance_ids = id_map[instance_ids]
    # --- MODIFICATION END ---

    # Get unique latent_ids (preserving first occurrence order) and build mapping
    unique_latent_ids = list(dict.fromkeys(latent_ids))
    latent_id_to_idx = {lid: idx for idx, lid in enumerate(unique_latent_ids)}

    # Gather all feats and coords for unique latents
    feats_list = [latents[lid]['feats'] for lid in unique_latent_ids]
    coords_list = [latents[lid]['coords'] for lid in unique_latent_ids]

    pbr_feats_list = [pbr_latents[lid]['feats'] for lid in unique_latent_ids]
    # pbr_coords_list = [pbr_latents[lid]['coords'] for lid in unique_latent_ids]

    # Batch normalize: concatenate, normalize, then split back
    split_sizes = [f.shape[0] for f in feats_list]
    all_feats = np.concatenate(feats_list, axis=0)
    num_feats = all_feats.shape[0]
    all_feats_normalized = (all_feats - feat_latent_mean) / feat_latent_std
    # debug: check the mean and std of the normalized feats
    # print(f"Normalized shape feats mean: {all_feats_normalized.reshape(-1, 16).mean(axis=0).tolist()}, std: {all_feats_normalized.reshape(-1, 16).std(axis=0).tolist()}")
    split_indices = np.cumsum(split_sizes[:-1])
    feats_list_normalized = np.split(all_feats_normalized, split_indices)

    split_sizes = [f.shape[0] for f in pbr_feats_list]
    all_pbr_feats = np.concatenate(pbr_feats_list, axis=0)
    num_pbr_feats = all_pbr_feats.shape[0]
    all_pbr_feats_normalized = (all_pbr_feats - pbr_feat_latent_mean) / pbr_feat_latent_std
    # debug: check the mean and std of the normalized feats
    # print(f"Normalized pbr feats mean: {all_pbr_feats_normalized.reshape(-1, 16).mean(axis=0).tolist()}, std: {all_pbr_feats_normalized.reshape(-1, 16).std(axis=0).tolist()}")
    split_indices = np.cumsum(split_sizes[:-1])
    pbr_feats_list_normalized = np.split(all_pbr_feats_normalized, split_indices)

    # Build output indices and assign using vectorized indexing
    output_indices = np.array([latent_id_to_idx[lid] for lid in latent_ids])
    feats_list_normalized = [feats_list_normalized[i] for i in output_indices]
    pbr_feats_list_normalized = [pbr_feats_list_normalized[i] for i in output_indices]
    coords_list = [coords_list[i] for i in output_indices]
    # pbr_coords_list = [pbr_coords_list[i] for i in output_indices]

    # Build output dict with correct dtypes
    return {
        "feats": feats_list_normalized,
        "pbr_feats": pbr_feats_list_normalized,
        "coords": coords_list,
        "transforms": new_transforms_inv.astype(np.float32),
        "instance_ids": out_instance_ids.astype(np.int32)
    }



def read_objects_to_tokens_wo_latents_with_instance_ids(objects_transforms, augment_info, existing_indices, instance_ids):
    """
    Combined and optimized version of read_objects_info + to_object_tokens.
    Directly returns the object tokens dict without intermediate dict creation.

    Args:
        objects_transforms: dict of obj_id -> transform dict
        augment_info: dict with 'augment_transform' and 'norm_transform'
        existing_indices: set/list of existing object indices
        instance_ids: (num_points,) np.array, original instance ids, need to be mapped to the new instance ids

    Returns:
        object_tokens_dict: {
            "translations": (N, 3) int32,
            "angles": (N, 3) int32,
            "scales": (N, 1) int32,
            "instance_ids": (num_points,) int32
        }
    """
    augment_transform = augment_info['augment_transform']
    norm_transform = augment_info['norm_transform']

    # Precompute combined transform
    combined_transform = np.asarray(norm_transform) @ np.asarray(augment_transform)

    # Convert to set for O(1) membership check
    existing_indices_set = set(existing_indices) if not isinstance(existing_indices, set) else existing_indices

    # Separate background from objects
    obj_ids = []
    bg_id = None
    for obj_id in objects_transforms:
        if not obj_id.startswith("layout_"):
            obj_ids.append(obj_id)
        else:
            bg_id = obj_id
    assert bg_id is not None, "Background ID is not found"

    obj_ids = sorted(obj_ids)
    # --- MODIFICATION START ---
    # We need to track the original index (1-based) alongside the object ID
    # to create the mapping later.
    obj_ids_existing = []
    # Initialize with [0] for the background (assuming bg is index 0 in original instance_ids)
    original_indices_filtered = [0]

    for i, obj_id in enumerate(obj_ids):
        # Original logic: checks if i+1 is in existing_indices
        if i + 1 in existing_indices_set:
            obj_ids_existing.append(obj_id)
            original_indices_filtered.append(i + 1)

    # --- MODIFICATION END ---

    # Build list: [bg_id] + existing objects
    obj_ids_to_process = [bg_id] + obj_ids_existing
    N = len(obj_ids_to_process)

    # Gather all transform data into arrays for batched processing
    scales_in = np.empty(N, dtype=np.float64)
    angles_in = np.empty((N, 3), dtype=np.float64)
    trans_in = np.empty((N, 3), dtype=np.float64)

    # Get the background transform
    bg_scale = np.empty(1, dtype=np.float64)
    bg_angles = np.empty((1, 3), dtype=np.float64)
    bg_trans = np.empty((1, 3), dtype=np.float64)
    bg_scale[0] = objects_transforms[bg_id]["scale"]
    bg_angles[0] = np.zeros(3).astype(np.float64) # we will keep background angles unchanged after transform, so set to 0 for now
    bg_trans[0] = objects_transforms[bg_id]["trans"]
    d_bg_scales, d_bg_angles, d_bg_trans = discrete_transform_batch(bg_scale, bg_angles, bg_trans)

    for i, obj_id in enumerate(obj_ids_to_process):
        transform_dict = objects_transforms[obj_id]
        scales_in[i] = transform_dict["scale"]
        angles_in[i] = transform_dict["angles"]
        trans_in[i] = transform_dict["trans"]

    # Batched transform and discretization
    new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
        scales_in, angles_in, trans_in, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)

    # the background angles should not be changed after transform
    d_angles[0] = d_bg_angles[0]

    # Sort non-background objects by d_trans (z, y, x priority)
    # Background stays at index 0, sort indices 1:N
    if N > 1:
        # Get sort order for objects (excluding background at index 0)
        obj_trans = d_trans[1:]  # (N-1, 3)
        # lexsort sorts by last key first, so order is (x, y, z) to get priority z, y, x
        sort_keys = (obj_trans[:, 0], obj_trans[:, 1], obj_trans[:, 2])
        sort_order = np.lexsort(sort_keys)  # indices into obj_trans

        # Build final order: [0] + [1 + sort_order]
        final_order = np.concatenate([[0], 1 + sort_order])
    else:
        final_order = np.array([0])

    # Reorder discretized transforms
    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]

    # --- MODIFICATION START: Instance ID Mapping ---
    # 1. Reorder the filtered original indices using the same spatial sort (final_order)
    # This gives us: ordered_original_indices[new_index] = original_index
    ordered_original_indices = [original_indices_filtered[i] for i in final_order]

    # 2. Create a dense mapping array
    # Determine size for mapping array (max id + 1)
    max_original_id = instance_ids.max() if instance_ids.size > 0 else len(obj_ids)
    # Initialize with 0.
    # Note: Any original ID not present in existing_indices will map to 0 (background).
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)

    # 3. Populate map: id_map[original_index] = new_index
    for new_idx, old_idx in enumerate(ordered_original_indices):
        if old_idx < id_map.shape[0]:
            id_map[old_idx] = new_idx

    # 4. Vectorized lookup to generate new instance ids
    out_instance_ids = id_map[instance_ids]
    # --- MODIFICATION END ---


    # Build output dict with correct dtypes
    return {
        "translations": d_trans.astype(np.int32),
        "angles": d_angles.astype(np.int32),
        "scales": d_scales.astype(np.int32).reshape(-1, 1),
        "instance_ids": out_instance_ids.astype(np.int32)
    }
