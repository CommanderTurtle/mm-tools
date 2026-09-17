import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def sparse_to_dense(feats, coords, resolution):
    """
    Converts sparse feature and coordinate arrays to dense tensors.

    Args:
        feats: (N, C) numpy array of features
        coords: (N, 3) numpy array of integer coordinates [x, y, z]
        resolution: int, the size of the grid dim

    Returns:
        dense_array: (Res, Res, Res, C)
        occupancy: (Res, Res, Res, 1) - uses +1 for occupied, -1 for empty
    """
    if HAS_NUMBA:
        return _sparse_to_dense_numba(feats, coords, resolution)
    return _sparse_to_dense_numpy(feats, coords, resolution)


def _sparse_to_dense_numpy(feats, coords, resolution):
    """Pure NumPy implementation."""
    N, C = feats.shape

    # Initialize dense arrays
    dense_array = np.zeros((resolution, resolution, resolution, C), dtype=feats.dtype)
    occupancy = np.full((resolution, resolution, resolution, 1), -1, dtype=np.float32)

    # Direct slicing avoids transpose overhead
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Scatter feats and set occupancy
    dense_array[x, y, z] = feats
    occupancy[x, y, z] = 1

    return dense_array, occupancy


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _sparse_to_dense_numba(feats, coords, resolution):
        """Numba-accelerated implementation with parallel execution."""
        N, C = feats.shape

        dense_array = np.zeros((resolution, resolution, resolution, C), dtype=feats.dtype)
        occupancy = np.full((resolution, resolution, resolution, 1), -1.0, dtype=np.float32)

        for i in prange(N):
            x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
            for c in range(C):
                dense_array[x, y, z, c] = feats[i, c]
            occupancy[x, y, z, 0] = 1.0

        return dense_array, occupancy


def sparse_to_dense_batch(feats_list, coords_list, resolution):
    """
    Batched version of sparse_to_dense for processing multiple latents at once.

    Args:
        feats_list: list of (N_i, C) numpy arrays of features
        coords_list: list of (N_i, 3) numpy arrays of integer coordinates [x, y, z]
        resolution: int, the size of the grid dim

    Returns:
        dense_array: (B, Res, Res, Res, C)
        occupancy: (B, Res, Res, Res, 1) - uses +1 for occupied, -1 for empty
    """
    B = len(feats_list)
    if B == 0:
        C = 16  # default channel dim
        return (np.zeros((0, resolution, resolution, resolution, C), dtype=np.float32),
                np.full((0, resolution, resolution, resolution, 1), -1, dtype=np.float32))

    C = feats_list[0].shape[1]
    dtype = feats_list[0].dtype

    # Initialize output arrays
    dense_array = np.zeros((B, resolution, resolution, resolution, C), dtype=dtype)
    occupancy = np.full((B, resolution, resolution, resolution, 1), -1, dtype=np.float32)

    # Build batch indices and concatenate all coords/feats
    batch_indices = []
    all_coords = []
    all_feats = []

    for i, (feats, coords) in enumerate(zip(feats_list, coords_list)):
        n = feats.shape[0]
        batch_indices.append(np.full(n, i, dtype=np.int64))
        all_coords.append(coords)
        all_feats.append(feats)

    batch_indices = np.concatenate(batch_indices)
    all_coords = np.concatenate(all_coords, axis=0)
    all_feats = np.concatenate(all_feats, axis=0)

    # Scatter using advanced indexing
    x, y, z = all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]
    dense_array[batch_indices, x, y, z] = all_feats
    occupancy[batch_indices, x, y, z] = 1

    return dense_array, occupancy


def dense_to_sparse(dense_array, occupancy):
    """
    Converts dense array and occupancy back to sparse feats and coords.

    Args:
        dense_array: (Res, Res, Res, C)
        occupancy: (Res, Res, Res, 1) - uses +1 for occupied, -1 for empty

    Returns:
        recovered_feats: (N, C)
        recovered_coords: (N, 3)
    """
    # Create boolean mask (squeeze last dim)
    mask = occupancy[..., 0] > 0

    # np.nonzero is faster than np.argwhere, then stack
    x, y, z = np.nonzero(mask)
    recovered_coords = np.column_stack((x, y, z))

    # Boolean masking for features
    recovered_feats = dense_array[mask]

    return recovered_feats, recovered_coords
