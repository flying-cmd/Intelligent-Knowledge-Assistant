import numpy as np
def normalize_sparse_vector(sparse_vec):
    """
    Apply L2 normalization to a sparse vector.
    Only non-zero dimensions are processed, zero dimensions are unaffected.
    :param sparse_vec: Original sparse vector as a dict: {dimension: value}
    :return: Normalized sparse vector
    """
    if not sparse_vec:  # Return empty vectors directly.
        return sparse_vec

    # Extract values from non-zero dimensions.
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # Compute the L2 norm while avoiding division by zero.
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # If the norm is near zero, return the original vector.
        return sparse_vec

    # Normalize each value by the L2 norm.
    normalized_values = values / l2_norm
    # normalized_values = (values / l2_norm).astype(np.float32)  # Cast to float32 if needed.
    # Rebuild the sparse vector dictionary.
    return dict(zip(sparse_vec.keys(), normalized_values))
