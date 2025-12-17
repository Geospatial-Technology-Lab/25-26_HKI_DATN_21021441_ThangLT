import numpy as np

def reconstruct_from_patches(patches, original_shape, patch_size, overlap):
    """
    Ghép các patches lại thành image hoàn chỉnh
    """
    result = np.zeros(original_shape, dtype=patches[0].dtype)
    step = patch_size - overlap
    patch_idx = 0

    for i in range(0, original_shape[0] - patch_size + 1, step):
        for j in range(0, original_shape[1] - patch_size + 1, step):
            if patch_idx < len(patches):
                result[i:i + patch_size, j:j + patch_size] = patches[patch_idx]
                patch_idx += 1

    return result