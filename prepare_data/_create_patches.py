def _create_patches(data, patch_size, overlap):
    """
    Chia data thành patches
    """
    patches = []
    step = patch_size - overlap

    for i in range(0, data.shape[0] - patch_size + 1, step):
        for j in range(0, data.shape[1] - patch_size + 1, step):
            patch = data[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches