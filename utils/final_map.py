import os
import numpy as np
import rasterio
from PIL import Image

def export_final_confusion_map(patch_size, overlap, step, thermal_path):
    os.makedirs("final_maps", exist_ok=True)
    
    # Read metadata from original image
    with rasterio.open(thermal_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    # Calculate patches
    n_rows = (height - overlap) // step
    n_cols = (width - overlap) // step

    # Initialize single-band array
    full_confusion = np.full((height, width), 255, dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    # Process patches
    for i in range(n_rows * n_cols):
        patch_path = f"confusion_maps/patch_{i + 1}_confusion.tif"
        if os.path.exists(patch_path):
            with rasterio.open(patch_path) as patch_ds:
                patch_data = patch_ds.read(1).astype(np.float32)

            patch_row = i // n_cols
            patch_col = i % n_cols
            row = patch_row * step
            col = patch_col * step

            non_white_mask = patch_data != 255
            full_confusion[row:row + patch_size, col:col + patch_size][non_white_mask] = \
                patch_data[non_white_mask]
            count_map[row:row + patch_size, col:col + patch_size][non_white_mask] += 1

    # Process overlapping regions
    mask = count_map > 0
    full_confusion[mask] /= count_map[mask]
    full_confusion[~mask] = 255
    full_confusion = np.clip(full_confusion, 0, 255).astype(np.uint8)

    # Save single-band GeoTIFF with metadata
    mosaic_path = "final_maps\\full_confusion_map.tif"
    meta.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "transform": transform,
        "crs": crs
    })

    with rasterio.open(mosaic_path, 'w', **meta) as dst:
        dst.write(full_confusion, 1)
        dst.update_tags(TIFFTAG_DOCUMENTNAME="Full Confusion Map")
        dst.update_tags(PATCH_SIZE=str(patch_size))
        dst.update_tags(OVERLAP=str(overlap))
        dst.update_tags(STEP=str(step))

    print("✅ Saved mosaic GeoTIFF at:", mosaic_path)

    # Convert to RGB for PNG viewing
    full_confusion_rgb = np.stack([full_confusion] * 3, axis=-1)
    png_path = "final_maps\\full_confusion_map.png"
    Image.fromarray(full_confusion_rgb).save(png_path)
    print("✅ Saved mosaic PNG at:", png_path)