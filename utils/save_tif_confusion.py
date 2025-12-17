from rasterio.transform import from_origin
import rasterio
from PIL import Image
import os
import logging

logger = logging.getLogger('PPO_Training')
def save_tif_confusion(original_shape, step, i, transform_affine, overlay, src_crs):
    os.makedirs("confusion_maps", exist_ok=True)
    
    # Calculate transform for patch
    row = i // (original_shape[1] // step)
    col = i % (original_shape[1] // step)
    x_pixel = col * step
    y_pixel = row * step
    x_geo, y_geo = transform_affine * (x_pixel, y_pixel)
    pixel_size_x = transform_affine.a
    pixel_size_y = transform_affine.e
    transform_patch = from_origin(x_geo, y_geo, pixel_size_x, -pixel_size_y)

    # Convert RGB overlay to single band
    overlay_gray = (overlay[:, :, 0] * 0.299 + 
                   overlay[:, :, 1] * 0.587 + 
                   overlay[:, :, 2] * 0.114).astype('uint8')

    # Save single-band GeoTIFF with metadata
    overlay_path = f"confusion_maps/patch_{i + 1}_confusion.tif"
    with rasterio.open(
            overlay_path,
            'w',
            driver='GTiff',
            height=overlay_gray.shape[0],
            width=overlay_gray.shape[1],
            count=1,
            dtype='uint8',
            crs=src_crs,
            transform=transform_patch,
            nodata=255
    ) as dst:
        dst.write(overlay_gray, 1)
        dst.update_tags(TIFFTAG_DOCUMENTNAME=f"Confusion Map Patch {i+1}")
        dst.update_tags(PATCH_ID=str(i+1))
        dst.update_tags(ORIGINAL_SHAPE=f"{original_shape}")

    logger.debug(f"Saved overlay GeoTIFF: {overlay_path}")
    # Save PNG for easy viewing
    Image.fromarray(overlay).save(f"confusion_maps/patch_{i + 1}_confusion.png")