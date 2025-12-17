import numpy as np
import rasterio
from rasterio.warp import transform
from patchify import patchify

def read_and_patch_thermal_image(image_path, patch_size=100, overlap=50):
    with rasterio.open(image_path) as src:
        thermal = src.read(1).astype(np.float32)
        transform_affine = src.transform  # Transform của ảnh gốc
        src_crs = src.crs  # EPSG:32649

        # Tính step dựa trên overlap
        step = patch_size - overlap

        # Cắt ảnh thành các patch với chồng lấp
        patches = patchify(thermal, (patch_size, patch_size), step=step)
        patch_list = []
        coords = []

        # Duyệt qua các patch
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, :, :]  # (patch_size, patch_size)
                patch_list.append(patch)
                # Tính tọa độ pixel của góc trên bên trái của patch
                x_pixel = j * step
                y_pixel = i * step
                # Chuyển từ pixel sang tọa độ hệ chiếu gốc (EPSG:32649)
                x_geo, y_geo = transform_affine * (x_pixel, y_pixel)
                # Chuyển sang EPSG:4326 (kinh độ/vĩ độ)
                lon, lat = transform(src_crs, "EPSG:4326", [x_geo], [y_geo])
                coords.append((lat[0], lon[0]))  # Trả về (vĩ độ, kinh độ)

        return patch_list, thermal.shape, coords, transform_affine, src_crs

