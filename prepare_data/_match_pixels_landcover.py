import rasterio
from rasterio.transform import from_bounds
import numpy as np
from rasterio.warp import reproject, Resampling

def _match_pixels_landcover(landcover_path, thermal_path):
    """
    Sử dụng rasterio để match pixels chính xác dựa trên coordinate system
    """
    print("→ Thực hiện pixel matching...")

    with rasterio.open(thermal_path) as thermal_src:
        # Đọc metadata của thermal
        thermal_profile = thermal_src.profile
        thermal_bounds = thermal_src.bounds
        thermal_transform = thermal_src.transform
        thermal_crs = thermal_src.crs
        thermal_shape = (thermal_src.height, thermal_src.width)

        # Tạo transform mới cho landcover dựa trên thermal bounds và shape
        new_transform = from_bounds(
            thermal_bounds.left, thermal_bounds.bottom,
            thermal_bounds.right, thermal_bounds.top,
            thermal_shape[1], thermal_shape[0]  # width, height
        )

        # Đọc và reproject landcover
        with rasterio.open(landcover_path) as landcover_src:
            landcover_data = landcover_src.read(1)

            # Tạo array đích
            aligned_landcover = np.zeros(thermal_shape, dtype=landcover_data.dtype)

            # Reproject với transform chính xác
            reproject(
                source=landcover_data,
                destination=aligned_landcover,
                src_transform=landcover_src.transform,
                src_crs=landcover_src.crs,
                dst_transform=new_transform,
                dst_crs=thermal_crs,
                resampling=Resampling.nearest
            )

            return aligned_landcover