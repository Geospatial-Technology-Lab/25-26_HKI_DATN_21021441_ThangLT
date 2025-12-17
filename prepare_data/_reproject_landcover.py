import numpy as np
from rasterio.warp import reproject, Resampling

def _reproject_landcover(landcover_data, landcover_crs, landcover_transform,
                         thermal_crs, thermal_transform, thermal_shape):
    """
    Reproject landcover về cùng CRS và grid với thermal
    """
    print("→ Thực hiện reprojection...")

    # Tạo array đích với kích thước thermal
    reprojected_landcover = np.zeros(thermal_shape, dtype=landcover_data.dtype)

    # Thực hiện reproject
    reproject(
        source=landcover_data,
        destination=reprojected_landcover,
        src_transform=landcover_transform,
        src_crs=landcover_crs,
        dst_transform=thermal_transform,
        dst_crs=thermal_crs,
        resampling=Resampling.nearest  # Dùng nearest neighbor cho landcover
    )

    return reprojected_landcover