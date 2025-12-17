import rasterio
import numpy as np

from prepare_data._reproject_landcover import _reproject_landcover
from prepare_data._resample_landcover import _resample_landcover
from prepare_data._match_pixels_landcover import _match_pixels_landcover
from prepare_data._validate_alignment import _validate_alignment
from prepare_data._create_patches import _create_patches

def read_and_align_landcover_to_thermal(landcover_path, thermal_path, patch_size, overlap, method='reproject'):
    """
    Căn chỉnh landcover với thermal image bằng nhiều phương pháp khác nhau

    Args:
        landcover_path: Đường dẫn đến file landcover
        thermal_path: Đường dẫn đến file thermal (làm reference)
        patch_size: Kích thước patch
        overlap: Độ chồng lấp
        method: Phương pháp căn chỉnh ('reproject', 'resample', 'match_pixels')

    Returns:
        landcover_patches: List các patch landcover đã căn chỉnh
    """

    print(f"→ Sử dụng phương pháp căn chỉnh: {method}")

    # Đọc thermal image để lấy thông tin tham chiếu
    with rasterio.open(thermal_path) as thermal_src:
        thermal_data = thermal_src.read(1)
        thermal_crs = thermal_src.crs
        thermal_transform = thermal_src.transform
        thermal_bounds = thermal_src.bounds
        thermal_shape = thermal_data.shape
        thermal_nodata = thermal_src.nodata

    # Đọc landcover image
    with rasterio.open(landcover_path) as landcover_src:
        landcover_data = landcover_src.read(1)
        landcover_crs = landcover_src.crs
        landcover_transform = landcover_src.transform
        landcover_bounds = landcover_src.bounds
        landcover_shape = landcover_data.shape
        landcover_nodata = landcover_src.nodata

    # Chọn phương pháp căn chỉnh
    if method == 'reproject':
        aligned_landcover = _reproject_landcover(
            landcover_data, landcover_crs, landcover_transform,
            thermal_crs, thermal_transform, thermal_shape
        )
    elif method == 'resample':
        aligned_landcover = _resample_landcover(
            landcover_data, landcover_bounds,
            thermal_bounds, thermal_shape
        )
    elif method == 'match_pixels':
        aligned_landcover = _match_pixels_landcover(
            landcover_path, thermal_path
        )
    else:
        raise ValueError(f"Phương pháp '{method}' không được hỗ trợ")

    # Kiểm tra chất lượng căn chỉnh
    _validate_alignment(aligned_landcover, thermal_data)

    # Chia thành patches
    landcover_patches = _create_patches(aligned_landcover, patch_size, overlap)

    print(f"→ Đã tạo {len(landcover_patches)} landcover patches")

    return landcover_patches