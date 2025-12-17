import numpy as np

def _resample_landcover(landcover_data, landcover_bounds, thermal_bounds, thermal_shape):
    """
    Resample landcover dựa trên bounds và shape của thermal
    """
    print("→ Thực hiện resampling based on bounds...")

    from scipy.ndimage import zoom

    # Tính toán tỷ lệ zoom
    height_ratio = thermal_shape[0] / landcover_data.shape[0]
    width_ratio = thermal_shape[1] / landcover_data.shape[1]

    # Thực hiện zoom với nearest neighbor
    resampled_landcover = zoom(
        landcover_data,
        (height_ratio, width_ratio),
        order=0,  # nearest neighbor
        mode='nearest'
    )

    # Đảm bảo kích thước chính xác
    if resampled_landcover.shape != thermal_shape:

        # Crop hoặc pad để match chính xác
        result = np.zeros(thermal_shape, dtype=landcover_data.dtype)

        end_h = min(thermal_shape[0], resampled_landcover.shape[0])
        end_w = min(thermal_shape[1], resampled_landcover.shape[1])

        result[:end_h, :end_w] = resampled_landcover[:end_h, :end_w]

        return result

    return resampled_landcover
