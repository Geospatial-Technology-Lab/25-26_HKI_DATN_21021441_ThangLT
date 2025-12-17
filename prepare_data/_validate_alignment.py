import numpy as np

def _validate_alignment(landcover_data, thermal_data):
    """
    Kiểm tra chất lượng căn chỉnh
    """
    print("→ Validating alignment...")

    # Kiểm tra kích thước
    if landcover_data.shape != thermal_data.shape:
        print(f"⚠️ WARNING: Shape mismatch - Landcover: {landcover_data.shape}, Thermal: {thermal_data.shape}")
        return False

    # Kiểm tra dữ liệu hợp lệ
    valid_landcover = np.sum(~np.isnan(landcover_data)) if landcover_data.dtype.kind == 'f' else np.sum(
        landcover_data >= 0)
    valid_thermal = np.sum(~np.isnan(thermal_data)) if thermal_data.dtype.kind == 'f' else np.sum(thermal_data >= 0)

    total_pixels = landcover_data.size

    # Kiểm tra overlap của vùng hợp lệ
    if landcover_data.dtype.kind == 'f':
        landcover_valid_mask = ~np.isnan(landcover_data)
    else:
        landcover_valid_mask = landcover_data >= 0

    if thermal_data.dtype.kind == 'f':
        thermal_valid_mask = ~np.isnan(thermal_data) & (thermal_data > 0)
    else:
        thermal_valid_mask = thermal_data > 0

    overlap = np.sum(landcover_valid_mask & thermal_valid_mask)
    overlap_ratio = overlap / total_pixels

    if overlap_ratio < 0.5:
        print("⚠️ WARNING: Low overlap ratio - có thể cần điều chỉnh phương pháp căn chỉnh")

    return True