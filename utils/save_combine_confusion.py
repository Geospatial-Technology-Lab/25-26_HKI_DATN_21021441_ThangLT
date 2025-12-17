import numpy as np
from utils.decimal_to_dms import decimal_to_dms
import matplotlib.pyplot as plt
import os

def save_combine_confusion(i, patch_coords, highest_temp_pos, overlay, T_Celsius):
    os.makedirs("combined_frames", exist_ok=True)  # Thêm dòng này trước khi savefig

    lat_top_left, lon_left_top = patch_coords[i]

    pixel_resolution_deg_lat = 30 / 111320.0  # Khoảng 1 độ vĩ độ là 111.32 km
    pixel_resolution_deg_lon = 30 / (
            111320.0 * np.cos(np.radians(lat_top_left)))  # Khoảng 1 độ kinh độ thay đổi theo vĩ độ

    # Tọa độ thực tế của điểm nóng nhất
    actual_lat_hotspot = lat_top_left - (highest_temp_pos[0] * pixel_resolution_deg_lat)
    actual_lon_hotspot = lon_left_top + (highest_temp_pos[1] * pixel_resolution_deg_lon)

    lat_dms, lon_dms = decimal_to_dms(lat_top_left, lon_left_top)
    lat_dms_hotspot, lon_dms_hotspot = decimal_to_dms(actual_lat_hotspot, actual_lon_hotspot)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(overlay)
    axes[0].set_title("Confusion Map")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    im = axes[1].imshow(T_Celsius, cmap='hot_r')
    axes[1].set_title("Thermal")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.colorbar(im, ax=axes[1], label="Temperature (°C)")

    fig.suptitle(
        f"Patch {i + 1} — {lat_dms}, {lon_dms} - Hotspot at {lat_dms_hotspot}, {lon_dms_hotspot} - Max Temp : {np.max(T_Celsius):.2f}℃",
        fontsize=12)
    fig.savefig(f"combined_frames/combined_frame_{i + 1}.png")
    plt.close(fig)

    return lat_dms, lon_dms, lat_dms_hotspot, lon_dms_hotspot