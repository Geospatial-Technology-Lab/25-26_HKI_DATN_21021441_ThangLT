import numpy as np

def enhanced_landcover_analysis(landcover_patch, thermal_patch, hotspot_pos):
    """
    Phân tích nâng cao về mối quan hệ landcover-thermal
    """
    x, y = hotspot_pos

    # Phân tích cơ bản tại hotspot
    if 0 <= x < landcover_patch.shape[0] and 0 <= y < landcover_patch.shape[1]:
        hotspot_landcover = landcover_patch[x, y]
        hotspot_temp = thermal_patch[x, y]

        # Phân tích vùng lân cận (5x5)
        radius = 2
        neighbors_landcover = []
        neighbors_temp = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < landcover_patch.shape[0] and 0 <= ny < landcover_patch.shape[1]:
                    neighbors_landcover.append(landcover_patch[nx, ny])
                    neighbors_temp.append(thermal_patch[nx, ny])

        # Tính các thống kê
        forest_ratio_around = np.mean([1 if lc == 1 else 0 for lc in neighbors_landcover])
        avg_temp_around = np.mean(neighbors_temp)

        # Phân tích gradient nhiệt độ trong rừng vs không rừng
        forest_temps = [temp for lc, temp in zip(neighbors_landcover, neighbors_temp) if lc == 1]
        non_forest_temps = [temp for lc, temp in zip(neighbors_landcover, neighbors_temp) if lc != 1]

        avg_forest_temp = np.mean(forest_temps) if forest_temps else 0
        avg_non_forest_temp = np.mean(non_forest_temps) if non_forest_temps else 0

        return {
            'hotspot_landcover': hotspot_landcover,
            'hotspot_temp': hotspot_temp,
            'is_forest_at_hotspot': hotspot_landcover == 1,
            'forest_ratio_around_hotspot': forest_ratio_around,
            'avg_temp_around_hotspot': avg_temp_around,
            'avg_forest_temp': avg_forest_temp,
            'avg_non_forest_temp': avg_non_forest_temp,
            'temp_diff_forest_vs_non': avg_forest_temp - avg_non_forest_temp
        }

    return None