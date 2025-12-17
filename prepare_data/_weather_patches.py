import numpy as np
from typing import Dict, Tuple

def process_weather_patches(weather_data: Dict[str, np.ndarray], 
                          coord: Tuple[int, int],
                          patch_size: int = 100) -> Dict[str, np.ndarray]:
    """Process weather data patches with dynamic normalization to [-1, 1]"""
    # Convert coordinates to integers
    x, y = int(coord[0]), int(coord[1])
    weather_patches = {}
    
    for name, data in weather_data.items():
        try:
            # Ensure coordinates are within bounds
            x_start = max(0, x)
            x_end = min(data.shape[1], x + patch_size)
            y_start = max(0, y)
            y_end = min(data.shape[0], y + patch_size)
            
            patch = data[y_start:y_end, x_start:x_end].copy()
            
            # Handle patches smaller than patch_size
            if patch.shape != (patch_size, patch_size):
                # Pad patch if necessary
                full_patch = np.zeros((patch_size, patch_size), dtype=patch.dtype)
                full_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = full_patch
            
            # Handle missing or invalid data
            if np.any(np.isnan(patch)):
                mask = np.isnan(patch)
                patch[mask] = np.nanmean(patch)
            
            # Dynamic normalization to [-1, 1] range
            patch_min = np.min(patch)
            patch_max = np.max(patch)
            
            # Avoid division by zero
            if patch_max - patch_min > 1e-6:
                # Normalize to [0, 1]
                patch = (patch - patch_min) / (patch_max - patch_min)
                # Scale to [-1, 1]
                patch = patch * 2 - 1
            else:
                # If all values are the same, set to 0
                patch = np.zeros_like(patch)
                
            weather_patches[name] = patch.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {name} patch at coord {coord}: {str(e)}")
            # Return zeros if there's an error
            weather_patches[name] = np.zeros((patch_size, patch_size), dtype=np.float32)
    
    return weather_patches