import numpy as np

def prepare_temp_data_balanced(T_Celsius, balance_method='adaptive'):
    """
    CHANGE: Enhanced data preparation with class balance consideration
    Replace your existing prepare_temp_data with this
    """
    T_Celsius = np.nan_to_num(T_Celsius, nan=0.0, posinf=0.0, neginf=0.0)
    range_temp = np.max(T_Celsius) - np.min(T_Celsius)
    normalized_temp = (T_Celsius - np.min(T_Celsius)) / range_temp if range_temp > 1e-10 else np.full_like(T_Celsius, 0.5)
    
    if balance_method == 'adaptive':
        # Dynamic threshold selection
        percentiles_to_try = [92, 94, 96, 98]
        best_percentile = 98
        best_ratio = float('inf')
        
        for p in percentiles_to_try:
            threshold = np.percentile(T_Celsius, p)
            fire_ratio = np.sum(T_Celsius > threshold) / T_Celsius.size
            target_ratio = 0.05  # Target 5% fire pixels
            if abs(fire_ratio - target_ratio) < best_ratio:
                best_ratio = abs(fire_ratio - target_ratio)
                best_percentile = p
        
        temperature_threshold = np.percentile(T_Celsius, best_percentile)
    else:
        temperature_threshold = np.percentile(T_Celsius, 98)
    
    y_true_binary = (T_Celsius > temperature_threshold).astype(int)
    
    # Calculate class weights
    fire_pixels = np.sum(y_true_binary)
    non_fire_pixels = y_true_binary.size - fire_pixels
    
    class_weights = {
        0: y_true_binary.size / (2 * non_fire_pixels) if non_fire_pixels > 0 else 1.0,
        1: y_true_binary.size / (2 * fire_pixels) if fire_pixels > 0 else 1.0
    }
    
    fire_ratio = fire_pixels / y_true_binary.size
    
    return T_Celsius, normalized_temp, y_true_binary, class_weights, fire_ratio