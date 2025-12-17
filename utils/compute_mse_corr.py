from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def compute_mse_corr(pred_map, normalized_temp):
    pred_map_flat = pred_map.flatten()
    pred_map_norm = (pred_map_flat - pred_map_flat.min()) / (pred_map_flat.max() - pred_map_flat.min())
    mse = mean_squared_error(normalized_temp.flatten(), pred_map_norm)
    corr, _ = pearsonr(normalized_temp.flatten(), pred_map_norm)
    return mse, corr