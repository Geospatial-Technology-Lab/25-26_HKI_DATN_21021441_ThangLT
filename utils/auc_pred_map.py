# utils/auc_pred_map.py - QUIET VERSION
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import logging

logger = logging.getLogger('PPO_Training')

def compute_auc_and_plot_full(grid_values, fire_ground_truth_binary,
                              fire_ground_truth_continuous, crop_id,
                              normalize=True, pred_map=None,
                              roc_dir="roc_curves",
                              no_value_iteration=False):
    """
    Optimized version - computes AUC without console spam
    """
    h, w = fire_ground_truth_continuous.shape
    y_true_flat = fire_ground_truth_binary.flatten()

    if no_value_iteration:
        pred_map = np.full((h, w), 0.0)
    else:
        if pred_map is None:
            pred_map = np.zeros((h, w))
            for (x, y), v in grid_values.items():
                if 0 <= x < h and 0 <= y < w:
                    pred_map[x, y] = v
        else:
            pred_map = pred_map.copy()

        if normalize and np.max(pred_map) > 0:
            pred_map = (pred_map - np.min(pred_map)) / (np.max(pred_map) - np.min(pred_map))

    auc = None
    if not no_value_iteration and np.unique(y_true_flat).size >= 2:
        auc = roc_auc_score(y_true_flat, pred_map.flatten())
        
        # Calculate optimal threshold but log to file only
        fpr, tpr, thresholds = roc_curve(y_true_flat, pred_map.flatten())
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Log to file instead of console
        logger.info(f"Crop {crop_id} - Optimal Threshold: {optimal_threshold:.4f}, AUC: {auc:.4f}")

    return auc, pred_map
