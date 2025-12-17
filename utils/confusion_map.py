# utils/confusion_map.py - QUIET VERSION
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import logging

logger = logging.getLogger('PPO_Training')

def compute_confusion_overlay(pred_map, gt_binary, crop_idx, threshold=None):
    """
    Optimized version - computes confusion overlay without console spam
    """
    if threshold is None:
        # Calculate optimal threshold using Youden's J Statistic
        fpr, tpr, thresholds = roc_curve(gt_binary.flatten(), pred_map.flatten())
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

    # Ensure input data has no NaN values
    pred_map = np.nan_to_num(pred_map)

    # 1. Binarize prediction
    pred_bin = (pred_map > threshold).astype(np.uint8)

    # 2. Create masks
    tp = (pred_bin == 1) & (gt_binary == 1)
    fp = (pred_bin == 1) & (gt_binary == 0)
    fn = (pred_bin == 0) & (gt_binary == 1)
    tn = (pred_bin == 0) & (gt_binary == 0)

    # 3. Create RGB overlay image
    h, w = pred_map.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[tp] = [255, 0, 0]  # Red - True Positive
    overlay[fp] = [0, 0, 255]  # Blue - False Positive
    overlay[fn] = [0, 128, 1]  # Green - False Negative
    overlay[tn] = [255, 255, 255]    # White - True Negative

    # Calculate metrics
    tp_count = np.sum(tp)
    fp_count = np.sum(fp)
    fn_count = np.sum(fn)
    tn_count = np.sum(tn)

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate PR-AUC without plotting
    precision_vals, recall_vals, _ = precision_recall_curve(gt_binary.flatten(), pred_map.flatten())
    pr_auc = auc(recall_vals, precision_vals)
    
    # Log to file instead of console
    logger.info(f"Crop {crop_idx}: Threshold={threshold:.4f}, "
                f"TP={tp_count}, FP={fp_count}, FN={fn_count}, TN={tn_count}, "
                f"Precision={precision:.4f}, Recall={recall:.4f}, "
                f"F1={f1:.4f}, PR-AUC={pr_auc:.4f}")

    return overlay, precision, recall, f1, pr_auc