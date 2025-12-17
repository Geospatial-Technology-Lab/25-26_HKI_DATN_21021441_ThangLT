"""
Evaluation Script for Integrated DRL Models
Creates prediction maps and computes metrics (AUC, F1, Precision, Recall)

Usage:
    python evaluate_integrated.py --algorithm a3c
    python evaluate_integrated.py --algorithm all
"""
import os
import sys
import argparse
import numpy as np
import torch
import rasterio
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATHS

# Import integrated agents
from a3c.integrated_a3c import IntegratedMultiAgentA3C
from a2c.integrated_a2c import IntegratedMultiAgentA2C
from ppo.integrated_ppo import IntegratedMultiAgentPPO
from dqn.integrated_dqn import IntegratedMultiAgentDQN
from sac.integrated_sac import IntegratedMultiAgentSAC
from ddpg.integrated_ddpg import IntegratedMultiAgentDDPG
from vpg.integrated_vpg import IntegratedMultiAgentVPG

# Import CNN environment
from environment.cnn_env import CNNCropThermalEnv


ALGORITHMS = ['a3c', 'a2c', 'ppo', 'dqn', 'sac', 'ddpg', 'vpg']


def normalize_array(arr):
    """Normalize array to [-1, 1] range"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return 2. * (arr - min_val) / (max_val - min_val) - 1.


def prepare_temp_data(thermal_data, high_threshold=0.95):
    """Prepare temperature data and create ground truth"""
    thermal_data = np.nan_to_num(thermal_data, nan=0.0)
    normalized = normalize_array(thermal_data)
    ground_truth = (normalized > high_threshold).astype(np.float32)
    return thermal_data, normalized, ground_truth


def create_prediction_map(agent, thermal_data, weather_patches, 
                          patch_size=11, device='cpu'):
    """Create prediction map by evaluating each position"""
    height, width = thermal_data.shape
    half_patch = patch_size // 2
    
    predictions = np.zeros((height, width), dtype=np.float32)
    
    # Pad data
    thermal_padded = np.pad(thermal_data, half_patch, mode='edge')
    weather_padded = {}
    weather_names = ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']
    
    for name in weather_names:
        if name in weather_patches:
            weather_padded[name] = np.pad(weather_patches[name], half_patch, mode='edge')
        else:
            weather_padded[name] = np.zeros_like(thermal_padded)
    
    # Get network
    network = agent.network if hasattr(agent, 'network') else agent.local_network
    network.eval()
    
    # Batch processing
    batch_size = 512 if device == 'cuda' else 128
    positions = [(x, y) for x in range(height) for y in range(width)]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(positions), batch_size), desc="Predicting", leave=False):
            batch_pos = positions[i:i+batch_size]
            batch_obs = []
            
            for x, y in batch_pos:
                px = x + half_patch
                py = y + half_patch
                
                obs = np.zeros((8, patch_size, patch_size), dtype=np.float32)
                obs[0] = thermal_padded[px-half_patch:px+half_patch+1, py-half_patch:py+half_patch+1]
                
                for j, name in enumerate(weather_names):
                    obs[j+1] = weather_padded[name][px-half_patch:px+half_patch+1, py-half_patch:py+half_patch+1]
                
                batch_obs.append(obs)
            
            batch_tensor = torch.FloatTensor(np.array(batch_obs)).to(device)
            
            # Get action probabilities
            policy, _ = network(batch_tensor)
            
            # Fire prediction is action 5 (Predict Fire)
            fire_probs = policy[:, 5].cpu().numpy()
            
            for j, (x, y) in enumerate(batch_pos):
                predictions[x, y] = fire_probs[j]
    
    return predictions


def compute_metrics(predictions, ground_truth, threshold=0.5):
    """Compute evaluation metrics"""
    y_true = ground_truth.flatten()
    y_pred_proba = predictions.flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Skip if no positive samples
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return None
    
    try:
        metrics = {
            'auc_roc': float(roc_auc_score(y_true, y_pred_proba)),
            'auc_pr': float(average_precision_score(y_true, y_pred_proba)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        }
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn))
        })
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None
    
    return metrics


def plot_results(predictions, ground_truth, metrics, save_path, algorithm):
    """Plot prediction map and comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    axes[0].imshow(ground_truth, cmap='Reds')
    axes[0].set_title('Ground Truth (Fire Areas)')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(predictions, cmap='Reds')
    axes[1].set_title('Model Predictions')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.zeros((*predictions.shape, 3))
    overlay[..., 0] = predictions  # Red = predicted
    overlay[..., 1] = ground_truth  # Green = ground truth  
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=Pred, Green=GT, Yellow=Match)')
    axes[2].axis('off')
    
    auc = metrics.get('auc_roc', 0) if metrics else 0
    f1 = metrics.get('f1', 0) if metrics else 0
    plt.suptitle(f'{algorithm.upper()} | AUC: {auc:.4f} | F1: {f1:.4f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved to {save_path}")


def evaluate_single_model(algorithm, device='cpu', use_sample=True, sample_size=500):
    """Evaluate a single trained model"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm.upper()}")
    print(f"{'='*60}")
    
    # Load thermal data
    thermal_path = PATHS.get('thermal_tif', 'data/thermal_raster_final.tif')
    
    if not os.path.exists(thermal_path):
        print(f"  ✗ Thermal file not found: {thermal_path}")
        return None
    
    print(f"  Loading thermal data...")
    with rasterio.open(thermal_path) as src:
        thermal_data = src.read(1)
    
    # Prepare data
    _, normalized_temp, ground_truth = prepare_temp_data(thermal_data)
    
    # For faster evaluation, use a sample region
    if use_sample and thermal_data.shape[0] > sample_size:
        # Find region with fire pixels
        fire_rows, fire_cols = np.where(ground_truth > 0)
        if len(fire_rows) > 0:
            center_y = int(np.median(fire_rows))
            center_x = int(np.median(fire_cols))
        else:
            center_y = thermal_data.shape[0] // 2
            center_x = thermal_data.shape[1] // 2
        
        half = sample_size // 2
        y1 = max(0, center_y - half)
        y2 = min(thermal_data.shape[0], center_y + half)
        x1 = max(0, center_x - half)
        x2 = min(thermal_data.shape[1], center_x + half)
        
        normalized_temp = normalized_temp[y1:y2, x1:x2]
        ground_truth = ground_truth[y1:y2, x1:x2]
        print(f"  Using sample region: {normalized_temp.shape}")
    
    # Load weather data
    weather_patches = {}
    weather_names = ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']
    for name in weather_names:
        path = PATHS.get(f'{name}_tif', f'database/aligned_{name}.tif')
        if os.path.exists(path):
            with rasterio.open(path) as src:
                data = src.read(1)
                if use_sample and sample_size:
                    data = data[y1:y2, x1:x2]
                weather_patches[name] = normalize_array(data)
    
    # Create dummy environment factory for trainer initialization
    def dummy_env_factory():
        return CNNCropThermalEnv(
            thermal_data=normalized_temp[:50, :50] if normalized_temp.shape[0] >= 50 else normalized_temp,
            start_pos=(25, 25) if normalized_temp.shape[0] >= 50 else (normalized_temp.shape[0]//2, normalized_temp.shape[1]//2),
            weather_patches={k: v[:50, :50] if v.shape[0] >= 50 else v for k, v in weather_patches.items()},
            landcover_data=np.ones((50, 50) if normalized_temp.shape[0] >= 50 else normalized_temp.shape),
            max_steps=50
        )
    
    # Get trainer class
    trainer_classes = {
        'a3c': IntegratedMultiAgentA3C,
        'a2c': IntegratedMultiAgentA2C,
        'ppo': IntegratedMultiAgentPPO,  
        'dqn': IntegratedMultiAgentDQN,
        'sac': IntegratedMultiAgentSAC,
        'ddpg': IntegratedMultiAgentDDPG,
        'vpg': IntegratedMultiAgentVPG
    }
    
    # Create trainer
    print(f"  Creating trainer...")
    trainer = trainer_classes[algorithm](
        env_factory=dummy_env_factory,
        num_agents=1,
        device=device,
        use_icm=False
    )
    
    # Load model
    if not trainer.load_best_model():
        print(f"  ✗ No trained model found for {algorithm}")
        return None
    
    print(f"  ✓ Model loaded")
    
    # Create prediction map
    print(f"  → Creating prediction map...")
    predictions = create_prediction_map(
        agent=trainer.agents[0],
        thermal_data=normalized_temp,
        weather_patches=weather_patches,
        device=device
    )
    
    # Compute metrics
    print(f"  → Computing metrics...")
    metrics = compute_metrics(predictions, ground_truth)
    
    if metrics is None:
        print(f"  ✗ Could not compute metrics")
        return None
    
    # Save results
    results_dir = f"{algorithm}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save plot
    plot_path = f"{results_dir}/evaluation_plot.png"
    plot_results(predictions, ground_truth, metrics, plot_path, algorithm)
    
    # Save metrics
    metrics['algorithm'] = algorithm
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(f"{results_dir}/evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print metrics
    print(f"\n  Metrics for {algorithm.upper()}:")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"    AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    
    return metrics


def compare_all_algorithms(device='cpu'):
    """Compare all algorithms"""
    
    print("\n" + "="*60)
    print("COMPARING ALL INTEGRATED ALGORITHMS")
    print("="*60)
    
    all_metrics = []
    
    for algorithm in ALGORITHMS:
        try:
            metrics = evaluate_single_model(algorithm, device)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {algorithm}: {e}")
    
    if not all_metrics:
        print("No models evaluated successfully")
        return
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Algorithm':<10} {'AUC-ROC':>10} {'AUC-PR':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*60)
    
    for m in sorted(all_metrics, key=lambda x: x['auc_roc'], reverse=True):
        print(f"{m['algorithm'].upper():<10} {m['auc_roc']:>10.4f} {m['auc_pr']:>10.4f} "
              f"{m['f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")
    
    # Save comparison
    with open('comparison_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Comparison saved to comparison_results.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Integrated DRL Models")
    parser.add_argument('--algorithm', type=str, default='a3c',
                       choices=ALGORITHMS + ['all'],
                       help="Algorithm to evaluate (or 'all')")
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use")
    parser.add_argument('--full', action='store_true',
                       help="Evaluate on full image (slow)")
    
    args = parser.parse_args()
    
    use_sample = not args.full
    
    if args.algorithm == 'all':
        compare_all_algorithms(args.device)
    else:
        evaluate_single_model(args.algorithm, args.device, use_sample=use_sample)


if __name__ == "__main__":
    main()
