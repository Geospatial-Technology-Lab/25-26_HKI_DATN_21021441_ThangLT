"""
Evaluation Script for Integrated DRL Models
Creates prediction maps and computes metrics (AUC, F1, Precision, Recall)

Usage:
    python evaluate_integrated.py --algorithm a3c
    python evaluate_integrated.py --algorithm all  # Evaluate all models
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

# Import utilities
from utils.prepare_temp import prepare_temp_data_balanced
from utils.process_weather import process_weather_patches
from utils.thermal_reader import read_and_patch_thermal_image
from utils.landcover_reader import read_and_align_landcover_to_thermal


ALGORITHMS = ['a3c', 'a2c', 'ppo', 'dqn', 'sac', 'ddpg', 'vpg']


def create_prediction_map(agent, thermal_data, weather_patches, landcover_data, 
                          patch_size=11, device='cpu'):
    """
    Create prediction map by evaluating each position
    """
    height, width = thermal_data.shape
    half_patch = patch_size // 2
    
    predictions = np.zeros((height, width), dtype=np.float32)
    
    # Pad data
    thermal_padded = np.pad(thermal_data, half_patch, mode='edge')
    weather_padded = {}
    for name, patch in weather_patches.items():
        weather_padded[name] = np.pad(patch, half_patch, mode='edge')
    
    # Add missing weather channels
    weather_names = ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']
    for name in weather_names:
        if name not in weather_padded:
            weather_padded[name] = np.zeros_like(thermal_padded)
    
    # Batch processing for speed
    batch_size = 1024 if device == 'cuda' else 256
    positions = [(x, y) for x in range(height) for y in range(width)]
    
    agent.network.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(positions), batch_size), desc="Evaluating", leave=False):
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
            if hasattr(agent.network, 'forward'):
                policy, _ = agent.network(batch_tensor)
            else:
                policy = agent.network(batch_tensor)
            
            # Prediction action is action 5
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
    if np.sum(y_true) == 0:
        return None
    
    metrics = {
        'auc_roc': float(roc_auc_score(y_true, y_pred_proba)),
        'auc_pr': float(average_precision_score(y_true, y_pred_proba)),
        'f1': float(f1_score(y_true, y_pred)),
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
    
    return metrics


def plot_results(predictions, ground_truth, metrics, save_path, algorithm):
    """Plot prediction map and confusion matrix"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    axes[0].imshow(ground_truth, cmap='Reds')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(predictions, cmap='Reds')
    axes[1].set_title('Predictions')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.zeros((*predictions.shape, 3))
    overlay[..., 0] = predictions  # Red = predicted
    overlay[..., 1] = ground_truth  # Green = ground truth  
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=Pred, Green=GT, Yellow=Match)')
    axes[2].axis('off')
    
    plt.suptitle(f'{algorithm.upper()} | AUC: {metrics["auc_roc"]:.4f} | F1: {metrics["f1"]:.4f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_single_model(algorithm, device='cpu'):
    """Evaluate a single trained model"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm.upper()}")
    print(f"{'='*60}")
    
    # Load data
    thermal_path = PATHS.get('thermal_tif', 'data/thermal_raster_final.tif')
    landcover_path = PATHS.get('landcover_tif', 'database/aligned_landcover.tif')
    
    # Weather data paths
    weather_tifs = {
        'humidity': PATHS.get('humidity_tif', 'database/aligned_humidity.tif'),
        'wind_speed': PATHS.get('wind_speed_tif', 'database/aligned_wind_speed.tif'),
        'soil_temp': PATHS.get('soil_temp_tif', 'database/aligned_soil_temp.tif'),
        'soil_moisture': PATHS.get('soil_moisture_tif', 'database/aligned_soil_moisture.tif'),
        'rainfall': PATHS.get('rainfall_tif', 'database/aligned_rainfall.tif'),
        'ndmi': PATHS.get('ndmi_tif', 'database/aligned_ndmi.tif'),
        'dem': PATHS.get('dem_tif', 'database/aligned_dem.tif'),
    }
    
    # Load thermal data
    with rasterio.open(thermal_path) as src:
        thermal_data = src.read(1)
    
    # Load weather data
    weather_data = {}
    for name, path in weather_tifs.items():
        if os.path.exists(path):
            with rasterio.open(path) as src:
                weather_data[name] = src.read(1)
    
    # Load landcover
    with rasterio.open(landcover_path) as src:
        landcover_data = src.read(1)
    
    # Prepare ground truth
    _, normalized_temp, ground_truth, _, _ = prepare_temp_data_balanced(thermal_data)
    
    # Process weather patches
    coord = (0, 0, thermal_data.shape[0], thermal_data.shape[1])
    weather_patches = process_weather_patches(weather_data, coord)
    
    # Create dummy environment factory for loading model
    def dummy_env_factory():
        return CNNCropThermalEnv(
            thermal_data=normalized_temp[:100, :100],
            start_pos=(50, 50),
            weather_patches={k: v[:100, :100] for k, v in weather_patches.items()},
            landcover_data=landcover_data[:100, :100],
            max_steps=100
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
    trainer = trainer_classes[algorithm](
        env_factory=dummy_env_factory,
        num_agents=1,
        device=device,
        use_icm=False  # Not needed for evaluation
    )
    
    # Load best model
    if not trainer.load_best_model():
        print(f"  ✗ No trained model found for {algorithm}")
        return None
    
    print(f"  ✓ Model loaded successfully")
    
    # Create prediction map
    print("  → Creating prediction map...")
    predictions = create_prediction_map(
        agent=trainer.agents[0],
        thermal_data=normalized_temp,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        device=device
    )
    
    # Compute metrics
    print("  → Computing metrics...")
    metrics = compute_metrics(predictions, ground_truth)
    
    if metrics is None:
        print(f"  ✗ No positive samples in ground truth")
        return None
    
    # Save results
    results_dir = f"{algorithm}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot results
    plot_path = f"{results_dir}/evaluation_plot.png"
    plot_results(predictions, ground_truth, metrics, plot_path, algorithm)
    print(f"  ✓ Plot saved to {plot_path}")
    
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
        metrics = evaluate_single_model(algorithm, device)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No models evaluated successfully")
        return
    
    # Create comparison table
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
    
    # Plot comparison
    plot_comparison(all_metrics)


def plot_comparison(all_metrics):
    """Plot comparison bar chart"""
    algorithms = [m['algorithm'].upper() for m in all_metrics]
    metrics_names = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    
    x = np.arange(len(algorithms))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric_name in enumerate(metrics_names):
        values = [m[metric_name] for m in all_metrics]
        ax.bar(x + i*width, values, width, label=metric_name.replace('_', ' ').title())
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.set_title('Integrated DRL Models Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)
    plt.close()
    print("✓ Comparison plot saved to comparison_plot.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Integrated DRL Models")
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=ALGORITHMS + ['all'],
                       help="Algorithm to evaluate (or 'all')")
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use")
    
    args = parser.parse_args()
    
    if args.algorithm == 'all':
        compare_all_algorithms(args.device)
    else:
        evaluate_single_model(args.algorithm, args.device)


if __name__ == "__main__":
    main()
