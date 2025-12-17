"""
Full Evaluation Script for Integrated DRL Models
Evaluates on all patches across the entire thermal image (similar to a3c_main.py)

Usage:
    python evaluate_integrated_full.py --algorithm a3c --device cuda
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
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

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
from utils.auc_pred_map import compute_auc_and_plot_full
from utils.confusion_map import compute_confusion_overlay
from utils.save_tif_confusion import save_tif_confusion
from utils.save_combine_confusion import save_combine_confusion
from utils.final_map import export_final_confusion_map
from prepare_data._crop_thermal_true_img import read_and_patch_thermal_image
from prepare_data._read_and_align_landcover_to_thermal import read_and_align_landcover_to_thermal
from prepare_data._find_threshold import prepare_temp_data_balanced as prepare_temp_balanced_orig

ALGORITHMS = ['a3c', 'a2c', 'ppo', 'dqn', 'sac', 'ddpg', 'vpg']


def normalize_array(arr):
    """Normalize array to [-1, 1] range"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return 2. * (arr - min_val) / (max_val - min_val) - 1.


def prepare_temp_data_balanced(T_Celsius):
    """Prepare temperature data with balanced thresholds"""
    T_Celsius = np.nan_to_num(T_Celsius, nan=0.0)
    
    # Normalize to [0, 1]
    min_temp = T_Celsius.min()
    max_temp = T_Celsius.max()
    if max_temp - min_temp < 1e-8:
        normalized_temp = np.zeros_like(T_Celsius)
    else:
        normalized_temp = (T_Celsius - min_temp) / (max_temp - min_temp)
    
    # Create binary ground truth
    high_threshold = 0.7
    y_true_binary = (normalized_temp > high_threshold).astype(np.float32)
    
    return T_Celsius, normalized_temp, y_true_binary, min_temp, max_temp


def process_weather_patches(weather_data, coord):
    """Extract weather patches for a given coordinate region"""
    y1, x1, y2, x2 = coord
    patches = {}
    for name, data in weather_data.items():
        if data.shape[0] > y2 and data.shape[1] > x2:
            patches[name] = normalize_array(data[y1:y2, x1:x2])
        else:
            patches[name] = np.zeros((y2-y1, x2-x1))
    return patches


def create_prediction_map_cnn(agent, thermal_data, weather_patches, landcover_data, device='cpu'):
    """Create prediction map using CNN model"""
    height, width = thermal_data.shape
    patch_size = 11
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
    batch_size = 1024 if device == 'cuda' else 256
    positions = [(x, y) for x in range(height) for y in range(width)]
    
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
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
            
            # Handle NaN
            if torch.isnan(batch_tensor).any():
                batch_tensor = torch.nan_to_num(batch_tensor, nan=0.0)
            
            # Get action probabilities
            policy, values = network(batch_tensor)
            
            # Fire prediction is action 5
            fire_probs = policy[:, 5].cpu().numpy()
            
            for j, (x, y) in enumerate(batch_pos):
                predictions[x, y] = fire_probs[j]
    
    return predictions


def evaluate_single_patch(patch_data, agent, device, patch_coords, transform_affine, src_crs, original_shape):
    """Evaluate a single patch"""
    try:
        predictions = create_prediction_map_cnn(
            agent=agent,
            thermal_data=patch_data['thermal_data'],
            weather_patches=patch_data['weather_patches'],
            landcover_data=patch_data['landcover_data'],
            device=device
        )
        
        y_true = patch_data['y_true_binary']
        
        # Compute metrics
        y_true_flat = y_true.flatten()
        y_pred_proba = predictions.flatten()
        
        # Adaptive threshold
        threshold = max(0.1, np.percentile(y_pred_proba, 90))
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Metrics
        if len(np.unique(y_true_flat)) > 1 and np.sum(y_true_flat) > 0:
            auc = roc_auc_score(y_true_flat, y_pred_proba)
            pr_auc = average_precision_score(y_true_flat, y_pred_proba)
        else:
            auc = 0.5
            pr_auc = 0.0
        
        precision = precision_score(y_true_flat, y_pred, zero_division=0)
        recall = recall_score(y_true_flat, y_pred, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred, zero_division=0)
        
        return {
            'success': True,
            'Patch': patch_data['index'] + 1,
            'AUC': float(auc),
            'PR_AUC': float(pr_auc),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'Fire_Ratio': float(patch_data.get('fire_ratio', 0)),
            'Max_Temp': float(np.max(patch_data.get('T_Celsius', patch_data['thermal_data'])))
        }
        
    except Exception as e:
        print(f"  Error processing patch {patch_data['index']}: {e}")
        return {'success': False, 'Patch': patch_data['index'] + 1}


def load_trainer(algorithm, device, dummy_env_factory):
    """Load trained model for given algorithm"""
    trainer_classes = {
        'a3c': IntegratedMultiAgentA3C,
        'a2c': IntegratedMultiAgentA2C,
        'ppo': IntegratedMultiAgentPPO,
        'dqn': IntegratedMultiAgentDQN,
        'sac': IntegratedMultiAgentSAC,
        'ddpg': IntegratedMultiAgentDDPG,
        'vpg': IntegratedMultiAgentVPG
    }
    
    trainer = trainer_classes[algorithm](
        env_factory=dummy_env_factory,
        num_agents=1,
        device=device,
        use_icm=False
    )
    
    if not trainer.load_best_model():
        return None
    
    return trainer


def evaluate_full(algorithm, device='cpu', max_patches=None):
    """Full evaluation on all patches"""
    
    print(f"\n{'='*80}")
    print(f"FULL EVALUATION - {algorithm.upper()}")
    print(f"{'='*80}")
    
    # Paths
    thermal_path = PATHS.get('thermal_tif', 'data/thermal_raster_final.tif')
    landcover_path = PATHS.get('landcover_tif', 'database/aligned_landcover.tif')
    
    if not os.path.exists(thermal_path):
        print(f"ERROR: Thermal file not found: {thermal_path}")
        return None
    
    # Load weather data
    print("\n→ Loading weather data...")
    weather_tifs = {
        'humidity': PATHS.get('humidity_tif', 'database/aligned_humidity.tif'),
        'wind_speed': PATHS.get('wind_speed_tif', 'database/aligned_wind_speed.tif'),
        'soil_temp': PATHS.get('soil_temp_tif', 'database/aligned_soil_temp.tif'),
        'soil_moisture': PATHS.get('soil_moisture_tif', 'database/aligned_soil_moisture.tif'),
        'rainfall': PATHS.get('rainfall_tif', 'database/aligned_rainfall.tif'),
        'ndmi': PATHS.get('ndmi_tif', 'database/aligned_ndmi.tif'),
        'dem': PATHS.get('dem_tif', 'database/aligned_dem.tif'),
    }
    
    weather_data = {}
    for name, path in weather_tifs.items():
        if os.path.exists(path):
            with rasterio.open(path) as src:
                weather_data[name] = src.read(1)
            print(f"  ✓ {name}: {weather_data[name].shape}")
    
    # Read thermal patches
    print("\n→ Reading thermal patches...")
    patch_size = 100
    overlap = 10
    patches, original_shape, patch_coords, transform_affine, src_crs = read_and_patch_thermal_image(
        thermal_path, patch_size, overlap
    )
    print(f"  Created {len(patches)} patches")
    
    # Read landcover
    print("\n→ Reading landcover...")
    landcover_patches = read_and_align_landcover_to_thermal(
        landcover_path, thermal_path, patch_size, overlap
    )
    
    # Prepare valid patches
    print("\n→ Preparing valid patches...")
    patch_list = []
    
    for i, (T_Celsius, landcover_patch, coord) in enumerate(zip(patches, landcover_patches, patch_coords)):
        T_Celsius, normalized_temp, y_true_binary, _, _ = prepare_temp_data_balanced(T_Celsius)
        
        # Skip invalid
        if np.any(T_Celsius <= 0.0) or np.sum(y_true_binary) == 0:
            continue
        
        weather_patches = process_weather_patches(weather_data, coord)
        
        patch_list.append({
            'index': i,
            'thermal_data': normalized_temp,
            'T_Celsius': T_Celsius,
            'start_pos': np.unravel_index(np.argmax(T_Celsius), T_Celsius.shape),
            'weather_patches': weather_patches,
            'landcover_data': landcover_patch,
            'y_true_binary': y_true_binary,
            'fire_ratio': np.sum(y_true_binary) / y_true_binary.size,
            'coord': coord
        })
    
    print(f"  {len(patch_list)} valid patches")
    
    if max_patches and len(patch_list) > max_patches:
        patch_list = patch_list[:max_patches]
        print(f"  Using first {max_patches} patches")
    
    # Create dummy env factory
    def dummy_env_factory():
        p = patch_list[0]
        return CNNCropThermalEnv(
            thermal_data=p['thermal_data'],
            start_pos=p['start_pos'],
            weather_patches=p['weather_patches'],
            landcover_data=p['landcover_data'],
            max_steps=50
        )
    
    # Load model
    print(f"\n→ Loading {algorithm.upper()} model...")
    trainer = load_trainer(algorithm, device, dummy_env_factory)
    
    if trainer is None:
        print(f"  ✗ Model not found for {algorithm}")
        return None
    
    print(f"  ✓ Model loaded")
    
    # Evaluate all patches
    print(f"\n→ Evaluating {len(patch_list)} patches...")
    all_results = []
    agent = trainer.agents[0]
    
    for patch_data in tqdm(patch_list, desc="Evaluating patches"):
        result = evaluate_single_patch(
            patch_data=patch_data,
            agent=agent,
            device=device,
            patch_coords=patch_coords,
            transform_affine=transform_affine,
            src_crs=src_crs,
            original_shape=original_shape
        )
        
        if result['success']:
            all_results.append(result)
        
        # Memory cleanup
        if len(all_results) % 100 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    if not all_results:
        print("No successful evaluations")
        return None
    
    # Compute summary
    df = pd.DataFrame(all_results)
    
    numeric_cols = ['AUC', 'PR_AUC', 'Precision', 'Recall', 'F1']
    avg_metrics = df[numeric_cols].mean()
    
    # Save results
    results_dir = f"{algorithm}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    df.to_csv(f"{results_dir}/full_evaluation_results.csv", index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{algorithm.upper()} EVALUATION SUMMARY ({len(all_results)} patches)")
    print(f"{'='*60}")
    print(f"Average AUC:       {avg_metrics['AUC']:.4f}")
    print(f"Average PR-AUC:    {avg_metrics['PR_AUC']:.4f}")
    print(f"Average Precision: {avg_metrics['Precision']:.4f}")
    print(f"Average Recall:    {avg_metrics['Recall']:.4f}")
    print(f"Average F1:        {avg_metrics['F1']:.4f}")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'algorithm': algorithm,
        'num_patches': len(all_results),
        'avg_auc': float(avg_metrics['AUC']),
        'avg_pr_auc': float(avg_metrics['PR_AUC']),
        'avg_precision': float(avg_metrics['Precision']),
        'avg_recall': float(avg_metrics['Recall']),
        'avg_f1': float(avg_metrics['F1']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{results_dir}/full_evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation of Integrated DRL Models")
    parser.add_argument('--algorithm', type=str, default='a3c',
                       choices=ALGORITHMS + ['all'],
                       help="Algorithm to evaluate")
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use")
    parser.add_argument('--max_patches', type=int, default=None,
                       help="Max patches to evaluate (None = all)")
    
    args = parser.parse_args()
    
    if args.algorithm == 'all':
        all_summaries = []
        for algo in ALGORITHMS:
            summary = evaluate_full(algo, args.device, args.max_patches)
            if summary:
                all_summaries.append(summary)
        
        if all_summaries:
            print("\n" + "="*60)
            print("COMPARISON OF ALL ALGORITHMS")
            print("="*60)
            print(f"{'Algorithm':<10} {'AUC':>10} {'PR-AUC':>10} {'F1':>10}")
            print("-"*40)
            for s in sorted(all_summaries, key=lambda x: x['avg_auc'], reverse=True):
                print(f"{s['algorithm'].upper():<10} {s['avg_auc']:>10.4f} {s['avg_pr_auc']:>10.4f} {s['avg_f1']:>10.4f}")
    else:
        evaluate_full(args.algorithm, args.device, args.max_patches)


if __name__ == "__main__":
    main()
