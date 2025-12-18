"""
Generate Full Fire Probability Raster Map

This script:
1. Loads trained integrated CNN model  
2. Processes the ENTIRE thermal image (not patches)
3. Outputs a GeoTIFF raster with fire probability values
4. Computes evaluation metrics against ground truth

Usage:
    python generate_prediction_raster.py --algorithm a3c --device cuda

Output:
    - {algorithm}_prediction_map.tif  (GeoTIFF with fire probabilities)
    - {algorithm}_binary_map.tif      (Binary fire/no-fire map)
    - evaluation_metrics.json         (AUC, F1, etc.)
"""
import os
import sys
import argparse
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime
from tqdm import tqdm
import json
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score
)

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

from environment.cnn_env import CNNCropThermalEnv

ALGORITHMS = ['a3c', 'a2c', 'ppo', 'dqn', 'sac', 'ddpg', 'vpg']


def normalize_array(arr):
    """Normalize array to [-1, 1] range"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return 2. * (arr - min_val) / (max_val - min_val) - 1.


def load_model(algorithm, device):
    """Load trained model"""
    trainer_classes = {
        'a3c': IntegratedMultiAgentA3C,
        'a2c': IntegratedMultiAgentA2C,
        'ppo': IntegratedMultiAgentPPO,
        'dqn': IntegratedMultiAgentDQN,
        'sac': IntegratedMultiAgentSAC,
        'ddpg': IntegratedMultiAgentDDPG,
        'vpg': IntegratedMultiAgentVPG
    }
    
    # Dummy env factory
    dummy_thermal = np.random.rand(50, 50).astype(np.float32)
    dummy_weather = {name: np.random.rand(50, 50).astype(np.float32) * 2 - 1 
                     for name in ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']}
    
    def dummy_env_factory():
        return CNNCropThermalEnv(
            thermal_data=dummy_thermal,
            start_pos=(25, 25),
            weather_patches=dummy_weather,
            landcover_data=np.ones((50, 50)),
            max_steps=50
        )
    
    trainer = trainer_classes[algorithm](
        env_factory=dummy_env_factory,
        num_agents=1,
        device=device,
        use_icm=False
    )
    
    if not trainer.load_best_model():
        return None
    
    return trainer


def create_full_prediction_map(network, thermal_data, weather_data, device='cpu', 
                                patch_size=11, batch_size=2048):
    """
    Create fire probability map for ENTIRE study area
    
    Returns:
        predictions: numpy array with same shape as thermal_data, values 0-1
    """
    height, width = thermal_data.shape
    half_patch = patch_size // 2
    
    # Initialize output
    predictions = np.zeros((height, width), dtype=np.float32)
    
    # Pad all data
    thermal_padded = np.pad(thermal_data, half_patch, mode='edge')
    
    weather_padded = {}
    weather_names = ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']
    for name in weather_names:
        if name in weather_data and weather_data[name] is not None:
            weather_padded[name] = np.pad(weather_data[name], half_patch, mode='edge')
        else:
            weather_padded[name] = np.zeros_like(thermal_padded)
    
    network.eval()
    
    # Generate all positions
    positions = [(i, j) for i in range(height) for j in range(width)]
    total_positions = len(positions)
    
    print(f"  Processing {total_positions:,} pixels...")
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, total_positions, batch_size), 
                                desc="  Generating predictions", leave=False):
            batch_end = min(batch_start + batch_size, total_positions)
            batch_positions = positions[batch_start:batch_end]
            
            # Build batch observations
            batch_obs = []
            for y, x in batch_positions:
                py = y + half_patch
                px = x + half_patch
                
                # Create 8-channel observation [C, H, W]
                obs = np.zeros((8, patch_size, patch_size), dtype=np.float32)
                
                # Channel 0: thermal
                obs[0] = thermal_padded[py-half_patch:py+half_patch+1, 
                                        px-half_patch:px+half_patch+1]
                
                # Channels 1-7: weather
                for c, name in enumerate(weather_names):
                    obs[c+1] = weather_padded[name][py-half_patch:py+half_patch+1,
                                                    px-half_patch:px+half_patch+1]
                
                batch_obs.append(obs)
            
            # Convert to tensor
            batch_tensor = torch.FloatTensor(np.array(batch_obs)).to(device)
            
            # Handle NaN
            if torch.isnan(batch_tensor).any():
                batch_tensor = torch.nan_to_num(batch_tensor, nan=0.0)
            
            # Get policy output
            policy, _ = network(batch_tensor)
            
            # Action 5 = Predict Fire, use its probability as fire probability
            fire_probs = policy[:, 5].cpu().numpy()
            
            # Store predictions
            for idx, (y, x) in enumerate(batch_positions):
                predictions[y, x] = fire_probs[idx]
    
    return predictions


def save_prediction_raster(predictions, reference_path, output_path):
    """
    Save prediction map as GeoTIFF with same CRS and transform as reference
    """
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw'
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(predictions.astype(np.float32), 1)
    
    print(f"  ✓ Saved: {output_path}")


def compute_metrics(predictions, ground_truth, threshold=0.5):
    """Compute evaluation metrics"""
    y_true = ground_truth.flatten()
    y_pred_proba = predictions.flatten()
    
    # Mask out invalid pixels (NaN, NoData)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_true = y_true[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    # Check for valid data
    n_positive = np.sum(y_true > 0)
    n_negative = np.sum(y_true == 0)
    
    print(f"\n  Ground truth: {n_positive:,} fire pixels, {n_negative:,} non-fire pixels")
    print(f"  Predictions: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")
    
    if n_positive > 0 and n_negative > 0:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred_proba))
        metrics['auc_pr'] = float(average_precision_score(y_true, y_pred_proba))
    else:
        metrics['auc_roc'] = 0.5
        metrics['auc_pr'] = 0.0
    
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate Full Fire Probability Raster")
    parser.add_argument('--algorithm', type=str, default='a3c',
                       choices=ALGORITHMS,
                       help="Algorithm to use")
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use")
    parser.add_argument('--threshold', type=float, default=0.5,
                       help="Threshold for binary classification")
    parser.add_argument('--batch_size', type=int, default=2048,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATE FIRE PROBABILITY RASTER MAP")
    print("="*70)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Paths
    thermal_path = PATHS.get('thermal_tif', 'data/thermal_raster_final.tif')
    output_dir = f"{args.algorithm}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check thermal file
    if not os.path.exists(thermal_path):
        print(f"ERROR: Thermal file not found: {thermal_path}")
        return
    
    # Load model
    print(f"\n→ Loading {args.algorithm.upper()} model...")
    trainer = load_model(args.algorithm, args.device)
    
    if trainer is None:
        print(f"  ✗ Model not found for {args.algorithm}")
        return
    
    network = trainer.agents[0].local_network if hasattr(trainer.agents[0], 'local_network') else trainer.agents[0].network
    print("  ✓ Model loaded")
    
    # Load thermal data
    print(f"\n→ Loading thermal data...")
    with rasterio.open(thermal_path) as src:
        thermal_data = src.read(1)
        thermal_shape = thermal_data.shape
    
    print(f"  Shape: {thermal_shape}")
    
    # Normalize thermal
    thermal_normalized = normalize_array(thermal_data)
    
    # Create ground truth (high temp = fire)
    ground_truth = (thermal_normalized > 0.7).astype(np.float32)
    
    # Load weather data
    print(f"\n→ Loading weather data...")
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
                data = src.read(1)
                # Ensure same shape
                if data.shape == thermal_shape:
                    weather_data[name] = normalize_array(data)
                    print(f"  ✓ {name}: {data.shape}")
                else:
                    print(f"  ✗ {name}: shape mismatch {data.shape} vs {thermal_shape}")
                    weather_data[name] = None
        else:
            print(f"  ✗ {name}: not found")
            weather_data[name] = None
    
    # Generate full prediction map
    print(f"\n→ Generating full prediction map...")
    predictions = create_full_prediction_map(
        network=network,
        thermal_data=thermal_normalized,
        weather_data=weather_data,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Save prediction raster
    print(f"\n→ Saving raster outputs...")
    
    # Fire probability map (0-1)
    prob_output = f"{output_dir}/{args.algorithm}_fire_probability.tif"
    save_prediction_raster(predictions, thermal_path, prob_output)
    
    # Binary fire map
    binary_predictions = (predictions >= args.threshold).astype(np.float32)
    binary_output = f"{output_dir}/{args.algorithm}_fire_binary.tif"
    save_prediction_raster(binary_predictions, thermal_path, binary_output)
    
    # Compute metrics
    print(f"\n→ Computing evaluation metrics...")
    metrics = compute_metrics(predictions, ground_truth, args.threshold)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{args.algorithm.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"AUC-ROC:    {metrics.get('auc_roc', 0):.4f}")
    print(f"AUC-PR:     {metrics.get('auc_pr', 0):.4f}")
    print(f"F1 Score:   {metrics.get('f1', 0):.4f}")
    print(f"Precision:  {metrics.get('precision', 0):.4f}")
    print(f"Recall:     {metrics.get('recall', 0):.4f}")
    print(f"Accuracy:   {metrics.get('accuracy', 0):.4f}")
    print(f"{'='*60}")
    
    # Save metrics
    metrics['algorithm'] = args.algorithm
    metrics['threshold'] = args.threshold
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['input_shape'] = list(thermal_shape)
    
    metrics_path = f"{output_dir}/raster_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved: {metrics_path}")
    
    print(f"\n✓ Outputs saved to {output_dir}/:")
    print(f"  - {args.algorithm}_fire_probability.tif (continuous 0-1)")
    print(f"  - {args.algorithm}_fire_binary.tif (binary 0/1)")
    print(f"  - raster_metrics.json")


if __name__ == "__main__":
    main()
