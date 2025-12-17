"""
Training Script for Integrated DRL Models with Real Data
Supports all 7 algorithms: A3C, A2C, PPO, DQN, SAC, DDPG, VPG

Usage:
    python train_integrated_main.py --algorithm a3c --episodes 100 --device cuda
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config for paths
from config import PATHS, TRAINING_CONFIG

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


def normalize_array(arr):
    """Normalize array to [-1, 1] range"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return 2. * (arr - min_val) / (max_val - min_val) - 1.


def prepare_temp_data(thermal_data, high_threshold=0.95, medium_threshold=0.85):
    """Prepare temperature data and create ground truth"""
    # Clean data
    thermal_data = np.nan_to_num(thermal_data, nan=0.0)
    
    # Normalize
    normalized = normalize_array(thermal_data)
    
    # Create ground truth (fire = 1 where normalized temp > threshold)
    ground_truth = (normalized > high_threshold).astype(np.float32)
    
    return thermal_data, normalized, ground_truth


def load_weather_data(weather_tifs: dict) -> dict:
    """Load weather TIF files"""
    weather_data = {}
    for name, path in weather_tifs.items():
        if os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    data = src.read(1)
                    weather_data[name] = normalize_array(data)
                print(f"  ✓ Loaded {name}: {data.shape}")
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")
        else:
            print(f"  ✗ Missing {name}: {path}")
    return weather_data


def create_patches(thermal_data, weather_data, landcover_data, patch_size=100, stride=50):
    """Create training patches from data"""
    height, width = thermal_data.shape
    patches = []
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            thermal_patch = thermal_data[y:y+patch_size, x:x+patch_size]
            
            # Skip invalid patches
            if np.all(thermal_patch == 0) or np.isnan(thermal_patch).any():
                continue
            
            # Normalize thermal patch
            _, normalized_thermal, ground_truth = prepare_temp_data(thermal_patch)
            
            # Skip if no fire pixels
            fire_ratio = np.sum(ground_truth) / ground_truth.size
            if fire_ratio == 0 or fire_ratio == 1:
                continue
            
            # Extract weather patches
            weather_patches = {}
            for name, data in weather_data.items():
                if data.shape == thermal_data.shape:
                    weather_patches[name] = data[y:y+patch_size, x:x+patch_size]
                else:
                    # Resize or create zeros
                    weather_patches[name] = np.zeros((patch_size, patch_size))
            
            # Landcover patch
            if landcover_data is not None and landcover_data.shape == thermal_data.shape:
                lc_patch = landcover_data[y:y+patch_size, x:x+patch_size]
            else:
                lc_patch = np.ones((patch_size, patch_size))
            
            # Find start position (hottest point)
            start_pos = np.unravel_index(np.argmax(normalized_thermal), normalized_thermal.shape)
            
            patches.append({
                'thermal_data': normalized_thermal,
                'start_pos': start_pos,
                'weather_patches': weather_patches,
                'landcover_data': lc_patch,
                'ground_truth': ground_truth,
                'fire_ratio': fire_ratio
            })
    
    return patches


def create_cnn_env_factory(patch_list, obs_patch_size=11):
    """Create factory function for CNN environment"""
    
    def env_factory():
        # Weighted sampling - prefer patches with moderate fire ratio
        weights = [1.0 / (abs(p['fire_ratio'] - 0.1) + 0.1) for p in patch_list]
        weights = np.array(weights) / np.sum(weights)
        patch = np.random.choice(patch_list, p=weights)
        
        return CNNCropThermalEnv(
            thermal_data=patch['thermal_data'],
            start_pos=patch['start_pos'],
            weather_patches=patch['weather_patches'],
            landcover_data=patch['landcover_data'],
            max_steps=min(200, patch['thermal_data'].size // 10),
            patch_size=obs_patch_size,
            verbose=False
        )
    
    return env_factory


def get_trainer(algorithm: str, env_factory, num_agents: int, device: str, use_icm: bool):
    """Get trainer based on algorithm name"""
    
    common_kwargs = {
        'env_factory': env_factory,
        'num_agents': num_agents,
        'num_channels': 8,
        'patch_size': 11,
        'action_size': 6,
        'device': device,
        'use_icm': use_icm
    }
    
    trainers = {
        'a3c': IntegratedMultiAgentA3C,
        'a2c': IntegratedMultiAgentA2C,
        'ppo': IntegratedMultiAgentPPO,
        'dqn': IntegratedMultiAgentDQN,
        'sac': IntegratedMultiAgentSAC,
        'ddpg': IntegratedMultiAgentDDPG,
        'vpg': IntegratedMultiAgentVPG
    }
    
    if algorithm not in trainers:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(trainers.keys())}")
    
    return trainers[algorithm](**common_kwargs)


def train_model(trainer, algorithm: str, max_episodes: int, steps_per_update: int,
                save_interval: int, device: str):
    """Train the model"""
    
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} for {max_episodes} episodes")
    print(f"{'='*60}")
    
    # Try to load existing model
    if hasattr(trainer, 'load_best_model'):
        if trainer.load_best_model():
            print("✓ Loaded existing model, continuing training...")
    
    rewards_history = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(1, max_episodes + 1), desc=f"Training {algorithm.upper()}"):
        # Collect experience
        trainer.collect_experience(steps_per_update)
        
        # Update agents
        for agent in trainer.agents:
            stats = agent.update() if hasattr(agent, 'update') else {}
        
        # Get average reward
        avg_reward = np.mean([a.reward_mean for a in trainer.agents])
        rewards_history.append(float(avg_reward))
        
        # Log progress
        if episode % 10 == 0:
            tqdm.write(f"Episode {episode:4d} | Avg Reward: {avg_reward:+8.2f}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            trainer.best_reward = best_reward
            if hasattr(trainer, 'save_model'):
                trainer.save_model()
        
        # Periodic save
        if episode % save_interval == 0:
            if hasattr(trainer, 'save_model'):
                trainer.save_model()
    
    print(f"\n✓ Training complete! Best reward: {best_reward:.2f}")
    return rewards_history, best_reward


def main():
    parser = argparse.ArgumentParser(description="Train Integrated DRL Models")
    
    parser.add_argument('--algorithm', type=str, default='a3c',
                       choices=['a3c', 'a2c', 'ppo', 'dqn', 'sac', 'ddpg', 'vpg'],
                       help="Algorithm to train")
    parser.add_argument('--episodes', type=int, default=500,
                       help="Number of training episodes")
    parser.add_argument('--agents', type=int, default=4,
                       help="Number of parallel agents")
    parser.add_argument('--steps', type=int, default=100,
                       help="Steps per update")
    parser.add_argument('--save_interval', type=int, default=50,
                       help="Save model every N episodes")
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device to use (cuda/cpu)")
    parser.add_argument('--no_icm', action='store_true',
                       help="Disable ICM exploration")
    parser.add_argument('--use_synthetic', action='store_true',
                       help="Use synthetic data instead of real data")
    
    args = parser.parse_args()
    
    print("="*60)
    print("INTEGRATED DRL TRAINING WITH ALL IMPROVEMENTS")
    print("="*60)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.episodes}")
    print(f"Agents: {args.agents}")
    print(f"ICM Exploration: {'Disabled' if args.no_icm else 'Enabled'}")
    print("="*60)
    
    if args.use_synthetic:
        # Use synthetic data for testing
        print("\n→ Using SYNTHETIC data for testing...")
        
        # Create synthetic thermal data with fire spots
        thermal_data = np.random.rand(200, 200).astype(np.float32)
        thermal_data[80:100, 80:100] = 0.95 + 0.05 * np.random.rand(20, 20)
        thermal_data[150:170, 30:50] = 0.90 + 0.10 * np.random.rand(20, 20)
        
        weather_data = {
            'humidity': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
            'wind_speed': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
            'soil_temp': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
            'soil_moisture': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
            'rainfall': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
            'ndmi': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
            'dem': np.random.rand(200, 200).astype(np.float32) * 2 - 1,
        }
        
        landcover_data = np.ones((200, 200), dtype=np.float32)
        
        _, normalized_thermal, _ = prepare_temp_data(thermal_data)
        
    else:
        # Load real data
        print("\n→ Loading REAL data...")
        
        thermal_path = PATHS.get('thermal_tif', 'data/thermal_raster_final.tif')
        landcover_path = PATHS.get('landcover_tif', 'database/aligned_landcover.tif')
        
        # Check if thermal file exists
        if not os.path.exists(thermal_path):
            print(f"ERROR: Thermal file not found: {thermal_path}")
            print("Try running with --use_synthetic flag for testing")
            return
        
        # Load thermal
        print(f"  Loading thermal: {thermal_path}")
        with rasterio.open(thermal_path) as src:
            thermal_data = src.read(1)
        _, normalized_thermal, _ = prepare_temp_data(thermal_data)
        print(f"  ✓ Thermal shape: {thermal_data.shape}")
        
        # Load weather
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
        weather_data = load_weather_data(weather_tifs)
        
        # Load landcover
        if os.path.exists(landcover_path):
            with rasterio.open(landcover_path) as src:
                landcover_data = src.read(1)
            print(f"  ✓ Landcover shape: {landcover_data.shape}")
        else:
            landcover_data = np.ones_like(thermal_data)
            print(f"  ✗ Landcover not found, using ones")
    
    # Create patches
    print("\n→ Creating training patches...")
    patch_list = create_patches(normalized_thermal, weather_data, landcover_data)
    print(f"  ✓ Created {len(patch_list)} valid patches")
    
    if len(patch_list) == 0:
        print("ERROR: No valid patches found!")
        print("Try running with --use_synthetic flag for testing")
        return
    
    # Create environment factory
    env_factory = create_cnn_env_factory(patch_list)
    
    # Create trainer
    trainer = get_trainer(
        algorithm=args.algorithm,
        env_factory=env_factory,
        num_agents=args.agents,
        device=args.device,
        use_icm=not args.no_icm
    )
    
    # Train
    rewards_history, best_reward = train_model(
        trainer=trainer,
        algorithm=args.algorithm,
        max_episodes=args.episodes,
        steps_per_update=args.steps,
        save_interval=args.save_interval,
        device=args.device
    )
    
    # Save training results
    results_dir = f"{args.algorithm}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'algorithm': args.algorithm,
        'best_reward': float(best_reward),
        'episodes': args.episodes,
        'use_icm': not args.no_icm,
        'rewards_history': rewards_history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{results_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/training_results.json")


if __name__ == "__main__":
    main()
