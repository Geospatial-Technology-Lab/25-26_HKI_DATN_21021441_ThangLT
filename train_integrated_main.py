"""
Training Script for Integrated DRL Models with Real Data
Supports all 7 algorithms: A3C, A2C, PPO, DQN, SAC, DDPG, VPG

Usage:
    python train_integrated_main.py --algorithm a3c --episodes 500 --device cuda
    python train_integrated_main.py --algorithm ppo --episodes 1000 --device cuda --no_icm

All algorithms use:
1. CNN-based observation (8x11x11 spatial features)
2. ICM curiosity-driven exploration (optional)
3. Balanced rewards
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

# Import utilities
from utils.normalize import NormalizeData
from utils.process_weather import process_weather_patches
from utils.prepare_temp import prepare_temp_data_balanced
from utils.thermal_reader import read_and_patch_thermal_image
from utils.landcover_reader import read_and_align_landcover_to_thermal


def load_weather_data(weather_tifs: dict) -> dict:
    """Load weather TIF files"""
    weather_data = {}
    for name, path in weather_tifs.items():
        if os.path.exists(path):
            with rasterio.open(path) as src:
                weather_data[name] = src.read(1)
            print(f"  ✓ Loaded {name}: {weather_data[name].shape}")
        else:
            print(f"  ✗ Missing {name}: {path}")
    return weather_data


def prepare_training_patches(thermal_path: str, landcover_path: str, weather_data: dict,
                             patch_size: int = 100, overlap: int = 10):
    """Prepare training patches from thermal and weather data"""
    
    print("\n→ Reading thermal patches...")
    patches, original_shape, patch_coords, transform_affine, src_crs = read_and_patch_thermal_image(
        thermal_path, patch_size, overlap
    )
    print(f"  Created {len(patches)} thermal patches")
    
    print("\n→ Aligning landcover...")
    landcover_patches = read_and_align_landcover_to_thermal(
        landcover_path, thermal_path, patch_size, overlap
    )
    
    print("\n→ Preparing valid patches...")
    patch_list = []
    
    for i, (T_Celsius, landcover_patch, coord) in enumerate(zip(patches, landcover_patches, patch_coords)):
        T_Celsius, normalized_temp, y_true_binary, _, _ = prepare_temp_data_balanced(T_Celsius)
        
        # Skip invalid patches
        if np.any(T_Celsius <= 0.0) or np.sum(y_true_binary) == 0:
            continue
        
        weather_patches = process_weather_patches(weather_data, coord)
        
        patch_list.append({
            'index': i,
            'thermal_data': normalized_temp,
            'start_pos': np.unravel_index(np.argmax(T_Celsius), T_Celsius.shape),
            'weather_patches': weather_patches,
            'landcover_data': landcover_patch,
            'y_true_binary': y_true_binary,
            'fire_ratio': np.sum(y_true_binary) / y_true_binary.size,
            'coord': coord
        })
    
    print(f"  Created {len(patch_list)} valid patches")
    return patch_list, original_shape, transform_affine, src_crs


def create_cnn_env_factory(patch_list, patch_size=11):
    """Create factory function for CNN environment"""
    
    def env_factory():
        # Weighted sampling - prefer patches with more fire pixels
        weights = [1.0 / (p['fire_ratio'] + 0.1) for p in patch_list]
        weights = np.array(weights) / np.sum(weights)
        patch = np.random.choice(patch_list, p=weights)
        
        return CNNCropThermalEnv(
            thermal_data=patch['thermal_data'],
            start_pos=patch['start_pos'],
            weather_patches=patch['weather_patches'],
            landcover_data=patch['landcover_data'],
            max_steps=min(200, patch['thermal_data'].size // 10),
            patch_size=patch_size,
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
        rewards_history.append(avg_reward)
        
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
    
    # Prepare patches
    thermal_path = PATHS.get('thermal_tif', 'data/thermal_raster_final.tif')
    landcover_path = PATHS.get('landcover_tif', 'database/aligned_landcover.tif')
    
    patch_list, original_shape, transform_affine, src_crs = prepare_training_patches(
        thermal_path, landcover_path, weather_data
    )
    
    if len(patch_list) == 0:
        print("ERROR: No valid patches found!")
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
        'best_reward': best_reward,
        'episodes': args.episodes,
        'use_icm': not args.no_icm,
        'rewards_history': rewards_history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{results_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n✓ Results saved to {results_dir}/training_results.json")


if __name__ == "__main__":
    main()
