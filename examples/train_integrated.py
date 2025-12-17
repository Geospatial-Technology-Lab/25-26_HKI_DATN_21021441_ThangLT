"""
Example Training Script demonstrating all 3 improvements:
1. Balanced Rewards (automatic via CNN environment)
2. CNN Observation (8x11x11 spatial features)
3. ICM Exploration (intrinsic curiosity rewards)

Usage:
    python examples/train_integrated.py --algorithm a3c --use_icm --device cuda
"""
import os
import sys
import argparse
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import integrated agents
from a3c.integrated_a3c import IntegratedMultiAgentA3C
from dqn.integrated_dqn import IntegratedMultiAgentDQN
from ppo.integrated_ppo import IntegratedMultiAgentPPO

# Import CNN environment
from environment.cnn_env import create_cnn_crop_thermal_env


def create_example_env(use_random_data=True):
    """
    Create example environment for testing.
    In production, replace with real thermal and weather data.
    """
    if use_random_data:
        # Create synthetic data for testing
        thermal = np.random.rand(100, 100).astype(np.float32)
        # Add some "fire" hotspots
        thermal[40:50, 40:50] = 0.95 + 0.05 * np.random.rand(10, 10)
        thermal[70:80, 20:30] = 0.90 + 0.10 * np.random.rand(10, 10)
        
        weather = {
            'humidity': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
            'wind_speed': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
            'soil_temp': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
            'soil_moisture': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
            'rainfall': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
            'ndmi': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
            'dem': np.random.rand(100, 100).astype(np.float32) * 2 - 1,
        }
        
        landcover = np.ones((100, 100), dtype=np.float32)
        start_pos = (45, 45)  # Start near a hotspot
    else:
        # Load real data (implement your data loading here)
        raise NotImplementedError("Implement real data loading")
    
    return create_cnn_crop_thermal_env(
        thermal_data=thermal,
        start_pos=start_pos,
        weather_patches=weather,
        landcover_data=landcover,
        max_steps=500,
        patch_size=11,
        verbose=False
    )


def train_integrated_agent(algorithm='a3c', num_agents=4, num_episodes=100,
                           steps_per_update=100, device='cpu', use_icm=True):
    """
    Train agent with all integrated improvements.
    
    Args:
        algorithm: 'a3c', 'dqn', or 'ppo'
        num_agents: Number of parallel agents
        num_episodes: Training episodes
        steps_per_update: Steps per training update
        device: 'cpu' or 'cuda'
        use_icm: Whether to use ICM exploration
    """
    print("=" * 60)
    print(f"Training Integrated {algorithm.upper()} Agent")
    print("=" * 60)
    print(f"Improvements enabled:")
    print(f"  ✓ Balanced Rewards (FP penalty 50, FN penalty 100)")
    print(f"  ✓ CNN Observation (8 channels, 11x11 patch)")
    print(f"  ✓ ICM Exploration: {'Enabled' if use_icm else 'Disabled'}")
    print(f"Device: {device}")
    print(f"Agents: {num_agents}")
    print("=" * 60)
    
    # Environment factory
    env_factory = lambda: create_example_env(use_random_data=True)
    
    # Create trainer based on algorithm
    if algorithm == 'a3c':
        trainer = IntegratedMultiAgentA3C(
            env_factory=env_factory,
            num_agents=num_agents,
            num_channels=8,
            patch_size=11,
            action_size=6,
            device=device,
            lr=1e-4,
            gamma=0.99,
            use_icm=use_icm
        )
    elif algorithm == 'dqn':
        trainer = IntegratedMultiAgentDQN(
            env_factory=env_factory,
            num_agents=num_agents,
            num_channels=8,
            patch_size=11,
            action_size=6,
            device=device,
            use_icm=use_icm,
            lr=1e-4,
            gamma=0.99
        )
    elif algorithm == 'ppo':
        trainer = IntegratedMultiAgentPPO(
            env_factory=env_factory,
            num_agents=num_agents,
            num_channels=8,
            patch_size=11,
            action_size=6,
            device=device,
            use_icm=use_icm,
            lr=3e-4,
            gamma=0.99
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Training loop
    best_reward = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        # Collect experience
        trainer.collect_experience(steps_per_update)
        
        # Update agents and get stats
        all_stats = []
        for agent in trainer.agents:
            stats = agent.update()
            if stats:
                all_stats.append(stats)
        
        # Aggregate stats
        if all_stats:
            avg_reward = np.mean([s.get('avg_episode_reward', 0) for s in all_stats])
            avg_loss = np.mean([s.get('total_loss', 0) for s in all_stats])
            avg_icm_loss = np.mean([s.get('icm_loss', 0) for s in all_stats]) if use_icm else 0
            
            print(f"Episode {episode:4d} | Reward: {avg_reward:+8.2f} | "
                  f"Loss: {avg_loss:.4f} | ICM: {avg_icm_loss:.4f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                trainer.best_reward = best_reward
                trainer.save_model()
                print(f"  └── New best model saved! (reward: {best_reward:.2f})")
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Best Reward: {best_reward:.2f}")
    print("=" * 60)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train Integrated DRL Agent")
    parser.add_argument('--algorithm', type=str, default='a3c',
                        choices=['a3c', 'dqn', 'ppo'],
                        help="Algorithm to use")
    parser.add_argument('--num_agents', type=int, default=4,
                        help="Number of parallel agents")
    parser.add_argument('--num_episodes', type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument('--steps_per_update', type=int, default=100,
                        help="Steps per training update")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use")
    parser.add_argument('--use_icm', action='store_true', default=True,
                        help="Use ICM exploration")
    parser.add_argument('--no_icm', action='store_true',
                        help="Disable ICM exploration")
    
    args = parser.parse_args()
    
    use_icm = args.use_icm and not args.no_icm
    
    train_integrated_agent(
        algorithm=args.algorithm,
        num_agents=args.num_agents,
        num_episodes=args.num_episodes,
        steps_per_update=args.steps_per_update,
        device=args.device,
        use_icm=use_icm
    )


if __name__ == "__main__":
    main()
