import os
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from typing import Dict, Tuple
from tqdm import tqdm
import gc
import warnings
import sys
import logging
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Import SAC-Discrete implementation
from sac.sac import MultiAgentSACDiscrete

# Import your existing modules
from prepare_data._crop_thermal_true_img import read_and_patch_thermal_image
from prepare_data._prepare_temp_data import prepare_temp_data_balanced
from prepare_data._reconstruct_from_patches import reconstruct_from_patches
from prepare_data._read_and_align_landcover_to_thermal import read_and_align_landcover_to_thermal
from prepare_data._weather_patches import process_weather_patches

from utils.auc_pred_map import compute_auc_and_plot_full
from utils.confusion_map import compute_confusion_overlay
from utils.final_map import export_final_confusion_map
from utils.save_tif_confusion import save_tif_confusion
from utils.save_combine_confusion import save_combine_confusion
from utils.compute_mse_corr import compute_mse_corr

from environment.env_src import create_enhanced_crop_thermal_env

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def setup_dual_logging():
    """Setup logging that actually writes to files"""
    os.makedirs('logs', exist_ok=True)
    
    # Clear existing handlers
    logging.getLogger().handlers = []
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training logger
    train_logger = logging.getLogger('Training_SAC')
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    
    # Console handler for training
    train_console = logging.StreamHandler()
    train_console.setLevel(logging.INFO)
    train_format = logging.Formatter('[SAC-TRAIN] %(message)s')
    train_console.setFormatter(train_format)
    train_logger.addHandler(train_console)
    
    # File handler for training
    train_file_path = f'logs/sac_training_{timestamp}.log'
    train_file = logging.FileHandler(train_file_path, mode='w')
    train_file.setLevel(logging.DEBUG)
    train_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    train_logger.addHandler(train_file)
    
    # Evaluation logger  
    eval_logger = logging.getLogger('SAC_Evaluation')
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    
    eval_file_path = f'logs/sac_evaluation_{timestamp}.log'
    eval_file = logging.FileHandler(eval_file_path, mode='w')
    eval_file.setLevel(logging.INFO)
    eval_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    eval_logger.addHandler(eval_file)
    
    # Write initial messages to confirm files are working
    train_logger.info("SAC Training Logger Initialized")
    eval_logger.info("SAC Evaluation Logger Initialized")
    
    print(f"Log files created: {train_file_path} and {eval_file_path}")
    
    return train_logger, eval_logger

def plot_sac_training_progress(rewards_history, save_path="sac_training_plots", 
                               training_stats=None):
    """Plot training metrics ONLY after training is complete"""
    os.makedirs(save_path, exist_ok=True)
    
    if not rewards_history:
        print("No rewards history to plot")
        return
    
    print("\nGenerating final training plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training rewards with smoothing
    plt.subplot(2, 3, 1)
    plt.plot(rewards_history, color='#2E86C1', linewidth=1, alpha=0.5, label='Raw Rewards')
    if len(rewards_history) > 10:
        # Add smoothed line
        window = min(20, len(rewards_history) // 5)
        smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), smoothed, color='#E74C3C', 
                linewidth=2, label=f'Smoothed (window={window})')
    plt.title('Training Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if training_stats and len(training_stats) > 0:
        # Plot 2: Policy Loss
        plt.subplot(2, 3, 2)
        policy_losses = [stats.get('policy_loss', 0) for stats in training_stats if stats]
        if policy_losses:
            plt.plot(policy_losses, color='#27AE60', linewidth=2)
            plt.title('Policy Loss')
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Q-Network Losses
        plt.subplot(2, 3, 3)
        q1_losses = [stats.get('q1_loss', 0) for stats in training_stats if stats]
        q2_losses = [stats.get('q2_loss', 0) for stats in training_stats if stats]
        if q1_losses:
            plt.plot(q1_losses, color='#E74C3C', linewidth=2, alpha=0.7, label='Q1 Loss')
            plt.plot(q2_losses, color='#3498DB', linewidth=2, alpha=0.7, label='Q2 Loss')
            plt.title('Q-Network Losses')
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Entropy
        plt.subplot(2, 3, 4)
        entropy_values = [stats.get('entropy', 0) for stats in training_stats if stats]
        if entropy_values:
            plt.plot(entropy_values, color='#8E44AD', linewidth=2)
            plt.title('Policy Entropy')
            plt.xlabel('Episodes')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Alpha (Temperature)
        plt.subplot(2, 3, 5)
        alpha_values = [stats.get('alpha', 0.2) for stats in training_stats if stats]
        if alpha_values:
            plt.plot(alpha_values, color='#F39C12', linewidth=2)
            plt.title('Temperature Parameter (α)')
            plt.xlabel('Episodes')
            plt.ylabel('Alpha')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Episodes Completed
        plt.subplot(2, 3, 6)
        episodes = [stats.get('episodes_completed', 0) for stats in training_stats if stats]
        if episodes:
            plt.plot(episodes, color='#16A085', linewidth=2)
            plt.title('Cumulative Episodes Completed')
            plt.xlabel('Training Steps')
            plt.ylabel('Total Episodes')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'sac_final_training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to: {plot_path}")

def create_prediction_map_from_sac(trainer, thermal_data, start_pos, weather_patches, 
                                   landcover_data, max_steps=500):
    """Generate prediction map using SAC agent"""
    env = create_enhanced_crop_thermal_env(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        verbose=False
    )
    
    agent = trainer.agents[0]
    agent.policy.eval()
    agent.q1.eval()
    agent.q2.eval()
    
    V = np.zeros_like(thermal_data)
    fire_prob_map = np.zeros_like(thermal_data)
    
    positions = [(i, j) for i in range(thermal_data.shape[0]) 
                 for j in range(thermal_data.shape[1])]
    
    batch_size = 1024 if agent.device == 'cuda' else 512  # Optimized for GPU
    
    with torch.no_grad():
        for batch_start in range(0, len(positions), batch_size):
            batch_end = min(batch_start + batch_size, len(positions))
            batch_positions = positions[batch_start:batch_end]
            
            batch_states = []
            for pos in batch_positions:
                state = env._get_observation(pos)
                batch_states.append(state)
            
            states_tensor = torch.FloatTensor(np.array(batch_states)).to(agent.device)
            
            # Get action probabilities from policy
            probs = agent.policy(states_tensor)
            
            # Get Q-values
            q1_values = agent.q1(states_tensor)
            q2_values = agent.q2(states_tensor)
            min_q_values = torch.min(q1_values, q2_values)
            
            # Calculate state values
            values = (probs * min_q_values).sum(dim=-1)
            
            for idx, (i, j) in enumerate(batch_positions):
                V[i, j] = values[idx].item()
                
                # Fire detection probability (action 5)
                fire_prob = probs[idx, 5].item()
                
                local_temp = thermal_data[i, j]
                base_threshold = 0.8 if local_temp < env.medium_temp_threshold else 0.7
                
                if landcover_data[i, j] != 1:
                    base_threshold += 0.2
                
                fire_prob_map[i, j] = fire_prob if fire_prob > base_threshold else 0.0
    
    env.close()
    return V, fire_prob_map

def process_single_patch_sac(patch_data, trainer, step, transform_affine, 
                             src_crs, patch_coords, original_shape):
    """Process a single patch using SAC"""
    eval_logger = logging.getLogger('SAC_Evaluation')
    
    try:
        i = patch_data['index']
        eval_logger.info(f"Processing patch {i+1}")
        
        # Generate prediction map using SAC
        V, _ = create_prediction_map_from_sac(
            trainer,
            patch_data['thermal_data'],
            patch_data['start_pos'],
            patch_data['weather_patches'],
            patch_data['landcover_data'],
            max_steps=200
        )
        
        # Suppress prints from utils
        with SuppressPrints():
            auc, pred_map = compute_auc_and_plot_full(
                grid_values=None,
                fire_ground_truth_binary=patch_data['y_true_binary'],
                fire_ground_truth_continuous=patch_data['thermal_data'],
                crop_id=i + 1,
                pred_map=V,
                no_value_iteration=False
            )
            
            overlay, precision, recall, f1, pr_auc = compute_confusion_overlay(
                pred_map, patch_data['y_true_binary'], i
            )
            
            save_tif_confusion(
                original_shape=original_shape,
                step=step,
                i=i,
                transform_affine=transform_affine,
                overlay=overlay,
                src_crs=src_crs
            )
        
        plt.close('all')  # Free memory
        
        lat_dms, lon_dms, lat_dms_hotspot, lon_dms_hotspot = save_combine_confusion(
            i, patch_coords, patch_data['start_pos'], overlay, patch_data['T_Celsius']
        )
        
        mse, corr = compute_mse_corr(pred_map, patch_data['thermal_data'])
        
        eval_logger.info(f"Patch {i+1} completed: AUC={auc:.4f}, F1={f1:.4f}")
        
        return {
            'success': True,
            'Patch': i + 1,
            'Lat': lat_dms,
            'Lon': lon_dms,
            'lat_hotspot': lat_dms_hotspot,
            'lon_hotspot': lon_dms_hotspot,
            'Max_Temp': float(np.max(patch_data['T_Celsius'])),
            'AUC': float(auc),
            'MSE': float(mse),
            'Pearson_Correlation': float(corr),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'PR_AUC': float(pr_auc),
        }
    except Exception as e:
        eval_logger.error(f"Error processing patch {patch_data.get('index', 'unknown')}: {str(e)}")
        return {'success': False, 'index': patch_data.get('index', -1), 'error': str(e)}


def enhanced_sac_main_with_real_data(thermal_path: str,
                                     landcover_path: str,
                                     weather_tifs: dict,
                                     alignment_method='match_pixels',
                                     num_workers=4,
                                     max_episodes=100,
                                     device='cpu',
                                     save_interval=10,
                                     steps_per_update=1000,
                                     force_retrain=False,
                                     use_parallel_processing=True):
    """Main function for SAC-Discrete training with real wildfire data"""
    
    train_logger, eval_logger = setup_dual_logging()
    
    print("="*80)
    print("SAC-DISCRETE TRAINING FOR WILDFIRE DETECTION")
    print(f"Training log: logs/sac_training_*.log")
    print(f"Evaluation log: logs/sac_evaluation_*.log")
    print("="*80)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Algorithm: SAC-Discrete (Soft Actor-Critic)")
    print(f"Number of workers: {num_workers}")
    print(f"Maximum episodes: {max_episodes}")
    print(f"Steps per update: {steps_per_update}")
    
    # Create directories
    result_dirs = ["sac_results", "sac_models", "sac_training_plots"]
    for folder in result_dirs:
        os.makedirs(folder, exist_ok=True)
    
    # Read weather data
    weather_data = {}
    for name, path in weather_tifs.items():
        with rasterio.open(path) as src:
            weather_data[name] = src.read(1)
    
    print("\n-> Reading and patching thermal image...")
    patch_size = 100
    overlap = 10
    patches, original_shape, patch_coords, transform_affine, src_crs = read_and_patch_thermal_image(
        thermal_path, patch_size, overlap)
    
    print("\n-> Aligning landcover with thermal data...")
    landcover_patches = read_and_align_landcover_to_thermal(
        landcover_path, thermal_path, patch_size, overlap, method=alignment_method)
    
    if len(landcover_patches) > 0:
        full_aligned = reconstruct_from_patches(landcover_patches, original_shape, patch_size, overlap)
    
    # Prepare patches
    patch_list = []
    valid_indices = []
    
    for i, (T_Celsius, landcover_patch, coord) in enumerate(zip(patches, landcover_patches, patch_coords)):
        T_Celsius, normalized_temp, y_true_binary, _, _ = prepare_temp_data_balanced(T_Celsius)
        
        if (
            np.any(T_Celsius <= 0.0) or 
            np.sum(y_true_binary) == 0 or
            np.sum(y_true_binary) == y_true_binary.size):
            continue
        
        valid_indices.append(i)
        weather_patches = process_weather_patches(weather_data, coord)
        
        patch_list.append({
            'index': i,
            'thermal_data': normalized_temp,
            'start_pos': np.unravel_index(np.argmax(T_Celsius), T_Celsius.shape),
            'weather_patches': weather_patches,
            'landcover_data': landcover_patch,
            'y_true_binary': y_true_binary,
            'T_Celsius': T_Celsius,
            'coord': coord,
            'fire_ratio': np.sum(y_true_binary) / y_true_binary.size
        })
    
    print(f"Created {len(patch_list)} valid patches for training")
    train_logger.info(f"Total valid patches: {len(patch_list)}")
    
    # Split data
    num_patches = len(patch_list)
    num_val = max(1, int(num_patches * 0.1))
    val_indices = np.random.choice(num_patches, num_val, replace=False)
    train_patches = [p for i, p in enumerate(patch_list) if i not in val_indices]
    
    def env_creator():
        """Create environment with weighted sampling"""
        weights = [1.0 / (p['fire_ratio'] + 0.1) for p in train_patches]
        patch = np.random.choice(train_patches, p=np.array(weights)/np.sum(weights))
        
        return create_enhanced_crop_thermal_env(
            thermal_data=patch['thermal_data'],
            start_pos=patch['start_pos'],
            weather_patches=patch['weather_patches'],
            landcover_data=patch['landcover_data'],
            max_steps=min(100, patch['thermal_data'].size // 10),
            verbose=False
        )
    
    # Get dimensions
    temp_env = env_creator()
    initial_obs = temp_env.reset()
    state_size = len(initial_obs)
    action_size = temp_env.action_space.n
    temp_env.close()
    
    print(f"State size: {state_size}, Action size: {action_size}")
    train_logger.info(f"Environment dimensions - State: {state_size}, Action: {action_size}")
    
    # Initialize SAC trainer with fixed hyperparameters
    trainer = MultiAgentSACDiscrete(
        env_factory=env_creator,
        num_agents=num_workers,
        state_size=state_size,
        action_size=action_size,
        device=device,
        lr=1e-4,  # Reduced from 3e-4
        gamma=0.99,
        tau=0.001,  # Reduced from 0.005 for slower target updates
        alpha=0.1,  # Start with smaller alpha
        automatic_entropy_tuning=True,
        target_update_interval=1,
        replay_buffer_size=100000
    )
    
    # Check if model exists and should be loaded
    model_exists = os.path.exists(trainer.best_model_path)
    
    if model_exists and not force_retrain:
        print(f"\nFound existing model at {trainer.best_model_path}")
        model_loaded = trainer.load_best_model()
        
        if model_loaded:
            print("Successfully loaded model. Starting evaluation...")
            eval_logger.info("Starting model evaluation")
            
            step = patch_size - overlap
            all_results = []
            
            if use_parallel_processing:
                print("\n-> Using PARALLEL PROCESSING for evaluation...")
                optimal_workers = min(os.cpu_count() - 1, 8, len(patch_list) // 10 + 1)
                
                with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    futures = []
                    for patch in patch_list:
                        future = executor.submit(
                            process_single_patch_sac,
                            patch, trainer, step, transform_affine,
                            src_crs, patch_coords, original_shape
                        )
                        futures.append(future)
                    
                    with tqdm(total=len(patch_list), desc="Evaluating patches") as pbar:
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=60)
                                if result['success']:
                                    all_results.append(result)
                            except Exception as e:
                                eval_logger.error(f"Error in parallel processing: {str(e)}")
                            finally:
                                pbar.update(1)
                                if len(all_results) % 100 == 0:
                                    gc.collect()
            else:
                print("\n-> Using SEQUENTIAL processing for evaluation...")
                for patch in tqdm(patch_list, desc="Evaluating patches"):
                    result = process_single_patch_sac(
                        patch, trainer, step, transform_affine,
                        src_crs, patch_coords, original_shape
                    )
                    if result['success']:
                        all_results.append(result)
                    
                    if len(all_results) % 100 == 0:
                        gc.collect()
            
            # Export final confusion map
            if all_results:
                try:
                    print("\n-> Exporting final confusion map...")
                    export_final_confusion_map(patch_size, overlap, step, thermal_path)
                    print("[OK] Final confusion map exported successfully")
                    eval_logger.info("Final confusion map exported")
                except Exception as e:
                    print(f"[WARNING] Could not export final confusion map: {str(e)}")
                    eval_logger.error(f"Failed to export final confusion map: {str(e)}")
                
                # Save results
                df_results = pd.DataFrame(all_results)
                
                numeric_cols = ['AUC', 'MSE', 'Pearson_Correlation', 'Precision', 'Recall', 'F1', 'PR_AUC']
                avg_metrics = df_results[numeric_cols].mean()
                
                summary_row = {col: float(avg_metrics[col]) for col in numeric_cols}
                summary_row.update({
                    'Patch': 'Average',
                    'Lat': '', 'Lon': '',
                    'lat_hotspot': '', 'lon_hotspot': '',
                    'Max_Temp': float(df_results['Max_Temp'].mean())
                })
                
                df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
                
                csv_path = "sac_results/sac_results_summary.csv"
                df_results.to_csv(csv_path, index=False)
                
                print(f"\n{'='*60}")
                print(f"SAC EVALUATION RESULTS ({len(all_results)} patches)")
                print(f"{'='*60}")
                print(f"Average AUC: {avg_metrics['AUC']:.4f}")
                print(f"Average Precision: {avg_metrics['Precision']:.4f}")
                print(f"Average Recall: {avg_metrics['Recall']:.4f}")
                print(f"Average F1: {avg_metrics['F1']:.4f}")
                print(f"Average PR-AUC: {avg_metrics['PR_AUC']:.4f}")
                print(f"Average MSE: {avg_metrics['MSE']:.4f}")
                print(f"Average Correlation: {avg_metrics['Pearson_Correlation']:.4f}")
                print(f"{'='*60}")
                print(f"Results saved to: {csv_path}")
                
                eval_logger.info(f"Evaluation complete. Average metrics: {avg_metrics.to_dict()}")
                
                return df_results, trainer
            else:
                print("[ERROR] No patches were successfully processed")
                return None, trainer
        else:
            print("Failed to load model. Starting training from scratch...")
            train_logger.warning("Model loading failed, starting new training")
    
    # Training mode (either no model exists or force_retrain=True)
    print("\n" + "="*60)
    print("STARTING SAC-DISCRETE TRAINING")
    print("="*60)
    train_logger.info("Starting SAC training")
    
    rewards_history = []
    training_stats_history = []
    best_reward = -float('inf')
    patience_counter = 0
    max_patience = 20  # Stop if no improvement for 20 episodes
    
    print(f"Training for {max_episodes} episodes")
    print(f"Replay buffer will be used for sample efficiency")
    print("-"*60)
    
    for episode in range(1, max_episodes + 1):
        print(f"\rEpisode {episode}/{max_episodes}", end="")
        
        # Collect experience and update
        trainer.collect_experience(steps_per_update)
        
        # Get statistics from all agents
        episode_rewards = []
        episode_stats = {}
        
        for agent in trainer.agents:
            if agent.episodes_completed > 0:
                episode_rewards.append(agent.reward_mean)
                
                # Get update statistics with reduced frequency
                stats = agent.update(batch_size=128, updates_per_step=1)  # Reduced from 256/10
                for key, value in stats.items():
                    if key not in episode_stats:
                        episode_stats[key] = []
                    episode_stats[key].append(value)
        
        # Average stats across agents
        if episode_stats:
            episode_stats = {key: np.mean(values) for key, values in episode_stats.items()}
            # Store update stats in agent for later retrieval
            for agent in trainer.agents:
                agent.last_update_stats = episode_stats
        
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            rewards_history.append(avg_reward)
            training_stats_history.append(episode_stats)
            
            print(f" | Reward: {avg_reward:+.2f}", end="")
            
            if episode_stats:
                print(f" | α: {episode_stats.get('alpha', 0.2):.3f}", end="")
                print(f" | Entropy: {episode_stats.get('entropy', 0):.3f}", end="")
            
            train_logger.info(f"Episode {episode}: Reward={avg_reward:.4f}, Stats={episode_stats}")
            
            # Save best model and check patience
            if avg_reward > trainer.best_reward:
                trainer.best_reward = avg_reward
                trainer.save_model()
                print(" [BEST]", end="")
                train_logger.info(f"New best model saved with reward: {avg_reward:.4f}")
                patience_counter = 0  # Reset patience
            else:
                patience_counter += 1
            
            # Early stopping check
            if episode > 20 and avg_reward < best_reward - 500:
                print(f"\n[WARNING] Performance degrading significantly!")
                print(f"Best reward: {best_reward:.2f}, Current: {avg_reward:.2f}")
                print("Stopping training to prevent further degradation.")
                train_logger.warning(f"Early stopping triggered at episode {episode}")
                break
            
            # Check patience limit
            if patience_counter >= max_patience:
                print(f"\n[INFO] No improvement for {max_patience} episodes. Stopping.")
                train_logger.info(f"Training stopped due to patience limit at episode {episode}")
                break
            
            # Update best reward tracking
            if avg_reward > best_reward:
                best_reward = avg_reward
        
        # Progress update every 10 episodes
        if episode % 10 == 0:
            print()  # New line for clean display
            buffer_size = len(trainer.agents[0].memory) if trainer.agents else 0
            total_episodes = sum(a.episodes_completed for a in trainer.agents)
            print(f"  Buffer size: {buffer_size} | Total episodes: {total_episodes}")
            train_logger.info(f"Checkpoint - Buffer: {buffer_size}, Episodes: {total_episodes}")
        
        # Save checkpoint models
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"sac_models/sac_checkpoint_episode_{episode}.pth"
            trainer.save_model(checkpoint_path)
            train_logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    print()  # Final newline
    
    # Generate final plots ONLY after training is complete
    plot_sac_training_progress(
        rewards_history,
        save_path="sac_training_plots",
        training_stats=training_stats_history
    )
    
    # Save training history
    training_history = {
        'rewards': rewards_history,
        'stats': training_stats_history,
        'best_reward': float(trainer.best_reward),
        'total_episodes': max_episodes
    }
    
    history_path = 'sac_models/sac_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SAC TRAINING COMPLETED")
    print(f"Final average reward: {rewards_history[-1] if rewards_history else 'N/A':.2f}")
    print(f"Best reward achieved: {trainer.best_reward:.2f}")
    print(f"Training history saved to: {history_path}")
    print(f"Best model saved to: {trainer.best_model_path}")
    print(f"{'='*60}")
    
    train_logger.info(f"Training complete. Best reward: {trainer.best_reward:.4f}")
    
    return None, trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAC-Discrete Training and Evaluation for Wildfire Detection")
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'eval', 'both'],
                       help="Mode: train, eval, or both (default: both)")
    parser.add_argument('--episodes', type=int, default=100,
                       help="Number of training episodes (default: 100)")
    parser.add_argument('--device', type=str, default='auto',
                       help="Device: cuda, cpu, or auto (default: auto)")
    parser.add_argument('--workers', type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    parser.add_argument('--steps', type=int, default=1000,
                       help="Steps per update (default: 1000)")
    parser.add_argument('--save_interval', type=int, default=20,
                       help="Save model every N episodes (default: 20)")
    parser.add_argument('--force_retrain', action='store_true',
                       help="Force retraining even if model exists")
    
    args = parser.parse_args()
    
    # Import centralized configuration
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from config import get_config, get_thermal_path, get_landcover_path, get_weather_tifs
        config = get_config()
        
        thermal_path = get_thermal_path()
        landcover_path = get_landcover_path()
        weather_tifs = get_weather_tifs()
        
        print("Using centralized configuration from config.py")
        
    except ImportError:
        print("[WARNING] config.py not found, using fallback paths")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        thermal_path = os.path.join(base_dir, 'data/thermal_raster_final.tif')
        landcover_path = os.path.join(base_dir, 'database/aligned_landcover.tif')
        
        weather_tifs = {
            'soil_moisture': os.path.join(base_dir, 'database/aligned_soil_moisture.tif'),
            'rainfall': os.path.join(base_dir, 'database/aligned_rainfall.tif'),
            'soil_temp': os.path.join(base_dir, 'database/aligned_soil_temp.tif'),
            'wind_speed': os.path.join(base_dir, 'database/aligned_wind_speed.tif'),
            'humidity': os.path.join(base_dir, 'database/aligned_humidity.tif'),
            'dem': os.path.join(base_dir, 'database/aligned_dem.tif'),
            'ndmi': os.path.join(base_dir, 'database/aligned_ndmi.tif')
        }

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*60)
    print("SAC-DISCRETE WILDFIRE DETECTION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Workers: {args.workers}")
    print(f"Steps per update: {args.steps}")
    print("="*60)
    
    try:
        results, trainer = enhanced_sac_main_with_real_data(
            weather_tifs=weather_tifs,
            thermal_path=thermal_path,
            landcover_path=landcover_path,
            alignment_method='match_pixels',
            num_workers=args.workers,
            max_episodes=args.episodes,
            device=device,
            save_interval=args.save_interval,
            steps_per_update=args.steps,
            force_retrain=args.force_retrain,
            use_parallel_processing=True
        )
        
        print("\nSAC-Discrete completed successfully!")
        
    except Exception as e:
        print(f"\nError during SAC execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up resources...")
        if 'trainer' in locals():
            for env in trainer.envs:
                try:
                    env.close()
                except:
                    pass
        print("Done!")