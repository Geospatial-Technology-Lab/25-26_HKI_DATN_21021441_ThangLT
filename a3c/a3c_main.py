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
from datetime import datetime

warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import data preparation modules
from prepare_data._crop_thermal_true_img import read_and_patch_thermal_image
from prepare_data._prepare_temp_data import prepare_temp_data_balanced
from prepare_data._reconstruct_from_patches import reconstruct_from_patches
from prepare_data._read_and_align_landcover_to_thermal import read_and_align_landcover_to_thermal
from prepare_data._weather_patches import process_weather_patches

# Import utils
from utils.auc_pred_map import compute_auc_and_plot_full
from utils.confusion_map import compute_confusion_overlay
from utils.final_map import export_final_confusion_map
from utils.save_tif_confusion import save_tif_confusion
from utils.save_combine_confusion import save_combine_confusion
from utils.compute_mse_corr import compute_mse_corr
from utils.convert import convert_to_serializable

# Import environment
from environment.env_src import create_enhanced_crop_thermal_env  

# Import IMPROVED A3C classes
from a3c.a3c import ImprovedMultiAgentA3C


# Disable console logging
logging.getLogger().handlers = []
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('A3C_Evaluation').propagate = False

# Safe print function
_real_print = print
def safe_print(*args, **kwargs):
    try:
        text = ' '.join(str(arg) for arg in args)
        replacements = {
            '✓': '[OK]',
            '→': '->',
            '⚠': '[WARNING]',
            '×': 'x',
            '•': '*'
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        _real_print(text, **kwargs)
    except Exception as e:
        try:
            text = ' '.join(str(arg).encode('ascii', 'ignore').decode('ascii') for arg in args)
            _real_print(text, **kwargs)
        except:
            _real_print("[Print error]", **kwargs)

print = safe_print

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def setup_dual_logging():
    """Setup separate loggers for training and evaluation"""
    os.makedirs('logs', exist_ok=True)
    logging.getLogger().handlers = []
    
    # Training logger
    train_logger = logging.getLogger('Training')
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    
    train_console = logging.StreamHandler()
    train_console.setLevel(logging.INFO)
    train_format = logging.Formatter('[TRAIN] %(message)s')
    train_console.setFormatter(train_format)
    train_logger.addHandler(train_console)
    
    train_file = logging.FileHandler(
        f'logs/training_a3c_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    train_file.setLevel(logging.DEBUG)
    train_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    train_logger.addHandler(train_file)
    
    # Evaluation logger
    eval_logger = logging.getLogger('A3C_Evaluation')
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    
    eval_file = logging.FileHandler(
        f'logs/evaluation_a3c_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    eval_file.setLevel(logging.INFO)
    eval_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    eval_logger.addHandler(eval_file)
    
    return train_logger, eval_logger

def plot_training_progress_a3c(rewards_history, save_path="a3c_training_plots", 
                                training_stats=None):
    """Plot A3C training progress"""
    os.makedirs(save_path, exist_ok=True)
    
    if not rewards_history:
        print("No rewards history to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training rewards
    plt.subplot(2, 3, 1)
    plt.plot(rewards_history, color='#2E86C1', linewidth=2, label='Training Rewards')
    plt.title('A3C Training Episode Rewards')
    plt.xlabel('Updates')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    if training_stats and len(training_stats) > 0:
        # Plot 2: Entropy
        plt.subplot(2, 3, 2)
        entropy_values = [stats.get('entropy', 0) for stats in training_stats]
        if entropy_values:
            plt.plot(entropy_values, color='#27AE60', linewidth=2)
            plt.title('Entropy (Exploration)')
            plt.xlabel('Updates')
            plt.ylabel('Entropy')
            plt.grid(True)
        
        # Plot 3: Total Loss
        plt.subplot(2, 3, 3)
        total_losses = [stats.get('total_loss', 0) for stats in training_stats]
        if total_losses:
            plt.plot(total_losses, color='#E74C3C', linewidth=2)
            plt.title('Total Loss')
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.grid(True)
        
        # Plot 4: Actor vs Critic Loss
        plt.subplot(2, 3, 4)
        actor_losses = [stats.get('actor_loss', 0) for stats in training_stats]
        critic_losses = [stats.get('critic_loss', 0) for stats in training_stats]
        if actor_losses and critic_losses:
            plt.plot(actor_losses, color='#8E44AD', linewidth=2, label='Actor Loss')
            plt.plot(critic_losses, color='#E67E22', linewidth=2, label='Critic Loss')
            plt.title('Actor vs Critic Loss')
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Plot 5: Average Value
        plt.subplot(2, 3, 5)
        avg_values = [stats.get('avg_value', 0) for stats in training_stats]
        if avg_values:
            plt.plot(avg_values, color='#16A085', linewidth=2)
            plt.title('Average Value Prediction')
            plt.xlabel('Updates')
            plt.ylabel('Value')
            plt.grid(True)
        
        # Plot 6: Policy Std (diversity)
        plt.subplot(2, 3, 6)
        policy_stds = [stats.get('policy_std', 0) for stats in training_stats]
        if policy_stds:
            plt.plot(policy_stds, color='#D35400', linewidth=2)
            plt.title('Policy Standard Deviation')
            plt.xlabel('Updates')
            plt.ylabel('Std')
            plt.grid(True)
     
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'a3c_training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"A3C training progress plot saved to {os.path.join(save_path, 'a3c_training_progress.png')}")

def create_prediction_map_from_a3c(trainer, thermal_data, start_pos, weather_patches, 
                                    landcover_data, max_steps=500):
    """Create prediction map using A3C model - OPTIMIZED VERSION with GPU acceleration"""
    env = create_enhanced_crop_thermal_env(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        verbose=False
    )
    
    # USE GLOBAL NETWORK DIRECTLY for evaluation
    model = trainer.global_network
    model.eval()
    device = trainer.device
    
    # Pre-allocate output arrays
    V = np.zeros_like(thermal_data, dtype=np.float32)
    fire_prob_map = np.zeros_like(thermal_data, dtype=np.float32)
    
    # Generate all positions
    positions = [(i, j) for i in range(thermal_data.shape[0]) 
                 for j in range(thermal_data.shape[1])]
    
    # OPTIMIZED: Larger batch size for GPU
    batch_size = 1024 if device == 'cuda' else 512
    
    # Pre-compute thresholds to avoid repeated calculations
    medium_threshold = env.medium_temp_threshold
    
    with torch.no_grad():
        for batch_start in range(0, len(positions), batch_size):
            batch_end = min(batch_start + batch_size, len(positions))
            batch_positions = positions[batch_start:batch_end]
            
            # OPTIMIZED: Use batch observation method if available
            if hasattr(env, '_get_batch_observations'):
                batch_states = env._get_batch_observations(batch_positions)
            else:
                batch_states = np.array([env._get_observation(pos) for pos in batch_positions], 
                                        dtype=np.float32)
            
            # Convert to tensor and move to device
            states_tensor = torch.from_numpy(batch_states).to(device)
            
            # Check for NaN and handle
            if torch.isnan(states_tensor).any():
                states_tensor = torch.nan_to_num(states_tensor, nan=0.0)
            
            # Get predictions
            policies, values = model(states_tensor)
            
            # Check for NaN in outputs
            if torch.isnan(policies).any() or torch.isnan(values).any():
                policies = torch.nan_to_num(policies, nan=0.0)
                values = torch.nan_to_num(values, nan=0.0)
            
            # OPTIMIZED: Move to CPU once and then process
            values_np = values.cpu().numpy().flatten()
            policies_np = policies.cpu().numpy()
            
            # Vectorized processing for batch positions
            for idx, (i, j) in enumerate(batch_positions):
                V[i, j] = values_np[idx]
                
                # Use action 5 (predict fire) as fire indicator
                pred_prob = policies_np[idx, 5]
                
                # Dynamic threshold based on temperature
                local_temp = thermal_data[i, j]
                base_threshold = 0.2 if local_temp < medium_threshold else 0.15
                
                # Adjust for landcover
                if landcover_data[i, j] != 1:
                    base_threshold += 0.1
                
                fire_prob_map[i, j] = pred_prob if pred_prob > base_threshold else 0.0
    
    env.close()
    return V, fire_prob_map


def process_single_patch_a3c(patch_data, trainer, step, transform_affine, 
                              src_crs, patch_coords, original_shape):
    """Process single patch with A3C"""
    try:
        i = patch_data['index']
        
        V, _ = create_prediction_map_from_a3c(
            trainer,
            patch_data['thermal_data'],
            patch_data['start_pos'],
            patch_data['weather_patches'],
            patch_data['landcover_data'],
            max_steps=200
        )
        
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
        
        plt.close('all')
        
        lat_dms, lon_dms, lat_dms_hotspot, lon_dms_hotspot = save_combine_confusion(
            i, patch_coords, patch_data['start_pos'], overlay, patch_data['T_Celsius']
        )
        
        mse, corr = compute_mse_corr(pred_map, patch_data['thermal_data'])
        
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
        print(f"Error processing patch {patch_data.get('index', 'unknown')}: {str(e)}")
        return {'success': False, 'index': patch_data.get('index', -1), 'error': str(e)}

def enhanced_a3c_main_with_real_data(thermal_path: str,
                                   landcover_path: str,
                                   weather_tifs: dict,
                                   alignment_method='match_pixels',
                                   num_workers=4,
                                   max_episodes=1000,
                                   device='cpu',
                                   save_interval=50,
                                   steps_per_update=2000,
                                   force_retrain=False,
                                   use_parallel=False):
    """
    Main function for Enhanced A3C training with real data - COMPLETE IMPROVED VERSION
    """
    
    train_logger, eval_logger = setup_dual_logging()

    print("="*80)
    print("ENHANCED A3C TRAINING WITH REAL DATA (IMPROVED)")
    print(f"Training log: logs/training_a3c_*.log")
    print(f"Evaluation log: logs/evaluation_a3c_*.log")
    print("="*80)

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")
    print(f"Maximum episodes: {max_episodes}")
    print(f"Steps per update: {steps_per_update}")
    print(f"Save interval: {save_interval}")
    print(f"Parallel training: {use_parallel}")
    print(f"Force retrain: {force_retrain}")

    result_dirs = ["a3c_results", "a3c_models", "a3c_plots"]
    for folder in result_dirs:
        os.makedirs(folder, exist_ok=True)

    # Backup old model if exists and force_retrain
    if force_retrain and os.path.exists('a3c_models/best_a3c_model.pth'):
        backup_path = f'a3c_models/best_a3c_model_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        os.rename('a3c_models/best_a3c_model.pth', backup_path)
        print(f"Backed up old model to {backup_path}")

    # Read weather TIF files
    print("\n-> Loading weather data...")
    weather_data = {}
    for name, path in weather_tifs.items():
        with rasterio.open(path) as src:
            weather_data[name] = src.read(1)
        print(f"  Loaded {name}: {weather_data[name].shape}")

    print("\n-> Reading and patching thermal image...")
    patch_size = 100
    overlap = 10
    patches, original_shape, patch_coords, transform_affine, src_crs = read_and_patch_thermal_image(
        thermal_path, patch_size, overlap)
    print(f"  Created {len(patches)} thermal patches")

    print("\n-> Aligning landcover with thermal data...")
    landcover_patches = read_and_align_landcover_to_thermal(
        landcover_path, thermal_path, patch_size, overlap, method=alignment_method)

    if len(landcover_patches) > 0:
        full_aligned = reconstruct_from_patches(landcover_patches, original_shape, patch_size, overlap)

    print("\n-> Preparing training patches...")
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
    
    fire_ratios = [p['fire_ratio'] for p in patch_list]
    print(f"Fire ratio statistics - Mean: {np.mean(fire_ratios):.3f}, "
          f"Std: {np.std(fire_ratios):.3f}, "
          f"Min: {np.min(fire_ratios):.3f}, "
          f"Max: {np.max(fire_ratios):.3f}")
    
    # Split patches
    num_patches = len(patch_list)
    num_val = max(1, int(num_patches * 0.1))
    val_indices = np.random.choice(num_patches, num_val, replace=False)
    train_patches = [p for i, p in enumerate(patch_list) if i not in val_indices]
    print(f"Training patches: {len(train_patches)}, Validation patches: {num_val}")

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

    # Create temporary environment to get dimensions
    temp_env = env_creator()
    initial_obs = temp_env.reset()
    state_size = len(initial_obs)
    action_size = temp_env.action_space.n
    temp_env.close()

    print(f"\nDetected state size: {state_size}, Action size: {action_size}")

    # Initialize IMPROVED A3C trainer
    print("\n-> Initializing Improved A3C trainer...")
    trainer = ImprovedMultiAgentA3C(
        env_factory=env_creator,
        num_agents=num_workers,
        state_size=state_size,
        action_size=action_size,
        device=device,
        lr=1e-4,  # Lower learning rate
        gamma=0.99,
        entropy_coeff=0.05,  # Higher entropy for exploration
        value_coeff=0.5,
        update_interval=20
    )
    
    # Handle model loading
    model_loaded = False if force_retrain else trainer.load_best_model()

    if model_loaded and not force_retrain:
        print("\n" + "="*80)
        print("LOADED EXISTING A3C MODEL - STARTING EVALUATION")
        print("="*80)
        
        # Verify model
        print("\n-> Verifying model functionality...")
        with torch.no_grad():
            trainer.global_network.eval()
            test_state = torch.randn(10, state_size).to(trainer.device)
            test_policies, test_values = trainer.global_network(test_state)
            
            print(f"  Policy output shape: {test_policies.shape}")
            print(f"  Policy mean per action: {test_policies.mean(0).cpu().numpy()}")
            print(f"  Policy entropy: {-(test_policies * torch.log(test_policies + 1e-8)).sum(1).mean():.4f}")
            print(f"  Value range: [{test_values.min():.2f}, {test_values.max():.2f}]")
            print(f"  Model is {'WORKING' if not torch.isnan(test_values).any() else 'BROKEN'}")
        
        step = patch_size - overlap
        
        print("\n-> Processing patches for evaluation...")
        all_results = []
        
        for i, patch in enumerate(tqdm(patch_list, desc="Processing patches")):
            result = process_single_patch_a3c(
                patch, trainer, step, transform_affine,
                src_crs, patch_coords, original_shape
            )
            
            if result['success']:
                all_results.append(result)
            
            if (i + 1) % 100 == 0:
                gc.collect()

        if not all_results:
            print("\n[ERROR] No patches were successfully processed.")
            return None, trainer

        try:
            print("\n-> Exporting final confusion map...")
            export_final_confusion_map(patch_size, overlap, step, thermal_path)
            print("[OK] Final confusion map exported")
        except Exception as e:
            print(f"[WARNING] Could not export final confusion map: {str(e)}")

        try:
            df_results = pd.DataFrame(all_results)
            
            numeric_cols = ['AUC', 'MSE', 'Pearson_Correlation', 'Precision', 'Recall', 'F1', 'PR_AUC']
            
            for col in numeric_cols:
                if col not in df_results.columns:
                    df_results[col] = 0.0
            
            df_numeric = df_results[numeric_cols].apply(pd.to_numeric, errors='coerce')
            avg_metrics = df_numeric.mean(numeric_only=True)

            summary_row = {col: float(avg_metrics.get(col, 0)) for col in numeric_cols}
            summary_row.update({
                'Patch': 'Average', 
                'Lat': '', 
                'Lon': '', 
                'lat_hotspot': '', 
                'lon_hotspot': '',
                'Max_Temp': float(df_results['Max_Temp'].mean()) if 'Max_Temp' in df_results else 0
            })
            df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
            
            csv_path = "a3c_results/a3c_results_summary.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"[OK] Results saved to {csv_path}")

            print(f"\n{'='*60}")
            print(f"A3C RESULTS SUMMARY ({len(all_results)} patches)")
            print(f"{'='*60}")
            print(f"Average AUC: {avg_metrics.get('AUC', 0):.4f}")
            print(f"Average Precision: {avg_metrics.get('Precision', 0):.4f}")
            print(f"Average Recall: {avg_metrics.get('Recall', 0):.4f}")
            print(f"Average F1: {avg_metrics.get('F1', 0):.4f}")
            print(f"Average PR-AUC: {avg_metrics.get('PR_AUC', 0):.4f}")
            print(f"Average MSE: {avg_metrics.get('MSE', 0):.4f}")
            print(f"Average Correlation: {avg_metrics.get('Pearson_Correlation', 0):.4f}")
            print(f"{'='*60}")

            return df_results, trainer
            
        except Exception as e:
            print(f"\n[ERROR] Failed to create results summary: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, trainer
    
    else:
        # TRAINING MODE
        train_logger = logging.getLogger('Training')
        train_logger.info("Starting Improved A3C training...")
        print("\n" + "="*80)
        print("STARTING IMPROVED A3C TRAINING")
        print("="*80)
        
        rewards_history = []
        training_stats_history = []
        update_count = 0
        
        num_updates = 100
        print(f"Training for {num_updates} updates ({steps_per_update} steps each)")
        print(f"Number of training patches: {len(train_patches)}")
        print("-"*80)
        
        for update in range(1, num_updates + 1):
            print(f"\n[Update {update}/{num_updates}]", end=" ")
            
            # Epsilon decay for exploration
            epsilon = max(0.01, 0.1 * (0.995 ** update))
            
            # Collect experience
            trainer.collect_experience(steps_per_agent=steps_per_update, epsilon=epsilon)
            
            # Aggregate statistics
            episode_rewards = []
            for agent in trainer.agents:
                if agent.episodes_completed > 0:
                    episode_rewards.append(agent.reward_mean)
            
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                rewards_history.append(avg_reward)
                
                print(f"Reward: {avg_reward:+.2f}", end=" ")
                
                # Get recent stats
                if len(trainer.training_stats) > 0:
                    recent_stats = trainer.training_stats[-10:]
                    avg_loss = np.mean([s.get('total_loss', 0) for s in recent_stats])
                    avg_entropy = np.mean([s.get('entropy', 0) for s in recent_stats])
                    avg_value = np.mean([s.get('avg_value', 0) for s in recent_stats])
                    policy_std = np.mean([s.get('policy_std', 0) for s in recent_stats])
                    
                    print(f"| Loss: {avg_loss:.3f} | Entropy: {avg_entropy:.3f} | Value: {avg_value:.2f} | Policy_std: {policy_std:.3f}", end=" ")
                    
                    training_stats_history.append({
                        'total_loss': avg_loss,
                        'entropy': avg_entropy,
                        'actor_loss': np.mean([s.get('actor_loss', 0) for s in recent_stats]),
                        'critic_loss': np.mean([s.get('critic_loss', 0) for s in recent_stats]),
                        'avg_value': avg_value,
                        'policy_std': policy_std
                    })
                
                # Save best model
                if avg_reward > trainer.best_reward:
                    trainer.best_reward = avg_reward
                    trainer.save_model()
                    print("[BEST]", end=" ")
                    train_logger.info(f"New best A3C model saved: {avg_reward:.4f}")
                
                train_logger.info(f"Update {update}: Avg Reward={avg_reward:.4f}, Entropy={avg_entropy:.4f}")
            
            # Detailed checkpoint every 10 updates
            if update % 10 == 0:
                print(f"\n{'='*80}")
                print(f"CHECKPOINT at update {update}/{num_updates}:")
                print(f"  Average reward (last 10): {np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history):.2f}")
                print(f"  Best reward so far: {trainer.best_reward:.2f}")
                print(f"  Episodes completed: {sum(a.episodes_completed for a in trainer.agents)}")
                print(f"  Current epsilon: {epsilon:.4f}")
                print(f"  Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
                
                # Check action distribution
                if len(trainer.agents) > 0:
                    action_dist = trainer.agents[0].get_action_distribution()
                    print(f"  Agent 0 action dist: {action_dist}")
                
                # Check policy diversity
                with torch.no_grad():
                    test_state = torch.randn(100, state_size).to(device)
                    policies, values = trainer.global_network(test_state)
                    policy_entropy = -(policies * torch.log(policies + 1e-8)).sum(1).mean()
                    print(f"  Global policy entropy: {policy_entropy:.4f}")
                    print(f"  Global value range: [{values.min():.2f}, {values.max():.2f}]")
                    print(f"  Policy mean per action: {policies.mean(0).cpu().numpy()}")
                
                print(f"{'='*80}")
                
            # Save checkpoint periodically
            if update % save_interval == 0:
                checkpoint_path = f"a3c_models/a3c_model_update_{update}.pth"
                trainer.save_model(checkpoint_path)
                print(f"\n  Checkpoint saved: {checkpoint_path}")
                
            update_count += 1
        
        # Final plot
        print("\n-> Generating training plots...")
        try:
            plot_training_progress_a3c(
                rewards_history, 
                save_path="a3c_plots",
                training_stats=training_stats_history)
        except Exception as e:
            print(f"Error in plotting: {e}")
        
        # Save training history
        training_history = {
            'rewards': convert_to_serializable(rewards_history),
            'stats': convert_to_serializable(training_stats_history),
            'best_reward': float(trainer.best_reward),
            'final_update': update_count
        }
        
        history_path = 'a3c_models/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        print(f"\n{'='*80}")
        print("A3C TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Final reward: {rewards_history[-1] if rewards_history else 'N/A':.2f}")
        print(f"Best reward: {trainer.best_reward:.2f}")
        print(f"Total updates: {update_count}")
        print(f"Total episodes: {sum(a.episodes_completed for a in trainer.agents)}")
        print(f"{'='*80}")
        
        # Now evaluate on all patches
        print("\n" + "="*80)
        print("STARTING POST-TRAINING EVALUATION")
        print("="*80)
        
        step = patch_size - overlap
        all_results = []
        
        for i, patch in enumerate(tqdm(patch_list, desc="Evaluating patches")):
            result = process_single_patch_a3c(
                patch, trainer, step, transform_affine,
                src_crs, patch_coords, original_shape
            )
            
            if result['success']:
                all_results.append(result)
            
            if (i + 1) % 100 == 0:
                gc.collect()

        if not all_results:
            print("\n[WARNING] No patches were successfully evaluated.")
            return None, trainer

        try:
            print("\n-> Exporting final confusion map...")
            export_final_confusion_map(patch_size, overlap, step, thermal_path)
            print("[OK] Final confusion map exported")
        except Exception as e:
            print(f"[WARNING] Could not export final confusion map: {str(e)}")

        try:
            df_results = pd.DataFrame(all_results)
            
            numeric_cols = ['AUC', 'MSE', 'Pearson_Correlation', 'Precision', 'Recall', 'F1', 'PR_AUC']
            
            for col in numeric_cols:
                if col not in df_results.columns:
                    df_results[col] = 0.0
            
            df_numeric = df_results[numeric_cols].apply(pd.to_numeric, errors='coerce')
            avg_metrics = df_numeric.mean(numeric_only=True)

            summary_row = {col: float(avg_metrics.get(col, 0)) for col in numeric_cols}
            summary_row.update({
                'Patch': 'Average', 
                'Lat': '', 
                'Lon': '', 
                'lat_hotspot': '', 
                'lon_hotspot': '',
                'Max_Temp': float(df_results['Max_Temp'].mean()) if 'Max_Temp' in df_results else 0
            })
            df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
            
            csv_path = "a3c_results/a3c_results_summary.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"[OK] Results saved to {csv_path}")

            print(f"\n{'='*80}")
            print(f"POST-TRAINING EVALUATION RESULTS ({len(all_results)} patches)")
            print(f"{'='*80}")
            print(f"Average AUC: {avg_metrics.get('AUC', 0):.4f}")
            print(f"Average Precision: {avg_metrics.get('Precision', 0):.4f}")
            print(f"Average Recall: {avg_metrics.get('Recall', 0):.4f}")
            print(f"Average F1: {avg_metrics.get('F1', 0):.4f}")
            print(f"Average PR-AUC: {avg_metrics.get('PR_AUC', 0):.4f}")
            print(f"Average MSE: {avg_metrics.get('MSE', 0):.4f}")
            print(f"Average Correlation: {avg_metrics.get('Pearson_Correlation', 0):.4f}")
            print(f"{'='*80}")

            return df_results, trainer
            
        except Exception as e:
            print(f"\n[ERROR] Failed to create results summary: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A3C Training and Evaluation for Wildfire Detection")
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'eval', 'both'],
                       help="Mode: train, eval, or both (default: both)")
    parser.add_argument('--episodes', type=int, default=100,
                       help="Number of training episodes (default: 100)")
    parser.add_argument('--device', type=str, default='auto',
                       help="Device: cuda, cpu, or auto (default: auto)")
    parser.add_argument('--workers', type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    parser.add_argument('--steps', type=int, default=2000,
                       help="Steps per update (default: 2000)")
    parser.add_argument('--save_interval', type=int, default=10,
                       help="Save model every N episodes (default: 10)")
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
    print("A3C WILDFIRE DETECTION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Workers: {args.workers}")
    print(f"Steps per update: {args.steps}")
    print("="*60)
    
    try:
        results, trainer = enhanced_a3c_main_with_real_data(
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
            use_parallel=True
        )

        if results is not None:
            print("\n" + "="*80)
            print("A3C PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Results saved to: a3c_results/a3c_results_summary.csv")
            print(f"Best model saved to: a3c_models/best_a3c_model.pth")
        else:
            print("\n[WARNING] Pipeline completed but no results generated")
        
    except Exception as e:
        print(f"\n[ERROR] Error during A3C execution: {str(e)}")
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