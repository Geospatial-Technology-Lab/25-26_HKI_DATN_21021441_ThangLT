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

# Import utils for visualization and metrics
from utils.auc_pred_map import compute_auc_and_plot_full
from utils.confusion_map import compute_confusion_overlay
from utils.final_map import export_final_confusion_map
from utils.save_tif_confusion import save_tif_confusion
from utils.save_combine_confusion import save_combine_confusion
from utils.compute_mse_corr import compute_mse_corr
from utils.convert import convert_to_serializable

# Import environment and VPG implementation
from environment.env_src import create_enhanced_crop_thermal_env  
from vpg.vpg import MultiAgentVPG

# Disable console output
logging.getLogger().handlers = []
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('VPG_Evaluation').propagate = False

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
    from datetime import datetime
    
    os.makedirs('logs', exist_ok=True)
    logging.getLogger().handlers = []
    
    # Setup TRAINING logger
    train_logger = logging.getLogger('Training')
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    
    train_console = logging.StreamHandler()
    train_console.setLevel(logging.INFO)
    train_format = logging.Formatter('[TRAIN] %(message)s')
    train_console.setFormatter(train_format)
    train_logger.addHandler(train_console)
    
    train_file = logging.FileHandler(
        f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    train_file.setLevel(logging.DEBUG)
    train_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    train_logger.addHandler(train_file)
    
    # Setup EVALUATION logger
    eval_logger = logging.getLogger('VPG_Evaluation')
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    
    eval_file = logging.FileHandler(
        f'logs/evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    eval_file.setLevel(logging.INFO)
    eval_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    eval_logger.addHandler(eval_file)
    
    return train_logger, eval_logger

def plot_training_progress_simple(rewards_history, save_path="vpg_training_plots", 
                                training_stats=None):
    """Plot training progress"""
    os.makedirs(save_path, exist_ok=True)
    
    if not rewards_history:
        print("No rewards history to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history, color='#2E86C1', linewidth=2, label='Training Rewards')
    plt.title('Training Episode Rewards')
    plt.xlabel('Updates')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    if training_stats and len(training_stats) > 0:
        # Plot 2: Entropy
        plt.subplot(2, 2, 2)
        entropy_values = []
        for stats in training_stats:
            if isinstance(stats.get('entropy'), dict):
                entropy_values.append(stats.get('entropy', {}).get('mean', 0))
            else:
                entropy_values.append(stats.get('entropy', 0))
        
        if entropy_values:
            plt.plot(entropy_values, color='#27AE60', linewidth=2)
            plt.title('Entropy')
            plt.xlabel('Updates')
            plt.ylabel('Entropy')
            plt.grid(True)
        
        # Plot 3: Policy Loss
        plt.subplot(2, 2, 3)
        policy_losses = []
        for stats in training_stats:
            if isinstance(stats.get('policy_loss'), dict):
                policy_losses.append(stats.get('policy_loss', {}).get('mean', 0))
            else:
                policy_losses.append(stats.get('policy_loss', 0))
        
        if policy_losses:
            plt.plot(policy_losses, color='#E74C3C', linewidth=2)
            plt.title('Policy Loss')
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.grid(True)
        
        # Plot 4: Return Statistics
        plt.subplot(2, 2, 4)
        return_means = []
        for stats in training_stats:
            if isinstance(stats.get('return_mean'), dict):
                return_means.append(stats.get('return_mean', {}).get('mean', 0))
            else:
                return_means.append(stats.get('return_mean', 0))
        
        if return_means:
            plt.plot(return_means, color='#8E44AD', linewidth=2)
            plt.title('Average Return')
            plt.xlabel('Updates')
            plt.ylabel('Return')
            plt.grid(True)
     
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {os.path.join(save_path, 'training_progress.png')}")

def create_prediction_map_from_vpg_fast(trainer, thermal_data, start_pos, weather_patches, landcover_data, max_steps=500):
    """Generate prediction map using trained VPG agent"""
    env = create_enhanced_crop_thermal_env(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        verbose=False
    )
    
    agent = trainer.agents[0]
    agent.network.eval()
    
    V = np.zeros_like(thermal_data)
    fire_prob_map = np.zeros_like(thermal_data)
    
    positions = [(i, j) for i in range(thermal_data.shape[0]) 
                 for j in range(thermal_data.shape[1])]
    
    batch_size = 128
    
    with torch.no_grad():
        for batch_start in range(0, len(positions), batch_size):
            batch_end = min(batch_start + batch_size, len(positions))
            batch_positions = positions[batch_start:batch_end]
            
            # Collect batch observations
            batch_states = []
            for pos in batch_positions:
                state = env._get_observation(pos)
                batch_states.append(state)
            
            # Process entire batch at once
            states_tensor = torch.FloatTensor(np.array(batch_states)).to(agent.device)
            policies, values = agent.network(states_tensor)
            
            # Apply results
            for idx, (i, j) in enumerate(batch_positions):
                V[i, j] = values[idx].item()
                pred_prob = policies[idx, 5].item()
                
                # Simplified thresholding
                local_temp = thermal_data[i, j]
                base_threshold = 0.8 if local_temp < env.medium_temp_threshold else 0.7
                
                # Quick landcover check
                if landcover_data[i, j] != 1:
                    base_threshold += 0.2
                
                fire_prob_map[i, j] = pred_prob if pred_prob > base_threshold else 0.0
    
    env.close()
    return V, fire_prob_map

def process_single_patch_fast_clean(patch_data, trainer, step, transform_affine, src_crs, patch_coords, original_shape):
    """Process a single patch with clean logging"""
    import logging
    eval_logger = logging.getLogger('VPG_Evaluation')
    
    try:
        i = patch_data['index']
        eval_logger.info(f"Starting to process patch {i+1}")
        
        # Generate prediction map
        V, _ = create_prediction_map_from_vpg_fast(
            trainer,
            patch_data['thermal_data'],
            patch_data['start_pos'],
            patch_data['weather_patches'],
            patch_data['landcover_data'],
            max_steps=200
        )
        
        eval_logger.info(f"Patch {i+1}: Prediction map generated")
        
        # Suppress all prints from utils functions
        with SuppressPrints():
            # Compute AUC quietly
            auc, pred_map = compute_auc_and_plot_full(
                grid_values=None,
                fire_ground_truth_binary=patch_data['y_true_binary'],
                fire_ground_truth_continuous=patch_data['thermal_data'],
                crop_id=i + 1,
                pred_map=V,
                no_value_iteration=False
            )
            
            eval_logger.info(f"Patch {i+1}: AUC computed = {auc:.4f}")
            
            # Compute confusion overlay quietly
            overlay, precision, recall, f1, pr_auc = compute_confusion_overlay(
                pred_map, patch_data['y_true_binary'], i
            )
            
            eval_logger.info(f"Patch {i+1}: Metrics - Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            
            # Save TIF confusion quietly
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
        
        # Compute MSE and correlation
        mse, corr = compute_mse_corr(pred_map, patch_data['thermal_data'])
        
        eval_logger.info(f"Patch {i+1} completed successfully: "
                   f"AUC={auc:.4f}, F1={f1:.4f}, MSE={mse:.4f}, Correlation={corr:.4f}")
        
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

def process_patches_parallel_clean(patch_list, trainer, step, transform_affine, 
                                  src_crs, patch_coords, original_shape, 
                                  num_workers=4):
    """Process all patches in parallel with clean progress display"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import gc
    import logging
    
    eval_logger = logging.getLogger('VPG_Evaluation')
    
    print(f"\n-> Processing {len(patch_list)} patches using {num_workers} workers...")
    print("Check logs/evaluation_*.log for detailed progress\n")
    
    all_results = []
    failed_patches = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for patch in patch_list:
            future = executor.submit(
                process_single_patch_fast_clean,
                patch, trainer, step, transform_affine, 
                src_crs, patch_coords, original_shape
            )
            futures.append(future)
        
        with tqdm(total=len(patch_list), desc="Processing", ncols=100, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    if result['success']:
                        all_results.append(result)
                    else:
                        failed_patches.append(result.get('index', -1))
                        if 'error' in result:
                            eval_logger.error(f"Patch {result['index']} failed: {result['error']}")
                except Exception as e:
                    eval_logger.error(f"Future execution error: {str(e)}")
                    failed_patches.append(-1)
                finally:
                    pbar.update(1)
                    
                    if len(all_results) % 100 == 0:
                        gc.collect()
                        if all_results:
                            avg_auc = np.mean([r['AUC'] for r in all_results[-100:]])
                            pbar.set_postfix({'Avg_AUC': f'{avg_auc:.3f}'})
    
    if failed_patches:
        eval_logger.warning(f"{len(failed_patches)} patches failed to process")
        print(f"\n[WARNING] {len(failed_patches)} patches failed (see log for details)")
    
    print(f"\n[OK] Successfully processed {len(all_results)}/{len(patch_list)} patches")
    
    if all_results:
        avg_metrics = {
            'AUC': np.mean([r['AUC'] for r in all_results]),
            'F1': np.mean([r['F1'] for r in all_results]),
            'Precision': np.mean([r['Precision'] for r in all_results]),
            'Recall': np.mean([r['Recall'] for r in all_results])
        }
        eval_logger.info(f"Final average metrics: {avg_metrics}")
    
    return all_results

def check_and_fix_model_compatibility(trainer, force_retrain=False):
    """Check if saved model is compatible with current environment dimensions"""
    if force_retrain:
        print("Force retrain requested - will create new model")
        if os.path.exists(trainer.best_model_path):
            backup_path = trainer.best_model_path + ".backup_force_retrain"
            os.rename(trainer.best_model_path, backup_path)
            print(f"Existing model backed up to {backup_path}")
        return False
    
    model_loaded = trainer.load_best_model()
    
    if not model_loaded:
        print("No compatible model found or failed to load - will train new model")
        return False
    
    print("Compatible model loaded successfully")
    return True

def enhanced_vpg_main_with_real_data(thermal_path: str,
                                   landcover_path: str,
                                   weather_tifs: dict,
                                   alignment_method='match_pixels',
                                   num_workers=4,
                                   max_episodes=1000,
                                   device='cpu',
                                   save_interval=50,
                                   steps_per_update=2000,
                                   force_retrain=False,
                                   use_parallel_processing=True,
                                   use_baseline=True):
    """Main function for Enhanced VPG training with real data (multi-patch RL)"""
    
    train_logger, eval_logger = setup_dual_logging()

    print("="*80)
    print("ENHANCED VPG TRAINING WITH REAL DATA (MULTI-PATCH) - OPTIMIZED VERSION")
    print(f"Baseline: {'Enabled (Actor-Critic)' if use_baseline else 'Disabled (Pure REINFORCE)'}")
    print(f"Training log: logs/training_*.log")
    print(f"Evaluation log: logs/evaluation_*.log")
    print("="*80)

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")
    print(f"Maximum episodes: {max_episodes}")
    print(f"Steps per update: {steps_per_update}")
    print(f"Save interval: {save_interval}")
    print(f"Parallel processing: {use_parallel_processing}")

    result_dirs = ["vpg_results", "vpg_models", "vpg_plots"]
    for folder in result_dirs:
        os.makedirs(folder, exist_ok=True)

    # Read weather TIF files
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
    
    # Split patches into train and validation sets
    num_patches = len(patch_list)
    num_val = max(1, int(num_patches * 0.1))
    val_indices = np.random.choice(num_patches, num_val, replace=False)
    train_patches = [p for i, p in enumerate(patch_list) if i not in val_indices]

    def env_creator():
        """Create environment with optimized parameters"""
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

    # Create a temporary environment to get the correct dimensions
    temp_env = env_creator()
    initial_obs = temp_env.reset()
    state_size = len(initial_obs)
    action_size = temp_env.action_space.n
    temp_env.close()

    print(f"Detected state size: {state_size}, Action size: {action_size}")

    # Initialize VPG trainer with correct dimensions
    trainer = MultiAgentVPG(
        env_factory=env_creator,
        num_agents=num_workers,
        state_size=state_size,
        action_size=action_size,
        device=device,
        lr=3e-4,
        gamma=0.99,
        entropy_coeff=0.01,
        value_coeff=0.5,
        use_baseline=use_baseline  # VPG-specific: whether to use value baseline
    )
    
    # Handle model loading with dimension checking
    model_loaded = check_and_fix_model_compatibility(trainer, force_retrain)

    if model_loaded and not force_retrain:
        print("Loaded existing model. Generating predictions...")
        
        step = patch_size - overlap
        
        if use_parallel_processing:
            print("\n-> Using PARALLEL PROCESSING for faster evaluation...")
            
            optimal_workers = min(os.cpu_count() - 1, 8, len(patch_list) // 100 + 1)
            print(f"Using {optimal_workers} parallel workers")
            
            all_results = process_patches_parallel_clean(
                patch_list, trainer, step, transform_affine, 
                src_crs, patch_coords, original_shape, 
                num_workers=optimal_workers
            )
        else:
            print("\n-> Using SEQUENTIAL processing (slower)...")
            all_results = []
            
            for i, patch in enumerate(tqdm(patch_list, desc="Processing patches")):
                result = process_single_patch_fast_clean(
                    patch, trainer, step, transform_affine,
                    src_crs, patch_coords, original_shape
                )
                if result['success']:
                    all_results.append(result)
                
                if (i + 1) % 100 == 0:
                    gc.collect()

        if not all_results:
            print("\n[ERROR] No patches were successfully processed. Cannot generate results.")
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
            
            csv_path = "vpg_results/vpg_results_summary.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"[OK] Results saved to {csv_path}")

            print(f"\n{'='*60}")
            print(f"RESULTS SUMMARY ({len(all_results)} patches processed)")
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
        # Training mode
        train_logger = logging.getLogger('Training')
        train_logger.info("Starting VPG training...")
        print("\n" + "="*60)
        print("STARTING VPG TRAINING")
        print("="*60)
        
        rewards_history = []
        training_stats_history = []
        best_reward = -float('inf')
        patience_counter = 0
        patience_limit = 50
        update_count = 0
        
        print(f"Training for 100 updates ({steps_per_update} steps each)")
        print(f"Number of training patches: {len(train_patches)}")
        print("-"*60)
        
        for update in range(1, 101):
            print(f"\nUpdate {update}/100", end="")
            
            # Collect experience
            trainer.collect_experience(max(steps_per_update, 500))
            
            episode_rewards = []
            update_stats = defaultdict(list)
            
            for agent in trainer.agents:
                if len(agent.states) > 32:
                    stats = agent.update()
                    if agent.episodes_completed > 0:
                        episode_rewards.append(agent.reward_mean)
                    
                    for key, value in stats.items():
                        update_stats[key].append(value)
            
            # Calculate statistics
            if update_stats:
                update_stats = {
                    key: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    } for key, values in update_stats.items()
                }
            
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                rewards_history.append(avg_reward)
                training_stats_history.append(update_stats)
                
                print(f" | Reward: {avg_reward:+.2f}", end="")
                
                if update_stats:
                    policy_loss = update_stats.get('policy_loss', {}).get('mean', 0)
                    entropy = update_stats.get('entropy', {}).get('mean', 0)
                    print(f" | P-Loss: {policy_loss:.3f} | Entropy: {entropy:.3f}", end="")
                
                # Save best model
                if avg_reward > trainer.best_reward:
                    trainer.best_reward = avg_reward
                    trainer.save_model()
                    print(" [BEST]", end="")
                    train_logger.info(f"New best model saved: {avg_reward:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                train_logger.info(f"Update {update}: Avg Reward={avg_reward:.4f}, Stats={update_stats}")
            
            # Periodic checkpoint
            if update % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Checkpoint at update {update}:")
                print(f"  Average reward (last 10): {np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history):.2f}")
                print(f"  Best reward so far: {trainer.best_reward:.2f}")
                print(f"  Episodes completed: {sum(a.episodes_completed for a in trainer.agents)}")
                print(f"{'='*60}")
                
            # Early stopping check
            if patience_counter >= patience_limit:
                print(f"\n[INFO] No improvement for {patience_limit} updates. Stopping training.")
                break
                
            # Save checkpoint periodically
            if update % save_interval == 0:
                trainer.save_model(f"vpg_models/vpg_model_update_{update}.pth")
                
            update_count += 1
        
        # Final plot
        try:
            plot_training_progress_simple(
                rewards_history, 
                save_path="vpg_plots",
                training_stats=training_stats_history)
        except Exception as e:
            print(f"Error in plotting: {e}")
        
        # Save training history
        training_history = {
            'rewards': convert_to_serializable(rewards_history),
            'stats': convert_to_serializable(training_stats_history)
        }
        
        with open('vpg_models/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"Final reward: {rewards_history[-1] if rewards_history else 'N/A':.2f}")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Total updates: {update_count}")
        print(f"{'='*60}")
        
        return None, trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VPG Training and Evaluation for Wildfire Detection")
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
    parser.add_argument('--save_interval', type=int, default=5,
                       help="Save model every N episodes (default: 5)")
    parser.add_argument('--force_retrain', action='store_true',
                       help="Force retraining even if model exists")
    parser.add_argument('--no_baseline', action='store_true',
                       help="Disable value baseline (use pure REINFORCE)")
    
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
    print("VPG WILDFIRE DETECTION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Workers: {args.workers}")
    print(f"Steps per update: {args.steps}")
    print(f"Use baseline: {not args.no_baseline}")
    print("="*60)

    try:
        results, trainer = enhanced_vpg_main_with_real_data(
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
            use_parallel_processing=True,
            use_baseline=not args.no_baseline
        )

        print("\nCompleted successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        if 'trainer' in locals():
            for env in trainer.envs:
                try:
                    env.close()
                except:
                    pass
        print("Done!")