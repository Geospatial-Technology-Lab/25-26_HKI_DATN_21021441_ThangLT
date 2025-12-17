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

# Import environment and DDPG implementation
from environment.env_src import create_enhanced_crop_thermal_env  
from ddpg.ddpg import MultiAgentDDPG

# Simple logging setup
logging.getLogger().handlers = []
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('DDPG_Evaluation').propagate = False

# Safe print function
_real_print = print
def safe_print(*args, **kwargs):
    """Print function that safely handles Unicode characters"""
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

# Suppress prints context manager
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
        f'logs/ddpg_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    train_file.setLevel(logging.DEBUG)
    train_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    train_logger.addHandler(train_file)
    
    # Setup EVALUATION logger
    eval_logger = logging.getLogger('DDPG_Evaluation')
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    
    eval_file = logging.FileHandler(
        f'logs/ddpg_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    eval_file.setLevel(logging.INFO)
    eval_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    eval_logger.addHandler(eval_file)
    
    return train_logger, eval_logger

def plot_ddpg_training_progress(rewards_history, critic_loss_history, actor_loss_history, 
                                noise_history=None, q_values_history=None, 
                                save_path="ddpg_training_plots"):
    """Plot DDPG training progress"""
    os.makedirs(save_path, exist_ok=True)
    
    if not rewards_history:
        print("No rewards history to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Training rewards
    axes[0, 0].plot(rewards_history, color='#2E86C1', linewidth=2, label='Training Rewards')
    axes[0, 0].set_title('DDPG Training Episode Rewards', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Updates')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Critic Loss
    if critic_loss_history:
        axes[0, 1].plot(critic_loss_history, color='#E74C3C', linewidth=2)
        axes[0, 1].set_title('Critic Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Updates')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Actor Loss
    if actor_loss_history:
        axes[0, 2].plot(actor_loss_history, color='#27AE60', linewidth=2)
        axes[0, 2].set_title('Actor Loss', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Updates')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Noise scale decay
    if noise_history:
        axes[1, 0].plot(noise_history, color='#F39C12', linewidth=2)
        axes[1, 0].set_title('Exploration Noise Scale', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Updates')
        axes[1, 0].set_ylabel('Noise Scale')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Q-values
    if q_values_history:
        axes[1, 1].plot(q_values_history, color='#9B59B6', linewidth=2)
        axes[1, 1].set_title('Average Q-Values', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Updates')
        axes[1, 1].set_ylabel('Q-Value')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Moving average reward
    if len(rewards_history) > 10:
        window = 10
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        axes[1, 2].plot(rewards_history, alpha=0.3, color='#2E86C1', label='Raw')
        axes[1, 2].plot(range(window-1, len(rewards_history)), moving_avg, 
                       color='#2E86C1', linewidth=2, label=f'{window}-update MA')
        axes[1, 2].set_title('Reward Moving Average', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Updates')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'ddpg_training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"DDPG training progress plot saved to {plot_path}")

def create_prediction_map_from_ddpg_fast(agent, thermal_data, start_pos, weather_patches, 
                                         landcover_data, max_steps=500):
    """
    Optimized version of prediction map generation for DDPG
    """
    env = create_enhanced_crop_thermal_env(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        verbose=False
    )
    
    agent.actor.eval()
    agent.critic.eval()
    
    V = np.zeros_like(thermal_data)
    fire_prob_map = np.zeros_like(thermal_data)
    
    # Batch processing for better performance
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
            
            # Get action probabilities from actor
            action_probs = agent.actor(states_tensor)
            
            # Get Q-values from critic
            q_values = agent.critic(states_tensor, action_probs)
            
            # Get fire prediction probabilities
            fire_probs = action_probs[:, 5]  # Action 5 = Predict Fire
            
            # Apply results
            for idx, (i, j) in enumerate(batch_positions):
                V[i, j] = q_values[idx].item()
                pred_prob = fire_probs[idx].item()
                
                # Simplified thresholding for speed
                local_temp = thermal_data[i, j]
                base_threshold = 0.8 if local_temp < env.medium_temp_threshold else 0.7
                
                # Quick landcover check
                if landcover_data[i, j] != 1:
                    base_threshold += 0.2
                
                fire_prob_map[i, j] = pred_prob if pred_prob > base_threshold else 0.0
    
    env.close()
    return V, fire_prob_map

def process_single_patch_ddpg_fast_clean(patch_data, agent, step, transform_affine, 
                                         src_crs, patch_coords, original_shape):
    """
    Process a single patch with DDPG agent - clean logging
    """
    eval_logger = logging.getLogger('DDPG_Evaluation')
    
    try:
        i = patch_data['index']
        eval_logger.info(f"Starting to process patch {i+1}")
        
        # Generate prediction map
        V, _ = create_prediction_map_from_ddpg_fast(
            agent,
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
        
        # Close any plots to free memory
        plt.close('all')
        
        # Save combined confusion
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

def process_patches_parallel_ddpg_clean(patch_list, agent, step, transform_affine, 
                                        src_crs, patch_coords, original_shape, 
                                        num_workers=4):
    """
    Process all patches in parallel with clean progress display - DDPG version
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    eval_logger = logging.getLogger('DDPG_Evaluation')
    
    print(f"\n-> Processing {len(patch_list)} patches using {num_workers} workers...")
    print("Check logs/ddpg_evaluation_*.log for detailed progress\n")
    
    all_results = []
    failed_patches = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for patch in patch_list:
            future = executor.submit(
                process_single_patch_ddpg_fast_clean,
                patch, agent, step, transform_affine, 
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

def check_and_fix_ddpg_model_compatibility(trainer, force_retrain=False):
    """
    Check if saved DDPG model is compatible with current environment dimensions
    """
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
    
    print("Compatible DDPG model loaded successfully")
    return True

def enhanced_ddpg_main_with_real_data(thermal_path: str,
                                     landcover_path: str,
                                     weather_tifs: dict,
                                     alignment_method='match_pixels',
                                     num_workers=4,
                                     max_episodes=1000,
                                     device='cpu',
                                     save_interval=50,
                                     steps_per_update=2000,
                                     force_retrain=False,
                                     use_parallel_processing=True):
    """
    Main function for Enhanced DDPG training with real data (multi-patch RL)
    Fully compatible with PPO and DQN versions
    """
    
    # Setup dual logging for training and evaluation
    train_logger, eval_logger = setup_dual_logging()

    print("="*80)
    print("ENHANCED DDPG TRAINING WITH REAL DATA (MULTI-PATCH) - OPTIMIZED VERSION")
    print(f"Training log: logs/ddpg_training_*.log")
    print(f"Evaluation log: logs/ddpg_evaluation_*.log")
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

    result_dirs = ["ddpg_results", "ddpg_models", "ddpg_plots"]
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

    # Initialize DDPG trainer with correct dimensions
    trainer = MultiAgentDDPG(
        env_factory=env_creator,
        num_agents=num_workers,
        state_size=state_size,
        action_size=action_size,
        device=device,
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        batch_size=128,
        noise_scale=0.2,
        noise_decay=0.995,
        use_prioritized_replay=False
    )
    
    # Handle model loading with dimension checking
    model_loaded = check_and_fix_ddpg_model_compatibility(trainer, force_retrain)

    if model_loaded and not force_retrain:
        print("Loaded existing DDPG model. Generating predictions...")
        
        # Calculate step for patches
        step = patch_size - overlap
        
        agent = trainer.agents[0]  # Use first agent for evaluation
        
        if use_parallel_processing:
            print("\n-> Using PARALLEL PROCESSING for faster evaluation...")
            optimal_workers = min(os.cpu_count() - 1, 8, len(patch_list) // 100 + 1)
            print(f"Using {optimal_workers} parallel workers")
            
            all_results = process_patches_parallel_ddpg_clean(
                patch_list, agent, step, transform_affine, 
                src_crs, patch_coords, original_shape, 
                num_workers=optimal_workers
            )
        else:
            print("\n-> Using SEQUENTIAL processing (slower)...")
            all_results = []
            
            for patch in tqdm(patch_list, desc="Processing patches"):
                result = process_single_patch_ddpg_fast_clean(
                    patch, agent, step, transform_affine,
                    src_crs, patch_coords, original_shape
                )
                if result['success']:
                    all_results.append(result)
                
                if len(all_results) % 100 == 0:
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

            # Add summary row
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
            
            # Save results
            csv_path = "ddpg_results/ddpg_results_summary.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"[OK] Results saved to {csv_path}")

            print(f"\n{'='*60}")
            print(f"DDPG RESULTS SUMMARY ({len(all_results)} patches processed)")
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
        train_logger.info("Starting DDPG training...")
        print("\n" + "="*60)
        print("STARTING DDPG TRAINING")
        print("="*60)
        
        rewards_history = []
        critic_loss_history = []
        actor_loss_history = []
        noise_history = []
        q_values_history = []
        best_reward = -float('inf')
        
        num_updates = 200  # Same as PPO and DQN
        print(f"Training for {num_updates} updates ({steps_per_update} steps each)")
        print(f"Number of training patches: {len(train_patches)}")
        print("-"*60)
        
        for update in range(1, num_updates + 1):
            # Training progress
            print(f"\nUpdate {update}/{num_updates}", end="")
            
            # Collect experience
            trainer.collect_experience(max(steps_per_update, 500))
            
            # Update all agents
            episode_rewards = []
            update_stats = defaultdict(list)
            
            for agent in trainer.agents:
                if len(agent.replay_buffer) > agent.batch_size:
                    # Multiple gradient updates per collection
                    stats = agent.update(update_epochs=4)
                    
                    if stats:
                        for key, value in stats.items():
                            update_stats[key].append(value)
                        
                        if agent.episodes_completed > 0:
                            episode_rewards.append(agent.reward_mean)
            
            # Calculate statistics
            if update_stats:
                avg_stats = {
                    key: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    } for key, values in update_stats.items()
                }
            else:
                avg_stats = {}
            
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                rewards_history.append(avg_reward)
                
                # Print training progress
                print(f" | Reward: {avg_reward:+.2f}", end="")
                
                if avg_stats:
                    critic_loss = avg_stats.get('critic_loss', {}).get('mean', 0)
                    actor_loss = avg_stats.get('actor_loss', {}).get('mean', 0)
                    noise = avg_stats.get('noise_scale', {}).get('mean', 0)
                    q_val = avg_stats.get('q_values', {}).get('mean', 0)
                    
                    critic_loss_history.append(critic_loss)
                    actor_loss_history.append(actor_loss)
                    noise_history.append(noise)
                    q_values_history.append(q_val)
                    
                    print(f" | C-Loss: {critic_loss:.3f} | A-Loss: {actor_loss:.3f} | Noise: {noise:.3f} | Q: {q_val:.2f}", end="")
                
                # Save best model
                if avg_reward > trainer.best_reward:
                    trainer.best_reward = avg_reward
                    trainer.save_model()
                    print(" [BEST]", end="")
                    train_logger.info(f"New best DDPG model saved: {avg_reward:.4f}")
                
                # Log detailed stats to file
                train_logger.info(f"Update {update}: Avg Reward={avg_reward:.4f}, Stats={avg_stats}")
            
            # Periodic checkpoint
            if update % 10 == 0:
                print(f"\n{'='*60}")
                print(f"Checkpoint at update {update}:")
                print(f"  Average reward (last 10): {np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history):.2f}")
                print(f"  Best reward so far: {trainer.best_reward:.2f}")
                print(f"  Total episodes: {sum(a.episodes_completed for a in trainer.agents)}")
                if noise_history:
                    print(f"  Current noise scale: {noise_history[-1]:.4f}")
                if q_values_history:
                    print(f"  Average Q-value: {np.mean(q_values_history[-10:]):.2f}")
                print(f"{'='*60}")
            
            # Save checkpoint periodically
            if update % save_interval == 0:
                trainer.save_model(f"ddpg_models/ddpg_model_update_{update}.pth")
        
        # Final plot
        try:
            plot_ddpg_training_progress(
                rewards_history, 
                critic_loss_history,
                actor_loss_history,
                noise_history,
                q_values_history,
                save_path="ddpg_plots"
            )
        except Exception as e:
            print(f"Error in plotting: {e}")
        
        # Save training history
        training_history = {
            'rewards': convert_to_serializable(rewards_history),
            'critic_loss': convert_to_serializable(critic_loss_history),
            'actor_loss': convert_to_serializable(actor_loss_history),
            'noise': convert_to_serializable(noise_history),
            'q_values': convert_to_serializable(q_values_history)
        }
        
        with open('ddpg_models/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("DDPG TRAINING COMPLETED")
        print(f"Final reward: {rewards_history[-1] if rewards_history else 'N/A':.2f}")
        print(f"Best reward: {trainer.best_reward:.2f}")
        print(f"Total updates: {len(rewards_history)}")
        print(f"Final noise scale: {noise_history[-1] if noise_history else 'N/A':.4f}")
        print(f"{'='*60}")
        
        return None, trainer


if __name__ == "__main__":
    thermal_path = 'C:/Users/unknown/Desktop/DRL/data/thermal_raster_final.tif'
    landcover_path = "C:/Users/unknown/Desktop/DRL/database/aligned_landcover.tif"
    
    # DDPG OPTIMIZED PARAMETERS
    num_workers = 10  # Parallel agents for experience collection
    max_episodes = 100  # Not directly used in DDPG, kept for compatibility
    save_interval = 20  # Save checkpoint every 10 updates
    steps_per_update = 2000  # Steps to collect before each update
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weather_tifs = {
        'soil_moisture': 'C:/Users/unknown/Desktop/DRL/database/aligned_soil_moisture.tif',
        'rainfall': 'C:/Users/unknown/Desktop/DRL/database/aligned_rainfall.tif', 
        'soil_temp': 'C:/Users/unknown/Desktop/DRL/database/aligned_soil_temp.tif',
        'wind_speed': 'C:/Users/unknown/Desktop/DRL/database/aligned_wind_speed.tif',
        'humidity': 'C:/Users/unknown/Desktop/DRL/database/aligned_humidity.tif',
        'dem': 'C:/Users/unknown/Desktop/DRL/database/aligned_dem.tif',
        'ndmi': 'C:/Users/unknown/Desktop/DRL/database/aligned_ndmi.tif'
    }

    try:
        results, trainer = enhanced_ddpg_main_with_real_data(
            weather_tifs=weather_tifs,
            thermal_path=thermal_path,
            landcover_path=landcover_path,
            alignment_method='match_pixels',
            num_workers=num_workers,
            max_episodes=max_episodes,
            device=device,
            save_interval=save_interval,
            steps_per_update=steps_per_update,
            force_retrain=False,  # Set to True to retrain from scratch
            use_parallel_processing=True  # ENABLE PARALLEL PROCESSING
        )

        print("\nDDPG training/evaluation completed successfully!")
        
        # Compare with other algorithms if results exist
        if results is not None:
            print("\n" + "="*60)
            print("COMPARING DDPG WITH OTHER ALGORITHMS")
            print("="*60)
            
            comparison_data = {}
            
            # Load DDPG results
            ddpg_avg = results[results['Patch'] == 'Average'].iloc[0]
            comparison_data['DDPG'] = ddpg_avg
            
            # Try to load PPO results
            if os.path.exists("ppo_results/ppo_results_summary.csv"):
                ppo_results = pd.read_csv("ppo_results/ppo_results_summary.csv")
                ppo_avg = ppo_results[ppo_results['Patch'] == 'Average'].iloc[0]
                comparison_data['PPO'] = ppo_avg
            
            # Try to load DQN results
            if os.path.exists("dqn_results/dqn_results_summary.csv"):
                dqn_results = pd.read_csv("dqn_results/dqn_results_summary.csv")
                dqn_avg = dqn_results[dqn_results['Patch'] == 'Average'].iloc[0]
                comparison_data['DQN'] = dqn_avg
            
            # Try to load SAC results
            if os.path.exists("sac_results/sac_results_summary.csv"):
                sac_results = pd.read_csv("sac_results/sac_results_summary.csv")
                sac_avg = sac_results[sac_results['Patch'] == 'Average'].iloc[0]
                comparison_data['SAC'] = sac_avg
            
            if len(comparison_data) > 1:
                print(f"\n{'Metric':<20}", end="")
                for algo in comparison_data.keys():
                    print(f"{algo:<12}", end="")
                print("Best")
                print("-" * (20 + 12 * len(comparison_data) + 12))
                
                metrics_to_compare = ['AUC', 'Precision', 'Recall', 'F1', 'PR_AUC']
                for metric in metrics_to_compare:
                    print(f"{metric:<20}", end="")
                    values = {}
                    for algo, data in comparison_data.items():
                        val = float(data[metric])
                        values[algo] = val
                        print(f"{val:<12.4f}", end="")
                    
                    best_algo = max(values, key=values.get)
                    print(f"{best_algo:<12}")
                
                # MSE (lower is better)
                print(f"{'MSE':<20}", end="")
                mse_values = {}
                for algo, data in comparison_data.items():
                    val = float(data['MSE'])
                    mse_values[algo] = val
                    print(f"{val:<12.4f}", end="")
                best_algo = min(mse_values, key=mse_values.get)
                print(f"{best_algo:<12}")
                
                # Correlation (higher is better)
                print(f"{'Correlation':<20}", end="")
                corr_values = {}
                for algo, data in comparison_data.items():
                    val = float(data['Pearson_Correlation'])
                    corr_values[algo] = val
                    print(f"{val:<12.4f}", end="")
                best_algo = max(corr_values, key=corr_values.get)
                print(f"{best_algo:<12}")
                
                print("="*60)
                
                # Overall winner
                scores = {algo: 0 for algo in comparison_data.keys()}
                
                # Score for each metric (1 point for best)
                for metric in ['AUC', 'Precision', 'Recall', 'F1', 'PR_AUC', 'Pearson_Correlation']:
                    values = {algo: float(data[metric]) for algo, data in comparison_data.items()}
                    best = max(values, key=values.get)
                    scores[best] += 1
                
                # Score for MSE (lower is better)
                mse_vals = {algo: float(data['MSE']) for algo, data in comparison_data.items()}
                best = min(mse_vals, key=mse_vals.get)
                scores[best] += 1
                
                print(f"\nOVERALL PERFORMANCE SCORES:")
                for algo, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {algo}: {score}/7 metrics")
                
                winner = max(scores, key=scores.get)
                print(f"\n🏆 BEST ALGORITHM: {winner}")
                print("="*60)
        
    except Exception as e:
        print(f"Error during DDPG training: {str(e)}")
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