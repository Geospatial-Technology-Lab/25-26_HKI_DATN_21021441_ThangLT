import os
import pandas as pd
import numpy as np
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
import json
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# IMPORTANT: Set multiprocessing start method for Windows
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

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

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def clean_old_logs(keep_last=5):
    """Xóa các file log cũ, chỉ giữ lại N file mới nhất"""
    import glob
    
    try:
        if not os.path.exists('logs'):
            return
            
        log_files = sorted(glob.glob('logs/q_learning_*.log'), 
                          key=os.path.getmtime, reverse=True)
        
        # Xóa các file cũ
        for log_file in log_files[keep_last:]:
            try:
                os.remove(log_file)
                print(f"Deleted old log: {os.path.basename(log_file)}")
            except:
                pass
    except Exception as e:
        print(f"Could not clean old logs: {e}")

def setup_logging(log_level='WARNING'):
    """Setup logging for Q-Learning - chỉ tạo một file log duy nhất cho main process"""
    os.makedirs('logs', exist_ok=True)
    
    # Clear existing handlers
    logging.getLogger().handlers = []
    
    # Setup logger
    logger = logging.getLogger('QLearning')
    logger.setLevel(getattr(logging, log_level))
    logger.propagate = False
    
    # Tạo file log với timestamp
    from datetime import datetime
    log_file = f'logs/q_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def run_q_learning_with_enhanced_env_optimized(thermal_data: np.ndarray,
                                               start_pos: Tuple[int, int],
                                               weather_patches: Dict[str, np.ndarray],
                                               landcover_data: np.ndarray,
                                               max_steps: int = 200,
                                               num_episodes: int = 100,
                                               early_stopping: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Optimized Q-Learning with EnhancedCropThermalEnv
    Không sử dụng logger để tránh tạo nhiều file log
    """
    # Import here to avoid loading torch in main process
    from q_learning.qlearn import QLearningOptimized
    from environment.env_src import create_enhanced_crop_thermal_env
    
    # Create EnhancedCropThermalEnv
    env = create_enhanced_crop_thermal_env(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        obey_prob=0.9,
        high_temp_threshold=0.95,
        medium_temp_threshold=0.85,
        verbose=False
    )
    
    # Create Q-Learning agent with optimized parameters
    q_agent = QLearningOptimized(
        env,
        alpha=0.15,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        logger=None,  # Không sử dụng logger trong worker processes
        early_stopping=early_stopping
    )
    
    # Train agent with optimized parameters
    training_results = q_agent.train(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        convergence_threshold=1e-3,
        patience=30
    )
    
    # Get value function and policy
    value_function = q_agent.get_value_function()
    policy = q_agent.get_policy()
    
    env.close()
    
    return value_function, policy

def process_single_patch_q_learning_optimized(args):
    """Optimized single patch processing for parallel execution - KHÔNG TẠO LOG FILE"""
    (patch_data, step, transform_affine, src_crs, 
     patch_coords, original_shape, num_episodes, early_stopping) = args
    
    # KHÔNG tạo logger cho worker processes để tránh tạo nhiều file log

    try:
        # Set thread count to 1 to avoid memory issues
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        i = patch_data['index']
        
        # Run Q-Learning with Enhanced Environment
        value_function, policy = run_q_learning_with_enhanced_env_optimized(
            thermal_data=patch_data['thermal_data'],
            start_pos=patch_data['start_pos'],
            weather_patches=patch_data['weather_patches'],
            landcover_data=patch_data['landcover_data'],
            max_steps=200,
            num_episodes=num_episodes,
            early_stopping=early_stopping
        )
        
        # Convert value function to dictionary for compatibility
        V = {}
        height, width = patch_data['thermal_data'].shape
        for x in range(height):
            for y in range(width):
                V[(x, y)] = value_function[x, y]
        
        # Suppress prints from utils
        with SuppressPrints():
            # Compute metrics
            auc, pred_map = compute_auc_and_plot_full(
                V, patch_data['y_true_binary'], patch_data['thermal_data'], 
                crop_id=i + 1, no_value_iteration=False
            )
            
            mse, corr = compute_mse_corr(pred_map, patch_data['thermal_data'])
            
            overlay, precision, recall, f1, pr_auc = compute_confusion_overlay(
                pred_map, patch_data['y_true_binary'], i + 1
            )
            
            save_tif_confusion(original_shape, step, i, transform_affine, overlay, src_crs)
        
        lat_dms, lon_dms, lat_dms_hotspot, lon_dms_hotspot = save_combine_confusion(
            i, patch_coords, patch_data['start_pos'], overlay, patch_data['T_Celsius']
        )
        
        # Close plots to free memory
        plt.close('all')
        gc.collect()

        return {
            'success': True,
            'Crop': i + 1,
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
        return {'success': False, 'index': patch_data.get('index', -1), 'error': str(e)}

def process_patches_parallel(patch_list, step, transform_affine, 
                            src_crs, patch_coords, original_shape, 
                            num_episodes=100, num_workers=None,
                            early_stopping=True, logger=None):
    """Process patches in parallel for better performance"""
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 8)  # Cap at 8 workers
    
    print(f"\nProcessing {len(patch_list)} patches with {num_workers} workers...")
    if logger:
        logger.info(f"Starting parallel processing with {num_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [
        (patch, step, transform_affine, src_crs, patch_coords, 
         original_shape, num_episodes, early_stopping)
        for patch in patch_list
    ]
    
    all_results = []
    failed_patches = []
    
    # Use ProcessPoolExecutor for better control
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_patch = {
            executor.submit(process_single_patch_q_learning_optimized, args): args[0]['index']
            for args in args_list
        }
        
        # Process results as they complete
        with tqdm(total=len(patch_list), desc="Processing patches", ncols=100) as pbar:
            for future in as_completed(future_to_patch):
                patch_idx = future_to_patch[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per patch
                    if result['success']:
                        all_results.append(result)
                        # Log vào main logger thay vì tạo logger mới
                        if logger:
                            logger.info(f"Patch {patch_idx} processed: AUC={result['AUC']:.4f}, F1={result['F1']:.4f}")
                    else:
                        failed_patches.append(result['index'])
                        if logger:
                            logger.warning(f"Patch {patch_idx} failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    failed_patches.append(patch_idx)
                    if logger:
                        logger.error(f"Patch {patch_idx} failed: {str(e)}")
                
                pbar.update(1)
                
                # Log progress
                if len(all_results) % 50 == 0 and len(all_results) > 0:
                    recent_results = all_results[-min(50, len(all_results)):]
                    avg_auc = np.mean([r['AUC'] for r in recent_results])
                    avg_f1 = np.mean([r['F1'] for r in recent_results])
                    pbar.set_postfix({'AUC': f'{avg_auc:.3f}', 'F1': f'{avg_f1:.3f}'})
    
    # Final summary
    if failed_patches:
        print(f"\n[WARNING] {len(failed_patches)} patches failed")
        if logger:
            logger.warning(f"{len(failed_patches)} patches failed: {failed_patches}")
    
    print(f"\n[OK] Successfully processed {len(all_results)}/{len(patch_list)} patches")
    if logger:
        logger.info(f"Successfully processed {len(all_results)}/{len(patch_list)} patches")
    
    return all_results

def q_learning_main_optimized(thermal_path: str,
                              landcover_path: str,
                              weather_tifs: dict,
                              alignment_method='match_pixels',
                              num_episodes=500,
                              use_parallel_processing=True,
                              num_workers=None,
                              sample_ratio=1.0,
                              early_stopping=True,
                              log_level='WARNING'):
    """
    Optimized main function for Q-Learning with parallel processing
    """
    
    # Dọn dẹp log cũ trước khi bắt đầu
    clean_old_logs(keep_last=5)
    
    # Chỉ tạo một logger duy nhất cho main process
    logger = setup_logging(log_level)
    
    print("="*80)
    print("Q-LEARNING WITH ENHANCED ENVIRONMENT (OPTIMIZED)")
    print(f"Episodes: {num_episodes} | Parallel: {use_parallel_processing} | Sample: {sample_ratio*100:.0f}%")
    print(f"Log file: logs/q_learning_*.log (single file only)")
    print("="*80)
    
    logger.info("="*80)
    logger.info("Q-LEARNING WITH ENHANCED ENVIRONMENT (OPTIMIZED)")
    logger.info(f"Configuration:")
    logger.info(f"  - Episodes per patch: {num_episodes}")
    logger.info(f"  - Parallel processing: {use_parallel_processing}")
    logger.info(f"  - Sample ratio: {sample_ratio}")
    logger.info(f"  - Early stopping: {early_stopping}")
    logger.info("="*80)
    
    # Create result directories
    result_dirs = ["combined_frames", "confusion_maps", "final_maps", "q_results"]
    for folder in result_dirs:
        os.makedirs(folder, exist_ok=True)
    
    # Read weather TIF files
    print("Loading weather data...")
    weather_data = {}
    for name, path in weather_tifs.items():
        with rasterio.open(path) as src:
            weather_data[name] = src.read(1)
        logger.info(f"Loaded weather data: {name}")
    
    # Read and patch thermal image
    print("Reading and patching thermal image...")
    patch_size = 100
    overlap = 10
    patches, original_shape, patch_coords, transform_affine, src_crs = read_and_patch_thermal_image(
        image_path=thermal_path, patch_size=patch_size, overlap=overlap
    )
    
    # Align landcover with thermal data
    print("Aligning landcover with thermal data...")
    landcover_patches = read_and_align_landcover_to_thermal(
        landcover_path, thermal_path, patch_size, overlap, method=alignment_method
    )
    
    if len(landcover_patches) > 0:
        full_aligned = reconstruct_from_patches(landcover_patches, original_shape, patch_size, overlap)
    
    print(f"Found {len(patches)} thermal patches and {len(landcover_patches)} landcover patches")
    logger.info(f"Found {len(patches)} thermal patches and {len(landcover_patches)} landcover patches")
    
    # Ensure matching number of patches
    min_patches = min(len(patches), len(landcover_patches))
    patches = patches[:min_patches]
    landcover_patches = landcover_patches[:min_patches]
    patch_coords = patch_coords[:min_patches]
    
    # Prepare patch list for processing
    patch_list = []
    
    for i, (T_Celsius, landcover_patch, coord) in enumerate(zip(patches, landcover_patches, patch_coords)):
        T_Celsius, normalized_temp, y_true_binary, _, _ = prepare_temp_data_balanced(T_Celsius)

        if ( 
            np.any(T_Celsius <= 0.0) or 
            np.sum(y_true_binary) == 0 or  
            np.sum(y_true_binary) == y_true_binary.size): 
            continue
        
        # Get highest temperature position
        highest_temp_pos = np.unravel_index(np.argmax(T_Celsius), T_Celsius.shape)
        
        # Process weather patches for this patch
        weather_patches = process_weather_patches(weather_data, coord, patch_size)
        
        patch_list.append({
            'index': i,
            'thermal_data': normalized_temp,
            'T_Celsius': T_Celsius,
            'start_pos': highest_temp_pos,
            'weather_patches': weather_patches,
            'landcover_data': landcover_patch,
            'y_true_binary': y_true_binary,
            'coord': coord
        })
    
    # Apply sampling if requested
    if sample_ratio < 1.0:
        original_count = len(patch_list)
        sample_size = max(1, int(len(patch_list) * sample_ratio))
        patch_list = random.sample(patch_list, sample_size)
        print(f"Sampling {sample_size}/{original_count} patches ({sample_ratio*100:.0f}%)")
        logger.info(f"Sampled {sample_size}/{original_count} patches")
    
    print(f"{len(patch_list)} patches ready for processing")
    logger.info(f"{len(patch_list)} patches ready for processing")
    
    # Calculate step for patches
    step = patch_size - overlap
    
    # Process patches
    if use_parallel_processing and len(patch_list) > 10:
        print(f"\nUsing PARALLEL processing with {num_workers or 'auto'} workers...")
        all_results = process_patches_parallel(
            patch_list, step, transform_affine,
            src_crs, patch_coords, original_shape,
            num_episodes=num_episodes,
            num_workers=num_workers,
            early_stopping=early_stopping,
            logger=logger  # Truyền logger từ main process
        )
    else:
        print("\nUsing SEQUENTIAL processing...")
        from q_learning.qlearn import process_patches_sequential
        all_results = process_patches_sequential(
            patch_list, step, transform_affine,
            src_crs, patch_coords, original_shape,
            num_episodes=num_episodes,
            logger=logger
        )
    
    # Check if we have results
    if not all_results:
        error_msg = "No patches were successfully processed."
        print(f"\n[ERROR] {error_msg}")
        logger.error(error_msg)
        return None
    
    # Export final confusion map
    try:
        print("\nExporting final confusion map...")
        export_final_confusion_map(patch_size, overlap, step, thermal_path)
        logger.info("Final confusion map exported successfully")
    except Exception as e:
        logger.warning(f"Could not export final confusion map: {str(e)}")
    
    # Create results dataframe
    try:
        df_results = pd.DataFrame(all_results)
        
        # Calculate average metrics
        numeric_cols = ['AUC', 'MSE', 'Pearson_Correlation', 'Precision', 'Recall', 'F1', 'PR_AUC']
        
        # Ensure all columns exist
        for col in numeric_cols:
            if col not in df_results.columns:
                df_results[col] = 0.0
        
        df_numeric = df_results[numeric_cols].apply(pd.to_numeric, errors='coerce')
        avg_metrics = df_numeric.mean(numeric_only=True)
        
        # Add summary row
        summary_row = {col: float(avg_metrics.get(col, 0)) for col in numeric_cols}
        summary_row.update({
            'Crop': 'Average',
            'Lat': '',
            'Lon': '',
            'lat_hotspot': '',
            'lon_hotspot': '',
            'Max_Temp': float(df_results['Max_Temp'].mean()) if 'Max_Temp' in df_results else 0
        })
        df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
        
        # Save results
        csv_path = "q_results/q_learning_results_summary_optimized.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save as JSON
        results_json = {
            'results': convert_to_serializable(all_results),
            'average_metrics': convert_to_serializable(dict(avg_metrics)),
            'method': 'q_learning_optimized',
            'num_patches_processed': len(all_results),
            'q_learning_episodes': num_episodes,
            'patch_size': patch_size,
            'overlap': overlap,
            'parallel_processing': use_parallel_processing,
            'sample_ratio': sample_ratio
        }
        
        with open('q_results/q_learning_results_optimized.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Print summary
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
        
        # Log summary
        logger.info("="*60)
        logger.info(f"RESULTS SUMMARY ({len(all_results)} patches processed)")
        logger.info(f"Average AUC: {avg_metrics.get('AUC', 0):.4f}")
        logger.info(f"Average F1: {avg_metrics.get('F1', 0):.4f}")
        logger.info(f"Average MSE: {avg_metrics.get('MSE', 0):.4f}")
        logger.info("="*60)
        
        return df_results
        
    except Exception as e:
        error_msg = f"Failed to create results summary: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        logger.error(error_msg)
        return None

if __name__ == "__main__":
    # Configuration paths
    thermal_path = 'C:/Users/unknown/Desktop/DRL/data/thermal_raster_final.tif'
    landcover_path = "C:/Users/unknown/Desktop/DRL/database/aligned_landcover.tif"
    
    weather_tifs = {
        'soil_moisture': 'C:/Users/unknown/Desktop/DRL/database/aligned_soil_moisture.tif',
        'rainfall': 'C:/Users/unknown/Desktop/DRL/database/aligned_rainfall.tif', 
        'soil_temp': 'C:/Users/unknown/Desktop/DRL/database/aligned_soil_temp.tif',
        'wind_speed': 'C:/Users/unknown/Desktop/DRL/database/aligned_wind_speed.tif',
        'humidity': 'C:/Users/unknown/Desktop/DRL/database/aligned_humidity.tif',
        'dem': 'C:/Users/unknown/Desktop/DRL/database/aligned_dem.tif',
        'ndmi': 'C:/Users/unknown/Desktop/DRL/database/aligned_ndmi.tif'
    }
    # Run optimized Q-Learning
    try:
        results = q_learning_main_optimized(
            thermal_path=thermal_path,
            landcover_path=landcover_path,
            weather_tifs=weather_tifs,
            alignment_method='match_pixels',
            num_episodes=200,
            use_parallel_processing=True,
            num_workers=4,
            sample_ratio=1.0,
            early_stopping=True,
            log_level='INFO'
        )
        
        print("\nOptimized Q-Learning completed successfully!")
        print("Note: Only ONE log file was created in logs/ directory")
        
    except Exception as e:
        print(f"Error during Q-Learning: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDone!")