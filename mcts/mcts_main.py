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

# IMPORTANT: Set multiprocessing start method for Windows
import multiprocessing as mp
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

def setup_logging():
    """Setup logging for MCTS - file only, no console"""
    os.makedirs('logs', exist_ok=True)
    
    # Clear existing handlers
    logging.getLogger().handlers = []
    
    # Setup logger
    logger = logging.getLogger('MCTS')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for file
    logger.propagate = False
    
    # File handler ONLY - no console handler
    from datetime import datetime
    file_handler = logging.FileHandler(
        f'logs/mcts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(file_handler)
    
    # Write header to log file
    logger.info("="*80)
    logger.info("MCTS LOG FILE")
    logger.info(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
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

def run_mcts_with_enhanced_env(thermal_data: np.ndarray,
                               start_pos: Tuple[int, int],
                               weather_patches: Dict[str, np.ndarray],
                               landcover_data: np.ndarray,
                               max_steps: int = 500,
                               num_iterations: int = 1000,
                               logger=None) -> Tuple[np.ndarray, Dict]:
    """
    Run MCTS with EnhancedCropThermalEnv
    
    Returns:
        value_map: Value map as numpy array
        policy: Policy dictionary
    """
    # Import here to avoid loading torch in main process
    from mcts.mcts import MCTSOptimized
    from environment.env_src import create_enhanced_crop_thermal_env
    
    if logger:
        logger.debug(f"Creating environment with shape {thermal_data.shape}")
    
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
    
    # Add gamma if not present
    if not hasattr(env, 'gamma'):
        env.gamma = 0.9
    
    # Create MCTS solver with logger
    import math
    mcts = MCTSOptimized(env, c_param=math.sqrt(2), max_rollout_steps=50, logger=logger)
    
    # Run MCTS search
    root = mcts.search(start_pos, num_iterations=num_iterations)
    
    # Get value map and policy
    value_map = mcts.get_value_map(root)
    policy = mcts.get_policy(root)
    
    env.close()
    
    return value_map, policy

def process_single_patch_mcts(patch_data, step, transform_affine, 
                              src_crs, patch_coords, original_shape, 
                              num_iterations=1000, logger=None):
    """Process a single patch using MCTS with EnhancedCropThermalEnv"""
    try:
        # Set thread count to 1 to avoid memory issues
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        i = patch_data['index']
        
        if logger:
            logger.info(f"Processing patch {i} with MCTS")
            logger.debug(f"  Thermal shape: {patch_data['thermal_data'].shape}")
            logger.debug(f"  Start position: {patch_data['start_pos']}")
            logger.debug(f"  MCTS iterations: {num_iterations}")
        
        # Run MCTS with Enhanced Environment
        value_map, policy = run_mcts_with_enhanced_env(
            thermal_data=patch_data['thermal_data'],
            start_pos=patch_data['start_pos'],
            weather_patches=patch_data['weather_patches'],
            landcover_data=patch_data['landcover_data'],
            max_steps=200,
            num_iterations=num_iterations,
            logger=logger
        )
        
        # Convert value map to dictionary for compatibility with existing utils
        V = {}
        height, width = patch_data['thermal_data'].shape
        for x in range(height):
            for y in range(width):
                V[(x, y)] = value_map[x, y]
        
        if logger:
            logger.debug(f"  Value map computed, mean={np.mean(value_map):.3f}")
        
        # Suppress prints from utils
        with SuppressPrints():
            # Compute AUC
            auc, pred_map = compute_auc_and_plot_full(
                V, patch_data['y_true_binary'], patch_data['thermal_data'], 
                crop_id=i + 1, no_value_iteration=False
            )
            
            # Compute MSE and correlation
            mse, corr = compute_mse_corr(pred_map, patch_data['thermal_data'])
            
            # Compute confusion overlay
            overlay, precision, recall, f1, pr_auc = compute_confusion_overlay(
                pred_map, patch_data['y_true_binary'], i + 1
            )
            
            # Save TIF confusion
            save_tif_confusion(original_shape, step, i, transform_affine, overlay, src_crs)
        
        # Save combined confusion
        lat_dms, lon_dms, lat_dms_hotspot, lon_dms_hotspot = save_combine_confusion(
            i, patch_coords, patch_data['start_pos'], overlay, patch_data['T_Celsius']
        )
        
        # Close plots to free memory
        plt.close('all')
        
        # Force garbage collection
        gc.collect()
        
        if logger:
            logger.info(f"  Patch {i} completed: AUC={auc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        
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
        if logger:
            logger.error(f"Error processing patch {patch_data.get('index', 'unknown')}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        return {'success': False, 'index': patch_data.get('index', -1), 'error': str(e)}

def process_patches_sequential(patch_list, step, transform_affine, 
                               src_crs, patch_coords, original_shape, 
                               num_iterations=1000, logger=None):
    """Process patches sequentially to avoid multiprocessing issues"""
    print(f"\nProcessing {len(patch_list)} patches sequentially...")
    
    if logger:
        logger.info("="*60)
        logger.info(f"Starting sequential processing of {len(patch_list)} patches")
        logger.info(f"MCTS iterations per patch: {num_iterations}")
        logger.info("="*60)
    
    all_results = []
    failed_patches = []
    
    # Use tqdm for progress bar only
    for patch in tqdm(patch_list, desc="Processing patches", ncols=100):
        try:
            if logger:
                logger.info(f"\n--- Processing patch {patch['index']} ---")
            
            result = process_single_patch_mcts(
                patch, step, transform_affine,
                src_crs, patch_coords, original_shape,
                num_iterations=num_iterations,
                logger=logger
            )
            
            if result['success']:
                all_results.append(result)
                
                # Log progress stats to file only
                if len(all_results) % 10 == 0 and logger:
                    recent_results = all_results[-min(10, len(all_results)):]
                    avg_auc = np.mean([r['AUC'] for r in recent_results])
                    avg_f1 = np.mean([r['F1'] for r in recent_results])
                    logger.info(f"Progress: {len(all_results)}/{len(patch_list)} | Avg AUC: {avg_auc:.3f} | Avg F1: {avg_f1:.3f}")
            else:
                failed_patches.append(result['index'])
                if logger:
                    logger.warning(f"Patch {result['index']} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            if logger:
                logger.error(f"Error processing patch {patch.get('index', -1)}: {str(e)}")
            failed_patches.append(patch.get('index', -1))
        
        # Memory cleanup every 10 patches
        if len(all_results) % 10 == 0:
            gc.collect()
    
    # Final summary
    if failed_patches:
        print(f"\n[WARNING] {len(failed_patches)} patches failed")
        if logger:
            logger.warning(f"{len(failed_patches)} patches failed: {failed_patches[:10]}...")
    
    print(f"\n[OK] Successfully processed {len(all_results)}/{len(patch_list)} patches")
    if logger:
        logger.info("="*60)
        logger.info(f"Successfully processed {len(all_results)}/{len(patch_list)} patches")
        logger.info("="*60)
    
    return all_results

def mcts_main_with_format(thermal_path: str,
                         landcover_path: str,
                         weather_tifs: dict,
                         alignment_method='match_pixels',
                         num_iterations=1000,
                         use_parallel_processing=False):
    """
    Main function for MCTS with clean console output
    Using MCTS with EnhancedCropThermalEnv
    """
    
    logger = setup_logging()
    
    print("="*80)
    print("MONTE CARLO TREE SEARCH WITH ENHANCED ENVIRONMENT")
    print(f"Log file: logs/mcts_*.log")
    print("="*80)
    
    # Write detailed info to log only
    logger.info("="*80)
    logger.info("MONTE CARLO TREE SEARCH WITH ENHANCED ENVIRONMENT")
    logger.info(f"Thermal path: {thermal_path}")
    logger.info(f"Landcover path: {landcover_path}")
    logger.info(f"MCTS iterations per patch: {num_iterations}")
    logger.info(f"Parallel processing: {use_parallel_processing}")
    logger.info("="*80)
    
    # Create result directories
    result_dirs = ["combined_frames", "confusion_maps", "final_maps", "mcts_results"]
    for folder in result_dirs:
        os.makedirs(folder, exist_ok=True)
    
    # Read weather TIF files
    print("Loading weather data...")
    weather_data = {}
    for name, path in weather_tifs.items():
        with rasterio.open(path) as src:
            weather_data[name] = src.read(1)
        logger.info(f"Loaded weather data: {name} from {path}")
    
    # Read and patch thermal image
    print("Reading and patching thermal image...")
    logger.info("Reading and patching thermal image...")
    patch_size = 100
    overlap = 10
    patches, original_shape, patch_coords, transform_affine, src_crs = read_and_patch_thermal_image(
        image_path=thermal_path, patch_size=patch_size, overlap=overlap
    )
    logger.info(f"Original shape: {original_shape}, Patch size: {patch_size}, Overlap: {overlap}")
    
    # Align landcover with thermal data
    print("Aligning landcover with thermal data...")
    logger.info("Aligning landcover with thermal data...")
    landcover_patches = read_and_align_landcover_to_thermal(
        landcover_path, thermal_path, patch_size, overlap, method=alignment_method
    )
    
    if len(landcover_patches) > 0:
        full_aligned = reconstruct_from_patches(landcover_patches, original_shape, patch_size, overlap)
        logger.info("Saved aligned landcover for debug")
    
    print(f"Found {len(patches)} thermal patches and {len(landcover_patches)} landcover patches")
    logger.info(f"Found {len(patches)} thermal patches and {len(landcover_patches)} landcover patches")
    
    # Ensure matching number of patches
    min_patches = min(len(patches), len(landcover_patches))
    patches = patches[:min_patches]
    landcover_patches = landcover_patches[:min_patches]
    patch_coords = patch_coords[:min_patches]
    
    # Prepare patch list for processing
    patch_list = []
    valid_count = 0
    invalid_count = 0
    
    logger.info("Preparing patches for processing...")
    for i, (T_Celsius, landcover_patch, coord) in enumerate(zip(patches, landcover_patches, patch_coords)):
        T_Celsius, normalized_temp, y_true_binary, _, _ = prepare_temp_data_balanced(T_Celsius)

        if (
            np.any(T_Celsius <= 0.0) or 
            np.sum(y_true_binary) == 0 or  
            np.sum(y_true_binary) == y_true_binary.size): 
            invalid_count += 1
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
        valid_count += 1
        logger.debug(f"Patch {i} added: max_temp={np.max(T_Celsius):.1f}, start_pos={highest_temp_pos}")
    
    print(f"{valid_count} valid patches ready for processing")
    logger.info(f"{valid_count} valid patches ready, {invalid_count} patches skipped")
    
    # Calculate step for patches
    step = patch_size - overlap
    
    # Process patches (MCTS is inherently sequential)
    print("\nUsing SEQUENTIAL processing (MCTS)...")
    logger.info("Using sequential processing for MCTS...")
    all_results = process_patches_sequential(
        patch_list, step, transform_affine,
        src_crs, patch_coords, original_shape,
        num_iterations=num_iterations,
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
        logger.info("Exporting final confusion map...")
        export_final_confusion_map(patch_size, overlap, step, thermal_path)
        logger.info("Final confusion map exported successfully")
    except Exception as e:
        warning_msg = f"Could not export final confusion map: {str(e)}"
        logger.warning(warning_msg)
    
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
        csv_path = "mcts_results/mcts_results_summary.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        logger.info(f"Results saved to {csv_path}")
        
        # Save as JSON for compatibility
        results_json = {
            'results': convert_to_serializable(all_results),
            'average_metrics': convert_to_serializable(dict(avg_metrics)),
            'method': 'monte_carlo_tree_search',
            'num_patches_processed': len(all_results),
            'mcts_iterations': num_iterations,
            'patch_size': patch_size,
            'overlap': overlap
        }
        
        with open('mcts_results/mcts_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info("Results saved to JSON file")
        
        # Print summary to console
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
        
        # Log detailed results
        logger.info("="*60)
        logger.info(f"FINAL RESULTS SUMMARY ({len(all_results)} patches processed)")
        logger.info("="*60)
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("="*60)
        logger.info("Processing completed successfully")
        
        return df_results
        
    except Exception as e:
        error_msg = f"Failed to create results summary: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
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
    
    # Run MCTS with safe configuration for Windows
    try:
        results = mcts_main_with_format(
            thermal_path=thermal_path,
            landcover_path=landcover_path,
            weather_tifs=weather_tifs,
            alignment_method='match_pixels',
            num_iterations=1000,  # MCTS iterations per patch
            use_parallel_processing=False  # MCTS is sequential by nature
        )
        
        print("\nMCTS completed successfully!")
        
    except Exception as e:
        print(f"Error during MCTS: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDone!")