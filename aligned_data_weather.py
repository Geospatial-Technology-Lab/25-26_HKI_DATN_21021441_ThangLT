import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os

def validate_and_align_tif_files(thermal_path, landcover_path, weather_tifs, output_dir="database"):
    """
    Validate and align all TIF files to match thermal image dimensions and spatial properties
    """
    print("="*80)
    print("VALIDATING AND ALIGNING TIF FILES")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read thermal image as reference
    print(f"\n1. Reading reference thermal image: {thermal_path}")
    with rasterio.open(thermal_path) as thermal_src:
        thermal_data = thermal_src.read(1)
        thermal_transform = thermal_src.transform
        thermal_crs = thermal_src.crs
        thermal_bounds = thermal_src.bounds
        thermal_shape = thermal_data.shape
        
        print(f"   Thermal shape: {thermal_shape}")
        print(f"   Thermal bounds: {thermal_bounds}")
        print(f"   Thermal CRS: {thermal_crs}")
        print(f"   Thermal transform: {thermal_transform}")
        print(f"   Thermal data range: {thermal_data.min():.3f} - {thermal_data.max():.3f}")
    
    reference_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': thermal_shape[1],
        'height': thermal_shape[0],
        'count': 1,
        'crs': thermal_crs,
        'transform': thermal_transform,
        'compress': 'lzw'
    }
    
    aligned_files = {}
    alignment_report = []
    
    # 2. Check and align landcover
    print(f"\n2. Checking landcover file: {landcover_path}")
    try:
        with rasterio.open(landcover_path) as landcover_src:
            landcover_data = landcover_src.read(1)
            landcover_shape = landcover_data.shape
            landcover_crs = landcover_src.crs
            landcover_bounds = landcover_src.bounds
            landcover_transform = landcover_src.transform
            
            print(f"   Landcover shape: {landcover_shape}")
            print(f"   Landcover bounds: {landcover_bounds}")
            print(f"   Landcover CRS: {landcover_crs}")
            print(f"   Landcover data range: {landcover_data.min()} - {landcover_data.max()}")
            
            # Check if alignment is needed
            needs_alignment = (
                landcover_shape != thermal_shape or
                landcover_crs != thermal_crs or
                not np.allclose([landcover_bounds[i] for i in range(4)], 
                              [thermal_bounds[i] for i in range(4)], rtol=1e-6)
            )
            
            if needs_alignment:
                print("LANDCOVER NEEDS ALIGNMENT")
                
                # Reproject landcover to match thermal
                aligned_landcover = np.empty(thermal_shape, dtype=landcover_data.dtype)
                
                reproject(
                    source=landcover_data,
                    destination=aligned_landcover,
                    src_transform=landcover_transform,
                    src_crs=landcover_crs,
                    dst_transform=thermal_transform,
                    dst_crs=thermal_crs,
                    resampling=Resampling.cubic  # Use nearest for categorical data
                )
                
                # Save aligned landcover
                aligned_landcover_path = os.path.join(output_dir, "aligned_landcover.tif")
                with rasterio.open(aligned_landcover_path, 'w', **reference_profile) as dst:
                    dst.write(aligned_landcover.astype(np.float32), 1)
                
                aligned_files['landcover'] = aligned_landcover_path
                alignment_report.append({
                    'file': 'landcover',
                    'original_shape': landcover_shape,
                    'aligned_shape': thermal_shape,
                    'alignment_needed': True,
                    'aligned_path': aligned_landcover_path
                })
                print(f"Aligned landcover saved to: {aligned_landcover_path}")
                
            else:
                print("Landcover already aligned")
                aligned_files['landcover'] = landcover_path
                alignment_report.append({
                    'file': 'landcover',
                    'original_shape': landcover_shape,
                    'aligned_shape': thermal_shape,
                    'alignment_needed': False,
                    'aligned_path': landcover_path
                })
                
    except Exception as e:
        print(f"Error processing landcover: {e}")
        raise
    
    # 3. Check and align weather files
    print(f"\n3. Checking weather files:")
    for weather_name, weather_path in weather_tifs.items():
        print(f"\n   Processing {weather_name}: {weather_path}")
        
        try:
            with rasterio.open(weather_path) as weather_src:
                weather_data = weather_src.read(1)
                weather_shape = weather_data.shape
                weather_crs = weather_src.crs
                weather_bounds = weather_src.bounds
                weather_transform = weather_src.transform
                
                print(f"     Shape: {weather_shape}")
                print(f"     Bounds: {weather_bounds}")
                print(f"     CRS: {weather_crs}")
                print(f"     Data range: {weather_data.min():.6f} - {weather_data.max():.6f}")
                print(f"     Has NaN: {np.isnan(weather_data).any()}")
                print(f"     Has Inf: {np.isinf(weather_data).any()}")
                
                # Check if alignment is needed
                needs_alignment = (
                    weather_shape != thermal_shape or
                    weather_crs != thermal_crs or
                    not np.allclose([weather_bounds[i] for i in range(4)], 
                                  [thermal_bounds[i] for i in range(4)], rtol=1e-6)
                )
                
                if needs_alignment:
                    print(f"{weather_name.upper()} NEEDS ALIGNMENT")
                    
                    # Reproject weather data to match thermal
                    aligned_weather = np.empty(thermal_shape, dtype=np.float32)
                    
                    reproject(
                        source=weather_data,
                        destination=aligned_weather,
                        src_transform=weather_transform,
                        src_crs=weather_crs,
                        dst_transform=thermal_transform,
                        dst_crs=thermal_crs,
                        resampling=Resampling.cubic  # Use bilinear for continuous data
                    )
                    
                    # Handle NaN values
                    if np.isnan(aligned_weather).any():
                        print(f"Fixing {np.sum(np.isnan(aligned_weather))} NaN values")
                        # Fill NaN with mean of valid values
                        valid_mask = ~np.isnan(aligned_weather)
                        if valid_mask.any():
                            mean_value = np.mean(aligned_weather[valid_mask])
                            aligned_weather[np.isnan(aligned_weather)] = mean_value
                        else:
                            aligned_weather[:] = 0.0
                    
                    # Save aligned weather data
                    aligned_weather_path = os.path.join(output_dir, f"aligned_{weather_name}.tif")
                    with rasterio.open(aligned_weather_path, 'w', **reference_profile) as dst:
                        dst.write(aligned_weather, 1)
                    
                    aligned_files[weather_name] = aligned_weather_path
                    alignment_report.append({
                        'file': weather_name,
                        'original_shape': weather_shape,
                        'aligned_shape': thermal_shape,
                        'alignment_needed': True,
                        'aligned_path': aligned_weather_path,
                        'data_range': f"{aligned_weather.min():.6f} - {aligned_weather.max():.6f}"
                    })
                    print(f"Aligned {weather_name} saved to: {aligned_weather_path}")
                    
                else:
                    print(f"{weather_name} already aligned")
                    aligned_files[weather_name] = weather_path
                    alignment_report.append({
                        'file': weather_name,
                        'original_shape': weather_shape,
                        'aligned_shape': thermal_shape,
                        'alignment_needed': False,
                        'aligned_path': weather_path,
                        'data_range': f"{weather_data.min():.6f} - {weather_data.max():.6f}"
                    })
                    
        except Exception as e:
            print(f"Error processing {weather_name}: {e}")
            raise
    
    # 4. Generate validation report
    print(f"\n4. Generating validation report")
    report_path = os.path.join(output_dir, "alignment_report.txt")
    with open(report_path, 'w') as f:
        f.write("TIF FILES ALIGNMENT REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Reference (Thermal): {thermal_path}\n")
        f.write(f"Reference shape: {thermal_shape}\n")
        f.write(f"Reference CRS: {thermal_crs}\n")
        f.write(f"Reference bounds: {thermal_bounds}\n\n")
        
        for report in alignment_report:
            f.write(f"File: {report['file']}\n")
            f.write(f"Original shape: {report['original_shape']}\n")
            f.write(f"Aligned shape: {report['aligned_shape']}\n")
            f.write(f"Alignment needed: {report['alignment_needed']}\n")
            f.write(f"Aligned path: {report['aligned_path']}\n")
            if 'data_range' in report:
                f.write(f"Data range: {report['data_range']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"Validation report saved to: {report_path}")
    
    # 5. Final validation - load all aligned files and verify
    print(f"\n5. Final verification of aligned files")
    verification_passed = True
    
    for file_type, file_path in aligned_files.items():
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                if data.shape != thermal_shape:
                    print(f"   ❌ {file_type}: Shape mismatch {data.shape} != {thermal_shape}")
                    verification_passed = False
                elif src.crs != thermal_crs:
                    print(f"   ❌ {file_type}: CRS mismatch {src.crs} != {thermal_crs}")
                    verification_passed = False
                elif not np.allclose([src.bounds[i] for i in range(4)], 
                                   [thermal_bounds[i] for i in range(4)], rtol=1e-6):
                    print(f"   ❌ {file_type}: Bounds mismatch")
                    verification_passed = False
                else:
                    print(f"   ✅ {file_type}: All checks passed")
        except Exception as e:
            print(f"   ❌ {file_type}: Verification error: {e}")
            verification_passed = False
    
    if verification_passed:
        print(f"\n🎉 ALL FILES SUCCESSFULLY ALIGNED AND VERIFIED!")
        print(f"📁 Aligned files directory: {os.path.abspath(output_dir)}")
    else:
        print(f"\n❌ VERIFICATION FAILED - Please check the alignment process")
        
    return aligned_files, alignment_report, verification_passed

def quick_tif_check(thermal_path, landcover_path, weather_tifs):
    """
    Quick check of TIF file dimensions without full alignment
    """
    print("QUICK TIF FILES DIMENSION CHECK")
    print("="*50)
    
    # Check thermal (reference)
    with rasterio.open(thermal_path) as src:
        thermal_shape = src.read(1).shape
        thermal_crs = src.crs
        thermal_bounds = src.bounds
    
    print(f"Thermal: {thermal_shape} | CRS: {thermal_crs}")
    
    # Check landcover
    with rasterio.open(landcover_path) as src:
        landcover_shape = src.read(1).shape
        landcover_crs = src.crs
        landcover_bounds = src.bounds
    
    match_landcover = (landcover_shape == thermal_shape and 
                      landcover_crs == thermal_crs and
                      np.allclose(landcover_bounds, thermal_bounds, rtol=1e-6))
    status = "✅ MATCH" if match_landcover else "❌ MISMATCH"
    print(f"Landcover: {landcover_shape} | CRS: {landcover_crs} | {status}")
    
    # Check weather files
    weather_status = {}
    for name, path in weather_tifs.items():
        with rasterio.open(path) as src:
            weather_shape = src.read(1).shape
            weather_crs = src.crs
            weather_bounds = src.bounds
        
        match_weather = (weather_shape == thermal_shape and 
                        weather_crs == thermal_crs and
                        np.allclose(weather_bounds, thermal_bounds, rtol=1e-6))
        status = "✅ MATCH" if match_weather else "❌ MISMATCH"
        weather_status[name] = match_weather
        print(f"{name}: {weather_shape} | CRS: {weather_crs} | {status}")
    
    all_match = match_landcover and all(weather_status.values())
    
    if all_match:
        print("\n🎉 ALL FILES HAVE MATCHING DIMENSIONS!")
        return True
    else:
        print("\n⚠️  FILES HAVE MISMATCHED DIMENSIONS - ALIGNMENT NEEDED")
        return False

# Usage example
if __name__ == "__main__":
    thermal_path = 'C:/Users/unknown/Desktop/DRL/data/thermal_raster_final.tif'
    landcover_path = "C:/Users/unknown/Desktop/DRL/data/lulc_area.tif"
    
    weather_tifs = {
        'soil_moisture': 'C:/Users/unknown/Desktop/DRL/data/SoilMoist_S_tavg_Cl_Resam.tif',
        'rainfall': 'C:/Users/unknown/Desktop/DRL/data/chirpsv3_Clip_Resample.tif', 
        'soil_temp': 'C:/Users/unknown/Desktop/DRL/data/soiltemp_Layer_Clip_Resample.tif',
        'wind_speed': 'C:/Users/unknown/Desktop/DRL/data/windspeed_raster_Res_1.tif',
        'humidity': 'C:/Users/unknown/Desktop/DRL/data/Qair_f_inst_Layer_C_Resam.tif',
        'ndmi': 'C:/Users/unknown/Desktop/DRL/data/NDMI_Clip_Resample.tif',
        'dem': 'C:/Users/unknown/Desktop/DRL/data/DEM_region_Clip.tif',
    }
    
    # Quick check first
    if not quick_tif_check(thermal_path, landcover_path, weather_tifs):
        # Full alignment if needed
        aligned_files, report, success = validate_and_align_tif_files(
            thermal_path, landcover_path, weather_tifs
        )
        
        if success:
            print("\nUse these aligned files in your training:")
            print(f"Landcover: {aligned_files['landcover']}")
            for weather_name, aligned_path in aligned_files.items():
                if weather_name != 'landcover':
                    print(f"{weather_name}: {aligned_path}")