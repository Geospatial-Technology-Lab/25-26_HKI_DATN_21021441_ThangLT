"""
Centralized Configuration Management for DRL Thesis Project
============================================================
This module provides centralized configuration for paths, training hyperparameters,
and evaluation settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


# Auto-detect project root
def get_project_root() -> str:
    """Get the project root directory"""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(current_file)


PROJECT_ROOT = get_project_root()


@dataclass
class PathConfig:
    """Data paths configuration"""
    base_dir: str = PROJECT_ROOT
    
    # Main data files
    thermal_path: str = "data/thermal_raster_final.tif"
    landcover_path: str = "database/aligned_landcover.tif"
    
    # Weather data paths (relative to base_dir)
    weather_tifs: Dict[str, str] = field(default_factory=lambda: {
        'soil_moisture': 'database/aligned_soil_moisture.tif',
        'rainfall': 'database/aligned_rainfall.tif',
        'soil_temp': 'database/aligned_soil_temp.tif',
        'wind_speed': 'database/aligned_wind_speed.tif',
        'humidity': 'database/aligned_humidity.tif',
        'dem': 'database/aligned_dem.tif',
        'ndmi': 'database/aligned_ndmi.tif'
    })
    
    def get_thermal_path(self) -> str:
        """Get full path to thermal raster"""
        return os.path.join(self.base_dir, self.thermal_path)
    
    def get_landcover_path(self) -> str:
        """Get full path to landcover data"""
        return os.path.join(self.base_dir, self.landcover_path)
    
    def get_weather_paths(self) -> Dict[str, str]:
        """Get full paths for all weather TIFs"""
        return {
            name: os.path.join(self.base_dir, path) 
            for name, path in self.weather_tifs.items()
        }
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all data files exist"""
        results = {}
        
        # Check thermal
        results['thermal'] = os.path.exists(self.get_thermal_path())
        
        # Check landcover
        results['landcover'] = os.path.exists(self.get_landcover_path())
        
        # Check weather files
        for name, path in self.get_weather_paths().items():
            results[f'weather_{name}'] = os.path.exists(path)
        
        return results


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    # Patch settings
    patch_size: int = 100
    overlap: int = 10
    
    # Episode settings
    max_steps: int = 200
    obey_prob: float = 0.9
    
    # Temperature thresholds
    high_temp_threshold: float = 0.95
    medium_temp_threshold: float = 0.85
    
    # Reward parameters
    true_positive_reward: float = 100.0
    false_positive_penalty: float = 300.0
    true_negative_reward: float = 10.0
    false_negative_penalty: float = 50.0
    exploration_reward: float = 0.1
    movement_cost: float = 0.2
    
    # Cache settings
    observation_cache_size: int = 5000
    use_precomputed_neighborhoods: bool = True


@dataclass  
class TrainingConfig:
    """Training hyperparameters"""
    # Multi-agent settings
    num_workers: int = 10
    max_episodes: int = 1000
    
    # Update settings
    save_interval: int = 10
    steps_per_update: int = 2000
    update_interval: int = 20
    
    # Optimization
    batch_size: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Regularization
    entropy_coeff: float = 0.05
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-5
    
    # Exploration
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Device settings
    device: str = "auto"  # auto, cuda, cpu
    use_mixed_precision: bool = True  # Use FP16 on GPU
    
    # Parallel training
    use_parallel_collection: bool = True
    
    def get_device(self) -> str:
        """Get the compute device to use"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    # Batch processing
    batch_size: int = 1024  # Larger batch for GPU evaluation
    max_steps: int = 200
    
    # GPU acceleration
    use_gpu: bool = True
    
    # Output settings
    save_predictions: bool = True
    export_confusion_map: bool = True
    
    # Alignment method
    alignment_method: str = 'match_pixels'


@dataclass
class Config:
    """Master configuration class"""
    paths: PathConfig = field(default_factory=PathConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def load_default(cls) -> 'Config':
        """Load default configuration"""
        return cls()
    
    def validate(self) -> bool:
        """Validate all configuration settings"""
        path_results = self.paths.validate_paths()
        all_valid = all(path_results.values())
        
        if not all_valid:
            print("Missing files:")
            for name, exists in path_results.items():
                if not exists:
                    print(f"  - {name}")
        
        return all_valid


# Global default configuration
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration"""
    return DEFAULT_CONFIG


# Convenience functions for backward compatibility
def get_weather_tifs() -> Dict[str, str]:
    """Get weather TIF paths (for backward compatibility)"""
    return DEFAULT_CONFIG.paths.get_weather_paths()


def get_thermal_path() -> str:
    """Get thermal raster path (for backward compatibility)"""
    return DEFAULT_CONFIG.paths.get_thermal_path()


def get_landcover_path() -> str:
    """Get landcover path (for backward compatibility)"""
    return DEFAULT_CONFIG.paths.get_landcover_path()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("DRL Thesis Configuration")
    print("=" * 50)
    print(f"Project root: {config.paths.base_dir}")
    print(f"Device: {config.training.get_device()}")
    print(f"\nPath validation:")
    for name, valid in config.paths.validate_paths().items():
        status = "OK" if valid else "MISSING"
        print(f"  {name}: {status}")
