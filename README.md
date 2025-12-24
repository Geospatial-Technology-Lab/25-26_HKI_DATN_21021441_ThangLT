# 🔥 Deep Reinforcement Learning for Wildfire Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bachelor's Thesis - University of Engineering and Technology, Vietnam National University Hanoi**

A comprehensive framework applying Deep Reinforcement Learning algorithms for wildfire hotspot detection using thermal imagery and multi-source geospatial data.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Algorithms](#-implemented-algorithms)
- [Installation](#-installation)
- [Usage: Original Version](#-usage-original-version)
- [Usage: Integrated CNN+ICM Version](#-usage-integrated-cnnicm-version)
- [Data Requirements](#-data-requirements)

---

## 🎯 Overview

This project implements and compares **11 Reinforcement Learning algorithms** for wildfire detection using satellite thermal imagery and weather data.

### Two Versions Available

| Version | Environment | Observation | Files |
|---------|-------------|-------------|-------|
| **Original** | `env_src.py` | 1D vector (16 features) | `*_main.py` |
| **Integrated CNN+ICM** | `cnn_env.py` | 3D tensor [8, 11, 11] | `integrated_*.py` |

---

## 📈 Results

### Evaluation Metrics (Original Version)

| Model | AUC | Precision | Recall | F1 | PR-AUC | MSE | Corr |
|-------|-----|-----------|--------|-----|--------|-----|------|
| **A3C** | **0.98** | **0.71** | **0.98** | **0.82** | **0.96** | 0.07 | 0.91 |
| DDPG | 0.97 | 0.52 | 0.97 | 0.67 | 0.86 | 0.10 | 0.78 |
| VPG | 0.96 | 0.46 | 0.92 | 0.60 | 0.83 | 0.02 | 0.74 |
| SAC | 0.95 | 0.45 | 0.91 | 0.58 | 0.80 | **0.007** | **0.94** |
| PPO | 0.87 | 0.23 | 0.88 | 0.33 | 0.48 | 0.07 | 0.78 |
| DQN | 0.87 | 0.49 | 0.89 | 0.41 | 0.32 | 0.13 | 0.55 |
| Q-Learning | 0.74 | 0.61 | 0.60 | 0.54 | 0.55 | 0.22 | 0.39 |
| A2C | 0.57 | 0.21 | 0.64 | 0.15 | 0.14 | 0.20 | 0.35 |
| VI | 0.57 | 0.36 | 0.17 | 0.23 | 0.41 | 0.22 | 0.20 |
| PI | 0.56 | 0.25 | 0.12 | 0.21 | 0.57 | 0.22 | 0.21 |
| MCTS | 0.32 | 0.26 | 0.11 | 0.20 | 0.35 | 0.11 | 0.10 |

> **Best:** A3C (highest AUC, F1, PR-AUC) | SAC (lowest MSE, highest correlation)

---

## 🧠 Implemented Algorithms

### Deep Reinforcement Learning
| Algorithm | Original File | Integrated File |
|-----------|---------------|-----------------|
| A3C | `a3c/a3c_main.py` | `a3c/integrated_a3c.py` |
| A2C | `a2c/a2c_main.py` | `a2c/integrated_a2c.py` |
| PPO | `ppo/ppo_main.py` | `ppo/integrated_ppo.py` |
| DQN | `dqn/dqn_main.py` | `dqn/integrated_dqn.py` |
| SAC | `sac/sac_main.py` | `sac/integrated_sac.py` |
| DDPG | `ddpg/ddpg_main.py` | `ddpg/integrated_ddpg.py` |
| VPG | `vpg/vpg_main.py` | `vpg/integrated_vpg.py` |

### Classical RL & Planning
| Algorithm | File |
|-----------|------|
| Q-Learning | `q_learning/qlearn_main.py` |
| Value Iteration | `value_iteration/vi_main.py` |
| Policy Iteration | `policy_iteration/piter_main.py` |
| MCTS | `mcts/mcts_main.py` |

---

## 🛠 Installation

```bash
# Clone repository
git clone https://github.com/Geospatial-Technology-Lab/25-26_HKI_DATN_21021441_ThangLT.git
cd 25-26_HKI_DATN_21021441_ThangLT

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Install PyTorch (with CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas matplotlib scipy rasterio tqdm gym scikit-learn
```

---

## 🔷 Usage: Original Version

> Uses `environment/env_src.py` with **1D observation** (16 features vector)

### Training Commands

```bash
# A3C (best performing)
python a3c/a3c_main.py --episodes 100 --device cuda

# DQN
python dqn/dqn_main.py --episodes 100 --device cuda

# SAC
python sac/sac_main.py --episodes 100 --device cuda

# VPG (with optional pure REINFORCE mode)
python vpg/vpg_main.py --episodes 100 --device cuda
python vpg/vpg_main.py --episodes 100 --device cuda --no_baseline  # Pure REINFORCE

# DDPG
python ddpg/ddpg_main.py --episodes 100 --device cuda

# A2C
python a2c/a2c_main.py --episodes 100 --device cuda
```

### CLI Options (All Original Algorithms)

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | both | `train`, `eval`, or `both` |
| `--episodes` | 100 | Number of training episodes |
| `--device` | auto | `cuda`, `cpu`, or `auto` |
| `--workers` | 10 | Number of parallel workers |
| `--steps` | varies | Steps per update (A3C: 2000, DQN/SAC: 1000) |
| `--save_interval` | varies | Save model every N episodes |
| `--force_retrain` | False | Force retraining even if model exists |
| `--no_baseline` | False | (VPG only) Use pure REINFORCE |

### Examples

```bash
# Train A3C with 200 episodes on GPU
python a3c/a3c_main.py --episodes 200 --device cuda --workers 8

# Force retrain DQN
python dqn/dqn_main.py --episodes 100 --force_retrain

# VPG without value baseline (pure REINFORCE)
python vpg/vpg_main.py --episodes 100 --no_baseline

# Run on CPU with fewer workers
python sac/sac_main.py --episodes 50 --device cpu --workers 4
```

### Output (Original)
- **GeoTIFF prediction maps** in `{algorithm}_results/`
- **CSV results** with per-patch metrics
- **Confusion maps** showing TP/FP/TN/FN

---

## 🔶 Usage: Integrated CNN+ICM Version

> Uses `environment/cnn_env.py` with **3D CNN observation** [8 channels, 11×11]

### What's Different?

| Feature | Original | Integrated CNN+ICM |
|---------|----------|-------------------|
| Observation | 1D vector (16) | 3D tensor [8, 11, 11] |
| Network | MLP | CNN (Conv2D layers) |
| Exploration | ε-greedy | ICM (curiosity-driven) |
| Spatial learning | ❌ | ✅ Fire edges, spread patterns |

### Training (Integrated)

```bash
# Train A3C with CNN + ICM
python train_integrated_main.py --algorithm a3c --episodes 500 --device cuda

# Train other algorithms
python train_integrated_main.py --algorithm sac --episodes 500 --device cuda
python train_integrated_main.py --algorithm ppo --episodes 500 --device cuda

# Disable ICM exploration
python train_integrated_main.py --algorithm a3c --episodes 500 --no_icm

# Use synthetic data (for testing)
python train_integrated_main.py --algorithm a3c --episodes 100 --use_synthetic
```

### CLI Options (Integrated)

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithm` | a3c | `a3c`, `a2c`, `ppo`, `dqn`, `sac`, `ddpg`, `vpg` |
| `--episodes` | 500 | Number of training episodes |
| `--device` | auto | `cuda`, `cpu`, or `auto` |
| `--agents` | 4 | Number of parallel agents |
| `--steps` | 100 | Steps per update |
| `--no_icm` | False | Disable ICM exploration |
| `--use_synthetic` | False | Use synthetic data |

### Generate Prediction Raster

```bash
# Generate full GeoTIFF fire probability map
python generate_prediction_raster.py --algorithm a3c --device cuda
```

**Output:**
- `a3c_results/a3c_fire_probability.tif` - Fire probability (0-1)
- `a3c_results/a3c_fire_binary.tif` - Binary fire map
- `a3c_results/raster_metrics.json` - Evaluation metrics

### Evaluation

```bash
# Evaluate on all patches
python evaluate_integrated_full.py --algorithm a3c --device cuda

# Quick evaluation (sample region)
python evaluate_integrated.py --algorithm a3c --device cuda
```

---

## 📊 Data Requirements

### Directory Structure

```
data/
└── thermal_raster_final.tif    # Thermal imagery (Landsat 8/9)

database/
├── aligned_landcover.tif       # Land cover classification
├── aligned_humidity.tif        # Humidity
├── aligned_wind_speed.tif      # Wind speed
├── aligned_rainfall.tif        # Rainfall
├── aligned_soil_temp.tif       # Soil temperature
├── aligned_soil_moisture.tif   # Soil moisture
├── aligned_dem.tif             # Digital Elevation Model
└── aligned_ndmi.tif            # NDMI
```

> **Note**: All rasters must share the same CRS, resolution, and extent.

---

## 📁 Project Structure

```
DRL_Thesis/
├── environment/
│   ├── env_src.py          # Original environment (1D obs)
│   ├── cnn_env.py          # CNN environment (3D obs)
│   └── vec_env.py          # Vectorized environments
│
├── models/
│   ├── cnn_network.py      # CNN Actor-Critic networks
│   └── icm.py              # Intrinsic Curiosity Module
│
├── a3c/
│   ├── a3c.py              # A3C algorithm
│   ├── a3c_main.py         # Original training (with CLI)
│   └── integrated_a3c.py   # CNN+ICM version
│
├── [other algorithms]/     # Similar structure (a2c, dqn, sac, ddpg, vpg, ppo)
│
├── train_integrated_main.py      # Unified CNN+ICM training
├── evaluate_integrated_full.py   # Full evaluation
├── generate_prediction_raster.py # GeoTIFF output
└── config.py                     # Configuration
```

---

## 👤 Author

**Le Toan Thang**  
Student ID: 21021441  
University of Engineering and Technology, VNU Hanoi  
Email: toanthangvietduc@gmail.com

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.