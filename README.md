# 🔥 Deep Reinforcement Learning for Wildfire Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bachelor's Thesis - University of Engineering and Technology, Vietnam National University Hanoi**

A comprehensive framework applying Deep Reinforcement Learning algorithms for wildfire hotspot detection using thermal imagery and multi-source geospatial data.

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Algorithms](#-implemented-algorithms)
- [New Features](#-new-features-v20)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data](#-data)

---

## 🎯 Overview

This project researches and compares the effectiveness of Deep Reinforcement Learning algorithms for wildfire detection. The agent navigates through thermal maps and predicts fire risk locations based on:

- **Thermal Data** (Satellite thermal imagery - Landsat 8/9)
- **Weather Data**: humidity, wind speed, rainfall, soil temperature, soil moisture
- **Terrain Data**: DEM, NDMI, Land cover

### Key Features

- ✅ Comparison of **11 RL/DRL algorithms**
- ✅ **CNN-based Observation** (8 channels × 11×11 spatial features)
- ✅ **ICM Exploration** (Curiosity-driven exploration)
- ✅ **Balanced Reward Structure**
- ✅ **GPU Acceleration** (CUDA support)
- ✅ **Full GeoTIFF Raster Output**

---

## 📈 Results

### Evaluation Metrics Comparison

| Model | AUC | Precision | Recall | F1 | PR-AUC | MSE | Correlation |
|-------|-----|-----------|--------|-----|--------|-----|-------------|
| **A3C** | **0.98** | **0.71** | **0.98** | **0.82** | **0.96** | 0.07 | **0.91** |
| **DDPG** | 0.97 | 0.52 | 0.97 | 0.67 | 0.86 | 0.10 | 0.78 |
| **VPG** | 0.96 | 0.46 | 0.92 | 0.60 | 0.83 | 0.02 | 0.74 |
| **SAC** | 0.95 | 0.45 | 0.91 | 0.58 | 0.80 | **0.007** | **0.94** |
| **PPO** | 0.87 | 0.23 | 0.88 | 0.33 | 0.48 | 0.07 | 0.78 |
| **DQN** | 0.87 | 0.49 | 0.89 | 0.41 | 0.32 | 0.13 | 0.55 |
| **Q-Learning** | 0.74 | 0.61 | 0.60 | 0.54 | 0.55 | 0.22 | 0.39 |
| **A2C** | 0.57 | 0.21 | 0.64 | 0.15 | 0.14 | 0.20 | 0.35 |
| **Value Iteration** | 0.57 | 0.36 | 0.17 | 0.23 | 0.41 | 0.22 | 0.20 |
| **Policy Iteration** | 0.56 | 0.25 | 0.12 | 0.21 | 0.57 | 0.22 | 0.21 |
| **MCTS** | 0.32 | 0.26 | 0.11 | 0.20 | 0.35 | 0.11 | 0.10 |

> **Best performer:** A3C achieves the highest AUC (0.98), F1 (0.82), and PR-AUC (0.96).  
> **Best correlation:** SAC achieves highest correlation (0.94) with lowest MSE (0.007).

---

## 🧠 Implemented Algorithms

### Deep Reinforcement Learning (7 algorithms)
| Algorithm | Description | Status |
|-----------|-------------|--------|
| **A3C** | Asynchronous Advantage Actor-Critic | ✅ Best overall |
| **PPO** | Proximal Policy Optimization | ✅ |
| **DQN** | Dueling Double Deep Q-Network | ✅ |
| **SAC** | Soft Actor-Critic (Discrete) | ✅ Best correlation |
| **DDPG** | Deep Deterministic Policy Gradient | ✅ |
| **VPG** | Vanilla Policy Gradient (REINFORCE) | ✅ |
| **A2C** | Advantage Actor-Critic | ✅ |

### Classical RL & Planning (4 algorithms)
| Algorithm | Description | Status |
|-----------|-------------|--------|
| **Q-Learning** | Tabular Q-Learning | ✅ Best classical |
| **Value Iteration** | Dynamic Programming | ✅ |
| **Policy Iteration** | Dynamic Programming | ✅ |
| **MCTS** | Monte Carlo Tree Search | ✅ |

---

## 🆕 New Features (v2.0)

### 1. CNN-based Observation
- Observation space: `[8, 11, 11]` spatial tensor
- 8 channels: thermal, humidity, wind_speed, soil_temp, soil_moisture, rainfall, ndmi, dem
- Enables learning spatial patterns (fire edges, spread direction)

### 2. ICM Exploration (Intrinsic Curiosity Module)
- Curiosity-driven exploration mechanism
- Intrinsic rewards based on prediction error
- Improves exploration in sparse reward environments

### 3. Balanced Reward Structure
- Optimized reward/penalty ratios for stable learning
- Graduated penalties based on temperature
- Proximity and discovery bonuses

### 4. Full GeoTIFF Output
- Generate fire probability raster matching input CRS
- Direct integration with GIS software (QGIS, ArcGIS)

---

## 📁 Project Structure

```
DRL_Thesis/
├── 📂 environment/
│   ├── env_src.py              # Original environment
│   ├── cnn_env.py              # CNN-based environment
│   └── vec_env.py              # Vectorized environments
│
├── 📂 models/
│   ├── cnn_network.py          # CNN Actor-Critic networks
│   ├── icm.py                  # Intrinsic Curiosity Module
│   └── __init__.py
│
├── 📂 a3c/                     # A3C implementation
│   ├── a3c.py                  # Core algorithm
│   ├── a3c_main.py             # Training script
│   └── integrated_a3c.py       # CNN + ICM integrated
│
├── 📂 [other algorithms]/      # Similar structure
│
├── 📂 prepare_data/            # Data preprocessing
├── 📂 utils/                   # Evaluation utilities
│
├── train_integrated_main.py    # Unified training script
├── evaluate_integrated_full.py # Full patch evaluation
├── generate_prediction_raster.py # GeoTIFF output
├── config.py                   # Configuration
└── README.md
```

---

## 🛠 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU)
- Git LFS (for large data files)

### Setup

```bash
# Clone repository
git clone https://github.com/Geospatial-Technology-Lab/25-26_HKI_DATN_21021441_ThangLT.git
cd 25-26_HKI_DATN_21021441_ThangLT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas matplotlib scipy rasterio tqdm gym scikit-learn
```

---

## 🚀 Usage

### 1. Training

```bash
# Train A3C (best performing)
python train_integrated_main.py --algorithm a3c --episodes 500 --device cuda

# Train other algorithms
python train_integrated_main.py --algorithm sac --episodes 500 --device cuda
python train_integrated_main.py --algorithm ppo --episodes 500 --device cuda

# Options
python train_integrated_main.py --help
  --algorithm    # a3c, a2c, ppo, dqn, sac, ddpg, vpg
  --episodes     # Number of training episodes (default: 500)
  --device       # cuda or cpu
  --no_icm       # Disable ICM exploration
```

### 2. Generate Prediction Raster

```bash
# Generate full fire probability GeoTIFF
python generate_prediction_raster.py --algorithm a3c --device cuda

# Output files:
#   - a3c_results/a3c_fire_probability.tif (continuous 0-1)
#   - a3c_results/a3c_fire_binary.tif (binary 0/1)
#   - a3c_results/raster_metrics.json
```

### 3. Evaluation

```bash
# Full evaluation on all patches
python evaluate_integrated_full.py --algorithm a3c --device cuda

# Compare all algorithms
python evaluate_integrated_full.py --algorithm all --device cuda
```

### 4. Original Training Scripts

```bash
# A3C (original implementation)
cd a3c && python a3c_main.py

# Other algorithms
cd dqn && python dqn_main.py
cd sac && python sac_main.py
```

---

## 📊 Data

### Required Data Structure

```
data/
└── thermal_raster_final.tif    # Thermal imagery (Landsat 8/9)

database/
├── aligned_landcover.tif       # Land cover classification
├── aligned_humidity.tif        # Humidity data
├── aligned_wind_speed.tif      # Wind speed data
├── aligned_rainfall.tif        # Rainfall data
├── aligned_soil_temp.tif       # Soil temperature
├── aligned_soil_moisture.tif   # Soil moisture
├── aligned_dem.tif             # Digital Elevation Model
└── aligned_ndmi.tif            # Normalized Difference Moisture Index
```

> **Note**: All rasters must be aligned to the same CRS and resolution.

---

## 👤 Author

**Le Toan Thang**
- Student ID: 21021441
- University: University of Engineering and Technology, VNU Hanoi
- Email: toanthangvietduc@gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Geospatial Technology Lab - VNU
- PyTorch Team
- OpenAI Gym