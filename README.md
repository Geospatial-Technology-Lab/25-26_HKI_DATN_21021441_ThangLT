# 🔥 Deep Reinforcement Learning for Wildfire Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bachelor's Thesis - University of Engineering and Technology, Vietnam National University Hanoi**

A comprehensive framework applying Deep Reinforcement Learning algorithms for wildfire hotspot detection using thermal imagery and weather data.

## 📋 Table of Contents

- [Overview](#-overview)
- [New Features](#-new-features-v20)
- [Algorithms](#-implemented-algorithms)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)

---

## 🎯 Overview

This project researches and compares the effectiveness of Deep Reinforcement Learning algorithms for wildfire detection. The agent navigates through thermal maps and predicts fire risk locations based on:

- **Thermal Data** (Satellite thermal imagery)
- **Weather Data**: humidity, wind speed, rainfall, soil temperature
- **Terrain Data**: DEM, NDMI, Land cover

### Key Features

- ✅ Comparison of **10+ RL/DRL algorithms**
- ✅ **CNN-based Observation** (8 channels × 11×11 spatial features)
- ✅ **ICM Exploration** (Curiosity-driven exploration)
- ✅ **Balanced Reward Structure**
- ✅ **GPU Acceleration** (CUDA support)
- ✅ **Multi-agent Parallel Training**

---

## 🆕 New Features (v2.0)

### 1. CNN-based Observation
- Observation space: `[8, 11, 11]` instead of 1D vector
- 8 channels: thermal, humidity, wind_speed, soil_temp, soil_moisture, rainfall, ndmi, dem
- Enables learning spatial patterns (fire edges, spread direction)

### 2. ICM Exploration (Intrinsic Curiosity Module)
- Curiosity-driven exploration mechanism
- Intrinsic rewards based on prediction error
- Improves exploration in sparse reward environments

### 3. Balanced Reward Structure
- `false_positive_penalty`: 300 → 50 (reduced)
- `false_negative_penalty`: 50 → 100 (increased)
- Added `proximity_reward_scale` and `discovery_bonus`

### 4. Integrated Models
All 7 DRL algorithms have integrated versions with CNN + ICM:

| Algorithm | Integrated File |
|-----------|-----------------|
| A3C | `a3c/integrated_a3c.py` |
| A2C | `a2c/integrated_a2c.py` |
| PPO | `ppo/integrated_ppo.py` |
| DQN | `dqn/integrated_dqn.py` |
| SAC | `sac/integrated_sac.py` |
| DDPG | `ddpg/integrated_ddpg.py` |
| VPG | `vpg/integrated_vpg.py` |

---

## 🧠 Implemented Algorithms

### Deep Reinforcement Learning
| Algorithm | Description | Files |
|-----------|-------------|-------|
| **A3C** | Asynchronous Advantage Actor-Critic | `a3c/a3c.py`, `a3c/integrated_a3c.py` |
| **PPO** | Proximal Policy Optimization | `ppo/ppo.py`, `ppo/integrated_ppo.py` |
| **DQN** | Dueling Double Deep Q-Network | `dqn/dqn.py`, `dqn/integrated_dqn.py` |
| **SAC** | Soft Actor-Critic (Discrete) | `sac/sac.py`, `sac/integrated_sac.py` |
| **DDPG** | Deep Deterministic Policy Gradient | `ddpg/ddpg.py`, `ddpg/integrated_ddpg.py` |
| **VPG** | Vanilla Policy Gradient (REINFORCE) | `vpg/vpg.py`, `vpg/integrated_vpg.py` |
| **A2C** | Advantage Actor-Critic | `a2c/a2c.py`, `a2c/integrated_a2c.py` |

### Classical RL & Planning
| Algorithm | Description | Directory |
|-----------|-------------|-----------|
| **Q-Learning** | Tabular Q-Learning | `q_learning/` |
| **Value Iteration** | Dynamic Programming | `value_iteration/` |
| **Policy Iteration** | Dynamic Programming | `policy_iteration/` |
| **MCTS** | Monte Carlo Tree Search | `mcts/` |

---

## 📁 Project Structure

```
DRL_Thesis/
├── 📂 environment/
│   ├── env_src.py              # Original environment
│   ├── cnn_env.py              # CNN-based environment (NEW)
│   └── vec_env.py              # Vectorized environments
│
├── 📂 models/
│   ├── cnn_network.py          # CNN Actor-Critic networks (NEW)
│   ├── icm.py                  # Intrinsic Curiosity Module (NEW)
│   └── __init__.py
│
├── 📂 a3c/                     # A3C implementation
│   ├── a3c.py                  # Original A3C
│   ├── a3c_main.py             # Training script
│   └── integrated_a3c.py       # CNN + ICM integrated (NEW)
│
├── 📂 [other algorithms]/      # Similar structure for each algorithm
│
├── 📂 prepare_data/            # Data preprocessing utilities
├── 📂 utils/                   # Evaluation & visualization utilities
│
├── train_integrated_main.py    # Unified training script (NEW)
├── evaluate_integrated.py      # Quick evaluation script (NEW)
├── evaluate_integrated_full.py # Full patch evaluation (NEW)
├── config.py                   # Centralized configuration
└── README.md
```

---

## 🛠 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- Git LFS (for large data files)

### Setup

```bash
# Install Git LFS first
git lfs install

# Clone repository
git clone https://github.com/Geospatial-Technology-Lab/25-26_HKI_DATN_21021441_ThangLT.git
cd 25-26_HKI_DATN_21021441_ThangLT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas matplotlib scipy rasterio tqdm gym scikit-learn
```

---

## 🚀 Usage

### 1. Training with Integrated Models (Recommended)

```bash
# Train A3C with CNN + ICM
python train_integrated_main.py --algorithm a3c --episodes 500 --device cuda

# Train PPO
python train_integrated_main.py --algorithm ppo --episodes 500 --device cuda

# Train all algorithms
for algo in a3c a2c ppo dqn sac ddpg vpg; do
    python train_integrated_main.py --algorithm $algo --episodes 500 --device cuda
done
```

**Available Options:**
```bash
python train_integrated_main.py --help
  --algorithm    # a3c, a2c, ppo, dqn, sac, ddpg, vpg
  --episodes     # Number of training episodes (default: 500)
  --agents       # Number of parallel agents (default: 4)
  --device       # cuda or cpu
  --no_icm       # Disable ICM exploration
  --use_synthetic # Use synthetic data for testing
```

### 2. Evaluation

```bash
# Quick evaluation (sample region)
python evaluate_integrated.py --algorithm a3c --device cuda

# Full evaluation (all patches)
python evaluate_integrated_full.py --algorithm a3c --device cuda

# Limit patches for faster testing
python evaluate_integrated_full.py --algorithm a3c --device cuda --max_patches 100

# Compare all algorithms
python evaluate_integrated_full.py --algorithm all --device cuda --max_patches 100
```

### 3. Training with Original Models

```bash
# A3C original
cd a3c && python a3c_main.py

# DQN original  
cd dqn && python dqn_main.py
```

---

## 📊 Data

### Data Structure

```
data/
└── thermal_raster_final.tif    # Thermal imagery (15132 × 6442)

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

> **Note**: Large data files are tracked with Git LFS.

---

## 📈 Results

Evaluation results are saved in `{algorithm}_results/`:

- `training_results.json` - Training history and rewards
- `full_evaluation_results.csv` - Per-patch evaluation metrics
- `full_evaluation_summary.json` - Average metrics summary
- `evaluation_plot.png` - Prediction visualization

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| AUC-ROC | Area Under ROC Curve |
| PR-AUC | Area Under Precision-Recall Curve |
| F1 | Harmonic mean of Precision & Recall |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |

---

## 🔧 Technical Details

### CNN Environment (`environment/cnn_env.py`)
- Observation space: `[8, 11, 11]` spatial tensor
- Integrated balanced reward structure
- Proximity and discovery bonus rewards

### ICM Module (`models/icm.py`)
- `CNNIntrinsicCuriosityModule` for spatial observations
- Forward/Inverse model architecture
- Configurable intrinsic reward scaling

### Optimizations
- Multi-agent parallel experience collection
- GPU-accelerated batch processing (batch size: 1024)
- LRU caching for observations
- Periodic model checkpointing

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