# 🔥 Deep Reinforcement Learning for Wildfire Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Đồ án tốt nghiệp - Đại học Công nghệ, Đại học Quốc gia Hà Nội**

Ứng dụng các thuật toán Deep Reinforcement Learning để phát hiện điểm nóng cháy rừng từ dữ liệu ảnh nhiệt và dữ liệu thời tiết.

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Tính Năng Mới](#-tính-năng-mới-v20)
- [Thuật Toán](#-thuật-toán-được-triển-khai)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Sử Dụng](#-sử-dụng)
- [Kết Quả](#-kết-quả)

---

## 🎯 Tổng Quan

Dự án này nghiên cứu và so sánh hiệu quả của các thuật toán Deep Reinforcement Learning trong bài toán phát hiện cháy rừng. Agent di chuyển trên bản đồ nhiệt và đưa ra dự đoán vị trí có nguy cơ cháy dựa trên:

- **Dữ liệu nhiệt độ** (Thermal imagery)
- **Dữ liệu thời tiết**: độ ẩm, tốc độ gió, lượng mưa, nhiệt độ đất
- **Dữ liệu địa hình**: DEM, NDMI, Land cover

### Đặc Điểm Nổi Bật

- ✅ So sánh **10+ thuật toán** RL/DRL
- ✅ **CNN-based Observation** (8 channels × 11×11)
- ✅ **ICM Exploration** (Curiosity-driven)
- ✅ **Balanced Reward Structure**
- ✅ Hỗ trợ **GPU acceleration** (CUDA)
- ✅ **Parallel training** với multi-agent

---

## 🆕 Tính Năng Mới (v2.0)

### 1. CNN-based Observation
- Observation space: `[8, 11, 11]` thay vì 1D vector
- 8 channels: thermal, humidity, wind_speed, soil_temp, soil_moisture, rainfall, ndmi, dem
- Học được spatial patterns (fire edges, spread direction)

### 2. ICM Exploration (Intrinsic Curiosity Module)
- Curiosity-driven exploration
- Intrinsic rewards dựa trên prediction error
- Giúp agent khám phá tốt hơn trong sparse reward environments

### 3. Balanced Reward Structure
- `false_positive_penalty`: 300 → 50 (giảm)
- `false_negative_penalty`: 50 → 100 (tăng)
- Thêm `proximity_reward_scale` và `discovery_bonus`

### 4. Integrated Models
Tất cả 7 DRL algorithms đều có phiên bản tích hợp với CNN + ICM:

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

## 🧠 Thuật Toán Được Triển Khai

### Deep Reinforcement Learning
| Thuật toán | Mô tả | Files |
|------------|-------|-------|
| **A3C** | Asynchronous Advantage Actor-Critic | `a3c/a3c.py`, `a3c/integrated_a3c.py` |
| **PPO** | Proximal Policy Optimization | `ppo/ppo.py`, `ppo/integrated_ppo.py` |
| **DQN** | Deep Q-Network (Dueling Double DQN) | `dqn/dqn.py`, `dqn/integrated_dqn.py` |
| **SAC** | Soft Actor-Critic (Discrete) | `sac/sac.py`, `sac/integrated_sac.py` |
| **DDPG** | Deep Deterministic Policy Gradient | `ddpg/ddpg.py`, `ddpg/integrated_ddpg.py` |
| **VPG** | Vanilla Policy Gradient | `vpg/vpg.py`, `vpg/integrated_vpg.py` |
| **A2C** | Advantage Actor-Critic | `a2c/a2c.py`, `a2c/integrated_a2c.py` |

### Classical RL & Planning
| Thuật toán | Mô tả | File |
|------------|-------|------|
| **Q-Learning** | Tabular Q-Learning | `q_learning/` |
| **Value Iteration** | Dynamic Programming | `value_iteration/` |
| **Policy Iteration** | Dynamic Programming | `policy_iteration/` |
| **MCTS** | Monte Carlo Tree Search | `mcts/` |

---

## 📁 Cấu Trúc Dự Án

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
├── 📂 a3c/
│   ├── a3c.py                  # Original A3C
│   ├── a3c_main.py             # Training script
│   └── integrated_a3c.py       # CNN + ICM integrated (NEW)
│
├── 📂 [other algorithms]/      # Similar structure
│
├── 📂 examples/
│   └── train_integrated.py     # Example training script
│
├── train_integrated_main.py    # Unified training script (NEW)
├── evaluate_integrated.py      # Quick evaluation (NEW)
├── evaluate_integrated_full.py # Full patch evaluation (NEW)
├── config.py                   # Configuration
└── README.md
```

---

## 🛠 Cài Đặt

### Yêu Cầu

- Python 3.8+
- CUDA 11.0+ (optional, for GPU)
- Git LFS (for large data files)

### Cài Đặt

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

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scipy rasterio tqdm gym scikit-learn
```

---

## 🚀 Sử Dụng

### 1. Training với Integrated Models (Khuyên dùng)

```bash
# Train A3C với CNN + ICM
python train_integrated_main.py --algorithm a3c --episodes 500 --device cuda

# Train PPO
python train_integrated_main.py --algorithm ppo --episodes 500 --device cuda

# Train tất cả algorithms
for algo in a3c a2c ppo dqn sac ddpg vpg; do
    python train_integrated_main.py --algorithm $algo --episodes 500 --device cuda
done

# Options
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

# Full evaluation (all patches - like a3c_main.py)
python evaluate_integrated_full.py --algorithm a3c --device cuda

# Limit patches for faster testing
python evaluate_integrated_full.py --algorithm a3c --device cuda --max_patches 100

# Compare all algorithms
python evaluate_integrated_full.py --algorithm all --device cuda --max_patches 100
```

### 3. Training với Original Models

```bash
# A3C original
cd a3c && python a3c_main.py

# DQN original  
cd dqn && python dqn_main.py
```

---

## 📊 Dữ Liệu

### Cấu Trúc Dữ Liệu

```
data/
└── thermal_raster_final.tif    # Ảnh nhiệt (15132 × 6442)

database/
├── aligned_landcover.tif       # Land cover
├── aligned_humidity.tif        # Độ ẩm
├── aligned_wind_speed.tif      # Tốc độ gió
├── aligned_rainfall.tif        # Lượng mưa
├── aligned_soil_temp.tif       # Nhiệt độ đất
├── aligned_soil_moisture.tif   # Độ ẩm đất
├── aligned_dem.tif             # Digital Elevation Model
└── aligned_ndmi.tif            # NDMI index
```

---

## 📈 Kết Quả

Kết quả đánh giá được lưu trong `{algorithm}_results/`:

- `training_results.json` - Training history
- `full_evaluation_results.csv` - Per-patch metrics
- `full_evaluation_summary.json` - Summary metrics
- `evaluation_plot.png` - Visualization

### Metrics
| Metric | Mô tả |
|--------|-------|
| AUC-ROC | Area Under ROC Curve |
| PR-AUC | Area Under Precision-Recall Curve |
| F1 | Harmonic mean of Precision & Recall |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |

---

## 🔧 Tính Năng Tối Ưu

### CNN Environment (`cnn_env.py`)
- Observation: `[8, 11, 11]` spatial features
- Integrated balanced rewards
- Proximity and discovery bonuses

### ICM Module (`models/icm.py`)
- `CNNIntrinsicCuriosityModule` for CNN observations
- Forward/Inverse model for curiosity
- Configurable intrinsic reward scale

### Training Optimizations
- Multi-agent parallel training
- GPU acceleration (batch size 1024)
- Running reward averaging
- Periodic model checkpointing

---

## 👤 Tác Giả

**Lê Toàn Thắng**
- Mã sinh viên: 21021441
- Trường: Đại học Công nghệ, Đại học Quốc gia Hà Nội
- Email: toanthangvietduc@gmail.com

---

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Acknowledgments

- Geospatial Technology Lab - VNU
- PyTorch Team
- OpenAI Gym