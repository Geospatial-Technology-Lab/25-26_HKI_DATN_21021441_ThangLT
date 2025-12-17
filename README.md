# 🔥 Deep Reinforcement Learning for Wildfire Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Đồ án tốt nghiệp - Đại học Khoa học Tự nhiên, ĐHQGHN**

Ứng dụng các thuật toán Deep Reinforcement Learning để phát hiện điểm nóng cháy rừng từ dữ liệu ảnh nhiệt và dữ liệu thời tiết.

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Thuật Toán](#-thuật-toán-được-triển-khai)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Sử Dụng](#-sử-dụng)
- [Dữ Liệu](#-dữ-liệu)
- [Kết Quả](#-kết-quả)

---

## 🎯 Tổng Quan

Dự án này nghiên cứu và so sánh hiệu quả của các thuật toán Deep Reinforcement Learning trong bài toán phát hiện cháy rừng. Agent di chuyển trên bản đồ nhiệt và đưa ra dự đoán vị trí có nguy cơ cháy dựa trên:

- **Dữ liệu nhiệt độ** (Thermal imagery)
- **Dữ liệu thời tiết**: độ ẩm, tốc độ gió, lượng mưa, nhiệt độ đất
- **Dữ liệu địa hình**: DEM, NDMI, Land cover

### Đặc Điểm Nổi Bật

- ✅ So sánh **10+ thuật toán** RL/DRL
- ✅ Environment tối ưu với **batch processing** và **caching**
- ✅ Hỗ trợ **GPU acceleration** (CUDA)
- ✅ **Parallel training** với multi-agent
- ✅ Đánh giá với metrics: AUC, F1, Precision, Recall, PR-AUC

---

## 🧠 Thuật Toán Được Triển Khai

### Deep Reinforcement Learning
| Thuật toán | Mô tả | File |
|------------|-------|------|
| **A3C** | Asynchronous Advantage Actor-Critic | `a3c/` |
| **PPO** | Proximal Policy Optimization | `ppo/` |
| **DQN** | Deep Q-Network (Double DQN) | `dqn/` |
| **SAC** | Soft Actor-Critic (Discrete) | `sac/` |
| **DDPG** | Deep Deterministic Policy Gradient | `ddpg/` |
| **VPG** | Vanilla Policy Gradient | `vpg/` |
| **A2C** | Advantage Actor-Critic | `a2c/` |

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
├── 📂 environment/          # RL Environment
│   ├── env_src.py          # Main environment class
│   └── vec_env.py          # Vectorized environments
│
├── 📂 a3c/                  # A3C algorithm
├── 📂 ppo/                  # PPO algorithm
├── 📂 dqn/                  # DQN algorithm
├── 📂 sac/                  # SAC algorithm
├── 📂 ddpg/                 # DDPG algorithm
├── 📂 vpg/                  # VPG algorithm
├── 📂 a2c/                  # A2C algorithm
├── 📂 q_learning/           # Q-Learning
├── 📂 value_iteration/      # Value Iteration
├── 📂 policy_iteration/     # Policy Iteration
├── 📂 mcts/                 # Monte Carlo Tree Search
│
├── 📂 prepare_data/         # Data preprocessing
├── 📂 utils/                # Utilities & visualization
├── 📂 result/               # Evaluation results (CSV)
│
├── config.py               # Centralized configuration
├── .gitignore
└── README.md
```

---

## 🛠 Cài Đặt

### Yêu Cầu

- Python 3.8+
- CUDA 11.0+ (optional, for GPU)

### Cài Đặt Dependencies

```bash
# Clone repository
git clone https://github.com/Geospatial-Technology-Lab/25-26_HKI_DATN_21021441_ThangLT.git
cd 25-26_HKI_DATN_21021441_ThangLT

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scipy rasterio tqdm gym
```

---

## 🚀 Sử Dụng

### 1. Cấu Hình

Chỉnh sửa file `config.py` để cập nhật đường dẫn dữ liệu:

```python
from config import get_config

config = get_config()
print(config.paths.validate_paths())  # Kiểm tra paths
```

### 2. Training

```bash
# Train A3C
cd a3c
python a3c_main.py

# Train DQN
cd dqn
python dqn_main.py

# Train PPO (xem ppo/test2.py)
cd ppo
python test2.py

# Train SAC
cd sac
python sac_main.py
```

### 3. Evaluation

Khi đã có model đã train (file `.pth`), chạy lại script tương ứng để evaluate:

```bash
python a3c_main.py  # Sẽ tự động load model và evaluate
```

---

## 📊 Dữ Liệu

### Cấu Trúc Dữ Liệu Cần Có

```
data/
├── thermal_raster_final.tif    # Ảnh nhiệt

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

> **Lưu ý**: Các file dữ liệu lớn không được upload lên GitHub. Liên hệ tác giả để lấy dữ liệu.

---

## 📈 Kết Quả

Kết quả đánh giá các thuật toán được lưu trong thư mục `result/`:

| Thuật toán | AUC | F1 | Precision | Recall |
|------------|-----|-----|-----------|--------|
| A3C | - | - | - | - |
| PPO | - | - | - | - |
| DQN | - | - | - | - |
| SAC | - | - | - | - |

*Chi tiết kết quả xem trong các file CSV tương ứng.*

---

## 🔧 Tính Năng Tối Ưu

### Environment Optimizations
- **Batch Observations**: Xử lý nhiều observations cùng lúc
- **LRU Cache**: Cache 5000 observations với eviction tự động
- **Precomputed Neighborhoods**: Tính trước neighborhood statistics với scipy

### Training Optimizations
- **Parallel Experience Collection**: Multi-threaded experience gathering
- **Vectorized Environments**: Chạy nhiều environments song song
- **GPU Acceleration**: Batch size 1024 cho GPU inference

---

## 👤 Tác Giả

**Lê Toàn Thắng**
- Mã sinh viên: 21021441
- Trường: Đại học Khoa học Tự nhiên, ĐHQGHN
- Email: [contact email]

---

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Acknowledgments

- Geospatial Technology Lab - VNU
- PyTorch Team
- OpenAI Gym