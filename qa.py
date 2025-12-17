import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Thiết lập font tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Định nghĩa features
features = ['Độ ẩm đất', 'Lượng mưa', 'Nhiệt độ đất', 'Tốc độ gió', 
            'Nhiệt độ KK', 'Độ cao', 'NDMI', 'Nhiệt độ BM', 'Lớp phủ']

# Feature importance cho dự báo cháy rừng
model_importance = {
    'V-Iteration': [0.08, 0.07, 0.12, 0.11, 0.10, 0.13, 0.10, 0.15, 0.14],
    'P-Iteration': [0.09, 0.08, 0.11, 0.10, 0.11, 0.12, 0.11, 0.14, 0.14],
    'Q-Learning': [0.09, 0.08, 0.13, 0.12, 0.10, 0.11, 0.09, 0.15, 0.13],
    'MCTS': [0.11, 0.10, 0.10, 0.09, 0.12, 0.13, 0.12, 0.11, 0.12],
    'VPG': [0.07, 0.06, 0.15, 0.14, 0.09, 0.10, 0.08, 0.18, 0.13],
    'A2C': [0.09, 0.08, 0.12, 0.11, 0.10, 0.11, 0.10, 0.15, 0.14],
    'A3C': [0.06, 0.05, 0.16, 0.15, 0.08, 0.09, 0.07, 0.20, 0.14],
    'PPO': [0.07, 0.06, 0.15, 0.14, 0.09, 0.10, 0.08, 0.18, 0.13],
    'SAC': [0.06, 0.05, 0.16, 0.15, 0.08, 0.09, 0.07, 0.20, 0.14],
    'DQN': [0.08, 0.07, 0.14, 0.13, 0.09, 0.10, 0.09, 0.17, 0.13],
    'DDPG': [0.06, 0.05, 0.16, 0.15, 0.08, 0.09, 0.07, 0.19, 0.15]
}

# Chỉ số hiệu suất
performance_data = {
    'V-Iteration': {'AUC': 0.57, 'Precision': 0.36, 'Recall': 0.17, 'F1': 0.23, 'Corr': 0.20},
    'P-Iteration': {'AUC': 0.56, 'Precision': 0.25, 'Recall': 0.12, 'F1': 0.21, 'Corr': 0.21},
    'Q-Learning': {'AUC': 0.74, 'Precision': 0.61, 'Recall': 0.60, 'F1': 0.54, 'Corr': 0.39},
    'MCTS': {'AUC': 0.32, 'Precision': 0.26, 'Recall': 0.11, 'F1': 0.20, 'Corr': 0.10},
    'VPG': {'AUC': 0.96, 'Precision': 0.46, 'Recall': 0.92, 'F1': 0.60, 'Corr': 0.74},
    'A2C': {'AUC': 0.57, 'Precision': 0.21, 'Recall': 0.64, 'F1': 0.15, 'Corr': 0.35},
    'A3C': {'AUC': 0.98, 'Precision': 0.71, 'Recall': 0.98, 'F1': 0.82, 'Corr': 0.91},
    'PPO': {'AUC': 0.87, 'Precision': 0.23, 'Recall': 0.88, 'F1': 0.33, 'Corr': 0.78},
    'SAC': {'AUC': 0.95, 'Precision': 0.45, 'Recall': 0.91, 'F1': 0.58, 'Corr': 0.94},
    'DQN': {'AUC': 0.87, 'Precision': 0.19, 'Recall': 0.89, 'F1': 0.31, 'Corr': 0.55},
    'DDPG': {'AUC': 0.97, 'Precision': 0.52, 'Recall': 0.97, 'F1': 0.67, 'Corr': 0.78}
}

colors = plt.cm.tab20(np.linspace(0, 1, len(model_importance)))

# ============================================================================
# 1. BIỂU ĐỒ CHO TỪNG MÔ HÌNH
# ============================================================================
def plot_individual_model(model_name, save=True):
    """Vẽ feature importance cho một mô hình"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Biểu đồ cột ngang
    importance = model_importance[model_name]
    colors_bar = plt.cm.viridis(np.array(importance) / max(importance))
    bars = ax1.barh(features, importance, color=colors_bar, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Mức độ quan trọng', fontsize=13, fontweight='bold')
    ax1.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=11)
    
    # Thêm giá trị - đặt bên trong thanh bar nếu giá trị lớn
    for i, (bar, val) in enumerate(zip(bars, importance)):
        if val > 0.15:  # Nếu thanh bar dài, đặt số bên trong
            ax1.text(val - 0.01, i, f'{val:.2f}', va='center', ha='right', 
                    fontsize=12, fontweight='bold', color='black')
        else:  # Nếu thanh bar ngắn, đặt số bên ngoài
            ax1.text(val + 0.003, i, f'{val:.2f}', va='center', ha='left',
                    fontsize=12, fontweight='bold', color='black')
    
    # Chỉ số hiệu suất
    metrics = performance_data[model_name]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars2 = ax2.bar(metric_names, metric_values, color='steelblue', 
                    edgecolor='black', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Điểm số', fontsize=13, fontweight='bold')
    ax2.set_title('Chỉ số đánh giá', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=11)
    
    # Thêm giá trị - đặt bên trong thanh bar nếu giá trị lớn
    for bar, val in zip(bars2, metric_values):
        height = bar.get_height()
        if val > 0.75:  # Nếu cột cao, đặt số bên trong
            ax2.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                    f'{val:.2f}', ha='center', va='top', fontsize=12, 
                    fontweight='bold', color='black')
        else:  # Nếu cột thấp, đặt số bên ngoài
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, 
                    fontweight='bold', color='black')
    
    plt.tight_layout()
    if save:
        plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Tạo biểu đồ cho tất cả các mô hình
print("Đang tạo biểu đồ cho từng mô hình...")
for model in model_importance.keys():
    plot_individual_model(model, save=True)
    print(f"✓ Đã lưu: feature_importance_{model}.png")

# ============================================================================
# 2. BIỂU ĐỒ SO SÁNH TẤT CẢ CÁC MÔ HÌNH
# ============================================================================
def plot_all_models_comparison(save=True):
    """Vẽ so sánh feature importance của tất cả mô hình"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, (model, importance) in enumerate(model_importance.items()):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        colors_bar = plt.cm.plasma(np.array(importance) / max(importance))
        bars = ax.barh(features, importance, color=colors_bar, edgecolor='black', linewidth=0.8)
        
        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.set_xlabel('Mức độ quan trọng', fontsize=8)
        ax.grid(axis='x', alpha=0.2, linestyle='--')
        ax.tick_params(axis='both', labelsize=7)
        
        # Thêm AUC score
        auc = performance_data[model]['AUC']
        ax.text(0.95, 0.95, f'AUC: {auc:.2f}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if save:
        plt.savefig('feature_importance_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nĐang tạo biểu đồ so sánh...")
plot_all_models_comparison(save=True)
print("✓ Đã lưu: feature_importance_all_models.png")

# ============================================================================
# 3. BẢNG NHIỆT - FEATURE IMPORTANCE
# ============================================================================
def plot_importance_heatmap(save=True):
    """Tạo bảng nhiệt feature importance"""
    models = list(model_importance.keys())
    importance_matrix = np.array([model_importance[model] for model in models])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(models, fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mức độ quan trọng', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Thêm giá trị
    for i in range(len(models)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{importance_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Đặc trưng', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mô hình', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    if save:
        plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nĐang tạo bảng nhiệt...")
plot_importance_heatmap(save=True)
print("✓ Đã lưu: feature_importance_heatmap.png")

# ============================================================================
# 4. TOP 3 ĐẶC TRƯNG QUAN TRỌNG NHẤT
# ============================================================================
def plot_top_features(save=True):
    """Vẽ top 3 đặc trưng quan trọng nhất cho mỗi mô hình"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    models = list(model_importance.keys())
    x_pos = np.arange(len(models))
    width = 0.25
    
    # Lấy top 3 features
    top_features_dict = {}
    for model, importance in model_importance.items():
        sorted_idx = np.argsort(importance)[::-1][:3]
        top_features_dict[model] = [(features[i], importance[i]) for i in sorted_idx]
    
    # Vẽ biểu đồ
    for i in range(3):
        values = [top_features_dict[model][i][1] for model in models]
        feature_names = [top_features_dict[model][i][0] for model in models]
        
        bars = ax.bar(x_pos + i*width, values, width, 
                      label=f'Đặc trưng thứ {i+1}', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Thêm tên đặc trưng
        for j, (bar, name) in enumerate(zip(bars, feature_names)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   name, ha='center', va='bottom', fontsize=8, rotation=90, fontweight='bold')
    
    ax.set_xlabel('Mô hình', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mức độ quan trọng', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save:
        plt.savefig('feature_importance_top3.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nĐang tạo biểu đồ top 3 đặc trưng...")
plot_top_features(save=True)
print("✓ Đã lưu: feature_importance_top3.png")

# ============================================================================
# 5. TƯƠNG QUAN: FEATURE IMPORTANCE vs HIỆU SUẤT
# ============================================================================
def plot_importance_vs_performance(save=True):
    """Vẽ mối quan hệ giữa feature importance và hiệu suất mô hình"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(model_importance.keys())
    thermal_imp = [model_importance[m][7] for m in models]
    soil_temp_imp = [model_importance[m][2] for m in models]
    aucs = [performance_data[m]['AUC'] for m in models]
    corrs = [performance_data[m]['Corr'] for m in models]
    
    # Biểu đồ 1: Nhiệt độ bề mặt vs AUC
    ax1 = axes[0, 0]
    ax1.scatter(thermal_imp, aucs, s=150, alpha=0.7, c=range(len(models)), 
                cmap='viridis', edgecolors='black', linewidth=1.5)
    for i, model in enumerate(models):
        ax1.annotate(model, (thermal_imp[i], aucs[i]), 
                    fontsize=9, ha='right', va='bottom', fontweight='bold')
    ax1.set_xlabel('Mức độ quan trọng - Nhiệt độ BM', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Chỉ số AUC', fontsize=10, fontweight='bold')
    ax1.set_title('Nhiệt độ bề mặt vs AUC', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Biểu đồ 2: Nhiệt độ đất vs AUC
    ax2 = axes[0, 1]
    ax2.scatter(soil_temp_imp, aucs, s=150, alpha=0.7, c=range(len(models)), 
                cmap='plasma', edgecolors='black', linewidth=1.5)
    for i, model in enumerate(models):
        ax2.annotate(model, (soil_temp_imp[i], aucs[i]), 
                    fontsize=9, ha='right', va='bottom', fontweight='bold')
    ax2.set_xlabel('Mức độ quan trọng - Nhiệt độ đất', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Chỉ số AUC', fontsize=10, fontweight='bold')
    ax2.set_title('Nhiệt độ đất vs AUC', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Biểu đồ 3: Nhiệt độ bề mặt vs Correlation
    ax3 = axes[1, 0]
    ax3.scatter(thermal_imp, corrs, s=150, alpha=0.7, c=range(len(models)), 
                cmap='coolwarm', edgecolors='black', linewidth=1.5)
    for i, model in enumerate(models):
        ax3.annotate(model, (thermal_imp[i], corrs[i]), 
                    fontsize=9, ha='right', va='bottom', fontweight='bold')
    ax3.set_xlabel('Mức độ quan trọng - Nhiệt độ BM', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Hệ số tương quan', fontsize=10, fontweight='bold')
    ax3.set_title('Nhiệt độ bề mặt vs Correlation', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Biểu đồ 4: Độ đa dạng đặc trưng vs Hiệu suất
    ax4 = axes[1, 1]
    feature_diversity = [np.std(model_importance[m]) for m in models]
    ax4.scatter(feature_diversity, aucs, s=150, alpha=0.7, c=range(len(models)), 
                cmap='Spectral', edgecolors='black', linewidth=1.5)
    for i, model in enumerate(models):
        ax4.annotate(model, (feature_diversity[i], aucs[i]), 
                    fontsize=9, ha='right', va='bottom', fontweight='bold')
    ax4.set_xlabel('Độ đa dạng đặc trưng (Std)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Chỉ số AUC', fontsize=10, fontweight='bold')
    ax4.set_title('Độ đa dạng vs AUC', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save:
        plt.savefig('feature_importance_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nĐang tạo biểu đồ tương quan...")
plot_importance_vs_performance(save=True)
print("✓ Đã lưu: feature_importance_vs_performance.png")

# ============================================================================
# 6. THỐNG KÊ TỔNG KẾT
# ============================================================================
print("\n" + "="*70)
print("TỔNG KẾT ĐỘ QUAN TRỌNG CÁC ĐẶC TRƯNG - DỰ BÁO CHÁY RỪNG")
print("="*70)

for model in model_importance.keys():
    importance = model_importance[model]
    top_3_idx = np.argsort(importance)[::-1][:3]
    
    print(f"\n{model}:")
    print(f"  AUC: {performance_data[model]['AUC']:.2f} | Corr: {performance_data[model]['Corr']:.2f}")
    print(f"  Top 3 đặc trưng quan trọng nhất:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"    {i}. {features[idx]}: {importance[idx]:.2f}")

print("\n" + "="*70)
print("Đã tạo tất cả các biểu đồ thành công!")
print("="*70)