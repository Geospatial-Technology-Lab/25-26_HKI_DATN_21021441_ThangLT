import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cấu hình style cho plot đẹp hơn
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Danh sách các mô hình
models = ['DQN', 'A2C', 'A3C', 'DDPG', 'VPG', 'Q_Learning', 
          'Value_Iteration', 'Policy_Iteration', 'MCTS', 'PPO', 'SAC']

# Các metrics cần phân tích
metrics = ['F1', 'Precision', 'Recall', 'Pearson_Correlation', 
           'PR_AUC', 'AUC', 'MSE']

# Đọc dữ liệu từ các file CSV
def load_data(folder_path='./'):
    """
    Load dữ liệu từ các file CSV
    folder_path: đường dẫn đến thư mục chứa các file CSV
    """
    data = {}
    
    for model in models:
        # Thử các pattern tên file khác nhau
        possible_names = [
            f"{model}.csv",
            f"{model.lower()}.csv",
            f"{model.replace('_', ' ')}.csv",
            f"{model.lower().replace('_', ' ')}.csv",
            f"{model.replace('_', '-')}.csv",
            f"{model.lower().replace('_', '-')}.csv",
        ]
        
        # Thêm các tên đặc biệt cho MCTS
        if model == 'MCTS':
            possible_names.extend([
                'Monte_Carlo_Tree_Search.csv',
                'monte_carlo_tree_search.csv',
                'monte carlo tree search.csv',
                'MonteCarloTreeSearch.csv',
            ])
        
        for filename in possible_names:
            filepath = Path(folder_path) / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                # Lấy dòng cuối cùng (dòng trung bình)
                avg_row = df.iloc[-1]
                data[model] = avg_row.to_dict()
                print(f"✓ Đã load: {filename} -> {model}")
                break
        else:
            print(f"✗ Không tìm thấy file cho model: {model}")
            print(f"   Đã thử: {', '.join(possible_names[:4])}...")
    
    if not data:
        print("\n⚠️ CẢNH BÁO: Không tìm thấy file CSV nào!")
        print("Vui lòng kiểm tra:")
        print("  - Đường dẫn thư mục đúng chưa?")
        print("  - Tên file có chứa tên model không?")
        print("  - File có định dạng .csv không?")
        return pd.DataFrame()
    
    # Chuyển đổi sang DataFrame và convert sang numeric
    result_df = pd.DataFrame(data).T
    
    # Convert tất cả các cột thành số (numeric), bỏ qua lỗi
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    # Loại bỏ các cột không phải số (nếu có)
    result_df = result_df.select_dtypes(include=[np.number])
    
    print(f"\n✓ Đã chuyển đổi dữ liệu sang dạng số")
    print(f"✓ Các cột số: {list(result_df.columns)}")
    
    return result_df


# 1. BIỂU ĐỒ SO SÁNH TỔNG QUAN CÁC METRICS CHÍNH
def plot_main_metrics_comparison(df, save=True):
    """So sánh các metrics chính giữa các models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))    
    main_metrics = ['F1', 'Precision', 'Recall', 'Pearson_Correlation']
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, main_metrics, colors)):
        data = df[metric].sort_values(ascending=False)
        bars = ax.barh(data.index, data.values, color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Thêm giá trị lên các cột
        for i, (value, bar) in enumerate(zip(data.values, bars)):
            ax.text(value + 0.01, i, f'{value:.4f}', 
                   va='center', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Giá trị', fontweight='bold')
        ax.set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold', fontsize=13)

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, data.max() * 1.15)
    
    plt.tight_layout()
    if save:
        plt.savefig('1_main_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 1_main_metrics_comparison.png")


# 2. BIỂU ĐỒ RADAR CHART (SO SÁNH ĐA CHIỀU)
def plot_radar_chart(df, save=True):
    """Radar chart so sánh đa chiều các models"""
    from math import pi
    
    # Chọn top 5 models theo F1 score
    top_models = df.nlargest(5, 'F1').index.tolist()
    metrics_to_plot = ['F1', 'Precision', 'Recall', 'Pearson_Correlation', 'PR_AUC', 'AUC']
    
    # Normalize data về scale 0-1
    df_norm = df[metrics_to_plot].copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    angles = [n / len(metrics_to_plot) * 2 * pi for n in range(len(metrics_to_plot))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    colors_list = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
    
    for idx, model in enumerate(top_models):
        values = df_norm.loc[model, metrics_to_plot].values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, 
               color=colors_list[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors_list[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_to_plot], 
                       fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=20, frameon=True, 
               shadow=True, fancybox=True, borderpad=1)
    
    if save:
        plt.savefig('2_radar_chart.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 2_radar_chart.png")


# 3. HEATMAP - MA TRẬN TƯƠNG QUAN
def plot_heatmap(df, save=True):
    """Heatmap hiển thị giá trị tất cả metrics"""
    metrics_to_show = ['F1', 'Precision', 'Recall', 
                       'Pearson_Correlation', 'PR_AUC', 'AUC', 'MSE']
    
    df_heatmap = df[metrics_to_show].T
    
    plt.figure(figsize=(14, 8))
    
    # Normalize từng hàng để dễ nhìn
    df_normalized = df_heatmap.div(df_heatmap.max(axis=1), axis=0)
    
    sns.heatmap(df_normalized, annot=df_heatmap.round(4), fmt='', 
                cmap='RdYlGn', cbar_kws={'label': 'Normalized Value'},
                linewidths=1, linecolor='white', 
                annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.ylabel('Metrics', fontweight='bold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save:
        plt.savefig('3_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 3_heatmap.png")


# 4. TOP 3 MODELS CHO MỖI METRIC
def plot_top3_each_metric(df, save=True):
    """Hiển thị top 3 models cho mỗi metric"""
    metrics_to_analyze = ['F1', 'Precision', 'Recall', 'PR_AUC', 'Pearson_Correlation', 'AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 3))
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_analyze)):
        top3 = df.nlargest(3, metric)[metric]
        
        bars = ax.bar(range(len(top3)), top3.values, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Thêm giá trị
        for i, (bar, value) in enumerate(zip(bars, top3.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xticks(range(len(top3)))
        ax.set_xticklabels(top3.index, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('Giá trị', fontweight='bold')
        ax.set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Thêm medals
        medals = ['🥇', '🥈', '🥉']
        for i, medal in enumerate(medals):
            ax.text(i, ax.get_ylim()[1] * 0.95, medal, 
                   ha='center', fontsize=20)
    
    plt.tight_layout()
    if save:
        plt.savefig('4_top3_each_metric.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 4_top3_each_metric.png")


# 5. BIỂU ĐỒ ĐƯỜNG - XU HƯỚNG
def plot_line_comparison(df, save=True):
    """Biểu đồ đường so sánh các metrics"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics_to_plot = ['F1', 'Precision', 'Recall', 'Pearson_Correlation', 'PR_AUC', 'AUC']
    
    # Sort models theo F1 score
    df_sorted = df.sort_values('F1', ascending=False)
    
    for metric in metrics_to_plot:
        ax.plot(df_sorted.index, df_sorted[metric], 
               marker='o', linewidth=2.5, markersize=8, 
               label=metric.upper().replace('_', ' '), alpha=0.8)
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Giá trị', fontweight='bold', fontsize=12)

    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    if save:
        plt.savefig('5_line_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 5_line_comparison.png")


# 6. BIỂU ĐỒ VIOLIN PLOT - PHÂN PHỐI CÁC METRICS
def plot_violinplot(df, save=True):
    """Violin plot cho các metrics chính - Hiển thị phân phối chi tiết hơn"""
    metrics_to_plot = ['F1', 'Precision', 'Recall', 'Pearson_Correlation', 'PR_AUC']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Tạo violin plot với màu sắc đẹp
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    parts = ax.violinplot([df[metric].values for metric in metrics_to_plot],
                          positions=range(len(metrics_to_plot)),
                          showmeans=True, showmedians=True, showextrema=True)
    
    # Tô màu cho violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Style cho các đường
    parts['cmeans'].set_edgecolor('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_edgecolor('blue')
    parts['cmedians'].set_linewidth(2)
    
    # Thêm scatter points để thấy rõ từng model
    for i, metric in enumerate(metrics_to_plot):
        y = df[metric].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=1)
    
    ax.set_xticks(range(len(metrics_to_plot)))
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_to_plot], 
                       rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylabel('Giá trị', fontweight='bold', fontsize=12)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Mean (Trung bình)'),
        Patch(facecolor='blue', alpha=0.7, label='Median (Trung vị)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)
    
    if save:
        plt.savefig('6_violinplot.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 6_violinplot.png")


# 7. BẢNG RANKING TỔNG HỢP
def plot_ranking_table(df, save=True):
    """Tạo bảng xếp hạng tổng hợp"""
    metrics_for_ranking = ['F1', 'Precision', 'Recall', 'PR_AUC', 'AUC']
    
    # Tính ranking cho mỗi metric
    rankings = pd.DataFrame()
    for metric in metrics_for_ranking:
        rankings[metric] = df[metric].rank(ascending=False)
    
    # Tính điểm trung bình
    rankings['Average_Rank'] = rankings.mean(axis=1)
    rankings = rankings.sort_values('Average_Rank')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo data cho bảng
    table_data = []
    for idx, (model, row) in enumerate(rankings.iterrows(), 1):
        rank_str = f"#{idx}"
        avg_rank = f"{row['Average_Rank']:.2f}"
        metric_ranks = [f"{int(row[m])}" for m in metrics_for_ranking]
        table_data.append([rank_str, model, avg_rank] + metric_ranks)
    
    headers = ['Rank', 'Model', 'Avg Rank'] + [m.upper() for m in metrics_for_ranking]
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.08, 0.2, 0.12] + [0.1]*len(metrics_for_ranking))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style cho header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3b82f6')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cho các hàng
    colors_gradient = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(table_data)))
    for i in range(len(table_data)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors_gradient[i] if j < 3 else 'white')
            if j < 3:
                table[(i+1, j)].set_text_props(weight='bold')

    
    if save:
        plt.savefig('7_ranking_table.png', dpi=300, bbox_inches='tight')
        print("✓ Đã lưu: 7_ranking_table.png")


def main(data_folder='./'):
    """
    Hàm chính để chạy toàn bộ phân tích
    
    Parameters:
    -----------
    data_folder : str
        Đường dẫn đến thư mục chứa các file CSV
        Ví dụ: './data/', 'C:/Users/YourName/Documents/csv_files/', etc.
    
    Example:
    --------
    # Chạy với thư mục hiện tại
    main('./')
    
    # Chạy với đường dẫn cụ thể
    main('C:/Users/YourName/Documents/RL_Models/')
    
    # Chạy với đường dẫn tương đối
    main('../data/csv_files/')
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH VÀ TẠO BIỂU ĐỒ MÔ HÌNH REINFORCEMENT LEARNING")
    print("="*60)
    
    # Tạo thư mục output nếu chưa có
    output_folder = Path(data_folder) / 'output_plots'
    output_folder.mkdir(exist_ok=True)
    print(f"\n📁 Thư mục output: {output_folder}")
    
    # Load dữ liệu
    print("\n" + "="*60)
    print("ĐANG TẢI DỮ LIỆU...")
    print("="*60)
    df = load_data(data_folder)
    
    if df.empty:
        print("\n❌ KHÔNG TÌM THẤY DỮ LIỆU! Vui lòng kiểm tra:")
        print(f"   - Đường dẫn: {data_folder}")
        print(f"   - Tên file phải chứa tên model (VD: DQN.csv, ppo.csv)")
        return
    
    print(f"\n✓ Đã load {len(df)} models thành công!")
    print(f"✓ Models: {', '.join(df.index.tolist())}")
    print(f"✓ Metrics: {', '.join(df.columns.tolist())}")
    
    # Chuyển working directory để lưu plots
    import os
    original_dir = os.getcwd()
    os.chdir(output_folder)
    
    # Tạo các plots
    print("\n" + "="*60)
    print("BẮT ĐẦU TẠO CÁC BIỂU ĐỒ...")
    print("="*60 + "\n")
    
    try:
        print("📊 [1/7] Đang tạo biểu đồ so sánh metrics chính...")
        plot_main_metrics_comparison(df)
        
        print("📊 [2/7] Đang tạo radar chart...")
        plot_radar_chart(df)
        
        print("📊 [3/7] Đang tạo heatmap...")
        plot_heatmap(df)
        
        print("📊 [4/7] Đang tạo biểu đồ top 3...")
        plot_top3_each_metric(df)
        
        print("📊 [5/7] Đang tạo biểu đồ đường...")
        plot_line_comparison(df)
        
        print("📊 [6/7] Đang tạo violin plot...")
        plot_violinplot(df)
        
        print("📊 [7/7] Đang tạo bảng xếp hạng...")
        plot_ranking_table(df)
        
        print("\n" + "="*60)
        print("✓ HOÀN THÀNH! ĐÃ TẠO 7 BIỂU ĐỒ CHUYÊN NGHIỆP")
        print("="*60)
        print(f"\n📁 Các file đã được lưu tại: {output_folder}")
        print("\n📄 Danh sách file:")
        print("  1. 1_main_metrics_comparison.png - So sánh 4 metrics chính")
        print("  2. 2_radar_chart.png - Radar chart top 5 models")
        print("  3. 3_heatmap.png - Ma trận nhiệt tất cả metrics")
        print("  4. 4_top3_each_metric.png - Top 3 cho mỗi metric")
        print("  5. 5_line_comparison.png - Xu hướng các metrics")
        print("  6. 6_violinplot.png - Phân phối các metrics (Violin Plot)")
        print("  7. 7_ranking_table.png - Bảng xếp hạng tổng hợp")
        print("\n💡 Tất cả biểu đồ đã được lưu với độ phân giải 300 DPI!")
        print("💡 Phù hợp để chèn trực tiếp vào Word/PowerPoint!")
        
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Quay lại thư mục gốc
        os.chdir(original_dir)


# CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    # ============================================================
    # THAY ĐỔI ĐƯỜNG DẪN Ở ĐÂY
    # ============================================================
    
    # Cách 1: Đường dẫn tương đối (thư mục hiện tại)
    main('C:/Users/unknown/Desktop/DRL/result')
    
    # Cách 2: Đường dẫn tuyệt đối Windows
    # main('C:/Users/YourName/Documents/CSV_Files/')
    
    # Cách 3: Đường dẫn tuyệt đối Linux/Mac
    # main('/home/username/data/csv_files/')
    
    # Cách 4: Đường dẫn tương đối (thư mục cha)
    # main('../data/')
    
    # ============================================================