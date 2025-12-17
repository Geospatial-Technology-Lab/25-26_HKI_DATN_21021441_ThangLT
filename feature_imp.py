import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class UniversalRLAnalyzer:
    """
    Phân tích feature importance cho TẤT CẢ các thuật toán RL
    Hỗ trợ: DQN, DDPG, A3C, A2C, PPO, SAC, VPG
    """
    
    def __init__(self, algorithm: str, model_path: str, device='cpu'):
        """
        Args:
            algorithm: 'DQN', 'DDPG', 'A3C', 'A2C', 'PPO', 'SAC', 'VPG'
            model_path: Đường dẫn đến best model
            device: 'cpu' hoặc 'cuda'
        """
        self.algorithm = algorithm.upper()
        self.device = device
        self.model_path = model_path
        
        # Cấu hình font tiếng Việt và kích thước lớn hơn
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 16  # Tăng từ 10 lên 16
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.labelsize'] = 18  # Kích thước label trục
        plt.rcParams['xtick.labelsize'] = 16  # Kích thước số trên trục x
        plt.rcParams['ytick.labelsize'] = 16  # Kích thước số trên trục y
        plt.rcParams['legend.fontsize'] = 14  # Kích thước chú thích
        
        print(f"\n{'='*70}")
        print(f"PHÂN TÍCH {self.algorithm}")
        print(f"{'='*70}")
        print(f"Đang tải mô hình từ {model_path}...")
        
        # Load model và detect architecture
        self.network, self.state_size = self._load_model(model_path)
        
        # Feature names
        self.feature_names = [
            'Vị trí X', 'Vị trí Y',
            'Nhiệt độ hiện tại', 'Cờ nhiệt cao', 'Cờ nhiệt trung bình',
            'Nhiệt độ láng giềng TB', 'Số lượng cháy láng giềng',
            'Nhiệt độ KK', 'Tốc độ gió', 'Nhiệt độ đất', 'Độ ẩm đất',
            'Lượng mưa', 'NDMI', 'Độ cao (DEM)',
            'Precision', 'Recall'
        ]
        
        if len(self.feature_names) != self.state_size:
            if len(self.feature_names) < self.state_size:
                self.feature_names += [f'Đặc trưng_{i}' for i in range(len(self.feature_names), self.state_size)]
            else:
                self.feature_names = self.feature_names[:self.state_size]
        
        print(f"Phát hiện state_size: {self.state_size}")
        print(f"Sử dụng {len(self.feature_names)} đặc trưng")
        
        # Create output directory
        self.output_dir = f'feature_importance_{self.algorithm.lower()}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_model(self, model_path: str) -> Tuple[nn.Module, int]:
        """Load model dựa trên algorithm type"""
        saved_data = torch.load(model_path, map_location=self.device)
        
        # Detect state_size
        if isinstance(saved_data, dict):
            if 'model_state_dict' in saved_data:
                state_dict = saved_data['model_state_dict']
                state_size = saved_data.get('state_size', 16)
            elif 'state_dict' in saved_data:
                state_dict = saved_data['state_dict']
                state_size = self._detect_state_size(state_dict)
            else:
                state_dict = saved_data
                state_size = self._detect_state_size(state_dict)
        else:
            state_dict = saved_data
            state_size = self._detect_state_size(state_dict)
        
        # Create network based on algorithm
        if self.algorithm in ['PPO', 'A3C', 'A2C', 'VPG']:
            # Actor-Critic architecture
            network = self._create_actor_critic(state_size)
        elif self.algorithm == 'DQN':
            # Q-Network
            network = self._create_q_network(state_size)
        elif self.algorithm == 'DDPG':
            # Actor network (deterministic policy)
            network = self._create_ddpg_actor(state_size)
        elif self.algorithm == 'SAC':
            # SAC Actor (stochastic)
            network = self._create_sac_actor(state_size)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        network.load_state_dict(state_dict, strict=False)
        network.to(self.device)
        network.eval()
        
        return network, state_size
    
    def _detect_state_size(self, state_dict: dict) -> int:
        """Auto-detect state_size from model architecture"""
        for key in state_dict.keys():
            if 'weight' in key and len(state_dict[key].shape) >= 2:
                # First linear layer's input size
                return state_dict[key].shape[1]
        return 16  # Default
    
    def _create_actor_critic(self, state_size: int) -> nn.Module:
        """Create Actor-Critic network for PPO/A3C/A2C/VPG"""
        class ActorCritic(nn.Module):
            def __init__(self, state_size):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.LayerNorm(state_size),
                    nn.Linear(state_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(256),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(128)
                )
                self.actor = nn.Sequential(
                    nn.Linear(128, 6),
                    nn.Softmax(dim=-1)
                )
                self.critic = nn.Linear(128, 1)
            
            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                shared_out = self.shared(x)
                policy = self.actor(shared_out)
                value = self.critic(shared_out)
                return policy, value
        
        return ActorCritic(state_size)
    
    def _create_q_network(self, state_size: int) -> nn.Module:
        """Create Q-Network for DQN"""
        class QNetwork(nn.Module):
            def __init__(self, state_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 6)  # 6 actions
                )
            
            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                q_values = self.network(x)
                # Return as (policy-like, value-like) for consistency
                policy = torch.softmax(q_values, dim=-1)
                value = q_values.max(dim=-1, keepdim=True)[0]
                return policy, value
        
        return QNetwork(state_size)
    
    def _create_ddpg_actor(self, state_size: int) -> nn.Module:
        """Create Actor for DDPG (deterministic)"""
        class DDPGActor(nn.Module):
            def __init__(self, state_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 6),
                    nn.Tanh()
                )
            
            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                action = self.network(x)
                # Convert to discrete-like policy
                policy = torch.softmax(action * 10, dim=-1)
                value = action.mean(dim=-1, keepdim=True)
                return policy, value
        
        return DDPGActor(state_size)
    
    def _create_sac_actor(self, state_size: int) -> nn.Module:
        """Create Actor for SAC (stochastic)"""
        class SACActor(nn.Module):
            def __init__(self, state_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                self.mean = nn.Linear(128, 6)
                self.log_std = nn.Linear(128, 6)
            
            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                features = self.network(x)
                mean = self.mean(features)
                log_std = self.log_std(features)
                std = log_std.exp()
                # Sample action
                policy = torch.softmax(mean, dim=-1)
                value = mean.mean(dim=-1, keepdim=True)
                return policy, value
        
        return SACActor(state_size)
    
    def generate_sample_states(self, n_samples=10000):
        """Tạo sample states"""
        print(f"\nĐang tạo {n_samples} trạng thái mẫu...")
        
        states = []
        for _ in range(n_samples):
            state = []
            state.extend(np.random.rand(2))  # Position
            temp = np.random.uniform(-1, 1)
            state.extend([temp, float(temp > 0.9), float(temp > 0.7)])  # Temp
            state.extend([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])  # Neighborhood
            state.extend([np.random.uniform(-1, 1) for _ in range(7)])  # Weather
            state.extend([np.random.uniform(0, 1), np.random.uniform(0, 1)])  # Metrics
            states.append(state[:self.state_size])
        
        return np.array(states, dtype=np.float32)
    
    def method_gradient_based(self, states):
        """Gradient-based importance"""
        print("\n[Phương pháp 1] Gradient-based importance...")
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        states_tensor.requires_grad = True
        
        policy, value = self.network(states_tensor)
        
        # Policy gradients
        policy_gradients = []
        for action_idx in range(6):
            if states_tensor.grad is not None:
                states_tensor.grad.zero_()
            policy[:, action_idx].sum().backward(retain_graph=True)
            policy_gradients.append(states_tensor.grad.abs().mean(0).cpu().numpy())
        
        policy_importance = np.mean(policy_gradients, axis=0)
        
        # Value gradients
        if states_tensor.grad is not None:
            states_tensor.grad.zero_()
        value.sum().backward()
        value_importance = states_tensor.grad.abs().mean(0).cpu().numpy()
        
        return {
            'policy': policy_importance,
            'value': value_importance,
            'combined': (policy_importance + value_importance) / 2
        }
    
    def method_permutation(self, states, n_permutations=50):
        """Permutation importance"""
        print("\n[Phương pháp 2] Permutation importance...")
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            policy_base, value_base = self.network(states_tensor)
            policy_base_pred = policy_base.argmax(1).cpu().numpy()
            value_base_pred = value_base.cpu().numpy()
        
        importance_scores = []
        
        for feature_idx in tqdm(range(self.state_size), desc="Đặc trưng"):
            policy_changes = []
            value_changes = []
            
            for _ in range(n_permutations):
                states_permuted = states.copy()
                np.random.shuffle(states_permuted[:, feature_idx])
                
                states_perm_tensor = torch.FloatTensor(states_permuted).to(self.device)
                
                with torch.no_grad():
                    policy_perm, value_perm = self.network(states_perm_tensor)
                    policy_perm_pred = policy_perm.argmax(1).cpu().numpy()
                    value_perm_pred = value_perm.cpu().numpy()
                
                policy_change = np.mean(policy_base_pred != policy_perm_pred)
                value_change = np.mean(np.abs(value_base_pred - value_perm_pred))
                
                policy_changes.append(policy_change)
                value_changes.append(value_change)
            
            importance_scores.append({
                'policy': np.mean(policy_changes),
                'value': np.mean(value_changes)
            })
        
        policy_importance = np.array([s['policy'] for s in importance_scores])
        value_importance = np.array([s['value'] for s in importance_scores])
        
        return {
            'policy': policy_importance,
            'value': value_importance,
            'combined': (policy_importance + value_importance) / 2
        }
    
    def method_ablation(self, states):
        """Ablation study"""
        print("\n[Phương pháp 3] Ablation study...")
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            policy_base, value_base = self.network(states_tensor)
            policy_entropy_base = -(policy_base * torch.log(policy_base + 1e-8)).sum(1).mean()
            value_std_base = value_base.std()
        
        importance_scores = []
        
        for feature_idx in tqdm(range(self.state_size), desc="Đặc trưng"):
            states_ablated = states.copy()
            states_ablated[:, feature_idx] = 0
            
            states_abl_tensor = torch.FloatTensor(states_ablated).to(self.device)
            
            with torch.no_grad():
                policy_abl, value_abl = self.network(states_abl_tensor)
                policy_entropy_abl = -(policy_abl * torch.log(policy_abl + 1e-8)).sum(1).mean()
                value_std_abl = value_abl.std()
            
            policy_change = abs(policy_entropy_base - policy_entropy_abl)
            value_change = abs(value_std_base - value_std_abl)
            
            importance_scores.append({
                'policy': policy_change.item(),
                'value': value_change.item()
            })
        
        policy_importance = np.array([s['policy'] for s in importance_scores])
        value_importance = np.array([s['value'] for s in importance_scores])
        
        return {
            'policy': policy_importance,
            'value': value_importance,
            'combined': (policy_importance + value_importance) / 2
        }
    
    def plot_comparison(self, results: Dict):
        """Vẽ biểu đồ so sánh - Từng plot riêng biệt"""
        print(f"\n[Vẽ biểu đồ] Đang tạo biểu đồ cho {self.algorithm}...")
        
        method_names_vi = {
            'Gradient': 'Phương Pháp Gradient',
            'Permutation': 'Phương Pháp Hoán Vị',
            'Ablation': 'Phương Pháp Loại Bỏ'
        }
        
        # ========== PLOT RIÊNG CHO MỖI PHƯƠNG PHÁP ==========
        for method_name, method_results in results.items():
            fig, ax = plt.subplots(figsize=(18, 16))
            
            importance = method_results['combined']
            sorted_idx = np.argsort(importance)[::-1]
            
            # Tô màu gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
            bars = ax.barh(range(len(importance)), importance[sorted_idx], 
                        color=colors, edgecolor='black', linewidth=2.5, height=0.75)
            
            # Tiêu đề và label
            ax.set_title(f'Độ Quan Trọng Đặc Trưng - {method_names_vi[method_name]}', 
                        fontsize=32, fontweight='bold', pad=25)
            ax.set_yticks(range(len(importance)))
            ax.set_yticklabels([self.feature_names[i] for i in sorted_idx], 
                            fontsize=20, fontweight='bold')
            ax.set_xlabel('Điểm Quan Trọng', fontsize=28, fontweight='bold')
            
            # Định dạng trục X - Hiển thị rõ ràng
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.5f}'))
            ax.tick_params(axis='x', labelsize=18)
            
            ax.grid(axis='x', alpha=0.5, linewidth=2.5, linestyle='--')
            
            # Hiển thị giá trị trên mỗi bar - VỊ TRÍ KHÔNG BỊ CHỒNG
            max_importance = max(importance[sorted_idx])
            for i, (bar, val) in enumerate(zip(bars, importance[sorted_idx])):
                x_pos = val + max_importance * 0.01  # Cách ra 1% từ cuối bar
                ax.text(x_pos, i, f'{val:.5f}', 
                    va='center', ha='left', fontsize=18, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.4, edgecolor='black', linewidth=1.5))
            
            # Tăng margin để hiển thị số
            ax.margins(x=0.2)
            ax.set_xlim(0, max_importance * 1.3)
            
            plt.tight_layout()
            
            # Lưu file
            filename = os.path.join(self.output_dir, 
                                f'{self.algorithm}_{method_name.lower()}_importance.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ Lưu: {filename}")
        
        # ========== PLOT TOP 10 - PLOT RIÊNG ==========
        fig, ax = plt.subplots(figsize=(18, 12))
        
        consensus = np.zeros(self.state_size)
        for method_results in results.values():
            importance = method_results['combined']
            ranks = np.argsort(np.argsort(importance)) + 1
            consensus += ranks
        consensus /= len(results)
        
        sorted_idx = np.argsort(consensus)[::-1][:10]
        
        colors_top10 = plt.cm.RdYlGn_r(np.linspace(0, 1, 10))
        bars = ax.bar(range(10), consensus[sorted_idx], 
                    color=colors_top10, edgecolor='black', linewidth=3, width=0.6, alpha=0.85)
        
        ax.set_title(f'10 Đặc Trưng Quan Trọng Nhất - {self.algorithm}', 
                    fontsize=32, fontweight='bold', pad=25)
        ax.set_xticks(range(10))
        ax.set_xticklabels([self.feature_names[i] for i in sorted_idx], 
                        rotation=35, ha='right', fontsize=20, fontweight='bold')
        ax.set_ylabel('Điểm Đồng Thuận', fontsize=28, fontweight='bold')
        
        # Định dạng trục Y - Hiển thị rõ ràng
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.2f}'))
        ax.tick_params(axis='y', labelsize=18)
        
        ax.set_ylim(0, max(consensus[sorted_idx]) * 1.2)
        ax.grid(axis='y', alpha=0.5, linewidth=2.5, linestyle='--')
        
        # Giá trị trên mỗi bar - Rõ ràng
        for bar, val in zip(bars, consensus[sorted_idx]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(consensus[sorted_idx]) * 0.03,
                f'{val:.2f}', ha='center', va='bottom', 
                fontsize=18, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'{self.algorithm}_top10_consensus.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Lưu: {filename}")
        
        print(f"[OK] Tất cả biểu đồ đã lưu vào '{self.output_dir}/'")
    
    def save_results(self, results: Dict) -> pd.DataFrame:
        """Lưu kết quả"""
        print("\n[Lưu file] Đang xuất kết quả...")
        
        data = {'Đặc trưng': self.feature_names}
        
        for method_name, method_results in results.items():
            data[f'{method_name}_combined'] = method_results['combined']
            data[f'{method_name}_policy'] = method_results['policy']
            data[f'{method_name}_value'] = method_results['value']
        
        df = pd.DataFrame(data)
        
        # Consensus rank
        consensus = np.zeros(self.state_size)
        for method_results in results.values():
            importance = method_results['combined']
            ranks = np.argsort(np.argsort(importance)) + 1
            consensus += ranks
        consensus /= len(results)
        df['Xếp_hạng'] = consensus
        
        df = df.sort_values('Xếp_hạng', ascending=False)
        
        csv_path = os.path.join(self.output_dir, f'{self.algorithm}_feature_importance.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f', encoding='utf-8-sig')
        print(f"[OK] Kết quả lưu tại '{csv_path}'")
        
        return df
    
    def run_analysis(self, n_samples=10000, n_permutations=50):
        """Chạy phân tích đầy đủ"""
        states = self.generate_sample_states(n_samples)
        
        results = {}
        results['Gradient'] = self.method_gradient_based(states)
        results['Permutation'] = self.method_permutation(states, n_permutations)
        results['Ablation'] = self.method_ablation(states)
        
        self.plot_comparison(results)
        df = self.save_results(results)
        
        print(f"\n{'='*70}")
        print(f"TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT - {self.algorithm}")
        print(f"{'='*70}")
        print(df[['Đặc trưng', 'Xếp_hạng']].head(10).to_string(index=False))
        print(f"{'='*70}\n")
        
        return results, df


def compare_all_algorithms(results_dict: Dict[str, pd.DataFrame], output_dir='comparison_plots'):
    """
    So sánh tất cả các thuật toán với nhau
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SO SÁNH TẤT CẢ THUẬT TOÁN")
    print("="*70)
    
    # Chuẩn bị dữ liệu
    all_data = {}
    feature_names = None
    
    for algo, df in results_dict.items():
        if feature_names is None:
            feature_names = df['Đặc trưng'].values
        
        # Lấy combined importance từ các phương pháp
        combined_cols = [col for col in df.columns if 'combined' in col]
        if combined_cols:
            # Average across methods
            avg_importance = df[combined_cols].mean(axis=1).values
            all_data[algo] = avg_importance
    
    # Plot 1: Heatmap so sánh
    fig, ax = plt.subplots(figsize=(18, 12))
    
    importance_matrix = np.array([all_data[algo] for algo in all_data.keys()])
    scaler = MinMaxScaler()
    importance_matrix_norm = scaler.fit_transform(importance_matrix.T).T
    
    sns.heatmap(importance_matrix_norm,
               xticklabels=feature_names,
               yticklabels=list(all_data.keys()),
               cmap='YlOrRd',
               annot=True,
               fmt='.2f',
               annot_kws={'size': 14, 'weight': 'bold'},
               cbar_kws={'label': 'Độ quan trọng chuẩn hóa'},
               ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Độ quan trọng chuẩn hóa', size=18, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_algorithms_heatmap.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Top 5 features per algorithm
    fig, ax = plt.subplots(figsize=(18, 12))
    
    x = np.arange(len(all_data))
    width = 0.15
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for feature_idx in range(5):
        values = []
        for algo in all_data.keys():
            sorted_idx = np.argsort(all_data[algo])[::-1]
            if feature_idx < len(sorted_idx):
                values.append(all_data[algo][sorted_idx[feature_idx]])
            else:
                values.append(0)
        
        feature_name = feature_names[np.argsort(importance_matrix.mean(axis=0))[::-1][feature_idx]]
        ax.bar(x + feature_idx * width, values, width, 
              label=feature_name, color=colors[feature_idx], alpha=0.8)
    
    ax.set_xlabel('Thuật toán', fontsize=20, fontweight='bold')
    ax.set_ylabel('Điểm độ quan trọng', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(list(all_data.keys()), fontsize=17)
    ax.legend(fontsize=15, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_algorithms_top5.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Biểu đồ so sánh đã lưu vào '{output_dir}/'")


# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    
    # CẤU HÌNH - Đường dẫn đến tất cả các model
    MODELS = {
        'PPO': 'C:/Users/unknown/Desktop/res_ppo_3_fullmap/ppo_models/best_ppo_model.pth',
        'A3C': 'C:/Users/unknown/Desktop/a3c_1_fullmap/a3c_models/best_a3c_model.pth',
        'A2C': 'C:/Users/unknown/Desktop/a2c_1_fullmap/a2c_models/best_a2c_model.pth',
        'DQN': 'C:/Users/unknown/Desktop/dqn_1_fullmap/dqn_models/best_dqn_model.pth',
        'DDPG': 'C:/Users/unknown/Desktop/ddpg_1_fullmap/ddpg_models/best_ddpg_model.pth',
        'SAC': 'C:/Users/unknown/Desktop/sac_3_fullmap/sac_models/best_sac_model.pth',
        'VPG': 'C:/Users/unknown/Desktop/vpg_v1_fullmap/vpg_models/best_vpg_model.pth'
    }
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SAMPLES = 10000
    N_PERMUTATIONS = 50
    
    print(f"\nThiết bị: {DEVICE}")
    print(f"Số lượng thuật toán: {len(MODELS)}")
    
    # Phân tích từng thuật toán
    all_results = {}
    
    for algo_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n[CẢNH BÁO] Không tìm thấy model: {model_path}")
            continue
        
        try:
            analyzer = UniversalRLAnalyzer(algo_name, model_path, device=DEVICE)
            results, df = analyzer.run_analysis(n_samples=N_SAMPLES, 
                                               n_permutations=N_PERMUTATIONS)
            all_results[algo_name] = df
            
        except Exception as e:
            print(f"\n[LỖI] Không thể phân tích {algo_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # So sánh tất cả thuật toán
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("BẮT ĐẦU SO SÁNH TỔNG THỂ")
        print("="*70)
        compare_all_algorithms(all_results)
    
    print("\n" + "="*70)
    print("HOÀN THÀNH TẤT CẢ PHÂN TÍCH!")
    print("="*70)
    print(f"Đã phân tích thành công: {len(all_results)}/{len(MODELS)} thuật toán")
    print("Kiểm tra các thư mục 'feature_importance_*/' để xem kết quả chi tiết")
    print("Kiểm tra thư mục 'comparison_plots/' để xem so sánh tổng thể")