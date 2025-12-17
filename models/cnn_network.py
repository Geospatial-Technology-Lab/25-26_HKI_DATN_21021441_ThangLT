"""
CNN-based Actor-Critic Network for Wildfire Detection
Designed to work with CNN environment observations
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class CNNActorCriticNetwork(nn.Module):
    """
    CNN-based Actor-Critic network for spatial observation processing.
    
    Input: [batch, 8, 11, 11] - 8 channels, 11x11 patch
    Output: 
        - policy: [batch, action_size] - action probabilities
        - value: [batch, 1] - state value
    """
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6, hidden_size=256):
        super().__init__()
        
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.action_size = action_size
        
        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Conv1: 8 -> 32 channels
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Conv2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11x11 -> 5x5
            nn.Dropout2d(0.1),
            
            # Conv3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        
        # Calculate flattened size after conv layers
        # After MaxPool2d(2): 11x11 -> 5x5
        self.flattened_size = 64 * 5 * 5
        
        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: [batch, 8, 11, 11] or [8, 11, 11] tensor
            
        Returns:
            policy: action probabilities
            value: state value
        """
        # Handle single sample
        single_sample = False
        if state.dim() == 3:
            state = state.unsqueeze(0)
            single_sample = True
        
        # Extract features
        features = self.features(state)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Shared FC
        shared = self.shared_fc(features)
        
        # Actor head with temperature scaling
        logits = self.actor(shared)
        temperature = 0.5  # Lower = more decisive
        policy = torch.softmax(logits / temperature, dim=-1)
        
        # Add epsilon to prevent zero probabilities
        policy = policy + 1e-8
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        # Critic head
        value = self.critic(shared)
        
        if single_sample:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
        
        return policy, value
    
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            policy, value = self.forward(state)
            
            if deterministic:
                action = torch.argmax(policy, dim=-1)
            else:
                dist = Categorical(policy)
                action = dist.sample()
            
            log_prob = torch.log(policy[action] + 1e-8)
            
        return action.item(), log_prob.item(), value.item()


class CNNDQNNetwork(nn.Module):
    """
    CNN-based DQN network for Q-value estimation.
    Uses Dueling architecture for better value estimation.
    """
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6, hidden_size=256):
        super().__init__()
        
        self.num_channels = num_channels
        self.action_size = action_size
        
        # CNN Feature Extractor (same as actor-critic)
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        
        self.flattened_size = 64 * 5 * 5
        
        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass with dueling architecture.
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        single_sample = False
        if state.dim() == 3:
            state = state.unsqueeze(0)
            single_sample = True
        
        features = self.features(state)
        features = features.view(features.size(0), -1)
        shared = self.shared_fc(features)
        
        # Dueling: Q = V + (A - mean(A))
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        if single_sample:
            q_values = q_values.squeeze(0)
        
        return q_values


class CNNPPOAgent:
    """
    PPO Agent using CNN network for spatial observations
    """
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6, 
                 device='cpu', lr=3e-4, gamma=0.99, clip_range=0.2,
                 gae_lambda=0.95, entropy_coeff=0.01, value_coeff=0.5):
        
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        
        # Network
        self.network = CNNActorCriticNetwork(
            num_channels=num_channels,
            patch_size=patch_size,
            action_size=action_size
        ).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Statistics
        self.episode_reward = 0
        self.episodes_completed = 0
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_history = []
        
        self.max_grad_norm = 0.5
    
    def get_action(self, state, deterministic=False):
        """Get action from state"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.network.get_action(state_tensor, deterministic)
    
    def store_experience(self, state, action, log_prob, value, reward, done):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.episode_reward += reward
        
        if done:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            self.reward_mean = np.mean(self.reward_history) if self.reward_history else 0
            self.reward_std = max(np.std(self.reward_history), 1.0) if self.reward_history else 1
            
            self.episodes_completed += 1
            self.episode_reward = 0
    
    def compute_gae(self, next_value=0):
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        rewards = np.array(self.rewards)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = np.clip(rewards, -5, 5)
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = self.values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, update_epochs=3, batch_size=32):
        """Update policy using PPO"""
        if len(self.states) < batch_size:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        advantages = torch.FloatTensor(self.compute_gae()).to(self.device)
        returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        total_loss = 0
        num_batches = 0
        
        for _ in range(update_epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_idx = indices[start:end]
                
                new_policy, new_values = self.network(states[batch_idx])
                new_dist = Categorical(new_policy)
                new_log_probs = new_dist.log_prob(actions[batch_idx])
                entropy = new_dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(new_values.squeeze(), returns[batch_idx])
                
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.scheduler.step()
        self.clear_buffer()
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': self.reward_mean
        }
    
    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
