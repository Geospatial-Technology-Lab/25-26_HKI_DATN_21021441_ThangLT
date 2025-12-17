import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from collections import deque
import random

class PolicyNetwork(nn.Module):
    """Actor network for SAC-Discrete with improved architecture"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.state_size = state_size
        
        self.network = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, state):
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        
        logits = self.network(state)
        # Temperature scaling for more stable probabilities
        logits = logits / 2.0  # Reduce confidence
        probs = F.softmax(logits, dim=-1)
        
        # Add small epsilon for numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        if single_sample:
            probs = probs.squeeze(0)
            
        return probs

class QNetwork(nn.Module):
    """Q-network for SAC-Discrete with improved architecture"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.state_size = state_size
        
        self.network = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, state):
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
            
        q_values = self.network(state)
        
        if single_sample:
            q_values = q_values.squeeze(0)
            
        return q_values

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for better sample efficiency"""
    def __init__(self, capacity=100000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return state, action, reward, next_state, done, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class SACDiscreteAgent:
    def __init__(self, state_size, action_size, device='cpu', lr=5e-5, gamma=0.995,
                 tau=0.005, alpha=0.05, automatic_entropy_tuning=True, 
                 target_update_interval=2, replay_buffer_size=50000):
        
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.q1 = QNetwork(state_size, action_size).to(device)
        self.q2 = QNetwork(state_size, action_size).to(device)
        self.q1_target = QNetwork(state_size, action_size).to(device)
        self.q2_target = QNetwork(state_size, action_size).to(device)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers with lower learning rate
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.ExponentialLR(
            self.policy_optimizer, gamma=0.995)
        self.q1_scheduler = optim.lr_scheduler.ExponentialLR(
            self.q1_optimizer, gamma=0.995)
        self.q2_scheduler = optim.lr_scheduler.ExponentialLR(
            self.q2_optimizer, gamma=0.995)
        
        # Automatic entropy tuning with fixes
        if self.automatic_entropy_tuning:
            # Reduce target entropy for more deterministic policy
            self.target_entropy = -np.log(1.0 / action_size) * 0.3
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            # Much lower learning rate for alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-6)
            self.max_alpha = 1.5  # Hard limit on alpha
            self.min_alpha = 0.01  # Minimum alpha
        
        # Use prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(replay_buffer_size)
        
        # Training tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.reward_mean = -1.0
        self.reward_std = 1.0
        self.updates = 0
        
        # Beta for importance sampling (will be annealed)
        self.beta = 0.4
        self.beta_increment = 0.001
    
    def process_wildfire_reward(self, reward):
        """Special reward processing for wildfire detection"""
        # Penalize false negatives more than false positives
        if reward < -100:  # Likely missed fire
            return np.clip(reward * 1.5, -10, 10)  # Increase penalty but clip
        elif reward > 0:  # Correct detection
            return np.clip(reward * 2.0, -10, 10)  # Increase reward but clip
        return np.clip(reward, -10, 10)
    
    def get_action(self, state, deterministic=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy(state)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                # Add small exploration noise
                if np.random.random() < 0.1:  # 10% random actions
                    action = torch.tensor(np.random.randint(0, self.action_size))
                else:
                    dist = Categorical(probs)
                    action = dist.sample()
            
            # Calculate value for logging
            q1_values = self.q1(state)
            q2_values = self.q2(state)
            min_q = torch.min(q1_values, q2_values)
            value = (probs * min_q).sum(dim=-1)
            
            log_prob = torch.log(probs.squeeze()[action.item()] + 1e-8)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state, action, log_prob, value, reward, done):
        """Store experience in replay buffer"""
        # Process reward for wildfire detection
        processed_reward = self.process_wildfire_reward(reward)
        
        self.episode_reward += reward  # Track original reward
        self.episode_length += 1
        
        if hasattr(self, 'prev_state'):
            self.memory.push(self.prev_state, self.prev_action, 
                           self.prev_processed_reward, state, done)
        
        self.prev_state = state
        self.prev_action = action
        self.prev_processed_reward = processed_reward
        
        if done:
            self.memory.push(state, action, processed_reward, state, done)
            
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            if len(self.reward_history) > 0:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = max(np.std(self.reward_history), 1.0)
            
            self.episodes_completed += 1
            self.episode_reward = 0
            self.episode_length = 0
            
            if hasattr(self, 'prev_state'):
                delattr(self, 'prev_state')
                delattr(self, 'prev_action')
                delattr(self, 'prev_processed_reward')
    
    def update(self, batch_size=128, updates_per_step=1):
        """Update SAC networks with all fixes"""
        if len(self.memory) < batch_size:
            return {}
        
        total_policy_loss = 0
        total_q1_loss = 0
        total_q2_loss = 0
        total_alpha_loss = 0
        total_entropy = 0
        
        for _ in range(updates_per_step):
            # Sample with priorities
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = \
                self.memory.sample(batch_size, self.beta)
            
            # Anneal beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.LongTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
            
            # Soft normalization with tanh (no brutal clipping)
            reward_mean = reward_batch.mean()
            reward_std = reward_batch.std() + 1e-8
            reward_batch = (reward_batch - reward_mean) / reward_std
            reward_batch = torch.tanh(reward_batch / 2) * 5  # Smooth scaling
            
            with torch.no_grad():
                next_probs = self.policy(next_state_batch)
                next_log_probs = torch.log(next_probs + 1e-8)
                
                next_q1 = self.q1_target(next_state_batch)
                next_q2 = self.q2_target(next_state_batch)
                next_q = torch.min(next_q1, next_q2)
                
                # Clip Q-values to prevent explosion
                next_q = torch.clamp(next_q, -50, 50)
                
                next_value = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
                target_q = reward_batch + (1 - done_batch) * self.gamma * next_value
                target_q = torch.clamp(target_q, -50, 50)
            
            # Update Q networks with importance sampling
            q1_pred = self.q1(state_batch).gather(1, action_batch.unsqueeze(1))
            q2_pred = self.q2(state_batch).gather(1, action_batch.unsqueeze(1))
            
            q1_loss = (weights * F.mse_loss(q1_pred, target_q, reduction='none')).mean()
            q2_loss = (weights * F.mse_loss(q2_pred, target_q, reduction='none')).mean()
            
            # Update priorities
            td_errors = torch.abs(q1_pred - target_q).detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors)
            
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
            self.q1_optimizer.step()
            
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
            self.q2_optimizer.step()
            
            # Update policy
            probs = self.policy(state_batch)
            log_probs = torch.log(probs + 1e-8)
            
            with torch.no_grad():
                q1_curr = self.q1(state_batch)
                q2_curr = self.q2(state_batch)
                min_q_curr = torch.min(q1_curr, q2_curr)
                min_q_curr = torch.clamp(min_q_curr, -50, 50)
            
            entropy = -(probs * log_probs).sum(dim=1).mean()
            policy_loss = (probs * (self.alpha * log_probs - min_q_curr)).sum(dim=1).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()
            
            # Update alpha with hard constraints
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                # FORCE hard clipping on log_alpha
                with torch.no_grad():
                    self.log_alpha.data = torch.clamp(
                        self.log_alpha.data,
                        min=np.log(self.min_alpha),
                        max=np.log(self.max_alpha)
                    )
                
                self.alpha = self.log_alpha.exp().item()
                self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)
                total_alpha_loss += alpha_loss.item()
            
            total_policy_loss += policy_loss.item()
            total_q1_loss += q1_loss.item()
            total_q2_loss += q2_loss.item()
            total_entropy += entropy.item()
            
            # Update target networks
            if self.updates % self.target_update_interval == 0:
                self.soft_update(self.q1_target, self.q1)
                self.soft_update(self.q2_target, self.q2)
            
            # Update learning rate
            if self.updates % 100 == 0:
                self.policy_scheduler.step()
                self.q1_scheduler.step()
                self.q2_scheduler.step()
            
            self.updates += 1
        
        return {
            'policy_loss': total_policy_loss / updates_per_step,
            'q1_loss': total_q1_loss / updates_per_step,
            'q2_loss': total_q2_loss / updates_per_step,
            'alpha_loss': total_alpha_loss / updates_per_step if self.automatic_entropy_tuning else 0,
            'entropy': total_entropy / updates_per_step,
            'alpha': self.alpha,
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': self.reward_mean,
            'reward_std': self.reward_std,
            'learning_rate': self.policy_scheduler.get_last_lr()[0]
        }
    
    def soft_update(self, target, source):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def clear_buffer(self):
        """Compatibility method"""
        pass

class MultiAgentSACDiscrete:
    def __init__(self, env_factory, num_agents, state_size, action_size, device='cpu', **kwargs):
        self.device = device
        self.agents = [SACDiscreteAgent(state_size, action_size, device, **kwargs) 
                      for _ in range(num_agents)]
        self.envs = [env_factory() for _ in range(num_agents)]
        self.best_model_path = "sac_models/best_sac_model.pth"
        self.best_reward = -float('inf')
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.episode_rewards = []
        self.training_stats = []
    
    def collect_experience(self, steps_per_update=1000):
        """Collect experience with reduced update frequency"""
        for agent, env in zip(self.agents, self.envs):
            state = env.reset() if not hasattr(agent, 'current_state') else agent.current_state
            steps_collected = 0
            
            while steps_collected < steps_per_update:
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(state, action, log_prob, value, reward, done)
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
                
                agent.current_state = state
                steps_collected += 1
                
                # Less frequent updates for stability
                if steps_collected % 200 == 0 and len(agent.memory) > 2000:
                    agent.update(batch_size=128, updates_per_step=1)
    
    def load_best_model(self):
        """Load the best saved model"""
        if os.path.exists(self.best_model_path):
            try:
                print(f"Loading SAC model from {self.best_model_path}")
                
                saved_data = torch.load(self.best_model_path, map_location=self.device)
                
                if isinstance(saved_data, dict) and 'policy_state_dict' in saved_data:
                    policy_state = saved_data['policy_state_dict']
                    q1_state = saved_data['q1_state_dict']
                    q2_state = saved_data['q2_state_dict']
                    
                    saved_state_size = saved_data.get('state_size', None)
                    saved_action_size = saved_data.get('action_size', None)
                    
                    if (saved_state_size == self.state_size and 
                        saved_action_size == self.action_size):
                        
                        for agent in self.agents:
                            agent.policy.load_state_dict(policy_state)
                            agent.q1.load_state_dict(q1_state)
                            agent.q2.load_state_dict(q2_state)
                            agent.q1_target.load_state_dict(q1_state)
                            agent.q2_target.load_state_dict(q2_state)
                        
                        self.best_reward = saved_data.get('best_reward', -float('inf'))
                        print("Model loaded successfully")
                        return True
                    else:
                        print("Dimension mismatch")
                        return False
                else:
                    print("Invalid model format")
                    return False
                    
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        else:
            print("No saved model found")
            return False
    
    def save_model(self, filepath=None):
        """Save the current model"""
        if filepath is None:
            filepath = self.best_model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'policy_state_dict': self.agents[0].policy.state_dict(),
            'q1_state_dict': self.agents[0].q1.state_dict(),
            'q2_state_dict': self.agents[0].q2.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'best_reward': self.best_reward
        }
        
        torch.save(save_data, filepath)
        print(f"Model saved to {filepath}")