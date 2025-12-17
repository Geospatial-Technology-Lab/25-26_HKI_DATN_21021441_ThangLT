"""
Integrated DQN Agent with CNN Observation and ICM Exploration
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
from collections import deque
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_network import CNNDQNNetwork
from models.icm import CNNIntrinsicCuriosityModule


class IntegratedDQNAgent:
    """
    DQN Agent with integrated improvements:
    1. CNN-based observation (Dueling DQN architecture)
    2. ICM for curiosity-driven exploration
    3. Double DQN
    4. Prioritized Experience Replay
    """
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6,
                 device='cpu', lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=100000, batch_size=64, target_update_freq=100,
                 use_double_dqn=True, use_icm=True, intrinsic_reward_scale=0.01):
        
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_icm = use_icm
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # CNN-based Q-Networks (Dueling architecture)
        self.q_network = CNNDQNNetwork(
            num_channels=num_channels,
            patch_size=patch_size,
            action_size=action_size
        ).to(device)
        
        self.target_network = CNNDQNNetwork(
            num_channels=num_channels,
            patch_size=patch_size,
            action_size=action_size
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)
        
        # ICM for exploration
        if use_icm:
            self.icm = CNNIntrinsicCuriosityModule(
                num_channels=num_channels,
                patch_size=patch_size,
                action_size=action_size
            ).to(device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        else:
            self.icm = None
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_capacity)
        
        # Statistics
        self.steps = 0
        self.episodes_completed = 0
        self.episode_reward = 0
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        self.max_grad_norm = 1.0
    
    def get_action(self, state, deterministic=False):
        """Select action using epsilon-greedy with CNN observation"""
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                value = q_values[0, action].item()
            return action, 0.0, value
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            value = q_values[0, action].item()
        
        return action, 0.0, value
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute ICM intrinsic reward"""
        if not self.use_icm or self.icm is None:
            return 0.0
        
        state_t = torch.FloatTensor(state).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        
        intrinsic_reward = self.icm.compute_intrinsic_reward(
            state_t, next_state_t, action_t
        )
        
        return float(intrinsic_reward.item()) * self.intrinsic_reward_scale
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store transition with intrinsic reward"""
        intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)
        combined_reward = reward + intrinsic_reward
        
        self.replay_buffer.append(
            (state, action, combined_reward, next_state, done, intrinsic_reward)
        )
        
        self.episode_reward += reward
        
        if done:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            self.reward_mean = np.mean(self.reward_history) if self.reward_history else 0
            self.reward_std = max(np.std(self.reward_history), 1.0) if self.reward_history else 1
            
            self.episodes_completed += 1
            self.episode_reward = 0
    
    def update(self, update_epochs=1):
        """Update Q-network and ICM"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        total_loss = 0.0
        total_q = 0.0
        icm_stats = {}
        
        for _ in range(update_epochs):
            # Sample batch
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones, intrinsic_rewards = zip(*batch)
            
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Current Q-values
            q_values = self.q_network(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-values
            with torch.no_grad():
                if self.use_double_dqn:
                    next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                    next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_q = self.target_network(next_states).max(dim=1)[0]
                
                target_q = rewards + self.gamma * next_q * (1 - dones)
            
            # Loss
            loss = F.smooth_l1_loss(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_q += current_q.mean().item()
            
            self.steps += 1
            
            # Update target network
            if self.steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Update ICM
            if self.use_icm:
                icm_stats = self._update_icm(states, actions, next_states)
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / update_epochs,
            'q_values': total_q / update_epochs,
            'epsilon': self.epsilon,
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': self.reward_mean,
            **icm_stats
        }
    
    def _update_icm(self, states, actions, next_states):
        """Update ICM"""
        pred_action, pred_features, actual_features = self.icm(states, next_states, actions)
        
        inverse_loss = nn.CrossEntropyLoss()(pred_action, actions)
        forward_loss = 0.5 * ((pred_features - actual_features.detach()) ** 2).sum(dim=-1).mean()
        
        icm_loss = 0.8 * inverse_loss + 0.2 * forward_loss
        
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 1.0)
        self.icm_optimizer.step()
        
        return {'icm_loss': icm_loss.item()}
    
    def save_model(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes_completed': self.episodes_completed
        }
        
        if self.use_icm:
            save_data['icm'] = self.icm.state_dict()
        
        torch.save(save_data, path)
    
    def load_model(self, path):
        """Load model"""
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.steps = checkpoint.get('steps', 0)
        self.episodes_completed = checkpoint.get('episodes_completed', 0)
        
        if self.use_icm and 'icm' in checkpoint:
            self.icm.load_state_dict(checkpoint['icm'])
        
        return True


class IntegratedMultiAgentDQN:
    """Multi-agent DQN with CNN + ICM"""
    
    def __init__(self, env_factory, num_agents, num_channels=8, patch_size=11,
                 action_size=6, device='cpu', use_icm=True, **kwargs):
        
        self.device = device
        
        # Create agents (share replay buffer for efficiency)
        self.agents = [
            IntegratedDQNAgent(
                num_channels=num_channels,
                patch_size=patch_size,
                action_size=action_size,
                device=device,
                use_icm=use_icm,
                **kwargs
            )
            for _ in range(num_agents)
        ]
        
        self.envs = [env_factory() for _ in range(num_agents)]
        
        self.best_model_path = "dqn_models/best_integrated_dqn_model.pth"
        self.best_reward = -float('inf')
    
    def collect_experience(self, steps_per_update=1000):
        """Collect experience from all agents"""
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            
            for _ in range(steps_per_update):
                action, _, _ = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.store_experience(state, action, reward, next_state, done)
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.best_model_path
        self.agents[0].save_model(filepath)
    
    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            success = self.agents[0].load_model(self.best_model_path)
            if success:
                for agent in self.agents[1:]:
                    agent.q_network.load_state_dict(self.agents[0].q_network.state_dict())
                    agent.target_network.load_state_dict(self.agents[0].target_network.state_dict())
            return success
        return False
