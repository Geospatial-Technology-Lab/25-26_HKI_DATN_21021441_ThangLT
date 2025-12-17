"""
Integrated SAC Agent with CNN Observation and ICM Exploration
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import sys
from collections import deque
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_network import CNNActorCriticNetwork, CNNDQNNetwork
from models.icm import CNNIntrinsicCuriosityModule


class CNNPolicyNetwork(nn.Module):
    """CNN-based Policy Network for SAC"""
    def __init__(self, num_channels=8, patch_size=11, action_size=6, hidden_size=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        x = self.features(state).view(state.size(0), -1)
        logits = self.fc(x)
        probs = F.softmax(logits / 0.5, dim=-1)
        return probs + 1e-8


class CNNQNetwork(nn.Module):
    """CNN-based Q Network for SAC"""
    def __init__(self, num_channels=8, patch_size=11, action_size=6, hidden_size=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        x = self.features(state).view(state.size(0), -1)
        return self.fc(x)


class IntegratedSACAgent:
    """SAC Agent with CNN + ICM"""
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6,
                 device='cpu', lr=5e-5, gamma=0.995, tau=0.005, alpha=0.05,
                 use_icm=True, intrinsic_reward_scale=0.01):
        
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.use_icm = use_icm
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # Networks
        self.policy = CNNPolicyNetwork(num_channels, patch_size, action_size).to(device)
        self.q1 = CNNQNetwork(num_channels, patch_size, action_size).to(device)
        self.q2 = CNNQNetwork(num_channels, patch_size, action_size).to(device)
        self.q1_target = CNNQNetwork(num_channels, patch_size, action_size).to(device)
        self.q2_target = CNNQNetwork(num_channels, patch_size, action_size).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        
        # ICM
        if use_icm:
            self.icm = CNNIntrinsicCuriosityModule(num_channels, patch_size, action_size).to(device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        
        # Replay buffer
        self.buffer = deque(maxlen=100000)
        
        # Stats
        self.episode_reward = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.reward_mean = 0
        self.reward_std = 1
    
    def get_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy(state_t).squeeze(0)
        
        if deterministic:
            action = probs.argmax().item()
        else:
            action = Categorical(probs).sample().item()
        
        return action, 0.0, 0.0
    
    def store_experience(self, state, action, reward, next_state, done):
        intrinsic = 0.0
        if self.use_icm:
            s = torch.FloatTensor(state).to(self.device)
            ns = torch.FloatTensor(next_state).to(self.device)
            a = torch.LongTensor([action]).to(self.device)
            intrinsic = float(self.icm.compute_intrinsic_reward(s, ns, a).item()) * self.intrinsic_reward_scale
        
        self.buffer.append((state, action, reward + intrinsic, next_state, done))
        self.episode_reward += reward
        
        if done:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            self.reward_mean = np.mean(self.reward_history) if self.reward_history else 0
            self.reward_std = max(np.std(self.reward_history), 1) if self.reward_history else 1
            self.episodes_completed += 1
            self.episode_reward = 0
    
    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return {}
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-learning
        with torch.no_grad():
            next_probs = self.policy(next_states)
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(-1)
            target_q = rewards + self.gamma * next_v * (1 - dones)
        
        q1_vals = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze()
        q2_vals = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        q_loss = F.mse_loss(q1_vals, target_q) + F.mse_loss(q2_vals, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Policy update
        probs = self.policy(states)
        q1_vals = self.q1(states)
        q2_vals = self.q2(states)
        min_q = torch.min(q1_vals, q2_vals)
        
        policy_loss = (probs * (self.alpha * torch.log(probs + 1e-8) - min_q)).sum(-1).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update
        for t, s in zip(self.q1_target.parameters(), self.q1.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.q2_target.parameters(), self.q2.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        
        return {'q_loss': q_loss.item(), 'policy_loss': policy_loss.item()}
    
    def clear_buffer(self):
        pass


class IntegratedMultiAgentSAC:
    """Multi-agent SAC with CNN + ICM"""
    
    def __init__(self, env_factory, num_agents, num_channels=8, patch_size=11,
                 action_size=6, device='cpu', use_icm=True, **kwargs):
        
        self.device = device
        self.agents = [
            IntegratedSACAgent(num_channels, patch_size, action_size, device, use_icm=use_icm, **kwargs)
            for _ in range(num_agents)
        ]
        self.envs = [env_factory() for _ in range(num_agents)]
        self.best_model_path = "sac_models/best_integrated_sac_model.pth"
        self.best_reward = -float('inf')
    
    def collect_experience(self, steps_per_update=1000):
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            for _ in range(steps_per_update):
                action, _, _ = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                state = env.reset() if done else next_state
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.best_model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy': self.agents[0].policy.state_dict(),
            'q1': self.agents[0].q1.state_dict(),
            'q2': self.agents[0].q2.state_dict()
        }, filepath)
    
    def load_best_model(self):
        if not os.path.exists(self.best_model_path):
            return False
        ckpt = torch.load(self.best_model_path, map_location=self.device)
        for agent in self.agents:
            agent.policy.load_state_dict(ckpt['policy'])
            agent.q1.load_state_dict(ckpt['q1'])
            agent.q2.load_state_dict(ckpt['q2'])
        return True
