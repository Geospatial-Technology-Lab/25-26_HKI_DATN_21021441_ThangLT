"""
Integrated DDPG Agent with CNN Observation and ICM Exploration
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
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.icm import CNNIntrinsicCuriosityModule


class CNNActorNetwork(nn.Module):
    """CNN-based Actor for DDPG (discrete actions)"""
    def __init__(self, num_channels=8, patch_size=11, action_size=6, hidden_size=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        x = self.features(state).view(state.size(0), -1)
        return self.fc(x)


class CNNCriticNetwork(nn.Module):
    """CNN-based Critic for DDPG"""
    def __init__(self, num_channels=8, patch_size=11, action_size=6, hidden_size=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5 + action_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.action_size = action_size
    
    def forward(self, state, action):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        x = self.features(state).view(state.size(0), -1)
        
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_size).float()
        else:
            action_onehot = action
        
        return self.fc(torch.cat([x, action_onehot], dim=1))


class IntegratedDDPGAgent:
    """DDPG Agent with CNN + ICM"""
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6,
                 device='cpu', actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 noise_scale=0.2, use_icm=True, intrinsic_reward_scale=0.01):
        
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.use_icm = use_icm
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # Networks
        self.actor = CNNActorNetwork(num_channels, patch_size, action_size).to(device)
        self.critic = CNNCriticNetwork(num_channels, patch_size, action_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # ICM
        if use_icm:
            self.icm = CNNIntrinsicCuriosityModule(num_channels, patch_size, action_size).to(device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        
        # Buffer
        self.buffer = deque(maxlen=50000)
        
        # Stats
        self.episode_reward = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.reward_mean = 0
        self.reward_std = 1
    
    def get_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t).squeeze(0)
        
        if deterministic:
            action = probs.argmax().item()
        else:
            if np.random.random() < self.noise_scale:
                action = np.random.randint(self.action_size)
            else:
                action = probs.argmax().item()
        
        q_val = self.critic(state_t, torch.LongTensor([action]).to(self.device)).item()
        return action, 0.0, q_val
    
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
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states).argmax(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions).squeeze()
        
        current_q = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        probs = self.actor(states)
        actor_loss = -self.critic(states, probs).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
    
    def clear_buffer(self):
        pass


class IntegratedMultiAgentDDPG:
    """Multi-agent DDPG with CNN + ICM"""
    
    def __init__(self, env_factory, num_agents, num_channels=8, patch_size=11,
                 action_size=6, device='cpu', use_icm=True, **kwargs):
        
        self.device = device
        self.agents = [
            IntegratedDDPGAgent(num_channels, patch_size, action_size, device, use_icm=use_icm, **kwargs)
            for _ in range(num_agents)
        ]
        self.envs = [env_factory() for _ in range(num_agents)]
        self.best_model_path = "ddpg_models/best_integrated_ddpg_model.pth"
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
            'actor': self.agents[0].actor.state_dict(),
            'critic': self.agents[0].critic.state_dict()
        }, filepath)
    
    def load_best_model(self):
        if not os.path.exists(self.best_model_path):
            return False
        ckpt = torch.load(self.best_model_path, map_location=self.device)
        for agent in self.agents:
            agent.actor.load_state_dict(ckpt['actor'])
            agent.critic.load_state_dict(ckpt['critic'])
        return True
