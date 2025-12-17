"""
Integrated VPG (Vanilla Policy Gradient / REINFORCE) Agent with CNN and ICM
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_network import CNNActorCriticNetwork
from models.icm import CNNIntrinsicCuriosityModule


class IntegratedVPGAgent:
    """VPG (REINFORCE) Agent with CNN + ICM"""
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6,
                 device='cpu', lr=3e-4, gamma=0.99, entropy_coeff=0.01,
                 value_coeff=0.5, use_baseline=True, use_icm=True,
                 intrinsic_reward_scale=0.01):
        
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.use_baseline = use_baseline
        self.use_icm = use_icm
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # CNN Network
        self.network = CNNActorCriticNetwork(num_channels, patch_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # ICM
        if use_icm:
            self.icm = CNNIntrinsicCuriosityModule(num_channels, patch_size, action_size).to(device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        
        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.intrinsic_rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
        
        # Stats
        self.episode_reward = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.reward_mean = 0
        self.reward_std = 1
        self.max_grad_norm = 0.5
    
    def get_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy, value = self.network(state_t)
        
        if deterministic:
            action = policy.argmax().item()
        else:
            action = Categorical(policy).sample().item()
        
        log_prob = torch.log(policy[action] + 1e-8).item()
        return action, log_prob, value.item()
    
    def store_experience(self, state, action, reward, value, log_prob, done, next_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        intrinsic = 0.0
        if next_state is not None and self.use_icm:
            s = torch.FloatTensor(state).to(self.device)
            ns = torch.FloatTensor(next_state).to(self.device)
            a = torch.LongTensor([action]).to(self.device)
            intrinsic = float(self.icm.compute_intrinsic_reward(s, ns, a).item()) * self.intrinsic_reward_scale
            self.next_states.append(next_state)
        self.intrinsic_rewards.append(intrinsic)
        
        self.episode_reward += reward
        if done:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            self.reward_mean = np.mean(self.reward_history) if self.reward_history else 0
            self.reward_std = max(np.std(self.reward_history), 1) if self.reward_history else 1
            self.episodes_completed += 1
            self.episode_reward = 0
    
    def compute_returns(self, next_value=0):
        """Monte Carlo returns (REINFORCE style)"""
        returns = []
        R = next_value
        combined = [r + ir for r, ir in zip(self.rewards, self.intrinsic_rewards)]
        
        for r in reversed(combined):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def update(self):
        if len(self.states) == 0:
            return {}
        
        returns = self.compute_returns()
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policies, values = self.network(states)
        dist = Categorical(policies)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # VPG loss
        if self.use_baseline:
            advantages = returns - values.squeeze().detach()
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
        else:
            actor_loss = -(log_probs * returns).mean()
            loss = actor_loss - self.entropy_coeff * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # ICM update
        if self.use_icm and len(self.next_states) > 0:
            self._update_icm()
        
        self.clear_buffer()
        return {'total_loss': loss.item(), 'episodes_completed': self.episodes_completed}
    
    def _update_icm(self):
        n = len(self.next_states)
        states = torch.FloatTensor(np.array(self.states[:n])).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        actions = torch.LongTensor(self.actions[:n]).to(self.device)
        
        pred_a, pred_f, actual_f = self.icm(states, next_states, actions)
        loss = 0.8 * nn.CrossEntropyLoss()(pred_a, actions) + 0.2 * ((pred_f - actual_f.detach())**2).sum(-1).mean()
        
        self.icm_optimizer.zero_grad()
        loss.backward()
        self.icm_optimizer.step()
    
    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.intrinsic_rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.next_states.clear()


class IntegratedMultiAgentVPG:
    """Multi-agent VPG with CNN + ICM"""
    
    def __init__(self, env_factory, num_agents, num_channels=8, patch_size=11,
                 action_size=6, device='cpu', use_icm=True, **kwargs):
        
        self.device = device
        self.agents = [
            IntegratedVPGAgent(num_channels, patch_size, action_size, device, use_icm=use_icm, **kwargs)
            for _ in range(num_agents)
        ]
        self.envs = [env_factory() for _ in range(num_agents)]
        self.best_model_path = "vpg_models/best_integrated_vpg_model.pth"
        self.best_reward = -float('inf')
    
    def collect_experience(self, steps_per_agent=1000):
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            for _ in range(steps_per_agent):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_experience(state, action, reward, value, log_prob, done, next_state)
                state = env.reset() if done else next_state
            agent.update()
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.best_model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({'network': self.agents[0].network.state_dict()}, filepath)
    
    def load_best_model(self):
        if not os.path.exists(self.best_model_path):
            return False
        ckpt = torch.load(self.best_model_path, map_location=self.device)
        for agent in self.agents:
            agent.network.load_state_dict(ckpt['network'])
        return True
