"""
Integrated PPO Agent with CNN Observation and ICM Exploration
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


class IntegratedPPOAgent:
    """
    PPO Agent with integrated improvements:
    1. CNN-based observation processing
    2. ICM for curiosity-driven exploration
    3. GAE for advantage estimation
    """
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6,
                 device='cpu', lr=3e-4, gamma=0.99, clip_range=0.2,
                 gae_lambda=0.95, entropy_coeff=0.01, value_coeff=0.5,
                 use_icm=True, intrinsic_reward_scale=0.01):
        
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.use_icm = use_icm
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # CNN-based network
        self.network = CNNActorCriticNetwork(
            num_channels=num_channels,
            patch_size=patch_size,
            action_size=action_size
        ).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)
        
        # ICM
        if use_icm:
            self.icm = CNNIntrinsicCuriosityModule(
                num_channels=num_channels,
                patch_size=patch_size,
                action_size=action_size
            ).to(device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        else:
            self.icm = None
        
        # Buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.intrinsic_rewards = []
        self.dones = []
        self.next_states = []
        
        # Stats
        self.episode_reward = 0
        self.episodes_completed = 0
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_history = []
        
        self.max_grad_norm = 0.5
    
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        state_t = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            policy, value = self.network(state_t)
        
        if deterministic:
            action = torch.argmax(policy).item()
        else:
            dist = Categorical(policy)
            action = dist.sample().item()
        
        log_prob = torch.log(policy[action] + 1e-8).item()
        
        return action, log_prob, value.item()
    
    def compute_intrinsic_reward(self, state, action, next_state):
        if not self.use_icm:
            return 0.0
        
        state_t = torch.FloatTensor(state).to(self.device)
        next_t = torch.FloatTensor(next_state).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        
        return float(self.icm.compute_intrinsic_reward(state_t, next_t, action_t).item()) * self.intrinsic_reward_scale
    
    def store_experience(self, state, action, log_prob, value, reward, done, next_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
        intrinsic = 0.0
        if next_state is not None and self.use_icm:
            intrinsic = self.compute_intrinsic_reward(state, action, next_state)
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
    
    def compute_gae(self, next_value=0):
        advantages = []
        gae = 0
        
        combined = [r + ir for r, ir in zip(self.rewards, self.intrinsic_rewards)]
        
        if len(combined) > 1:
            arr = np.array(combined)
            combined = np.clip((arr - arr.mean()) / (arr.std() + 1e-8), -5, 5).tolist()
        
        for i in reversed(range(len(combined))):
            next_val = next_value if i == len(combined) - 1 else self.values[i + 1]
            next_non_term = 1.0 - self.dones[i]
            
            delta = combined[i] + self.gamma * next_val * next_non_term - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_term * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, update_epochs=3, batch_size=32):
        if len(self.states) < batch_size:
            return {}
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        advantages = torch.FloatTensor(self.compute_gae()).to(self.device)
        returns = advantages + values
        
        advantages = torch.clamp((advantages - advantages.mean()) / (advantages.std() + 1e-6), -5, 5)
        
        total_loss = 0
        num_batches = 0
        
        for _ in range(update_epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                idx = indices[start:end]
                
                policy, new_values = self.network(states[idx])
                dist = Categorical(policy)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(new_values.squeeze(), returns[idx])
                
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Update ICM
        icm_stats = {}
        if self.use_icm and len(self.next_states) > 0:
            icm_stats = self._update_icm()
        
        self.scheduler.step()
        self.clear_buffer()
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': self.reward_mean,
            **icm_stats
        }
    
    def _update_icm(self):
        n = len(self.next_states)
        states = torch.FloatTensor(np.array(self.states[:n])).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        actions = torch.LongTensor(self.actions[:n]).to(self.device)
        
        pred_action, pred_feat, actual_feat = self.icm(states, next_states, actions)
        
        inv_loss = nn.CrossEntropyLoss()(pred_action, actions)
        fwd_loss = 0.5 * ((pred_feat - actual_feat.detach()) ** 2).sum(-1).mean()
        
        loss = 0.8 * inv_loss + 0.2 * fwd_loss
        
        self.icm_optimizer.zero_grad()
        loss.backward()
        self.icm_optimizer.step()
        
        return {'icm_loss': loss.item()}
    
    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.intrinsic_rewards.clear()
        self.dones.clear()
        self.next_states.clear()


class IntegratedMultiAgentPPO:
    """Multi-agent PPO with CNN + ICM"""
    
    def __init__(self, env_factory, num_agents, num_channels=8, patch_size=11,
                 action_size=6, device='cpu', use_icm=True, **kwargs):
        
        self.device = device
        
        self.agents = [
            IntegratedPPOAgent(
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
        
        self.best_model_path = "ppo_models/best_integrated_ppo_model.pth"
        self.best_reward = -float('inf')
    
    def collect_experience(self, steps_per_update=1000):
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            
            for _ in range(steps_per_update):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.store_experience(state, action, log_prob, value, reward, done, next_state)
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.best_model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'network': self.agents[0].network.state_dict(),
            'best_reward': self.best_reward
        }
        
        if self.agents[0].use_icm:
            save_data['icm'] = self.agents[0].icm.state_dict()
        
        torch.save(save_data, filepath)
    
    def load_best_model(self):
        if not os.path.exists(self.best_model_path):
            return False
        
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        
        for agent in self.agents:
            agent.network.load_state_dict(checkpoint['network'])
            if agent.use_icm and 'icm' in checkpoint:
                agent.icm.load_state_dict(checkpoint['icm'])
        
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        return True
