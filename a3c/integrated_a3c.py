"""
Integrated A3C Agent with CNN Observation and ICM Exploration
Combines all 3 high-priority improvements
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_network import CNNActorCriticNetwork
from models.icm import ICMWrapper, CNNIntrinsicCuriosityModule


class IntegratedA3CAgent:
    """
    A3C Agent with integrated improvements:
    1. CNN-based observation processing (spatial patterns)
    2. ICM for curiosity-driven exploration
    3. Works with balanced reward environment
    """
    
    def __init__(self, agent_id, global_network, optimizer, 
                 num_channels=8, patch_size=11, action_size=6,
                 device='cpu', gamma=0.99, entropy_coeff=0.05, value_coeff=0.5,
                 use_icm=True, icm_beta=0.2, intrinsic_reward_scale=0.01):
        
        self.agent_id = agent_id
        self.device = device
        self.action_size = action_size
        self.use_icm = use_icm
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # CNN-based local network
        self.local_network = CNNActorCriticNetwork(
            num_channels=num_channels,
            patch_size=patch_size,
            action_size=action_size
        ).to(device)
        self.local_network.load_state_dict(global_network.state_dict())
        
        self.global_network = global_network
        self.optimizer = optimizer
        
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        
        # ICM for curiosity-driven exploration
        if use_icm:
            self.icm = CNNIntrinsicCuriosityModule(
                num_channels=num_channels,
                patch_size=patch_size,
                action_size=action_size
            ).to(device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
            self.icm_beta = icm_beta
        else:
            self.icm = None
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.intrinsic_rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        self.max_grad_norm = 1.0
    
    def sync_with_global(self):
        """Synchronize local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def get_action(self, state, deterministic=False, epsilon=0.0):
        """
        Select action from CNN observation
        
        Args:
            state: [C, H, W] observation tensor
            deterministic: If True, select argmax action
            epsilon: Epsilon for random exploration
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            policy, value = self.local_network(state_tensor)
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon and not deterministic:
            action = np.random.randint(0, self.action_size)
            log_prob = torch.log(policy[action] + 1e-8).item()
        elif deterministic:
            action = torch.argmax(policy).item()
            log_prob = torch.log(policy[action] + 1e-8).item()
        else:
            try:
                dist = Categorical(policy)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
            except:
                action = np.random.randint(0, self.action_size)
                log_prob = torch.log(policy[action] + 1e-8).item()
        
        return action, log_prob, value.item()
    
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
    
    def store_experience(self, state, action, reward, value, log_prob, done, next_state=None):
        """Store experience with optional intrinsic reward"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        # Compute and store intrinsic reward
        intrinsic_reward = 0.0
        if next_state is not None and self.use_icm:
            intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)
            self.next_states.append(next_state)
        self.intrinsic_rewards.append(intrinsic_reward)
        
        self.episode_reward += reward
        self.episode_intrinsic_reward += intrinsic_reward
        
        if done:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            if len(self.reward_history) > 0:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = max(np.std(self.reward_history), 1.0)
            
            self.episodes_completed += 1
            self.episode_reward = 0
            self.episode_intrinsic_reward = 0
    
    def compute_returns(self, next_value=0):
        """Compute returns with combined extrinsic + intrinsic rewards"""
        returns = []
        advantages = []
        gae_lambda = 0.95
        gae = 0
        
        # Combine extrinsic and intrinsic rewards
        combined_rewards = [
            r + ir for r, ir in zip(self.rewards, self.intrinsic_rewards)
        ]
        
        # Normalize rewards
        if len(combined_rewards) > 1:
            rewards_array = np.array(combined_rewards)
            normalized_rewards = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)
            normalized_rewards = np.clip(normalized_rewards, -10, 10)
        else:
            normalized_rewards = combined_rewards
        
        for i in reversed(range(len(normalized_rewards))):
            if i == len(normalized_rewards) - 1:
                next_non_terminal = 1.0 - float(self.dones[i])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(self.dones[i])
                next_val = self.values[i + 1]
            
            delta = normalized_rewards[i] + self.gamma * next_val * next_non_terminal - self.values[i]
            gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return returns, advantages
    
    def update_global(self, next_value=0):
        """Update global network with A3C gradients + ICM update"""
        if len(self.states) == 0:
            return {}
        
        returns, advantages = self.compute_returns(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        policies, values = self.local_network(states)
        dist = Categorical(policies)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.max_grad_norm)
        
        # Transfer gradients to global
        for local_param, global_param in zip(self.local_network.parameters(), 
                                              self.global_network.parameters()):
            if local_param.grad is not None:
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad
        
        self.optimizer.step()
        self.sync_with_global()
        
        # ICM update
        icm_stats = {}
        if self.use_icm and len(self.next_states) > 0:
            icm_stats = self._update_icm()
        
        stats = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'avg_intrinsic_reward': np.mean(self.intrinsic_rewards) if self.intrinsic_rewards else 0,
            **icm_stats
        }
        
        self.clear_buffer()
        return stats
    
    def _update_icm(self):
        """Update ICM module"""
        if len(self.next_states) < 2:
            return {}
        
        states = torch.FloatTensor(np.array(self.states[:len(self.next_states)])).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        actions = torch.LongTensor(self.actions[:len(self.next_states)]).to(self.device)
        
        pred_action, pred_features, actual_features = self.icm(states, next_states, actions)
        
        inverse_loss = nn.CrossEntropyLoss()(pred_action, actions)
        forward_loss = 0.5 * ((pred_features - actual_features.detach()) ** 2).sum(dim=-1).mean()
        
        icm_loss = (1 - self.icm_beta) * inverse_loss + self.icm_beta * forward_loss
        
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 1.0)
        self.icm_optimizer.step()
        
        return {
            'icm_loss': icm_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'forward_loss': forward_loss.item()
        }
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.intrinsic_rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.next_states.clear()


class IntegratedMultiAgentA3C:
    """
    Multi-agent A3C trainer with CNN + ICM
    """
    
    def __init__(self, env_factory, num_agents, num_channels=8, patch_size=11, 
                 action_size=6, device='cpu', lr=1e-4, gamma=0.99,
                 entropy_coeff=0.05, value_coeff=0.5, update_interval=20,
                 use_icm=True):
        
        self.device = device
        self.num_agents = num_agents
        self.update_interval = update_interval
        self.use_icm = use_icm
        
        # Global CNN network
        self.global_network = CNNActorCriticNetwork(
            num_channels=num_channels,
            patch_size=patch_size,
            action_size=action_size
        ).to(device)
        self.global_network.share_memory()
        
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        
        # Create agents
        self.agents = [
            IntegratedA3CAgent(
                agent_id=i,
                global_network=self.global_network,
                optimizer=self.optimizer,
                num_channels=num_channels,
                patch_size=patch_size,
                action_size=action_size,
                device=device,
                gamma=gamma,
                entropy_coeff=entropy_coeff,
                value_coeff=value_coeff,
                use_icm=use_icm
            )
            for i in range(num_agents)
        ]
        
        # Create environments (should use CNN environment)
        self.envs = [env_factory() for _ in range(num_agents)]
        
        self.best_model_path = "a3c_models/best_integrated_a3c_model.pt"
        self.best_reward = -float('inf')
    
    def collect_experience(self, steps_per_agent=100, use_parallel=False):
        """Collect experience from all agents"""
        all_stats = []
        
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            
            for step in range(steps_per_agent):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(
                    state, action, reward, value, log_prob, done, next_state
                )
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
                
                # Update periodically
                if (step + 1) % self.update_interval == 0:
                    stats = agent.update_global()
                    if stats:
                        all_stats.append(stats)
            
            # Final update
            if len(agent.states) > 0:
                stats = agent.update_global()
                if stats:
                    all_stats.append(stats)
        
        self.scheduler.step()
        return all_stats
    
    def save_model(self, filepath=None):
        """Save model"""
        if filepath is None:
            filepath = self.best_model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'model_state_dict': self.global_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'use_icm': self.use_icm
        }
        
        # Save ICM if used
        if self.use_icm and self.agents:
            save_data['icm_state_dict'] = self.agents[0].icm.state_dict()
        
        torch.save(save_data, filepath)
        print(f"Integrated A3C model saved to {filepath}")
    
    def load_best_model(self):
        """Load best model"""
        if not os.path.exists(self.best_model_path):
            return False
        
        try:
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.global_network.load_state_dict(checkpoint['model_state_dict'])
            
            for agent in self.agents:
                agent.sync_with_global()
                if self.use_icm and 'icm_state_dict' in checkpoint:
                    agent.icm.load_state_dict(checkpoint['icm_state_dict'])
            
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            print(f"Loaded integrated A3C model. Best reward: {self.best_reward:.2f}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
