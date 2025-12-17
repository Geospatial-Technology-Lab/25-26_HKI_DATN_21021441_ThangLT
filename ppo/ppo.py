import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        # Store state_size for compatibility checking
        self.state_size = state_size
        
        self.shared = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size // 2)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state):
        # Handle single sample case for BatchNorm
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        elif state.shape[0] == 1:
            single_sample = True
            # For single sample, set network to eval mode temporarily
            was_training = self.training
            if was_training:
                self.eval()
        
        shared_features = self.shared(state)
        policy = self.actor(shared_features)
        value = self.critic(shared_features)
        
        # Restore training mode if it was changed
        if single_sample and 'was_training' in locals() and was_training:
            self.train()
        
        # Squeeze if input was single sample
        if single_sample:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
            
        return policy, value

class PPOAgent:
    def __init__(self, state_size, action_size, device='cpu', lr=1e-4, gamma=0.99, 
                 clip_range=0.2, gae_lambda=0.95, entropy_coeff=0.01, value_coeff=0.5):
        self.device = device
        self.state_size = state_size  # Store for compatibility checking
        self.action_size = action_size
        self.network = ActorCriticNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)  # Less aggressive
        
        self.gamma = gamma
        self.clip_range = clip_range
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.reward_mean = -1.0
        self.reward_std = 1.0
        self.episodes_completed = 0
        self.min_episodes_before_update = 1
        
        # Running statistics for reward normalization
        self.reward_history = []
    
        # Gradient clipping value
        self.max_grad_norm = 0.5
        # Early stopping KL threshold
        self.target_kl = 0.05

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.network(state)
            
        if deterministic:
            action = torch.argmax(policy, dim=-1)
            # FIX: Use dim=0 for 1D tensor indexing
            log_prob = torch.log(policy.gather(0, action)).squeeze()
        else:
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        if done:
            self.reward_history.append(self.episode_reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            if len(self.reward_history) > 0:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = max(np.std(self.reward_history), 1.0)
            
            self.episodes_completed += 1
            self.episode_reward = 0
            self.episode_length = 0

    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Normalize rewards
        if len(self.rewards) > 1:
            rewards_array = np.array(self.rewards)
            normalized_rewards = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)
            normalized_rewards = np.clip(normalized_rewards, -5, 5)
        else:
            normalized_rewards = self.rewards
            
        for i in reversed(range(len(normalized_rewards))):
            if i == len(normalized_rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = self.values[i + 1]
                
            delta = normalized_rewards[i] + self.gamma * next_value * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self, update_epochs=3, batch_size=32):
        """Update the policy using PPO"""
        if len(self.states) < batch_size:
            return {}
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        # Compute advantages and returns
        advantages = torch.FloatTensor(self.compute_gae()).to(self.device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        advantages = torch.clamp(advantages, -5.0, 5.0)  # Clip extreme advantages  

        dataset_size = len(states)
        
        total_loss = 0
        total_entropy = 0
        total_kl_div = 0
        num_batches = 0
        
        # Multiple epochs of optimization
        for epoch in range(update_epochs):
            # Create random batches
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                new_policy, new_values = self.network(batch_states)
                new_dist = Categorical(new_policy)
                new_log_probs = new_dist.log_prob(batch_actions)
                entropy = new_dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
                
                # KL divergence for monitoring
                kl_div = (batch_old_log_probs - new_log_probs).mean()
                
                # Total loss
                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses for logging
                total_loss += loss.item()
                total_entropy += entropy.item()
                total_kl_div += kl_div.item()
                num_batches += 1
        # Update learning rate
        self.scheduler.step()
        
        # Clear experience buffer
        self.clear_buffer()
        
        # Return training statistics
        if num_batches > 0:
            return {
                'total_loss': total_loss / num_batches,
                'entropy': total_entropy / num_batches,
                'kl_divergence': total_kl_div / num_batches,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'episodes_completed': self.episodes_completed,
                'avg_episode_reward': self.reward_mean,
                'reward_std': self.reward_std
            }
        else:
            return {}
    
    def clear_buffer(self):
        """Clear the experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

class MultiAgentPPO:
    def __init__(self, env_factory, num_agents, state_size, action_size, device='cpu', **kwargs):
        self.device = device
        self.agents = [PPOAgent(state_size, action_size, device, **kwargs) for _ in range(num_agents)]
        self.envs = [env_factory() for _ in range(num_agents)]
        self.best_model_path = "ppo_models/best_ppo_model.pth"
        self.best_reward = -float('inf')
        
        # Store current dimensions for compatibility checking
        self.state_size = state_size
        self.action_size = action_size
        
        # Training statistics
        self.episode_rewards = []
        self.training_stats = []

    def collect_experience(self, steps_per_update=1000):
        """Collect experience from all agents until minimum episodes completed"""
        min_episodes_required = 1
        
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            steps_collected = 0
            episodes_completed_before = agent.episodes_completed
            
            while (steps_collected < steps_per_update or 
                   agent.episodes_completed - episodes_completed_before < min_episodes_required):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(state, action, log_prob, value, reward, done)
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
                    
                steps_collected += 1

            if agent.episodes_completed == 0:
                while not done:
                    action, log_prob, value = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.store_experience(state, action, log_prob, value, reward, done)
                    state = next_state

    def load_best_model(self):
        """Load the best saved model with improved dimension compatibility checking"""
        if os.path.exists(self.best_model_path):
            try:
                print(f"Attempting to load best model from {self.best_model_path}")
                
                # Load the saved data
                saved_data = torch.load(self.best_model_path, map_location=self.device)
                
                # Handle different save formats
                if isinstance(saved_data, dict) and 'model_state_dict' in saved_data:
                    # New format with metadata
                    state_dict = saved_data['model_state_dict']
                    saved_state_size = saved_data.get('state_size', None)
                    saved_action_size = saved_data.get('action_size', None)
                    
                    print(f"Saved model dimensions: state_size={saved_state_size}, action_size={saved_action_size}")
                    print(f"Current dimensions: state_size={self.state_size}, action_size={self.action_size}")
                    
                    if (saved_state_size is not None and saved_state_size != self.state_size) or \
                       (saved_action_size is not None and saved_action_size != self.action_size):
                        print("Dimension mismatch detected - creating new model")
                        # Backup the incompatible model
                        backup_path = self.best_model_path + f".backup_dim_{saved_state_size}_{saved_action_size}"
                        torch.save(saved_data, backup_path)
                        print(f"Incompatible model backed up to {backup_path}")
                        return False
                    
                    self.best_reward = saved_data.get('best_reward', -float('inf'))
                else:
                    # Old format (just state dict) - check dimensions by examining layers
                    state_dict = saved_data
                    
                    # Check first BatchNorm layer for dimension compatibility
                    bn_key = 'shared.0.weight'  # First BatchNorm layer
                    if bn_key in state_dict:
                        saved_state_size = state_dict[bn_key].shape[0]
                        print(f"Detected saved state_size: {saved_state_size}")
                        print(f"Current state_size: {self.state_size}")
                        
                        if saved_state_size != self.state_size:
                            print("Dimension mismatch in old format model - creating new model")
                            backup_path = self.best_model_path + f".backup_old_format_{saved_state_size}"
                            torch.save(saved_data, backup_path)
                            print(f"Incompatible model backed up to {backup_path}")
                            return False
                
                # Load the compatible model
                for agent in self.agents:
                    agent.network.load_state_dict(state_dict)
                
                print("Successfully loaded compatible model")
                return True
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                # Move problematic model to backup
                if os.path.exists(self.best_model_path):
                    backup_path = self.best_model_path + ".backup_error"
                    os.rename(self.best_model_path, backup_path)
                    print(f"Problematic model moved to {backup_path}")
                return False
        else:
            print("No saved model found")
            return False
    
    def save_model(self, filepath=None):
        """Save the current model with metadata for better compatibility checking"""
        if filepath is None:
            filepath = self.best_model_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save with metadata
        save_data = {
            'model_state_dict': self.agents[0].network.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'best_reward': self.best_reward
        }
        
        torch.save(save_data, filepath)
        print(f"Model with metadata saved to {filepath}")