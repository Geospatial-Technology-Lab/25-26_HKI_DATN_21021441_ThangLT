import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
import random
import copy


class ActorNetwork(nn.Module):
    """
    Actor Network for DDPG
    Outputs action probabilities (policy) for discrete actions
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
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
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)  # Output action probabilities
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state):
        """Forward pass - returns action probabilities"""
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        
        action_probs = self.network(state)
        
        if single_sample:
            action_probs = action_probs.squeeze(0)
        
        return action_probs


class CriticNetwork(nn.Module):
    """
    Critic Network for DDPG
    Estimates Q-value given state and action
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # State processing
        self.state_net = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        
        # Action processing
        self.action_net = nn.Sequential(
            nn.Linear(action_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state, action):
        """
        Forward pass
        Args:
            state: State tensor [batch, state_size]
            action: Action tensor [batch, action_size] (one-hot or probabilities)
        Returns:
            Q-value [batch, 1]
        """
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            single_sample = True
        
        # Process state and action separately
        state_features = self.state_net(state)
        action_features = self.action_net(action)
        
        # Combine features
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.combined_net(combined)
        
        if single_sample:
            q_value = q_value.squeeze(0)
        
        return q_value


class ReplayBuffer:
    """
    Experience Replay Buffer for DDPG
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        priority = abs(reward) + 0.1
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Sample random batch"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def sample_prioritized(self, batch_size, alpha=0.6):
        """Sample with prioritization"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities, dtype=np.float32)
        priorities = priorities ** alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent
    Adapted for discrete action spaces
    """
    def __init__(self, 
                 state_size, 
                 action_size, 
                 device='cpu',
                 actor_lr=1e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 buffer_capacity=50000,
                 batch_size=64,
                 noise_scale=0.2,
                 noise_decay=0.995,
                 use_prioritized_replay=False):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            device: 'cpu' or 'cuda'
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter (Polyak averaging)
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            noise_scale: Exploration noise scale
            noise_decay: Noise decay rate
            use_prioritized_replay: Whether to use prioritized replay
        """
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.use_prioritized_replay = use_prioritized_replay
        
        # Actor networks
        self.actor = ActorNetwork(state_size, action_size).to(device)
        self.actor_target = ActorNetwork(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = CriticNetwork(state_size, action_size).to(device)
        self.critic_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=200, gamma=0.95
        )
        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=200, gamma=0.95
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training statistics
        self.steps = 0
        self.episodes_completed = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.reward_history = []
        self.loss_history = []
        
        # For compatibility with PPO/DQN interface
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Gradient clipping
        self.max_grad_norm = 1.0
    
    def get_action(self, state, deterministic=False):
        """
        Select action using actor network with exploration noise
        
        Args:
            state: Current state
            deterministic: If True, select best action (no noise)
        
        Returns:
            action: Selected action
            log_prob: Not used in DDPG but kept for interface compatibility
            value: Q-value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor).cpu().numpy()[0]
        
        if deterministic:
            # Greedy action selection
            action = np.argmax(action_probs)
        else:
            # Add exploration noise (Gumbel noise for discrete actions)
            noise = np.random.gumbel(size=action_probs.shape) * self.noise_scale
            noisy_probs = action_probs + noise
            noisy_probs = np.exp(noisy_probs) / np.exp(noisy_probs).sum()
            
            # Sample action
            action = np.random.choice(self.action_size, p=noisy_probs)
        
        # Get Q-value for selected action
        with torch.no_grad():
            action_onehot = torch.zeros(1, self.action_size).to(self.device)
            action_onehot[0, action] = 1.0
            value = self.critic(state_tensor, action_onehot).item()
        
        return action, 0.0, value
    
    def get_action_distribution(self, state):
        """
        Get action distribution for ensemble
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        return action_probs.cpu().numpy()[0]
    
    def store_experience(self, state, action, log_prob, value, reward, done):
        """
        Store transition in replay buffer
        Note: log_prob and value not used but kept for interface compatibility
        """
        # Need next_state, so store temporarily
        if not hasattr(self, '_last_state'):
            self._last_state = state
            self._last_action = action
            self._last_reward = reward
            return
        
        # Add transition to buffer
        self.replay_buffer.push(
            self._last_state,
            self._last_action,
            self._last_reward,
            state,
            done
        )
        
        # Update episode statistics
        self.episode_reward += reward
        self.episode_length += 1
        
        # If episode ended
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
            
            # Decay noise
            self.noise_scale = max(0.01, self.noise_scale * self.noise_decay)
            
            # Reset temporary storage
            delattr(self, '_last_state')
        else:
            # Update for next transition
            self._last_state = state
            self._last_action = action
            self._last_reward = reward
    
    def update(self, update_epochs=1):
        """
        Update actor and critic networks using DDPG algorithm
        
        Args:
            update_epochs: Number of update iterations
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_q_values = 0.0
        num_updates = 0
        
        for _ in range(update_epochs):
            # Sample batch from replay buffer
            if self.use_prioritized_replay:
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample_prioritized(self.batch_size)
            else:
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample(self.batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Convert actions to one-hot
            actions_onehot = F.one_hot(actions, self.action_size).float()
            
            # ============ Update Critic ============
            with torch.no_grad():
                # Get target actions from target actor
                next_action_probs = self.actor_target(next_states)
                
                # Compute target Q-values using target critic
                target_q = self.critic_target(next_states, next_action_probs)
                target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
            
            # Current Q-values
            current_q = self.critic(states, actions_onehot)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q, target_q)
            
            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                max_norm=self.max_grad_norm
            )
            self.critic_optimizer.step()
            
            # ============ Update Actor ============
            # Get current action probabilities
            action_probs = self.actor(states)
            
            # Actor loss: maximize Q-value (negative because we minimize)
            actor_loss = -self.critic(states, action_probs).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                max_norm=self.max_grad_norm
            )
            self.actor_optimizer.step()
            
            # ============ Soft update target networks ============
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            # Statistics
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_q_values += current_q.mean().item()
            num_updates += 1
            
            self.steps += 1
        
        # Update learning rates
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # Record loss
        avg_loss = (total_critic_loss + total_actor_loss) / (2 * num_updates) if num_updates > 0 else 0
        self.loss_history.append(avg_loss)
        
        # Return statistics
        return {
            'total_loss': avg_loss,
            'critic_loss': total_critic_loss / num_updates if num_updates > 0 else 0,
            'actor_loss': total_actor_loss / num_updates if num_updates > 0 else 0,
            'q_values': total_q_values / num_updates if num_updates > 0 else 0,
            'noise_scale': self.noise_scale,
            'actor_lr': self.actor_scheduler.get_last_lr()[0],
            'critic_lr': self.critic_scheduler.get_last_lr()[0],
            'buffer_size': len(self.replay_buffer),
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': self.reward_mean,
            'reward_std': self.reward_std
        }
    
    def _soft_update(self, source, target):
        """
        Soft update target network using Polyak averaging
        target = tau * source + (1 - tau) * target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def clear_buffer(self):
        """Clear experience buffer - for interface compatibility"""
        pass  # DDPG doesn't clear buffer like PPO
    
    def save_model(self, path):
        """Save DDPG model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'noise_scale': self.noise_scale,
            'steps': self.steps,
            'episodes_completed': self.episodes_completed,
            'reward_history': self.reward_history,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, path)
        
        print(f"DDPG model saved to {path}")
    
    def load_model(self, path):
        """Load DDPG model"""
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check dimension compatibility
            saved_state_size = checkpoint.get('state_size', None)
            saved_action_size = checkpoint.get('action_size', None)
            
            if saved_state_size and saved_state_size != self.state_size:
                print(f"State size mismatch: saved={saved_state_size}, current={self.state_size}")
                return False
            
            if saved_action_size and saved_action_size != self.action_size:
                print(f"Action size mismatch: saved={saved_action_size}, current={self.action_size}")
                return False
            
            # Load networks
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
            
            # Load training state
            self.noise_scale = checkpoint['noise_scale']
            self.steps = checkpoint['steps']
            self.episodes_completed = checkpoint['episodes_completed']
            self.reward_history = checkpoint['reward_history']
            
            if len(self.reward_history) > 0:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = max(np.std(self.reward_history), 1.0)
            
            print(f"DDPG model loaded from {path}")
            print(f"  Episodes completed: {self.episodes_completed}")
            print(f"  Current noise scale: {self.noise_scale:.4f}")
            print(f"  Average reward: {self.reward_mean:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading DDPG model: {str(e)}")
            return False


class MultiAgentDDPG:
    """
    Multi-agent DDPG trainer
    Similar to MultiAgentPPO and MultiAgentDQN
    """
    def __init__(self, 
                 env_factory, 
                 num_agents, 
                 state_size, 
                 action_size, 
                 device='cpu',
                 **kwargs):
        """
        Args:
            env_factory: Function that creates environment
            num_agents: Number of parallel agents
            state_size: State dimension
            action_size: Action dimension
            device: 'cpu' or 'cuda'
            **kwargs: Additional arguments for DDPGAgent
        """
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # Create agents
        self.agents = [
            DDPGAgent(state_size, action_size, device, **kwargs) 
            for _ in range(num_agents)
        ]
        
        # Create environments
        self.envs = [env_factory() for _ in range(num_agents)]
        
        # Model paths
        self.best_model_path = "ddpg_models/best_ddpg_model.pth"
        self.best_reward = -float('inf')
        
        # Training statistics
        self.episode_rewards = []
        self.training_stats = []
    
    def collect_experience(self, steps_per_update=2000):
        """
        Collect experience from all agents
        
        Args:
            steps_per_update: Number of steps to collect
        """
        for agent, env in zip(self.agents, self.envs):
            state = env.reset()
            steps_collected = 0
            
            while steps_collected < steps_per_update:
                # Get action
                action, log_prob, value = agent.get_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, log_prob, value, reward, done)
                
                steps_collected += 1
                
                if done:
                    state = env.reset()
                else:
                    state = next_state
    
    def save_model(self, filepath=None):
        """Save best model"""
        if filepath is None:
            filepath = self.best_model_path
        
        # Save first agent as representative
        self.agents[0].save_model(filepath)
        
        # Save additional metadata
        metadata_path = filepath.replace('.pth', '_metadata.pth')
        torch.save({
            'best_reward': self.best_reward,
            'num_agents': len(self.agents),
            'episode_rewards': self.episode_rewards
        }, metadata_path)
    
    def load_best_model(self):
        """Load best model"""
        if os.path.exists(self.best_model_path):
            # Load to all agents
            success = self.agents[0].load_model(self.best_model_path)
            
            if success:
                # Copy to other agents
                for agent in self.agents[1:]:
                    agent.actor.load_state_dict(
                        self.agents[0].actor.state_dict()
                    )
                    agent.actor_target.load_state_dict(
                        self.agents[0].actor_target.state_dict()
                    )
                    agent.critic.load_state_dict(
                        self.agents[0].critic.state_dict()
                    )
                    agent.critic_target.load_state_dict(
                        self.agents[0].critic_target.state_dict()
                    )
                
                # Load metadata
                metadata_path = self.best_model_path.replace('.pth', '_metadata.pth')
                if os.path.exists(metadata_path):
                    metadata = torch.load(metadata_path, map_location=self.device)
                    self.best_reward = metadata['best_reward']
                    self.episode_rewards = metadata['episode_rewards']
                
                return True
        
        return False