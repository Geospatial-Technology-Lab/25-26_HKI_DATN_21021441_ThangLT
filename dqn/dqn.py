import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
import random


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture
    Tương tự ActorCriticNetwork nhưng chỉ output Q-values
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extraction layers
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
        
        # Q-value head - outputs Q(s,a) for each action
        self.q_values = nn.Sequential(
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state):
        """
        Forward pass
        Returns Q-values for all actions
        """
        # Handle single sample case
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        
        features = self.shared(state)
        q_vals = self.q_values(features)
        
        if single_sample:
            q_vals = q_vals.squeeze(0)
        
        return q_vals


class ReplayBuffer:
    """
    Experience Replay Buffer cho DQN
    Lưu trữ transitions và sample random batches
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        # Simple priority: higher absolute reward = higher priority
        priority = abs(reward) + 0.1
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Sample random batch"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Simple uniform sampling
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
        
        # Convert priorities to probabilities
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


class DQNAgent:
    """
    Deep Q-Network Agent
    Implements Double DQN with Experience Replay
    """
    def __init__(self, 
                 state_size, 
                 action_size, 
                 device='cpu',
                 lr=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_capacity=50000,
                 batch_size=64,
                 target_update_freq=100,
                 use_double_dqn=True,
                 use_prioritized_replay=False):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of actions
            device: 'cpu' or 'cuda'
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            target_update_freq: How often to update target network
            use_double_dqn: Whether to use Double DQN
            use_prioritized_replay: Whether to use prioritized experience replay
        """
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Networks
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.95
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
        
        # For compatibility with PPO interface
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Gradient clipping
        self.max_grad_norm = 1.0
    
    def get_action(self, state, deterministic=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            deterministic: If True, always select best action (no exploration)
        
        Returns:
            action: Selected action
            log_prob: Not used in DQN but included for interface compatibility
            value: Q-value of selected action
        """
        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
            # Get Q-value for random action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                value = q_values[0, action].item()
            return action, 0.0, value  # log_prob=0 for random action
        
        # Greedy action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            value = q_values[0, action].item()
        
        return action, 0.0, value  # log_prob not used in DQN
    
    def get_action_distribution(self, state):
        """
        Get action distribution for ensemble
        Convert Q-values to probabilities using softmax
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            # Convert Q-values to probabilities
            action_probs = F.softmax(q_values / 0.1, dim=1)  # Temperature=0.1
        
        return action_probs.cpu().numpy()[0]
    
    def store_experience(self, state, action, log_prob, value, reward, done):
        """
        Store transition in replay buffer
        Note: log_prob and value not used in DQN but kept for interface compatibility
        """
        # We need next_state, so store temporarily
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
            
            # Reset temporary storage
            delattr(self, '_last_state')
        else:
            # Update for next transition
            self._last_state = state
            self._last_action = action
            self._last_reward = reward
    
    def update(self, update_epochs=1):
        """
        Update Q-network using experience replay
        
        Args:
            update_epochs: Number of update iterations (not used much in DQN)
        
        Returns:
            Dictionary of training statistics
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        total_loss = 0.0
        total_q_values = 0.0
        num_updates = 0
        
        # Multiple updates per call
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
            
            # Current Q-values
            current_q_values = self.q_network(states)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: use Q-network to select action, target network to evaluate
                    next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                    next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                else:
                    # Standard DQN: use target network for both
                    next_q = self.target_network(next_states).max(dim=1)[0]
                
                target_q = rewards + self.gamma * next_q * (1 - dones)
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q, target_q)  # Huber loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), 
                max_norm=self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_q_values += current_q.mean().item()
            num_updates += 1
            
            self.steps += 1
            
            # Update target network periodically
            if self.steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update learning rate
        self.scheduler.step()
        
        # Record loss
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        self.loss_history.append(avg_loss)
        
        # Return statistics
        return {
            'total_loss': avg_loss,
            'q_values': total_q_values / num_updates if num_updates > 0 else 0,
            'epsilon': self.epsilon,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'buffer_size': len(self.replay_buffer),
            'episodes_completed': self.episodes_completed,
            'avg_episode_reward': self.reward_mean,
            'reward_std': self.reward_std
        }
    
    def clear_buffer(self):
        """Clear experience buffer - for interface compatibility"""
        pass  # DQN doesn't clear buffer like PPO
    
    def save_model(self, path):
        """Save DQN model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes_completed': self.episodes_completed,
            'reward_history': self.reward_history,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, path)
        
        print(f"DQN model saved to {path}")
    
    def load_model(self, path):
        """Load DQN model"""
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
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Load training state
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.episodes_completed = checkpoint['episodes_completed']
            self.reward_history = checkpoint['reward_history']
            
            if len(self.reward_history) > 0:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = max(np.std(self.reward_history), 1.0)
            
            print(f"DQN model loaded from {path}")
            print(f"  Episodes completed: {self.episodes_completed}")
            print(f"  Current epsilon: {self.epsilon:.4f}")
            print(f"  Average reward: {self.reward_mean:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading DQN model: {str(e)}")
            return False


class MultiAgentDQN:
    """
    Multi-agent DQN trainer
    Similar to MultiAgentPPO but for DQN
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
            **kwargs: Additional arguments for DQNAgent
        """
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # Create agents
        self.agents = [
            DQNAgent(state_size, action_size, device, **kwargs) 
            for _ in range(num_agents)
        ]
        
        # Create environments
        self.envs = [env_factory() for _ in range(num_agents)]
        
        # Model paths
        self.best_model_path = "dqn_models/best_dqn_model.pth"
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
                    agent.q_network.load_state_dict(
                        self.agents[0].q_network.state_dict()
                    )
                    agent.target_network.load_state_dict(
                        self.agents[0].target_network.state_dict()
                    )
                
                # Load metadata
                metadata_path = self.best_model_path.replace('.pth', '_metadata.pth')
                if os.path.exists(metadata_path):
                    metadata = torch.load(metadata_path, map_location=self.device)
                    self.best_reward = metadata['best_reward']
                    self.episode_rewards = metadata['episode_rewards']
                
                return True
        
        return False