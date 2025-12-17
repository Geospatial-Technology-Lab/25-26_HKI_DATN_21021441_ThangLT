import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import threading

class ImprovedActorCriticNetwork(nn.Module):
    """Improved network architecture to prevent collapse"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        self.state_size = state_size
        
        # Shared feature extractor with skip connections
        self.input_norm = nn.LayerNorm(state_size)
        
        self.shared_layer1 = nn.Linear(state_size, hidden_size)
        self.shared_norm1 = nn.LayerNorm(hidden_size)
        self.shared_dropout1 = nn.Dropout(0.1)
        
        self.shared_layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.shared_norm2 = nn.LayerNorm(hidden_size // 2)
        self.shared_dropout2 = nn.Dropout(0.1)
        
        # Separate heads for actor and critic
        self.actor_layer1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.actor_norm = nn.LayerNorm(hidden_size // 4)
        self.actor_output = nn.Linear(hidden_size // 4, action_size)
        
        self.critic_layer1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.critic_norm = nn.LayerNorm(hidden_size // 4)
        self.critic_output = nn.Linear(hidden_size // 4, 1)
        
        # Initialize with Xavier/Glorot to prevent vanishing gradients
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # Handle batching
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        
        # Input normalization
        x = self.input_norm(state)
        
        # Shared layers with residual-like connections
        x1 = torch.relu(self.shared_norm1(self.shared_layer1(x)))
        x1 = self.shared_dropout1(x1)
        
        x2 = torch.relu(self.shared_norm2(self.shared_layer2(x1)))
        x2 = self.shared_dropout2(x2)
        
        # Actor head
        actor_features = torch.relu(self.actor_norm(self.actor_layer1(x2)))
        actor_logits = self.actor_output(actor_features)
        
        # Use temperature scaling to prevent collapse
        temperature = 0.5  # Lower = more diverse actions
        policy = torch.softmax(actor_logits / temperature, dim=-1)
        
        # Add small epsilon to prevent exactly zero probabilities
        policy = policy + 1e-8
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        # Critic head
        critic_features = torch.relu(self.critic_norm(self.critic_layer1(x2)))
        value = self.critic_output(critic_features)
        
        if single_sample:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
            
        return policy, value


class ImprovedA3CAgent:
    """A3C Agent with improvements to prevent collapse"""
    
    def __init__(self, agent_id, global_network, optimizer, state_size, action_size, 
                 device='cpu', gamma=0.99, entropy_coeff=0.05, value_coeff=0.5):
        self.agent_id = agent_id
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # Local network
        self.local_network = ImprovedActorCriticNetwork(state_size, action_size).to(device)
        self.local_network.load_state_dict(global_network.state_dict())
        
        self.global_network = global_network
        self.optimizer = optimizer
        
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff  # Higher to encourage exploration
        self.value_coeff = value_coeff
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_length = 0
        self.episodes_completed = 0
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Action tracking for debugging
        self.action_counts = np.zeros(action_size)
        
        self.max_grad_norm = 1.0  # Increased for stability
    
    def sync_with_global(self):
        """Synchronize local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def get_action(self, state, deterministic=False, epsilon=0.0):
        """Select action with optional epsilon-greedy for exploration"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.local_network(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon and not deterministic:
            action = np.random.randint(0, self.action_size)
            log_prob = torch.log(policy[0, action]).item()
        elif deterministic:
            action = torch.argmax(policy, dim=-1).item()
            log_prob = torch.log(policy[0, action]).item()
        else:
            try:
                dist = Categorical(policy)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
            except:
                action = np.random.randint(0, self.action_size)
                log_prob = torch.log(policy[0, action]).item()
        
        # Track action distribution
        self.action_counts[action] += 1
        
        return action, log_prob, value.item()
    
    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in local buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
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
    
    def compute_returns(self, next_value=0):
        """Compute returns with GAE (Generalized Advantage Estimation)"""
        returns = []
        advantages = []
        
        # GAE parameters
        gae_lambda = 0.95
        
        gae = 0
        next_value_bootstrap = next_value
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_value_bootstrap = 0
                gae = 0
            
            # TD error
            delta = self.rewards[i] + self.gamma * next_value_bootstrap - self.values[i]
            
            # GAE
            gae = delta + self.gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
            
            next_value_bootstrap = self.values[i]
        
        return returns, advantages
    
    def update_global(self, next_value=0):
        """Compute gradients and update global network"""
        if len(self.states) == 0:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        policies, values = self.local_network(states)
        
        # Create distribution
        dist = Categorical(policies)
        
        # Calculate losses
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Actor loss with advantage
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss with entropy bonus
        total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.max_grad_norm)
        
        # Transfer gradients to global
        for local_param, global_param in zip(self.local_network.parameters(), 
                                             self.global_network.parameters()):
            if local_param.grad is not None:
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad
        
        # Update global network
        self.optimizer.step()
        
        # Sync with global
        self.sync_with_global()
        
        # Get statistics
        stats = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'avg_return': returns.mean().item(),
            'avg_value': values.mean().item(),
            'policy_std': policies.std().item()
        }
        
        # Clear buffer
        self.clear_buffer()
        
        return stats
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_action_distribution(self):
        """Get action distribution for debugging"""
        if self.action_counts.sum() == 0:
            return np.zeros(self.action_size)
        return self.action_counts / self.action_counts.sum()


class ImprovedMultiAgentA3C:
    """Improved Multi-agent A3C with better training dynamics"""
    
    def __init__(self, env_factory, num_agents, state_size, action_size, 
                 device='cpu', lr=1e-4, gamma=0.99, entropy_coeff=0.05, 
                 value_coeff=0.5, update_interval=20):
        
        self.device = device
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.update_interval = update_interval
        
        # Global network
        self.global_network = ImprovedActorCriticNetwork(state_size, action_size).to(device)
        self.global_network.share_memory()
        
        # Optimizer with better hyperparameters
        self.optimizer = optim.Adam(
            self.global_network.parameters(), 
            lr=lr,
            eps=1e-5,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Create agents
        self.agents = [
            ImprovedA3CAgent(
                agent_id=i,
                global_network=self.global_network,
                optimizer=self.optimizer,
                state_size=state_size,
                action_size=action_size,
                device=device,
                gamma=gamma,
                entropy_coeff=entropy_coeff,
                value_coeff=value_coeff
            )
            for i in range(num_agents)
        ]
        
        # Create environments
        self.envs = [env_factory() for _ in range(num_agents)]
        
        # Model saving
        self.best_model_path = "a3c_models/best_a3c_model.pth"
        self.best_reward = -float('inf')
        
        # Training statistics
        self.episode_rewards = []
        self.training_stats = []
        
        # Thread lock
        self.lock = threading.Lock()
        
        self.update_count = 0
    
    def collect_experience(self, steps_per_agent=1000, epsilon=0.05, parallel=False):
        """Collect experience with epsilon-greedy exploration - supports parallel collection"""
        
        # Decay epsilon over time
        current_epsilon = epsilon * (0.99 ** self.update_count)
        
        def collect_single_agent(agent_idx):
            """Collect experience for a single agent"""
            agent = self.agents[agent_idx]
            env = self.envs[agent_idx]
            
            state = env.reset()
            steps_collected = 0
            local_stats = []
            
            while steps_collected < steps_per_agent:
                action, log_prob, value = agent.get_action(state, epsilon=current_epsilon)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(state, action, reward, value, log_prob, done)
                
                if done:
                    with self.lock:
                        stats = agent.update_global(next_value=0)
                        if stats:
                            local_stats.append(stats)
                    state = env.reset()
                else:
                    state = next_state
                
                steps_collected += 1
                
                # Periodic update
                if len(agent.states) >= self.update_interval:
                    with torch.no_grad():
                        _, next_value = agent.local_network(
                            torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        )
                        next_value = next_value.item()
                    
                    with self.lock:
                        stats = agent.update_global(next_value)
                        if stats:
                            local_stats.append(stats)
            
            return agent_idx, local_stats
        
        if parallel and self.num_agents > 1:
            # Parallel collection using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=min(self.num_agents, 4)) as executor:
                futures = {
                    executor.submit(collect_single_agent, idx): idx 
                    for idx in range(self.num_agents)
                }
                
                for future in as_completed(futures):
                    agent_idx, local_stats = future.result()
                    self.training_stats.extend(local_stats)
                    
                    # Print action distribution for first agent
                    if agent_idx == 0 and self.update_count % 10 == 0:
                        action_dist = self.agents[0].get_action_distribution()
                        print(f"\n  Agent 0 action distribution: {action_dist}")
        else:
            # Sequential collection (original behavior)
            for agent_idx, (agent, env) in enumerate(zip(self.agents, self.envs)):
                state = env.reset()
                steps_collected = 0
                
                while steps_collected < steps_per_agent:
                    action, log_prob, value = agent.get_action(state, epsilon=current_epsilon)
                    next_state, reward, done, info = env.step(action)
                    
                    agent.store_experience(state, action, reward, value, log_prob, done)
                    
                    if done:
                        stats = agent.update_global(next_value=0)
                        if stats:
                            self.training_stats.append(stats)
                        state = env.reset()
                    else:
                        state = next_state
                    
                    steps_collected += 1
                    
                    # Periodic update
                    if len(agent.states) >= self.update_interval:
                        with torch.no_grad():
                            _, next_value = agent.local_network(
                                torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                            )
                            next_value = next_value.item()
                        
                        stats = agent.update_global(next_value)
                        if stats:
                            self.training_stats.append(stats)
                
                # Print action distribution for first agent
                if agent_idx == 0 and self.update_count % 10 == 0:
                    action_dist = agent.get_action_distribution()
                    print(f"\n  Agent 0 action distribution: {action_dist}")
        
        self.update_count += 1
        self.scheduler.step()
    
    def save_model(self, filepath=None):
        """Save model"""
        if filepath is None:
            filepath = self.best_model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'model_state_dict': self.global_network.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'best_reward': self.best_reward,
            'update_count': self.update_count
        }
        
        torch.save(save_data, filepath)
    
    def load_best_model(self):
        """Load model"""
        if os.path.exists(self.best_model_path):
            try:
                saved_data = torch.load(self.best_model_path, map_location=self.device)
                
                if isinstance(saved_data, dict) and 'model_state_dict' in saved_data:
                    saved_state_size = saved_data.get('state_size')
                    saved_action_size = saved_data.get('action_size')
                    
                    if (saved_state_size != self.state_size or 
                        saved_action_size != self.action_size):
                        print("Dimension mismatch - model not loaded")
                        return False
                    
                    # Check if it's the improved architecture
                    state_dict = saved_data['model_state_dict']
                    if 'actor_layer1.weight' not in state_dict:
                        print("Old architecture detected - not loading")
                        return False
                    
                    self.global_network.load_state_dict(state_dict)
                    self.best_reward = saved_data.get('best_reward', -float('inf'))
                    self.update_count = saved_data.get('update_count', 0)
                    
                    for agent in self.agents:
                        agent.sync_with_global()
                    
                    print("Successfully loaded improved A3C model")
                    return True
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        print("No saved model found")
        return False