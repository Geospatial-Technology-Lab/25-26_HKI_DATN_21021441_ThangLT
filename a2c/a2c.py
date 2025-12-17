import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

class ImprovedActorCriticNetwork(nn.Module):
    """Improved network architecture for A2C"""
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
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
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
        
        # Shared layers
        x1 = torch.relu(self.shared_norm1(self.shared_layer1(x)))
        x1 = self.shared_dropout1(x1)
        
        x2 = torch.relu(self.shared_norm2(self.shared_layer2(x1)))
        x2 = self.shared_dropout2(x2)
        
        # Actor head
        actor_features = torch.relu(self.actor_norm(self.actor_layer1(x2)))
        actor_logits = self.actor_output(actor_features)
        
        # Temperature scaling
        temperature = 0.5
        policy = torch.softmax(actor_logits / temperature, dim=-1)
        
        # Prevent exactly zero probabilities
        policy = policy + 1e-8
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        # Critic head
        critic_features = torch.relu(self.critic_norm(self.critic_layer1(x2)))
        value = self.critic_output(critic_features)
        
        if single_sample:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
            
        return policy, value


class A2CAgent:
    """A2C Agent - Synchronous version of A3C"""
    
    def __init__(self, state_size, action_size, device='cpu', lr=1e-4, 
                 gamma=0.99, entropy_coeff=0.01, value_coeff=0.5, 
                 gae_lambda=0.95):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        
        # Single network (no global/local separation like A3C)
        self.network = ImprovedActorCriticNetwork(state_size, action_size).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=lr,
            eps=1e-5,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)
        
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.gae_lambda = gae_lambda
        
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
        
        # Action tracking
        self.action_counts = np.zeros(action_size)
        
        self.max_grad_norm = 1.0
    
    def get_action(self, state, deterministic=False, epsilon=0.0):
        """Select action with optional epsilon-greedy exploration"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.network(state)
        
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
        """Store experience in buffer"""
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
    
    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        returns = []
        advantages = []
        
        gae = 0
        next_value_bootstrap = next_value
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_value_bootstrap = 0
                gae = 0
            
            # TD error
            delta = self.rewards[i] + self.gamma * next_value_bootstrap - self.values[i]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
            
            next_value_bootstrap = self.values[i]
        
        return returns, advantages
    
    def update(self, next_value=0):
        """Update network using collected experiences"""
        if len(self.states) < 32:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # Forward pass
        policies, values = self.network(states)
        
        # Create distribution
        dist = Categorical(policies)
        
        # Calculate losses
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss
        total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        # Statistics
        stats = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'avg_return': returns.mean().item(),
            'avg_value': values.mean().item(),
            'policy_std': policies.std().item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
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


class MultiAgentA2C:
    """Multi-agent A2C with synchronous updates"""
    
    def __init__(self, env_factory, num_agents, state_size, action_size, 
                 device='cpu', lr=1e-4, gamma=0.99, entropy_coeff=0.01, 
                 value_coeff=0.5, gae_lambda=0.95):
        
        self.device = device
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        # Create agents (each with their own network)
        self.agents = [
            A2CAgent(
                state_size=state_size,
                action_size=action_size,
                device=device,
                lr=lr,
                gamma=gamma,
                entropy_coeff=entropy_coeff,
                value_coeff=value_coeff,
                gae_lambda=gae_lambda
            )
            for i in range(num_agents)
        ]
        
        # Create environments
        self.envs = [env_factory() for _ in range(num_agents)]
        
        # Model saving
        self.best_model_path = "a2c_models/best_a2c_model.pth"
        self.best_reward = -float('inf')
        
        # Training statistics
        self.episode_rewards = []
        self.training_stats = []
        
        self.update_count = 0
    
    def collect_experience(self, steps_per_agent=1000, epsilon=0.05):
        """Collect experience from all agents synchronously"""
        all_stats = []
        
        for agent_idx, (agent, env) in enumerate(zip(self.agents, self.envs)):
            state = env.reset()
            steps_collected = 0
            
            # Decay epsilon over time
            current_epsilon = epsilon * (0.99 ** self.update_count)
            
            while steps_collected < steps_per_agent:
                action, log_prob, value = agent.get_action(state, epsilon=current_epsilon)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(state, action, reward, value, log_prob, done)
                
                if done:
                    # Update on episode end
                    stats = agent.update(next_value=0)
                    if stats:
                        all_stats.append(stats)
                    state = env.reset()
                else:
                    state = next_state
                
                steps_collected += 1
                
                # Periodic update every 20 steps
                if len(agent.states) >= 20:
                    with torch.no_grad():
                        _, next_value = agent.network(
                            torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        )
                        next_value = next_value.item()
                    
                    stats = agent.update(next_value)
                    if stats:
                        all_stats.append(stats)
            
            # Print action distribution for first agent periodically
            if agent_idx == 0 and self.update_count % 10 == 0:
                action_dist = agent.get_action_distribution()
                print(f"\n  Agent 0 action distribution: {action_dist}")
        
        # Aggregate statistics from all agents
        if all_stats:
            self.training_stats.extend(all_stats)
        
        self.update_count += 1
    
    def save_model(self, filepath=None):
        """Save model (save first agent's network as representative)"""
        if filepath is None:
            filepath = self.best_model_path
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'model_state_dict': self.agents[0].network.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'best_reward': self.best_reward,
            'update_count': self.update_count
        }
        
        torch.save(save_data, filepath)
        print(f"A2C model saved to {filepath}")
    
    def load_best_model(self):
        """Load model to all agents"""
        if os.path.exists(self.best_model_path):
            try:
                print(f"Attempting to load A2C model from {self.best_model_path}")
                
                saved_data = torch.load(self.best_model_path, map_location=self.device)
                
                if isinstance(saved_data, dict) and 'model_state_dict' in saved_data:
                    saved_state_size = saved_data.get('state_size')
                    saved_action_size = saved_data.get('action_size')
                    
                    print(f"Saved model dimensions: state_size={saved_state_size}, action_size={saved_action_size}")
                    print(f"Current dimensions: state_size={self.state_size}, action_size={self.action_size}")
                    
                    if (saved_state_size != self.state_size or 
                        saved_action_size != self.action_size):
                        print("Dimension mismatch - creating new model")
                        backup_path = self.best_model_path + f".backup_dim_{saved_state_size}_{saved_action_size}"
                        torch.save(saved_data, backup_path)
                        print(f"Incompatible model backed up to {backup_path}")
                        return False
                    
                    # Check architecture
                    state_dict = saved_data['model_state_dict']
                    if 'actor_layer1.weight' not in state_dict:
                        print("Old architecture detected - not loading")
                        return False
                    
                    # Load to all agents
                    for agent in self.agents:
                        agent.network.load_state_dict(state_dict)
                    
                    self.best_reward = saved_data.get('best_reward', -float('inf'))
                    self.update_count = saved_data.get('update_count', 0)
                    
                    print("Successfully loaded A2C model to all agents")
                    return True
                else:
                    print("Invalid save format")
                    return False
                    
            except Exception as e:
                print(f"Error loading model: {e}")
                if os.path.exists(self.best_model_path):
                    backup_path = self.best_model_path + ".backup_error"
                    os.rename(self.best_model_path, backup_path)
                    print(f"Problematic model moved to {backup_path}")
                return False
        else:
            print("No saved model found")
            return False