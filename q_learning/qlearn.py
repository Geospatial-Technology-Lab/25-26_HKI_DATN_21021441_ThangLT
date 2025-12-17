import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class QLearningOptimized:
    """
    Optimized Q-Learning for EnhancedCropThermalEnv with faster convergence
    """
    
    def __init__(self, env, alpha=0.15, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, 
                 min_epsilon=0.05, logger=None, early_stopping=True):
        """
        Args:
            env: EnhancedCropThermalEnv environment
            alpha: Learning rate (increased for faster learning)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            min_epsilon: Minimum epsilon value
            logger: Logger instance
            early_stopping: Enable early stopping
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.logger = logger
        self.early_stopping = early_stopping
        
        # Initialize Q-table with better initialization
        self.Q = {}
        self._initialize_q_table_smart()
        
        # Pre-compute static components for optimization
        self._precompute_static_components()
        
        # Statistics tracking
        self.episode_rewards = []
        self.visited_states = defaultdict(int)
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.convergence_history = []
        
        # Early stopping tracking
        self.best_avg_reward = float('-inf')
        self.patience_counter = 0
        
        if self.logger:
            self.logger.info(f"Initialized Optimized Q-Learning with {env.height}x{env.width} grid")
    
    def _initialize_q_table_smart(self):
        """Smart Q-table initialization based on domain knowledge"""
        for x in range(self.env.height):
            for y in range(self.env.width):
                state = (x, y)
                self.Q[state] = {}
                
                # Get temperature at this position
                temp = self.env.thermal_data[x, y] if hasattr(self.env, 'thermal_data') else 0.5
                
                for action in range(self.env.action_space.n):
                    if action == 5:  # Predict action
                        # Initialize predict action based on temperature
                        if temp >= 0.9:
                            self.Q[state][action] = 0.5  # Higher initial value for hot spots
                        elif temp >= 0.8:
                            self.Q[state][action] = 0.2
                        else:
                            self.Q[state][action] = -0.1
                    elif action == 4:  # Stay action
                        self.Q[state][action] = 0.0
                    else:  # Movement actions
                        # Slight positive bias for exploration
                        self.Q[state][action] = 0.1
    
    def _precompute_static_components(self):
        """Pre-compute static components for optimization"""
        height, width = self.env.height, self.env.width
        
        # Pre-compute temperature-based rewards
        self.temp_rewards = np.zeros((height, width))
        for x in range(height):
            for y in range(width):
                temp = self.env.thermal_data[x, y]
                if temp >= self.env.high_temp_threshold:
                    self.temp_rewards[x, y] = 10.0
                elif temp >= self.env.medium_temp_threshold:
                    self.temp_rewards[x, y] = 5.0
                else:
                    self.temp_rewards[x, y] = 0.1
        
        # Pre-compute weather risks (simplified)
        self.weather_risks = self._precompute_weather_risks_fast()
        
        # Pre-compute fire probabilities
        self.fire_probs = self._precompute_fire_probabilities_fast()
        
        # High priority mask (areas near fire ground truth)
        self.high_priority_mask = self.env.fire_ground_truth.copy()
        from scipy.ndimage import binary_dilation
        self.high_priority_mask = binary_dilation(self.high_priority_mask, iterations=1)
        
        # Pre-compute valid actions for each state
        self.valid_actions_cache = {}
        for x in range(height):
            for y in range(width):
                self.valid_actions_cache[(x, y)] = self.env._get_valid_actions((x, y))
    
    def _precompute_weather_risks_fast(self):
        """Fast pre-computation of weather risk"""
        height, width = self.env.height, self.env.width
        
        # Simple risk based on normalized thermal data
        risks = self.env.thermal_data.copy()
        
        # Add simple weather influence if available
        if hasattr(self.env, 'weather_patches') and self.env.weather_patches:
            if 'humidity' in self.env.weather_patches:
                risks *= (1.0 - self.env.weather_patches['humidity'] * 0.3)
            if 'wind_speed' in self.env.weather_patches:
                risks *= (1.0 + self.env.weather_patches['wind_speed'] * 0.2)
        
        return np.clip(risks, 0.0, 1.0)
    
    def _precompute_fire_probabilities_fast(self):
        """Fast pre-computation of fire probabilities"""
        # Base probabilities from temperature
        probs = self.env.thermal_data.copy()
        
        # Apply thresholds
        probs[probs >= self.env.high_temp_threshold] = 0.9
        probs[(probs >= self.env.medium_temp_threshold) & 
              (probs < self.env.high_temp_threshold)] = 0.5
        probs[probs < self.env.medium_temp_threshold] = 0.1
        
        # Landcover influence
        if hasattr(self.env, 'landcover_data'):
            forest_mask = self.env.landcover_data == 1
            probs[forest_mask] *= 1.2
            probs[~forest_mask] *= 0.8
        
        return np.clip(probs, 0.0, 1.0)
    
    def get_state_from_observation(self, obs):
        """Extract state (x, y) from observation"""
        # Fast extraction assuming first two elements are positions
        x = int(obs[0] * self.env.height) if obs[0] <= 1.0 else int(obs[0])
        y = int(obs[1] * self.env.width) if obs[1] <= 1.0 else int(obs[1])
        
        # Clip to bounds
        x = min(max(0, x), self.env.height - 1)
        y = min(max(0, y), self.env.width - 1)
        
        return (x, y)
    
    def select_action(self, state: Tuple[int, int], epsilon=None) -> int:
        """Optimized action selection with caching"""
        if epsilon is None:
            epsilon = self.epsilon
        
        # Use cached valid actions
        valid_actions = self.valid_actions_cache.get(state, [4])
        if not valid_actions:
            return 4  # Stay action as default
        
        if np.random.random() < epsilon:
            # Smart exploration
            return self._guided_exploration_fast(state, valid_actions)
        else:
            # Exploitation - vectorized for speed
            q_values = [self.Q[state][a] for a in valid_actions]
            return valid_actions[np.argmax(q_values)]
    
    def _guided_exploration_fast(self, state: Tuple[int, int], valid_actions: List[int]) -> int:
        """Fast guided exploration"""
        x, y = state
        
        # Quick decision based on fire probability
        if np.random.random() < self.fire_probs[x, y] and 5 in valid_actions:
            return 5  # Predict with high probability in high-risk areas
        
        # Otherwise, prefer movement towards higher temperatures
        if len(valid_actions) > 1:
            # Remove stay action for exploration
            move_actions = [a for a in valid_actions if a != 4]
            if move_actions:
                return np.random.choice(move_actions)
        
        return np.random.choice(valid_actions)
    
    def update_q_value_batch(self, transitions):
        """Batch Q-value updates for efficiency"""
        for state, action, reward, next_state, done in transitions:
            if done:
                target = reward
            else:
                valid_next = self.valid_actions_cache.get(next_state, [])
                if valid_next:
                    max_q_next = max([self.Q[next_state][a] for a in valid_next])
                    target = reward + self.gamma * max_q_next
                else:
                    target = reward
            
            # Q-learning update
            self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def train(self, num_episodes: int = 100, max_steps_per_episode: int = 200,
              convergence_threshold: float = 1e-3, patience: int = 30) -> Dict:
        """
        Optimized training with early stopping and batch updates
        """
        if self.logger:
            self.logger.info(f"Starting optimized training: {num_episodes} episodes")
        
        # Use deque for efficient reward tracking
        recent_rewards = deque(maxlen=20)
        convergence_check_interval = 10
        
        for episode in range(num_episodes):
            # Reset environment
            obs = self.env.reset()
            state = self.get_state_from_observation(obs)
            
            episode_reward = 0
            transitions = []
            
            # Episode loop
            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(state, self.epsilon)
                
                # Take action
                next_obs, reward, done, info = self.env.step(action)
                next_state = self.get_state_from_observation(next_obs)
                
                # Store transition
                transitions.append((state, action, reward, next_state, done))
                
                # Update statistics
                episode_reward += reward
                self.visited_states[state] += 1
                
                state = next_state
                
                if done:
                    break
            
            # Batch update Q-values
            self.update_q_value_batch(transitions)
            
            # Update epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Track rewards
            self.episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            
            # Early stopping check
            if self.early_stopping and episode % convergence_check_interval == 0 and episode > 0:
                avg_reward = np.mean(recent_rewards)
                
                # Check for convergence
                if len(recent_rewards) == 20:
                    reward_std = np.std(recent_rewards)
                    
                    # If rewards have stabilized
                    if reward_std < convergence_threshold:
                        if self.logger:
                            self.logger.info(f"Converged at episode {episode} (std={reward_std:.4f})")
                        break
                    
                    # Check for improvement
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                        if self.patience_counter >= patience // convergence_check_interval:
                            if self.logger:
                                self.logger.info(f"Early stopping at episode {episode} (no improvement)")
                            break
            
            # Minimal logging
            if self.logger and episode % 50 == 0 and episode > 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                self.logger.info(f"Episode {episode}: Avg reward={avg_reward:.3f}, ε={self.epsilon:.3f}")
        
        if self.logger:
            self.logger.info(f"Training completed. Final epsilon: {self.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'final_epsilon': self.epsilon,
            'episodes_trained': episode + 1,
            'states_visited': len(self.visited_states)
        }
    
    def get_value_function(self) -> np.ndarray:
        """Fast extraction of value function"""
        V = np.zeros((self.env.height, self.env.width))
        
        # Vectorized computation where possible
        for x in range(self.env.height):
            for y in range(self.env.width):
                state = (x, y)
                valid_actions = self.valid_actions_cache.get(state, [])
                
                if valid_actions:
                    V[x, y] = max([self.Q[state][a] for a in valid_actions])
        
        # Light smoothing for visualization
        from scipy.ndimage import gaussian_filter
        V_smoothed = gaussian_filter(V, sigma=0.5)
        
        # Fast normalization
        v_min, v_max = V_smoothed.min(), V_smoothed.max()
        if v_max > v_min:
            V_smoothed = (V_smoothed - v_min) / (v_max - v_min)
        
        return V_smoothed
    
    def get_policy(self) -> Dict[Tuple[int, int], int]:
        """Extract greedy policy from Q-table"""
        policy = {}
        
        # Only extract policy for visited or important states
        for state in self.visited_states:
            x, y = state
            if self.high_priority_mask[x, y] or self.visited_states[state] > 2:
                valid_actions = self.valid_actions_cache.get(state, [])
                if valid_actions:
                    q_values = [self.Q[state][a] for a in valid_actions]
                    policy[state] = valid_actions[np.argmax(q_values)]
        
        return policy