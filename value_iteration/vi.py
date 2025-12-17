import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import heapq
import warnings
warnings.filterwarnings('ignore')

class AsynchronousValueIterationOptimized:
    """
    Highly optimized Asynchronous Value Iteration with vectorization and early stopping
    """

    def __init__(self, env, gamma: float = 0.9, theta: float = 1e-4, logger=None):
        """
        Args:
            env: EnhancedCropThermalEnv environment
            gamma: Discount factor
            theta: Convergence threshold (increased for speed)
            logger: Logger instance for file logging only
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.logger = logger if logger else self._setup_default_logger()

        # Value function
        self.V = np.zeros((env.height, env.width))
        
        # Pre-compute static rewards and masks for vectorization
        self._precompute_static_components()
        
        # Policy
        self.policy = {}

        self.logger.info(f"Initialized Optimized Async VI with {env.height}x{env.width} grid")
        self.logger.info(f"Parameters: gamma={gamma}, theta={theta}")

    def _setup_default_logger(self):
        """Setup a default logger if none provided - file only"""
        import logging
        logger = logging.getLogger('ValueIterationOptimized')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            # Only create file handler, no console handler
            from datetime import datetime
            file_handler = logging.FileHandler(
                f'logs/value_iteration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        return logger

    def _precompute_static_components(self):
        """Pre-compute components that don't change during iteration"""
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
        
        # Pre-compute weather risk scores
        self.weather_risks = self._precompute_weather_risks()
        
        # Pre-compute fire probabilities
        self.fire_probs = self._precompute_fire_probabilities()
        
        # Create masks for high-priority regions (fire ground truth areas)
        self.high_priority_mask = self.env.fire_ground_truth.copy()
        
        # Expand priority mask to include neighbors of fire pixels
        from scipy.ndimage import binary_dilation
        self.high_priority_mask = binary_dilation(self.high_priority_mask, iterations=2)

    def _precompute_weather_risks(self):
        """Pre-compute weather risk for all positions"""
        height, width = self.env.height, self.env.width
        risks = np.ones((height, width)) * 0.5  # Default risk
        
        if not hasattr(self.env, 'weather_patches') or not self.env.weather_patches:
            return risks
        
        factor_count = 0
        risk_accumulator = np.zeros((height, width))
        
        if 'humidity' in self.env.weather_patches:
            risk_accumulator += (1.0 - self.env.weather_patches['humidity'])
            factor_count += 1
        
        if 'wind_speed' in self.env.weather_patches:
            risk_accumulator += self.env.weather_patches['wind_speed']
            factor_count += 1
        
        if 'soil_moisture' in self.env.weather_patches:
            risk_accumulator += (1.0 - self.env.weather_patches['soil_moisture'])
            factor_count += 1
        
        if 'rainfall' in self.env.weather_patches:
            risk_accumulator += (1.0 - self.env.weather_patches['rainfall'])
            factor_count += 1
        
        if 'soil_temp' in self.env.weather_patches:
            risk_accumulator += self.env.weather_patches['soil_temp']
            factor_count += 1
        
        if factor_count > 0:
            risks = risk_accumulator / factor_count
        
        return np.clip(risks, 0.0, 1.0)

    def _precompute_fire_probabilities(self):
        """Pre-compute fire probabilities for all positions"""
        height, width = self.env.height, self.env.width
        probs = np.zeros((height, width))
        
        for x in range(height):
            for y in range(width):
                temp = self.env.thermal_data[x, y]
                if temp >= self.env.high_temp_threshold:
                    base_prob = 0.9
                elif temp >= self.env.medium_temp_threshold:
                    base_prob = 0.4
                else:
                    base_prob = 0.05
                
                # Apply modifiers
                weather_risk = self.weather_risks[x, y]
                landcover_modifier = 1.3 if self.env.landcover_data[x, y] == 1 else 0.7
                
                probs[x, y] = np.clip(base_prob * (1 + weather_risk) * landcover_modifier, 0.0, 1.0)
        
        return probs

    def bellman_update_vectorized(self, states: List[Tuple[int, int]]) -> List[float]:
        """Vectorized Bellman update for multiple states"""
        deltas = []
        
        for state in states:
            x, y = state
            old_value = self.V[x, y]
            
            # Fast computation using pre-computed values
            max_action_value = self._compute_max_action_value_fast(state)
            
            self.V[x, y] = max_action_value
            deltas.append(abs(self.V[x, y] - old_value))
        
        return deltas

    def _compute_max_action_value_fast(self, state: Tuple[int, int]) -> float:
        """Fast computation using pre-computed values"""
        x, y = state
        valid_actions = self.env._get_valid_actions(state)
        
        if not valid_actions:
            return 0.0
        
        max_value = float('-inf')
        
        for action in valid_actions:
            if action == 5:  # Predict action
                # Use pre-computed fire probability
                fire_prob = self.fire_probs[x, y]
                is_fire_gt = self.env.fire_ground_truth[x, y]
                
                if is_fire_gt:
                    reward = self.env.true_positive_reward * fire_prob
                    reward += self.weather_risks[x, y] * 20
                else:
                    penalty_reduction = self.weather_risks[x, y] * 0.5
                    reward = -self.env.false_positive_penalty * fire_prob * (1 - penalty_reduction)
                
                value = reward + self.gamma * self.V[x, y]
            
            else:  # Movement actions
                if action == 4:  # Stay
                    next_x, next_y = x, y
                else:
                    dx, dy = self.env.action_to_direction[action]
                    next_x = np.clip(x + dx, 0, self.env.height - 1)
                    next_y = np.clip(y + dy, 0, self.env.width - 1)
                
                # Use pre-computed rewards
                base_reward = self.temp_rewards[next_x, next_y]
                movement_cost = -self.env.movement_cost if action != 4 else 0
                
                # Simplified stochastic transitions
                value = self.env.obey_prob * (base_reward + movement_cost + self.gamma * self.V[next_x, next_y])
                
                # Add average of random transitions (simplified)
                if self.env.obey_prob < 1.0:
                    avg_neighbor_value = self._get_average_neighbor_value(x, y)
                    value += (1 - self.env.obey_prob) * self.gamma * avg_neighbor_value
            
            max_value = max(max_value, value)
        
        return max_value

    def _get_average_neighbor_value(self, x: int, y: int) -> float:
        """Get average value of neighboring states"""
        total = 0
        count = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.height and 0 <= ny < self.env.width:
                    total += self.V[nx, ny]
                    count += 1
        
        return total / count if count > 0 else 0

    def async_value_iteration_optimized(self, max_iterations: int = 5000) -> Tuple[np.ndarray, Dict]:
        """
        Highly optimized async VI with early stopping and focused updates
        """
        self.logger.info("="*60)
        self.logger.info("Starting Optimized Async Value Iteration")
        self.logger.info(f"Max iterations: {max_iterations}")
        self.logger.info(f"Environment size: {self.env.height}x{self.env.width}")
        self.logger.info("="*60)
        
        # Use heap-based priority queue for efficiency
        priority_heap = []
        in_heap = set()
        
        # Initialize with high-priority states only
        for x in range(self.env.height):
            for y in range(self.env.width):
                if self.high_priority_mask[x, y]:
                    state = (x, y)
                    # Negative priority for max-heap behavior
                    heapq.heappush(priority_heap, (-self.temp_rewards[x, y], state))
                    in_heap.add(state)
        
        iteration = 0
        converged_count = 0
        last_max_delta = float('inf')
        
        # Early stopping parameters
        patience = 100
        no_improvement_count = 0
        
        while iteration < max_iterations and priority_heap:
            # Process batch of high-priority states
            batch_size = min(50, len(priority_heap))
            batch_states = []
            
            for _ in range(batch_size):
                if not priority_heap:
                    break
                _, state = heapq.heappop(priority_heap)
                in_heap.discard(state)
                batch_states.append(state)
            
            # Vectorized update
            deltas = self.bellman_update_vectorized(batch_states)
            max_delta = max(deltas) if deltas else 0
            
            # Add neighbors of updated states back to queue if needed
            for i, state in enumerate(batch_states):
                if deltas[i] > self.theta:
                    self._add_neighbors_to_queue(state, priority_heap, in_heap)
            
            # Check for convergence
            if max_delta < last_max_delta:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            last_max_delta = max_delta
            
            # Early stopping conditions
            if no_improvement_count >= patience:
                self.logger.info(f"Early stopping at iteration {iteration} (no improvement)")
                break
            
            if max_delta < self.theta * 10:  # Relaxed convergence
                converged_count += 1
                if converged_count > 10:
                    self.logger.info(f"Converged after {iteration} iterations")
                    break
            else:
                converged_count = 0
            
            iteration += batch_size
            
            # Log progress every 500 iterations
            if iteration % 500 == 0:
                self.logger.debug(f"Iteration {iteration}: Queue size = {len(priority_heap)}, Max delta = {max_delta:.6f}")
        
        self.logger.info(f"Optimized Async VI completed after {iteration} iterations")
        self.logger.info(f"Final max delta: {last_max_delta:.6f}")
        self.logger.info(f"Value function stats: mean={np.mean(self.V):.3f}, max={np.max(self.V):.3f}, min={np.min(self.V):.3f}")
        
        # Fast policy extraction focusing on important states
        self.policy = self._extract_policy_fast()
        
        return self.V, self.policy

    def _add_neighbors_to_queue(self, state: Tuple[int, int], priority_heap: list, in_heap: set):
        """Add neighboring states to priority queue"""
        x, y = state
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if (0 <= nx < self.env.height and 
                    0 <= ny < self.env.width and 
                    neighbor not in in_heap):
                    
                    # Priority based on temperature and weather risk
                    priority = -(self.temp_rewards[nx, ny] + self.weather_risks[nx, ny] * 10)
                    heapq.heappush(priority_heap, (priority, neighbor))
                    in_heap.add(neighbor)

    def _extract_policy_fast(self) -> Dict[Tuple[int, int], int]:
        """Fast policy extraction focusing on important states"""
        policy = {}
        
        # Only extract policy for high-priority states
        for x in range(self.env.height):
            for y in range(self.env.width):
                if self.high_priority_mask[x, y] or self.V[x, y] > 0.1:
                    state = (x, y)
                    valid_actions = self.env._get_valid_actions(state)
                    
                    if valid_actions:
                        best_action = max(valid_actions, 
                                        key=lambda a: self._compute_action_value_simple(state, a))
                        policy[state] = best_action
        
        return policy

    def _compute_action_value_simple(self, state: Tuple[int, int], action: int) -> float:
        """Simplified action value computation for policy extraction"""
        x, y = state
        
        if action == 5:  # Predict
            return self.fire_probs[x, y] * self.env.true_positive_reward if self.env.fire_ground_truth[x, y] else -self.env.false_positive_penalty
        elif action == 4:  # Stay
            return self.V[x, y]
        else:  # Movement
            dx, dy = self.env.action_to_direction[action]
            next_x = np.clip(x + dx, 0, self.env.height - 1)
            next_y = np.clip(y + dy, 0, self.env.width - 1)
            return self.V[next_x, next_y] - self.env.movement_cost