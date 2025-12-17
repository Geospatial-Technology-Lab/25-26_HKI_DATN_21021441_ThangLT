import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class PolicyIterationOptimized:
    """
    Optimized Policy Iteration for EnhancedCropThermalEnv
    Compatible with the same environment used in Value Iteration
    """
    
    def __init__(self, env, gamma: float = 0.9, theta: float = 1e-4, logger=None):
        """
        Args:
            env: EnhancedCropThermalEnv environment
            gamma: Discount factor
            theta: Convergence threshold
            logger: Logger instance
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.logger = logger if logger else self._setup_default_logger()
        
        # Initialize value function and policy
        self.V = np.zeros((env.height, env.width))
        self.policy = {}
        
        # Pre-compute static components for optimization
        self._precompute_static_components()
        
        # Initialize policy intelligently
        self._initialize_smart_policy()
        
        self.logger.info(f"Initialized Policy Iteration with {env.height}x{env.width} grid")
        self.logger.info(f"Parameters: gamma={gamma}, theta={theta}")
    
    def _setup_default_logger(self):
        """Setup a default logger if none provided"""
        import logging
        logger = logging.getLogger('PolicyIterationOptimized')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[PI] %(message)s'))
            logger.addHandler(handler)
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
        
        # Create high-priority mask for focused computation
        self.high_priority_mask = self.env.fire_ground_truth.copy()
        from scipy.ndimage import binary_dilation
        self.high_priority_mask = binary_dilation(self.high_priority_mask, iterations=2)
    
    def _precompute_weather_risks(self):
        """Pre-compute weather risk for all positions"""
        height, width = self.env.height, self.env.width
        risks = np.ones((height, width)) * 0.5
        
        if not hasattr(self.env, 'weather_patches') or not self.env.weather_patches:
            return risks
        
        factor_count = 0
        risk_accumulator = np.zeros((height, width))
        
        weather_factors = {
            'humidity': lambda x: 1.0 - x,  # Lower humidity = higher risk
            'wind_speed': lambda x: x,       # Higher wind = higher risk
            'soil_moisture': lambda x: 1.0 - x,  # Lower moisture = higher risk
            'rainfall': lambda x: 1.0 - x,   # Lower rainfall = higher risk
            'soil_temp': lambda x: x          # Higher temp = higher risk
        }
        
        for name, transform in weather_factors.items():
            if name in self.env.weather_patches:
                risk_accumulator += transform(self.env.weather_patches[name])
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
                
                # Base probability from temperature
                if temp >= self.env.high_temp_threshold:
                    base_prob = 0.9
                elif temp >= self.env.medium_temp_threshold:
                    base_prob = 0.4
                else:
                    base_prob = 0.05
                
                # Apply modifiers
                weather_risk = self.weather_risks[x, y]
                landcover_modifier = 1.3 if self.env.landcover_data[x, y] == 1 else 0.7
                
                probs[x, y] = np.clip(
                    base_prob * (1 + weather_risk) * landcover_modifier, 
                    0.0, 1.0
                )
        
        return probs
    
    def _initialize_smart_policy(self):
        """Initialize policy intelligently based on environment characteristics"""
        for x in range(self.env.height):
            for y in range(self.env.width):
                state = (x, y)
                valid_actions = self.env._get_valid_actions(state)
                
                if not valid_actions:
                    self.policy[state] = 4  # Stay
                    continue
                
                # Smart initialization based on conditions
                temp = self.env.thermal_data[x, y]
                is_forest = self.env.landcover_data[x, y] == 1
                
                # High temp + fire ground truth -> Predict
                if self.env.fire_ground_truth[x, y] and temp >= self.env.high_temp_threshold:
                    if 5 in valid_actions:  # Predict action
                        self.policy[state] = 5
                    else:
                        self.policy[state] = 4  # Stay
                
                # High temp area -> Stay or move to higher temp
                elif temp >= self.env.high_temp_threshold:
                    if is_forest and 4 in valid_actions:
                        self.policy[state] = 4  # Stay in forest
                    else:
                        # Move to higher temperature neighbor
                        best_action = self._find_action_to_higher_temp(state, valid_actions)
                        self.policy[state] = best_action if best_action is not None else valid_actions[0]
                
                # Medium temp -> Move toward high temp areas
                elif temp >= self.env.medium_temp_threshold:
                    best_action = self._find_action_to_higher_temp(state, valid_actions)
                    self.policy[state] = best_action if best_action is not None else valid_actions[0]
                
                # Low temp -> Explore
                else:
                    # Random exploration
                    self.policy[state] = np.random.choice(valid_actions)
    
    def _find_action_to_higher_temp(self, state: Tuple[int, int], valid_actions: List[int]) -> Optional[int]:
        """Find action that leads to higher temperature"""
        x, y = state
        current_temp = self.env.thermal_data[x, y]
        best_action = None
        best_temp = current_temp
        
        for action in valid_actions:
            if action == 5:  # Skip predict action
                continue
            
            if action == 4:  # Stay
                continue
            
            dx, dy = self.env.action_to_direction[action]
            new_x = np.clip(x + dx, 0, self.env.height - 1)
            new_y = np.clip(y + dy, 0, self.env.width - 1)
            
            new_temp = self.env.thermal_data[new_x, new_y]
            if new_temp > best_temp:
                best_temp = new_temp
                best_action = action
        
        return best_action
    
    def policy_evaluation(self, max_iterations: int = 100) -> np.ndarray:
        """
        Policy Evaluation: Compute value function for current policy
        """
        self.logger.debug("Starting Policy Evaluation")
        
        for iteration in range(max_iterations):
            V_old = self.V.copy()
            max_delta = 0.0
            
            # Focus on high-priority states first
            states_evaluated = 0
            for x in range(self.env.height):
                for y in range(self.env.width):
                    if not self.high_priority_mask[x, y] and self.V[x, y] < 0.1:
                        continue  # Skip low-priority states
                    
                    state = (x, y)
                    if state not in self.policy:
                        continue
                    
                    action = self.policy[state]
                    
                    # Compute expected value for this state-action pair
                    expected_value = self._compute_expected_value(state, action)
                    self.V[x, y] = expected_value
                    
                    delta = abs(self.V[x, y] - V_old[x, y])
                    max_delta = max(max_delta, delta)
                    states_evaluated += 1
            
            # Log progress every 10 iterations
            if iteration % 10 == 0:
                self.logger.debug(f"  Eval iteration {iteration}: max_delta={max_delta:.6f}, states={states_evaluated}")
            
            # Check convergence
            if max_delta < self.theta:
                self.logger.info(f"  Policy Evaluation converged after {iteration + 1} iterations (max_delta={max_delta:.6f})")
                break
        
        return self.V
    
    def policy_improvement(self) -> bool:
        """
        Policy Improvement: Update policy to be greedy with respect to value function
        Returns True if policy changed
        """
        policy_stable = True
        
        for x in range(self.env.height):
            for y in range(self.env.width):
                state = (x, y)
                
                if state not in self.policy:
                    continue
                
                old_action = self.policy[state]
                valid_actions = self.env._get_valid_actions(state)
                
                if not valid_actions:
                    continue
                
                # Find best action
                best_action = valid_actions[0]
                best_value = float('-inf')
                
                for action in valid_actions:
                    action_value = self._compute_expected_value(state, action)
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                self.policy[state] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def _compute_expected_value(self, state: Tuple[int, int], action: int) -> float:
        """Compute expected value for a state-action pair"""
        x, y = state
        
        if action == 5:  # Predict action
            # Immediate reward for prediction
            fire_prob = self.fire_probs[x, y]
            is_fire_gt = self.env.fire_ground_truth[x, y]
            
            if is_fire_gt:
                # True positive with weather bonus
                reward = self.env.true_positive_reward * fire_prob
                reward += self.weather_risks[x, y] * 20
            else:
                # False positive with weather mitigation
                penalty_reduction = self.weather_risks[x, y] * 0.5
                reward = -self.env.false_positive_penalty * fire_prob * (1 - penalty_reduction)
            
            # Future value (stay in same position after prediction)
            future_value = self.gamma * self.V[x, y]
            return reward + future_value
        
        else:  # Movement actions
            # Compute next state
            if action == 4:  # Stay
                next_x, next_y = x, y
                movement_cost = 0
            else:
                dx, dy = self.env.action_to_direction[action]
                next_x = np.clip(x + dx, 0, self.env.height - 1)
                next_y = np.clip(y + dy, 0, self.env.width - 1)
                movement_cost = -self.env.movement_cost
            
            # Base reward from temperature
            base_reward = self.temp_rewards[next_x, next_y]
            
            # Stochastic transitions
            if self.env.obey_prob >= 1.0:
                # Deterministic case
                return base_reward + movement_cost + self.gamma * self.V[next_x, next_y]
            else:
                # Stochastic case
                expected = self.env.obey_prob * (base_reward + movement_cost + self.gamma * self.V[next_x, next_y])
                
                # Add probability of random movement
                if self.env.obey_prob < 1.0:
                    # Simplified: use average of neighbor values
                    avg_neighbor_value = self._get_average_neighbor_value(x, y)
                    expected += (1 - self.env.obey_prob) * self.gamma * avg_neighbor_value
                
                return expected
    
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
    
    def policy_iteration_optimized(self, max_iterations: int = 50) -> Tuple[np.ndarray, Dict]:
        """
        Main Policy Iteration algorithm with optimizations
        """
        self.logger.info("="*60)
        self.logger.info("Starting Optimized Policy Iteration")
        self.logger.info(f"Max iterations: {max_iterations}")
        self.logger.info(f"Environment size: {self.env.height}x{self.env.width}")
        self.logger.info("="*60)
        
        for iteration in range(max_iterations):
            self.logger.info(f"\n>>> Policy Iteration - Iteration {iteration + 1}")
            
            # Policy Evaluation
            self.logger.info("  Running Policy Evaluation...")
            self.V = self.policy_evaluation(max_iterations=100)
            
            # Log value function statistics
            v_mean = np.mean(self.V)
            v_max = np.max(self.V)
            v_min = np.min(self.V)
            self.logger.info(f"  Value function stats: mean={v_mean:.3f}, max={v_max:.3f}, min={v_min:.3f}")
            
            # Policy Improvement
            self.logger.info("  Running Policy Improvement...")
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                self.logger.info(f">>> Policy Iteration CONVERGED after {iteration + 1} iterations!")
                break
            
            # Log action distribution
            action_counts = {i: 0 for i in range(6)}
            for action in self.policy.values():
                if action in action_counts:
                    action_counts[action] += 1
            
            self.logger.info("  Current policy action distribution:")
            action_names = ['Up', 'Down', 'Left', 'Right', 'Stay', 'Predict']
            for i, name in enumerate(action_names):
                if i in action_counts:
                    percentage = (action_counts[i] / len(self.policy)) * 100 if len(self.policy) > 0 else 0
                    self.logger.info(f"    {name}: {action_counts[i]} ({percentage:.1f}%)")
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Policy Iteration Completed")
        self.logger.info("="*60)
        
        # Convert policy to focus on important states only
        final_policy = {}
        important_states = 0
        for state, action in self.policy.items():
            x, y = state
            if self.high_priority_mask[x, y] or self.V[x, y] > 0.1:
                final_policy[state] = action
                important_states += 1
        
        self.logger.info(f"Final policy contains {important_states} important states (out of {len(self.policy)} total)")
        
        return self.V, final_policy
    
    def _print_iteration_stats(self):
        """Print statistics about current policy"""
        action_counts = {i: 0 for i in range(6)}
        for action in self.policy.values():
            if action in action_counts:
                action_counts[action] += 1
        
        print("  Action distribution:")
        action_names = ['Up', 'Down', 'Left', 'Right', 'Stay', 'Predict']
        for i, name in enumerate(action_names):
            if i in action_counts:
                print(f"    {name}: {action_counts[i]}")
        
        print(f"  Value function stats:")
        print(f"    Mean: {np.mean(self.V):.3f}")
        print(f"    Max: {np.max(self.V):.3f}")
        print(f"    Min: {np.min(self.V):.3f}")


def run_policy_iteration_with_enhanced_env(thermal_data: np.ndarray,
                                          start_pos: Tuple[int, int],
                                          weather_patches: Dict[str, np.ndarray],
                                          landcover_data: np.ndarray,
                                          max_steps: int = 500) -> Tuple[np.ndarray, Dict]:
    """
    Run Optimized Policy Iteration with EnhancedCropThermalEnv
    
    Returns:
        V_array: Value function as numpy array
        policy: Optimal policy dictionary
    """
    # Import here to avoid loading torch in main process
    from environment.env_src import create_enhanced_crop_thermal_env
    
    # Create EnhancedCropThermalEnv
    env = create_enhanced_crop_thermal_env(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        obey_prob=0.9,
        high_temp_threshold=0.95,
        medium_temp_threshold=0.85,
        verbose=False
    )
    
    # Suppress verbose output
    import sys
    import os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # Create Policy Iteration solver
        policy_iter = PolicyIterationOptimized(env, gamma=0.9, theta=1e-4)
        
        # Run policy iteration
        V_array, policy = policy_iter.policy_iteration_optimized(max_iterations=50)
        
    finally:
        sys.stdout = old_stdout
        env.close()
    
    return V_array, policy