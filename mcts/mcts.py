import math
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MCTSNode:
    """Node in MCTS tree for EnhancedCropThermalEnv"""
    
    def __init__(self, state: Tuple[int, int], parent=None, action=None):
        self.state = state  # (x, y) position
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None
        
        # Enhanced statistics
        self.fire_predictions = 0
        self.true_positives = 0
        self.false_positives = 0
        self.forest_visits = 0
        self.high_temp_visits = 0
    
    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def ucb1_score(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c=math.sqrt(2)):
        return max(self.children.values(), key=lambda child: child.ucb1_score(c))
    
    def add_child(self, action, child_state):
        child = MCTSNode(child_state, parent=self, action=action)
        self.children[action] = child
        if self.untried_actions and action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

class MCTSOptimized:
    """
    Optimized MCTS for EnhancedCropThermalEnv
    Compatible with the same environment used in Value/Policy Iteration
    """
    
    def __init__(self, env, c_param=math.sqrt(2), max_rollout_steps=50, logger=None):
        """
        Args:
            env: EnhancedCropThermalEnv environment
            c_param: UCB1 exploration constant
            max_rollout_steps: Maximum steps in rollout
            logger: Logger instance
        """
        self.env = env
        self.c_param = c_param
        self.max_rollout_steps = max_rollout_steps
        self.logger = logger if logger else self._setup_default_logger()
        
        # Pre-compute static components
        self._precompute_static_components()
        
        # Statistics tracking
        self.visited_states = defaultdict(int)
        self.state_rewards = defaultdict(float)
        self.forest_states = set()
        self.fire_prediction_states = set()
        
        self.logger.info(f"Initialized MCTS with {env.height}x{env.width} grid")
        self.logger.info(f"Parameters: c={c_param}, max_rollout={max_rollout_steps}")
    
    def _setup_default_logger(self):
        """Setup a default logger if none provided - file only"""
        import logging
        logger = logging.getLogger('MCTSOptimized')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            from datetime import datetime
            file_handler = logging.FileHandler(
                f'logs/mcts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        return logger
    
    def _precompute_static_components(self):
        """Pre-compute components for faster MCTS"""
        height, width = self.env.height, self.env.width
        
        # Pre-compute temperature-based priorities
        self.temp_priorities = np.zeros((height, width))
        for x in range(height):
            for y in range(width):
                temp = self.env.thermal_data[x, y]
                if temp >= self.env.high_temp_threshold:
                    self.temp_priorities[x, y] = 10.0
                elif temp >= self.env.medium_temp_threshold:
                    self.temp_priorities[x, y] = 5.0
                else:
                    self.temp_priorities[x, y] = 1.0
        
        # Pre-compute weather risks
        self.weather_risks = self._precompute_weather_risks()
        
        # Pre-compute fire probabilities
        self.fire_probs = self._precompute_fire_probabilities()
        
        # High priority mask
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
            'humidity': lambda x: 1.0 - x,
            'wind_speed': lambda x: x,
            'soil_moisture': lambda x: 1.0 - x,
            'rainfall': lambda x: 1.0 - x,
            'soil_temp': lambda x: x
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
                
                if temp >= self.env.high_temp_threshold:
                    base_prob = 0.9
                elif temp >= self.env.medium_temp_threshold:
                    base_prob = 0.4
                else:
                    base_prob = 0.05
                
                weather_risk = self.weather_risks[x, y]
                landcover_modifier = 1.3 if self.env.landcover_data[x, y] == 1 else 0.7
                
                probs[x, y] = np.clip(
                    base_prob * (1 + weather_risk) * landcover_modifier,
                    0.0, 1.0
                )
        
        return probs
    
    def _get_valid_actions(self, state: Tuple[int, int]):
        """Get valid actions for a state"""
        return self.env._get_valid_actions(state)
    
    def _get_action_priorities(self, state: Tuple[int, int]) -> Dict[int, float]:
        """Get action priorities based on state"""
        x, y = state
        valid_actions = self._get_valid_actions(state)
        priorities = {}
        
        for action in valid_actions:
            if action == 5:  # Predict action
                # High priority if high fire probability
                priorities[action] = self.fire_probs[x, y] * 10
            elif action == 4:  # Stay action
                priorities[action] = self.temp_priorities[x, y]
            else:  # Movement actions
                dx, dy = self.env.action_to_direction[action]
                next_x = np.clip(x + dx, 0, self.env.height - 1)
                next_y = np.clip(y + dy, 0, self.env.width - 1)
                
                # Priority based on temperature and forest
                temp_priority = self.temp_priorities[next_x, next_y]
                forest_bonus = 2.0 if self.env.landcover_data[next_x, next_y] == 1 else 0
                
                priorities[action] = temp_priority + forest_bonus
        
        return priorities
    
    def selection(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using UCB1"""
        while not node.is_leaf() and node.is_fully_expanded():
            node = node.best_child(self.c_param)
        return node
    
    def expansion(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a child"""
        if node.untried_actions is None:
            node.untried_actions = list(self._get_valid_actions(node.state))
        
        if node.untried_actions:
            # Choose action based on priorities
            priorities = self._get_action_priorities(node.state)
            untried_priorities = {a: priorities.get(a, 1.0) for a in node.untried_actions}
            
            # Weighted random choice
            actions = list(untried_priorities.keys())
            weights = list(untried_priorities.values())
            action = random.choices(actions, weights=weights)[0]
            
            # Simulate action to get next state
            next_state = self._simulate_action(node.state, action)
            child = node.add_child(action, next_state)
            
            # Update statistics
            x, y = next_state
            if self.env.landcover_data[x, y] == 1:
                child.forest_visits += 1
            if self.env.thermal_data[x, y] >= self.env.high_temp_threshold:
                child.high_temp_visits += 1
            
            return child
        
        return node
    
    def _simulate_action(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Simulate taking an action from a state"""
        x, y = state
        
        if action == 5 or action == 4:  # Predict or Stay
            return state
        else:  # Movement
            dx, dy = self.env.action_to_direction[action]
            next_x = np.clip(x + dx, 0, self.env.height - 1)
            next_y = np.clip(y + dy, 0, self.env.width - 1)
            
            # Apply stochasticity
            if np.random.random() >= self.env.obey_prob:
                # Random movement
                valid_actions = [a for a in range(5) if a in self._get_valid_actions(state)]
                if valid_actions:
                    random_action = np.random.choice(valid_actions)
                    if random_action != 4:
                        dx, dy = self.env.action_to_direction[random_action]
                        next_x = np.clip(x + dx, 0, self.env.height - 1)
                        next_y = np.clip(y + dy, 0, self.env.width - 1)
            
            return (next_x, next_y)
    
    def simulation(self, node: MCTSNode) -> float:
        """Run a rollout simulation from node"""
        total_reward = 0.0
        current_state = node.state
        visited_in_rollout = set()
        
        for step in range(self.max_rollout_steps):
            if current_state in visited_in_rollout:
                break
            visited_in_rollout.add(current_state)
            
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break
            
            # Smart action selection during rollout
            priorities = self._get_action_priorities(current_state)
            actions = list(priorities.keys())
            weights = [priorities[a] for a in actions]
            action = random.choices(actions, weights=weights)[0]
            
            # Calculate reward
            reward = self._calculate_reward(current_state, action)
            total_reward += reward * (self.env.gamma ** step)
            
            # Get next state
            if action not in [4, 5]:  # Not Stay or Predict
                current_state = self._simulate_action(current_state, action)
        
        return total_reward
    
    def _calculate_reward(self, state: Tuple[int, int], action: int) -> float:
        """Calculate reward for state-action pair"""
        x, y = state
        
        if action == 5:  # Predict action
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
            
            return reward
        
        elif action == 4:  # Stay
            return self.temp_priorities[x, y] * 0.1
        
        else:  # Movement
            next_state = self._simulate_action(state, action)
            next_x, next_y = next_state
            
            # Base reward from temperature
            base_reward = self.temp_priorities[next_x, next_y] * 0.1
            
            # Exploration bonus
            if next_state not in self.visited_states:
                base_reward += 0.5
            
            # Forest bonus
            if self.env.landcover_data[next_x, next_y] == 1:
                base_reward += 1.0
            
            return base_reward - self.env.movement_cost
    
    def backpropagation(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            
            # Update state statistics
            self.visited_states[node.state] += 1
            self.state_rewards[node.state] += reward
            
            # Track special states
            x, y = node.state
            if self.env.landcover_data[x, y] == 1:
                self.forest_states.add(node.state)
            
            node = node.parent
    
    def search(self, start_state: Tuple[int, int], num_iterations: int = 1000) -> MCTSNode:
        """Run MCTS search from start state"""
        self.logger.info("="*60)
        self.logger.info("Starting MCTS Search")
        self.logger.info(f"Iterations: {num_iterations}")
        self.logger.info(f"Start position: {start_state}")
        self.logger.info("="*60)
        
        root = MCTSNode(start_state)
        
        for i in range(num_iterations):
            # Selection
            selected_node = self.selection(root)
            
            # Expansion
            if not selected_node.is_fully_expanded():
                selected_node = self.expansion(selected_node)
            
            # Simulation
            reward = self.simulation(selected_node)
            
            # Backpropagation
            self.backpropagation(selected_node, reward)
            
            # Log progress
            if (i + 1) % 100 == 0:
                avg_reward = root.total_reward / max(root.visits, 1)
                self.logger.debug(f"Iteration {i+1}: Root visits={root.visits}, Avg reward={avg_reward:.3f}")
        
        self.logger.info(f"MCTS completed. Root visits: {root.visits}")
        self.logger.info(f"Average reward: {root.total_reward/max(root.visits, 1):.3f}")
        self.logger.info(f"States visited: {len(self.visited_states)}")
        
        return root
    
    def get_value_map(self, root: MCTSNode) -> np.ndarray:
        """Extract value map from MCTS tree"""
        value_map = np.zeros((self.env.height, self.env.width))
        
        # Fill from visited states
        for state, visits in self.visited_states.items():
            x, y = state
            if visits > 0:
                avg_reward = self.state_rewards[state] / visits
                value_map[x, y] = avg_reward
        
        # Traverse tree for additional information
        def traverse_tree(node, depth=0):
            if depth > 10:  # Limit depth
                return
            
            if node.visits > 0:
                x, y = node.state
                avg_reward = node.total_reward / node.visits
                # Use maximum between existing and new value
                value_map[x, y] = max(value_map[x, y], avg_reward)
            
            for child in node.children.values():
                traverse_tree(child, depth + 1)
        
        traverse_tree(root)
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter
        value_map = gaussian_filter(value_map, sigma=1.0)
        
        # Normalize
        if np.max(value_map) > np.min(value_map):
            value_map = (value_map - np.min(value_map)) / (np.max(value_map) - np.min(value_map))
        
        return value_map
    
    def get_policy(self, root: MCTSNode) -> Dict[Tuple[int, int], int]:
        """Extract policy from MCTS tree"""
        policy = {}
        
        def extract_policy(node):
            if node.children:
                # Choose best action based on visit count
                best_action = None
                best_visits = -1
                
                for action, child in node.children.items():
                    if child.visits > best_visits:
                        best_visits = child.visits
                        best_action = action
                
                if best_action is not None:
                    policy[node.state] = best_action
                
                # Recursively extract from children
                for child in node.children.values():
                    extract_policy(child)
        
        extract_policy(root)
        return policy


def run_mcts_with_enhanced_env(thermal_data: np.ndarray,
                               start_pos: Tuple[int, int],
                               weather_patches: Dict[str, np.ndarray],
                               landcover_data: np.ndarray,
                               max_steps: int = 500,
                               num_iterations: int = 1000,
                               logger=None) -> Tuple[np.ndarray, Dict]:
    """
    Run MCTS with EnhancedCropThermalEnv
    
    Returns:
        value_map: Value map as numpy array
        policy: Policy dictionary
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
    
    # Add gamma if not present
    if not hasattr(env, 'gamma'):
        env.gamma = 0.9
    
    # Create MCTS solver
    mcts = MCTSOptimized(env, c_param=math.sqrt(2), max_rollout_steps=50, logger=logger)
    
    # Run MCTS search
    root = mcts.search(start_pos, num_iterations=num_iterations)
    
    # Get value map and policy
    value_map = mcts.get_value_map(root)
    policy = mcts.get_policy(root)
    
    env.close()
    
    return value_map, policy