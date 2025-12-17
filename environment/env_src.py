import gym
import torch
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def compute_weather_statistics(weather_patches: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute min/max statistics of weather patches for reference.
    Weather data is already normalized to [-1, 1], so stats should be close to that range.
    """
    stats = {}
    for name, patch in weather_patches.items():
        # Exclude NaN values
        valid_data = patch[~np.isnan(patch)]
        if len(valid_data) > 0:
            stats[name] = {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data))
            }
    return stats

class EnhancedCropThermalEnv(gym.Env):
    def __init__(self,
                thermal_data: np.ndarray,
                start_pos: Tuple[int, int],
                weather_patches: Optional[Dict[str, np.ndarray]] = None,
                landcover_data: Optional[np.ndarray] = None,
                max_steps: int = 1000,
                obey_prob: float = 0.9,
                high_temp_threshold: float = 0.95,
                medium_temp_threshold: float = 0.85,
                verbose: bool = False):
        super(EnhancedCropThermalEnv, self).__init__()

        self.thermal_data = thermal_data
        self.start_pos = start_pos
        self.weather_patches = weather_patches or {}
        self.max_steps = max_steps
        self.obey_prob = obey_prob
        
        self.landcover_data = landcover_data if landcover_data is not None else np.ones_like(thermal_data)
        
        # Balanced reward parameters
        self.true_positive_reward = 100.0
        self.false_positive_penalty = 300.0  # Significantly increased
        self.true_negative_reward = 10.0     # Increased
        self.false_negative_penalty = 50.0
        self.exploration_reward = 0.1        # Reduced
        self.movement_cost = 0.2             # Increased

        self.high_temp_threshold = high_temp_threshold
        self.medium_temp_threshold = medium_temp_threshold

        self.height, self.width = thermal_data.shape

        self.action_space = spaces.Discrete(6)
        self.action_to_direction = {
            0: (-1, 0),  # U (Up)
            1: (1, 0),   # D (Down)
            2: (0, -1),  # L (Left)
            3: (0, 1),   # R (Right)
            4: (0, 0),   # S (Stay)
            5: (0, 0)    # P (Predict Fire)
        }
        self.action_names = ['U', 'D', 'L', 'R', 'S', 'P']

        # Tính toán động số lượng weather variables
        # Base features: 2 (position) + 3 (temp) + 2 (neighborhood) + 2 (metrics)
        num_base_features = 9
        # Weather variables: humidity, wind_speed, soil_temp, soil_moisture, rainfall, ndmi, dem
        num_weather_features = 7
        total_features = num_base_features + num_weather_features

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_features,),  # Use the calculated total
            dtype=np.float32
        )

        self.weather_stats = compute_weather_statistics(self.weather_patches) if self.weather_patches else {}
        
        self.verbose = verbose

        self.fire_ground_truth = self.thermal_data >= self.high_temp_threshold
        self.total_fire_pixels = int(np.sum(self.fire_ground_truth))
        self.total_non_fire_pixels = int(np.sum(~self.fire_ground_truth))

        # Tracking variables
        self.fire_predictions = np.zeros_like(self.thermal_data, dtype=bool)
        self.prediction_counts = np.zeros_like(self.thermal_data, dtype=int)
        self.visited_positions = set()

        # Episode statistics
        self.true_positives = 0
        self.false_positives = 0
        self.total_predictions = 0
        self.steps_since_last_prediction = 0
        
        # Initialize
        self.current_pos = None
        self.step_count = 0
        self.episode_reward = 0

        # Improved observation cache with statistics
        self._observation_cache = {}
        self._max_cache_size = 5000  # Increased cache size
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Pre-compute neighborhood statistics for faster lookups
        self._precompute_neighborhood_stats()
    
    def _precompute_neighborhood_stats(self):
        """Pre-compute neighborhood statistics for all positions using vectorized operations"""
        if HAS_SCIPY:
            # Use scipy for fast uniform filtering
            self._avg_temp_map = uniform_filter(self.thermal_data, size=3, mode='nearest')
            self._fire_neighbor_count = uniform_filter(
                self.fire_ground_truth.astype(np.float32), size=3, mode='constant', cval=0
            ) * 9  # 9 cells in 3x3 kernel, subtract center later
        else:
            # Fallback: no precomputation
            self._avg_temp_map = None
            self._fire_neighbor_count = None
    
    def _get_batch_observations(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        """Get observations for multiple positions in batch - much faster for evaluation"""
        observations = np.zeros((len(positions), self.observation_space.shape[0]), dtype=np.float32)
        
        for idx, pos in enumerate(positions):
            observations[idx] = self._get_observation(pos)
        
        return observations
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics for debugging"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            'cache_size': len(self._observation_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }

    def _get_observation(self, pos: Tuple[int, int]) -> np.ndarray:
        cache_key = (pos[0], pos[1])
        if cache_key in self._observation_cache:
            self._cache_hits += 1
            return self._observation_cache[cache_key]
        
        self._cache_misses += 1
        x, y = pos
        observation_features = []
        
        # 1. Position features (2)
        observation_features.extend([
            x / self.height,  # Normalized x
            y / self.width    # Normalized y
        ])
        
        # 2. Essential temperature features (3)
        current_temp = self.thermal_data[x, y]
        observation_features.extend([
            current_temp,                                    # Current temperature 
            float(current_temp >= self.high_temp_threshold), # High temp flag
            float(current_temp >= self.medium_temp_threshold)# Medium temp flag
        ])
        
        # 3. Simplified neighborhood features (2)
        temp_neighbors = self._get_neighborhood_features(pos)
        observation_features.extend([
            temp_neighbors['avg_temp'],  # Average neighborhood temp
            temp_neighbors['fire_count'] # Number of fire neighbors
        ])
        
        # 4. Most important weather features (7)
        weather_feature_names = ['humidity', 'wind_speed', 'soil_temp', 
                                'soil_moisture', 'rainfall', 'ndmi', 'dem']
        for weather_name in weather_feature_names:
            if weather_name in self.weather_patches:
                weather_value = self.weather_patches[weather_name][x, y]
                observation_features.append(float(weather_value))
        
        # 5. Critical performance metrics (2)
        precision = self._get_current_precision() 
        recall = self._get_current_recall()
        observation_features.extend([precision, recall])

        # Total features: 16 (2 + 3 + 2 + 7 + 2)
        observation_array = np.array(observation_features, dtype=np.float32)
        
        # Cache with LRU-style eviction
        if len(self._observation_cache) >= self._max_cache_size:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._observation_cache.keys())[:self._max_cache_size // 5]
            for key in keys_to_remove:
                del self._observation_cache[key]
        
        self._observation_cache[cache_key] = observation_array
        return observation_array

    def _get_neighborhood_features(self, pos: Tuple[int, int]) -> Dict:
        x, y = pos
        
        # Use precomputed maps if available (much faster)
        if self._avg_temp_map is not None:
            avg_temp = (self._avg_temp_map[x, y] * 2 - 1)  # Normalize to [-1, 1]
            # Estimate fire count from precomputed map
            fire_count_raw = self._fire_neighbor_count[x, y]
            # Subtract center cell contribution
            fire_count = max(0, fire_count_raw - float(self.fire_ground_truth[x, y]))
            fire_count_norm = (fire_count / 8.0) * 2 - 1
            
            # Compute max/min/gradient from local neighborhood (still need these)
            x_start, x_end = max(0, x-1), min(self.height, x+2)
            y_start, y_end = max(0, y-1), min(self.width, y+2)
            neighborhood = self.thermal_data[x_start:x_end, y_start:y_end]
            max_temp = (np.max(neighborhood) * 2 - 1)
            min_temp = (np.min(neighborhood) * 2 - 1)
            temp_gradient = max_temp - min_temp
            
            return {
                'avg_temp': avg_temp,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'temp_gradient': temp_gradient,
                'fire_count': fire_count_norm
            }
        
        # Fallback: original implementation
        temps = []
        fire_count = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    temp = self.thermal_data[nx, ny]
                    temps.append(temp)
                    if self.fire_ground_truth[nx, ny]:
                        fire_count += 1
        
        if temps:
            avg_temp = (np.mean(temps) * 2 - 1)  # Normalize to [-1, 1]
            max_temp = (np.max(temps) * 2 - 1)   # Normalize to [-1, 1]
            min_temp = (np.min(temps) * 2 - 1)   # Add missing min_temp calculation
            temp_gradient = ((np.max(temps) - np.min(temps)) * 2 - 1)  # Normalize
        else:
            avg_temp = (self.thermal_data[x, y] * 2 - 1)
            max_temp = avg_temp
            min_temp = avg_temp  # Add missing min_temp for else case
            temp_gradient = 0
        
        fire_count_norm = (fire_count / 8.0) * 2 - 1  # Max 8 neighbors, normalize to [-1, 1]
        
        return {
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'min_temp': min_temp,  # Add missing min_temp to return dict
            'temp_gradient': temp_gradient,
            'fire_count': fire_count_norm
        }

    def _calculate_fire_probability(self, pos: Tuple[int, int]) -> float:
        """Optimized fire probability calculation"""
        x, y = pos
        
        # Pre-compute neighborhoods for all weather parameters at once
        x_start = max(0, x-2)
        x_end = min(x+3, self.height) 
        y_start = max(0, y-2)
        y_end = min(y+3, self.width)
        
        # Cache neighborhoods
        weather_neighborhoods = {}
        weather_stats = {}
        
        for name, patch in self.weather_patches.items():
            neighborhood = patch[x_start:x_end, y_start:y_end]
            weather_neighborhoods[name] = neighborhood
            
            # Pre-compute statistics
            weather_stats[name] = {
                'current': patch[x, y],
                'mean': np.mean(neighborhood),
                'std': np.std(neighborhood),
                'min': np.min(neighborhood),
                'max': np.max(neighborhood)
            }
        
        # Temperature-based probability
        temp = self.thermal_data[x, y]
        temp_prob = (0.9 if temp >= self.high_temp_threshold else 
                    0.2 if temp >= self.medium_temp_threshold else 
                    0.001)
        
        # Define weather impact weights
        weather_weights = {
            'rainfall': -0.45,  # Negative impact
            'humidity': -0.30,  # Negative impact
            'wind_speed': 0.25,
            'soil_temp': 0.40,
            'soil_moisture': -0.35,  # Negative impact
            'dem': -0.15,  # Higher elevation = less fire risk
            'ndmi': -0.20   # Higher moisture index = less fire
        }
        
        # Calculate weather modifier using normalized values from [-1, 1]
        weather_modifier = 1.0
        for name, weight in weather_weights.items():
            if name in weather_stats:
                # Weather data is already normalized to [-1, 1] from process_weather_patches
                norm_val = weather_stats[name]['current']
                
                # Interpret normalized value:
                # -1 means minimum in the region (worst conditions for this variable)
                # +1 means maximum in the region (best conditions for this variable)
                
                # For negative impact variables, invert (low values = bad)
                if weight < 0:
                    norm_val = -norm_val  # Convert: low value -> high impact
                    weight = abs(weight)
                
                # Apply weight to modifier
                weather_modifier *= (1 + norm_val * weight)
        
        # Combine probabilities
        final_prob = temp_prob * weather_modifier
        
        # Apply landcover modifier
        forest_modifier = 1.2 if self.landcover_data[x, y] == 1 else 0.4
        final_prob *= forest_modifier
        
        return np.clip(final_prob, 0, 1)

    def _get_current_precision(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.true_positives / self.total_predictions

    def _get_current_recall(self) -> float:
        if self.total_fire_pixels == 0:
            return 0.0
        return self.true_positives / self.total_fire_pixels

    def step(self, action: int):
        if self.current_pos is None:
            raise ValueError("Environment not reset!")

        # Check for valid action
        valid_actions = self._get_valid_actions(self.current_pos)
        if action not in valid_actions:
            return self._get_observation(self.current_pos), -1.0, False, {
                'invalid_action': True,
                'valid_actions': valid_actions
            }

        reward = 0.0
        info = {'made_prediction': False}
        
        if action == 5:  # Predict Fire
            reward = self._handle_fire_prediction()
            info['made_prediction'] = True
            self.steps_since_last_prediction = 0
            new_pos = self.current_pos
        else:
            # Movement action
            reward = self._handle_movement(action)
            new_pos = self._get_new_position(action)
            self.steps_since_last_prediction += 1
            
            # Small penalty for not making predictions for too long
            if self.steps_since_last_prediction > 20:
                reward -= 0.5

        # Update position and counters
        self.current_pos = new_pos
        self.step_count += 1
        self.episode_reward += reward

        # Check termination conditions
        done = False
        if self.step_count >= self.max_steps:
            done = True
            # End-of-episode bonus based on performance
            f1_score = self._calculate_f1_score()
            reward += f1_score * 5.0  # Bonus for good overall performance

        # Add information to info dict
        info.update({
            'step_count': self.step_count,
            'position': self.current_pos,
            'temperature': self.thermal_data[new_pos[0], new_pos[1]],
            'is_fire_gt': self.fire_ground_truth[new_pos[0], new_pos[1]],
            'fire_probability': self._calculate_fire_probability(new_pos),
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'total_predictions': self.total_predictions,
            'precision': self._get_current_precision(),
            'recall': self._get_current_recall(),
            'episode_reward': self.episode_reward
        })

        return self._get_observation(new_pos), reward, done, info

    def _get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        x, y = pos
        valid = []
        
        # Movement actions
        for action in range(5):
            if action == 4:  # Stay action is always valid
                valid.append(action)
            else:
                dx, dy = self.action_to_direction[action]
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.height and 0 <= new_y < self.width:
                    valid.append(action)
        
        # Predict action is always valid
        valid.append(5)
        return valid

    def _get_new_position(self, action: int) -> Tuple[int, int]:
        if action == 4 or action == 5:  # Stay or Predict
            return self.current_pos
        
        # Apply some randomness to movement
        if np.random.random() < self.obey_prob:
            actual_action = action
        else:
            # Choose a random valid movement action
            valid_moves = [a for a in range(5) if a in self._get_valid_actions(self.current_pos)]
            actual_action = np.random.choice(valid_moves)
        
        dx, dy = self.action_to_direction[actual_action]
        new_x = np.clip(self.current_pos[0] + dx, 0, self.height - 1)
        new_y = np.clip(self.current_pos[1] + dy, 0, self.width - 1)
        
        return (new_x, new_y)

    def _handle_movement(self, action: int) -> float:
        new_pos = self._get_new_position(action)
        reward = -self.movement_cost  # Base movement cost
        
        # Reward for exploring new areas
        if new_pos not in self.visited_positions:
            fire_prob = self._calculate_fire_probability(new_pos)
            reward += self.exploration_reward * fire_prob
            self.visited_positions.add(new_pos)
        
        # Reward for moving toward higher temperatures
        current_temp = self.thermal_data[self.current_pos[0], self.current_pos[1]]
        new_temp = self.thermal_data[new_pos[0], new_pos[1]]
        if new_temp > current_temp and new_temp > self.medium_temp_threshold:
            reward += 0.5
        
        return reward

    def _handle_fire_prediction(self) -> float:
        x, y = self.current_pos
        
        if self.fire_predictions[x, y]:
            return -2.0  # Penalty for redundant prediction
        
        # Calculate weather-based risk score from normalized values [-1, 1]
        risk_score = 0.0
        weather_count = 0
        
        # All weather data is already normalized to [-1, 1]
        # So we can directly use them without manual normalization
        
        for weather_name in ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']:
            if weather_name in self.weather_patches:
                weather_value = self.weather_patches[weather_name][x, y]
                
                # Interpret based on fire risk impact:
                if weather_name in ['humidity', 'soil_moisture', 'rainfall', 'ndmi']:
                    # Low values = high risk (invert)
                    risk_contribution = (1 - weather_value) / 2  # Convert [-1,1] to [0,1]
                elif weather_name in ['wind_speed', 'soil_temp', 'dem']:
                    # High values = higher risk
                    risk_contribution = (weather_value + 1) / 2  # Convert [-1,1] to [0,1]
                else:
                    risk_contribution = 0.0
                
                risk_score += risk_contribution
                weather_count += 1
        
        # Average risk score
        if weather_count > 0:
            risk_score /= weather_count
        
        # Make prediction
        self.fire_predictions[x, y] = True
        self.prediction_counts[x, y] += 1
        self.total_predictions += 1
        
        # Check if prediction is correct
        is_fire_gt = self.fire_ground_truth[x, y]
        
        if is_fire_gt:
            # True Positive
            self.true_positives += 1
            base_reward = self.true_positive_reward
            # Increase reward if weather conditions also indicate high risk
            weather_bonus = risk_score * 50.0  # Up to 50 additional reward points
            reward = base_reward + weather_bonus
            if self.verbose:
                print(f"TP at {self.current_pos}: +{reward:.1f} (weather bonus: +{weather_bonus:.1f})")
        else:
            # False Positive
            self.false_positives += 1
            base_penalty = -self.false_positive_penalty
            # Reduce penalty if weather conditions suggested high risk
            weather_mitigation = risk_score * 100.0  # Up to 100 points of penalty reduction
            reward = base_penalty + weather_mitigation
            if self.verbose:
                print(f"FP at {self.current_pos}: {reward:.1f} (weather mitigation: +{weather_mitigation:.1f})")
        
        return reward

    def _calculate_f1_score(self) -> float:
        precision = self._get_current_precision()
        recall = self._get_current_recall()
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def reset(self):
        # Reset all tracking variables
        self.fire_predictions = np.zeros_like(self.thermal_data, dtype=bool)
        self.prediction_counts = np.zeros_like(self.thermal_data, dtype=int)
        self.visited_positions = set()
        
        self.true_positives = 0
        self.false_positives = 0
        self.total_predictions = 0
        self.steps_since_last_prediction = 0
        self.step_count = 0
        self.episode_reward = 0
        
        # Choose starting position strategically
        if np.random.random() < 0.3:
            # Sometimes start near fire pixels
            fire_positions = list(zip(*np.where(self.fire_ground_truth)))
            if fire_positions:
                idx = np.random.choice(len(fire_positions))
                self.current_pos = fire_positions[idx]
            else:
                self.current_pos = self.start_pos

        elif np.random.random() < 0.5:
            # Sometimes start at high temperature areas
            high_temp_positions = list(zip(*np.where(self.thermal_data >= self.medium_temp_threshold)))
            if high_temp_positions:
                idx = np.random.choice(len(high_temp_positions))
                self.current_pos = high_temp_positions[idx]
            else:
                self.current_pos = self.start_pos
        else:
            # Random start
            self.current_pos = (
                np.random.randint(0, self.height),
                np.random.randint(0, self.width)
            )
        
        self.visited_positions.add(self.current_pos)
        return self._get_observation(self.current_pos)
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current position: {self.current_pos}")
            print(f"Temperature: {self.thermal_data[self.current_pos[0], self.current_pos[1]]:.3f}")
            print(f"Is fire (GT): {self.fire_ground_truth[self.current_pos[0], self.current_pos[1]]}")
            print(f"Fire probability: {self._calculate_fire_probability(self.current_pos):.3f}")
            print(f"Step: {self.step_count}/{self.max_steps}")
            print(f"Predictions: {self.total_predictions} (TP: {self.true_positives}, FP: {self.false_positives})")
            print(f"Precision: {self._get_current_precision():.3f}, Recall: {self._get_current_recall():.3f}")
            print(f"Episode reward: {self.episode_reward:.2f}")

    def close(self):
        pass
        
def create_enhanced_crop_thermal_env(thermal_data, start_pos, weather_patches=None, 
                                   landcover_data=None, **kwargs):
    return EnhancedCropThermalEnv(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        **kwargs
    )