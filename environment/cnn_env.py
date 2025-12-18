"""
CNN-based Environment for Wildfire Detection
Uses 2D spatial observations instead of 1D features
"""
import gym
import torch
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class CNNCropThermalEnv(gym.Env):
    """
    Enhanced environment with CNN-compatible observations.
    
    Observation space: [C, H, W] tensor where:
    - C = 8 channels (thermal + 7 weather features)
    - H, W = patch_size (default 11x11)
    
    This allows the agent to learn spatial patterns like:
    - Fire edges
    - Fire spread direction
    - Local hotspots
    """
    
    def __init__(self,
                 thermal_data: np.ndarray,
                 start_pos: Tuple[int, int],
                 weather_patches: Optional[Dict[str, np.ndarray]] = None,
                 landcover_data: Optional[np.ndarray] = None,
                 max_steps: int = 1000,
                 obey_prob: float = 0.9,
                 high_temp_threshold: float = 0.95,
                 medium_temp_threshold: float = 0.85,
                 patch_size: int = 11,  # Size of observation patch
                 verbose: bool = False):
        super(CNNCropThermalEnv, self).__init__()
        
        self.thermal_data = thermal_data
        self.start_pos = start_pos
        self.weather_patches = weather_patches or {}
        self.max_steps = max_steps
        self.obey_prob = obey_prob
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        
        self.landcover_data = landcover_data if landcover_data is not None else np.ones_like(thermal_data)
        
        # REBALANCED reward parameters for better learning
        # Key: Lower penalties to encourage exploration and prediction
        self.true_positive_reward = 10.0       # Reward for correct fire prediction
        self.false_positive_penalty = 5.0      # Much lower penalty (was 50)
        self.true_negative_reward = 1.0
        self.false_negative_penalty = 0.0      # No penalty, just no reward
        self.exploration_reward = 0.5
        self.movement_cost = 0.05              # Lower movement cost
        self.proximity_reward_scale = 2.0
        self.discovery_bonus = 1.0
        
        # Lower thresholds for fire detection (more fire pixels to learn from)
        self.high_temp_threshold = 0.7         # Was 0.95 - too strict!
        self.medium_temp_threshold = 0.5       # Was 0.85
        self.height, self.width = thermal_data.shape
        
        # Action space: 6 actions (U, D, L, R, Stay, Predict)
        self.action_space = spaces.Discrete(6)
        self.action_to_direction = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
            4: (0, 0),   # Stay
            5: (0, 0)    # Predict Fire
        }
        self.action_names = ['U', 'D', 'L', 'R', 'S', 'P']
        
        # CNN Observation space: [channels, height, width]
        # Channels: thermal, humidity, wind_speed, soil_temp, soil_moisture, rainfall, ndmi, dem
        self.num_channels = 8
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_channels, patch_size, patch_size),
            dtype=np.float32
        )
        
        self.verbose = verbose
        
        # Pre-compute fire ground truth
        self.fire_ground_truth = self.thermal_data >= self.high_temp_threshold
        self.total_fire_pixels = int(np.sum(self.fire_ground_truth))
        self.total_non_fire_pixels = int(np.sum(~self.fire_ground_truth))
        
        # Tracking variables
        self.fire_predictions = np.zeros_like(self.thermal_data, dtype=bool)
        self.prediction_counts = np.zeros_like(self.thermal_data, dtype=int)
        self.visited_positions = set()
        
        # Statistics
        self.true_positives = 0
        self.false_positives = 0
        self.total_predictions = 0
        self.steps_since_last_prediction = 0
        
        # Initialize
        self.current_pos = None
        self.step_count = 0
        self.episode_reward = 0
        
        # Pre-compute fire distance map for shaped rewards
        if HAS_SCIPY:
            self._fire_distance_map = distance_transform_edt(~self.fire_ground_truth)
            max_dist = np.max(self._fire_distance_map)
            if max_dist > 0:
                self._fire_distance_map = self._fire_distance_map / max_dist
        
        # Pad data for edge handling
        self._pad_data()
    
    def _pad_data(self):
        """Pad all data arrays to handle edge positions"""
        pad_width = self.half_patch
        
        # Pad thermal data with edge values
        self.thermal_padded = np.pad(
            self.thermal_data, 
            pad_width, 
            mode='edge'
        )
        
        # Pad weather data
        self.weather_padded = {}
        weather_names = ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']
        for name in weather_names:
            if name in self.weather_patches:
                self.weather_padded[name] = np.pad(
                    self.weather_patches[name],
                    pad_width,
                    mode='edge'
                )
            else:
                # Create zeros if weather data not available
                self.weather_padded[name] = np.zeros_like(self.thermal_padded)
        
        # Pad landcover
        self.landcover_padded = np.pad(
            self.landcover_data,
            pad_width,
            mode='edge'
        )
    
    def _get_observation(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Get CNN-compatible observation: [C, H, W] tensor
        """
        x, y = pos
        # Adjust for padding
        px = x + self.half_patch
        py = y + self.half_patch
        
        # Extract patches
        observation = np.zeros((self.num_channels, self.patch_size, self.patch_size), dtype=np.float32)
        
        # Channel 0: Thermal data (normalized)
        observation[0] = self.thermal_padded[
            px - self.half_patch : px + self.half_patch + 1,
            py - self.half_patch : py + self.half_patch + 1
        ]
        
        # Channels 1-7: Weather features
        weather_names = ['humidity', 'wind_speed', 'soil_temp', 'soil_moisture', 'rainfall', 'ndmi', 'dem']
        for i, name in enumerate(weather_names):
            observation[i + 1] = self.weather_padded[name][
                px - self.half_patch : px + self.half_patch + 1,
                py - self.half_patch : py + self.half_patch + 1
            ]
        
        return observation
    
    def reset(self):
        """Reset environment"""
        self.fire_predictions = np.zeros_like(self.thermal_data, dtype=bool)
        self.prediction_counts = np.zeros_like(self.thermal_data, dtype=int)
        self.visited_positions = set()
        
        self.true_positives = 0
        self.false_positives = 0
        self.total_predictions = 0
        self.steps_since_last_prediction = 0
        
        self.step_count = 0
        self.episode_reward = 0
        
        self.current_pos = self.start_pos
        self.visited_positions.add(self.current_pos)
        
        return self._get_observation(self.current_pos)
    
    def step(self, action: int):
        """Execute action and return observation, reward, done, info"""
        if self.current_pos is None:
            raise ValueError("Environment not reset!")
        
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
            reward = self._handle_movement(action)
            new_pos = self._get_new_position(action)
            self.steps_since_last_prediction += 1
            
            if self.steps_since_last_prediction > 20:
                reward -= 0.5
        
        self.current_pos = new_pos
        self.step_count += 1
        self.episode_reward += reward
        
        done = False
        if self.step_count >= self.max_steps:
            done = True
            f1_score = self._calculate_f1_score()
            reward += f1_score * 5.0
        
        info.update({
            'step_count': self.step_count,
            'position': self.current_pos,
            'temperature': self.thermal_data[new_pos[0], new_pos[1]],
            'is_fire_gt': self.fire_ground_truth[new_pos[0], new_pos[1]],
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
        
        for action in range(5):
            if action == 4:
                valid.append(action)
            else:
                dx, dy = self.action_to_direction[action]
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.height and 0 <= new_y < self.width:
                    valid.append(action)
        
        valid.append(5)
        return valid
    
    def _get_new_position(self, action: int) -> Tuple[int, int]:
        if action == 4 or action == 5:
            return self.current_pos
        
        if np.random.random() < self.obey_prob:
            actual_action = action
        else:
            valid_moves = [a for a in range(5) if a in self._get_valid_actions(self.current_pos)]
            actual_action = np.random.choice(valid_moves)
        
        dx, dy = self.action_to_direction[actual_action]
        new_x = np.clip(self.current_pos[0] + dx, 0, self.height - 1)
        new_y = np.clip(self.current_pos[1] + dy, 0, self.width - 1)
        
        return (new_x, new_y)
    
    def _handle_movement(self, action: int) -> float:
        new_pos = self._get_new_position(action)
        reward = -self.movement_cost
        
        if new_pos not in self.visited_positions:
            reward += self.exploration_reward
            self.visited_positions.add(new_pos)
            
            new_temp = self.thermal_data[new_pos[0], new_pos[1]]
            if new_temp >= self.high_temp_threshold:
                reward += self.discovery_bonus * 2.0
            elif new_temp >= self.medium_temp_threshold:
                reward += self.discovery_bonus
        
        # Temperature gradient reward
        current_temp = self.thermal_data[self.current_pos[0], self.current_pos[1]]
        new_temp = self.thermal_data[new_pos[0], new_pos[1]]
        if new_temp > current_temp and new_temp > self.medium_temp_threshold:
            reward += 0.5
        
        # Proximity reward
        if hasattr(self, '_fire_distance_map'):
            current_dist = self._fire_distance_map[self.current_pos[0], self.current_pos[1]]
            new_dist = self._fire_distance_map[new_pos[0], new_pos[1]]
            if new_dist < current_dist:
                proximity_bonus = (current_dist - new_dist) * self.proximity_reward_scale
                reward += proximity_bonus
        
        return reward
    
    def _handle_fire_prediction(self) -> float:
        x, y = self.current_pos
        
        # Small penalty for repeated predictions
        if self.fire_predictions[x, y]:
            return -0.5
        
        self.fire_predictions[x, y] = True
        self.prediction_counts[x, y] += 1
        self.total_predictions += 1
        
        is_fire_gt = self.fire_ground_truth[x, y]
        current_temp = self.thermal_data[x, y]
        
        if is_fire_gt:
            # True positive - big reward!
            self.true_positives += 1
            reward = self.true_positive_reward
        else:
            self.false_positives += 1
            # Graduated penalty based on temperature
            # Lower penalty if temp is high (reasonable guess)
            if current_temp >= self.medium_temp_threshold:
                reward = -self.false_positive_penalty * 0.5  # Half penalty
            else:
                reward = -self.false_positive_penalty
        
        return reward
    
    def _get_current_precision(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.true_positives / self.total_predictions
    
    def _get_current_recall(self) -> float:
        if self.total_fire_pixels == 0:
            return 0.0
        return self.true_positives / self.total_fire_pixels
    
    def _calculate_f1_score(self) -> float:
        precision = self._get_current_precision()
        recall = self._get_current_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass


def create_cnn_crop_thermal_env(thermal_data, start_pos, weather_patches=None, 
                                 landcover_data=None, max_steps=1000, 
                                 patch_size=11, verbose=False):
    """Factory function to create CNN environment"""
    return CNNCropThermalEnv(
        thermal_data=thermal_data,
        start_pos=start_pos,
        weather_patches=weather_patches,
        landcover_data=landcover_data,
        max_steps=max_steps,
        patch_size=patch_size,
        verbose=verbose
    )
