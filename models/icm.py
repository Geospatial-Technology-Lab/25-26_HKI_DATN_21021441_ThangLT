"""
Intrinsic Curiosity Module (ICM) for Exploration
Provides intrinsic rewards based on prediction error
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class IntrinsicCuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) as described in:
    "Curiosity-driven Exploration by Self-Supervised Prediction"
    https://arxiv.org/abs/1705.05363
    
    Components:
    1. Feature Encoder: Encodes states into feature space
    2. Forward Model: Predicts next state features from (state, action)
    3. Inverse Model: Predicts action from (state, next_state)
    
    Intrinsic Reward = prediction error of forward model
    """
    
    def __init__(self, state_size, action_size, feature_size=64, hidden_size=128):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.feature_size = feature_size
        
        # Feature Encoder: state -> features
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )
        
        # Inverse Model: (phi(s), phi(s')) -> action
        # Predicts what action was taken to go from s to s'
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Forward Model: (phi(s), action) -> phi(s')
        # Predicts next state features given current features and action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, state):
        """Encode state into feature space"""
        return self.encoder(state)
    
    def forward(self, state, next_state, action):
        """
        Forward pass
        
        Args:
            state: Current state [batch, state_size]
            next_state: Next state [batch, state_size]
            action: Action taken [batch] (integer indices)
            
        Returns:
            pred_action: Predicted action logits [batch, action_size]
            pred_next_features: Predicted next state features [batch, feature_size]
            actual_next_features: Actual next state features [batch, feature_size]
        """
        # Encode states
        phi_s = self.encode(state)
        phi_s_next = self.encode(next_state)
        
        # Inverse model: predict action from state transition
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        pred_action = self.inverse_model(inverse_input)
        
        # Forward model: predict next state features
        action_onehot = torch.zeros(action.size(0), self.action_size, device=state.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        pred_next_features = self.forward_model(forward_input)
        
        return pred_action, pred_next_features, phi_s_next
    
    def compute_intrinsic_reward(self, state, next_state, action):
        """
        Compute intrinsic reward based on forward model prediction error.
        Higher error = more curiosity = more reward
        
        Args:
            state: Current state [batch, state_size] or [state_size]
            next_state: Next state [batch, state_size] or [state_size]
            action: Action taken [batch] or scalar
            
        Returns:
            intrinsic_reward: Curiosity-based reward [batch] or scalar
        """
        # Handle single samples
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0) if isinstance(action, torch.Tensor) else torch.tensor([action])
            single_sample = True
        
        with torch.no_grad():
            _, pred_next_features, actual_next_features = self.forward(state, next_state, action)
            
            # Intrinsic reward = MSE between predicted and actual next features
            intrinsic_reward = 0.5 * ((pred_next_features - actual_next_features) ** 2).sum(dim=-1)
            
        if single_sample:
            intrinsic_reward = intrinsic_reward.squeeze(0)
        
        return intrinsic_reward


class ICMWrapper:
    """
    Wrapper to integrate ICM with RL agents.
    Handles training ICM and computing intrinsic rewards.
    """
    
    def __init__(self, state_size, action_size, device='cpu', 
                 lr=1e-3, beta=0.2, feature_size=64):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of actions
            device: 'cpu' or 'cuda'
            lr: Learning rate for ICM
            beta: Weight for forward model loss vs inverse model loss
            feature_size: Size of encoded feature space
        """
        self.device = device
        self.beta = beta
        self.intrinsic_reward_scale = 0.01  # Scale factor for intrinsic rewards
        
        # ICM module
        self.icm = IntrinsicCuriosityModule(
            state_size=state_size,
            action_size=action_size,
            feature_size=feature_size
        ).to(device)
        
        self.optimizer = optim.Adam(self.icm.parameters(), lr=lr)
        
        # Experience buffer for ICM training
        self.states = []
        self.next_states = []
        self.actions = []
        
        # Statistics
        self.avg_intrinsic_reward = 0.0
        self.update_count = 0
    
    def store_transition(self, state, action, next_state):
        """Store transition for ICM training"""
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
    
    def compute_reward(self, state, action, next_state):
        """
        Compute intrinsic reward for a single transition.
        
        Returns:
            intrinsic_reward: float
        """
        state_t = torch.FloatTensor(state).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        
        intrinsic_reward = self.icm.compute_intrinsic_reward(
            state_t, next_state_t, action_t
        )
        
        return float(intrinsic_reward.item()) * self.intrinsic_reward_scale
    
    def update(self, batch_size=64):
        """
        Update ICM using stored transitions.
        
        Returns:
            dict: Training statistics
        """
        if len(self.states) < batch_size:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        
        # Sample batch
        indices = np.random.choice(len(states), min(batch_size, len(states)), replace=False)
        batch_states = states[indices]
        batch_next_states = next_states[indices]
        batch_actions = actions[indices]
        
        # Forward pass
        pred_action, pred_next_features, actual_next_features = self.icm(
            batch_states, batch_next_states, batch_actions
        )
        
        # Inverse model loss (cross-entropy)
        inverse_loss = nn.CrossEntropyLoss()(pred_action, batch_actions)
        
        # Forward model loss (MSE)
        forward_loss = 0.5 * ((pred_next_features - actual_next_features.detach()) ** 2).sum(dim=-1).mean()
        
        # Total loss
        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update statistics
        self.update_count += 1
        
        return {
            'icm_loss': loss.item(),
            'inverse_loss': inverse_loss.item(),
            'forward_loss': forward_loss.item()
        }
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.next_states.clear()
        self.actions.clear()


class CNNIntrinsicCuriosityModule(nn.Module):
    """
    ICM variant for CNN-based observations.
    Encodes spatial observations before computing curiosity.
    """
    
    def __init__(self, num_channels=8, patch_size=11, action_size=6, feature_size=128):
        super().__init__()
        
        self.action_size = action_size
        self.feature_size = feature_size
        
        # CNN Encoder for spatial observations
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, feature_size)
        )
        
        # Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size)
        )
    
    def encode(self, state):
        return self.encoder(state)
    
    def forward(self, state, next_state, action):
        phi_s = self.encode(state)
        phi_s_next = self.encode(next_state)
        
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        pred_action = self.inverse_model(inverse_input)
        
        # Ensure action is 1D tensor for scatter
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        batch_size = phi_s.size(0)
        action_onehot = torch.zeros(batch_size, self.action_size, device=state.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1)
        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        pred_next_features = self.forward_model(forward_input)
        
        return pred_action, pred_next_features, phi_s_next
    
    def compute_intrinsic_reward(self, state, next_state, action):
        single_sample = False
        if state.dim() == 3:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            single_sample = True
        
        # Ensure action is tensor on correct device
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], device=state.device)
        else:
            action = action.to(state.device)
        
        # Ensure action has correct shape
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        with torch.no_grad():
            _, pred_next_features, actual_next_features = self.forward(state, next_state, action)
            intrinsic_reward = 0.5 * ((pred_next_features - actual_next_features) ** 2).sum(dim=-1)
        
        if single_sample:
            intrinsic_reward = intrinsic_reward.squeeze(0)
        
        return intrinsic_reward
