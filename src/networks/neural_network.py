#!/usr/bin/env python3
"""
Neural Network implementation for Alpha Zero agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .neural_network_interface_minimal import NeuralNetworkInterface


def encode_observation_simple(obs):
    """
    Encode observation for LINE SWITCHING + BUS SWITCHING actions
    
    Includes essential features for both disconnect/reconnect and bus topology decisions:
    1. Line loadings (which lines are overloaded)
    2. Line status (which lines are connected/disconnected)  
    3. Line cooldowns (which lines can be switched)
    4. Bus topology (current bus assignment for switchable elements)
    
    Parameters:
    -----------
    obs : grid2op.Observation
        Grid2Op observation object
        
    Returns:
    --------
    state_vector : np.ndarray
        Encoded state vector for neural network (83 features)
    """
    try:
        features = []
        
        # 1. Line loadings (rho) - 20 features [0:20]
        # Critical for identifying overloaded lines that need disconnection
        if hasattr(obs, 'rho'):
            rho_values = [float(x) for x in obs.rho]
            features.extend(rho_values)
            
        # 2. Line status - 20 features [20:40]
        # Which lines are currently connected (True/1) or disconnected (False/0)
        if hasattr(obs, 'line_status'):
            status_values = [float(x) for x in obs.line_status]
            features.extend(status_values)
            
        # 3. Line cooldowns - 20 features [40:60]
        # Time before each line can be switched (0 = can switch now)
        if hasattr(obs, 'time_before_cooldown_line'):
            cooldown_values = [float(x) for x in obs.time_before_cooldown_line]
            features.extend(cooldown_values)
        
        # 4. Bus topology for switchable substations - 23 features [60:83]
        # Current bus assignment (0=bus1, 1=bus2) for elements in substations 3, 4, 5, 8
        if hasattr(obs, 'topo_vect'):
            # Substation 3: indices 13:19 (6 elements)
            sub3_topo = obs.topo_vect[13:19] - 1  # Convert from 1/2 to 0/1
            features.extend([float(x) for x in sub3_topo])
            
            # Substation 4: indices 19:24 (5 elements)
            sub4_topo = obs.topo_vect[19:24] - 1  # Convert from 1/2 to 0/1
            features.extend([float(x) for x in sub4_topo])
            
            # Substation 5: indices 24:31 (7 elements)  
            sub5_topo = obs.topo_vect[24:31] - 1  # Convert from 1/2 to 0/1
            features.extend([float(x) for x in sub5_topo])
            
            # Substation 8: indices 36:41 (5 elements)
            sub8_topo = obs.topo_vect[36:41] - 1  # Convert from 1/2 to 0/1
            features.extend([float(x) for x in sub8_topo])
        
        # Total: 20 + 20 + 20 + 23 = 83 features
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        # Fallback: return minimal state if encoding fails
        print(f"⚠️ Observation encoding failed: {e}")
        # Return rho + zeros for topology as fallback
        try:
            rho_features = obs.rho.tolist()
            fallback_features = rho_features + [0.0] * (83 - len(rho_features))
            return np.array(fallback_features[:83], dtype=np.float32)
        except:
            return np.zeros(83, dtype=np.float32)  # Full 83-feature fallback


def create_neural_network(input_size, num_actions, config):
    """
    Create Alpha Zero neural network with shared trunk + policy/value heads.
    
    Parameters:
    -----------
    input_size : int
        Size of input state vector (number of transmission lines)
    num_actions : int
        Maximum number of actions (lines + do nothing)
    config : dict
        Network configuration parameters
        
    Returns:
    --------
    network : AlphaZeroNetwork
        Shared neural network with policy and value heads
    """
    hidden_size = config.get('hidden_size', 256)
    
    # Create shared network with policy and value heads
    network = AlphaZeroNetwork(
        input_size=input_size,
        num_actions=num_actions, 
        hidden_size=hidden_size
    )
    
    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    
    print(f"Created AlphaZero network on {device}")
    return network


def neural_network_forward(neural_network, obs, num_actions, mask=None):
    """
    Forward pass through shared neural network with policy and value heads.
    
    Parameters:
    -----------
    neural_network : AlphaZeroNetwork or None
        Shared neural network with policy and value heads
    obs : grid2op.Observation
        Grid2Op observation
    num_actions : int
        Number of valid actions
        
    Returns:
    --------
    policy_probs : np.array
        Action probabilities
    value : float
        State value estimate
    """
    # Encode observation to state vector (simple implementation)
    state_vector = encode_observation_simple(obs)
    
    if neural_network is None:
        # Placeholder: return uniform policy and zero value
        uniform_policy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([1.0])
        return uniform_policy, 0.0
    
    # Convert to torch tensor and move to correct device
    device = next(neural_network.parameters()).device
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)  # Add batch dimension
    
    # Set to eval mode for inference (important for batch norm / dropout)
    neural_network.eval()
    
    # Forward pass through shared network -> policy and value heads
    with torch.no_grad():
        policy_logits, value = neural_network.forward(state_tensor)

    # Truncate to valid actions
    policy_logits = policy_logits.squeeze(0)[:num_actions]

    # Apply optional mask (masked actions get large negative logit)
    if mask is not None:
        # Expect mask as 1D array-like of length num_actions, True=keep
        mask_arr = torch.as_tensor(mask, dtype=torch.bool, device=policy_logits.device)
        if mask_arr.shape[0] != num_actions:
            raise ValueError(f"Mask length {mask_arr.shape[0]} != num_actions {num_actions}")
        # Fill invalid logits
        invalid = ~mask_arr
        if invalid.any():
            policy_logits = policy_logits.clone()
            policy_logits[invalid] = -1e9

    policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()
    
    # Extract value (remove batch dimension)
    value = value.squeeze(0).cpu().item()
    
    return policy_probs, value


def train_neural_network(neural_network, training_examples, config, optimizer=None):
    """
    Train the neural network on collected MCTS examples.
    
    Parameters:
    -----------
    neural_network : torch.nn.Module
        Neural network to train
    training_examples : list
        List of {'state': array, 'mcts_policy': array, 'value': float} from self-play
    config : dict
        Training configuration
    optimizer : torch.optim.Optimizer, optional
        If provided, use this optimizer (for persistent LR decay). Otherwise create new one.
        
    Returns:
    --------
    loss_info : dict
        Training loss information
    """
    if neural_network is None or len(training_examples) == 0:
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
    
    device = next(neural_network.parameters()).device
    neural_network.train()
    
    # Training hyperparameters
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('training_epochs', 10)
    weight_decay = config.get('weight_decay', 1e-4)
    
    # Loss balancing weights - more balanced approach
    policy_weight = config.get('policy_weight', 1.0)  # Equal weight
    value_weight = config.get('value_weight', 1.0)    # Equal weight
    
    # Convert training examples to tensors
    states = []
    mcts_policies = []
    values = []
    
    for example in training_examples:
        states.append(example['state'])
        mcts_policies.append(example['mcts_policy'])
        values.append(example['value'])
    
    # Ensure proper dtypes - states and policies should already be np.float32 arrays
    # All policies are exactly 21 elements (fixed-size action space)
    states = [np.array(s, dtype=np.float32).flatten() for s in states]
    mcts_policies = [np.array(p, dtype=np.float32) for p in mcts_policies]
    values = [float(v) for v in values]
    
    # Convert to tensors with explicit float32 dtype
    states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32)).to(device)
    policies_tensor = torch.FloatTensor(np.array(mcts_policies, dtype=np.float32)).to(device)
    values_tensor = torch.FloatTensor(np.array(values, dtype=np.float32)).to(device).unsqueeze(1)
    
    # Create DataLoader for batching
    dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer - use provided one or create new
    if optimizer is None:
        optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        
        for batch_states, batch_policies, batch_values in dataloader:
            # Forward pass
            policy_logits, predicted_values = neural_network(batch_states)
            
            # Policy loss (stable cross-entropy between MCTS policy and network policy)
            # Add small epsilon to prevent log(0) and numerical instability
            epsilon = 1e-8
            
            # Normalize MCTS policies to ensure they sum to 1
            batch_policies_safe = batch_policies + epsilon
            batch_policies_safe = batch_policies_safe / batch_policies_safe.sum(dim=1, keepdim=True)
            
            # Stable log softmax for network predictions
            log_probs = F.log_softmax(policy_logits, dim=1)
            
            # Cross-entropy: -sum(target * log(predicted))
            policy_loss = -(batch_policies_safe * log_probs).sum(dim=1).mean()
            
            # Value loss (MSE between actual returns and predicted values)
            value_loss = F.mse_loss(predicted_values, batch_values)
            
            # Combined loss with weights from config (no additional scaling)
            total_loss = policy_weight * policy_loss + value_weight * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses (both weighted and unweighted for reporting)
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            total_batches += 1
        
        total_policy_loss += epoch_policy_loss
        total_value_loss += epoch_value_loss
    
    # Average losses (report actual weighted losses)
    avg_policy_loss = (total_policy_loss / total_batches) * policy_weight
    avg_value_loss = (total_value_loss / total_batches) * value_weight
    avg_total_loss = avg_policy_loss + avg_value_loss
    
    print(f"Training complete: {len(training_examples)} examples, {epochs} epochs")
    print(f"  Policy Loss: {avg_policy_loss:.4f}")
    print(f"  Value Loss: {avg_value_loss:.4f}")
    print(f"  Total Loss: {avg_total_loss:.4f}")
    import sys
    sys.stdout.flush()  # Force output to appear immediately

    return {
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss,
        'total_loss': avg_total_loss
    }


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper networks.
    Architecture: Linear -> ReLU -> Dropout -> Linear -> Add residual -> ReLU
    """
    def __init__(self, hidden_size, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class AlphaZeroNetwork(nn.Module, NeuralNetworkInterface):
    """
    Alpha Zero neural network: shared trunk + policy head + value head
    (Single network with two outputs, as used in the original paper)
    
    Enhanced with:
    - Larger hidden size (1024) for more capacity
    - Residual connections for deeper networks (6 residual blocks)
    - Reduced dropout (0.1) for more effective capacity
    
    Implements NeuralNetworkInterface for proper type checking and contract enforcement.
    """
    
    def __init__(self, input_size, num_actions, hidden_size=256):
        super(AlphaZeroNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        # Initial projection layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual trunk (3 blocks instead of 6 - smaller network for limited data)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size, dropout=0.1),
            ResidualBlock(hidden_size, dropout=0.1),
            ResidualBlock(hidden_size, dropout=0.1)
        )
        
        # Policy head (action probabilities)  
        self.policy_head = nn.Linear(hidden_size, num_actions)
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
            # No activation - linear output for unbounded cumulative reward prediction
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"AlphaZero network created: {input_size} -> projection({hidden_size}) -> 3x residual blocks -> policy({num_actions}) + value(1)")
        print(f"  Total parameters: ~{self._count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Forward pass through shared network with dual heads.
        
        Implements NeuralNetworkInterface.forward()
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, input_size]
            Input state vectors (line loads)
        
        Returns:
        --------
        policy_logits : torch.Tensor [batch_size, num_actions]
            Raw policy outputs (before softmax)
        value : torch.Tensor [batch_size, 1] 
            State value estimates (unbounded - cumulative future rewards)
        """
        # Initial projection
        x = self.input_layer(x)
        
        # Deep residual trunk
        shared_features = self.residual_blocks(x)
        
        # Policy head - raw logits for actions
        policy_logits = self.policy_head(shared_features)
        
        # Value head - estimated state value
        value = self.value_head(shared_features)
        
        return policy_logits, value