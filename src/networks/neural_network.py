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

# Try to import gym dependencies for traditional observation space
try:
    from grid2op.gym_compat import GymEnv, BoxGymObsSpace
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: Grid2Op gym compatibility not available. Only custom observation encoding supported.")


def encode_observation_simple(obs):
    """
    Encode observation for LINE SWITCHING + BUS SWITCHING actions
    
    Includes essential features for both disconnect/reconnect and bus topology decisions:
    1. Line loadings (which lines are overloaded)
    2. Line status (which lines are connected/disconnected)  
    3. Line cooldowns (which lines can be switched)
    4. Bus topology (current bus assignment for ALL substations)
    
    Parameters:
    -----------
    obs : grid2op.Observation
        Grid2Op observation object
        
    Returns:
    --------
    state_vector : np.ndarray
        Encoded state vector for neural network (dynamic size based on environment)
    """
    try:
        features = []
        
        # 1. Line loadings (rho) - n_line features
        # Critical for identifying overloaded lines that need disconnection
        if hasattr(obs, 'rho'):
            rho_values = [float(x) for x in obs.rho]
            features.extend(rho_values)
            n_lines = len(rho_values)
            
        # 2. Line status - n_line features
        # Which lines are currently connected (True/1) or disconnected (False/0)
        if hasattr(obs, 'line_status'):
            status_values = [float(x) for x in obs.line_status]
            features.extend(status_values)
            
        # 3. Line cooldowns - n_line features
        # Time before each line can be switched (0 = can switch now)
        if hasattr(obs, 'time_before_cooldown_line'):
            cooldown_values = [float(x) for x in obs.time_before_cooldown_line]
            features.extend(cooldown_values)
        
        # 4. Bus topology - topo_vect_size features
        # Current bus assignment (0=bus1, 1=bus2) for all elements in topology vector
        # NOTE: Actual size depends on environment (l2rpn_case14_sandbox vs rte_case14_realistic)
        if hasattr(obs, 'topo_vect'):
            # Include all elements from topology vector
            # Convert from 1-based (1=bus1, 2=bus2) to 0-based (0=bus1, 1=bus2)
            # Handle disconnected elements: -1 stays -1
            all_topo = []
            for topo_val in obs.topo_vect:
                if topo_val == -1:
                    all_topo.append(-1.0)  # Keep disconnected as -1
                else:
                    all_topo.append(float(topo_val - 1))  # Convert 1->0, 2->1
            features.extend(all_topo)
            topo_size = len(all_topo)
        
        # Total: 3*n_line + topo_vect_size features (depends on environment)
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        # Encoding failed - this should not happen in normal operation
        print(f"⚠️ Critical: Observation encoding failed: {e}")
        raise RuntimeError(f"Cannot encode observation: {e}. Check environment compatibility.")


def encode_observation_minimal(obs):
    """
    MINIMAL observation encoding - ONLY line loadings (rho values).
    
    Tests the hypothesis that topology details are noise and only line loads matter.
    This gives the absolute minimum information needed for grid management:
    - Which lines are overloaded (rho > 1.0)
    - How close other lines are to their limits
    
    Parameters:
    -----------
    obs : grid2op.Observation
        Grid2Op observation object
        
    Returns:
    --------
    state_vector : np.ndarray
        Minimal encoded state vector (only 20 features for l2rpn_case14_sandbox)
    """
    try:
        # ONLY line loadings (rho) - the most critical information
        if hasattr(obs, 'rho'):
            rho_values = [float(x) for x in obs.rho]
            return np.array(rho_values, dtype=np.float32)
        else:
            raise RuntimeError("No rho attribute in observation")
            
    except Exception as e:
        print(f"⚠️ Critical: Minimal observation encoding failed: {e}")
        raise RuntimeError(f"Cannot encode observation: {e}. Check environment compatibility.")


def encode_observation_gym(obs, env, obs_attr_to_keep=None, normalize=True):
    """
    Encode observation using traditional gym-based approach (like stable-baselines3).
    
    This provides a more standard RL observation space that includes various grid attributes
    in a flattened vector format, similar to what's used in traditional RL environments.
    
    Parameters:
    -----------
    obs : grid2op.Observation
        Grid2Op observation object
    env : grid2op.Environment
        Grid2Op environment (used for gym space setup)
    obs_attr_to_keep : list, optional
        List of observation attributes to include. If None, uses default set.
    normalize : bool
        Whether to normalize the observation values
        
    Returns:
    --------
    state_vector : np.ndarray
        Encoded state vector for neural network (dynamic size based on attributes)
    """
    if not GYM_AVAILABLE:
        raise ImportError("Grid2Op gym compatibility not available. Install with: pip install grid2op[gym]")
    
    try:
        # Default observation attributes if not specified
        if obs_attr_to_keep is None:
            obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                               "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                               "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                               "storage_power", "storage_charge"]
        
        # Create gym observation space if it doesn't exist
        if not hasattr(env, '_gym_obs_space') or env._gym_obs_space is None:
            env._gym_obs_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr_to_keep)
            if normalize:
                for attr_nm in obs_attr_to_keep:
                    try:
                        env._gym_obs_space.normalize_attr(attr_nm)
                    except:
                        # Skip normalization if it fails for this attribute
                        pass
        
        # Convert observation to gym format
        gym_obs = env._gym_obs_space.to_gym(obs)
        
        # Flatten if needed (gym observation might be multi-dimensional)
        if isinstance(gym_obs, (list, tuple)):
            features = []
            for item in gym_obs:
                if isinstance(item, np.ndarray):
                    features.extend(item.flatten())
                else:
                    features.append(float(item))
            gym_obs = np.array(features, dtype=np.float32)
        elif isinstance(gym_obs, np.ndarray):
            gym_obs = gym_obs.flatten().astype(np.float32)
        
        # Return flattened gym observation
        return gym_obs
        
    except Exception as e:
        print(f"⚠️ Critical: Gym observation encoding failed: {e}")
        raise RuntimeError(f"Cannot encode gym observation: {e}. Check gym compatibility.")


def encode_observation(obs, config=None, env=None):
    """
    Unified observation encoder that chooses between custom and gym-based encoding.
    
    Parameters:
    -----------
    obs : grid2op.Observation
        Grid2Op observation object
    config : dict, optional
        Configuration dictionary with observation space settings
    env : grid2op.Environment, optional
        Grid2Op environment (needed for gym encoding)
        
    Returns:
    --------
    state_vector : np.ndarray
        Encoded state vector for neural network
    """
    if config is None:
        config = {'obs_space_type': 'custom'}
    
    obs_space_type = config.get('obs_space_type', 'custom')
    
    if obs_space_type == 'gym':
        if env is None:
            raise ValueError("Environment is required for gym-based observation encoding")
        obs_attr_to_keep = config.get('gym_obs_attr_to_keep', None)
        normalize = config.get('gym_obs_normalize', True)
        return encode_observation_gym(obs, env, obs_attr_to_keep, normalize)
    elif obs_space_type == 'essential':
        if env is None:
            raise ValueError("Environment is required for essential observation encoding")
        obs_attr_to_keep = config.get('essential_obs_attr_to_keep', ["prod_p", "load_p", "rho", "timestep_overflow", "topo_vect"])
        normalize = config.get('gym_obs_normalize', True)
        return encode_observation_gym(obs, env, obs_attr_to_keep, normalize)
    elif obs_space_type == 'minimal':
        return encode_observation_minimal(obs)
    else:
        return encode_observation_simple(obs)


def get_observation_size(env, config=None):
    """
    Get the observation size for a given environment and configuration.
    
    Parameters:
    -----------
    env : grid2op.Environment
        Grid2Op environment
    config : dict, optional
        Configuration dictionary with observation space settings
        
    Returns:
    --------
    int : Size of the observation vector
    """
    if config is None:
        config = {'obs_space_type': 'custom'}
        
    obs_space_type = config.get('obs_space_type', 'custom')
    
    # Get a sample observation to determine size
    obs = env.reset()
    
    if obs_space_type == 'gym':
        sample_encoded = encode_observation_gym(obs, env, 
                                              config.get('gym_obs_attr_to_keep', None),
                                              config.get('gym_obs_normalize', True))
    elif obs_space_type == 'essential':
        sample_encoded = encode_observation_gym(obs, env, 
                                              config.get('essential_obs_attr_to_keep', ["prod_p", "load_p", "rho", "timestep_overflow", "topo_vect"]),
                                              config.get('gym_obs_normalize', True))
    elif obs_space_type == 'minimal':
        sample_encoded = encode_observation_minimal(obs)
    else:
        sample_encoded = encode_observation_simple(obs)
    
    return len(sample_encoded)


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
    
    # Debug output for network creation
    print(f"🔍 NETWORK CREATION: input_size={input_size}, num_actions={num_actions}, hidden_size={hidden_size}")
    
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


def load_neural_network(filepath):
    """
    Load neural network from file automatically detecting type.
    
    This is a factory function that loads any saved neural network
    without requiring you to create an empty object first.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model file
        
    Returns:
    --------
    network : NeuralNetworkInterface
        Loaded neural network ready for inference
    """
    import torch
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Detect model type with explicit backward compatibility
        model_type = checkpoint.get('model_type', None)
        
        if model_type == 'alphazero':
            # Modern checkpoint with explicit type
            pass
        elif model_type is None and 'model_state_dict' in checkpoint:
            # Legacy checkpoint without model_type - assume AlphaZero for backward compatibility
            print("⚠️ Loading legacy checkpoint without model_type, assuming AlphaZero")
            model_type = 'alphazero'
        
        if model_type == 'alphazero':
            # Load AlphaZero model
            if 'input_size' not in checkpoint or 'num_actions' not in checkpoint:
                raise ValueError(f"AlphaZero checkpoint missing required fields: {filepath}")
                
            network = AlphaZeroNetwork(
                input_size=checkpoint['input_size'],
                num_actions=checkpoint['num_actions'],
                hidden_size=checkpoint.get('hidden_size', 256)
            )
            network.load_state_dict(checkpoint['model_state_dict'])
            network.eval()
            
            print(f"✅ AlphaZero model loaded from: {filepath}")
            print(f"✅ Architecture: {checkpoint['input_size']} inputs -> {checkpoint['num_actions']} actions")
            return network
            
        elif model_type == 'rbm':
            # Load RBM model (implement when RBM is ready)
            raise NotImplementedError("RBM loading not yet implemented")
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load neural network from {filepath}: {e}")


def neural_network_forward(neural_network, obs, num_actions, mask=None, config=None, env=None):
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
    config : dict, optional
        Configuration dict for observation encoding
    env : grid2op.Environment, optional
        Grid2Op environment (required for some observation encodings)
        
    Returns:
    --------
    policy_probs : np.array
        Action probabilities
    value : float
        State value estimate
    """
    # Encode observation to state vector using config-aware encoding
    state_vector = encode_observation(obs, config, env)
    
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
    epochs = config.get('training_epochs', 5)  # Use config value or default to 1
    weight_decay = config.get('weight_decay', 1e-4)
    
    # Loss balancing weights - more balanced approach
    policy_weight = config.get('policy_weight', 1.0)  # Equal weight
    value_weight = config.get('value_weight', 1.0)    # Equal weight
    
    # Convert training examples to tensors
    states = []
    mcts_policies = []
    values = []
    has_value_targets = True  # Track if we're training value network
    
    for example in training_examples:
        states.append(example['state'])
        mcts_policies.append(example['mcts_policy'])
        # Check if value is None (heuristic mode - policy-only training)
        if example['value'] is None:
            values.append(0.0)  # Placeholder, will be ignored
            has_value_targets = False
        else:
            values.append(example['value'])
    
    # Override value_weight to 0 if no value targets (heuristic mode)
    if not has_value_targets:
        value_weight = 0.0
        print("  [Heuristic mode] Training policy only, value network frozen")
    
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
            # Skip batches with size 1 to avoid BatchNorm error
            if batch_states.size(0) == 1:
                continue
                
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
            
            # Value loss with Huber loss + clipping (PPO-style stability)
            # Clip predicted values to prevent extreme deviations from targets
            value_clip_range = config.get('value_clip_range', 10.0)  # Allow ±N deviation from target
            predicted_values_clipped = torch.clamp(
                predicted_values,
                batch_values - value_clip_range,
                batch_values + value_clip_range
            )
            
            # Huber loss: acts like MSE for small errors, linear for large errors
            # This prevents explosion from extreme MCTS values while still learning
            huber_delta = config.get('huber_delta', 5.0)
            value_loss_unclipped = F.huber_loss(predicted_values, batch_values, delta=huber_delta)
            value_loss_clipped = F.huber_loss(predicted_values_clipped, batch_values, delta=huber_delta)
            
            # Take the maximum loss (conservative clipping like PPO)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
            
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
    Residual block with skip connection and batch normalization.
    Architecture: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> Add residual -> ReLU
    """
    def __init__(self, hidden_size, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class AlphaZeroNetwork(nn.Module):
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
        
        # Input normalization (stabilize training)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Initial projection layer with batch norm
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Residual trunk (6 blocks for better capacity)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )
        
        # Policy head (action probabilities)  
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, num_actions)
        )
        
        # Value head (state value estimate) - larger capacity
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # No activation - linear output for unbounded cumulative reward prediction
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"AlphaZero network created: {input_size} -> projection({hidden_size}) -> 6x residual blocks -> policy({num_actions}) + value(1)")
        print(f"  Total parameters: ~{self._count_parameters():,}")
        print(f"  Architecture: Wider ({hidden_size}), deeper (6 blocks), BatchNorm, no dropout")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize policy head final layer to zeros for uniform priors at start
        # When all logits are 0, softmax gives uniform distribution
        for module in self.policy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize value head final layer to zeros for neutral value output (0.0)
        # This ensures consistent neutral value estimates before training
        if len(self.value_head) > 0:
            final_value_layer = self.value_head[-1]  # Last layer in Sequential
            if isinstance(final_value_layer, nn.Linear):
                nn.init.zeros_(final_value_layer.weight)
                if final_value_layer.bias is not None:
                    nn.init.zeros_(final_value_layer.bias)
    
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
    
    def save_model(self, filepath):
        """Save the AlphaZero model using PyTorch format."""
        import torch
        torch.save({
            'model_type': 'alphazero',  # Add type for auto-detection
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'num_actions': self.num_actions,
            'hidden_size': self.hidden_size
        }, filepath)
        print(f"✅ AlphaZero model saved to: {filepath}")