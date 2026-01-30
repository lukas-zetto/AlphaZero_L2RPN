#!/usr/bin/env python3
"""
Neural Network Interface - Minimal Version

True interface defining only the functions actually called from outside neural_network.py:

1. create_neural_network() - Factory to create neural network instances
2. neural_network_forward() - Forward pass for action probabilities + value
3. train_neural_network() - Train network on batch of examples  
4. encode_observation_simple() - Convert Grid2Op obs to state vector (legacy)
5. encode_observation() - Unified observation encoder (custom/gym)
6. get_observation_size() - Get observation vector size

Plus required PyTorch methods on neural network instances: .forward(), .train(), .eval()
Plus required PyTorch methods called directly: .state_dict(), .load_state_dict()
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np
import queno


class RBM(ABC):
    """Interface defining required methods for neural network instances."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = queno.load_npz_model(config['rbm_model_file'])
        self.raw_model = np.load(config['rbm_model_file'])
        self.bins = self.raw_model['bins']

    @abstractmethod
    def forward(self, x):
        """Forward pass through network."""
        pass
    
    @abstractmethod
    def train(self):
        """Set network to training mode."""
        pass
    
    @abstractmethod
    def eval(self):
        """Set network to evaluation mode."""
        pass
    
    @abstractmethod
    def state_dict(self):
        """Get model weights."""
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load model weights."""
        pass


# Function signatures 
def create_neural_network(input_size: int, num_actions: int, config: Dict[str, Any]) -> RBM:
    network = RBM(config) 
    return 


def neural_network_forward(rbm, obs, num_actions: int) -> Tuple[np.ndarray, float]:
    """Forward pass through neural network for action selection."""
    enc_observation = encode_observation_simple(obs)
    dsc_observation = queno.disctretize(enc_observation, rbm.bins)
    H = 65
    Z = np.array([queno.missing]*(N*H)).reshape(N,H)
    X = np.hstack((dsc_observation, Z)).astype(np.uint16)
    
    S = queno.nat.sample(backend=rbm.model, obs=X, num_samples=1000)
    pol = S[:,rbm.get_observation_size()]
    P, C = np.unique(pol, return_counts=True)
    n_pols = queno.nat.nstates(backend=rbm.model)[rbm.get_observation_size()]
    pols_full = np.zeros(n_pols)
    for i in range(len(P)):
        pols_full[P[i]] = C[i]
    return pols_full / np.sum(pols_full), 0.0  # Value is dummy


def train_neural_network(neural_network: NeuralNetworkInterface, training_examples: List[Dict], config: Dict[str, Any]) -> Dict[str, float]:
    """Train the neural network on a batch of examples."""
    raise NotImplementedError("Import from neural_network.py")


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



def encode_observation(obs, config=None, env=None) -> np.ndarray:
    return encode_observation_simple(obs)
    


def get_observation_size(env, config=None) -> int:
    """Get the observation vector size for given environment and configuration."""
    return len(encode_observation_simple(env.reset()))