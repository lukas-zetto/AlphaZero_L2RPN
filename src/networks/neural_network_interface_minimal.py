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


class NeuralNetworkInterface(ABC):
    """Interface defining required methods for neural network instances."""
    
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


# Function signatures (implementations in neural_network.py)
def create_neural_network(input_size: int, num_actions: int, config: Dict[str, Any]) -> NeuralNetworkInterface:
    """Create a neural network instance."""
    raise NotImplementedError("Import from neural_network.py")


def neural_network_forward(neural_network: NeuralNetworkInterface, obs, num_actions: int) -> Tuple[np.ndarray, float]:
    """Forward pass through neural network for action selection."""
    raise NotImplementedError("Import from neural_network.py")


def train_neural_network(neural_network: NeuralNetworkInterface, training_examples: List[Dict], config: Dict[str, Any]) -> Dict[str, float]:
    """Train the neural network on a batch of examples."""
    raise NotImplementedError("Import from neural_network.py")


def encode_observation_simple(obs) -> np.ndarray:
    """Encode Grid2Op observation into neural network input (legacy method)."""
    raise NotImplementedError("Import from neural_network.py")


def encode_observation(obs, config=None, env=None) -> np.ndarray:
    """Unified observation encoder that chooses between custom and gym-based encoding."""
    raise NotImplementedError("Import from neural_network.py")


def get_observation_size(env, config=None) -> int:
    """Get the observation vector size for given environment and configuration."""
    raise NotImplementedError("Import from neural_network.py")