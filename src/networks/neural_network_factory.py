#!/usr/bin/env python3
"""
Neural Network Factory

Factory pattern to dynamically choose between different neural network implementations
based on configuration. Keeps the interface clean and enables easy switching.

Usage:
    # In any file that needs neural network functions
    from neural_network_factory import get_neural_network_functions
    
    nn_funcs = get_neural_network_functions(config)
    network = nn_funcs.create_neural_network(input_size, num_actions, config)
    probs, value = nn_funcs.neural_network_forward(network, obs, num_actions)
"""

from typing import Dict, Any


def get_neural_network_module(config: Dict[str, Any]):
    """
    Get the neural network module based on config.
    
    Args:
        config: Configuration with 'neural_network_implementation' key
        
    Returns:
        The neural network module (neural_network.py or neural_network_v2.py etc.)
    """
    implementation = config.get('neural_network_implementation', 'v1')
    
    if implementation == 'v1':
        from src.networks import neural_network
        return neural_network
    elif implementation == 'v2':
        from src.networks import neural_network_v2  # Alternative implementation (to be created)
        return neural_network_v2
    else:
        raise ValueError(f"Unknown neural network implementation: {implementation}")


class NeuralNetworkFunctions:
    """Wrapper class that provides the 4 interface functions from the chosen implementation."""
    
    def __init__(self, module):
        self._module = module
    
    def create_neural_network(self, input_size: int, num_actions: int, config: Dict[str, Any]):
        """Create neural network using the chosen implementation."""
        return self._module.create_neural_network(input_size, num_actions, config)
    
    def neural_network_forward(self, neural_network, obs, num_actions: int, mask=None):
        """Forward pass using the chosen implementation."""
        return self._module.neural_network_forward(neural_network, obs, num_actions, mask=mask)
    
    def train_neural_network(self, neural_network, training_examples, config: Dict[str, Any]):
        """Train neural network using the chosen implementation."""
        return self._module.train_neural_network(neural_network, training_examples, config)
    
    def encode_observation_simple(self, obs):
        """Encode observation using the chosen implementation."""
        return self._module.encode_observation_simple(obs)


def get_neural_network_functions(config: Dict[str, Any]) -> NeuralNetworkFunctions:
    """
    Get neural network functions based on config.
    
    Args:
        config: Configuration with 'neural_network_implementation' key
        
    Returns:
        NeuralNetworkFunctions object with the 4 interface functions
        
    Example:
        nn_funcs = get_neural_network_functions(config)
        network = nn_funcs.create_neural_network(60, 21, config)
    """
    module = get_neural_network_module(config)
    return NeuralNetworkFunctions(module)


# Convenience function for direct module access if needed
def get_implementation_name(config: Dict[str, Any]) -> str:
    """Get the name of the current neural network implementation."""
    return config.get('neural_network_implementation', 'v1')


# Test function
def test_factory():
    """Test that the factory can load the default implementation."""
    config = {'neural_network_implementation': 'v1'}
    try:
        nn_funcs = get_neural_network_functions(config)
        print(f"✅ Factory successfully loaded implementation: {get_implementation_name(config)}")
        return True
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False


if __name__ == "__main__":
    test_factory()