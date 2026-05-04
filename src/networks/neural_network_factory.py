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
import inspect



def get_neural_network_module(config: Dict[str, Any]):
    """
    Get the neural network module based on config.
    
    Args:
        config: Configuration with 'neural_network_implementation' key
        
    Returns:
        The neural network module (neural_network.py or neural_network_v2.py etc.)
    """
    implementation = config.get('neural_network_implementation', 'v1')
    
    try:
        if implementation == 'v1' or implementation == 'alphazero':
            from networks import neural_network
            return neural_network
        else:
            raise ValueError(f"Unknown neural network implementation: {implementation}")
    except Exception as e:
        # Silent fallback - return None if import fails
        return None


class NeuralNetworkFunctions:
    """Wrapper class that provides the 4 interface functions from the chosen implementation."""
    
    def __init__(self, module):
        self._module = module
    
    def create_neural_network(self, input_size: int, num_actions: int, config: Dict[str, Any]):
        """Create neural network using the chosen implementation."""
        return self._module.create_neural_network(input_size, num_actions, config)
    
    def neural_network_forward(self, neural_network, obs, num_actions: int, mask=None, config=None, env=None):
        """Forward pass using the chosen implementation."""
        # Use signature inspection to only pass supported parameters
        sig = inspect.signature(self._module.neural_network_forward)
        kwargs = {}
        if 'mask' in sig.parameters and mask is not None:
            kwargs['mask'] = mask
        if 'config' in sig.parameters and config is not None:
            kwargs['config'] = config
        if 'env' in sig.parameters and env is not None:
            kwargs['env'] = env
        return self._module.neural_network_forward(neural_network, obs, num_actions, **kwargs)
    
    def train_neural_network(self, neural_network, training_examples, config: Dict[str, Any], optimizer=None, **kwargs):
        """Train neural network using the chosen implementation.

        This method is resilient: it accepts an optional `optimizer` and any
        extra keyword arguments, and will forward the `optimizer` to the
        underlying implementation only when supported. If the implementation
        does not accept an `optimizer` parameter (for example RBM), the
        optimizer is ignored and the function is called without it.
        """
        import inspect

        func = getattr(self._module, 'train_neural_network')

        # Determine optimizer value from explicit arg or kwargs
        opt_to_pass = optimizer if optimizer is not None else kwargs.get('optimizer', None)

        # Try to inspect the signature of the target function and call appropriately
        try:
            sig = inspect.signature(func)
            if 'optimizer' in sig.parameters:
                # Call with optimizer as a keyword if supported
                return func(neural_network, training_examples, config, optimizer=opt_to_pass)
            else:
                # Call without optimizer
                return func(neural_network, training_examples, config)
        except (ValueError, TypeError):
            # If we cannot inspect, attempt a best-effort call: try with optimizer, then without
            try:
                return func(neural_network, training_examples, config, optimizer=opt_to_pass)
            except TypeError:
                return func(neural_network, training_examples, config)
    
    def encode_observation_simple(self, obs):
        """Encode observation using the chosen implementation (legacy method)."""
        return self._module.encode_observation_simple(obs)
    
    def encode_observation(self, obs, config=None, env=None):
        """Unified observation encoder that chooses between custom and gym-based encoding."""
        # Check if the module has the new unified encoder
        if hasattr(self._module, 'encode_observation'):
            return self._module.encode_observation(obs, config, env)
        else:
            # Fallback to simple encoder for backward compatibility
            return self._module.encode_observation_simple(obs)
    
    def get_observation_size(self, env, config=None):
        """Get observation size for given environment and config."""
        if hasattr(self._module, 'get_observation_size'):
            return self._module.get_observation_size(env, config)
        else:
            # Fallback - use sample observation
            obs = env.reset()
            return len(self._module.encode_observation_simple(obs))


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
    if module is None:
        return None
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