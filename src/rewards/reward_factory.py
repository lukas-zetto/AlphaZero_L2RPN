"""
Reward Factory for dynamically loading different reward classes from files
Similar to neural_network_factory.py pattern
"""

import importlib


# Reward mapping configuration
REWARD_MAPPINGS = {
    'AlphaZero': {
        'module': 'rewards.custom_reward',
        'class': 'MyCustomReward'
    },
    'D3QN-2022': {
        'module': 'rewards.d3qn_reward',
        'class': 'D3QNSurvivalReward'
    },
    # Add more reward mappings here as needed
    # 'ActionBased': {
    #     'module': 'action_reward',
    #     'class': 'ActionReward'
    # },
}


def get_reward_class(reward_name):
    """
    Factory function to dynamically load reward class from name
    
    Parameters:
    -----------
    reward_name : str
        Name of the reward type (e.g., 'AlphaZero')
        Must be defined in REWARD_MAPPINGS
        
    Returns:
    --------
    reward_class : BaseReward
        The reward class to use
    """
    
    if reward_name not in REWARD_MAPPINGS:
        available = list(REWARD_MAPPINGS.keys())
        raise ValueError(f"Unknown reward name '{reward_name}'. Available: {available}")
    
    mapping = REWARD_MAPPINGS[reward_name]
    module_name = mapping['module']
    class_name = mapping['class']
    
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        
        # Get the class from the module
        if not hasattr(module, class_name):
            raise AttributeError(f"Module '{module_name}' has no class '{class_name}'")
        
        reward_class = getattr(module, class_name)
        
        return reward_class
        
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading reward class '{class_name}' from '{module_name}': {e}")


def create_reward(reward_name):
    """
    Create reward instance from name
    
    Parameters:
    -----------
    reward_name : str
        Name of the reward type
        
    Returns:
    --------
    reward : BaseReward
        Reward instance
    """
    
    reward_class = get_reward_class(reward_name)
    return reward_class()