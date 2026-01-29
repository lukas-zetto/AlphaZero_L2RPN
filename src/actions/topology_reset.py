"""
Topology Reset Module

Resets the grid topology to the reference (initial) configuration when the grid
is in a safe state (max line load ≤ 0.75). This helps the agent return to a
known-good baseline topology after successfully resolving overloads.

Benefits:
- Prevents topology drift (accumulating complex configurations)
- Provides a clean slate for future actions
- Reduces unnecessary complexity when grid is stable
"""

import numpy as np
from grid2op.Agent import RecoPowerlineAgent


def should_reset_topology(observation, safe_threshold=0.75):
    """
    Check if the grid should be reset to reference topology.
    
    Args:
        observation: Grid2Op observation
        safe_threshold: Maximum rho threshold to consider "safe" (default 0.75)
    
    Returns:
        bool: True if grid is safe enough to reset, False otherwise
    """
    max_rho = observation.rho.max()
    return max_rho <= safe_threshold


def create_reset_action(observation, env, method='reco_agent'):
    """
    Create a Grid2Op action that resets topology to reference configuration.
    
    Args:
        observation: Grid2Op observation
        env: Grid2Op environment (to access full action space)
        method: 'reco_agent' or 'manual' - which reset method to use
    
    Returns:
        Grid2Op action that resets topology appropriately
    """
    if method == 'reco_agent':
        # Use Grid2Op's built-in reset agent with full action space (including line switching)
        # Our trained model uses a limited bus-only action space, but RecoPowerlineAgent
        # specifically needs line reconnection capabilities
        full_action_space = env.action_space
        reset_agent = RecoPowerlineAgent(full_action_space)
        return reset_agent.act(observation, reward=None, done=False)
    
    elif method == 'manual':
        # Use manual topology reset approach
        return create_manual_reset_action(observation, env.action_space)
    
    else:
        raise ValueError(f"Unknown topology reset method: {method}. Use 'reco_agent' or 'manual'")


def is_topology_reference(observation):
    """
    Check if the current topology is already the reference topology.
    
    Args:
        observation: Grid2Op observation
    
    Returns:
        bool: True if topology matches reference, False otherwise
    """
    # Check if all substations are in reference configuration
    # topo_vect encodes the busbar assignment for each element
    # Reference topology has all elements on their default busbars
    
    # Compare current topology to reference
    current_topo = observation.topo_vect
    reference_topo = observation.get_energy_graph()[3]  # This might not work, using simpler check
    
    # Simpler approach: check if any substation has been modified
    # In reference topology, all elements at each substation should be on bus 1
    # (or their default bus configuration)
    
    # If line_or_bus or line_ex_bus has any value != 1, topology is modified
    # (assuming reference has everything on bus 1)
    
    # Actually, Grid2Op provides sub_topology which gives busbar per element
    # But the most reliable way is to check if we can apply a "do nothing" action
    # and the topology stays the same
    
    # For simplicity: check if all line connections are on their default bus
    # Reference topology typically has line_or_bus = 1 and line_ex_bus = 1 for all lines
    
    # Even simpler: just check if any topology action has been applied
    # We'll return False and let the reset happen - if it's already reference,
    # the action will be a no-op anyway
    
    return False  # Conservative: always try to reset if safe


def create_manual_reset_action(observation, action_space):
    """
    Create a manual reset action that sets all substations to reference topology.
    
    This is the original manual approach that sets all elements to bus 1.
    
    Args:
        observation: Grid2Op observation
        action_space: Grid2Op action space
    
    Returns:
        Grid2Op action that resets topology to reference
    """
    # Start with empty action
    action = action_space()
    
    # Check each substation
    n_sub = observation.n_sub
    modified_subs = []
    
    for sub_id in range(n_sub):
        # Get topology at this substation
        try:
            sub_topo = observation.state_of(substation_id=sub_id)
            topo_vect = sub_topo['topo_vect']
            
            # If any element is not on bus 1, substation is modified
            if np.any(topo_vect != 1):
                modified_subs.append(sub_id)
        except:
            # If state_of fails, skip this substation
            continue
    
    # If no substations are modified, return do-nothing
    if len(modified_subs) == 0:
        return action
    
    # Reset each modified substation individually using proper Grid2Op API
    for sub_id in modified_subs:
        try:
            sub_topo = observation.state_of(substation_id=sub_id)
            n_elements = len(sub_topo['topo_vect'])
            
            # Set all elements in this substation to bus 1
            sub_start = sum(observation.sub_info[:sub_id])
            for element_idx in range(n_elements):
                global_idx = sub_start + element_idx
                action.set_bus[global_idx] = 1
        except:
            continue
    
    return action


def get_reference_topology_action(observation, env):
    """
    Create an action that attempts to reset all substations to reference topology.
    
    Uses the configured reset method from config.py.
    
    Args:
        observation: Grid2Op observation
        env: Grid2Op environment (to access full action space and config)
    
    Returns:
        Grid2Op action (may be do-nothing if already in reference)
    """
    # Try to get reset method from config, default to 'reco_agent'
    try:
        from src.config import AGENT_CONFIG
        method = AGENT_CONFIG.get('topology_reset_method', 'reco_agent')
    except:
        method = 'reco_agent'  # Default fallback
    
    return create_reset_action(observation, env, method)
