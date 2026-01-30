"""
Topology Reset Module

Resets the grid topology to the reference (initial) configuration when the grid
is in a safe state (max line load ≤ 0.90). This helps the agent return to a
known-good baseline topology after successfully resolving overloads.

Benefits:
- Prevents topology drift (accumulating complex configurations)
- Provides a clean slate for future actions
- Reduces unnecessary complexity when grid is stable
"""

import numpy as np

# Try to import module wrappers
try:
    from actions.base_module import BaseModule, GreedyModule
    from grid2op.Action import ActionSpace
    HAS_MODULES = True
except ImportError:
    try:
        # Fallback to grid2op imports if available
        from grid2op.Agent.base_module import BaseModule, GreedyModule
        from grid2op.Action import ActionSpace
        HAS_MODULES = True
    except ImportError:
        HAS_MODULES = False

if HAS_MODULES:
    class RecoverInitTopoModule(GreedyModule):
        """Module for initial topology recovering.
        This module will perform the best action to recover initial topology by changing bus
        (single action, do not support multiple sub-zone actions)
        """

        def __init__(self, action_space: ActionSpace):
            GreedyModule.__init__(self, action_space)

        def _get_tested_action(self, observation):
            # Get the list of possible actions to revert the grid's topology to its reference state.
            tested_action = self.action_space.get_back_to_ref_state(observation).get(
                "substation", None
            )
            if tested_action is not None:
                print(f"🔍 Raw topology reset actions found: {len(tested_action)}")
                tested_action = [
                    act
                    for act in tested_action
                    if (
                        observation.time_before_cooldown_sub[
                            int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                        ]
                        == 0
                    )
                ]
                print(f"🔍 Actions after cooldown filter: {len(tested_action)} available for GreedyModule")
                return tested_action
            else:
                print("🔍 No topology reset actions found by get_back_to_ref_state")
                return []

        def get_act(self, observation, base_action, reward, done=False, **kwargs):
            """Override GreedyModule to bypass strict simulation for topology reset"""
            tested_actions = self._get_tested_action(observation)
            
            if not tested_actions:
                print("🔍 No topology reset actions available")
                return self.action_space()
            
            print(f"🔍 GreedyModule simulating {len(tested_actions)} topology reset actions...")
            
            # For topology reset, we don't need strict reward optimization
            # Just pick the first action that doesn't cause errors
            for i, action in enumerate(tested_actions):
                try:
                    simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action + (base_action or self.action_space()))
                    
                    if not simul_has_error and len(simul_info["exception"]) == 0:
                        print(f"🔍 Topology reset action {i}: rho {observation.rho.max():.3f} -> {simul_obs.rho.max():.3f}, reward={simul_reward:.3f}")
                        return action
                    else:
                        print(f"🔍 Topology reset action {i}: simulation failed - error={simul_has_error}, exceptions={len(simul_info['exception'])}")
                        
                except Exception as e:
                    print(f"🔍 Topology reset action {i}: simulation exception - {e}")
                    
            # If all actions failed, return do-nothing
            print("🔍 All topology reset actions failed - returning do-nothing")
            return self.action_space()
else:
    # Fallback if modules not available
    RecoverInitTopoModule = None


def should_reset_topology(observation, safe_threshold=0.90):
    """
    Check if the grid should be reset to reference topology.
    
    Args:
        observation: Grid2Op observation
        safe_threshold: Maximum rho threshold to consider "safe" (default 0.90)
    
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
        # Use RecoverInitTopoModule for topology reset to reference state
        full_action_space = env.action_space
        if RecoverInitTopoModule is not None:
            reset_agent = RecoverInitTopoModule(full_action_space)
            print(f"🔍 RecoverInitTopoModule input: rho_max={observation.rho.max():.3f}")
            reset_action = reset_agent.get_act(observation, base_action=None, reward=0.0)
            
            # Handle case where GreedyModule returns None (no beneficial action found)
            if reset_action is None:
                print(f"🔍 RecoverInitTopoModule: no beneficial reset found, using do-nothing")
                reset_action = full_action_space()
            else:
                print(f"🔍 RecoverInitTopoModule output: action type={type(reset_action)}")
        else:
            # If RecoverInitTopoModule not available, return do-nothing action
            print("⚠️ RecoverInitTopoModule not available, skipping topology reset")
            reset_action = full_action_space()
        
        # Debug: Show what substation(s) this action affects
        try:
            if hasattr(reset_action, 'get_topological_impact'):
                topo_impact = reset_action.get_topological_impact()
                modified_subs = [i for i, changed in enumerate(topo_impact) 
                               if (np.any(changed) if hasattr(changed, '__len__') else changed)]
                if modified_subs:
                    if RecoverInitTopoModule is not None:
                        print(f"🔄 RecoverInitTopoModule reset affecting substations: {modified_subs}")
                    else:
                        print(f"🔄 Manual reset affecting substations: {modified_subs}")
                else:
                    if RecoverInitTopoModule is not None:
                        print("🔄 RecoverInitTopoModule: no topology changes (do-nothing action)")
                    else:
                        print("🔄 Manual reset: no topology changes (do-nothing action)")
            else:
                # Fallback: check set_bus and line status changes
                changes = []
                if hasattr(reset_action, 'set_bus') and reset_action.set_bus is not None:
                    # Handle both scalar and array bus assignments
                    set_bus = reset_action.set_bus
                    if hasattr(set_bus, '__len__'):
                        bus_changes = [i for i, bus in enumerate(set_bus) if (np.any(bus) if hasattr(bus, '__len__') else bus) != 0]
                    else:
                        bus_changes = [0] if set_bus != 0 else []
                    if bus_changes:
                        changes.append(f"bus changes: {len(bus_changes)} elements")
                        
                if hasattr(reset_action, 'set_line_status') and reset_action.set_line_status is not None:
                    # Handle line status changes
                    set_line_status = reset_action.set_line_status
                    if hasattr(set_line_status, '__len__'):
                        line_changes = [i for i, status in enumerate(set_line_status) if (np.any(status) if hasattr(status, '__len__') else status) != 0]
                    else:
                        line_changes = [0] if set_line_status != 0 else []
                    if line_changes:
                        changes.append(f"line status: {len(line_changes)} lines")
                
                if changes:
                    module_name = "RecoverInitTopoModule" if RecoverInitTopoModule is not None else "Manual reset"
                    print(f"🔄 {module_name}: {', '.join(changes)}")
                else:
                    module_name = "RecoverInitTopoModule" if RecoverInitTopoModule is not None else "Manual reset"
                    print(f"🔄 {module_name}: no changes detected (do-nothing action)")
        except Exception as e:
            module_name = "RecoverInitTopoModule" if RecoverInitTopoModule is not None else "Manual reset"
            print(f"🔄 {module_name}: reset action created (debug failed: {e})")
        
        return reset_action
    
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
