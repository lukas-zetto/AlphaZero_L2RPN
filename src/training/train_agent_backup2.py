#!/usr/bin/env python3
"""
Alpha Zero MCTS Training Agent for Grid2Op Line Switching

This implements the TRUE Alpha Zero algorithm with:
- Real Grid2Op environment execution during MCTS
- Actual reward collection and state transitions  
- Proper discounted reward backpropagation
- Recovery node detection for early stopping
- Maximum reachable steps for action selectidef run_alpha_zero_mcts(env, neural_network=None, num_simulations=100, c_puct=1.5,n

This matches the Alpha Zero paper's approach of using real environment
execution during tree search rather than just neural network predictions.
"""

import os
import sys
import random
import math
import numpy as np
import torch
import grid2op
from grid2op import make
from collections import defaultdict

# Try to import LightSimBackend
try:
    from lightsim2grid import LightSimBackend
except ImportError:
    try:
        from grid2op.Backend import LightSimBackend
    except ImportError:
        print("⚠️ LightSimBackend not available, using default backend")
        LightSimBackend = None

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.config import AGENT_CONFIG
    from src.networks.neural_network_factory import get_neural_network_functions
    from src.networks.neural_network import AlphaZeroNetwork  # Still need the class for type checking
    
    # Get neural network functions based on config
    nn_funcs = get_neural_network_functions(AGENT_CONFIG)
    NEURAL_NETWORK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Neural network components not available: {e}")
    NEURAL_NETWORK_AVAILABLE = False
    # Placeholder config for testing
    AGENT_CONFIG = {
        'hidden_size': 256, 'learning_rate': 0.001,
        'mcts_simulations': 50, 'training_epochs': 1,
        'num_training_episodes': 10, 'gamma': 0.99, 'puct_c': 1.5,
        'input_size': 60  # 20 rho + 20 status + 20 cooldowns
    }


class MCTSNode:
    """
    Alpha Zero MCTS Node with real Grid2Op environment execution.
    
    Key features:
    - Stores actual Grid2Op rewards from environment execution
    - Tracks maximum reachable episode steps (recovery detection)
    - Proper discounted reward backpropagation
    - Early stopping when recovery nodes found
    """
    
    def __init__(self, parent=None, action=None, prior_prob=0.0, action_idx=None):
        self.parent = parent
        self.action = action  # Grid2Op action that led to this node
        self.action_idx = action_idx  # Index in action catalog/space (for training data)
        self.children = {}    # Dict mapping action strings to child nodes
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        
        # Alpha Zero specific - real environment execution results
        self.immediate_reward = 0.0  # Reward from executing action to reach this node
        self.cumulative_reward = 0.0  # Discounted cumulative reward
        self.max_reachable_steps = 0  # Maximum episode steps reached from this node
        self.is_recovery_node = False  # Whether this leads to grid recovery
        
        # Terminal state information
        self.is_terminal = False
        self.terminal_reason = None  # 'done', 'illegal', 'ambiguous', 'timeout'
        
    def is_expanded(self):
        """Check if this node has been expanded (has children)."""
        return len(self.children) > 0
    
    def select_child(self, c_puct=1.5, t_skipped=10):
        """
        Select child using PUCT formula with recovery node prioritization.
        
        PUCT = Q(s,a) + U(s,a) + Recovery_Bonus(s,a)
        """
        best_score = -float('inf')
        best_action = None
        
        for action_str, child in self.children.items():
            if child.visit_count == 0:
                # Unvisited nodes get infinite exploration value
                q_value = 0.0
                u_value = float('inf')
            else:
                # Q-value: average discounted reward (clipped to reasonable range)
                q_value = np.clip(child.value_sum / child.visit_count, -10.0, 10.0)
                
                # U-value: exploration bonus
                u_value = (c_puct * child.prior_prob * 
                          math.sqrt(self.visit_count) / (1 + child.visit_count))
            
            # Recovery bonus: prioritize nodes that lead to longer episodes
            recovery_bonus = 0.0
            if child.is_recovery_node:
                recovery_bonus = 2.0  # Strong bias toward recovery actions
            elif child.max_reachable_steps > t_skipped:  # Use t_skipped from config
                recovery_bonus = 1.0
            
            score = q_value + u_value + recovery_bonus
            
            if score > best_score:
                best_score = score
                best_action = action_str
        
        return self.children[best_action] if best_action is not None else None
    
    def expand(self, valid_actions, action_probs):
        """Expand this node by adding children for all valid actions."""
        for i, action in enumerate(valid_actions):
            action_str = str(action)
            if action_str not in self.children:
                prob = action_probs[i] if i < len(action_probs) else (1.0 / len(valid_actions))
                self.children[action_str] = MCTSNode(parent=self, action=action, prior_prob=prob, action_idx=i)

    def expand_one_child(self, valid_actions, action_probs):
        """Expand this node by adding ONE child for ONE unexplored action (proper AlphaZero)."""
        for i, action in enumerate(valid_actions):
            action_str = str(action)
            if action_str not in self.children:  # Find first unexplored action
                prob = action_probs[i] if i < len(action_probs) else (1.0 / len(valid_actions))
                new_child = MCTSNode(parent=self, action=action, prior_prob=prob, action_idx=i)
                self.children[action_str] = new_child
                return new_child  # Return the new leaf node
        return None  # All actions already explored
    
    def backup_value(self, accumulated_value, max_steps_reached, t_skipped=10):
        """
        Backup accumulated value up the tree (AlphaZero style).
        
        This implements proper Alpha Zero backup:
        - Start with leaf neural network value
        - Add discounted immediate rewards going up the path
        - Each node receives: leaf_value + Σ(γ^k * step_rewards)
        """
        self.visit_count += 1
        self.value_sum += accumulated_value
        
        # Track maximum episode performance
        if max_steps_reached > self.max_reachable_steps:
            self.max_reachable_steps = max_steps_reached
            
        # Mark as recovery node based on performance
        if max_steps_reached > t_skipped:  # Use t_skipped from config
            self.is_recovery_node = True
    
    def get_depth(self):
        """Get depth of this node in the tree."""
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth
    
    def get_action_probs(self, temperature=1.0, config=None):
        """Get action probabilities based on visit counts (standard AlphaZero).
        
        Uses pure MCTS visit counts transformed by temperature, as per AlphaZero paper.
        Temperature controls exploration vs exploitation:
        - temp = 0: Greedy (pick most visited)
        - temp = 1: Proportional to visit counts
        - temp < 1: Sharper (favor most visited)
        - temp > 1: Flatter (more uniform)
        
        Returns list of (action_idx, prob) tuples where action_idx can be used directly
        for training data indexing.
        """
        if not self.children:
            return []

        action_indices = []
        visit_counts = []

        for action_str, child in self.children.items():
            # Use action_idx if available (catalog mode), otherwise use action object
            action_indices.append(child.action_idx if child.action_idx is not None else child.action)
            # Use pure visit counts (standard AlphaZero)
            visit_counts.append(child.visit_count)

        visit_counts = np.array(visit_counts, dtype=np.float32)

        if temperature == 0:
            # Greedy selection: pick most visited action
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(action_indices), dtype=np.float32)
            probs[best_idx] = 1.0
        else:
            # Temperature-based selection using pure visit counts
            if temperature != 1.0:
                # Apply temperature: visits^(1/temperature)
                # temp < 1 sharpens (favors most visited)
                # temp > 1 flattens (more uniform)
                visit_counts = visit_counts ** (1.0 / temperature)
            
            # Normalize to probabilities
            total = np.sum(visit_counts)
            if total > 0:
                probs = visit_counts / total
            else:
                # Fallback to uniform if no visits (shouldn't happen)
                probs = np.ones(len(action_indices), dtype=np.float32) / len(action_indices)

        return list(zip(action_indices, probs))


def is_critical_state(observation, threshold=None):
    """
    Check if grid is in critical state requiring MCTS intervention.

    According to the paper, MCTS should only run when line load > threshold
    to focus computational resources on situations that actually need intervention.

    Parameters:
    -----------
    observation : grid2op.Observation
        Current grid observation
    threshold : float, optional
        Critical threshold for maximum line loading (if None, taken from config)

    Returns:
    --------
    is_critical : bool
        True if any line load exceeds threshold
    """
    if threshold is None:
        threshold = AGENT_CONFIG.get('critical_threshold', 0.95)
    max_rho = np.max(observation.rho)
    return max_rho > threshold


def get_action_index_from_bus_handler(action, bus_actions):
    """
    Get action index using bus switching handler's reverse lookup.
    
    Parameters:
    -----------
    action : grid2op.Action
        Grid2Op action to find index for
    bus_actions : BusSwitchingActions
        Bus switching handler
        
    Returns:
    --------
    index : int
        Action index (0 to max_actions-1), or 0 if not found
    """
    # Try to find the action by comparing with all generated actions
    for action_idx in range(bus_actions.get_action_space_size()):
        try:
            test_action = bus_actions.create_action(action_idx)
            # Simple comparison - check if actions have same effect
            if str(action) == str(test_action):
                return action_idx
        except:
            continue
    
    # If no exact match found, return do-nothing action (index 0)
    return 0


def simulate_with_safe_skipping(sim_env, action, max_skip_steps=None, safe_threshold=None):
    """
    Execute action and automatically skip safe steps.
    
    This implements the paper's "Grid State Observer" concept where
    safe states (line load < 95%) are automatically skipped with do-nothing actions
    to focus MCTS computation on critical decisions.
    
    Parameters:
    -----------
    sim_env : grid2op.Environment
        Simulation environment (copied)
    action : grid2op.Action
        Action to execute first
    max_skip_steps : int or None
        Maximum number of safe steps to skip automatically (None = unlimited)
    safe_threshold : float, optional
        Threshold below which state is considered safe (if None, taken from config)
        
    Returns:
    --------
    obs : grid2op.Observation
        Final observation after action + skipped steps
    total_reward : float
        Cumulative reward from action + skipped steps
    done : bool
        Episode termination status
    info : dict
        Final step info
    skipped_steps : int
        Number of safe steps that were automatically skipped
    """
    if safe_threshold is None:
        safe_threshold = AGENT_CONFIG.get('critical_threshold', 0.95)
    # Execute the main action
    obs, reward, done, info = sim_env.step(action)
    total_reward = reward
    skipped_steps = 0
    # Auto-skip safe steps with do-nothing actions
    while (not done and 
           not info.get('is_illegal', False) and 
           np.max(obs.rho) < safe_threshold and 
           (max_skip_steps is None or skipped_steps < max_skip_steps)):
        obs, step_reward, done, info = sim_env.step(sim_env.action_space())  # Do nothing
        total_reward += step_reward
        skipped_steps += 1
    return obs, total_reward, done, info, skipped_steps


def simulate_with_safe_skipping(sim_env, action, max_skip_steps=None, safe_threshold=None):
    """
    Execute action and automatically skip safe steps.
    
    This implements the paper's "Grid State Observer" concept where
    safe states (line load < 95%) are automatically skipped with do-nothing actions
    to focus MCTS computation on critical decisions.
    
    Parameters:
    -----------
    sim_env : grid2op.Environment
        Simulation environment (copied)
    action : grid2op.Action
        Action to execute first
    max_skip_steps : int or None
        Maximum number of safe steps to skip automatically (None = unlimited)
    safe_threshold : float, optional
        Threshold below which state is considered safe (if None, taken from config)
        
    Returns:
    --------
    obs : grid2op.Observation
        Final observation after action + skipped steps
    total_reward : float
        Cumulative reward from action + skipped steps
    done : bool
        Episode termination status
    info : dict
        Final step info
    skipped_steps : int
        Number of safe steps that were automatically skipped
    """
    if safe_threshold is None:
        safe_threshold = AGENT_CONFIG.get('critical_threshold', 0.95)
    # Execute the main action
    obs, reward, done, info = sim_env.step(action)
    total_reward = reward
    skipped_steps = 0
    # Auto-skip safe steps with do-nothing actions
    while (not done and 
           not info.get('is_illegal', False) and 
           np.max(obs.rho) < safe_threshold and 
           (max_skip_steps is None or skipped_steps < max_skip_steps)):
        obs, step_reward, done, info = sim_env.step(sim_env.action_space())  # Do nothing
        total_reward += step_reward
        skipped_steps += 1
    return obs, total_reward, done, info, skipped_steps


def action_probs_to_fixed_policy(action_probs, observation, num_actions, bus_actions=None, action_catalog=None):
    """
    Convert MCTS action probabilities to fixed-size policy vector.
    
    NOTE: When using catalog, action_probs should contain (action_index, prob) tuples
    where action_index is the position in the catalog (0-59 for N1 mode).
    
    Parameters:
    -----------
    action_probs : list of (action_idx, prob) tuples OR (action, prob) tuples
        MCTS action probabilities with indices or action objects
    observation : grid2op.Observation
        Current observation 
    num_actions : int
        Fixed action space size (60 for catalog, 38 for bus switching, 21 for line switching only)
    bus_actions : BusSwitchingActions, optional
        Bus switching handler (not used if action_catalog provided)
    action_catalog : ActionCatalog, optional
        Catalog-based action space (takes priority if provided)
        
    Returns:
    --------
    fixed_policy : np.ndarray
        Fixed-size policy vector with probabilities in correct positions
    """
    # Create fixed-size policy vector (all zeros initially)
    fixed_policy = np.zeros(num_actions, dtype=np.float32)
    
    # Map each MCTS action probability to its correct position
    for item, prob in action_probs:
        try:
            # Check if item is already an index (integer) or an action object
            if isinstance(item, int):
                action_idx = item  # Already an index
            else:
                # Legacy: item is a Grid2Op action, need to find its index
                if bus_actions is not None:
                    action_idx = get_action_index_from_bus_handler(item, bus_actions)
                else:
                    # Simple line switching logic for legacy support
                    try:
                        lines_impacted, _ = item.get_topological_impact(observation)
                        if not lines_impacted.any():
                            action_idx = 0  # Do-nothing
                        else:
                            line_id = np.where(lines_impacted)[0][0]
                            action_idx = line_id + 1  # Lines 0-19 map to indices 1-20
                    except:
                        action_idx = 0  # Fallback to do-nothing
            
            if 0 <= action_idx < num_actions:  # Ensure index is within bounds
                fixed_policy[action_idx] = float(prob)
        except Exception as e:
            # If mapping fails, skip this action
            continue
    
    return fixed_policy


def build_valid_actions(observation, action_space, bus_actions=None, action_catalog=None):
    """
    Build list of valid actions based on current observation and action space configuration.
    
    Priority: action_catalog > bus_actions > line switching only
    
    Parameters:
    -----------
    observation : grid2op.Observation
        Current grid state
    action_space : grid2op.ActionSpace
        Grid2Op action space
    bus_actions : BusSwitchingActions, optional
        LEGACY bus switching handler (ignored if action_catalog provided)
    action_catalog : ActionCatalog, optional
        NEW catalog-based action space (preferred)
        
    Returns:
    --------
    valid_actions : list
        List of valid Grid2Op actions (do nothing + line switches + bus switches)
    """
    # Priority 1: Use catalog if available
    if action_catalog is not None:
        valid_actions = []
        for layout_action in action_catalog.actions:
            try:
                action = layout_action.apply(action_space)
                valid_actions.append(action)
            except Exception:
                pass  # Skip actions that fail to create
        # Prepend do-nothing action
        valid_actions.insert(0, action_space())
        return valid_actions
    
    # Priority 2: Use legacy bus_actions if catalog not available
    if bus_actions is not None:
        # Use expanded action space with bus switching
        num_actions = bus_actions.get_action_space_size()
        valid_actions = []
        
        for action_idx in range(num_actions):
            try:
                action = bus_actions.create_action(action_idx, action_space)
                valid_actions.append(action)
            except Exception as e:
                # If action creation fails, skip this action
                print(f"⚠️ Failed to create action {action_idx}: {e}")
                continue
        
        return valid_actions
    else:
        # Fallback to line switching only
        return build_line_switching_actions(observation, action_space)


def build_line_switching_actions(observation, action_space):
    """
    Build all valid line switching actions for current observation.
    
    Focuses on discrete line switching actions only as per Alpha Zero approach.
    
    Parameters:
    -----------
    observation : grid2op.Observation
        Current grid observation
    action_space : grid2op.ActionSpace  
        Grid2Op action space
        
    Returns:
    --------
    valid_actions : list
        List of valid Grid2Op actions (do nothing + line switches)
    """
    valid_actions = []
    
    try:
        # Always include "do nothing" action
        do_nothing = action_space()
        valid_actions.append(do_nothing)
        
        # Line switching actions for lines not in cooldown
        for line_id in range(observation.n_line):
            if observation.time_before_cooldown_line[line_id] == 0:
                
                # Disconnect action (if line is connected)
                if observation.line_status[line_id]:
                    try:
                        disconnect_action = action_space.disconnect_powerline(line_id=line_id)
                        valid_actions.append(disconnect_action)
                    except:
                        pass
                
                # Reconnect action (if line is disconnected)  
                else:
                    try:
                        reconnect_action = action_space.reconnect_powerline(line_id=line_id)
                        valid_actions.append(reconnect_action)
                    except:
                        pass
        
        return valid_actions
        
    except Exception as e:
        print(f"⚠️ Error building actions: {e}")
        return [action_space()]  # Return only do-nothing as fallback


def is_grid_in_recovery(observation, recovery_threshold=None):
    """
    Check if the grid is in a recovery state (safe operation).
    
    Parameters:
    -----------
    observation : grid2op.Observation
        Current grid observation
    recovery_threshold : float, optional
        Maximum line load threshold for recovery (if None, taken from config)
        
    Returns:
    --------
    bool : True if grid is in recovery (all lines under threshold)
    """
    if recovery_threshold is None:
        recovery_threshold = AGENT_CONFIG.get('critical_threshold', 0.95)
    try:
        max_rho = np.max(observation.rho)
        return max_rho < recovery_threshold
    except:
        return False

def should_use_mcts(observation, threshold=None):
    """
    Decide whether to run expensive MCTS or use simple heuristics.
    
    According to the paper, MCTS should only be used in critical states
    to focus computational resources efficiently.
    
    Parameters:
    -----------
    observation : grid2op.Observation
        Current observation
    threshold : float, optional
        Critical threshold (if None, taken from config)
        
    Returns:
    --------
    use_mcts : bool
        True if MCTS should be used, False for simple heuristics
    """
    if threshold is None:
        threshold = AGENT_CONFIG.get('critical_threshold', 0.95)
    return is_critical_state(observation, threshold)


def run_alpha_zero_mcts(env, neural_network=None, num_simulations=None, c_puct=None, 
                        max_depth=20, gamma=None, early_stop_recovery=True, config=None, bus_actions=None, action_catalog=None):
    """
    TRUE Alpha Zero MCTS with real Grid2Op environment execution.
    
    This implements the paper's approach of using actual environment execution
    during MCTS search rather than just neural network predictions.
    
    Key Features:
    - Executes actual Grid2Op actions during tree search
    - Collects real rewards and state transitions  
    - Proper discounted reward backpropagation
    - Recovery node detection for early stopping
    - Maximum reachable steps tracking
    
    Parameters:
    -----------
    env : grid2op.Environment
        Current Grid2Op environment (will be copied for simulations)
    neural_network : torch.nn.Module, optional
        Neural network for policy/value predictions (uses uniform if None)
    num_simulations : int
        Number of MCTS simulations to run
    c_puct : float
        Exploration constant for PUCT formula
    max_depth : int
        Maximum depth for each simulation
    gamma : float
        Discount factor for reward backpropagation
    early_stop_recovery : bool
        Whether to stop early when recovery nodes are found
        
    Returns:
    --------
    best_action : grid2op.Action
        Best action found by MCTS (leads to maximum reachable steps)
    action_probs : list
        List of (action, probability) tuples based on visit counts
    search_stats : dict  
        MCTS search statistics and performance metrics
    """
    
    # Extract config parameters (use defaults if config not provided)
    if config is None:
        config = {}
    
    # Apply config parameters with proper defaults
    if num_simulations is None:
        num_simulations = config.get('mcts_simulations', 200)
    if c_puct is None:
        c_puct = config.get('puct_c', 1.4)
    if gamma is None:
        gamma = config.get('gamma', 0.99)
        
    t_skipped = config.get('t_skipped', 5)
    t_stopping = config.get('t_stopping', 3)
    
    # Get current observation and build action space
    observation = env.get_obs()
    valid_actions = build_valid_actions(observation, env.action_space, action_catalog=action_catalog)
    
    if len(valid_actions) <= 1:
        # Only do-nothing available
        return valid_actions[0], [(valid_actions[0], 1.0)], {'simulations': 0}
    
    # Initialize root node with neural network policy (or uniform)
    root = MCTSNode()
    
    # Get policy from neural network or use uniform
    # CRITICAL: Use full action space size for neural network, not just valid actions
    if neural_network is not None:
        try:
            # Determine full action space size based on what's being used
            if action_catalog is not None:
                full_action_space_size = action_catalog.size  # 60 catalog actions
            elif bus_actions is not None:
                full_action_space_size = bus_actions.get_action_space_size()
            else:
                full_action_space_size = 21  # Line switching only
            
            # Get network policy (60 actions for catalog)
            network_policy, root_value = nn_funcs.neural_network_forward(neural_network, observation, full_action_space_size)
            
            # Add do-nothing probability if using catalog (do-nothing prepended to valid_actions)
            if action_catalog is not None:
                # Network doesn't model do-nothing, so add small uniform probability
                do_nothing_prob = 1.0 / (full_action_space_size + 1)
                # Scale down other probabilities to make room
                network_policy = np.array(network_policy) * (1.0 - do_nothing_prob)
                # Prepend do-nothing probability
                policy_probs = np.concatenate([[do_nothing_prob], network_policy])
            else:
                # Truncate to match valid actions
                policy_probs = network_policy[:len(valid_actions)]
                
        except Exception as e:
            print(f"⚠️ Neural network forward failed: {e}")
            policy_probs = [1.0 / len(valid_actions)] * len(valid_actions)
    else:
        policy_probs = [1.0 / len(valid_actions)] * len(valid_actions)
    
    # Add Dirichlet noise at root to encourage exploration (AlphaZero technique)
    # This prevents the policy from collapsing to a single action
    dirichlet_alpha = config.get('dirichlet_alpha', 0.3)  # Standard value for AlphaZero
    dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)  # Mix 75% network, 25% noise
    
    if len(valid_actions) > 1:
        # Generate Dirichlet noise: random distribution that sums to 1
        # Use len(valid_actions) to match the actual number of actions (61 = do-nothing + 60 catalog)
        noise = np.random.dirichlet([dirichlet_alpha] * len(valid_actions))
        
        # Mix network priors with noise: 75% network policy, 25% random exploration
        policy_probs = np.array(policy_probs)
        policy_probs = (1 - dirichlet_epsilon) * policy_probs + dirichlet_epsilon * noise
        policy_probs = policy_probs.tolist()
    
    # Expand root with mixed policy (network + noise)
    root.expand(valid_actions, policy_probs)
    
    # MCTS Simulations with REAL Grid2Op environment execution
    recovery_nodes_found = 0
    max_episode_steps_reached = 0
    
    # Get the current observation to use for simulation
    current_obs = env.get_obs()
    
    for sim in range(num_simulations):
        # 🔑 CRUCIAL: Use observation.simulate() instead of env.copy()
        # This avoids deep copy issues and is the recommended Grid2Op approach
        # We'll simulate from the current observation state
        
        # Selection Phase: Traverse tree using PUCT + recovery bonus
        node = root
        path = [root]
        depth = 0
        sim_obs = current_obs  # Start from current observation
        
        while node.is_expanded() and depth < max_depth:
            node = node.select_child(c_puct, t_skipped)
            if node is None:
                break
            path.append(node)
                
            # 🔑 SIMULATE ACTION using obs.simulate() instead of env.copy()
            try:
                # Use Grid2Op's built-in simulation from observation
                sim_obs_result, sim_reward, sim_done, sim_info = sim_obs.simulate(node.action)
                
                # Store results
                node.immediate_reward = sim_reward
                node.observation = sim_obs_result  # Store observation for neural network evaluation
                node.skipped_steps = 0  # No safe-skipping in simulation mode
                
                # Update simulation observation for next iteration
                sim_obs = sim_obs_result
                
                # Check terminal conditions
                if sim_done or sim_info.get('is_illegal', False) or sim_info.get('is_ambiguous', False):
                    node.is_terminal = True
                    node.terminal_reason = 'done' if sim_done else 'illegal'
                    break
                    
            except Exception as e:
                # Suppress common NoForecastAvailable warnings - this is normal at end of chronics
                if "NoForecastAvailable" not in str(e):
                    print(f"[MCTS Exception] {e}")
                node.is_terminal = True
                node.terminal_reason = 'exception'
                node.immediate_reward = -10.0
                break
                
            depth += 1
        
        # Expansion and Rollout Phase
        episode_reward = 0.0
        max_steps_in_this_simulation = depth
        
        if not node.is_terminal and depth < max_depth:
            # Expand current node
            try:
                # Use the current simulation observation
                current_sim_obs = sim_obs
                current_actions = build_valid_actions(current_sim_obs, env.action_space, action_catalog=action_catalog)
                
                if len(current_actions) > 1:
                    # Get policy for expansion (neural network or uniform)
                    if neural_network is not None:
                        try:
                            if action_catalog is not None:
                                full_action_space_size = len(action_catalog)
                            else:
                                full_action_space_size = 21  # Line switching only
                            expansion_policy, _ = nn_funcs.neural_network_forward(neural_network, current_sim_obs, full_action_space_size)
                            # Truncate to only valid actions for MCTS
                            expansion_policy = expansion_policy[:len(current_actions)]
                        except:
                            expansion_policy = [1.0 / len(current_actions)] * len(current_actions)
                    else:
                        expansion_policy = [1.0 / len(current_actions)] * len(current_actions)
                    
                    # Proper AlphaZero: expand only ONE leaf node per simulation
                    new_leaf = node.expand_one_child(current_actions, expansion_policy)
                    
                    if new_leaf is not None:
                        # Don't execute the action yet - just evaluate the new leaf state
                        # The leaf represents the state AFTER executing the action
                        new_leaf.observation = current_sim_obs  # Current observation before action
                        
                        # Use neural network to evaluate this leaf, or do rollout from current state
                        # with the new leaf's action as the first step of rollout
                    
                    # 🔑 SIMULATION ROLLOUT: Use obs.simulate() for faster rollout
                    rollout_reward = 0.0
                    rollout_steps = 0
                    rollout_obs = current_sim_obs
                    
                    # First step: execute the new leaf's action
                    first_action = new_leaf.action if new_leaf else random.choice(current_actions)
                    
                    # Run simulation rollout until done
                    while True:
                        try:
                            # Use leaf action for first step, then random
                            rollout_action = first_action if rollout_steps == 0 else random.choice(current_actions)
                            rollout_obs, reward, done, info = rollout_obs.simulate(rollout_action)
                            
                            rollout_reward += reward * (gamma ** rollout_steps)
                            rollout_steps += 1
                            
                            # Check if we reached a recovery state
                            if is_grid_in_recovery(rollout_obs):
                                recovery_nodes_found += 1
                                node.is_recovery_node = True
                                
                                # Early stopping if we found recovery
                                if early_stop_recovery and recovery_nodes_found >= t_stopping:
                                    break
                            
                            if done or info.get('is_illegal', False):
                                break
                                
                        except:
                            break
                    
                    episode_reward = rollout_reward
                    max_steps_in_this_simulation = depth + rollout_steps
                    
            except Exception as e:
                episode_reward = -1.0
        else:
            # Terminal node - use immediate reward
            episode_reward = node.immediate_reward if hasattr(node, 'immediate_reward') else 0.0
        
        # Update maximum episode steps reached
        max_episode_steps_reached = max(max_episode_steps_reached, max_steps_in_this_simulation)
        
        # 🔑 ALPHAZERO BACKUP: Start with leaf value, add discounted rewards up the path
        
        # Get leaf value from neural network (or use rollout result)
        leaf_node = path[-1] if path else None
        if leaf_node and neural_network is not None:
            try:
                # Get neural network value for leaf state
                leaf_obs = getattr(leaf_node, 'observation', None)
                if leaf_obs is not None:
                    _, leaf_value = nn_funcs.neural_network_forward(neural_network, leaf_obs, 1)
                else:
                    leaf_value = episode_reward  # Fallback to episode reward
            except:
                leaf_value = episode_reward
        else:
            leaf_value = episode_reward
        
        # Backup from leaf to root: leaf_value + Σ(γ^k * step_rewards)
        accumulated_value = leaf_value
        
        # Store the accumulated values for each node (for training data)
        node_values = {}
        
        for i, node in enumerate(reversed(path)):
            # Add immediate reward for this step (standard Bellman backup)
            if i > 0:  # Skip leaf node (already has its value)
                step_reward = getattr(node, 'immediate_reward', 0.0)
                # FIXED: Standard discounted return formula (was causing value explosion!)
                accumulated_value = step_reward + gamma * accumulated_value
                # Clip to prevent extreme values
                accumulated_value = np.clip(accumulated_value, -20.0, 20.0)
            
            # Store the accumulated value for this node
            node_values[id(node)] = accumulated_value
            
            # Backup the accumulated value to this node
            node.backup_value(accumulated_value, max_steps_in_this_simulation, t_skipped)
        
        # Early stopping if enough recovery found
        if early_stop_recovery and recovery_nodes_found >= t_stopping:
            break
    
    # Action Selection: Choose action using MCTS policy with configured temperature
    training_temperature = config.get('temperature', 1.5) if config else 1.5
    action_probs = root.get_action_probs(temperature=training_temperature, config=config)
    
    if action_probs:
        # action_probs contains (action_idx, prob) or (action_object, prob) tuples
        # Need to find the child with best performance and return its actual action object
        best_child = None
        best_score = (-1, -1, -1)  # (max_steps, visit_count, prob)
        
        for action_str, child in root.children.items():
            # Find this child in action_probs
            child_idx_or_action = child.action_idx if child.action_idx is not None else child.action
            child_prob = 0.0
            for idx_or_action, prob in action_probs:
                if idx_or_action == child_idx_or_action:
                    child_prob = prob
                    break
            
            score = (child.max_reachable_steps, child.visit_count, child_prob)
            if score > best_score:
                best_score = score
                best_child = child
        
        best_action = best_child.action if best_child else valid_actions[0]
    else:
        best_action = valid_actions[0]
    
    # Calculate the MCTS value for the root state (average of all simulations)
    root_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
    
    # Comprehensive statistics
    stats = {
        'simulations': sim,
        'recovery_nodes_found': recovery_nodes_found,
        'max_episode_steps': max_episode_steps_reached,
        'root_visits': root.visit_count,
        'num_children': len(root.children),
        'best_action_max_steps': best_child.max_reachable_steps if best_child else 0,
        'avg_child_visits': np.mean([child.visit_count for child in root.children.values()]) if root.children else 0,
        'root_value': root_value  # Add the MCTS-calculated value for training
    }
    
    return best_action, action_probs, stats


def collect_training_data(env, num_episodes=5, mcts_simulations=30, neural_network=None, config=None, line_logger=None, chronic_id=None, action_catalog=None):
    """
    Collect training data focusing on critical states only.
    
    This function implements the paper's approach:
    1. Skip safe states (line load < 95%) with do-nothing actions
    2. Run MCTS only in critical states (line load > 95%)
    3. Collect (state, action_probs, reward) tuples from critical decisions
    4. Use real Grid2Op environment rewards with safe step skipping
    
    This dramatically reduces computational requirements by focusing
    MCTS cycles only on states that actually require intervention.
    
    Parameters:
    -----------
    env : grid2op.Environment
        Grid2Op environment for data collection
    num_episodes : int
        Number of episodes to run
    mcts_simulations : int
        Number of MCTS simulations per action
    config : dict, optional
        Configuration dictionary with MCTS parameters
    action_catalog : ActionCatalog
        Catalog-based action space (REQUIRED)
        
    Returns:
    --------
    training_examples : list
        List of (observation, action_probs, reward) for neural network training
    """
    
    training_examples = []
    
    # Action catalog is required
    if action_catalog is None:
        raise ValueError("action_catalog is required for training")
    
    num_actions = action_catalog.size
    print(f"   Using catalog action space: {num_actions} actions")
    
    for episode in range(num_episodes):
        print(f"📊 Collecting episode {episode + 1}/{num_episodes}")
        
        # Reset environment for new episode
        observation = env.reset()
        episode_data = []
        if line_logger is not None:
            try:
                line_logger.start_episode(chronic_id if chronic_id is not None else -1, episode)
                line_logger.update(observation)
            except Exception as _log_err:
                print(f"   [LineLog] start/update failed: {_log_err}")
        
        # Skip to first critical state or run full episode
        safe_steps_skipped = 0
        done = False
        max_episode_steps = env.max_episode_duration()
        
        while not is_critical_state(observation) and safe_steps_skipped < max_episode_steps and not done:
            observation, _, done, _ = env.step(env.action_space())  # Do nothing in safe states
            safe_steps_skipped += 1
            if line_logger is not None:
                try:
                    line_logger.update(observation)
                except Exception:
                    pass
        
        if done:
            print(f"    Episode completed without critical state (ran {safe_steps_skipped} steps)")
            continue
        
        print(f"    Reached critical state after {safe_steps_skipped} safe steps (max_rho={np.max(observation.rho):.3f})")
        
        step = safe_steps_skipped  # Continue from where we left off
        
        # Collect training data from critical states, continue episode until done
        while not done:
            
            # Track if we ran MCTS this step (before executing action)
            ran_mcts = False
            observation_before_action = observation  # Save state BEFORE action
            
            if is_critical_state(observation):
                # Run MCTS search for critical states
                ran_mcts = True
                try:
                    best_action, action_probs, stats = run_alpha_zero_mcts(
                        env, 
                        neural_network=neural_network,  # Use the neural network passed to function
                        num_simulations=mcts_simulations,
                        config=config,  # Pass config - this will read puct_c, gamma, etc.
                        action_catalog=action_catalog  # Pass catalog
                    )
                    
                except Exception as e:
                    print(f"    ❌ MCTS failed at step {step}: {e}")
                    break
            else:
                # Use do-nothing for safe states
                best_action = env.action_space()  # Do nothing action
                # Create action_probs with same format as MCTS (do-nothing gets prob 1.0)
                valid_actions = build_valid_actions(observation, env.action_space, action_catalog=action_catalog)
                do_nothing_probs = []
                for action in valid_actions:
                    if str(action) == str(best_action):  # This is do-nothing
                        do_nothing_probs.append((action, 1.0))
                    else:
                        do_nothing_probs.append((action, 0.0))
                action_probs = do_nothing_probs
                stats = {'simulations': 0, 'safe_state': True}
            
            # Execute the action
            try:
                observation, reward, done, info = env.step(best_action)
                if line_logger is not None:
                    try:
                        line_logger.update(observation)
                    except Exception:
                        pass
                
                # Get the MCTS-calculated value for this state (the accumulated discounted return)
                mcts_value = stats.get('root_value', 0.0)
                
                # Collect training examples whenever we ran MCTS (regardless of resulting state)
                if ran_mcts:
                    # Convert action_probs to fixed-size policy vector 
                    # This ensures consistent shape regardless of which lines are on cooldown
                    probs_only = action_probs_to_fixed_policy(action_probs, observation_before_action, num_actions=num_actions, action_catalog=action_catalog)
                    
                    # Encode the observation BEFORE action for neural network training
                    try:
                        encoded_state = nn_funcs.encode_observation_simple(observation_before_action)
                    except:
                        # Fallback encoding if function not available
                        encoded_state = np.array(observation.rho.tolist(), dtype=np.float32)
                    
                    episode_data.append({
                        'state': encoded_state,      # Encoded state for neural network
                        'mcts_policy': probs_only,   # Fixed-size policy (38 elements with bus switching, 21 without)
                        'value': mcts_value,         # MCTS-calculated value (discounted returns)
                        'step': step,
                        'mcts_stats': stats
                    })
                    training_examples.append(episode_data[-1])  # Add immediately
            
                # Check termination conditions
                if done:
                    print(f"    Episode done at step {step}")
                    break
                    
                if info.get('is_illegal', False) or info.get('is_ambiguous', False):
                    print(f"    Invalid action at step {step}")
                    break
                    
            except Exception as e:
                print(f"    ❌ Action execution failed at step {step}: {e}")
                break
            
            step += 1
        
        # Calculate average episode value (from MCTS calculations)
        avg_value = np.mean([ex['value'] for ex in episode_data]) if episode_data else 0.0
        
        print(f"    ✅ Episode: {len(episode_data)} steps, avg MCTS value: {avg_value:.3f}")
        if line_logger is not None:
            try:
                line_logger.end_episode()
            except Exception as _end_err:
                print(f"   [LineLog] end_episode failed: {_end_err}")
    
    print(f"📈 Collected {len(training_examples)} training examples")
    return training_examples


def test_mcts_implementation():
    """
    Test the MCTS implementation with real Grid2Op environment.
    
    This validates:
    - Environment copying works correctly
    - MCTS search runs without errors  
    - Real rewards are collected properly
    - Action selection is reasonable
    """
    
    print("🧪 Testing Alpha Zero MCTS Implementation")
    print("=" * 50)
    
    try:
        # Create test environment
        print("🌍 Creating test environment...")
        # Create environment with appropriate backend
        if LightSimBackend is not None:
            env = make("l2rpn_case14_sandbox", backend=LightSimBackend())
        else:
            env = make("l2rpn_case14_sandbox")
        print(f"   Environment: {env.name}")
        print(f"   Lines: {env.n_line}")
        
        # Reset environment  
        observation = env.reset()
        print(f"   Initial max line load: {np.max(observation.rho):.3f}")
        
        # Test action building
        print("\n🔧 Testing action building...")
        bus_actions = BusSwitchingActions(env)
        valid_actions = build_valid_actions(observation, env.action_space, bus_actions)
        print(f"   Built {len(valid_actions)} valid actions ({bus_actions.line_switch_actions} line + {bus_actions.bus_switch_actions} bus)")
        
        # Test MCTS search
        print("\n🌲 Testing MCTS search...")
        best_action, action_probs, stats = run_alpha_zero_mcts(
            env,
            neural_network=None,
            num_simulations=20,  # Small number for testing
            config=AGENT_CONFIG,  # Pass config - will read puct_c, gamma, etc.
            bus_actions=bus_actions  # Pass bus actions
        )
        
        print(f"   ✅ MCTS completed successfully!")
        print(f"   Simulations: {stats['simulations']}")
        print(f"   Root visits: {stats['root_visits']}")
        print(f"   Children: {stats['num_children']}")
        print(f"   Action probabilities: {len(action_probs)}")
        
        # Test action execution
        print(f"\n⚡ Testing action execution...")
        obs, reward, done, info = env.step(best_action)
        print(f"   ✅ Action executed!")
        print(f"   Reward: {reward:.3f}")
        print(f"   Done: {done}")
        print(f"   New max line load: {np.max(obs.rho):.3f}")
        
        # Test training data collection
        print(f"\n📊 Testing training data collection...")
        env.reset()  # Reset for clean test
        training_data = collect_training_data(env, num_episodes=2, mcts_simulations=10, config=AGENT_CONFIG, bus_actions=bus_actions)
        print(f"   ✅ Collected {len(training_data)} training examples")
        
        if training_data:
            example = training_data[0]
            print(f"   Example keys: {list(example.keys())}")
            if 'mcts_policy' in example:
                print(f"   MCTS policy: {len(example['mcts_policy'])}")
            elif 'action_probs' in example:
                print(f"   Action probs: {len(example['action_probs'])}")
            else:
                print("   No policy/probs field found")
            print(f"   Reward: {example['reward']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_alpha_zero_agent():
    """
    Full Alpha Zero training with neural network integration.
    
    This creates and trains the neural network using MCTS-generated training data.
    """
    print("🚀 Full Alpha Zero Training with Neural Network")
    print("=" * 60)
    
    # Create environment
    env_name = "rte_case14_realistic" 
    try:
        if LightSimBackend:
            env = make(env_name, backend=LightSimBackend())
        else:
            env = make(env_name)
    except Exception as e:
        print(f"⚠️ Failed to create {env_name}: {e}")
        env = make("l2rpn_case14_sandbox")
    
    # Get initial observation to determine network size
    obs = env.reset()
    
    # Initialize bus switching actions for expanded action space
    try:
        bus_actions = BusSwitchingActions(env)
        num_actions = bus_actions.get_action_space_size()  # 41 actions
        print(f"✅ Bus switching initialized: {num_actions} total actions")
    except Exception as e:
        print(f"⚠️ Bus switching initialization failed: {e}")
        print("   Falling back to line switching only")
        bus_actions = None
        num_lines = len(obs.rho)
        num_actions = num_lines + 1  # All lines + do nothing
    
    # Create neural network
    try:
        input_size = AGENT_CONFIG['input_size']
        neural_network = nn_funcs.create_neural_network(
            input_size=input_size,
            num_actions=num_actions,
            config=AGENT_CONFIG
        )
        print(f"✅ Neural network created: {num_actions} actions (input_size={input_size})")
    except Exception as e:
        print(f"❌ Failed to create neural network: {e}")
        neural_network = None
    
    # Training loop
    training_examples = []
    num_episodes = AGENT_CONFIG.get('num_training_episodes', 5)
    
    for episode in range(num_episodes):
        print(f"\n� Training Episode {episode + 1}/{num_episodes}")
        # Reset environment
        obs = env.reset()
        done = False
        episode_data = []
        while not done:
            # Only run MCTS in critical states for efficiency
            if is_critical_state(obs):
                print(f"⚠️ Critical state detected - running MCTS")
                # Run MCTS with current neural network
                best_action, action_probs, training_data = run_alpha_zero_mcts(
                    env, 
                    neural_network=neural_network,
                    num_simulations=AGENT_CONFIG.get('mcts_simulations', 50),
                    config=AGENT_CONFIG,  # Pass config for t_skipped and t_stopping
                    bus_actions=bus_actions  # Pass bus actions for expanded action space
                )
                # Collect training data
                episode_data.extend(training_data)
                action = best_action
            else:
                # In safe states, just do nothing
                action = env.action_space()
            # Execute action
            obs, reward, done, info = env.step(action)
            if done:
                break
        # Add episode data to training set
        training_examples.extend(episode_data)
        print(f"✅ Episode {episode + 1} complete - collected {len(episode_data)} training examples")
        # Train neural network every few episodes
        if neural_network and len(training_examples) > 20 and (episode + 1) % 2 == 0:
            print(f"🧠 Training neural network on {len(training_examples)} examples...")
            try:
                nn_funcs.train_neural_network(neural_network, training_examples, AGENT_CONFIG)
                print(f"✅ Neural network training complete")
            except Exception as e:
                print(f"❌ Neural network training failed: {e}")
    
    # Save final model
    if neural_network:
        try:
            torch.save(neural_network.state_dict(), 'trained_agent_final.pth')
            print(f"✅ Model saved to trained_agent_final.pth")
        except Exception as e:
            print(f"⚠️ Failed to save model: {e}")
    
    print(f"\n🎉 Alpha Zero Training Complete!")
    print(f"✅ Total training examples collected: {len(training_examples)}")
    return neural_network, training_examples


def main():
    """
    Main function - choose between testing MCTS or full training.
    """
    
    print("🚀 Alpha Zero for Grid2Op Line Switching")
    print("=" * 60)
    print("1. Test MCTS implementation (no neural network)")
    print("2. Full Alpha Zero training (with neural network)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        # Full training
        neural_network, training_examples = train_alpha_zero_agent()
        return True
    else:
        # Just test MCTS
        test_success = test_mcts_implementation()
        return test_success


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready for Neural Network Integration!")
    else:
        print("\n❌ Fix MCTS issues before proceeding")