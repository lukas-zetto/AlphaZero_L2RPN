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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

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
    """

    def __init__(self, parent=None, action=None, prior_prob=0.0, action_idx=None):
        self.parent = parent
        self.action = action
        self.action_idx = action_idx
        self.children = {}

        # MCTS stats
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob

        # Environment-related stats
        self.immediate_reward = 0.0
        self.cumulative_reward = 0.0
        self.max_reachable_steps = 0
        self.is_recovery_node = False

        # Terminal info
        self.is_terminal = False
        self.terminal_reason = None

        # Per-node priors and action references
        self._prior_map = {}
        self._action_map = {}
        self._action_idx_map = {}
        self._priors_initialized = False

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct=1.5, t_skipped=10):
        best_score = -float('inf')
        best_child = None
        for action_str, child in self.children.items():
            q_value = (child.value_sum / child.visit_count) if child.visit_count > 0 else 0.0
            q_value = float(np.clip(q_value, -10.0, 10.0))
            parent_visits = max(1, self.visit_count)
            u_value = (c_puct * child.prior_prob * math.sqrt(parent_visits) / (1 + child.visit_count))
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, valid_actions, action_probs):
        for i, action in enumerate(valid_actions):
            action_str = str(action)
            if action_str not in self.children:
                prob = action_probs[i] if i < len(action_probs) else (1.0 / max(1, len(valid_actions)))
                self.children[action_str] = MCTSNode(parent=self, action=action, prior_prob=prob, action_idx=i)

    def expand_one_child(self, valid_actions, action_probs):
        for i, action in enumerate(valid_actions):
            action_str = str(action)
            if action_str not in self.children:
                prob = action_probs[i] if i < len(action_probs) else (1.0 / max(1, len(valid_actions)))
                new_child = MCTSNode(parent=self, action=action, prior_prob=prob, action_idx=i)
                self.children[action_str] = new_child
                return new_child
        return None

    def set_priors(self, valid_actions, action_probs):
        self._prior_map = {}
        self._action_map = {}
        self._action_idx_map = {}
        for i, action in enumerate(valid_actions):
            action_str = str(action)
            prob = action_probs[i] if i < len(action_probs) else (1.0 / max(1, len(valid_actions)))
            self._prior_map[action_str] = float(prob)
            self._action_map[action_str] = action
            self._action_idx_map[action_str] = i
        self._priors_initialized = True

    def pick_action_with_puct(self, c_puct=1.5, progressive=None):
        if not self._priors_initialized:
            return None, None
        best_score = -float('inf')
        best_key = None
        best_child = None
        parent_visits = max(1, self.visit_count)
        # Progressive widening control
        items = list(self._prior_map.items())
        if progressive is not None:
            alpha, kappa = progressive
            n_allowed = max(1, int(kappa * (parent_visits) ** alpha))
            items = sorted(items, key=lambda kv: kv[1], reverse=True)[:n_allowed]
        for action_str, prior in items:
            child = self.children.get(action_str)
            if child is not None and child.visit_count > 0:
                q_value = float(np.clip(child.value_sum / child.visit_count, -10.0, 10.0))
                n_child = child.visit_count
            else:
                q_value = 0.0
                n_child = 0
            u_value = (c_puct * prior * math.sqrt(parent_visits) / (1 + n_child))
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_key = action_str
                best_child = child
        return best_key, best_child

    def backup_value(self, accumulated_value, max_steps_reached, t_skipped=10):
        self.visit_count += 1
        self.value_sum += accumulated_value
        if max_steps_reached > self.max_reachable_steps:
            self.max_reachable_steps = max_steps_reached
        node_skipped_steps = getattr(self, 'skipped_steps', 0)
        if node_skipped_steps > t_skipped:
            self.is_recovery_node = True

    def get_depth(self):
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth

    def get_action_probs(self, temperature=1.0):
        if not self.children:
            return []
        action_indices = []
        max_steps_list = []
        for action_str, child in self.children.items():
            action_indices.append(child.action_idx if child.action_idx is not None else child.action)
            max_steps_list.append(child.max_reachable_steps)
        max_steps_array = np.array(max_steps_list, dtype=np.float32)
        total = np.sum(max_steps_array)
        if total > 0:
            probs = max_steps_array / total
        else:
            probs = np.ones(len(action_indices)) / len(action_indices)
        return list(zip(action_indices, probs))
        
        
        # COMMENTED OUT: Pure AlphaZero visit count version
        # if not self.children:
        #     return []
        # 
        # action_indices = []
        # visit_counts = []
        # 
        # for action_str, child in self.children.items():
        #     action_indices.append(child.action_idx if child.action_idx is not None else child.action)
        #     visit_counts.append(child.visit_count)
        # 
        # visit_counts = np.array(visit_counts, dtype=np.float32)
        # 
        # if temperature == 0:
        #     best_idx = np.argmax(visit_counts)
        #     probs = np.zeros(len(action_indices))
        #     probs[best_idx] = 1.0
        # else:
        #     visit_counts_temp = visit_counts ** (1.0 / temperature)
        #     total = np.sum(visit_counts_temp)
        #     if total > 0:
        #         probs = visit_counts_temp / total
        #     else:
        #         probs = np.ones(len(action_indices)) / len(action_indices)
        # 
        # return list(zip(action_indices, probs))


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
                        max_depth=20, gamma=None, early_stop_recovery=True, config=None, bus_actions=None, action_catalog=None, verbose=False):
    """
    Pure Alpha Zero MCTS with neural network leaf evaluation only.
    
    This implements the true AlphaZero approach:
    - Executes actual Grid2Op actions during tree traversal
    - Expands one leaf node per simulation
    - Evaluates leaves using ONLY neural network (no rollouts)
    - Backs up neural network values with discounted step rewards
    
    Key Features:
    - Pure neural network leaf evaluation (no Monte Carlo rollouts)
    - Real Grid2Op environment execution for tree traversal
    - Proper discounted reward backpropagation
    - Recovery node detection for early stopping
    
    Parameters:
    -----------
    env : grid2op.Environment
        Current Grid2Op environment (will be copied for simulations)
    neural_network : torch.nn.Module, REQUIRED
        Neural network for policy/value predictions (required for leaf evaluation)
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
    
    # Get max_depth from config if not specified
    if max_depth is None or max_depth == 20:  # 20 is the default parameter value
        max_depth = config.get('max_depth', 20)
        
    t_skipped = config.get('t_skipped', 200)
    t_stopping = config.get('t_stopping', 10)
    
    # Get current observation and build action space
    observation = env.get_obs()
    valid_actions = build_valid_actions(observation, env.action_space, bus_actions, action_catalog)
    
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
                full_action_space_size = action_catalog.size + 1  # Catalog + do-nothing
            elif bus_actions is not None:
                full_action_space_size = bus_actions.get_action_space_size()
            else:
                full_action_space_size = 21  # Line switching only
            policy_probs, root_value = nn_funcs.neural_network_forward(neural_network, observation, full_action_space_size)
            # Truncate to only valid actions for MCTS (should be same size if catalog used)
            policy_probs = policy_probs[:len(valid_actions)]
        except:
            policy_probs = [1.0 / len(valid_actions)] * len(valid_actions)
    else:
        policy_probs = [1.0 / len(valid_actions)] * len(valid_actions)
    
    # Add Dirichlet noise at root for exploration (AlphaZero technique)
    dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
    dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
    if len(valid_actions) > 1:
        noise = np.random.dirichlet([dirichlet_alpha] * len(valid_actions))
        policy_probs = (1 - dirichlet_epsilon) * np.array(policy_probs) + dirichlet_epsilon * noise
        policy_probs = policy_probs.tolist()
    
    # Root expansion behavior:
    # If pre_expand_root is True, expand all root children immediately (AlphaZero-style root init).
    # If False (default), do NOT pre-expand root; expand lazily when the root is selected during traversal.
    if config.get('pre_expand_root', False):
        # Expand root with initial policy over ALL valid actions
        # This allows priors and Dirichlet noise to guide selection among all root actions.
        root.expand(valid_actions, policy_probs)
    
    # MCTS Simulations with REAL Grid2Op environment execution
    recovery_nodes_found = 0
    max_episode_steps_reached = 0
    
    # Get the current observation to use for simulation
    # IMPORTANT: env.get_obs() may return stale data if action catalog was just built
    # (build_action_catalog calls env.reset() internally)
    # In training loop, the catalog should be built ONCE before the episode loop
    if hasattr(env, '_obs') and env._obs is not None:
        current_obs = env._obs
    else:
        current_obs = env.get_obs()
    starting_step = getattr(current_obs, 'current_step', 0)  # Track where we started
    
    for sim in range(num_simulations):
        # Full tree analysis every 100 simulations
        if verbose and sim > 0 and sim % 100 == 0:
            print(f"\n{'='*80}")
            print(f"📊 MCTS Progress Report - Simulation {sim}/{num_simulations}")
            print(f"{'='*80}")
            print(f"Root visits: {root.visit_count}, Children: {len(root.children)}")
            print(f"Recovery nodes found: {recovery_nodes_found}")
            
            # Count nodes at each depth level recursively up to the configured max_depth
            # Use the outer scope `max_depth` so logs follow the configured MCTS depth (e.g. 20)
            def count_nodes_at_depth(node, max_depth=max_depth):
                """Recursively count nodes at each depth"""
                depth_counts = {i: 0 for i in range(max_depth + 1)}

                def traverse(n, d):
                    if d <= max_depth:
                        depth_counts[d] += 1
                        for child in n.children.values():
                            traverse(child, d + 1)

                traverse(node, 0)
                return depth_counts
            
            depth_counts = count_nodes_at_depth(root)
            total_nodes = sum(depth_counts.values())
            
            print(f"\nTree structure (nodes at each depth):")
            # Show counts for all depths from 0..max_depth (inclusive)
            for depth in range(max_depth + 1):
                count = depth_counts.get(depth, 0)
                print(f"  Depth {depth}: {count} nodes")
            print(f"  Total nodes in tree: {total_nodes}")
            
            # Count recovery nodes in entire tree
            def count_recovery_nodes(node):
                """Count recovery nodes recursively"""
                count = 0
                if getattr(node, 'is_recovery_node', False):
                    count += 1
                for child in node.children.values():
                    count += count_recovery_nodes(child)
                return count
            
            recovery_in_tree = count_recovery_nodes(root)
            print(f"  Recovery nodes in tree: {recovery_in_tree}")
            
            # Analyze max_steps distribution
            if root.children:
                max_steps_dist = {}
                recovery_count = 0
                for child in root.children.values():
                    steps = child.max_reachable_steps
                    max_steps_dist[steps] = max_steps_dist.get(steps, 0) + 1
                    if getattr(child, 'is_recovery_node', False):
                        recovery_count += 1
                
                unique_steps = len(max_steps_dist)
                print(f"\nmax_reachable_steps: {unique_steps} unique values among {len(root.children)} children")
                sorted_dist = sorted(max_steps_dist.items(), reverse=True)[:5]
                print(f"  Top 5 distributions: {sorted_dist}")
                print(f"  Recovery children: {recovery_count}/{len(root.children)}")
                
                # Show top 3 actions
                sorted_children = sorted(root.children.items(), 
                                       key=lambda x: (x[1].max_reachable_steps, x[1].visit_count), 
                                       reverse=True)
                print(f"\nTop 3 actions by max_reachable_steps:")
                for i, (action_str, child) in enumerate(sorted_children[:3]):
                    action_idx = getattr(child, 'action_idx', '?')
                    skipped = getattr(child, 'skipped_steps', 0)
                    recovery = '✓ RECOVERY' if getattr(child, 'is_recovery_node', False) else ''
                    print(f"  {i+1}. Action {action_idx}: max_steps={child.max_reachable_steps}, "
                          f"visits={child.visit_count}, skipped={skipped} {recovery}")
            print(f"{'='*80}\n")
        
        # Selection + Expansion Phase (true AlphaZero traversal with PUCT over expanded/unexpanded)
        node = root
        path = [root]
        depth = 0
        sim_obs = current_obs
        total_skipped_steps = 0
        new_leaf = None
        
        while depth < max_depth:
            # Build legal actions for the current simulated observation
            try:
                current_actions = build_valid_actions(sim_obs, env.action_space, bus_actions, action_catalog)
            except Exception as e:
                # If we cannot build actions here, stop traversal
                break

            if len(current_actions) <= 1:
                # Only do-nothing available; stop traversal
                break

            # Get policy (NN or uniform) for this node
            if neural_network is not None:
                try:
                    if action_catalog is not None:
                        full_action_space_size = action_catalog.size + 1
                    elif bus_actions is not None:
                        full_action_space_size = bus_actions.get_action_space_size()
                    else:
                        full_action_space_size = 21
                    expansion_policy, _ = nn_funcs.neural_network_forward(neural_network, sim_obs, full_action_space_size)
                    expansion_policy = expansion_policy[:len(current_actions)]
                except Exception:
                    expansion_policy = [1.0 / len(current_actions)] * len(current_actions)
            else:
                expansion_policy = [1.0 / len(current_actions)] * len(current_actions)

            # Ensure priors exist at this node
            if not getattr(node, '_priors_initialized', False):
                try:
                    node.set_priors(current_actions, expansion_policy)
                except Exception:
                    # If persisting priors fails, abort traversal this simulation
                    break

            # PUCT over all actions: if unexpanded selected -> expand; else descend
            # Progressive widening parameters from config (if any)
            pw = None
            if config is not None and config.get('progressive_widening', True):
                pw = (config.get('pw_alpha', 0.5), config.get('pw_kappa', 2.0))
            chosen_key, chosen_child = node.pick_action_with_puct(c_puct, progressive=pw)

            if chosen_key is None:
                # Could not select action (should be rare); stop traversal
                break

            if chosen_child is None:
                # Unexpanded move chosen: expand it here and evaluate as leaf
                try:
                    action = node._action_map[chosen_key]
                    prior = node._prior_map.get(chosen_key, 1.0 / max(1, len(current_actions)))
                    action_idx = node._action_idx_map.get(chosen_key, None)
                    new_leaf = MCTSNode(parent=node, action=action, prior_prob=prior, action_idx=action_idx)
                    node.children[chosen_key] = new_leaf
                except Exception:
                    new_leaf = None
                # Stop traversal at newly created leaf
                node = new_leaf if new_leaf is not None else node
                path.append(node) if node is not None else None
                break
            else:
                # Descend into already-expanded child and update simulated observation
                path.append(chosen_child)
                depth += 1
                action_to_apply = chosen_child.action
                try:
                    next_obs, _, done_flag, _ = sim_obs.simulate(action_to_apply)
                    # Skip safe states until next critical or limit
                    skipped_steps = 0
                    cur = next_obs
                    while not is_critical_state(cur) and skipped_steps < 100:
                        try:
                            cur_next, _, cur_done, _ = cur.simulate(env.action_space())
                            if cur_done:
                                done_flag = True
                                break
                            cur = cur_next
                            skipped_steps += 1
                        except Exception as e:
                            if "NoForecastAvailable" not in str(e):
                                print(f"[MCTS Skip Exception] {e}")
                            break
                    total_skipped_steps += skipped_steps
                    sim_obs = cur
                except Exception as e:
                    if "NoForecastAvailable" not in str(e):
                        print(f"[MCTS Replay Exception] {e}")
                    break
                node = chosen_child
        
        # Expansion and Rollout Phase
        episode_reward = 0.0
        
        # Calculate how far we've gotten in time
        # depth = number of critical nodes traversed
        # total_skipped_steps = safe states skipped along the way
        
        if node.is_terminal and hasattr(node, 'terminal_reason') and node.terminal_reason in ['done', 'illegal', 'exception']:
            max_steps_in_this_simulation = starting_step  # Failed immediately
        else:
            max_steps_in_this_simulation = starting_step + depth + total_skipped_steps
        
        if not node.is_terminal and depth < max_depth:
            # If we created a new leaf above, evaluate it; otherwise, no expansion this sim
            try:
                current_sim_obs = sim_obs
                if new_leaf is not None:
                    # Evaluate the leaf state AFTER executing the action with neural network
                    try:
                        # Apply new action on the current simulated observation
                        leaf_sim_obs, _, leaf_done, _ = current_sim_obs.simulate(new_leaf.action)
                        if leaf_done:
                            new_leaf.is_terminal = True
                            new_leaf.terminal_reason = 'done'
                            new_leaf.observation = leaf_sim_obs
                            new_leaf.skipped_steps = 0
                        else:
                            # Now apply safe state skipping starting from leaf_sim_obs
                            leaf_skipped_steps = 0
                            current_leaf_obs = leaf_sim_obs
                            
                            while not is_critical_state(current_leaf_obs):
                                try:
                                    # Skip safe states by simulating do-nothing
                                    next_obs, _, done, _ = current_leaf_obs.simulate(env.action_space())
                                    if done:
                                        new_leaf.is_terminal = True
                                        new_leaf.terminal_reason = 'done'
                                        break
                                    current_leaf_obs = next_obs
                                    leaf_skipped_steps += 1
                                    
                                    # Safety limit
                                    if leaf_skipped_steps >= 100:
                                        break
                                except Exception as skip_err:
                                    # If skipping fails, stop and use current state
                                    break
                            
                            new_leaf.observation = current_leaf_obs  # Store state AFTER skipping
                            new_leaf.skipped_steps = leaf_skipped_steps
                            
                            # Check if leaf state is a recovery state
                            if is_grid_in_recovery(current_leaf_obs):
                                recovery_nodes_found += 1
                                new_leaf.is_recovery_node = True
                    except Exception as e:
                        # If action replay or simulation fails, mark as terminal
                        if "NoForecastAvailable" not in str(e):
                            print(f"[MCTS Expansion Failed at depth {depth}] {type(e).__name__}: {e}")
                        if new_leaf is not None:
                            new_leaf.is_terminal = True
                            new_leaf.terminal_reason = 'simulation_failed'
                            new_leaf.observation = None
                            new_leaf.skipped_steps = 0
                        episode_reward = -5.0
                        max_steps_in_this_simulation = starting_step + depth + total_skipped_steps
                    else:
                        # Set episode reward to 0 for neural network evaluation (no rollout)
                        episode_reward = 0.0
                        # Expanded one more node; account for depth so far, skipped steps, and leaf skipped steps
                        max_steps_in_this_simulation = starting_step + depth + total_skipped_steps + 1 + getattr(new_leaf, 'skipped_steps', 0)
                else:
                    # No leaf to expand - use immediate reward
                    episode_reward = 0.0
                    max_steps_in_this_simulation = starting_step + depth + total_skipped_steps
            except Exception as e:
                episode_reward = -1.0
                max_steps_in_this_simulation = starting_step + depth
        else:
            # Terminal node - use immediate reward
            episode_reward = node.immediate_reward if hasattr(node, 'immediate_reward') else 0.0
        
        # Update maximum episode steps reached
        max_episode_steps_reached = max(max_episode_steps_reached, max_steps_in_this_simulation)
        
        # 🔑 ALPHAZERO BACKUP: Use ONLY neural network evaluation for leaf values
        
        # Get leaf value from neural network ONLY (no rollout fallback)
        leaf_node = path[-1] if path else None
        leaf_value = 0.0  # Default if no neural network
        
        if leaf_node and neural_network is not None:
            try:
                # Get neural network value for leaf state AFTER action execution
                leaf_obs = getattr(leaf_node, 'observation', None)
                if leaf_obs is not None:
                    # Use neural network to evaluate the leaf state
                    _, leaf_value = nn_funcs.neural_network_forward(neural_network, leaf_obs, 1)
                else:
                    # If no observation stored, evaluate current state
                    leaf_value = 0.0
            except Exception as e:
                # If neural network evaluation fails, re-raise with clear context
                raise RuntimeError(f"Neural network evaluation failed for leaf node: {e}") from e
        elif neural_network is None and config is not None and config.get('diagnostic_leaf_value', False):
            # Optional diagnostic heuristic leaf value when NN is absent
            leaf_value = float(max_steps_in_this_simulation - starting_step)
            leaf_value = max(min(leaf_value, 1000.0), -1000.0)
        elif leaf_node and leaf_node.is_terminal:
            # Terminal nodes get immediate reward only
            leaf_value = getattr(leaf_node, 'immediate_reward', 0.0)
        else:
            # No neural network available - use zero value
            leaf_value = 0.0
        
        # Backup from leaf to root: leaf_value + Σ(γ^k * step_rewards)
        accumulated_value = leaf_value
        
        # Store the accumulated values for each node (for training data)
        node_values = {}
        
        # CRITICAL: Propagate max_steps_reached upward through the tree
        # max_steps_in_this_simulation is the deepest timestep reached in THIS simulation
        # As we backup, ALL parent nodes learn that one of their descendants reached this timestep
        # Each node's max_reachable_steps = max over ALL simulations through that node
        propagating_max_steps = max_steps_in_this_simulation
        
        for i, node in enumerate(reversed(path)):
            # Add immediate reward for this step (standard Bellman backup)
            if i > 0:  # Skip leaf node (already has its value)
                step_reward = getattr(node, 'immediate_reward', 0.0)
                # FIXED: Standard discounted return formula (was causing value explosion!)
                accumulated_value = step_reward + gamma * accumulated_value
                # Clip to prevent extreme values
                # accumulated_value = np.clip(accumulated_value, -20.0, 20.0)
            
            # Store the accumulated value for this node
            node_values[id(node)] = accumulated_value
            
            # Backup: update visit count, value sum, and max_reachable_steps
            # The max propagates up - parent learns the max any child can reach
            node.backup_value(accumulated_value, propagating_max_steps, t_skipped)
        
        # Early stopping if enough recovery found
        if early_stop_recovery and recovery_nodes_found >= t_stopping:
            break
    
    # Action Selection: Choose action using MCTS policy with configured temperature
    training_temperature = AGENT_CONFIG.get('temperature', 1.5)
    action_probs = root.get_action_probs(temperature=training_temperature)
    
    if action_probs:
        # action_probs contains (action_idx_or_obj, prob) tuples
        # action_idx_or_obj can be either integer index (catalog mode) or action object (legacy)
        best_child = None
        best_score = (-1, -1, -1)  # (max_steps, visit_count, prob)
        
        for action_str, child in root.children.items():
            # Find this child in action_probs by matching indices or action objects
            child_prob = 0.0
            for action_idx_or_obj, prob in action_probs:
                # Match by index (catalog mode) or by action string (legacy mode)
                if isinstance(action_idx_or_obj, int):
                    # Catalog mode: compare indices
                    if action_idx_or_obj == child.action_idx:
                        child_prob = prob
                        break
                else:
                    # Legacy mode: compare action objects by string
                    if str(action_idx_or_obj) == str(child.action):
                        child_prob = prob
                        break
            
            # Use max_reachable_steps as primary criterion (how far in time can we get)
            # This now properly represents the furthest timestep reached, not just tree depth
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
        'root_value': root_value,  # Add the MCTS-calculated value for training
        'root_node': root,  # Expose root for diagnostics
        'starting_step': starting_step  # Expose starting timestep
    }
    
    # Debugging output for MCTS tree analysis
    if verbose:
        print(f"\n{'='*80}")
        print(f"🌳 MCTS Tree Analysis (step {starting_step})")
        print(f"{'='*80}")
        print(f"   Simulations: {sim}/{num_simulations}")
        print(f"   Recovery nodes: {recovery_nodes_found}")
        print(f"   Root visits: {root.visit_count}")
        print(f"   Children: {len(root.children)}")
        
        # Count tree depth
        grandchildren = sum(1 for child in root.children.values() if child.children)
        great_grandchildren = sum(len(gc.children) for child in root.children.values() for gc in child.children.values())
        print(f"   Tree depth - Grandchildren: {grandchildren}, Great-grandchildren: {great_grandchildren}")
        
        # Analyze action distribution
        if root.children:
            max_steps_dist = {}
            for child in root.children.values():
                steps = child.max_reachable_steps
                max_steps_dist[steps] = max_steps_dist.get(steps, 0) + 1
            
            unique_steps = len(max_steps_dist)
            print(f"   Unique max_steps values: {unique_steps}/{len(root.children)}")
            if unique_steps < len(root.children) * 0.2:
                print(f"   ⚠️ Low differentiation - distribution: {dict(sorted(max_steps_dist.items(), reverse=True)[:5])}")
            
            # Show top actions
            sorted_children = sorted(root.children.items(), 
                                   key=lambda x: (x[1].max_reachable_steps, x[1].visit_count), 
                                   reverse=True)
            print(f"\n   📋 Top 5 actions by max_reachable_steps:")
            for i, (action_str, child) in enumerate(sorted_children[:5]):
                action_idx = getattr(child, 'action_idx', '?')
                recovery = '✓' if getattr(child, 'is_recovery_node', False) else ''
                print(f"      {i+1}. Action {action_idx}: max_steps={child.max_reachable_steps}, "
                      f"visits={child.visit_count}, Q={child.value_sum/child.visit_count if child.visit_count > 0 else 0:.3f} {recovery}")
        
        # Policy entropy analysis
        if action_probs:
            probs = [p for _, p in action_probs]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            print(f"\n   🎯 Policy entropy: {entropy:.3f}/{max_entropy:.3f} ({100*entropy/max_entropy:.1f}%)")
            if entropy / max_entropy > 0.95:
                print(f"      ⚠️ Nearly uniform policy (high entropy)")
        
        print(f"{'='*80}\n")
    
    return best_action, action_probs, stats


def collect_training_data(env, num_episodes=5, mcts_simulations=30, neural_network=None, config=None, bus_actions=None, line_logger=None, chronic_id=None, action_catalog=None):
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
    bus_actions : BusSwitchingActions, optional
        LEGACY bus switching handler (only used if action_catalog is None)
    action_catalog : ActionCatalog, optional
        NEW catalog-based action space (preferred over bus_actions)
        
    Returns:
    --------
    training_examples : list
        List of (observation, action_probs, reward) for neural network training
    """
    
    training_examples = []
    
    # Determine action space size: prefer catalog over legacy bus_actions
    if action_catalog is not None:
        num_actions = action_catalog.size + 1  # Catalog actions + do-nothing
        print(f"   Using catalog action space: {num_actions} actions")
        bus_actions = None  # Ignore legacy if catalog present
    elif bus_actions is not None:
        try:
            # Recreate bus_actions with current environment to avoid env mismatch
            from bus_switching_actions import BusSwitchingActions
            bus_actions = BusSwitchingActions(env)
            num_actions = bus_actions.get_action_space_size()
            print(f"   Using legacy bus_actions: {num_actions} actions")
        except Exception as e:
            print(f"⚠️ Bus switching recreation failed: {e}")
            bus_actions = None
            num_actions = 21  # Default line switching only
    else:
        num_actions = 21  # Default line switching only
        print(f"   Using line switching only: {num_actions} actions")
    
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
        
        # DON'T skip to first critical state - go through ENTIRE chronic!
        # Collect data at EVERY critical moment throughout the timeline
        step = 0
        done = False
        max_episode_steps = env.max_episode_duration()
        
        print(f"    Collecting data throughout entire chronic (max {max_episode_steps} steps)")
        
        # Collect training data from critical states, continue episode until done
        critical_steps = 0
        while not done and step < max_episode_steps:
            
            # Track if we ran MCTS this step (before executing action)
            ran_mcts = False
            observation_before_action = observation  # Save state BEFORE action
            
            if is_critical_state(observation):
                critical_steps += 1
                # Run MCTS search for critical states
                ran_mcts = True
                try:
                    # Prefer verbose diagnostics when supported; fall back gracefully if not
                    try:
                        best_action, action_probs, stats = run_alpha_zero_mcts(
                            env,
                            neural_network=neural_network,  # Use the neural network passed to function
                            num_simulations=mcts_simulations,
                            config=config,  # Pass config - this will read puct_c, gamma, etc.
                            bus_actions=bus_actions,  # Pass bus actions for expanded action space
                            action_catalog=action_catalog,  # Pass catalog (priority over bus_actions)
                            verbose=True  # Enable debugging output when available
                        )
                    except TypeError as te:
                        # Backward compatibility: some variants don't accept 'verbose'
                        if "unexpected keyword argument 'verbose'" in str(te):
                            best_action, action_probs, stats = run_alpha_zero_mcts(
                                env,
                                neural_network=neural_network,
                                num_simulations=mcts_simulations,
                                config=config,
                                bus_actions=bus_actions,
                                action_catalog=action_catalog
                            )
                        else:
                            raise
                    
                except Exception as e:
                    print(f"    ⚠️ MCTS failed at step {step}: {e}, using do-nothing and continuing")
                    # Don't break! Use do-nothing and continue to next step
                    best_action = env.action_space()
                    # Create uniform do-nothing probs
                    valid_actions = build_valid_actions(observation, env.action_space, bus_actions, action_catalog)
                    action_probs = [(0, 1.0)] + [(i, 0.0) for i in range(1, len(valid_actions))]
                    stats = {'simulations': 0, 'mcts_failed': True}
                    ran_mcts = False  # Don't collect this as training data
            else:
                # Use do-nothing for safe states
                best_action = env.action_space()  # Do nothing action
                # Create action_probs with same format as MCTS (indices with do-nothing getting prob 1.0)
                valid_actions = build_valid_actions(observation, env.action_space, bus_actions, action_catalog)
                do_nothing_probs = []
                for idx, action in enumerate(valid_actions):
                    if str(action) == str(best_action):  # This is do-nothing (index 0)
                        do_nothing_probs.append((idx, 1.0))
                    else:
                        do_nothing_probs.append((idx, 0.0))
                action_probs = do_nothing_probs
                stats = {'simulations': 0, 'safe_state': True}
            
            # Execute the action
            try:
                # Print which action was selected - find the index in valid_actions
                action_idx = None
                action_prob = None
                
                # Build a mapping from action strings to indices in valid_actions
                valid_actions_list = build_valid_actions(observation, env.action_space, bus_actions, action_catalog)
                action_to_idx = {str(action): idx for idx, action in enumerate(valid_actions_list)}
                best_action_str = str(best_action)
                
                if best_action_str in action_to_idx:
                    action_idx = action_to_idx[best_action_str]
                    # Find the probability for this action index
                    for idx_or_action, prob in action_probs:
                        if idx_or_action == action_idx:
                            action_prob = prob
                            break
                
                # Action logging removed for cleaner output
                
                # Log line loads before action (only for critical states where MCTS ran)
                if ran_mcts:
                    rho_before = observation.rho
                    max_rho_before = np.max(rho_before)
                    top_lines_before = np.argsort(rho_before)[-3:][::-1]  # Top 3 loaded lines
                
                observation_after_action = observation  # Save for comparison
                observation, reward, done, info = env.step(best_action)
                
                # Log line loads after action (only for critical states where MCTS ran)
                if ran_mcts:
                    rho_after = observation.rho
                    max_rho_after = np.max(rho_after)
                    top_lines_after = np.argsort(rho_after)[-3:][::-1]  # Top 3 loaded lines
                    
                    # Print the comparison
                    rho_change = max_rho_after - max_rho_before
                    change_symbol = "📈" if rho_change > 0 else "📉" if rho_change < 0 else "➡️"
                    print(f"    {change_symbol} Line loads: max_rho {max_rho_before:.3f} → {max_rho_after:.3f} (Δ {rho_change:+.3f})")
                    print(f"       Top 3 before: {[(int(i), f'{rho_before[i]:.3f}') for i in top_lines_before]}")
                    print(f"       Top 3 after:  {[(int(i), f'{rho_after[i]:.3f}') for i in top_lines_after]}")
                
                # Debug why episode ended
                if done:
                    if step < max_episode_steps - 10:  # Episode ended early
                        print(f"    ⚠️ Episode ended early at step {step}/{max_episode_steps}")
                        print(f"       Reason: is_illegal={info.get('is_illegal', False)}, is_ambiguous={info.get('is_ambiguous', False)}")
                        print(f"       exception={info.get('exception', None)}")
                
                if line_logger is not None:
                    try:
                        line_logger.update(observation)
                    except Exception:
                        pass
                
                # Get the MCTS-calculated value for this state (the accumulated discounted return)
                mcts_value = stats.get('root_value', 0.0)
                # NO CLIPPING - let the network learn the true value range
                
                # Collect training examples whenever we ran MCTS (regardless of resulting state)
                if ran_mcts:
                    # Convert action_probs to fixed-size policy vector 
                    # This ensures consistent shape regardless of which lines are on cooldown
                    probs_only = action_probs_to_fixed_policy(action_probs, observation_before_action, num_actions=num_actions, bus_actions=bus_actions, action_catalog=action_catalog)
                    
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
            
                # Check termination conditions - ONLY break on illegal/ambiguous, NOT on done
                # This allows us to continue collecting data even if action causes issues
                if info.get('is_illegal', False) or info.get('is_ambiguous', False):
                    print(f"    ⚠️ Invalid action at step {step}, using do-nothing and continuing")
                    # Don't break - try to recover with do-nothing
                    observation, reward, done, info = env.step(env.action_space())
                    if line_logger is not None:
                        try:
                            line_logger.update(observation)
                        except Exception:
                            pass
                
                # If done naturally (end of chronic data or grid failure), finish the episode gracefully
                if done:
                    print(f"    Episode done at step {step} (grid failure or chronic ended)")
                    break
                    
            except Exception as e:
                print(f"    ⚠️ Action execution failed at step {step}: {e}, continuing with do-nothing")
                # Don't break - try to continue
                try:
                    observation, reward, done, info = env.step(env.action_space())
                    if line_logger is not None:
                        try:
                            line_logger.update(observation)
                        except Exception:
                            pass
                except:
                    print(f"    ❌ Cannot recover, ending episode")
                    break
            
            step += 1
        
        # Calculate average episode value (from MCTS calculations)
        avg_value = np.mean([ex['value'] for ex in episode_data]) if episode_data else 0.0
        
        print(f"    ✅ Episode: {len(episode_data)} MCTS steps collected (saw {critical_steps} critical states total), avg MCTS value: {avg_value:.3f}")
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
        # Determine input size dynamically from environment
        try:
            dummy_encoded = nn_funcs.encode_observation_simple(obs)
            input_size = len(dummy_encoded)
            print(f"🔍 AGENT: Dynamic input_size={input_size}")
        except:
            input_size = AGENT_CONFIG.get('input_size', 83)  # Fallback
            print(f"🔍 AGENT: Fallback input_size={input_size}")
        neural_network = nn_funcs.create_neural_network(
            input_size=input_size,  # Now taken from config
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
                best_action, action_probs, stats = run_alpha_zero_mcts(
                    env, 
                    neural_network=neural_network,
                    num_simulations=AGENT_CONFIG.get('mcts_simulations', 50),
                    config=AGENT_CONFIG,  # Pass config for t_skipped and t_stopping
                    bus_actions=bus_actions  # Pass bus actions for expanded action space
                )
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
    Main function - automatically run full Alpha Zero training.
    """
    
    print("🚀 Alpha Zero for Grid2Op Line Switching")
    print("=" * 60)
    print("Starting Full Alpha Zero training (with neural network)")
    print()
    
    # Run the REAL training function with 903 chronics
    train_mcts_with_auto_reconnect()
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready for Neural Network Integration!")
    else:
        print("\n❌ Fix MCTS issues before proceeding")