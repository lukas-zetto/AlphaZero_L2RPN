"""
AlphaZero-style MCTS for Grid2Op - Version 2
Properly handles multi-step lookahead by maintaining environment copies.

Key difference from v1: Instead of chaining observation.simulate() calls (which fails),
we maintain environment copies at each node that can be stepped forward.
"""

import numpy as np
import copy
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import defaultdict
import sys
sys.path.append('/workspace/src')
from rewards.custom_reward import MyCustomReward

# Initialize reward function
custom_reward_fn = MyCustomReward()


class MCTSNodeV2:
    """
    MCTS node with environment state management for multi-step simulation.
    
    Each node stores an environment copy that can be stepped from this state.
    This allows us to explore depth > 1 despite Grid2Op's simulate() limitation.
    
    SAFE STATE SKIPPING: Only critical states (max_rho > threshold) create nodes.
    Safe states are skipped with do-nothing actions, and we track steps_to_reach.
    """
    
    def __init__(self, env, observation, parent=None, action_taken=None, 
                 action_idx=None, edge_reward=0.0, steps_to_reach=0):
        """
        Args:
            env: Grid2Op environment copy at this state (can be stepped forward)
            observation: Grid2Op observation at this node
            parent: Parent node
            action_taken: Action object that led here from parent
            action_idx: Index of action in action catalog
            edge_reward: Immediate reward r(s,a) from parent to this node
            steps_to_reach: Number of steps (including do-nothings) to reach this node
        """
        self.env = env  # Environment copy at this state
        self.observation = observation
        self.parent = parent
        self.action_taken = action_taken
        self.action_idx = action_idx
        self.edge_reward = edge_reward
        self.steps_to_reach = steps_to_reach  # Total steps including do-nothing skips
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action_idx -> MCTSNodeV2
        self.unexpanded_actions = []  # List of action indices not yet tried
        
        # State flags
        self.is_terminal = False
        self.max_reachable_steps = 0
        
        # Prior probabilities (from neural network or uniform)
        self.action_priors = {}  # action_idx -> prior probability
    
    def is_fully_expanded(self):
        return len(self.unexpanded_actions) == 0
    
    def value(self):
        """Average value Q(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_depth(self):
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth
    
    def __repr__(self):
        return (f"NodeV2(depth={self.get_depth()}, visits={self.visit_count}, "
                f"Q={self.value():.3f}, children={len(self.children)}, "
                f"unexpanded={len(self.unexpanded_actions)})")


def puct_score(node: MCTSNodeV2, child_idx: int, c_puct: float, 
               parent_value: float = 0.0, depth_bonus: float = 0.05,
               virtual_loss_weight: float = 0.5) -> float:
    """
    Calculate PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child) + depth_bonus
    
    For unexpanded actions, use parent's Q-value instead of 0.
    Small depth_bonus to slightly encourage exploration without causing narrow trees.
    Virtual loss: Penalize frequently visited nodes to encourage width.
    """
    if child_idx in node.children:
        child = node.children[child_idx]
        q_value = child.value()
        n_child = child.visit_count
        
        # Virtual loss: penalize nodes with many visits to encourage exploring other children
        # This helps create wider trees instead of narrow deep paths
        parent_visits = max(1, node.visit_count)
        visit_ratio = n_child / parent_visits
        virtual_loss = -virtual_loss_weight * visit_ratio  # Penalize if this child dominates visits
        
        # Small depth bonus for nodes with no children (encourages going deeper)
        if len(child.children) == 0 and not child.is_terminal:
            depth_reward = depth_bonus
        else:
            depth_reward = 0.0
    else:
        # Unexpanded action: use parent's value as estimate
        q_value = parent_value
        n_child = 0
        virtual_loss = 0.0
        depth_reward = depth_bonus * 0.5  # Small bonus for unexpanded
    
    prior = node.action_priors.get(child_idx, 1.0 / len(node.action_priors))
    exploration = c_puct * prior * np.sqrt(node.visit_count) / (1 + n_child)
    
    return q_value + exploration + depth_reward + virtual_loss


def select_child(node: MCTSNodeV2, c_puct: float, epsilon: float = 0.0) -> Tuple[int, bool]:
    """
    Select child using PUCT with epsilon-greedy exploration.
    
    With probability epsilon, selects a random action.
    With probability 1-epsilon, selects best action according to PUCT.
    
    Returns (action_idx, is_expanded).
    
    If all actions expanded, select best child.
    Otherwise, select from unexpanded actions with highest PUCT.
    """
    if node.is_terminal:
        return None, False
    
    # Epsilon-greedy: random action with probability epsilon
    if epsilon > 0 and np.random.random() < epsilon:
        # Random selection
        if len(node.unexpanded_actions) > 0:
            # Random from unexpanded
            action_idx = np.random.choice(node.unexpanded_actions)
            return action_idx, False
        elif len(node.children) > 0:
            # Random from expanded
            action_idx = np.random.choice(list(node.children.keys()))
            return action_idx, True
        else:
            return None, False
    
    # Standard PUCT selection
    parent_q = node.value()
    
    # If we have unexpanded actions, choose from them
    if len(node.unexpanded_actions) > 0:
        best_score = -np.inf
        best_action = None
        
        for action_idx in node.unexpanded_actions:
            score = puct_score(node, action_idx, c_puct, parent_q)
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action, False  # Not yet expanded
    
    # All actions expanded, select best child (skip terminal children)
    elif len(node.children) > 0:
        best_score = -np.inf
        best_action = None
        
        for action_idx, child in node.children.items():
            # Skip terminal children - can't explore further
            if child.is_terminal:
                continue
            
            score = puct_score(node, action_idx, c_puct, parent_q)
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action, True  # Already expanded
    
    else:
        # No actions available (shouldn't happen unless terminal)
        return None, False


def skip_safe_states(env_copy, max_steps=100, critical_threshold=0.90):
    """
    Skip through safe states with do-nothing actions until we hit:
    1. A critical state (max_rho > threshold), OR
    2. A terminal state (done=True), OR  
    3. Max steps reached
    
    Returns:
        (final_env, final_obs, steps_taken, done, accumulated_reward)
    """
    steps = 0
    done = False
    accumulated_reward = 0.0
    
    try:
        obs = env_copy.get_obs()
    except Exception as e:
        # Environment not initialized (terminal state)
        return env_copy, None, 0, True, 0.0
    
    while steps < max_steps and not done:
        max_rho = obs.rho.max()
        
        # Stop if we reached a critical state
        if max_rho > critical_threshold:
            break
        
        # Take do-nothing action
        obs, reward, done, info = env_copy.step(env_copy.action_space())
        accumulated_reward += reward
        steps += 1
    
    return env_copy, obs, steps, done, accumulated_reward


def expand_node(node: MCTSNodeV2, action_catalog, action_idx: int,
                critical_threshold: float = 0.90,
                penalty_for_failure: float = -5.0,
                auto_reconnect: bool = True,
                max_reconnections: int = 1,
                prefilter_rho_increase: float = 0.20) -> Optional[MCTSNodeV2]:
    """
    Expand a new child by taking action_idx in node's environment.
    
    Returns new child node, or None if action fails.
    
    CRITICAL: Uses env.step() on node's environment copy, NOT observation.simulate()
    This allows us to go depth > 1.
    
    Args:
        auto_reconnect: If True, automatically try to reconnect disconnected lines
        max_reconnections: Maximum number of lines to reconnect per action
        prefilter_rho_increase: Maximum allowed increase in max_rho (e.g., 0.20 = 20%)
    """
    if action_idx >= len(action_catalog.actions):
        return None
    
    # Don't expand from terminal nodes
    if node.is_terminal:
        return None
    
    # Don't expand if node has no environment (shouldn't happen but safety check)
    if node.env is None:
        print(f"[BUG] Tried to expand from node with env=None but is_terminal={node.is_terminal}")
        print(f"  This is a bug - nodes with env=None should have is_terminal=True")
        print(f"  Node depth: {node.depth if hasattr(node, 'depth') else 'unknown'}")
        print(f"  Parent is_terminal: {node.parent.is_terminal if node.parent else 'no parent'}")
        return None
    
    layout_action = action_catalog.actions[action_idx]
    
    # Create environment copy and step it
    try:
        # Copy the environment at this node's state
        try:
            env_copy = node.env.copy()
        except Exception as e:
            # env.copy() itself failed - node.env might be corrupted
            print(f"[ERROR] env.copy() failed: {type(e).__name__}: {e}")
            child = MCTSNodeV2(
                env=None,
                observation=None,
                parent=node,
                action_taken=layout_action,
                action_idx=action_idx,
                edge_reward=penalty_for_failure,
                steps_to_reach=node.steps_to_reach + 1
            )
            child.is_terminal = True
            return child
        
        # Check if the copied environment is in a valid state
        try:
            test_obs = env_copy.get_obs()
        except Exception as e:
            # Environment copy is in terminal/invalid state
            print(f"[DEBUG] env.copy() returned terminal environment:")
            print(f"  Error: {type(e).__name__}: {e}")
            print(f"  Parent node.is_terminal: {node.is_terminal}")
            print(f"  Parent node.env is not None: {node.env is not None}")
            print(f"  This suggests parent env is in terminal state but is_terminal=False")
            child = MCTSNodeV2(
                env=None,
                observation=None,
                parent=node,
                action_taken=layout_action,
                action_idx=action_idx,
                edge_reward=penalty_for_failure,
                steps_to_reach=node.steps_to_reach + 1
            )
            child.is_terminal = True
            return child
        
        # Convert layout action to Grid2Op action
        grid2op_action = layout_action.apply(env_copy.action_space)
        
        # AUTO-RECONNECT: Try to reconnect disconnected lines after topology action
        # This helps the agent recover from line outages faster
        if auto_reconnect:
            try:
                from actions.reconnection import get_reconnectable_lines, prioritize_lines_by_capacity
                
                # Check for reconnectable lines
                reconnectable = get_reconnectable_lines(node.observation)
                if len(reconnectable) > 0:
                    # Prioritize by capacity and reconnect top N lines
                    reconnectable = prioritize_lines_by_capacity(node.observation, reconnectable)
                    lines_to_reconnect = reconnectable[:max_reconnections]
                    
                    # Add reconnections to the action
                    for line_id in lines_to_reconnect:
                        grid2op_action.line_set_status = [(line_id, 1)]
            except Exception as reconnect_error:
                # If reconnection fails, just proceed with topology action
                pass
        
        # Take the action
        try:
            obs, reward_raw, done, info = env_copy.step(grid2op_action)
        except Exception as e:
            # Step failed - probably environment in bad state
            print(f"[ERROR] env.step() failed: {type(e).__name__}: {e}")
            child = MCTSNodeV2(
                env=None,
                observation=None,
                parent=node,
                action_taken=layout_action,
                action_idx=action_idx,
                edge_reward=penalty_for_failure,
                steps_to_reach=node.steps_to_reach + 1
            )
            child.is_terminal = True
            return child
        
        # Use custom reward function
        is_illegal = info.get('is_illegal', False)
        is_ambiguous = info.get('is_ambiguous', False)
        has_error = done and reward_raw <= penalty_for_failure
        
        # Calculate custom shaped reward
        # Pass the observation we already have to avoid calling env.get_obs() on terminal states
        custom_reward = custom_reward_fn(
            action=grid2op_action,
            env=env_copy,
            has_error=has_error,
            is_done=done,
            is_illegal=is_illegal,
            is_ambiguous=is_ambiguous,
            obs=obs  # Pass the observation from step() to avoid get_obs() on terminal env
        )
        
        # Use custom reward as edge reward (shaped between 0 and 1)
        edge_reward = custom_reward
        
        # PREFILTER: Check if action increases max_rho too much
        # This prevents catastrophic actions that significantly worsen grid state
        parent_max_rho = node.observation.rho.max()
        new_max_rho = obs.rho.max()
        rho_increase = new_max_rho - parent_max_rho
        
        # Allow action 0 (do-nothing) to bypass prefilter
        if action_idx != 0 and rho_increase > prefilter_rho_increase:
            # Action increases rho too much - treat as filtered (terminal failure)
            child = MCTSNodeV2(
                env=None,
                observation=None,
                parent=node,
                action_taken=layout_action,
                action_idx=action_idx,
                edge_reward=penalty_for_failure,
                steps_to_reach=node.steps_to_reach + 1
            )
            child.is_terminal = True
            try:
                env_copy.close()
            except:
                pass
            return child
        
        # HARD TERMINAL CHECK: Extreme overloads (>200%) are immediate failures
        # This prevents MCTS from treating catastrophic actions as "acceptable"
        if new_max_rho > 2.0:
            done = True
            edge_reward = penalty_for_failure
        
        # Check if action was legal
        if done and reward_raw <= penalty_for_failure:
            # Failed action
            child = MCTSNodeV2(
                env=None,  # Don't keep env for terminal nodes
                observation=None,
                parent=node,
                action_taken=layout_action,
                action_idx=action_idx,
                edge_reward=edge_reward,
                steps_to_reach=node.steps_to_reach + 1
            )
            child.is_terminal = True
            return child
        
        # SAFE STATE SKIPPING: Skip through safe states with do-nothing actions
        # Until we reach a critical state or terminal state
        skip_steps = 0
        skip_reward = 0.0
        if not done:
            try:
                env_copy, obs, skip_steps, done, skip_reward = skip_safe_states(
                    env_copy, 
                    max_steps=100, 
                    critical_threshold=critical_threshold
                )
                # Add skip reward to edge reward
                # This gives positive signal for actions that lead to many safe steps
                edge_reward += skip_reward * 0.01  # Small weight so custom reward dominates
            except Exception as e:
                # If skip_safe_states fails (e.g., env not initialized), treat as terminal
                done = True
        
        # Successful action - create child with env copy
        # steps_to_reach includes the action step + any skipped do-nothing steps
        child = MCTSNodeV2(
            env=env_copy if not done else None,  # Only store env if non-terminal
            observation=obs,
            parent=node,
            action_taken=layout_action,
            action_idx=action_idx,
            edge_reward=edge_reward,
            steps_to_reach=node.steps_to_reach + 1 + skip_steps  # Action + skip steps
        )
        child.is_terminal = done
        child.skip_steps = skip_steps  # Store for early stopping logic
        
        # Initialize child's action space
        if not done:
            child.unexpanded_actions = list(range(len(action_catalog.actions)))
            # Uniform priors for now (could use neural network)
            n_actions = len(action_catalog.actions)
            child.action_priors = {i: 1.0/n_actions for i in range(n_actions)}
        else:
            # Terminal node - close the env copy to avoid leaks
            try:
                env_copy.close()
            except:
                pass
        
        return child
        
    except Exception as e:
        # Action failed (illegal, diverging, etc.)
        # Print first few errors with full context, then suppress to avoid spam
        if not hasattr(expand_node, '_error_count'):
            expand_node._error_count = 0
        if expand_node._error_count < 5:
            import traceback
            print(f"[ERROR] Expansion failed: {type(e).__name__}: {e}")
            if expand_node._error_count < 2:  # Only print traceback for first 2 errors
                traceback.print_exc()
            expand_node._error_count += 1
        
        child = MCTSNodeV2(
            env=None,
            observation=None,
            parent=node,
            action_taken=layout_action,
            action_idx=action_idx,
            edge_reward=penalty_for_failure,
            steps_to_reach=node.steps_to_reach + 1
        )
        child.is_terminal = True
        return child


def evaluate_node(node: MCTSNodeV2, value_fn: Optional[Callable] = None) -> float:
    """
    Evaluate a node to get value estimate.
    
    Uses neural network value function if provided, otherwise returns 0.
    No heuristics - we only use real NN values.
    """
    # If observation is None (terminal/failed node), return 0
    if node.observation is None:
        return 0.0
    
    if value_fn is not None:
        return value_fn(node.observation)
    
    # No heuristic - return 0 until we have a trained neural network
    return 0.0


def backup(node: MCTSNodeV2, value: float, gamma: float = 0.99):
    """
    Backup value up the tree from node to root.
    
    Standard AlphaZero backup:
    - N(s,a) += 1
    - W(s,a) += v
    - Q(s,a) = W(s,a) / N(s,a)
    
    Also propagates max_steps_reached: the deepest steps_to_reach value 
    observed in any simulation through this node.
    
    IMPORTANT: Discount is applied based on actual step differences, accounting
    for skipped safe states. If parent is at step 10 and child at step 70,
    we apply gamma^60, not gamma^1.
    """
    discounted_value = value
    leaf_steps = node.steps_to_reach  # Remember the leaf's total steps
    
    while node is not None:
        node.visit_count += 1
        node.value_sum += discounted_value
        # Update max_reachable_steps to be the highest steps_to_reach seen from any leaf
        node.max_reachable_steps = max(node.max_reachable_steps, leaf_steps)
        
        # Move up tree with proper discounting based on step difference
        if node.parent is not None:
            # Calculate how many steps separate parent from this node
            step_diff = node.steps_to_reach - node.parent.steps_to_reach
            # Apply discount: gamma^(step_diff)
            discounted_value = node.edge_reward + (gamma ** step_diff) * discounted_value
        
        node = node.parent


def run_simulation(root: MCTSNodeV2, action_catalog, c_puct: float = 1.0,
                   gamma: float = 0.99, value_fn: Optional[Callable] = None,
                   max_depth: int = 100, epsilon: float = 0.0,
                   force_expand_root: bool = True,
                   critical_threshold: float = 0.90,
                   auto_reconnect: bool = True,
                   max_reconnections: int = 1,
                   prefilter_rho_increase: float = 0.20) -> MCTSNodeV2:
    """
    Run one MCTS simulation: Selection -> Expansion -> Evaluation -> Backup
    
    Args:
        epsilon: Probability of random action selection (epsilon-greedy)
        force_expand_root: If True, expand all root children before going deeper
        auto_reconnect: If True, automatically try to reconnect lines after topology actions
        max_reconnections: Maximum number of lines to reconnect per action
        prefilter_rho_increase: Maximum allowed increase in max_rho (e.g., 0.20 = 20%)
    
    Returns the leaf node that was evaluated.
    """
    # === FORCE ROOT EXPANSION ===
    # If enabled, expand root children first to ensure all actions are tried
    if force_expand_root and len(root.unexpanded_actions) > 0:
        # Expand next root action
        action_idx = root.unexpanded_actions[0]
        node = root
        depth = 0
        is_expanded = False
    else:
        # === 1. SELECTION ===
        # Traverse tree using PUCT until we reach a leaf or unexpanded node
        node = root
        depth = 0
        action_idx = None
        is_expanded = True
        
        while depth < max_depth:
            if node.is_terminal:
                # Can't expand terminal nodes
                break
            
            action_idx, is_expanded = select_child(node, c_puct, epsilon)
            
            if action_idx is None:
                # No actions available
                break
            
            if not is_expanded:
                # Found unexpanded action - break to expand it
                break
            
            # Descend to child
            node = node.children[action_idx]
            depth += 1
    
    # === 2. EXPANSION ===
    # If we stopped at an unexpanded action, expand it
    if action_idx is not None and not is_expanded and not node.is_terminal:
        child = expand_node(node, action_catalog, action_idx, 
                           critical_threshold=critical_threshold,
                           auto_reconnect=auto_reconnect,
                           max_reconnections=max_reconnections,
                           prefilter_rho_increase=prefilter_rho_increase)
        if child is not None:
            node.children[action_idx] = child
            node.unexpanded_actions.remove(action_idx)
            node = child  # Move to newly created child
    
    # === 3. EVALUATION ===
    value = evaluate_node(node, value_fn)
    
    # === 4. BACKUP ===
    backup(node, value, gamma)
    
    return node


def run_mcts(env, observation, action_catalog, num_simulations: int,
             c_puct: float = 1.0, gamma: float = 0.99,
             value_fn: Optional[Callable] = None,
             max_depth: int = 100, epsilon: float = 0.0,
             force_expand_root: bool = True, verbose: bool = False,
             critical_threshold: float = 0.90,
             t_skipped: int = 50,
             t_stopping: int = 20,
             auto_reconnect: bool = True,
             max_reconnections: int = 1,
             prefilter_rho_increase: float = 0.20,
             dirichlet_alpha: float = 0.3,
             dirichlet_epsilon: float = 0.25) -> Tuple[MCTSNodeV2, Dict]:
    """
    Run MCTS from current state with safe state skipping.
    
    Only critical states (max_rho > critical_threshold) create tree nodes.
    Safe states are skipped with do-nothing actions.
    
    Action selection uses epsilon-greedy based on max steps_to_reach:
    - With probability (1-epsilon): Select action with max steps_to_reach
    - With probability epsilon: Select random action
    
    Args:
        env: Grid2Op environment (will be copied for simulations)
        observation: Current observation
        action_catalog: ActionCatalog with available actions
        num_simulations: Number of MCTS simulations to run
        c_puct: Exploration constant
        gamma: Discount factor
        value_fn: Optional neural network value function
        max_depth: Maximum tree depth
        epsilon: Probability of random action selection (epsilon-greedy)
        force_expand_root: If True, expand all root children before going deeper
        verbose: Print progress
        critical_threshold: Only states with max_rho > this create nodes
        t_skipped: Number of skipped safe states to qualify as recovery node
        t_stopping: Stop early if this many recovery nodes found
        auto_reconnect: If True, automatically try to reconnect lines after topology actions
        max_reconnections: Maximum number of lines to reconnect per action
        prefilter_rho_increase: Maximum allowed increase in max_rho (e.g., 0.20 = 20%)
    
    Returns:
        (root_node, stats_dict)
    """
    # Create root node with environment copy
    root_env = env.copy()
    root = MCTSNodeV2(
        env=root_env,
        observation=observation,
        parent=None,
        action_taken=None,
        action_idx=None,
        edge_reward=0.0,
        steps_to_reach=0
    )
    
    # Initialize root's action space
    n_actions = len(action_catalog.actions)
    root.unexpanded_actions = list(range(n_actions))
    root.action_priors = {i: 1.0/n_actions for i in range(n_actions)}
    
    # Add Dirichlet noise to root priors for exploration (AlphaZero technique)
    # dirichlet_alpha controls concentration (lower = more uniform noise)
    # dirichlet_epsilon controls mixing weight (higher = more exploration)
    noise = np.random.dirichlet([dirichlet_alpha] * n_actions)
    for i in range(n_actions):
        root.action_priors[i] = (1 - dirichlet_epsilon) * root.action_priors[i] + dirichlet_epsilon * noise[i]
    
    # Track recovery nodes for early stopping
    recovery_node_count = 0
    
    # Run simulations
    for sim_idx in range(num_simulations):
        leaf = run_simulation(root, action_catalog, c_puct, gamma, value_fn, max_depth, epsilon, force_expand_root, 
                             critical_threshold, auto_reconnect, max_reconnections, prefilter_rho_increase)
        
        # Check if leaf is a recovery node (skipped many safe states)
        # Recovery nodes indicate the action leads to long-term safety
        if hasattr(leaf, 'skip_steps') and leaf.skip_steps >= t_skipped:
            recovery_node_count += 1
            
            # Early stopping: if we found enough recovery nodes, we have good options
            if recovery_node_count >= t_stopping:
                if verbose:
                    print(f"Early stopping at simulation {sim_idx + 1}/{num_simulations}: "
                          f"found {recovery_node_count} recovery nodes (threshold={t_stopping})")
                break
        
        if verbose and (sim_idx + 1) % 100 == 0:
            print(f"Simulation {sim_idx + 1}/{num_simulations}: "
                  f"Root visits={root.visit_count}, "
                  f"Children={len(root.children)}, "
                  f"Recovery nodes={recovery_node_count}, "
                  f"Leaf depth={leaf.get_depth()}")
    
    # Collect statistics
    stats = collect_tree_stats(root)
    
    # Add recovery node count to stats
    stats['recovery_nodes'] = recovery_node_count
    stats['simulations_run'] = sim_idx + 1  # Actual number of simulations (may be less than num_simulations if early stopped)
    
    # Cleanup: close all environment copies to avoid double-close errors
    cleanup_tree_envs(root)
    
    return root, stats


def cleanup_tree_envs(root: MCTSNodeV2):
    """Close all environment copies in the tree to prevent memory leaks."""
    nodes_to_process = [root]
    processed = set()
    
    while nodes_to_process:
        node = nodes_to_process.pop()
        node_id = id(node)
        
        if node_id in processed:
            continue
        processed.add(node_id)
        
        # Close this node's env if it exists
        if node.env is not None:
            try:
                node.env.close()
                node.env = None
            except:
                pass  # Ignore errors if already closed
        
        # Add children to queue
        for child in node.children.values():
            nodes_to_process.append(child)


def collect_tree_stats(root: MCTSNodeV2) -> Dict:
    """Collect statistics about the MCTS tree."""
    stats = {
        'total_nodes': 0,
        'nodes_by_depth': defaultdict(int),
        'visits_by_depth': defaultdict(int),
        'terminal_by_depth': defaultdict(int),
        'max_depth': 0,
        'root_children_visits': [],
    }
    
    # BFS to collect stats
    queue = [root]
    while queue:
        node = queue.pop(0)
        depth = node.get_depth()
        
        stats['total_nodes'] += 1
        stats['nodes_by_depth'][depth] += 1
        stats['visits_by_depth'][depth] += node.visit_count
        if node.is_terminal:
            stats['terminal_by_depth'][depth] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        # Track root children
        if depth == 0:
            for child in node.children.values():
                stats['root_children_visits'].append({
                    'action_idx': child.action_idx,
                    'visits': child.visit_count,
                    'value': child.value(),
                    'is_terminal': child.is_terminal,
                    'steps_to_reach': child.steps_to_reach,  # Immediate steps
                    'max_reachable_steps': child.max_reachable_steps,  # Best path through this action
                })
        
        # Add children to queue
        for child in node.children.values():
            queue.append(child)
    
    return stats


def select_action(root: MCTSNodeV2, temperature: float = 1.0, epsilon: float = 0.0) -> int:
    """
    Select action based on visit counts with temperature-based sampling.
    
    Selection modes:
    - temperature = 0: Greedy (select action with max visits deterministically)
    - temperature > 0: Sample from visit distribution with temperature
      Higher temp = more random, lower temp = more deterministic
    - epsilon > 0: Random action with probability epsilon
    
    CRITICAL: Filters out terminal nodes (immediate failures) to prevent catastrophic actions
    
    Args:
        root: Root MCTS node
        temperature: Controls randomness (0=greedy, higher=more random)
        epsilon: Probability of random action selection
    
    Returns:
        Selected action index
    """
    if len(root.children) == 0:
        return None
    
    # Filter out terminal children (immediate failures), but ALWAYS keep action 0 (do-nothing)
    non_terminal_actions = {
        action_idx: child 
        for action_idx, child in root.children.items() 
        if not child.is_terminal or action_idx == 0
    }
    
    # If all actions lead to terminal states, always do-nothing (safest option)
    if len(non_terminal_actions) == 0:
        print("  ⚠️ WARNING: All actions lead to terminal states! Falling back to do-nothing (action 0)")
        return 0
    
    # Check if all actions' best paths only reach 1 step (all effectively terminal)
    max_reachable = max(child.max_reachable_steps for child in non_terminal_actions.values())
    if max_reachable <= 1:
        print("  ⚠️ WARNING: All action paths only reach 1 step (grid doomed)! Falling back to do-nothing (action 0)")
        return 0
    
    actions = list(non_terminal_actions.keys())
    
    # Epsilon-greedy: random with probability epsilon (only from non-terminal actions)
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    
    # Get visit counts for all non-terminal actions
    visits = np.array([non_terminal_actions[a].visit_count for a in actions])
    
    # Temperature-based selection
    if temperature == 0 or visits.sum() == 0:
        # Greedy: select action with most visits
        best_idx = np.argmax(visits)
        return actions[best_idx]
    else:
        # Sample from visit distribution with temperature
        # Higher temperature = more uniform, lower = more peaked
        visits_temp = np.power(visits, 1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
        return np.random.choice(actions, p=probs)


def print_tree_stats(stats: Dict, action_catalog):
    """Print MCTS tree statistics."""
    print("\n" + "="*60)
    print("MCTS TREE STATISTICS")
    print("="*60)
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Max depth reached: {stats['max_depth']}")
    print()
    
    print("Nodes per depth:")
    for depth in sorted(stats['nodes_by_depth'].keys()):
        total = stats['nodes_by_depth'][depth]
        terminal = stats['terminal_by_depth'][depth]
        visits = stats['visits_by_depth'][depth]
        print(f"  Depth {depth}: {total} nodes, {terminal} terminal ({100*terminal/total:.1f}%), "
              f"{visits} total visits")
    print()
    
    if stats['root_children_visits']:
        print("Top 10 root actions by visit count:")
        sorted_children = sorted(stats['root_children_visits'], 
                                key=lambda x: x['visits'], reverse=True)[:10]
        for i, child_info in enumerate(sorted_children, 1):
            action_idx = child_info['action_idx']
            action_desc = str(action_catalog.actions[action_idx])[:60]
            print(f"  {i}. Action {action_idx}: {child_info['visits']} visits, "
                  f"Q={child_info['value']:.3f}, terminal={child_info['is_terminal']}")
            print(f"      {action_desc}")
        
        # Show visit distribution
        all_visits = [c['visits'] for c in stats['root_children_visits']]
        print(f"\nRoot visit distribution:")
        print(f"  Total root visits: {sum(all_visits)}")
        print(f"  Min visits: {min(all_visits)}")
        print(f"  Max visits: {max(all_visits)}")
        print(f"  Mean visits: {sum(all_visits)/len(all_visits):.1f}")
        print(f"  Actions with >100 visits: {sum(1 for v in all_visits if v > 100)}")
        print(f"  Actions with <10 visits: {sum(1 for v in all_visits if v < 10)}")
    
    print("="*60)
