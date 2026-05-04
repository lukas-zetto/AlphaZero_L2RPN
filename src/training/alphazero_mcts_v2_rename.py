"""
AlphaZero-style MCTS for Grid2Op - Version 2
Properly handles multi-step lookahead by maintaining environment copies.

Key difference from v1: Instead of chaining observation.simulate() calls (which fails),
we maintain environment copies at each node that can be stepped forward.
"""

import numpy as np
import copy
import os
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
               parent_value: float = 0.0, depth_bonus: float = 0.0,
               virtual_loss_weight: float = 0.0,
               value_fn: Optional[Callable] = None) -> float:
    """
    Calculate PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
    
    Standard AlphaZero PUCT with optional modifications:
    - depth_bonus: Encourage exploration of unexpanded/leaf nodes (non-standard)
    - virtual_loss_weight: Penalize frequently visited nodes for wider trees (non-standard)
    
    For unexpanded actions, uses NN value estimate if available, else parent's Q-value.
    """
    if child_idx in node.children:
        child = node.children[child_idx]
        q_value = child.value()
        n_child = child.visit_count
        
        # Virtual loss: penalize nodes with many visits to encourage exploring other children
        # This helps create wider trees instead of narrow deep paths
        if virtual_loss_weight > 0:
            parent_visits = max(1, node.visit_count)
            visit_ratio = n_child / parent_visits
            virtual_loss = -virtual_loss_weight * visit_ratio  # Penalize if this child dominates visits
        else:
            virtual_loss = 0.0
        
        # Small depth bonus for nodes with no children (encourages going deeper)
        if depth_bonus > 0 and len(child.children) == 0 and not child.is_terminal:
            depth_reward = depth_bonus
        else:
            depth_reward = 0.0
    else:
        # Unexpanded action: use NN value estimate if available, else parent's value
        if value_fn is not None and node.observation is not None:
            try:
                q_value = value_fn(node.observation)
            except:
                q_value = parent_value
        else:
            q_value = parent_value
        n_child = 0
        virtual_loss = 0.0
        depth_reward = depth_bonus * 0.5 if depth_bonus > 0 else 0.0  # Small bonus for unexpanded
    
    prior = node.action_priors.get(child_idx, 1.0 / len(node.action_priors))
    exploration = c_puct * prior * np.sqrt(node.visit_count) / (1 + n_child)
    
    return q_value + exploration + depth_reward + virtual_loss


def select_child(node: MCTSNodeV2, c_puct: float, epsilon: float = 0.0,
                depth_bonus: float = 0.0, virtual_loss_weight: float = 0.0,
                value_fn: Optional[Callable] = None, debug_root: bool = False) -> Tuple[int, bool]:
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
            score = puct_score(node, action_idx, c_puct, parent_q, depth_bonus, virtual_loss_weight, value_fn)
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        return best_action, False  # Not yet expanded
    
    # All actions expanded, select best child (skip terminal children)
    elif len(node.children) > 0:
        best_score = -np.inf
        best_action = None
        
        # Debug: print PUCT scores at root
        if debug_root:
            print(f"    🔍 PUCT selection at depth {node.get_depth()}, parent_visits={node.visit_count}:")
            scores = []
            for action_idx, child in node.children.items():
                if not child.is_terminal:
                    score = puct_score(node, action_idx, c_puct, parent_q, depth_bonus, virtual_loss_weight, value_fn)
                    prior = node.action_priors.get(action_idx, 1.0 / len(node.action_priors))
                    q_val = child.value()
                    visits = child.visit_count
                    exploration_term = c_puct * prior * np.sqrt(node.visit_count) / (1 + visits)
                    rho = child.observation.rho.max() if child.observation is not None else 0
                    scores.append((action_idx, score, q_val, prior, visits, exploration_term, rho, child.edge_reward))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"      Top 5 actions by PUCT:")
            for i, (action_idx, score, q_val, prior, visits, expl, rho, edge_r) in enumerate(scores[:5]):
                print(f"        #{i+1} Action {action_idx}: PUCT={score:.3f} = Q={q_val:.3f} + Expl={expl:.3f} "
                      f"(Prior={prior:.4f}, Visits={visits}, EdgeR={edge_r:.3f}, Rho={rho:.3f})")
            print(f"      → Selected: Action {scores[0][0]}")
        
        for action_idx, child in node.children.items():
            # Skip terminal children - can't explore further
            if child.is_terminal:
                continue
            
            score = puct_score(node, action_idx, c_puct, parent_q, depth_bonus, virtual_loss_weight, value_fn)
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
                prefilter_rho_increase: float = 0.20,
                policy_fn: Optional[Callable] = None) -> Optional[MCTSNodeV2]:
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
    
    # FILTER IDENTITY ACTIONS: Check if action would result in same topology as current state
    # Special case: sub_id=-1 is do-nothing action, always allow it
    if layout_action.sub_id != -1 and node.observation is not None:
        current_obs = node.observation
        
        # Check if this action would change any bus assignment
        would_change_topology = False
        
        # Check line origins
        for line_idx in layout_action.line_or_indices:
            current_bus = current_obs.line_or_bus[line_idx]
            target_bus = layout_action.assignment[("line_or", line_idx)]
            if current_bus != target_bus:
                would_change_topology = True
                break
        
        # Check line extremities
        if not would_change_topology:
            for line_idx in layout_action.line_ex_indices:
                current_bus = current_obs.line_ex_bus[line_idx]
                target_bus = layout_action.assignment[("line_ex", line_idx)]
                if current_bus != target_bus:
                    would_change_topology = True
                    break
        
        # Check generators
        if not would_change_topology:
            for gen_idx in layout_action.gen_indices:
                current_bus = current_obs.gen_bus[gen_idx]
                target_bus = layout_action.assignment[("gen", gen_idx)]
                if current_bus != target_bus:
                    would_change_topology = True
                    break
        
        # Check loads
        if not would_change_topology:
            for load_idx in layout_action.load_indices:
                current_bus = current_obs.load_bus[load_idx]
                target_bus = layout_action.assignment[("load", load_idx)]
                if current_bus != target_bus:
                    would_change_topology = True
                    break
        
        # If action wouldn't change topology, treat as terminal with penalty
        if not would_change_topology:
            child = MCTSNodeV2(
                env=None,
                observation=None,
                parent=node,
                action_taken=layout_action,
                action_idx=action_idx,
                edge_reward=penalty_for_failure,  # Penalize no-op actions
                steps_to_reach=node.steps_to_reach + 1
            )
            child.is_terminal = True
            return child
    
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
            if enable_debug:
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
        
        # SAFE STATE SKIPPING: DISABLED - causes unreliable max_reachable_steps
        # The simulated environment diverges from reality, reporting safe states that actually fail
        skip_steps = 0
        skip_reward = 0.0
        if False:  # DISABLED
            try:
                env_copy, obs, skip_steps, done, skip_reward = skip_safe_states(
                    env_copy, 
                    max_steps=0,  # DISABLED - set to 0 to prevent skipping
                    critical_threshold=critical_threshold
                )
                # DON'T add skip_reward to edge_reward - it inflates values too much
                # The heuristic value function will extrapolate the edge_reward anyway
                # edge_reward += skip_reward * 0.01  # REMOVED - causes reward inflation
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
            n_actions = len(action_catalog.actions)
            
            # Get action priors from neural network policy for this child state
            if policy_fn is not None and obs is not None:
                try:
                    policy_probs = policy_fn(obs)
                    policy_probs = np.array(policy_probs)
                    if policy_probs.sum() > 0:
                        policy_probs = policy_probs / policy_probs.sum()
                    else:
                        policy_probs = np.ones(n_actions) / n_actions
                    child.action_priors = {i: float(policy_probs[i]) for i in range(n_actions)}
                except:
                    # Fallback to uniform if NN fails
                    child.action_priors = {i: 1.0/n_actions for i in range(n_actions)}
            else:
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
    
    Uses neural network value function if provided.
    If value_fn returns a heuristic-based value, it will use the node's actual
    edge_reward to extrapolate future value.
    """
    # If observation is None (terminal/failed node), return 0
    if node.observation is None:
        return 0.0
    
    if value_fn is not None:
        return value_fn(node.observation)
    
    # No value function - return 0
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
                   prefilter_rho_increase: float = 0.20,
                   depth_bonus: float = 0.0,
                   virtual_loss_weight: float = 0.0,
                   penalty_for_failure: float = -5.0,
                   policy_fn: Optional[Callable] = None,
                   debug_action: int = None,
                   sim_idx: int = -1) -> MCTSNodeV2:
    """
    Run one MCTS simulation: Selection -> Expansion -> Evaluation -> Backup
    
    Args:
        epsilon: Probability of random action selection (epsilon-greedy)
        force_expand_root: If True, expand all root children before going deeper
        auto_reconnect: If True, automatically try to reconnect lines after topology actions
        max_reconnections: Maximum number of lines to reconnect per action
        prefilter_rho_increase: Maximum allowed increase in max_rho (e.g., 0.20 = 20%)
        depth_bonus: Bonus for unexpanded/leaf nodes (0 = disabled)
        virtual_loss_weight: Penalty for frequently visited nodes (0 = disabled)
        debug_action: If set, log detailed path when this action is taken from root
    
    Returns the leaf node that was evaluated.
    """
    # Track path for debugging
    path_actions = []
    path_rewards = []
    path_rhos = []
    
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
        
        # Enable debug for first few simulations
        enable_debug = (sim_idx >= 0 and sim_idx < 5)
        
        if enable_debug:
            print(f"\n  === Simulation {sim_idx}: Starting tree traversal ===")
        
        while depth < max_depth:
            if node.is_terminal:
                # Can't expand terminal nodes
                break
            
            action_idx, is_expanded = select_child(node, c_puct, epsilon, depth_bonus, virtual_loss_weight, value_fn, debug_root=enable_debug)
            
            if action_idx is None:
                # No actions available
                break
            
            if not is_expanded:
                # Found unexpanded action - break to expand it
                if enable_debug:
                    print(f"      → Action {action_idx} unexpanded, will expand it")
                break
            
            # Track path for debug action
            if depth == 0 and debug_action is not None and action_idx == debug_action:
                path_actions.append(action_idx)
                if node.observation is not None:
                    path_rhos.append(node.observation.rho.max())
            elif len(path_actions) > 0:
                # Continue tracking if we started from debug_action
                path_actions.append(action_idx)
                if node.children[action_idx].observation is not None:
                    path_rhos.append(node.children[action_idx].observation.rho.max())
                path_rewards.append(node.children[action_idx].edge_reward)
            
            # Descend to child
            if enable_debug:
                child_rho = node.children[action_idx].observation.rho.max() if node.children[action_idx].observation is not None else 0
                print(f"      → Descending to action {action_idx} (depth {depth} → {depth+1}, rho={child_rho:.3f})")
            node = node.children[action_idx]
            depth += 1
    
    # === 2. EXPANSION ===
    # If we stopped at an unexpanded action, expand it
    if action_idx is not None and not is_expanded and not node.is_terminal:
        child = expand_node(node, action_catalog, action_idx, 
                           critical_threshold=critical_threshold,
                           penalty_for_failure=penalty_for_failure,
                           auto_reconnect=auto_reconnect,
                           max_reconnections=max_reconnections,
                           prefilter_rho_increase=prefilter_rho_increase,
                           policy_fn=policy_fn)
        if child is not None:
            node.children[action_idx] = child
            node.unexpanded_actions.remove(action_idx)
            node = child  # Move to newly created child
    
    # === 3. EVALUATION ===
    value = evaluate_node(node, value_fn)
    
    # Debug logging for tracked paths
    if len(path_actions) > 0 and debug_action is not None and enable_debug:
        print(f"    [DEBUG] Path through action {debug_action}:")
        print(f"      Actions: {path_actions}")
        print(f"      Rewards: {[f'{r:.3f}' for r in path_rewards]}")
        print(f"      Rhos: {[f'{r:.3f}' for r in path_rhos]}")
        print(f"      Leaf value: {value:.3f}, Depth: {len(path_actions)}")
    
    # === 4. BACKUP ===
    backup(node, value, gamma)
    
    return node


def run_mcts(env, observation, action_catalog, num_simulations: int,
             c_puct: float = 1.0, gamma: float = 0.99,
             policy_fn: Optional[Callable] = None,
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
             dirichlet_epsilon: float = 0.25,
             depth_bonus: float = 0.0,
             virtual_loss_weight: float = 0.0,
             penalty_for_failure: float = -5.0,
             enable_debug: bool = False) -> Tuple[MCTSNodeV2, Dict]:
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
        depth_bonus: Bonus for unexpanded/leaf nodes (0 = disabled)
        virtual_loss_weight: Penalty for frequently visited nodes (0 = disabled)
    
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
    
    # Get action priors from neural network policy, or use uniform if not available
    if policy_fn is not None:
        try:
            policy_probs = policy_fn(observation)
            # Normalize to ensure valid probability distribution
            policy_probs = np.array(policy_probs)
            if policy_probs.sum() > 0:
                policy_probs = policy_probs / policy_probs.sum()
            else:
                policy_probs = np.ones(n_actions) / n_actions
            root.action_priors = {i: float(policy_probs[i]) for i in range(n_actions)}
        except:
            # Fallback to uniform if NN fails
            root.action_priors = {i: 1.0/n_actions for i in range(n_actions)}
    else:
        root.action_priors = {i: 1.0/n_actions for i in range(n_actions)}
    
    if enable_debug:
        print(f"[DEBUG] policy_fn is None: {policy_fn is None}")
        print(f"[DEBUG] n_actions: {n_actions}, expected uniform prior: {1.0/n_actions}")
        print(f"[DEBUG] Priors BEFORE Dirichlet: {list(root.action_priors.values())[:10]}")
    
    # Add Dirichlet noise to root priors for exploration (AlphaZero technique)
    # dirichlet_alpha controls concentration (lower = more uniform noise)
    # dirichlet_epsilon controls mixing weight (higher = more exploration)
    if enable_debug:
        print(f"[DEBUG] dirichlet_epsilon = {dirichlet_epsilon}, adding noise = {dirichlet_epsilon > 0}")
    if dirichlet_epsilon > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * n_actions)
        for i in range(n_actions):
            root.action_priors[i] = (1 - dirichlet_epsilon) * root.action_priors[i] + dirichlet_epsilon * noise[i]
    if enable_debug:
        print(f"[DEBUG] Sample priors: {list(root.action_priors.values())[:5]}")
    
    # Track recovery nodes for early stopping
    recovery_node_count = 0
    
    # Debug: Track top action from initial visit counts to follow its path
    debug_action = None
    
    # Run simulations
    for sim_idx in range(num_simulations):
        # Enable debug logging after 100 simulations for the most visited action
        if sim_idx == 100 and verbose:
            # Find action with most visits
            max_visits = 0
            for action_idx, child in root.children.items():
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    debug_action = action_idx
            if debug_action is not None:
                print(f"\n  🔍 Debugging action {debug_action} (current visits: {max_visits})")
        
        # Only log paths for simulations 101-105 to avoid spam
        current_debug = debug_action if (100 <= sim_idx < 105 and verbose) else None
        
        leaf = run_simulation(root, action_catalog, c_puct, gamma, value_fn, max_depth, epsilon, force_expand_root, 
                             critical_threshold, auto_reconnect, max_reconnections, prefilter_rho_increase,
                             depth_bonus, virtual_loss_weight, penalty_for_failure, policy_fn, current_debug, sim_idx)
        
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
    
    # Debug: Print PUCT analysis for top actions
    if verbose:
        print("\n  🔍 PUCT Analysis at root (top 10 actions):")
        analysis = []
        for action_idx, child in root.children.items():
            if not child.is_terminal:
                prior = root.action_priors.get(action_idx, 1.0 / len(root.action_priors))
                q_val = child.value()
                visits = child.visit_count
                exploration_term = c_puct * prior * np.sqrt(root.visit_count) / (1 + visits)
                puct = q_val + exploration_term
                rho = child.observation.rho.max() if child.observation is not None else 0
                analysis.append((action_idx, puct, q_val, prior, visits, exploration_term, child.edge_reward, child.steps_to_reach, rho, child))
        
        # Sort by visits (what was actually chosen)
        analysis.sort(key=lambda x: x[4], reverse=True)
        for i, (action_idx, puct, q_val, prior, visits, expl, edge_r, steps, rho, child) in enumerate(analysis[:10]):
            print(f"      Action {action_idx}: Visits={visits:4d}, Q={q_val:6.3f}, EdgeR={edge_r:6.3f}, Rho={rho:.3f}, "
                  f"Prior={prior:.4f}, Expl={expl:6.3f}, PUCT={puct:6.3f}")
        
        # Print detailed subtree for top action
        if len(analysis) > 0:
            top_action_idx, _, top_q, _, top_visits, _, top_edge_r, _, top_rho, top_child = analysis[0]
            print(f"\n  📊 Subtree analysis for action {top_action_idx} (Q={top_q:.3f}, visits={top_visits}):")
            print(f"      Edge reward: {top_edge_r:.3f}, Rho: {top_rho:.3f}")
            
            # Show top children of this action
            if len(top_child.children) > 0:
                child_analysis = []
                for child_action_idx, grandchild in top_child.children.items():
                    if not grandchild.is_terminal:
                        gc_q = grandchild.value()
                        gc_visits = grandchild.visit_count
                        gc_rho = grandchild.observation.rho.max() if grandchild.observation is not None else 0
                        child_analysis.append((child_action_idx, gc_q, gc_visits, grandchild.edge_reward, gc_rho))
                
                child_analysis.sort(key=lambda x: x[2], reverse=True)
                print(f"      Top 5 children (depth 2):")
                for ca_idx, ca_q, ca_visits, ca_edge_r, ca_rho in child_analysis[:5]:
                    print(f"        Action {ca_idx}: Visits={ca_visits:3d}, Q={ca_q:6.3f}, EdgeR={ca_edge_r:6.3f}, Rho={ca_rho:.3f}")
    
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


def collect_all_tree_nodes(root: MCTSNodeV2, encode_fn: Callable) -> List[Dict]:
    """
    Collect training data from ALL nodes in the MCTS tree (not just root).
    This is the proper AlphaZero approach - use every position visited during search.
    
    Each node provides:
    - state: encoded observation
    - policy: visit distribution over children (or uniform if terminal/leaf)
    - value: MCTS Q-value at this node
    
    Args:
        root: Root of MCTS tree
        encode_fn: Function to encode observation to state vector
        
    Returns:
        List of training examples (state, policy, value) for each visited node
    """
    training_data = []
    
    # BFS through tree
    queue = [root]
    visited = set()
    
    while queue:
        node = queue.pop(0)
        node_id = id(node)
        
        if node_id in visited:
            continue
        visited.add(node_id)
        
        # Only include nodes that were actually visited during MCTS
        if node.visit_count == 0:
            continue
            
        # Encode state
        state_vector = encode_fn(node.observation)
        
        # Get policy distribution (visit counts over children)
        if len(node.children) > 0:
            # Use visit counts to create policy target
            child_actions = list(node.children.keys())
            visits = np.array([node.children[a].visit_count for a in child_actions])
            
            # Create full policy vector (size = action space)
            num_actions = max(child_actions) + 1 if child_actions else 1
            policy = np.zeros(num_actions)
            
            if visits.sum() > 0:
                # Normalize visits to probabilities
                visit_probs = visits / visits.sum()
                for action_idx, prob in zip(child_actions, visit_probs):
                    policy[action_idx] = prob
            else:
                # Uniform over expanded actions if no visits
                for action_idx in child_actions:
                    policy[action_idx] = 1.0 / len(child_actions)
        else:
            # Leaf or terminal node - no children
            # Use uniform policy (no information)
            num_actions = 61  # Default action space size
            policy = np.ones(num_actions) / num_actions
        
        # Get value (MCTS Q-value)
        value = node.value()
        
        training_data.append({
            'state': state_vector,
            'policy': policy,
            'value': value,
            'visit_count': node.visit_count,
            'depth': node.get_depth(),
        })
        
        # Add children to queue
        for child in node.children.values():
            queue.append(child)
    
    return training_data


def compute_heuristic_value(node: MCTSNodeV2, gamma: float = 0.95, horizon: int = 100) -> float:
    """
    Compute heuristic value function based on discounted future rewards.
    
    From the paper: "The heuristic value function for state St is based on the assumption 
    that the current reward will remain similar for subsequent states... the heuristic value 
    is defined as the asymptotic value νt = Σ(j=t to t+h) γ^j * r_j"
    
    Since we don't have access to future rewards at a leaf node during MCTS, we use:
    - The current raw Grid2Op reward at this node as an estimate
    - Asymptotic sum: r * Σ(γ^i for i=0 to h) = r * (1 - γ^h) / (1 - γ)
    
    Args:
        node: MCTS node
        gamma: Discount factor
        horizon: Lookahead horizon
        
    Returns:
        Heuristic value estimate
    """
    # Get current reward (edge reward to reach this node)
    current_reward = node.edge_reward
    
    # Compute geometric series sum: (1 - γ^horizon) / (1 - γ)
    if gamma >= 1.0:
        # Undiscounted case
        heuristic_value = current_reward * horizon
    else:
        # Discounted case
        heuristic_value = current_reward * (1.0 - gamma**horizon) / (1.0 - gamma)
    
    return heuristic_value


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
        if os.getenv("AGENT_DEBUG", "false").lower() == "true":
            print("  ⚠️ WARNING: All actions lead to terminal states! Falling back to do-nothing (action 0)")
        return 0
    
    # Check if all actions' best paths only reach 1 step (all effectively terminal)
    max_reachable = max(child.max_reachable_steps for child in non_terminal_actions.values())
    if max_reachable <= 1:
        if os.getenv("AGENT_DEBUG", "false").lower() == "true":
            print("  ⚠️ WARNING: All action paths only reach 1 step (grid doomed)! Falling back to do-nothing (action 0)")
        return 0
    
    actions = list(non_terminal_actions.keys())
    
    # Epsilon-greedy: random with probability epsilon (only from non-terminal actions)
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    
    # Multi-factor selection: max_reachable_steps (primary) → visits (secondary) → Q-value (tertiary)
    children = [non_terminal_actions[a] for a in actions]
    max_steps = np.array([child.max_reachable_steps for child in children])
    visits = np.array([child.visit_count for child in children])
    q_values = np.array([child.value() for child in children])
    
    # Temperature-based selection
    if temperature == 0 or visits.sum() == 0:
        # Greedy: select by (max_steps, visits, Q-value) lexicographic ordering
        # Find all actions with maximum reachable steps
        best_steps = np.max(max_steps)
        best_steps_mask = (max_steps == best_steps)
        
        # Among those, find ones with most visits
        best_visits = np.max(visits[best_steps_mask])
        best_visits_mask = best_steps_mask & (visits == best_visits)
        
        # Among those, select by highest Q-value
        best_q_idx = np.argmax(np.where(best_visits_mask, q_values, -np.inf))
        return actions[best_q_idx]
    else:
        # Sample from visit distribution with temperature (standard AlphaZero during training)
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
