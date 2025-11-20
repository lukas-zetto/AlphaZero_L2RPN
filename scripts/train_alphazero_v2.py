"""
AlphaZero training with MCTS v2 (environment copy approach).
Trains a neural network using self-play with MCTS search.

Supports both sequential and parallel episode collection.
"""

import sys
sys.path.append('/workspace/src')

import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
import numpy as np
import torch
import os
from collections import deque
import multiprocessing as mp
import warnings
import traceback

from actions.action_catalog import build_action_catalog
from training.alphazero_mcts_v2 import run_mcts, select_action, MCTSNodeV2
from networks.neural_network import create_neural_network, encode_observation_simple, neural_network_forward, train_neural_network

# Suppress warnings in parallel workers
warnings.filterwarnings("ignore")


def _parallel_episode_worker(args):
    """
    Worker function for parallel episode collection.
    Runs in separate process - must be self-contained.
    
    Returns episode data in same format as self_play_episode.
    """
    chronic_id, episode_id, config, worker_id = args
    
    try:
        # Re-import in worker process
        import grid2op
        from lightsim2grid import LightSimBackend
        from grid2op.Parameters import Parameters
        from actions.action_catalog import build_action_catalog
        from training.alphazero_mcts_v2 import run_mcts, select_action
        from networks.neural_network import create_neural_network, encode_observation_simple, neural_network_forward
        
        # Create environment
        params = Parameters()
        params.NO_OVERFLOW_DISCONNECTION = True
        env = grid2op.make(
            "l2rpn_case14_sandbox",
            backend=LightSimBackend(),
            param=params
        )
        
        # Build action catalog
        from config import ACTIONS_CONFIG
        catalog = build_action_catalog(
            env,
            substations=ACTIONS_CONFIG['substations'],
            reduction=ACTIONS_CONFIG['reduction'],
            drop_identity=ACTIONS_CONFIG['drop_identity'],
            include_do_nothing=ACTIONS_CONFIG.get('include_do_nothing', True)
        )
        
        # Create neural network and load shared weights
        input_size = 83
        num_actions = len(catalog.actions)
        
        # Import torch for state dict loading
        import torch
        
        # Create network structure (suppress print messages in workers)
        import os
        # Temporarily suppress stdout for network creation
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        neural_network = create_neural_network(input_size=input_size, num_actions=num_actions, config=config)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Load shared network weights if provided
        if 'network_state_dict' in config:
            neural_network.load_state_dict(config['network_state_dict'])
            neural_network.eval()  # Set to evaluation mode for inference
        
        # Set chronic and reset
        env.set_id(chronic_id)
        obs = env.reset()
        
        # Print worker start (will appear in log)
        print(f"  [Worker {worker_id}] Started: Chronic {chronic_id}", flush=True)
        
        done = False
        step = 0
        episode_data = []
        max_steps = config['max_episode_steps']
        critical_threshold = config.get('critical_threshold', 0.90)
        
        # Track episode statistics
        max_rho_seen = 0.0
        overloaded_lines = []
        actions_taken = []
        critical_states_count = 0
        
        while not done and step < max_steps:
            max_rho = np.max(obs.rho)
            is_critical = max_rho > critical_threshold
            
            # Track max rho seen
            if max_rho > max_rho_seen:
                max_rho_seen = max_rho
                # Track which lines are overloaded
                overloaded_lines = [i for i, rho in enumerate(obs.rho) if rho > 1.0]
            
            if is_critical:
                critical_states_count += 1
                
                # Store state before action
                rho_before = obs.rho.copy()
                max_rho_before = np.max(rho_before)
                overloaded_before = [i for i, rho in enumerate(rho_before) if rho > 1.0]
                
                # Run MCTS
                root, stats = run_mcts(
                    env,
                    obs,
                    catalog,
                    num_simulations=config['mcts_simulations'],
                    c_puct=config['c_puct'],
                    gamma=config['gamma'],
                    max_depth=config['max_depth'],
                    epsilon=config.get('mcts_epsilon', 0.0),
                    value_fn=lambda o: neural_network_forward(neural_network, o, num_actions)[1],
                    critical_threshold=critical_threshold
                )
                
                # Select action (based on max_reachable_steps)
                action_idx = select_action(root, temperature=0, epsilon=0.0)
                
                # Get MCTS policy target based on config
                policy_target_method = config.get('policy_target_method', 'visits')
                policy_temperature = config.get('policy_temperature', 1.0)
                
                if policy_target_method == 'one_hot':
                    # One-hot: target the selected action (which has best max_reachable_steps)
                    mcts_policy = np.zeros(num_actions)
                    mcts_policy[action_idx] = 1.0
                    
                elif policy_target_method == 'max_steps':
                    # Distribution based on max_reachable_steps with temperature sharpening
                    max_reachable_values = np.array([root.children[i].max_reachable_steps if i in root.children else 0 
                                         for i in range(num_actions)])
                    
                    if max_reachable_values.max() > 0:
                        # Apply temperature sharpening (lower temp = more peaked toward best action)
                        # Shift to positive values to avoid issues with negative powers
                        max_steps_shifted = max_reachable_values + 1  # Add 1 to avoid zero
                        steps_powered = np.power(max_steps_shifted, 1.0/policy_temperature)
                        mcts_policy = steps_powered / steps_powered.sum()
                    else:
                        # Fallback if no valid actions
                        mcts_policy = np.zeros(num_actions)
                        mcts_policy[action_idx] = 1.0
                        
                elif policy_target_method == 'visits_with_selection_bias':
                    # Visit distribution with extra weight on selected action
                    visits = np.array([root.children[i].visit_count if i in root.children else 0 
                                      for i in range(num_actions)])
                    if visits.sum() > 0:
                        visits_powered = np.power(visits, 1.0/policy_temperature)
                        # Add selection bias: boost the actually selected action (but not action 0 - do-nothing)
                        if action_idx != 0:
                            selection_bias = config.get('selection_bias_weight', 0.5)
                            visits_powered[action_idx] *= (1.0 + selection_bias)
                        mcts_policy = visits_powered / visits_powered.sum()
                    else:
                        mcts_policy = np.ones(num_actions) / num_actions
                        
                else:  # 'visits' or default
                    # Visit distribution: traditional AlphaZero (exploration pattern)
                    visits = np.array([root.children[i].visit_count if i in root.children else 0 
                                      for i in range(num_actions)])
                    if visits.sum() > 0:
                        # Apply temperature sharpening if specified
                        visits_powered = np.power(visits, 1.0/policy_temperature)
                        mcts_policy = visits_powered / visits_powered.sum()
                    else:
                        mcts_policy = np.ones(num_actions) / num_actions
                
                # Store training example
                state_vector = encode_observation_simple(obs)
                root_value = root.value()
                episode_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': root_value,
                })
                
                # Take action
                action = catalog.actions[action_idx]
                grid2op_action = action.apply(env.action_space)
                obs, reward, done, info = env.step(grid2op_action)
                step += 1
                
                # Track action result
                rho_after = obs.rho.copy()
                max_rho_after = np.max(rho_after)
                overloaded_after = [i for i, rho in enumerate(rho_after) if rho > 1.0]
                rho_change = max_rho_after - max_rho_before
                
                action_desc = catalog.actions[action_idx].description() if hasattr(catalog.actions[action_idx], 'description') else f"Action {action_idx}"
                actions_taken.append({
                    'step': step,
                    'action_idx': action_idx,
                    'action_desc': action_desc,
                    'rho_before': max_rho_before,
                    'rho_after': max_rho_after,
                    'rho_change': rho_change,
                    'overloaded_before': overloaded_before,
                    'overloaded_after': overloaded_after
                })
            else:
                # Safe state - do nothing
                obs, reward, done, info = env.step(env.action_space())
                step += 1
        
        env.close()
        
        # Print completion summary with statistics
        print(f"  [Worker {worker_id}] Completed: Chronic {chronic_id}", flush=True)
        print(f"    Steps: {step}, Critical states: {critical_states_count}, Training examples: {len(episode_data)}", flush=True)
        print(f"    Max rho: {max_rho_seen:.3f}", flush=True)
        if overloaded_lines:
            print(f"    Peak overloaded lines (rho>1.0): {overloaded_lines[:5]}" + (" ..." if len(overloaded_lines) > 5 else ""), flush=True)
        if actions_taken:
            print(f"    Actions taken: {len(actions_taken)} topology changes", flush=True)
            # Show ALL actions with detailed info
            for action_info in actions_taken:
                a = action_info
                improvement = "✓" if a['rho_change'] < 0 else "✗"
                print(f"      Step {a['step']}: Action {a['action_idx']} {improvement} rho {a['rho_before']:.3f} → {a['rho_after']:.3f} ({a['rho_change']:+.3f})", flush=True)
                if a['overloaded_before'] or a['overloaded_after']:
                        before_str = f"{len(a['overloaded_before'])} lines" if a['overloaded_before'] else "none"
                        after_str = f"{len(a['overloaded_after'])} lines" if a['overloaded_after'] else "none"
                        print(f"        Overloaded: {before_str} → {after_str}", flush=True)
        
        return {
            'success': True,
            'chronic_id': chronic_id,
            'episode_id': episode_id,
            'worker_id': worker_id,
            'episode_data': episode_data,
            'steps': step,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Worker {worker_id} failed: {str(e)}\n{traceback.format_exc()}"
        return {
            'success': False,
            'chronic_id': chronic_id,
            'episode_id': episode_id,
            'worker_id': worker_id,
            'episode_data': [],
            'steps': 0,
            'error': error_msg
        }


class AlphaZeroTrainerV2:
    """AlphaZero self-play trainer using MCTS v2."""
    
    def __init__(self, env, action_catalog, config):
        self.env = env
        self.action_catalog = action_catalog
        self.config = config
        self.use_replay_buffer = config.get('use_replay_buffer', True)
        # Create neural network
        self.input_size = 83  # From encode_observation_simple
        self.num_actions = len(action_catalog.actions)
        self.neural_network = create_neural_network(
            input_size=self.input_size,
            num_actions=self.num_actions,
            config=config
        )
        # Create optimizer once (for learning rate decay)
        self.learning_rate = config['learning_rate']
        self.optimizer = torch.optim.Adam(
            self.neural_network.parameters(), 
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.0001)
        )
        # Learning rate decay params
        self.lr_decay = config.get('learning_rate_decay', 1.0)
        self.min_lr = config.get('min_learning_rate', 0.00001)
        
        # Temperature for action selection
        self.current_temperature = config.get('temperature', 1.0)
        
        # Training data buffer - stores EPISODES (each episode is a list of experiences)
        self.replay_buffer = deque(maxlen=config.get('replay_buffer_size', 100))
        # Split chronics into 90% train / 10% test
        total_chronics = len(env.chronics_handler.real_data.subpaths)
        train_size = int(total_chronics * 0.9)
        self.train_chronics = list(range(train_size))
        self.test_chronics = list(range(train_size, total_chronics))
        self.current_chronic_idx = 0  # Index into train_chronics list
        print(f"Created neural network: input={self.input_size}, actions={self.num_actions}")
        print(f"Chronic split: {len(self.train_chronics)} train ({self.train_chronics[0]}-{self.train_chronics[-1]}), {len(self.test_chronics)} test ({self.test_chronics[0]}-{self.test_chronics[-1]})")
        print(f"Initial learning rate: {self.learning_rate:.6f}, decay: {self.lr_decay}, min: {self.min_lr:.6f}")
        print(f"Replay buffer: {'ENABLED' if self.use_replay_buffer else 'DISABLED'}, size: {config.get('replay_buffer_size', 0)} episodes")
    
    def get_policy_value(self, observation):
        """Get policy and value from neural network."""
        if observation is None:
            # Return uniform policy and zero value for None observations
            uniform_policy = np.ones(self.num_actions) / self.num_actions
            return uniform_policy, 0.0
        
        policy, value = neural_network_forward(
            self.neural_network,
            observation,
            self.num_actions
        )
        return policy, value
    
    def self_play_episode(self, episode_id):
        """Run one self-play episode collecting training data."""
        print(f"\n{'='*60}")
        print(f"SELF-PLAY EPISODE {episode_id}")
        print(f"{'='*60}")
        
        # Reset environment with next training chronic (cycle through ALL training chronics)
        chronic_id = self.train_chronics[self.current_chronic_idx % len(self.train_chronics)]
        print(f"Using training chronic {chronic_id} ({self.current_chronic_idx + 1}/{len(self.train_chronics)} in cycle)")
        self.env.set_id(chronic_id)
        obs = self.env.reset()
        self.current_chronic_idx += 1
        
        done = False
        step = 0
        episode_data = []
        is_failure = False  # Track if episode ended due to failure
        
        while not done and step < self.config['max_episode_steps']:
            max_rho = np.max(obs.rho)
            critical_threshold = self.config.get('critical_threshold', 0.90)
            is_critical = max_rho > critical_threshold
            
            print(f"\nStep {step}: rho_max={max_rho:.3f} {'🔴 CRITICAL' if is_critical else '🟢 SAFE'} (threshold={critical_threshold})")
            
            # Only run MCTS if in critical state
            if not is_critical:
                # Check if we should reset topology to reference when safe
                reset_threshold = self.config.get('topology_reset_threshold', 0.75)
                if max_rho <= reset_threshold:
                    from actions.topology_reset import get_reference_topology_action
                    try:
                        reset_action = get_reference_topology_action(obs, self.env.action_space)
                        # Check if this is actually a reset (not do-nothing)
                        # For now, just apply it
                        print(f"  🔄 Grid is very safe (rho={max_rho:.3f} ≤ {reset_threshold}) - resetting to reference topology")
                        obs, reward, done, info = self.env.step(reset_action)
                    except Exception as e:
                        print(f"  ⚠️ Topology reset failed: {e}, using do-nothing")
                        obs, reward, done, info = self.env.step(self.env.action_space())
                else:
                    print(f"  ⏭️ Skipping MCTS - grid is safe, taking do-nothing action")
                    # Take do-nothing action
                    obs, reward, done, info = self.env.step(self.env.action_space())
                step += 1
                continue
            
            # Run MCTS with neural network (only in critical states)
            epsilon = self.config.get('mcts_epsilon', 0.0)
            critical_threshold = self.config.get('critical_threshold', 0.90)
            
            # COMPARISON: What would do-nothing do?
            try:
                do_nothing_action = self.env.action_space()
                sim_obs, sim_reward, sim_done, sim_info = obs.simulate(do_nothing_action)
                do_nothing_rho = sim_obs.rho.max()
                rho_delta = do_nothing_rho - max_rho
                if sim_done:
                    print(f"  🚫 Do-nothing would FAIL (rho={do_nothing_rho:.3f})")
                elif rho_delta > 0:
                    print(f"  ⬆️ Do-nothing would increase rho: {max_rho:.3f} → {do_nothing_rho:.3f} (+{rho_delta:.3f})")
                elif rho_delta < 0:
                    print(f"  ⬇️ Do-nothing would decrease rho: {max_rho:.3f} → {do_nothing_rho:.3f} ({rho_delta:.3f})")
                else:
                    print(f"  ➡️ Do-nothing would maintain rho: {max_rho:.3f}")
            except Exception as e:
                print(f"  ⚠️ Could not simulate do-nothing: {e}")
            
            print(f"  Running MCTS with {self.config['mcts_simulations']} simulations (epsilon={epsilon}, threshold={critical_threshold})...")
            
            root, stats = run_mcts(
                env=self.env,
                observation=obs,
                action_catalog=self.action_catalog,
                num_simulations=self.config['mcts_simulations'],
                c_puct=self.config['c_puct'],  # Fixed: use 'c_puct' not 'puct_c'
                gamma=self.config['gamma'],
                value_fn=lambda o: self.get_policy_value(o)[1],  # Use NN value
                max_depth=self.config['max_depth'],
                epsilon=epsilon,
                verbose=True,  # Enable to see pre-filter messages
                critical_threshold=critical_threshold,
                t_skipped=self.config.get('t_skipped', 80),  # Use config value or default
                t_stopping=self.config.get('t_stopping', 30),  # Use config value or default
                auto_reconnect=self.config.get('auto_reconnect', True),
                max_reconnections=self.config.get('max_reconnections_per_action', 1),
                prefilter_rho_increase=self.config.get('action_prefilter_rho_increase', 0.15)
            )
            
            # Print tree statistics for debugging
            recovery_info = f" ({stats.get('recovery_nodes', 0)} recovery nodes)" if stats.get('recovery_nodes', 0) > 0 else ""
            early_stop_info = f" - early stopped at {stats.get('simulations_run', 0)}/{self.config['mcts_simulations']}" if stats.get('simulations_run', 0) < self.config['mcts_simulations'] else ""
            print(f"  MCTS complete: {len(root.children)} root children, max_depth={stats['max_depth']}{recovery_info}{early_stop_info}")
            print(f"    Tree structure:")
            for d in range(min(10, stats['max_depth']+1)):  # Show up to depth 10
                total = stats['nodes_by_depth'].get(d, 0)
                terminal = stats['terminal_by_depth'].get(d, 0)
                visits = stats['visits_by_depth'].get(d, 0)
                print(f"      Depth {d}: {total} nodes ({terminal} terminal, {visits} visits)")
            
            # Show root visit distribution and max_reachable_steps
            if stats['root_children_visits']:
                visits = [c['visits'] for c in stats['root_children_visits']]
                values = [c['value'] for c in stats['root_children_visits']]
                max_steps = [c['max_reachable_steps'] for c in stats['root_children_visits']]
                print(f"    Root visits: max={max(visits)}, mean={sum(visits)/len(visits):.1f}, >10={sum(1 for v in visits if v > 10)}")
                print(f"    Root Q-values: max={max(values):.3f}, min={min(values):.3f}, mean={sum(values)/len(values):.3f}")
                print(f"    Max reachable steps: max={max(max_steps)}, mean={sum(max_steps)/len(max_steps):.1f}")
                
                # Show top 3 actions by max_reachable_steps (most important metric!)
                sorted_children = sorted(stats['root_children_visits'], key=lambda x: x['max_reachable_steps'], reverse=True)[:3]
                print(f"    Top 3 actions by max_reachable_steps:")
                for i, child_info in enumerate(sorted_children, 1):
                    action_idx = child_info['action_idx']
                    print(f"      {i}. Action {action_idx}: max={child_info['max_reachable_steps']} steps, {child_info['visits']} visits, Q={child_info['value']:.3f}")
            
            # Get action probabilities from visit counts
            visits = np.array([root.children.get(i, MCTSNodeV2(None, None)).visit_count 
                              for i in range(self.num_actions)])
            
            # Normalize to get policy
            total_visits = visits.sum()
            if total_visits > 0:
                mcts_policy = visits / total_visits
            else:
                mcts_policy = np.ones(self.num_actions) / self.num_actions
            
            # Calculate policy entropy
            policy_nonzero = mcts_policy[mcts_policy > 0]
            policy_entropy = -np.sum(policy_nonzero * np.log(policy_nonzero + 1e-10))
            max_entropy = np.log(self.num_actions)
            normalized_entropy = policy_entropy / max_entropy
            print(f"  🎯 Policy entropy: {policy_entropy:.3f} (normalized: {normalized_entropy:.3f})")
            if normalized_entropy > 0.9:
                print(f"    ⚠️ Warning: Policy is very uniform (entropy close to max)")
            
            # Count terminal children for debugging
            terminal_actions = sum(1 for child in root.children.values() if child.is_terminal)
            if terminal_actions > 0:
                print(f"    ⚠️ Filtered out {terminal_actions} terminal actions (immediate failures)")
            
            # Select action using max_reachable_steps (greedy, no epsilon here - epsilon only for MCTS exploration)
            from training.alphazero_mcts_v2 import select_action
            action_idx = select_action(root, temperature=0, epsilon=0.0)  # Greedy selection for actual action
            
            # Get MCTS policy target based on config
            policy_target_method = self.config.get('policy_target_method', 'visits')
            policy_temperature = self.config.get('policy_temperature', 1.0)
            
            if policy_target_method == 'one_hot':
                # One-hot: target the selected action (which has best max_reachable_steps)
                mcts_policy = np.zeros(self.num_actions)
                mcts_policy[action_idx] = 1.0
                
            elif policy_target_method == 'max_steps':
                # Distribution based on max_reachable_steps with temperature sharpening
                max_reachable_values = np.array([root.children.get(i, MCTSNodeV2(None, None)).max_reachable_steps 
                                     for i in range(self.num_actions)])
                
                if max_reachable_values.max() > 0:
                    # Apply temperature sharpening (lower temp = more peaked toward best action)
                    # Shift to positive values to avoid issues with negative powers
                    max_steps_shifted = max_reachable_values + 1  # Add 1 to avoid zero
                    steps_powered = np.power(max_steps_shifted, 1.0/policy_temperature)
                    mcts_policy = steps_powered / steps_powered.sum()
                else:
                    # Fallback if no valid actions
                    mcts_policy = np.zeros(self.num_actions)
                    mcts_policy[action_idx] = 1.0
                    
            elif policy_target_method == 'visits_with_selection_bias':
                # Visit distribution with extra weight on selected action
                if total_visits > 0:
                    visits_powered = np.power(visits, 1.0/policy_temperature)
                    # Add selection bias: boost the actually selected action (but not action 0 - do-nothing)
                    if action_idx != 0:
                        selection_bias = self.config.get('selection_bias_weight', 0.5)
                        visits_powered[action_idx] *= (1.0 + selection_bias)
                    mcts_policy = visits_powered / visits_powered.sum()
                else:
                    mcts_policy = np.ones(self.num_actions) / self.num_actions
                    
            else:  # 'visits' or default
                # Visit distribution: already computed above, just apply temperature
                if total_visits > 0:
                    visits_powered = np.power(visits, 1.0/policy_temperature)
                    mcts_policy = visits_powered / visits_powered.sum()
                # else: mcts_policy already set to uniform above
            
            if action_idx in root.children:
                selected_max_steps = root.children[action_idx].max_reachable_steps
                selected_child = root.children[action_idx]
                terminal_flag = " [TERMINAL]" if selected_child.is_terminal else ""
                print(f"  ✅ Selected action {action_idx} (max_steps={selected_max_steps}, visits={visits[action_idx]}{terminal_flag})")
            else:
                print(f"  ✅ Selected action {action_idx} (default/fallback)")
            
            # Store training example with either MCTS value or placeholder for binary outcome
            state_vector = encode_observation_simple(obs)
            
            if self.config.get('use_mcts_values', True):
                # Use MCTS Q-values (AlphaZero approach)
                root_value = root.value()  # Average Q-value from all MCTS simulations
                episode_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': root_value,  # MCTS value estimate
                })
            else:
                # Use binary episode outcomes (will be set after episode completes)
                episode_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': 0.0,  # Placeholder - will be updated with episode outcome
                })
            
            # Get line loads before action
            rho_before = obs.rho.copy()
            top_lines_before = np.argsort(rho_before)[-3:][::-1]
            
            # Take action in environment
            action = self.action_catalog.actions[action_idx]
            grid2op_action = action.apply(self.env.action_space)
            
            print(f"  📋 Action {action_idx}: {str(action)[:80]}")
            
            obs, reward, done, info = self.env.step(grid2op_action)
            step += 1
            
            # Show line load changes
            if not done:
                rho_after = obs.rho.copy()
                top_lines_after = np.argsort(rho_after)[-3:][::-1]
                actual_rho_change = rho_after.max() - rho_before.max()
                
                print(f"  📉 Line loads:")
                print(f"    Before (top 3): " + ", ".join([f"L{i}={rho_before[i]:.2f}" for i in top_lines_before]))
                print(f"    After  (top 3): " + ", ".join([f"L{i}={rho_after[i]:.2f}" for i in top_lines_after]))
                print(f"    Max change: {actual_rho_change:+.3f}")
                
                # Compare with do-nothing
                if 'do_nothing_rho' in locals():
                    do_nothing_change = do_nothing_rho - rho_before.max()
                    actual_change = rho_after.max() - rho_before.max()
                    if actual_change < do_nothing_change:
                        improvement = do_nothing_change - actual_change
                        print(f"    ✅ Better than do-nothing by {improvement:.3f} (do-nothing: {do_nothing_change:+.3f}, actual: {actual_change:+.3f})")
                    elif actual_change > do_nothing_change:
                        worse = actual_change - do_nothing_change
                        print(f"    ❌ WORSE than do-nothing by {worse:.3f} (do-nothing: {do_nothing_change:+.3f}, actual: {actual_change:+.3f})")
                    else:
                        print(f"    ➡️ Same as do-nothing ({actual_change:+.3f})")

            
            if done:
                # Check if episode ended due to failure or natural completion
                is_failure = (info.get('is_illegal', False) or 
                             info.get('is_ambiguous', False) or 
                             'exception' in info)
                
                if is_failure:
                    print(f"\n  ❌ Episode FAILED at step {step}")
                    print(f"    Reason: {info.get('exception', 'Unknown')}")
                    print(f"    Is illegal: {info.get('is_illegal', False)}")
                    print(f"    Is ambiguous: {info.get('is_ambiguous', False)}")
                else:
                    print(f"\n  ✅ Episode completed naturally at step {step}")
                break
        
        # Episode complete - assign final values based on configuration
        if not self.config.get('use_mcts_values', True) and len(episode_data) > 0:
            # Binary episode outcome assignment: +1 for success, -1 for failure
            episode_outcome = 1.0 if not is_failure else -1.0
            for data in episode_data:
                data['value'] = episode_outcome
            print(f"  📊 Assigned binary values: {episode_outcome} to all {len(episode_data)} states")
        
        print(f"\n✅ Episode Summary:")
        print(f"  Steps: {step}/{self.config['max_episode_steps']}")
        print(f"  MCTS states collected: {len(episode_data)}")
        print(f"  Success: {'Yes' if not is_failure else 'No'}")
        print(f"  Value assignment: {'MCTS Q-values' if self.config.get('use_mcts_values', True) else 'Binary outcomes'}")
        
        if len(episode_data) > 0:
            # Show value distribution
            values = [data['value'] for data in episode_data]
            print(f"  Value range: [{min(values):.3f}, {max(values):.3f}], mean={np.mean(values):.3f}")
        
        # Store ENTIRE EPISODE as a unit in replay buffer (always store for collection)
        # We'll decide whether to use it for multi-iteration training based on use_replay_buffer flag
        if len(episode_data) > 0:
            self.replay_buffer.append(episode_data)
        
        return len(episode_data), 1.0 if not is_failure else -1.0  # Episode outcome for logging only
    
    def train_network(self, iteration, current_iteration_data=None):
        """Train neural network on replay buffer or just current iteration's data."""
        if not self.use_replay_buffer:
            # Train only on current iteration's data
            if not current_iteration_data or len(current_iteration_data) == 0:
                print(f"No current iteration data, skipping training")
                return None
            print(f"\n{'='*60}")
            print(f"TRAINING NEURAL NETWORK - Iteration {iteration}")
            print(f"{'='*60}")
            print(f"Training on {len(current_iteration_data)} experiences (no replay buffer)")
            training_examples = [
                {
                    'state': d['state'],
                    'mcts_policy': d['policy'],
                    'value': d['value']
                } for d in current_iteration_data
            ]
        else:
            # Train on experiences from replay buffer (episode-based)
            if len(self.replay_buffer) == 0:
                print(f"Replay buffer is empty, skipping training")
                return None
            
            # Flatten all episodes in replay buffer to get individual experiences
            all_experiences = []
            for episode in self.replay_buffer:
                all_experiences.extend(episode)
            
            print(f"\n{'='*60}")
            print(f"TRAINING NEURAL NETWORK - Iteration {iteration}")
            print(f"{'='*60}")
            print(f"Replay buffer: {len(self.replay_buffer)} episodes, {len(all_experiences)} total experiences")
            print(f"Current learning rate: {self.learning_rate:.6f}")
            
            # Use ALL experiences (will be batched inside train_neural_network)
            training_examples = []
            for data in all_experiences:
                # Handle both 'policy' and 'mcts_policy' keys (parallel vs sequential)
                policy = data.get('mcts_policy', data.get('policy'))
                training_examples.append({
                    'state': data['state'],
                    'mcts_policy': policy,
                    'value': data['value']
                })
        
        # Train using the persistent optimizer
        loss_info = train_neural_network(
            self.neural_network,
            training_examples,
            self.config,
            optimizer=self.optimizer  # Pass our persistent optimizer
        )
        
        # Log value statistics for debugging
        value_targets = [ex['value'] for ex in training_examples]
        value_min, value_max, value_mean = np.min(value_targets), np.max(value_targets), np.mean(value_targets)
        
        print(f"Training losses:")
        print(f"  Policy loss: {loss_info['policy_loss']:.4f}")
        print(f"  Value loss: {loss_info['value_loss']:.4f}")
        print(f"  Total loss: {loss_info['total_loss']:.4f}")
        print(f"  Value targets - min: {value_min:.3f}, max: {value_max:.3f}, mean: {value_mean:.3f}")
        return loss_info
    
    def decay_learning_rate(self):
        """Apply learning rate decay."""
        old_lr = self.learning_rate
        self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        
        if old_lr != self.learning_rate:
            print(f"📉 Learning rate decayed: {old_lr:.9f} → {self.learning_rate:.9f}")
        else:
            print(f"📉 Learning rate at minimum: {self.learning_rate:.9f}")
    
    def decay_temperature(self):
        """Apply exploration temperature decay."""
        old_temp = self.current_temperature
        temp_decay = self.config.get('exploration_temperature_decay', 1.0)
        min_temp = self.config.get('min_temperature', 0.0)
        self.current_temperature = max(min_temp, self.current_temperature * temp_decay)
        
        if old_temp != self.current_temperature:
            print(f"🌡️ Temperature decayed: {old_temp:.3f} → {self.current_temperature:.3f}")
    
    def self_play_parallel(self, num_episodes, num_workers=None):
        """
        Collect multiple episodes in parallel using multiprocessing.
        
        Args:
            num_episodes: Number of episodes to collect
            num_workers: Number of parallel workers (default: min(cpu_count, num_episodes))
        
        Returns:
            total_collected: Total number of training examples collected
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), num_episodes)
        
        print(f"\n⚡ PARALLEL EPISODE COLLECTION")
        print(f"  Episodes: {num_episodes}")
        print(f"  Workers: {num_workers}")
        
        # Create config copy with shared network weights
        worker_config = self.config.copy()
        worker_config['network_state_dict'] = self.neural_network.state_dict()
        
        # Prepare worker arguments
        worker_args = []
        for episode in range(num_episodes):
            chronic_id = self.train_chronics[self.current_chronic_idx % len(self.train_chronics)]
            self.current_chronic_idx += 1
            worker_args.append((chronic_id, episode, worker_config, episode))
        
        # Run parallel collection
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(_parallel_episode_worker, worker_args)
            
            # Process results
            total_examples = 0
            successful = 0
            failed = 0
            
            for result in results:
                if result['success']:
                    successful += 1
                    episode_data = result['episode_data']
                    
                    # Convert to expected format and add to replay buffer
                    formatted_data = []
                    for data in episode_data:
                        formatted_data.append({
                            'state': data['state'],
                            'mcts_policy': data['policy'],
                            'value': data['value']
                        })
                    
                    if len(formatted_data) > 0:
                        self.replay_buffer.append(formatted_data)
                        total_examples += len(formatted_data)
                    
                    print(f"  ✅ Chronic {result['chronic_id']}: {len(episode_data)} examples, {result['steps']} steps")
                else:
                    failed += 1
                    print(f"  ❌ Chronic {result['chronic_id']}: FAILED")
                    if result['error']:
                        print(f"     Error: {result['error'][:150]}...")
            
            print(f"\n  Summary: {successful}/{num_episodes} successful, {total_examples} total examples")
            return total_examples
            
        except Exception as e:
            print(f"❌ Parallel collection failed: {e}")
            print(traceback.format_exc())
            # Fall back to sequential
            print("⚠️ Falling back to sequential collection...")
            for episode in range(num_episodes):
                self.self_play_episode(episode)
            return 0
    
    def train(self, num_iterations):
        """Run full AlphaZero training loop."""
        print(f"\n{'='*60}")
        print(f"ALPHAZERO TRAINING V2")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Episodes per iteration: {self.config['episodes_per_iteration']}")
        print(f"MCTS simulations: {self.config['mcts_simulations']}")
        
        # Check if parallel training is enabled
        parallel_workers = self.config.get('parallel_workers', 0)
        use_parallel = parallel_workers > 0
        
        if use_parallel:
            print(f"⚡ Parallel mode: {parallel_workers} workers")
        else:
            print(f"🔄 Sequential mode")
        print()
        
        for iteration in range(num_iterations):
            print(f"\n{'#'*60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'#'*60}")
            
            # Self-play phase - parallel or sequential
            current_iteration_data = []
            
            if use_parallel:
                # PARALLEL COLLECTION
                self.self_play_parallel(
                    num_episodes=self.config['episodes_per_iteration'],
                    num_workers=parallel_workers
                )
            else:
                # SEQUENTIAL COLLECTION (original behavior)  
                for episode in range(self.config['episodes_per_iteration']):
                    episode_len, episode_value = self.self_play_episode(episode)
                    # Collect all episode data if not using replay buffer
                    if not self.use_replay_buffer:
                        # Get the last episode from replay buffer (the episode we just played)
                        if len(self.replay_buffer) > 0:
                            current_iteration_data.extend(self.replay_buffer[-1])  # Last episode
            
            # Training phase (always sequential)
            self.train_network(iteration, current_iteration_data=current_iteration_data if not self.use_replay_buffer else None)
            
            # Decay learning rate and temperature after training
            self.decay_learning_rate()
            self.decay_temperature()
            
            # Save checkpoint
            if (iteration + 1) % self.config['save_every'] == 0:
                checkpoint_path = f"/workspace/checkpoints/alphazero_v2_iter{iteration+1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': self.neural_network.state_dict(),
                    'replay_buffer_size': len(self.replay_buffer),
                    'num_actions': self.num_actions,
                    'input_size': self.input_size,
                    'hidden_size': self.config.get('hidden_size', 256),
                }, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")


def main():
    # Set random seeds for reproducibility
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup environment
    env_name = "l2rpn_case14_sandbox"
    print(f"Loading environment: {env_name}")
    
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        param=params
    )
    
    # Set Grid2Op seed for deterministic chronics
    env.seed(seed)
    
    # Build action catalog from config
    from config import ACTIONS_CONFIG
    substations = ACTIONS_CONFIG['substations']
    reduction = ACTIONS_CONFIG['reduction']
    drop_identity = ACTIONS_CONFIG['drop_identity']
    include_do_nothing = ACTIONS_CONFIG.get('include_do_nothing', True)
    
    print(f"Building action catalog: substations={substations}, reduction={reduction}, drop_identity={drop_identity}, include_do_nothing={include_do_nothing}")
    catalog = build_action_catalog(env, substations=substations, reduction=reduction, 
                                   drop_identity=drop_identity, include_do_nothing=include_do_nothing)
    print(f"Action catalog: {len(catalog.actions)} actions")
    if include_do_nothing:
        print(f"  Action 0: do-nothing (explicit)")
        print(f"  Actions 1-{len(catalog.actions)-1}: topology changes")
    
    # Load training configuration from config.py
    from config import AGENT_CONFIG, TRAINING_CONFIG
    
    config = {
        # MCTS parameters from config.py
        'mcts_simulations': AGENT_CONFIG['mcts_simulations'],
        'c_puct': AGENT_CONFIG['puct_c'],
        'gamma': AGENT_CONFIG['gamma'],
        'max_depth': AGENT_CONFIG['max_depth'],
        'temperature': AGENT_CONFIG['temperature'],
        'critical_threshold': AGENT_CONFIG['critical_threshold'],  # Only act when rho > threshold
        'mcts_epsilon': AGENT_CONFIG['mcts_epsilon'],  # Epsilon-greedy exploration in MCTS
        'action_prefilter_rho_increase': AGENT_CONFIG.get('action_prefilter_rho_increase', 0.15),  # Pre-filter bad actions
        't_skipped': AGENT_CONFIG['t_skipped'],  # Recovery node threshold
        't_stopping': AGENT_CONFIG['t_stopping'],  # Early stopping threshold
        
        # Policy target parameters
        'policy_target_method': AGENT_CONFIG.get('policy_target_method', 'visits'),
        'policy_temperature': AGENT_CONFIG.get('policy_temperature', 1.0),


        # Training parameters
        'episodes_per_iteration': AGENT_CONFIG['episodes_per_iteration'],  # From config
        'parallel_workers': AGENT_CONFIG.get('parallel_workers', 0),  # Number of parallel workers for episode collection
        'max_episode_steps': TRAINING_CONFIG.get('max_steps_per_episode', 10000) or 10000,  # From TRAINING_CONFIG, default 10000 if None
        'replay_buffer_size': AGENT_CONFIG.get('replay_buffer_size', 100),  # From config
        'use_replay_buffer': AGENT_CONFIG.get('use_replay_buffer', True),  # From config
        'batch_size': AGENT_CONFIG['batch_size'],
        'learning_rate': AGENT_CONFIG['learning_rate'],
        'learning_rate_decay': AGENT_CONFIG.get('learning_rate_decay', 1.0),
        'min_learning_rate': AGENT_CONFIG['min_learning_rate'],
        'weight_decay': AGENT_CONFIG['weight_decay'],
        'training_epochs': AGENT_CONFIG['training_epochs'],  # Correct key name
        
        # Checkpointing
        'save_every': 1,  # Save checkpoint every iteration
        
        # Neural network architecture
        'hidden_size': AGENT_CONFIG['hidden_size'],  # Use singular to match network code
        'dropout': 0.0,
    }
    
    print(f"\nTraining configuration:")
    print(f"  MCTS simulations: {config['mcts_simulations']}")
    print(f"  c_puct: {config['c_puct']}")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()
    
    # Create trainer
    trainer = AlphaZeroTrainerV2(env, catalog, config)
    
    # Run training
    num_cycles = AGENT_CONFIG.get('num_cycles', 50)
    episodes_per_iteration = AGENT_CONFIG.get('episodes_per_iteration', 2)
    
    # Calculate total iterations: (num_cycles * train_chronics) / episodes_per_iteration
    num_train_chronics = len(trainer.train_chronics)
    total_iterations = (num_cycles * num_train_chronics) // episodes_per_iteration
    
    print(f"Training plan:")
    print(f"  Training chronics: {num_train_chronics}")
    print(f"  Cycles through all chronics: {num_cycles}")
    print(f"  Episodes per iteration: {episodes_per_iteration}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Total episodes: {total_iterations * episodes_per_iteration}")
    print()
    
    trainer.train(num_iterations=total_iterations)


if __name__ == "__main__":
    # Set multiprocessing start method for parallel training
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
