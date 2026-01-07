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
from training.alphazero_mcts_v2 import run_mcts, select_action, MCTSNodeV2, collect_all_tree_nodes, compute_heuristic_value
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
        params.NO_OVERFLOW_DISCONNECTION = False
        env = grid2op.make(
            "l2rpn_case14_sandbox",
            backend=LightSimBackend(),
            param=params
        )
        
        # Build action catalog
        from config import ACTIONS_CONFIG, USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS
        full_catalog = build_action_catalog(
            env,
            substations=ACTIONS_CONFIG['substations'],
            reduction=ACTIONS_CONFIG['reduction'],
            drop_identity=ACTIONS_CONFIG['drop_identity'],
            include_do_nothing=ACTIONS_CONFIG.get('include_do_nothing', True)
        )
        
        # Apply reduced action space if enabled
        if USE_REDUCED_ACTION_SPACE:
            from dataclasses import replace
            reduced_actions = [full_catalog.actions[idx] for idx in REDUCED_ACTIONS if idx < len(full_catalog.actions)]
            catalog = replace(full_catalog, actions=reduced_actions)
        else:
            catalog = full_catalog
        
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
        
        # DISABLED: Don't load any network weights - use fresh random initialization
        # Each worker starts with its own random network (pure exploration/heuristic mode)
        # if 'network_state_dict' in config:
        #     neural_network.load_state_dict(config['network_state_dict'])
        #     neural_network.eval()  # Set to evaluation mode for inference
        
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
        cumulative_reward = 0.0
        
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
                    policy_fn=None,  # Heuristic mode: uniform priors
                    value_fn=None,   # Heuristic mode: use heuristic value function
                    critical_threshold=critical_threshold,
                    dirichlet_alpha=config.get('dirichlet_alpha', 0.3),
                    dirichlet_epsilon=config.get('dirichlet_epsilon', 0.25),
                    penalty_for_failure=config.get('penalty_for_failure', -5.0)
                )
                
                # Track tree depth statistics for this episode
                if 'tree_depths' not in locals():
                    tree_depths = []
                tree_depths.append(stats.get('max_depth', 0))
                
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
                        if policy_temperature == 0:
                            # Greedy: one-hot on action with max reachable steps
                            mcts_policy = np.zeros(num_actions)
                            mcts_policy[action_idx] = 1.0
                        else:
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
                        if policy_temperature == 0:
                            # Greedy: one-hot on selected action
                            mcts_policy = np.zeros(num_actions)
                            mcts_policy[action_idx] = 1.0
                        else:
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
                        if policy_temperature == 0:
                            # Greedy: one-hot on selected action
                            mcts_policy = np.zeros(num_actions)
                            mcts_policy[action_idx] = 1.0
                        else:
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
                cumulative_reward += reward
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
                cumulative_reward += reward
                step += 1
        
        env.close()
        
        # Print completion summary with statistics
        avg_tree_depth = sum(tree_depths) / len(tree_depths) if 'tree_depths' in locals() and tree_depths else 0
        print(f"  [Worker {worker_id}] Completed: Chronic {chronic_id}", flush=True)
        print(f"    Steps: {step}, Critical states: {critical_states_count}, Training examples: {len(episode_data)}", flush=True)
        print(f"    Total reward: {cumulative_reward:.2f}, Max rho: {max_rho_seen:.3f}, Avg tree depth: {avg_tree_depth:.1f}", flush=True)
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
        self.input_size = 117  # From encode_observation_simple (60 line features + 57 topology bits)
        self.num_actions = len(action_catalog.actions)
        self.neural_network = create_neural_network(
            input_size=self.input_size,
            num_actions=self.num_actions,
            config=config
        )
        
        # Load from checkpoint if model_path is specified
        # DISABLED: Start training from scratch with random initialization
        # from config import AGENT_CONFIG
        # model_path = AGENT_CONFIG.get('model_path', None)
        # if model_path and os.path.exists(model_path):
        #     print(f"Loading checkpoint from: {model_path}")
        #     checkpoint = torch.load(model_path, map_location='cpu')
        #     # Handle both 'model_state_dict' and 'network_state_dict' keys
        #     state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'network_state_dict'
        #     self.neural_network.load_state_dict(checkpoint[state_dict_key])
        #     print(f"✓ Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")
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
        
        # Split chronics into train/test with random test selection
        from config import TRAINING_CONFIG
        total_chronics = len(env.chronics_handler.real_data.subpaths)
        train_test_split = TRAINING_CONFIG.get('train_test_split', 0.9)
        num_test_chronics = TRAINING_CONFIG.get('num_test_chronics', None)
        chronic_seed = TRAINING_CONFIG.get('chronic_seed', None)
        
        # Create split: first X% for training pool, remaining for test pool
        train_pool_size = int(total_chronics * train_test_split)
        self.train_chronics = list(range(train_pool_size))
        
        # Get test pool from remaining chronics
        test_pool = list(range(train_pool_size, total_chronics))
        
        # Use seed for deterministic selection if provided
        import random
        if chronic_seed is not None:
            random.seed(chronic_seed)
            
        # Select test chronics: either a random subset or all test chronics
        if num_test_chronics is None:
            # Use all chronics from test pool
            self.test_chronics = test_pool
        elif len(test_pool) >= num_test_chronics:
            # Randomly select subset from test pool
            self.test_chronics = sorted(random.sample(test_pool, num_test_chronics))
        else:
            # Test pool smaller than requested, use all
            self.test_chronics = test_pool
            print(f"⚠️  Warning: Only {len(test_pool)} chronics available for testing (requested {num_test_chronics})")
        
        # Shuffle training chronics to prevent temporal correlation
        if chronic_seed is not None:
            random.seed(chronic_seed)
        random.shuffle(self.train_chronics)
        
        self.current_chronic_idx = 0  # Index into train_chronics list
        print(f"Created neural network: input={self.input_size}, actions={self.num_actions}")
        print(f"Chronic selection (seed={chronic_seed}):")
        print(f"  Total chronics: {total_chronics}")
        print(f"  Training pool: 0-{train_pool_size-1} ({train_pool_size} chronics, SHUFFLED)")
        print(f"  Test pool: {train_pool_size}-{total_chronics-1} ({len(test_pool)} chronics)")
        print(f"  Test chronics selected: {len(self.test_chronics)} chronics - {self.test_chronics}")
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
    
    def self_play_episode(self, episode_id, enable_debug=False):
        """Run one self-play episode collecting training data.
        
        Args:
            episode_id: Episode number for logging
            enable_debug: If True, enable verbose debug output in MCTS (default False for parallel mode)
        """
        print(f"\n{'='*60}")
        print(f"SELF-PLAY EPISODE {episode_id}")
        print(f"{'='*60}")
        
        # Reset environment with next training chronic (cycle through ALL training chronics)
        chronic_id = self.train_chronics[self.current_chronic_idx % len(self.train_chronics)]
        
        # Re-shuffle training chronics at the start of each new cycle for variety
        if self.current_chronic_idx > 0 and self.current_chronic_idx % len(self.train_chronics) == 0:
            import random
            import time
            # Use current time as seed for randomization at each cycle
            cycle_seed = int(time.time() * 1000) % (2**32)
            random.seed(cycle_seed)
            random.shuffle(self.train_chronics)
            print(f"🔀 Starting new cycle - reshuffled training chronics with seed {cycle_seed}")
        
        print(f"Using training chronic {chronic_id} ({self.current_chronic_idx + 1}/{len(self.train_chronics)} in cycle)")
        self.env.set_id(chronic_id)
        obs = self.env.reset()
        self.current_chronic_idx += 1
        
        done = False
        step = 0
        episode_data = []
        is_failure = False  # Track if episode ended due to failure
        cumulative_reward = 0.0  # Track total episode reward
        
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
                        cumulative_reward += reward
                    except Exception as e:
                        print(f"  ⚠️ Topology reset failed: {e}, using do-nothing")
                        obs, reward, done, info = self.env.step(self.env.action_space())
                        cumulative_reward += reward
                else:
                    print(f"  ⏭️ Skipping MCTS - grid is safe, taking do-nothing action")
                    # Take do-nothing action
                    obs, reward, done, info = self.env.step(self.env.action_space())
                    cumulative_reward += reward
                step += 1
                continue
            
            # Run MCTS with neural network (only in critical states)
            epsilon = self.config.get('mcts_epsilon', 0.0)
            critical_threshold = self.config.get('critical_threshold', 0.90)
            
            # COMPARISON: What would do-nothing do?
            do_nothing_failed = False
            try:
                do_nothing_action = self.env.action_space()
                sim_obs, sim_reward, sim_done, sim_info = obs.simulate(do_nothing_action)
                do_nothing_rho = sim_obs.rho.max()
                rho_delta = do_nothing_rho - max_rho
                if sim_done:
                    print(f"  🚫 Do-nothing would FAIL (rho={do_nothing_rho:.3f})")
                    do_nothing_failed = True
                elif rho_delta > 0:
                    print(f"  ⬆️ Do-nothing would increase rho: {max_rho:.3f} → {do_nothing_rho:.3f} (+{rho_delta:.3f})")
                elif rho_delta < 0:
                    print(f"  ⬇️ Do-nothing would decrease rho: {max_rho:.3f} → {do_nothing_rho:.3f} ({rho_delta:.3f})")
                else:
                    print(f"  ➡️ Do-nothing would maintain rho: {max_rho:.3f}")
            except Exception as e:
                print(f"  ⚠️ Could not simulate do-nothing: {e}")
                do_nothing_failed = True
            
            # Determine value function based on value_target_method
            value_method = self.config.get('value_target_method', 'mcts_root')
            if value_method == 'heuristic':
                # Use heuristic value function during MCTS search
                from training.alphazero_mcts_v2 import compute_heuristic_value
                from rewards.custom_reward import MyCustomReward
                heuristic_reward_fn = MyCustomReward()
                
                def heuristic_value_fn(observation):
                    # Compute actual reward for this observation
                    actual_reward = heuristic_reward_fn(
                        action=None,
                        env=None,
                        has_error=False,
                        is_done=False,
                        is_illegal=False,
                        is_ambiguous=False,
                        obs=observation
                    )
                    # Create temporary node with actual reward
                    temp_node = MCTSNodeV2(
                        env=None, 
                        observation=observation,
                        edge_reward=actual_reward  # Use actual reward based on state
                    )
                    return compute_heuristic_value(
                        temp_node,
                        gamma=self.config.get('gamma', 0.95),
                        horizon=self.config.get('heuristic_value_horizon', 100)
                    )
                value_fn = heuristic_value_fn
                # Use NN for policy priors (should be ~uniform at initialization)
                policy_fn = lambda o: self.get_policy_value(o)[0]
            else:
                # Use neural network for both policy and value
                policy_fn = lambda o: self.get_policy_value(o)[0]
                value_fn = lambda o: self.get_policy_value(o)[1]
            
            print(f"  Running MCTS with {self.config['mcts_simulations']} simulations (epsilon={epsilon}, threshold={critical_threshold})...")
            
            root, stats = run_mcts(
                env=self.env,
                observation=obs,
                action_catalog=self.action_catalog,
                num_simulations=self.config['mcts_simulations'],
                c_puct=self.config['c_puct'],  # Fixed: use 'c_puct' not 'puct_c'
                gamma=self.config['gamma'],
                policy_fn=policy_fn,  # NN policy for priors
                value_fn=value_fn,    # Heuristic or NN value
                max_depth=self.config['max_depth'],
                epsilon=epsilon,
                verbose=True,  # Enable to see pre-filter messages
                critical_threshold=critical_threshold,
                t_skipped=self.config.get('t_skipped', 80),  # Use config value or default
                t_stopping=self.config.get('t_stopping', 30),  # Use config value or default
                auto_reconnect=self.config.get('auto_reconnect', True),
                max_reconnections=self.config.get('max_reconnections_per_action', 1),
                prefilter_rho_increase=self.config.get('action_prefilter_rho_increase', 0.15),
                dirichlet_alpha=self.config.get('dirichlet_alpha', 0.3),
                dirichlet_epsilon=self.config.get('dirichlet_epsilon', 0.25),
                penalty_for_failure=self.config.get('penalty_for_failure', -5.0),
                enable_debug=enable_debug  # Only show debug in sequential mode
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
            
            # Select action using GREEDY selection (best action from MCTS)
            # Temperature is used only for policy target, not action selection
            from training.alphazero_mcts_v2 import select_action
            action_idx = select_action(root, temperature=0, epsilon=0.0)  # Greedy: always pick best action
            
            # Get MCTS policy target based on config
            policy_target_method = self.config.get('policy_target_method', 'visits')
            # Use decaying temperature for policy target (exploration in training)
            policy_temperature = self.current_temperature
            
            if policy_target_method == 'one_hot':
                # One-hot: target the selected action (which has best max_reachable_steps)
                mcts_policy = np.zeros(self.num_actions)
                mcts_policy[action_idx] = 1.0
                
            elif policy_target_method == 'max_steps':
                # Distribution based on max_reachable_steps with temperature sharpening
                max_reachable_values = np.array([root.children.get(i, MCTSNodeV2(None, None)).max_reachable_steps 
                                     for i in range(self.num_actions)])
                
                if max_reachable_values.max() > 0:
                    if policy_temperature == 0:
                        # Greedy: one-hot on selected action
                        mcts_policy = np.zeros(self.num_actions)
                        mcts_policy[action_idx] = 1.0
                    else:
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
                    if policy_temperature == 0:
                        # Greedy: one-hot on action with most visits
                        mcts_policy = np.zeros(self.num_actions)
                        mcts_policy[action_idx] = 1.0
                    else:
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
            
            # Store training examples based on value_target_method
            value_method = self.config.get('value_target_method', 'mcts_root')
            
            if value_method == 'mcts_root':
                # Original: Use MCTS Q-value from root node only
                state_vector = encode_observation_simple(obs)
                root_value = root.value()
                episode_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': root_value,
                    'source': 'mcts_root'
                })
                
            elif value_method == 'binary_root':
                # Original: Use binary outcome from root node only (set after episode)
                state_vector = encode_observation_simple(obs)
                episode_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': 0.0,  # Placeholder - updated after episode
                    'source': 'binary_root'
                })
                
            elif value_method == 'mcts_all_nodes':
                # Use MCTS Q-values from ALL tree nodes
                all_nodes_data = collect_all_tree_nodes(root, encode_observation_simple)
                episode_data.extend(all_nodes_data)
                print(f"  📦 Collected {len(all_nodes_data)} nodes from MCTS tree (depths 0-{max([d['depth'] for d in all_nodes_data])})")
                
            elif value_method == 'binary_all_nodes':
                # PROPER ALPHAZERO: Collect ALL tree nodes, label with final episode outcome
                # This is how AlphaZero actually works (Go/Chess/Shogi)
                all_nodes_data = collect_all_tree_nodes(root, encode_observation_simple)
                # Set placeholder values (will be updated after episode completes)
                for data in all_nodes_data:
                    data['value'] = 0.0  # Placeholder
                    data['source'] = 'binary_all_nodes'
                episode_data.extend(all_nodes_data)
                print(f"  📦 Collected {len(all_nodes_data)} nodes from MCTS tree (depths 0-{max([d['depth'] for d in all_nodes_data])}), will label with episode outcome")
                
            elif value_method == 'heuristic':
                # Heuristic mode: Only train policy, not value
                # Value is computed via heuristic formula, not learned
                state_vector = encode_observation_simple(obs)
                episode_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': None,  # No value target - we don't train value network
                    'source': 'heuristic'
                })
                print(f"  📦 Collected root for policy training (heuristic mode - no value training)")
            else:
                raise ValueError(f"Unknown value_target_method: {value_method}")
            
            # Get line loads before action
            rho_before = obs.rho.copy()
            top_lines_before = np.argsort(rho_before)[-3:][::-1]
            
            # Take action in environment
            action = self.action_catalog.actions[action_idx]
            grid2op_action = action.apply(self.env.action_space)
            
            print(f"  📋 Action {action_idx}: {str(action)[:80]}")
            
            obs, reward, done, info = self.env.step(grid2op_action)
            cumulative_reward += reward
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
                
                # Compare with do-nothing (only if do-nothing didn't fail)
                if 'do_nothing_rho' in locals() and not do_nothing_failed:
                    do_nothing_change = do_nothing_rho - rho_before.max()
                    actual_change = rho_after.max() - rho_before.max()
                    
                    # For comparison: negative = rho decreased (good), positive = rho increased (bad)
                    # Better action = more negative change (bigger decrease) or less positive change (smaller increase)
                    if actual_change < do_nothing_change:
                        # Actual is more negative (better decrease) or less positive (smaller increase)
                        improvement = do_nothing_change - actual_change
                        print(f"    ✅ Better than do-nothing by {improvement:.3f} (do-nothing: {do_nothing_change:+.3f}, actual: {actual_change:+.3f})")
                    elif actual_change > do_nothing_change:
                        # Actual is less negative (worse decrease) or more positive (bigger increase)
                        worse = actual_change - do_nothing_change
                        print(f"    ❌ WORSE than do-nothing by {worse:.3f} (do-nothing would have changed by {do_nothing_change:+.3f}, actual: {actual_change:+.3f})")
                    else:
                        print(f"    ➡️ Same as do-nothing ({actual_change:+.3f})")
                elif do_nothing_failed:
                    print(f"    ℹ️ Any action is better than do-nothing (which would fail)")

            
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
        
        # Episode complete - assign final values based on value_target_method
        value_method = self.config.get('value_target_method', 'mcts_root')
        
        if value_method == 'binary_root' and len(episode_data) > 0:
            # Binary episode outcome assignment: +1 for success, -1 for failure (root only)
            episode_outcome = 1.0 if not is_failure else -1.0
            for data in episode_data:
                if data.get('source') == 'binary_root':  # Only update placeholders
                    data['value'] = episode_outcome
            print(f"  📊 Assigned binary outcome: {episode_outcome} to all {len(episode_data)} root states")
            
        elif value_method == 'binary_all_nodes' and len(episode_data) > 0:
            # PROPER ALPHAZERO: Binary outcome for ALL nodes in MCTS trees
            episode_outcome = 1.0 if not is_failure else -1.0
            for data in episode_data:
                if data.get('source') == 'binary_all_nodes':  # Only update placeholders
                    data['value'] = episode_outcome
            print(f"  📊 Assigned binary outcome: {episode_outcome} to all {len(episode_data)} tree nodes (AlphaZero style)")
        
        # Determine method name for logging
        method_names = {
            'mcts_root': 'MCTS Q-values (root only)',
            'binary_root': 'Binary outcomes (root only)',
            'mcts_all_nodes': 'MCTS Q-values (all tree nodes)',
            'binary_all_nodes': 'Binary outcomes (all nodes) - PROPER ALPHAZERO',
            'heuristic': 'Heuristic value function'
        }
        method_name = method_names.get(value_method, value_method)
        
        print(f"\n✅ Episode Summary:")
        print(f"  Steps: {step}/{self.config['max_episode_steps']}")
        print(f"  Total reward: {cumulative_reward:.2f}")
        print(f"  Training samples collected: {len(episode_data)}")
        print(f"  Success: {'Yes' if not is_failure else 'No'}")
        print(f"  Value assignment: {method_name}")
        
        if len(episode_data) > 0:
            # Show value distribution (skip None values for heuristic mode)
            values = [data['value'] for data in episode_data if data['value'] is not None]
            if len(values) > 0:
                print(f"  Value range: [{min(values):.3f}, {max(values):.3f}], mean={np.mean(values):.3f}")
            
            # Show depth distribution if using all nodes
            if 'depth' in episode_data[0]:
                depths = [data['depth'] for data in episode_data]
                from collections import Counter
                depth_counts = Counter(depths)
                print(f"  Samples by depth: {dict(sorted(depth_counts.items()))}")
        
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
        
        # Save training data if enabled
        save_enabled = self.config.get('save_training_data', False)
        print(f"DEBUG: save_training_data = {save_enabled}, type = {type(save_enabled)}")
        if save_enabled:
            try:
                print(f"💾 Attempting to save training data for iteration {iteration}...")
                self._save_training_data_sample(iteration, training_examples)
            except Exception as e:
                print(f"⚠️ ERROR saving training data: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"DEBUG: Training data saving is DISABLED in config")
        
        # Train using the persistent optimizer
        loss_info = train_neural_network(
            self.neural_network,
            training_examples,
            self.config,
            optimizer=self.optimizer  # Pass our persistent optimizer
        )
        
        # Log value statistics for debugging
        value_targets = [ex['value'] for ex in training_examples if ex['value'] is not None]
        if len(value_targets) > 0:
            value_min, value_max, value_mean = np.min(value_targets), np.max(value_targets), np.mean(value_targets)
        else:
            value_min, value_max, value_mean = 0, 0, 0
        
        print(f"Training losses:")
        print(f"  Policy loss: {loss_info['policy_loss']:.4f}")
        print(f"  Value loss: {loss_info['value_loss']:.4f}")
        print(f"  Total loss: {loss_info['total_loss']:.4f}")
        if len(value_targets) > 0:
            print(f"  Value targets - min: {value_min:.3f}, max: {value_max:.3f}, mean: {value_mean:.3f}")
        else:
            print(f"  Value targets - (none, using heuristic mode)")
        return loss_info
    
    def _save_training_data_sample(self, iteration, training_examples):
        """Save all training data for analysis in .npz format."""
        import numpy as np
        
        data_dir = self.config.get('training_data_dir', 'training_data_samples')
        os.makedirs(data_dir, exist_ok=True)
        
        # Convert training examples to numpy arrays
        states = np.array([ex['state'] for ex in training_examples], dtype=np.float32)
        policies = np.array([ex['mcts_policy'] for ex in training_examples], dtype=np.float32)
        values = np.array([ex['value'] for ex in training_examples], dtype=np.float32)
        
        # Save as .npz (NumPy compressed format)
        filepath = os.path.join(data_dir, f'training_data_iter_{iteration}.npz')
        np.savez_compressed(
            filepath,
            states=states,
            policies=policies,
            values=values,
            iteration=iteration,
            learning_rate=self.learning_rate,
            replay_buffer_size=len(self.replay_buffer)
        )
        
        print(f"💾 Saved {len(training_examples)} training examples to {filepath}")
        print(f"   States: {states.shape}, Policies: {policies.shape}, Values: {values.shape}")
    
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
        """Run full AlphaZero training loop with proper replay buffer logic."""
        print(f"\n{'='*60}")
        print(f"ALPHAZERO TRAINING V2 - Continuous Training")
        print(f"{'='*60}")
        print(f"Total iterations: {num_iterations}")
        print(f"MCTS simulations: {self.config['mcts_simulations']}")
        
        # Check if parallel training is enabled
        parallel_workers = self.config.get('parallel_workers', 0)
        use_parallel = parallel_workers > 0
        
        if use_parallel:
            print(f"⚡ Parallel mode: {parallel_workers} workers")
        else:
            print(f"🔄 Sequential mode")
        
        # AlphaZero-style replay buffer parameters
        replay_buffer_min_size = self.config.get('replay_buffer_min_size', 903)
        train_every = self.config.get('train_every', 300)
        
        print(f"📊 Replay buffer: min={replay_buffer_min_size}, max={self.config.get('replay_buffer_size', 3612)} (FIFO)")
        print(f"📊 Training frequency: every {train_every} episodes after buffer reaches {replay_buffer_min_size}")
        print()
        
        episodes_collected = 0  # Track total episodes
        episodes_since_last_training = 0  # Track episodes since last training
        training_iteration = 0
        
        for iteration in range(num_iterations):
            print(f"\n{'#'*60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'#'*60}")
            
            # Collect episodes - adjust based on buffer state
            replay_buffer_size = self.config.get('replay_buffer_size', 3612)
            if len(self.replay_buffer) >= replay_buffer_size:
                # Buffer is full - collect smaller batches
                episodes_per_iteration = self.config.get('train_every', 300)
            else:
                # Still filling buffer - collect full cycles
                episodes_per_iteration = self.config['episodes_per_iteration']
            
            # Self-play phase - parallel or sequential
            current_iteration_data = []
            
            if use_parallel:
                # PARALLEL COLLECTION
                self.self_play_parallel(
                    num_episodes=episodes_per_iteration,
                    num_workers=parallel_workers
                )
            else:
                # SEQUENTIAL COLLECTION (original behavior)  
                for episode in range(episodes_per_iteration):
                    episode_len, episode_value = self.self_play_episode(episode, enable_debug=True)
                    # Collect all episode data if not using replay buffer
                    if not self.use_replay_buffer:
                        # Get the last episode from replay buffer (the episode we just played)
                        if len(self.replay_buffer) > 0:
                            current_iteration_data.extend(self.replay_buffer[-1])  # Last episode
            
            episodes_collected += episodes_per_iteration
            episodes_since_last_training += episodes_per_iteration
            
            # Check if we should train
            should_train = False
            print(f"DEBUG: buffer_size={len(self.replay_buffer)}, min_size={replay_buffer_min_size}, training_iter={training_iteration}, episodes_since={episodes_since_last_training}, train_every={train_every}")
            if len(self.replay_buffer) < replay_buffer_min_size:
                print(f"📊 Buffer filling: {len(self.replay_buffer)}/{replay_buffer_min_size} episodes - NOT training yet")
            elif training_iteration == 0 or episodes_since_last_training >= train_every:
                # Train on first iteration once buffer is ready, OR every train_every episodes after that
                should_train = True
                print(f"📊 Buffer: {len(self.replay_buffer)} episodes, {episodes_since_last_training} new episodes - TRAINING")
            else:
                print(f"📊 Buffer: {len(self.replay_buffer)} episodes, {episodes_since_last_training} new episodes - waiting for {train_every - episodes_since_last_training} more")
            
            # Training phase (only if buffer is ready and enough new episodes)
            if should_train:
                self.train_network(training_iteration, current_iteration_data=current_iteration_data if not self.use_replay_buffer else None)
                
                # Save checkpoint after each training update
                if 'CHECKPOINT_DIR' in os.environ:
                    checkpoint_dir = f"/workspace/{os.environ['CHECKPOINT_DIR']}"
                else:
                    value_method = self.config.get('value_target_method', 'default')
                    checkpoint_dir = f"/workspace/checkpoints_{value_method}" if value_method != 'default' else "/workspace/checkpoints"
                checkpoint_path = f"{checkpoint_dir}/alphazero_v2_train{training_iteration}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'training_iteration': training_iteration,
                    'episodes_collected': episodes_collected,
                    'model_state_dict': self.neural_network.state_dict(),
                    'replay_buffer_size': len(self.replay_buffer),
                    'num_actions': self.num_actions,
                    'input_size': self.input_size,
                    'hidden_size': self.config.get('hidden_size', 256),
                    'config': self.config,
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
                
                training_iteration += 1
                episodes_since_last_training = 0
                
                # Decay learning rate and temperature after training
                self.decay_learning_rate()
                self.decay_temperature()
            
            # Decay learning rate and temperature after training
            self.decay_learning_rate()
            self.decay_temperature()
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")


def main():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train AlphaZero agent')
    parser.add_argument('--method', type=str, default='mcts_all_nodes',
                       choices=['heuristic', 'mcts_root', 'mcts_all_nodes', 'binary_root', 'binary_all_nodes'],
                       help='Value function method')
    parser.add_argument('--num_iterations', type=int, default=None,
                       help='Total number of training iterations (overrides calculation)')
    parser.add_argument('--episodes_per_iteration', type=int, default=None,
                       help='Episodes per iteration (overrides config)')
    args = parser.parse_args()
    
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
    params.NO_OVERFLOW_DISCONNECTION = False
    
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        param=params
    )
    
    # Set Grid2Op seed for deterministic chronics
    env.seed(seed)
    
    # Build action catalog from config
    from config import ACTIONS_CONFIG, USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS
    substations = ACTIONS_CONFIG['substations']
    reduction = ACTIONS_CONFIG['reduction']
    drop_identity = ACTIONS_CONFIG['drop_identity']
    include_do_nothing = ACTIONS_CONFIG.get('include_do_nothing', True)
    
    print(f"Building action catalog: substations={substations}, reduction={reduction}, drop_identity={drop_identity}, include_do_nothing={include_do_nothing}")
    full_catalog = build_action_catalog(env, substations=substations, reduction=reduction, 
                                   drop_identity=drop_identity, include_do_nothing=include_do_nothing)
    
    # Apply reduced action space if enabled
    if USE_REDUCED_ACTION_SPACE:
        from dataclasses import replace
        reduced_actions = [full_catalog.actions[idx] for idx in REDUCED_ACTIONS if idx < len(full_catalog.actions)]
        catalog = replace(full_catalog, actions=reduced_actions)
        print(f"🎯 Using REDUCED action space: {len(catalog.actions)} actions (indices: {REDUCED_ACTIONS})")
    else:
        catalog = full_catalog
        print(f"Action catalog: {len(catalog.actions)} actions")
    
    if include_do_nothing:
        print(f"  Action 0: do-nothing (explicit)")
        print(f"  Actions 1-{len(catalog.actions)-1}: topology changes")
    
    # Load training configuration from config.py
    from config import AGENT_CONFIG, TRAINING_CONFIG
    
    # Override episodes_per_iteration from command-line if provided
    if args.episodes_per_iteration is not None:
        AGENT_CONFIG['episodes_per_iteration'] = args.episodes_per_iteration
    
    # Override value_function_method from command-line
    AGENT_CONFIG['value_function_method'] = args.method
    
    # Override config with environment variables (for Optuna trials or comparison)
    def get_config_value(key, default):
        """Get config value from environment variable or default"""
        # Check both OPTUNA_ and direct env var names
        env_key_optuna = f'OPTUNA_{key.upper()}'
        env_key_direct = key.upper()
        
        value = None
        if env_key_optuna in os.environ:
            value = os.environ[env_key_optuna]
        elif env_key_direct in os.environ:
            value = os.environ[env_key_direct]
        
        if value is not None:
            # Parse value type
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            try:
                # Try int first, then float
                if '.' not in value:
                    return int(value)
                return float(value)
            except ValueError:
                return value  # Return as string
        return default
    
    config = {
        # MCTS parameters from config.py (can be overridden by env vars)
        'mcts_simulations': get_config_value('mcts_simulations', AGENT_CONFIG['mcts_simulations']),
        'c_puct': get_config_value('puct_c', AGENT_CONFIG['puct_c']),
        'gamma': AGENT_CONFIG['gamma'],
        'max_depth': AGENT_CONFIG['max_depth'],
        'temperature': get_config_value('temperature', AGENT_CONFIG['temperature']),
        'critical_threshold': get_config_value('critical_threshold', AGENT_CONFIG['critical_threshold']),
        'mcts_epsilon': AGENT_CONFIG['mcts_epsilon'],
        'action_prefilter_rho_increase': AGENT_CONFIG.get('action_prefilter_rho_increase', 0.15),
        't_skipped': AGENT_CONFIG['t_skipped'],
        't_stopping': AGENT_CONFIG['t_stopping'],
        
        # Policy target parameters
        'policy_target_method': get_config_value('policy_target_method', AGENT_CONFIG.get('policy_target_method', 'visits')),
        'policy_temperature': AGENT_CONFIG.get('policy_temperature', 1.0),
        'selection_bias_weight': get_config_value('selection_bias_weight', AGENT_CONFIG.get('selection_bias_weight', 0.0)),
        
        # PUCT modifications
        'use_depth_bonus': get_config_value('use_depth_bonus', AGENT_CONFIG.get('use_depth_bonus', False)),
        'depth_bonus': get_config_value('depth_bonus', AGENT_CONFIG.get('depth_bonus', 0.0)),
        'use_virtual_loss': get_config_value('use_virtual_loss', AGENT_CONFIG.get('use_virtual_loss', False)),
        'virtual_loss_weight': get_config_value('virtual_loss_weight', AGENT_CONFIG.get('virtual_loss_weight', 0.0)),
        'penalty_for_failure': get_config_value('penalty_for_failure', AGENT_CONFIG.get('penalty_for_failure', -5.0)),
        
        # Dirichlet noise
        'dirichlet_epsilon': get_config_value('dirichlet_epsilon', AGENT_CONFIG.get('dirichlet_epsilon', 0.25)),
        'dirichlet_alpha': get_config_value('dirichlet_alpha', AGENT_CONFIG.get('dirichlet_alpha', 0.3)),
        
        # Temperature decay
        'temperature_decay': get_config_value('temperature_decay', AGENT_CONFIG.get('temperature_decay', 0.95)),

        # Value target method (for comparison trials)
        'value_target_method': get_config_value('value_target_method', AGENT_CONFIG.get('value_target_method', 'mcts_root')),
        'heuristic_value_horizon': get_config_value('heuristic_value_horizon', AGENT_CONFIG.get('heuristic_value_horizon', 100)),

        # Training parameters
        'episodes_per_iteration': get_config_value('episodes_per_iteration', AGENT_CONFIG['episodes_per_iteration']),
        'parallel_workers': get_config_value('parallel_workers', AGENT_CONFIG.get('parallel_workers', 0)),
        'max_episode_steps': TRAINING_CONFIG.get('max_steps_per_episode', 10000) or 10000,
        'replay_buffer_size': get_config_value('replay_buffer_size', AGENT_CONFIG.get('replay_buffer_size', 100)),
        'replay_buffer_min_size': get_config_value('replay_buffer_min_size', AGENT_CONFIG.get('replay_buffer_min_size', 903)),
        'train_every': get_config_value('train_every', AGENT_CONFIG.get('train_every', 300)),
        'use_replay_buffer': AGENT_CONFIG.get('use_replay_buffer', True),
        'batch_size': AGENT_CONFIG['batch_size'],
        'learning_rate': get_config_value('learning_rate', AGENT_CONFIG['learning_rate']),
        'learning_rate_decay': get_config_value('learning_rate_decay', AGENT_CONFIG.get('learning_rate_decay', 1.0)),
        'min_learning_rate': AGENT_CONFIG['min_learning_rate'],
        'weight_decay': AGENT_CONFIG['weight_decay'],
        'training_epochs': get_config_value('training_epochs', AGENT_CONFIG['training_epochs']),
        
        # Loss weights
        'policy_weight': get_config_value('policy_weight', AGENT_CONFIG.get('policy_weight', 3.0)),
        'value_weight': get_config_value('value_weight', AGENT_CONFIG.get('value_weight', 1.0)),
        
        # Checkpointing
        'save_every': 1,  # Save checkpoint every iteration
        'save_training_data': AGENT_CONFIG.get('save_training_data', False),
        'training_data_dir': AGENT_CONFIG.get('training_data_dir', 'training_data_samples'),
        
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
    num_cycles = get_config_value('num_cycles', AGENT_CONFIG.get('num_cycles', 50))
    episodes_per_iteration = AGENT_CONFIG.get('episodes_per_iteration', 2)
    
    # Calculate total iterations: (num_cycles * train_chronics) / episodes_per_iteration
    num_train_chronics = len(trainer.train_chronics)
    total_iterations = (num_cycles * num_train_chronics) // episodes_per_iteration
    
    # Override with command-line argument if provided
    if args.num_iterations is not None:
        total_iterations = args.num_iterations
    
    # Apply max_training_iterations limit if set via environment variable
    max_training_iterations = get_config_value('max_training_iterations', None)
    if max_training_iterations is not None:
        total_iterations = min(total_iterations, max_training_iterations)
    
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
