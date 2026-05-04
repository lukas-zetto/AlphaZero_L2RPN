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
import copy
from collections import deque
import multiprocessing as mp
import warnings
import traceback

from actions.action_catalog import build_action_catalog
from training.alphazero_mcts_v2 import run_mcts, select_action, MCTSNodeV2, collect_all_tree_nodes, compute_heuristic_value
from networks.neural_network_factory import get_neural_network_functions
from networks.neural_network import encode_observation, encode_observation_simple, get_observation_size

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
        # Comprehensive warnings suppression for this worker process
        import warnings
        import os
        import sys
        
        # Apply all suppression settings FIRST
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*numba.*")
        warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        
        # Set environment variables to suppress numba warnings
        os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        
        # Monkey-patch warnings.warn to completely suppress output
        def silent_warn(*args, **kwargs):
            pass
        warnings.warn = silent_warn
        
        # Suppress stdout/stderr for imports that still show warnings 
        from io import StringIO
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = StringIO()
        sys.stdout = StringIO()
        
        # Re-import in worker process (with stderr suppressed)
        import grid2op
        from lightsim2grid import LightSimBackend
        from grid2op.Parameters import Parameters
        from actions.action_catalog import build_action_catalog
        from training.alphazero_mcts_v2 import run_mcts, select_action
        from networks.neural_network_factory import get_neural_network_functions
        from networks.neural_network import encode_observation, encode_observation_simple
        
        # Restore stderr and stdout after imports
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        
        # Create environment with reward class support (parallel worker)
        params = Parameters()
        params.NO_OVERFLOW_DISCONNECTION = False
        
        # Get reward class from config (passed from main process)
        reward_class_name = config.get('environment', {}).get('reward_class', None)
        reward_class = None
        
        if reward_class_name:
            match reward_class_name:
                case 'AlphaZero':
                    from src.rewards.alphazero_reward import AlphaZeroReward
                    reward_class = AlphaZeroReward
                case 'D3QN-2022':
                    from src.rewards.d3qn_reward import D3QNSurvivalReward
                    reward_class = D3QNSurvivalReward
                case 'D3QN-2020':
                    from src.rewards.d3qn_2020_reward import D3QN2020Reward
                    reward_class = D3QN2020Reward
                case 'Loss':
                    from src.rewards.loss_reward import LossReward
                    reward_class = LossReward
                case 'MaxRho':
                    from src.rewards.maxrho_reward import MaxRhoReward
                    reward_class = MaxRhoReward
                case 'PPO':
                    from src.rewards.ppo_reward import PPO_Reward
                    reward_class = PPO_Reward
                case 'LinesCapacity':
                    from src.rewards.linescapacity_reward import LinesCapacityReward
                    reward_class = LinesCapacityReward
                case _:
                    reward_class = None
        
        # Create environment with optional reward class
        env_kwargs = {
            "backend": LightSimBackend(),
            "param": params
        }
        if reward_class is not None:
            env_kwargs["reward_class"] = reward_class
            
        env = grid2op.make("l2rpn_case14_sandbox", **env_kwargs)
        
        # Build action catalog
        from config import ACTIONS_CONFIG, USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS
        
        full_catalog = build_action_catalog(
            env,
            substations=ACTIONS_CONFIG['substations'],
            reduction=config.get('reduction', ACTIONS_CONFIG['reduction']),  # Use config from main process (includes env vars)
            drop_identity=ACTIONS_CONFIG['drop_identity'],
            include_do_nothing=ACTIONS_CONFIG.get('include_do_nothing', True)
        )
        
        # Apply reduced action space if enabled
        use_reduced_action_space = config.get('use_reduced_action_space', USE_REDUCED_ACTION_SPACE)
        if use_reduced_action_space:
            from dataclasses import replace
            reduced_actions = [full_catalog.actions[idx] for idx in REDUCED_ACTIONS if idx < len(full_catalog.actions)]
            catalog = replace(full_catalog, actions=reduced_actions)
        else:
            catalog = full_catalog
            
        # Create neural network and load shared weights
        # Dynamically determine input size from environment
        dummy_obs = env.reset()
        from networks.neural_network import encode_observation, get_observation_size
        # Use new unified size calculation
        input_size = get_observation_size(env, config)
        num_actions = len(catalog.actions)
        # Workers run silently - no verbose output
        
        # Get neural network functions based on config
        nn_funcs = get_neural_network_functions(config)
        
        # Import torch for state dict loading
        import torch
        
        # Create network structure (suppress print messages in workers)
        import os
        # Temporarily suppress stdout for network creation
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        neural_network = nn_funcs.create_neural_network(input_size=input_size, num_actions=num_actions, config=config)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # ENABLE: Load shared network weights from main training process
        if 'network_state_dict' in config:
            neural_network.load_state_dict(config['network_state_dict'])
            neural_network.eval()  # Set to evaluation mode for inference
            # Workers run silently
        else:
            # Workers run silently - use random initialization
            pass
        
        # Create agent instance with full capabilities (same as main training process)
        from src.agent.my_agent import MyCustomAgent
        # Temporarily suppress agent output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        agent = MyCustomAgent(action_space=env.action_space, config=config)
        agent.set_env(env)  # Set environment reference
        agent.set_mode('train')  # Workers use training mode (MCTS + reconnections)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Pass required objects to agent for MCTS
        agent.neural_network = neural_network  # Use shared weights
        agent.mcts_config = config  # Full config for MCTS
        agent.action_catalog_for_mcts = catalog  # Action catalog for MCTS
        
        # Get neural network functions for agent
        nn_funcs = get_neural_network_functions(config)
        agent.nn_funcs = nn_funcs
        
        # Workers run silently - no verbose output
        
        # Set chronic and reset
        print(f"  [Worker {worker_id}] Started: Chronic {chronic_id}", flush=True)
        try:
            env.set_id(chronic_id)
            obs = env.reset()
        except Exception as e:
            print(f"  [Worker {worker_id}] Environment setup failed: {e}", flush=True)
            raise
        
        # Workers run silently - no startup messages
        
        done = False
        step = 0
        reward = 0.0  # Initialize reward for first agent.act() call
        episode_data = []
        max_steps = config['max_episode_steps']
        critical_threshold = config.get('critical_threshold', 0.90)
        
        # Track episode statistics
        max_rho_seen = 0.0
        cumulative_reward = 0.0
        
        while not done and step < max_steps:
            # Check for debug mode without printing
            is_debug_mode = config.get('debug_fast_training', False)
            num_samples = len(episode_data)

            if is_debug_mode and num_samples >= 2:
                print(f"  ⏩ DEBUG: Reached {num_samples} training samples, forcing episode end.")
                break

            # Use agent for all decision making (handles MCTS in critical states, reconnections in all states)
            try:
                action = agent.act(obs, reward, done)
            except Exception as e:
                print(f"  [Worker {worker_id}] Step {step}: Agent action failed: {e}", flush=True)
                raise
            
            # Track basic episode statistics
            max_rho = np.max(obs.rho)
            if max_rho > max_rho_seen:
                max_rho_seen = max_rho
            
            # Collect training data from agent if available
            training_data = getattr(agent, 'last_training_data', None)
            if training_data:
                episode_data.extend(training_data)
            
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            step += 1
        
        env.close()
        
        # Calculate average tree depth from MCTS runs (proper method)
        tree_depths = getattr(agent, 'episode_tree_depths', [])
        avg_depth = np.mean(tree_depths) if tree_depths else 0.0
        
        # Print compact completion summary with requested statistics
        print(f"  [Worker {worker_id}] Completed: Chronic {chronic_id} - {step} steps, {len(episode_data)} samples, avg_depth={avg_depth:.1f}, reward={cumulative_reward:.1f}, max_rho={max_rho_seen:.3f}", flush=True)
        
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
        print(f"  [Worker {worker_id}] Episode FAILED with exception: {e}", flush=True)
        import traceback
        error_msg = f"Worker {worker_id} failed: {str(e)}\n{traceback.format_exc()}"
        print(f"  [Worker {worker_id}] Full error: {error_msg}", flush=True)
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
        
        # Create agent for safe-state action selection
        from src.agent.my_agent import MyCustomAgent
        self.agent = MyCustomAgent(action_space=env.action_space, config=config)
        self.agent.set_env(env)  # Set environment reference for action modules
        
        # Set agent to training mode and pass required objects for MCTS
        self.agent.set_mode('train')
        self.agent.mcts_config = config
        self.agent.action_catalog_for_mcts = action_catalog
        
        # Get neural network functions based on config
        self.nn_funcs = get_neural_network_functions(config)
        self.agent.nn_funcs = self.nn_funcs  # Pass to agent for MCTS
        
        # Create neural network with dynamic input size
        # Determine input size from actual environment observation using unified method
        self.input_size = get_observation_size(env, config)  # Dynamic based on config and environment
        self.num_actions = len(action_catalog.actions)
        print(f"🔍 TRAINER: Dynamic input_size={self.input_size}, num_actions={self.num_actions}, obs_type={config.get('obs_space_type', 'custom')}")
        print(f"🔍 Using neural network implementation: {config.get('neural_network_implementation', 'default')}")
        
        self.neural_network = self.nn_funcs.create_neural_network(
            input_size=self.input_size,
            num_actions=self.num_actions,
            config=config
        )
        
        # Share neural network with agent for MCTS
        self.agent.neural_network = self.neural_network
        
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
        # Create optimizer once (for learning rate decay) - only for non-RBM implementations
        self.learning_rate = config['learning_rate']
        if config.get('neural_network_implementation') != 'rbm':
            self.optimizer = torch.optim.Adam(
                self.neural_network.parameters(), 
                lr=self.learning_rate,
                weight_decay=config.get('weight_decay', 0.0001)
            )
        else:
            self.optimizer = None  # RBM doesn't use PyTorch optimizers
        # Learning rate decay params
        self.lr_decay = config.get('learning_rate_decay', 1.0)
        self.min_lr = config.get('min_learning_rate', 0.00001)
        
        # Temperature for action selection
        self.current_temperature = config.get('temperature', 1.0)
        
        # Training data buffer - stores EPISODES (each episode is a list of experiences)
        self.replay_buffer = deque(maxlen=config.get('replay_buffer_size', 100))
        
        # Performance tracking for plotting
        self.episode_rewards = []  # Cumulative reward per episode
        self.episode_success = []  # 1 if success, 0 if failure
        self.episode_lengths = []  # Steps per episode
        
        # Split chronics into train/test with random test selection
        from config import TRAINING_CONFIG
        total_chronics = len(env.chronics_handler.real_data.subpaths)
        train_test_split = TRAINING_CONFIG.get('train_test_split', 0.9)
        num_test_chronics = TRAINING_CONFIG.get('num_test_chronics', None)
        chronic_seed = config.get('chronic_seed') or TRAINING_CONFIG.get('chronic_seed', None)
        
        # Create random split: shuffle all chronics first, then split
        import random
        all_chronics = list(range(total_chronics))
        
        # Use seed for deterministic random split if provided
        if chronic_seed is not None:
            random.seed(chronic_seed)
        
        # Randomly shuffle all chronics before splitting
        random.shuffle(all_chronics)
        
        train_pool_size = int(total_chronics * train_test_split)
        self.train_chronics = all_chronics[:train_pool_size]  # First X% after shuffle
        test_pool = all_chronics[train_pool_size:]  # Remaining chronics after shuffle
        
        # Select test chronics: either a random subset or all test chronics
        if num_test_chronics is None:
            # Use all chronics from test pool
            self.test_chronics = test_pool
        elif len(test_pool) >= num_test_chronics:
            # Randomly select subset from test pool (re-seed for consistency)
            if chronic_seed is not None:
                random.seed(chronic_seed + 1)  # Different seed for test selection
            self.test_chronics = sorted(random.sample(test_pool, num_test_chronics))
        else:
            # Test pool smaller than requested, use all
            self.test_chronics = test_pool
            print(f"⚠️  Warning: Only {len(test_pool)} chronics available for testing (requested {num_test_chronics})")
        
        # Shuffle training chronics to prevent temporal correlation
        if chronic_seed is not None:
            random.seed(chronic_seed + 2)  # Different seed for training order
        random.shuffle(self.train_chronics)
        
        self.current_chronic_idx = 0  # Index into train_chronics list
        print(f"Created neural network: input={self.input_size}, actions={self.num_actions}")
        print(f"Chronic selection (seed={chronic_seed}) - RANDOM SPLIT:")
        print(f"  Total chronics: {total_chronics}")
        print(f"  Training chronics: {len(self.train_chronics)} randomly selected - {sorted(self.train_chronics)}")
        print(f"  Test chronics selected: {len(self.test_chronics)} chronics - {sorted(self.test_chronics)}")
        print(f"Initial learning rate: {self.learning_rate:.6f}, decay: {self.lr_decay}, min: {self.min_lr:.6f}")
        print(f"Replay buffer: {'ENABLED' if self.use_replay_buffer else 'DISABLED'}, size: {config.get('replay_buffer_size', 0)} episodes")

    
    def get_policy_value(self, observation):
        """Get policy and value from neural network."""
        if observation is None:
            # Return uniform policy and zero value for None observations
            uniform_policy = np.ones(self.num_actions) / self.num_actions
            return uniform_policy, 0.0
        
        policy, value = self.nn_funcs.neural_network_forward(
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
        reward = 0.0  # Initialize reward for first step
        
        while not done and step < self.config['max_episode_steps']:
            # Forceful debug check at every step
            is_debug_mode = self.config.get('debug_fast_training', False)
            num_samples = len(episode_data)
            print(f"  [DEBUG] Step {step}: debug_fast_training={is_debug_mode}, collected_samples={num_samples}")

            if is_debug_mode and num_samples >= 2:
                print(f"  ⏩ DEBUG: Reached {num_samples} training samples, forcing episode end.")
                break
            max_rho = np.max(obs.rho)
            critical_threshold = self.config.get('core_agent', {}).get('critical_threshold', 0.98)
            is_critical = max_rho > critical_threshold
            
            print(f"\nStep {step}: rho_max={max_rho:.3f} {'🔴 CRITICAL' if is_critical else '🟢 SAFE'} (threshold={critical_threshold})")
            
            # Use agent for all decision making (handles MCTS in critical states, reconnections in all states)
            action = self.agent.act(obs, reward, done)
            
            # Collect training data from agent if available
            training_data = getattr(self.agent, 'last_training_data', None)
            if training_data:
                episode_data.extend(training_data)
                print(f"  📦 Collected {len(training_data)} training examples from agent")
            
            obs, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            step += 1
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
            'heuristic': 'Heuristic value function',
            'no_guidance': 'Pure MCTS (no neural network guidance)'
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
        
        # Track performance metrics
        self.episode_rewards.append(cumulative_reward)
        self.episode_success.append(0.0 if is_failure else 1.0)
        self.episode_lengths.append(step)
        
        # Return: (num_samples, episode_outcome, num_steps)
        # num_steps = actual environment interactions (actions taken in real env, not MCTS sims)
        return len(episode_data), 1.0 if not is_failure else -1.0, step
    
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
        # Check if RBM implementation (doesn't use optimizer parameter)
        if self.config.get('neural_network_implementation') == 'rbm':
            loss_info = self.nn_funcs.train_neural_network(
                self.neural_network,
                training_examples,
                self.config
            )
        else:
            loss_info = self.nn_funcs.train_neural_network(
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
        
        # Update agent's neural network reference after training
        self.agent.neural_network = self.neural_network
        
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
        
        # Update optimizer learning rate (only if optimizer exists)
        if self.optimizer is not None:
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
            (total_collected, total_steps): Number of training examples and environment steps
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
            worker_id = episode % num_workers  # Assign worker ID based on episode index (0-3 for 4 workers)
            worker_args.append((chronic_id, episode, worker_config, worker_id))
        
        # Run parallel collection
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(_parallel_episode_worker, worker_args)
            
            # Process results
            total_examples = 0
            total_steps = 0  # Track environment steps
            successful = 0
            failed = 0
            
            for result in results:
                if result['success']:
                    successful += 1
                    episode_data = result['episode_data']
                    total_steps += result['steps']  # Accumulate steps from each episode
                    
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
            
            print(f"\n  Summary: {successful}/{num_episodes} successful, {total_examples} total examples, {total_steps:,} total steps")
            return total_examples, total_steps
            
        except Exception as e:
            print(f"❌ Parallel collection failed: {e}")
            print(traceback.format_exc())
            # Fall back to sequential
            print("⚠️ Falling back to sequential collection...")
            total_steps_fallback = 0
            for episode in range(num_episodes):
                _, _, episode_steps = self.self_play_episode(episode)
                total_steps_fallback += episode_steps
            return 0, total_steps_fallback
    
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
        total_steps = 0  # Track total environment steps (actual actions taken)
        
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
                _, parallel_steps = self.self_play_parallel(
                    num_episodes=episodes_per_iteration,
                    num_workers=parallel_workers
                )
                total_steps += parallel_steps  # Accumulate steps from parallel episodes
            else:
                # SEQUENTIAL COLLECTION (original behavior)  
                for episode in range(episodes_per_iteration):
                    episode_len, episode_value, episode_steps = self.self_play_episode(episode, enable_debug=True)
                    total_steps += episode_steps  # Accumulate environment steps
                    # Collect all episode data if not using replay buffer
                    if not self.use_replay_buffer:
                        # Get the last episode from replay buffer (the episode we just played)
                        if len(self.replay_buffer) > 0:
                            current_iteration_data.extend(self.replay_buffer[-1])  # Last episode
            
            episodes_collected += episodes_per_iteration
            episodes_since_last_training += episodes_per_iteration
            
            # Log progress with step count
            print(f"\n📊 Progress: {episodes_collected} episodes, {total_steps:,} environment steps")
            
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
                    checkpoint_dir = os.environ['CHECKPOINT_DIR']  # Use absolute path directly
                else:
                    value_method = self.config.get('value_target_method', 'default')
                    checkpoint_dir = f"checkpoints_{value_method}" if value_method != 'default' else "checkpoints"
                checkpoint_path = f"{checkpoint_dir}/alphazero_v2_train{training_iteration}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                # Calculate recent performance (last 100 episodes or all available)
                recent_window = min(100, len(self.episode_rewards))
                if recent_window > 0:
                    recent_avg_reward = np.mean(self.episode_rewards[-recent_window:])
                    recent_success_rate = np.mean(self.episode_success[-recent_window:])
                    recent_avg_length = np.mean(self.episode_lengths[-recent_window:])
                else:
                    recent_avg_reward = 0.0
                    recent_success_rate = 0.0
                    recent_avg_length = 0.0
                
                # Save checkpoint using the updated save interface
                checkpoint_data = {
                    'training_iteration': training_iteration,
                    'episodes_collected': episodes_collected,
                    'total_steps': total_steps,  # Total environment interactions
                    'replay_buffer_size': len(self.replay_buffer),
                    'num_actions': self.num_actions,
                    'input_size': self.input_size,
                    'hidden_size': self.config.get('hidden_size', 256),
                    'config': self.config,
                    # Performance metrics for plotting
                    'performance': {
                        'avg_reward_recent': recent_avg_reward,
                        'success_rate_recent': recent_success_rate,
                        'avg_episode_length_recent': recent_avg_length,
                        'total_episodes': len(self.episode_rewards),
                        'all_episode_rewards': self.episode_rewards,
                        'all_episode_success': self.episode_success,
                        'all_episode_lengths': self.episode_lengths,
                    }
                }
                
                # Save checkpoint with standardized model format + training metadata
                torch.save({
                    # Standardized model data (same format as neural_network.save_model)
                    'model_type': 'alphazero',
                    'model_state_dict': self.neural_network.state_dict(),
                    'input_size': self.input_size,
                    'num_actions': self.num_actions,
                    'hidden_size': self.config.get('hidden_size', 256),
                    # Training-specific metadata
                    **checkpoint_data
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
                print(f"   Steps: {total_steps:,} | Avg reward (last {recent_window}): {recent_avg_reward:.2f} | Success rate: {recent_success_rate:.1%}")
                
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
        print(f"📊 Final Statistics:")
        print(f"  Total episodes: {episodes_collected}")
        print(f"  Total environment steps: {total_steps:,}")
        print(f"  Training updates: {training_iteration}")
        print(f"  Replay buffer size: {len(self.replay_buffer)} episodes")
        if episodes_collected > 0:
            print(f"  Average steps per episode: {total_steps / episodes_collected:.1f}")
        print()


def main():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train AlphaZero agent')
    parser.add_argument('--method', type=str, default=None,
                       choices=['heuristic', 'mcts_root', 'mcts_all_nodes', 'binary_root', 'binary_all_nodes', 'no_guidance'],
                       help='Value function method (overrides config)')
    parser.add_argument('--obs_space_type', type=str, default=None,
                       choices=['minimal', 'custom', 'gym', 'essential'],
                       help='Observation space type (overrides config)')
    parser.add_argument('--num_iterations', type=int, default=None,
                       help='Total number of training iterations (overrides calculation)')
    parser.add_argument('--episodes_per_iteration', type=int, default=None,
                       help='Episodes per iteration (overrides config)')
    parser.add_argument('--main-seed', type=int, default=None,
                       help='Main seed for training process (overrides hardcoded seed)')
    parser.add_argument('--chronic-seed', type=int, default=None,
                       help='Chronic seed for deterministic scenario selection (overrides config)')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    import random
    seed = args.main_seed if args.main_seed is not None else 1415  # Default to RUN _8, override via command line
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Using main seed: {seed}")
    if args.chronic_seed is not None:
        print(f"🎲 Chronic seed override: {args.chronic_seed}")
    
    # Helper function to get config values from environment variables
    def get_config_value(key, default):
        """Get config value from environment variable or default"""
        # Check both OPTUNA_ and direct env var names
        env_key_optuna = f'OPTUNA_{key.upper()}'
        env_key_direct = key.upper()
        
        value = None
        if env_key_optuna in os.environ:
            value = os.environ[env_key_optuna]
            print(f"🔧 DEBUG: Found {env_key_optuna} = {value}")
        elif env_key_direct in os.environ:
            value = os.environ[env_key_direct]
            print(f"🔧 DEBUG: Found {env_key_direct} = {value}")
        else:
            print(f"🔧 DEBUG: Environment variable {key.upper()} not found, using default: {default}")
        
        if value is not None:
            # Parse value type
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            try:
                # Try int first, then float
                if '.' not in value:
                    parsed = int(value)
                    print(f"🔧 DEBUG: Parsed {key} as int: {parsed}")
                    return parsed
                parsed = float(value)
                print(f"🔧 DEBUG: Parsed {key} as float: {parsed}")
                return parsed
            except ValueError:
                print(f"🔧 DEBUG: Parsed {key} as string: {value}")
                return value  # Return as string
        return default
    
    # Setup environment with reward class support
    env_name = "l2rpn_case14_sandbox"
    print(f"Loading environment: {env_name}")
    
    # Import custom reward class based on environment variable
    reward_class = None
    reward_class_name = get_config_value('reward_class', 'AlphaZero')
    
    if reward_class_name:
        match reward_class_name:
            case 'AlphaZero':
                from src.rewards.alphazero_reward import AlphaZeroReward
                reward_class = AlphaZeroReward
            case 'D3QN-2022':
                from src.rewards.d3qn_reward import D3QNSurvivalReward
                reward_class = D3QNSurvivalReward
            case 'D3QN-2020':
                from src.rewards.d3qn_2020_reward import D3QN2020Reward
                reward_class = D3QN2020Reward
            case 'Loss':
                from src.rewards.loss_reward import LossReward
                reward_class = LossReward
            case 'MaxRho':
                from src.rewards.maxrho_reward import MaxRhoReward
                reward_class = MaxRhoReward
            case 'PPO':
                from src.rewards.ppo_reward import PPO_Reward
                reward_class = PPO_Reward
            case 'LinesCapacity':
                from src.rewards.linescapacity_reward import LinesCapacityReward
                reward_class = LinesCapacityReward
            case _:
                reward_class = None
        print(f"🎯 Using reward class: {reward_class_name} -> {reward_class}")
        
        # Verify reward class has proper identification
        if reward_class is not None:
            temp_reward = reward_class()
            if hasattr(temp_reward, 'REWARD_ID'):
                print(f"🔍 REWARD VERIFICATION: ID={temp_reward.REWARD_ID}")
            else:
                print(f"⚠️  REWARD WARNING: {reward_class.__name__} missing REWARD_ID verification")
    
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = False
    
    # Create environment with optional reward class
    env_kwargs = {
        "backend": LightSimBackend(),
        "param": params
    }
    if reward_class is not None:
        env_kwargs["reward_class"] = reward_class
        
    env = grid2op.make(env_name, **env_kwargs)
    
    # Set Grid2Op seed for deterministic chronics
    env.seed(seed)
    
    # Load training configuration from config.py
    from config import AGENT_CONFIG, TRAINING_CONFIG, MASTER_CONFIG
    from config import ACTIONS_CONFIG, USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS
    
    # Override episodes_per_iteration from command-line if provided
    if args.episodes_per_iteration is not None:
        AGENT_CONFIG['episodes_per_iteration'] = args.episodes_per_iteration
    
    # Override value_target_method from command-line if provided
    if args.method is not None:
        AGENT_CONFIG['value_target_method'] = args.method
    
    # Override obs_space_type from command-line if provided
    if args.obs_space_type is not None:
        AGENT_CONFIG['obs_space_type'] = args.obs_space_type
    
    # Create config dictionary using environment variables
    actions_config = copy.deepcopy(MASTER_CONFIG['actions'])
    actions_config.setdefault('topology_reset_module', {})
    actions_config['topology_reset_module']['safety_threshold'] = get_config_value(
        'topology_reset_threshold',
        actions_config['topology_reset_module'].get('safety_threshold', 0.75)
    )

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
        'debug_fast_training': TRAINING_CONFIG.get('debug_fast_training', False),  # Add debug flag from TRAINING_CONFIG
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
        
        # Cycles and iterations
        'num_cycles': get_config_value('num_cycles', AGENT_CONFIG.get('num_cycles', 50)),
        
        # Topology reset parameters (ADD THIS!)
        'topology_reset_threshold': get_config_value('topology_reset_threshold', AGENT_CONFIG.get('topology_reset_threshold', 0.90)),
        
        # Loss weights
        'policy_weight': get_config_value('policy_weight', AGENT_CONFIG.get('policy_weight', 3.0)),
        'value_weight': get_config_value('value_weight', AGENT_CONFIG.get('value_weight', 1.0)),
        
        # Checkpointing
        'save_every': 1,  # Save checkpoint every iteration
        'save_training_data': AGENT_CONFIG.get('save_training_data', False),
        'training_data_dir': AGENT_CONFIG.get('training_data_dir', 'training_data_samples'),
        
        # Neural network architecture
        'hidden_size': AGENT_CONFIG['hidden_size'],  # Use singular to match network code
        'neural_network_implementation': AGENT_CONFIG.get('neural_network_implementation', 'v1'),  # Add neural network implementation key
        
        # RBM-specific parameters
        'rbm_num_hidden_units': AGENT_CONFIG.get('rbm_num_hidden_units', 64),
        'rbm_disc_steps': AGENT_CONFIG.get('rbm_disc_steps', 8),
        'rbm_epochs': AGENT_CONFIG.get('rbm_epochs', 5),
        'rbm_regularization': AGENT_CONFIG.get('rbm_regularization', 'weak'),
        'rbm_outer_epochs': AGENT_CONFIG.get('rbm_outer_epochs', 5),
        'max_actions': 100,  # Will be updated after catalog creation
        
        'dropout': 0.0,
        
        # CRITICAL: Missing parameters that broke all experiments
        'obs_space_type': get_config_value('obs_space_type', AGENT_CONFIG.get('obs_space_type', 'custom')),
        
        # Action space parameters (for action space experiments)  
        'reduction': get_config_value('reduction', ACTIONS_CONFIG.get('reduction', 'N0')),
        'use_reduced_action_space': get_config_value('use_reduced_action_space', USE_REDUCED_ACTION_SPACE),
        
        # Safety parameters (for safety experiments)
        't_skipped': get_config_value('t_skipped', AGENT_CONFIG.get('t_skipped', 4)),
        't_stopping': get_config_value('t_stopping', AGENT_CONFIG.get('t_stopping', 2)),
        
        # Add actions configuration for agent compatibility
        'actions': actions_config,
        'core_agent': MASTER_CONFIG['core_agent'],
        
        # CRITICAL: Environment configuration for MCTS reward functions
        'environment': {
            'reward_class': get_config_value('reward_class', 'AlphaZero')
        },
        
        # Command line argument overrides
        'chronic_seed': args.chronic_seed,  # Add chronic seed from command line args
    }
    
    print(f"\nTraining configuration:")
    print(f"  MCTS simulations: {config['mcts_simulations']}")
    print(f"  c_puct: {config['c_puct']}")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()
    
    # Build action catalog using processed config (includes environment variables)
    substations = ACTIONS_CONFIG['substations']
    reduction = config['reduction']  # Use processed config (includes env vars)
    drop_identity = ACTIONS_CONFIG['drop_identity']
    use_reduced_action_space = config['use_reduced_action_space']  # Use processed config
    include_do_nothing = ACTIONS_CONFIG.get('include_do_nothing', True)
    
    print(f"Building action catalog: substations={substations}, reduction={reduction}, drop_identity={drop_identity}, include_do_nothing={include_do_nothing}")
    print(f"🔧 Using action reduction: {reduction} (from environment variables)")
    full_catalog = build_action_catalog(env, substations=substations, reduction=reduction, 
                                   drop_identity=drop_identity, include_do_nothing=include_do_nothing)
    
    # Apply reduced action space if enabled
    if use_reduced_action_space:
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
    
    # Update config with actual catalog size for RBM networks
    config['max_actions'] = len(catalog.actions)
    
    # Create trainer
    trainer = AlphaZeroTrainerV2(env, catalog, config)
    
    # Run training
    num_cycles = config['num_cycles']  # Use config dict (already processed env vars)
    episodes_per_iteration = config['episodes_per_iteration']  # Use config dict that has env vars
    
    # Calculate total iterations: (num_cycles * train_chronics) / episodes_per_iteration
    num_train_chronics = len(trainer.train_chronics)
    total_iterations = (num_cycles * num_train_chronics) // episodes_per_iteration
    
    # Override for debug mode
    if config.get('debug_fast_training', False):
        print("🔧 DEBUG MODE: Overriding total_iterations to 1")
        total_iterations = 1
    
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
