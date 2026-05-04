"""
Configuration file for MyCustomAgent - Unified Config Structure
"""

import grid2op

# MASTER CONFIGURATION - Single source of truth for all parameters
MASTER_CONFIG = {
    # =============================
    # Core Agent Parameters
    # =============================
    'core_agent': {
        # Agent selection
        'agent_type': 'custom',  # 'custom' for MyCustomAgent, 'do_nothing' for Grid2Op DoNothingAgent
        'compare_with_baseline': True,  # Set to True to compare custom agent vs do nothing agent
        
        # Learning parameters
        'learning_rate': 0.000005,  # Conservative, stable learning
        'learning_rate_decay': 0.98,  # Decay LR by 10% per iteration  
        'min_learning_rate': 0.0000005,  # Minimum learning rate floor
        'exploration_rate': 0.0,  # Lower exploration - policy already learned good actions
        'exploration_decay': 0.999,  # Gradual decay over training
        'min_exploration_rate': 0.00,  # Small minimum to maintain some exploration
        
        # Safety parameters  
        'intervention_threshold': 0.98,  # Agent acts when max_rho > 98% (evaluation - match training threshold)
        'critical_threshold': 0.98,      # MCTS training when max_rho > 98% (training - less frequent MCTS)
        
        # Action logic selection
        'use_simplified_action_logic': False,  # Use simplified prioritized action logic (reconnect -> reset -> AlphaZero)
        
        # Memory parameters
        'memory_size': 50000,  # Larger memory for more diverse experiences
        'batch_size': 64,  # Standard batch size for stable training
        
        # Observation space configuration
        'obs_space_type': 'minimal',  # 'custom' for full 117 features, 'gym' for traditional, 'minimal' for ONLY rho (20 features), 'essential' for power+topo only
        'gym_obs_attr_to_keep': ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                                "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                                "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                                "storage_power", "storage_charge"],  # Attributes for gym observation space
        'essential_obs_attr_to_keep': ["prod_p", "load_p", "rho", "timestep_overflow", "topo_vect"],  # Essential features: power + topology
        'gym_obs_normalize': True,  # Normalize gym observations
        
        # Training parameters
        'target_update_frequency': 2000,  # Less frequent updates for stability
        'gamma': 0.95,  # Standard discount factor for long-term planning
        
        # Value loss stability parameters (PPO-style)
        'value_clip_range': 20.0,  # Clip value predictions to ±10 from target
        'huber_delta': None,  # Huber loss delta for value head - reduces sensitivity to outliers
        'max_training_chronics': None,
        
        # Value assignment method
        'value_target_method': 'heuristic',  # Options:
            # 'mcts_root': Use MCTS Q-values from root node only
            # 'binary_root': Use binary episode outcomes for root node only (+1/-1)
            # 'binary_all_nodes': Use binary outcomes for ALL tree nodes (PROPER ALPHAZERO like Go/Chess)
            # 'heuristic': Use heuristic value function (discounted future rewards from current state)
        'heuristic_value_horizon': 30,  # Set from trial_params.json

        # MCTS parameters
        'mcts_simulations': 250,  # Increased for better policy quality
        'max_depth': 25,  # Limit tree depth to 25 levels
        
        # Training data logging
        'save_training_data': False,  # Save training batches to disk for analysis
        'training_data_dir': 'training_data_fresh',  # Directory to save data
        'puct_c': 1.624074561769746,  # Set from trial_params.json
        'mcts_epsilon': 0.0,  # No random exploration during MCTS
        'temperature': 1.0,  # No sharpening - use raw visit counts (1.0 = no temperature effect)
        'temperature_decay': 1.0,  # Decay temperature by 5% each iteration
        'min_temperature': 1.0,  # Minimum temperature (keep at 1.0 for no sharpening)
        't_skipped': 200,  # Number of skipped safe states to be considered recovery node
        't_stopping': 50,  # Stop MCTS early if this many recovery nodes found (good actions exist)
        'action_prefilter_rho_increase': 0.10,  # RELAXED: Allow more actions (was 0.10)
        
        # PUCT modifications for exploration
        'use_depth_bonus': False,  # Add bonus for unexpanded/leaf nodes to encourage deeper exploration
        'depth_bonus': 0.05,  # Magnitude of depth bonus (only used if use_depth_bonus=True)
        'use_virtual_loss': False,  # Penalize frequently visited nodes to encourage wider trees
        'virtual_loss_weight': 0.5,  # Magnitude of virtual loss penalty (only used if use_virtual_loss=True)
        'penalty_for_failure': 0.0,  # Penalty reward for failed episodes (negative value)
        
        # Policy target method
        'policy_target_method': 'visits',  # 'one_hot' = one-hot on selected action, 'visits' = visit count distribution
        'policy_temperature': 1.0,  # Temperature for policy distribution (1.0 = no sharpening)
        'selection_bias_weight': 0.0,  # Set to 0 to prevent action collapse
        
        # Dirichlet noise for exploration (AlphaZero technique)
        'dirichlet_alpha': 0.15227525095137953,    # Set from trial_params.json
        'dirichlet_epsilon': 0.05,  # Optimized value from best_hyperparameters_cycle.json

        # Neural network parameters
        'hidden_size': 256,  # Larger network for better capacity
        'neural_network_implementation': 'alphazero',  # Use standard AlphaZero NN
        
        # Debug parameters
        'debug_fast_training': False,  # Set to False for normal training runs

        # Model training parameters
        'num_cycles': 30,
        'episodes_per_iteration': 301,
        'episodes_per_iteration_after_buffer_full': 301,  # Train after full circles
        'parallel_workers': 99,  # 15 workers for parallel training
        'training_epochs': 5,  # Increased for better convergence per iteration
        'weight_decay': 0.0001,  # Standard L2 regularization
        'policy_weight': 1.0,  # Balanced loss weighting
        'value_weight': 1.0,   # Equal importance for policy and value
        
        # Experience replay
        'use_replay_buffer': True,  # Enable replay buffer for stable training
        'replay_buffer_min_size': 301,  # Smaller buffer for faster training start
        'replay_buffer_size': 3612,  # Smaller buffer size
        'train_every': 301,  # Train more frequently
        
        # Action space
        'max_actions': 82,  # Number of possible actions (all 14 substations + do-nothing)
        
        # Auto-reconnect parameters
        'auto_reconnect': True,  # Automatically try to reconnect disconnected lines after topology actions
        'max_reconnections_per_action': 1,  # Number of lines to reconnect per MCTS action (1 = safest)
        
        # Topology reset parameters 
        'topology_reset_method': 'reco_agent',  # 'reco_agent' or 'manual' - how to reset topology to reference
        'topology_reset_threshold': 0.75,  # Reset to reference topology when max_rho <= this value

        
        # Model saving
        'model_path': None,  # Path to load trained model for evaluation (None = start from scratch)
        'checkpoint_dir': 'checkpoints_heuristics_minimal',  # Directory to save model checkpoints
    },
    
    # =============================
    # Actions Configuration
    # =============================
    'actions': {
        'type': 'catalog_bus_switch',  # Set to 'catalog_bus_switch' to activate new catalog
        'substations': list(range(14)),  # All 14 substations (0-13)
        'reduction': 'N1',             # One of: 'SYM', 'N0', 'N1'
        'drop_identity': False,        # Keep all actions for consistent NN indices (filter in MCTS instead)
        'include_do_nothing': True,    # Include explicit do-nothing action at index 0
        'masking': True,               # Enable runtime masking of illegal / cooldown actions
        'include_line_status_toggles': False,  # Hook for later extension
        'include_multi_sub_composites': False, # Hook for later extension
        'use_reduced_action_space': False,  # Set to True to use only a subset of actions
        'reduced_actions': [0, 9, 17, 22, 26, 29, 54, 59],  # Indices from full catalog to keep
        
        # =============================
        # Reconnection Module (RecoPowerlineModule)
        # =============================
        'reconnection_module': {
            'enabled': True,  # ENABLED by default - intelligent line reconnections
            'cooldown_respect': True,  # Respect line cooldowns (don't reconnect if on cooldown)
            'fallback_to_manual': True,  # Fallback to manual reconnection if RecoPowerlineModule fails
        },
        
        # =============================
        # Topology Reset Module
        # =============================
        'topology_reset_module': {
            'enabled': True,  # ENABLED by default - reset topology when very safe
            'safety_threshold': 0.75,  # Only reset when max_rho <= this value
        },
        
        # =============================
        # Emergency Disconnect Module
        # =============================
        'disconnect_module': {
            'enabled': False,  # DISABLED by default - sophisticated emergency disconnection
            'min_sustained_timesteps': 2,  # Minimum consecutive overflow timesteps before considering disconnection
            'max_disconnections_per_step': 2,  # Maximum lines to disconnect in single timestep
            'stress_reduction_required': True,  # Only disconnect if it reduces overall system stress (cur_max_rho > new_max_rho)
            'emergency_threshold': 1.0,  # Only consider lines with rho > this value for disconnection
            'cooldown_respect': True,  # Respect line cooldowns (don't disconnect if on cooldown)
        },
    },
    
    # =============================
    # Environment Configuration  
    # =============================
    'environment': {
        'name': 'l2rpn_case14_sandbox',  # Different environments = different action spaces
        'backend': 'LightSimBackend',  # or 'PandaPowerBackend'
        'reward_class': 'AlphaZero',  # Custom reward with line load penalties and topology change costs
        'action_class': None,  # Use default action class (or specify custom one)
        'observation_class': None,  # Use default observation class
    },
    
    # =============================
    # Training Configuration
    # =============================
    'training': {
        'episodes': 50,  # Increased for proper training
        'max_steps_per_episode': None,  # None = no limit, let episodes run to completion
        'validation_episodes': 50,
        'save_frequency': 10,  # Save model every N episodes (more frequent for fresh training)
        'log_frequency': 5,    # Log progress every N episodes (more frequent for monitoring)
        'early_stopping_patience': 200,
        'chronic_seed': 359,  # Seed for deterministic chronic selection - RUN _8
        'train_test_split': 0.9,  # 90% of chronics for training, 10% for test pool
        'num_test_chronics': None,  # Number of chronics to randomly select from test pool
    },
    
    # =============================
    # Evaluation Configuration
    # =============================
    'evaluation': {
        'episodes': None,  # None = use all test scenarios
        'max_steps': None,  # None = no limit, run episodes to completion
        'metrics': ['survival_time', 'reward', 'grid_losses', 'renewable_usage', 'operational_cost']
    },
    
    # =============================
    # Line Load Logging Configuration
    # =============================
    'line_logging': {
        'enabled': True,                    # Enable episode line load summaries
        'top_k': 5,                         # Number of highest-loaded lines to record per episode
        'episode_csv': 'logs/episode_line_summary.csv',  # Summary CSV output
        'ensure_dir': True,                 # Create directory if missing
        'print_summary': True,              # Print a concise summary line after each episode
    }
}


AGENT_CONFIG = MASTER_CONFIG['core_agent']
ACTIONS_CONFIG = MASTER_CONFIG['actions']
ENV_CONFIG = MASTER_CONFIG['environment']
TRAINING_CONFIG = MASTER_CONFIG['training']
EVAL_CONFIG = MASTER_CONFIG['evaluation']
LINE_LOGGING_CONFIG = MASTER_CONFIG['line_logging']

USE_REDUCED_ACTION_SPACE = MASTER_CONFIG['actions']['use_reduced_action_space']
REDUCED_ACTIONS = MASTER_CONFIG['actions']['reduced_actions']