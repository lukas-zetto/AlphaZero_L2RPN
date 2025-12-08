"""
Configuration file for MyCustomAgent
"""

import grid2op

# Agent hyperparameters
AGENT_CONFIG = {
    # Agent selection
    'agent_type': 'custom',  # 'custom' for MyCustomAgent, 'do_nothing' for Grid2Op DoNothingAgent
    'compare_with_baseline': True,  # Set to True to compare custom agent vs do nothing agent
    
    # Learning parameters
    'learning_rate': 0.0005,  # 
    'learning_rate_decay': 0.98,  # Decay LR by 10% per iteration  
    'min_learning_rate': 0.000005,  # Minimum learning rate floor
    'exploration_rate': 0.0,  # Lower exploration - policy already learned good actions
    'exploration_decay': 0.999,  # Gradual decay over training
    'min_exploration_rate': 0.00,  # Small minimum to maintain some exploration
    
    # Safety parameters  
    'intervention_threshold': 0.95,  # Agent acts when max_rho > 95% (evaluation - match training threshold)
    'critical_threshold': 0.98,      # MCTS training when max_rho > 98% (training - less frequent MCTS)
    # Simple logic: Act whenever rho > 95% to match training behavior
    
    # Memory parameters
    'memory_size': 50000,  # Larger memory for more diverse experiences
    'batch_size': 64,  # Standard batch size for stable training
    
    # Training parameters
    'target_update_frequency': 2000,  # Less frequent updates for stability
    'gamma': 0.95,  # Standard discount factor for long-term planning
    
    # Value loss stability parameters (PPO-style)
    'value_clip_range': 20.0,  # Clip value predictions to ±10 from target
    'huber_delta': None,  # Huber loss delta for value head - reduces sensitivity to outliers
    'max_training_chronics': None,  #
    
    # Value assignment method
    'value_target_method': 'heuristic',  # Options:
        # 'mcts_root': Use MCTS Q-values from root node only
        # 'binary_root': Use binary episode outcomes for root node only (+1/-1)
        # 'mcts_all_nodes': Use MCTS Q-values from ALL tree nodes (50-200x more training data)
        # 'binary_all_nodes': Use binary outcomes for ALL tree nodes (PROPER ALPHAZERO like Go/Chess)
        # 'heuristic': Use heuristic value function (discounted future rewards from current state)
    'heuristic_value_horizon': 10,  # Reduced from 20 to minimize do-nothing bias

    # MCTS parameters
    'mcts_simulations': 300,  # Number of MCTS simulations per critical state
    'max_depth': 25,  # Limit tree depth to 25 levels
    'puct_c': 3.0,  # Increased to 3.0 for more exploration to overcome heuristic do-nothing bias
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
    'policy_target_method': 'visits',  # 'one_hot' = one-hot on selected action, 'visits' = visit count distribution, 'max_steps' = distribution based on max_reachable_steps, 'visits_with_selection_bias' = visits + boost selected action
    'policy_temperature': 1.0,  # Temperature for policy distribution (1.0 = no sharpening, just raw visit counts)
    'selection_bias_weight': 0.0,  # Set to 0 to prevent action collapse
    
    # Dirichlet noise for exploration (AlphaZero technique)
    'dirichlet_alpha': 0.3,    # Chess/Shogi value (0.03 for Go, 0.3 for Chess/Shogi)
    'dirichlet_epsilon': 0.25,  # Standard AlphaZero: 25% noise, 75% policy

    # Neural network parameters
    'hidden_size': 256,  # Larger network for better capacity (enliteAI used 512 for larger grids)
    'neural_network_implementation': 'v1',  # Use stable implementation
    
    # Model training parameters
    'num_cycles': 10,  # 10 cycles through all training chronics
    'episodes_per_iteration': 903,  # Full cycle = 903 chronics (train after each complete cycle)
    'parallel_workers': 14,  # 14 parallel workers for faster training
    'training_epochs': 10,  # More epochs since we train less frequently (once per cycle)
    'weight_decay': 0.0001,  # Standard L2 regularization
    'policy_weight': 1.0,  # Balanced loss weighting
    'value_weight': 1.0,   # Equal importance for policy and value
    'input_size': 83,  # Extended encoding: 60 line features + 23 bus topology bits
    # Experience replay toggle
    'use_replay_buffer': True,  # Maintain experience diversity across iterations
    'replay_buffer_size': 1806,  # 2 full cycles worth (2*903) for stable diverse training
        # Input encoding breakdown:
        # - Line loads (rho): 20 bits [0:20]
        # - Line status: 20 bits [20:40] 
        # - Line cooldowns: 20 bits [40:60]
        # - Bus topology (subs 3,4,5,8): 23 bits [60:83]
        #   * Sub 3: 6 bits (3 line_or + 2 line_ex + 1 load)
        #   * Sub 4: 5 bits (1 line_or + 3 line_ex + 1 load)
        #   * Sub 5: 7 bits (3 line_or + 1 line_ex + 2 gen + 1 load)  
        #   * Sub 8: 5 bits (3 line_or + 1 line_ex + 1 load)
        # Action space
    'max_actions': 61,  # Number of possible actions (60 catalog actions + 1 do-nothing)
    
    # Recovery parameters
    'recovery_score_norm': 100.0,  # Normalization factor for recovery_score
    # critical_threshold is defined above in Safety parameters (0.95)
    
    # Line reconnection parameters
    'auto_reconnect': True,  # Automatically try to reconnect disconnected lines after topology actions
    'max_reconnections_per_action': 1,  # Number of lines to reconnect per MCTS action (1 = safest)
    
    # Topology reset parameters
    'topology_reset_threshold': 0.75,  # Reset to reference topology when max_rho ≤ this value (0.75 = 75% load)

    # Grid operation parameters
    'max_redispatch': 50.0,  # MW
    'storage_efficiency': 0.9,
    'curtailment_penalty': 0.1,
    
    # Model saving
    'model_path': 'checkpoints_heuristic/alphazero_v2_iter4.pt',  # Path to save/load trained model - use the latest trained model
}

# Reduced action space configuration
USE_REDUCED_ACTION_SPACE = False  # Set to True to use only a subset of actions
REDUCED_ACTIONS = [0, 9, 17, 22, 26, 29, 54, 59]  # Indices from full catalog to keep (0=do-nothing always included)

# New actions configuration (catalog-based bus switching)
# This enables dynamic enumeration of full substation bus layouts.
ACTIONS_CONFIG = {
    'type': 'catalog_bus_switch',  # Set to 'catalog_bus_switch' to activate new catalog; any other value keeps legacy behavior
    'substations': [3, 4, 5, 8],   # Default substation set (config-driven, extendable)
    'reduction': 'N1',             # One of: 'SYM', 'N0', 'N1'
    'drop_identity': False,        # Keep all actions for consistent NN indices (filter in MCTS instead)
    'include_do_nothing': True,    # Include explicit do-nothing action at index 0
    'masking': True,               # Enable runtime masking of illegal / cooldown actions
    # Future flags (placeholders):
    'include_line_status_toggles': False,  # Hook for later extension
    'include_multi_sub_composites': False, # Hook for later extension
}

# Line load logging configuration
LINE_LOGGING_CONFIG = {
    'enabled': True,                    # Enable episode line load summaries
    'top_k': 5,                         # Number of highest-loaded lines to record per episode
    'episode_csv': 'logs/episode_line_summary.csv',  # Summary CSV output
    'ensure_dir': True,                 # Create directory if missing
    'print_summary': True,              # Print a concise summary line after each episode
}

# Environment configuration - THIS DETERMINES THE ACTION SPACE
ENV_CONFIG = {
    'name': 'l2rpn_case14_sandbox',  # Different environments = different action spaces
    'backend': 'LightSimBackend',  # or 'PandaPowerBackend'
    'reward_class': 'AlphaZero',  # Custom reward with line load penalties and topology change costs
    
    # Action space customization options:
    'action_class': None,  # Use default action class (or specify custom one)
    'observation_class': None,  # Use default observation class
    
    # Environment parameters that affect available actions:
    # 'parameters': {...}  # Grid2Op parameters to restrict actions
    # 'gamerules_class': None,  # Custom game rules
    # 'volagecontroler_class': None,  # Custom voltage controller
    
    # Available environments with different action spaces:
    # - 'l2rpn_case14_sandbox': 14-bus system (small, good for development)
    # - 'l2rpn_neurips_2020_track1_small': 36-bus system  
    # - 'l2rpn_neurips_2020_track2_small': 36-bus with storage
    # - 'l2rpn_wcci_2022': 118-bus system (large)
}

# Training configuration
TRAINING_CONFIG = {
    'episodes': 1000,
    'max_steps_per_episode': None,  # None = no limit, let episodes run to completion
    'validation_episodes': 50,
    'save_frequency': 100,  # Save model every N episodes
    'log_frequency': 10,    # Log progress every N episodes
    'early_stopping_patience': 200,
    
    # Chronic selection configuration
    'chronic_seed': 123,  # Seed for deterministic chronic selection (set None for random)
    'train_test_split': 0.9,  # 90% of chronics for training, 10% for test pool
    'num_test_chronics': None,  # Number of chronics to randomly select from test pool (None = use all test chronics)
}


# Evaluation configuration
EVAL_CONFIG = {
    'episodes': None,  # None = use all test scenarios (101 chronics)
    'max_steps': 1000,  # Reduced from 2000 for faster testing
    'metrics': [
        'survival_time',
        'reward',
        'grid_losses', 
        'renewable_usage',
        'operational_cost'
    ]
}