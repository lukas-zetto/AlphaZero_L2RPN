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
    'learning_rate': 0.0001,  # REDUCED: Lower initial learning rate for stability (was 0.0003)
    'learning_rate_decay': 0.98,  # NEW: Decay LR by 2% per iteration
    'min_learning_rate': 0.00001,  # NEW: Minimum learning rate floor
    'exploration_rate': 0.3,  # Balanced exploration rate
    'exploration_decay': 0.995,  # Gradual decay over training
    'min_exploration_rate': 0.05,  # Small minimum to maintain some exploration
    
    # Safety parameters  
    'intervention_threshold': 0.98,  # Agent acts when max_rho > 98% (evaluation/normal operation)
    'critical_threshold': 0.95,      # MCTS training when max_rho > 95% (training - balanced difficulty)
    # Simple logic: Train on critical cases (95%+), act only in near-emergency (98%+)
    
    # Memory parameters
    'memory_size': 50000,  # Larger memory for more diverse experiences
    'batch_size': 64,  # Standard batch size for stable training
    
    # Training parameters
    'target_update_frequency': 1000,  # Less frequent updates for stability
    'gamma': 0.99,  # Standard discount factor for long-term planning
    'max_training_chronics': None,  #

    # MCTS parameters
    'mcts_simulations': 1000,  # Number of MCTS simulations per critical state
    'max_depth': 20,  # Limit tree depth to 10 levels
    'puct_c': 1.4,  # REDUCED: Less exploration, more exploitation of known good actions
    'mcts_epsilon': 0.2,  # Epsilon-greedy for MCTS node selection: 30% random, 70% PUCT for tree width
    'temperature': 0,  # Deterministic action selection (argmax)
    't_skipped': 80,  # Number of skipped safe states to be considered recovery node
    't_stopping': 30,  # Stop MCTS early if this many recovery nodes found (good actions exist)
    'action_prefilter_rho_increase': 0.10,  # Pre-filter actions that increase max_rho by more than this (0.10 = 10%)
    
    # Dirichlet noise for exploration (AlphaZero technique)
    'dirichlet_alpha': 0.3,    # Standard value: generates somewhat uniform noise
    'dirichlet_epsilon': 0.05,  # REDUCED: 95% network policy + 5% noise for more deterministic behavior

    # Neural network parameters
    'hidden_size': 256,  # Reduced from 1024 - smaller network for limited data
    'neural_network_implementation': 'v1',  # Use stable implementation
    
    # Model training parameters
    'num_cycles': 1,  # Number of times to cycle through all training chronics
    'episodes_per_iteration': 8,  # INCREASED: More episodes for stable gradient estimates (was 2)
    'training_epochs': 5,  # Multiple epochs for proper convergence
    'weight_decay': 0.0001,  # Standard L2 regularization
    'policy_weight': 1.0,  # Balanced loss weighting
    'value_weight': 1.0,   # Equal importance for policy and value
    'input_size': 83,  # Extended encoding: 60 line features + 23 bus topology bits
    # Experience replay toggle
    'use_replay_buffer': True,  # ENABLED: Use experience replay for stability
    'replay_buffer_size': 100,  # NEW: Keep last 100 episodes (~800 experiences with 8 eps/iter)
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
    'model_path': 'checkpoints/alphazero_v2_iter1.pt',  # Path to save/load trained model - use the latest trained model
}

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
    'reward_class': 'D3QN-2022',  # D3QN reward: line load penalties + survival bonus
    
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