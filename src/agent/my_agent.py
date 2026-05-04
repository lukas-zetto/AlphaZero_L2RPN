import os
import numpy as np
import warnings

# Suppress warnings to clean up output  
warnings.filterwarnings("ignore")

# Global debug flag - can be controlled via environment variable
AGENT_DEBUG = os.environ.get('AGENT_DEBUG', 'true').lower() == 'true'

from grid2op.Agent import BaseAgent
from grid2op.Action import ActionSpace
from grid2op.Observation import BaseObservation

# Catalog-based action space (REQUIRED)
from src.actions.action_catalog import build_action_catalog, compute_action_mask, summarize_catalog


class MyCustomAgent(BaseAgent):
    """
    Custom Grid2Op Agent for L2RPN Challenge
    
    This agent template provides the basic structure for implementing
    a custom reinforcement learning or optimization-based agent.
    """
    
    def __init__(self, action_space: ActionSpace, config=None):
        """
        Initialize the agent with both line switching and bus switching capabilities
        
        Parameters:
        -----------
        action_space : ActionSpace
            The action space of the environment
        config : dict, optional
            Configuration parameters for the agent
        """
        super().__init__(action_space)
        
        self.action_space = action_space
        self.config = config or {}

        # Action configuration (new unified config structure)
        self.actions_config = self.config.get('actions') or self.config.get('ACTIONS_CONFIG') or self.config.get('actions_config')

        # Catalog attributes
        self.action_catalog = None
        self.catalog_ready = False
        self._baseline_env = None  # temporary env used for catalog build
        self._initialize_actions()

        # Initialize neural network for training
        self.neural_network = None
        self._initialize_neural_network()

        # Initialize agent parameters - use correct defaults matching config.py
        self.intervention_threshold = self.config.get('intervention_threshold', 0.98)
        self.critical_threshold = self.config.get('critical_threshold', 0.98)  # Fixed: was 1.0

        # Store reference to current environment
        self.current_env = None
        
        # Track last action taken for evaluation statistics
        self.last_action_taken = None
        
        # Track overflow history for sustained overflow detection
        self.overflow_history = {}

        # Allow switching between action logic
        self.use_simplified_action_logic = self.config.get('use_simplified_action_logic', False)
        
        # Track whether topology is in reference state to avoid expensive checks
        self.is_reference_topology = True  # Start assuming reference topology
        
        # Mode system for train/test behavior
        self.mode = 'train'  # default to training mode
        
        # MCTS-related properties (set by trainer)
        self.mcts_config = None
        self.action_catalog_for_mcts = None  # Will be set by trainer for MCTS
        self.nn_funcs = None  # Will be set by trainer for MCTS
        
        # Training data storage (populated during training mode)
        self.last_training_data = None
    
    def set_env(self, env):
        """Update environment reference."""
        self.current_env = env
    
    def set_mode(self, mode):
        """Set agent mode: 'train' or 'test'"""
        if mode not in ['train', 'test']:
            raise ValueError("Mode must be 'train' or 'test'")
        self.mode = mode
        print(f"🔧 Agent mode set to: {mode}")
    
    def reset(self, obs):
        """Reset agent state for a new episode."""
        # Reset any episode-specific state here
        self.last_action_taken = None
        self.overflow_history = {}  # Clear overflow tracking
        self.is_reference_topology = True  # Reset assumes reference topology
        # BaseAgent.reset() handles the observation
        return super().reset(obs)
    
    def _initialize_actions(self):
        """Initialize the action catalog (REQUIRED)."""
        if not self.actions_config or self.actions_config.get('type') != 'catalog_bus_switch':
            raise ValueError("Action catalog configuration is required! Set ACTIONS_CONFIG['type'] = 'catalog_bus_switch'")

        try:
            import grid2op
            try:
                from lightsim2grid import LightSimBackend
            except ImportError:
                LightSimBackend = None
            if LightSimBackend is not None:
                backend = LightSimBackend()
                env = grid2op.make("l2rpn_case14_sandbox", backend=backend)
            else:
                env = grid2op.make("l2rpn_case14_sandbox")
            self._baseline_env = env

            subs = self.actions_config.get('substations', list(range(14)))
            # Use top-level config first (processed env vars), then fall back to actions config
            reduction = self.config.get('reduction') or self.actions_config.get('reduction', 'N1')
            drop_identity = self.actions_config.get('drop_identity', False)  # Fixed: was True
            
            # Build full catalog first
            full_catalog = build_action_catalog(env, subs, reduction, drop_identity)
            
            # Apply reduced action space if enabled
            try:
                # Use top-level config first (processed env vars), then fall back to actions config
                use_reduced = self.config.get('use_reduced_action_space') or self.actions_config.get('use_reduced_action_space', False)
                reduced_actions = self.actions_config.get('reduced_actions', [0, 9, 17, 22, 26, 29, 54, 59])
                if use_reduced:
                    from dataclasses import replace
                    selected_actions = [full_catalog.actions[idx] for idx in reduced_actions if idx < len(full_catalog.actions)]
                    self.action_catalog = replace(full_catalog, actions=selected_actions)
                    print(f"✅ Built REDUCED catalog-based action space with {len(selected_actions)} actions (indices: {reduced_actions})")
                else:
                    self.action_catalog = full_catalog
                    print("✅ Built catalog-based action space")
            except ImportError:
                self.action_catalog = full_catalog
                print("✅ Built catalog-based action space")
            if summarize_catalog:
                print(summarize_catalog(self.action_catalog))
            self.catalog_ready = True
        except Exception as e:
            print(f"❌ Catalog build failed: {e}")
            raise ValueError(f"Failed to initialize action catalog: {e}")
    
    def _initialize_neural_network(self):
        """Initialize the neural network for training"""
        try:
            from src.networks.neural_network_factory import get_neural_network_functions
            nn_funcs = get_neural_network_functions(self.config)
            
            # Calculate input size dynamically using the baseline environment
            if self._baseline_env is not None:
                input_size = nn_funcs.get_observation_size(self._baseline_env, self.config)
                if AGENT_DEBUG:
                    print(f"🔍 Dynamic input size calculation: {input_size}")
            else:
                # Fallback to config if available
                input_size = self.config.get('input_size', None)
                if input_size is None:
                    raise ValueError("Cannot determine input size: no baseline environment and no config['input_size']")

            if self.catalog_ready and self.action_catalog is not None:
                num_actions = self.action_catalog.size  # Catalog already includes do-nothing if configured
                print(f"🔧 Using catalog-based action space with {num_actions} actions")
            else:
                raise ValueError("Action catalog is required but not initialized!")

            self.neural_network = nn_funcs.create_neural_network(
                input_size=input_size,
                num_actions=num_actions,
                config=self.config
            )
            print(f"✅ Neural network initialized: {input_size} inputs -> {num_actions} actions")
        except Exception as e:
            print(f"⚠️ Neural network initialization failed: {e}")
            self.neural_network = None
    
    def act(self, observation: BaseObservation, reward: float = None, done: bool = False):
        """
        Choose a line switching action based on the current observation
        
        Parameters:
        -----------
        observation : BaseObservation
            Current state of the power grid
        reward : float, optional
            Reward received from previous action (unused)
        done : bool
            Whether the episode is finished
            
        Returns:
        --------
        action : Action
            The line switching action to take in the power grid
        """
        
        # Update overflow history for sustained overflow detection
        self._update_overflow_history(observation)
        
        # Initialize episode tree depths tracking if not exists  
        if not hasattr(self, 'episode_tree_depths'):
            self.episode_tree_depths = []
        
        # Reset last training data
        self.last_training_data = None
        
        # No dynamic recreation needed for catalog; for legacy keep previous safeguard.
        if not self.catalog_ready and self.bus_actions is not None:
            try:
                test_action = self.action_space()
            except Exception:
                print("   🔄 Recreating legacy bus actions for current environment...")
                self._initialize_bus_switching_legacy()
        
        # Get action based on grid state
        action, action_idx = self._get_action(observation)
        
        # Store action index for evaluation statistics
        self.last_action_taken = action_idx
        
        return action
        
    def _get_action(self, observation: BaseObservation):
        """
        Select an action based on the configured logic (simplified or combined).
        """
        if self.use_simplified_action_logic:
            if AGENT_DEBUG:
                print("🔍 Using SIMPLIFIED action logic")
            return self._get_action_simplified(observation)
        else:
            if AGENT_DEBUG:
                print("🔍 Using COMBINED action logic")
            return self._get_action_combined(observation)

    def _get_action_simplified(self, observation: BaseObservation):
        """
        Select an action using a simplified, prioritized approach.
        1. Analyze grid state to get information.
        2. Auto-reconnect lines if any are disconnected.
        3. Reset topology if grid is safe.
        4. Use AlphaZero for intervention.
        """
        # Analyze grid state to determine if action is needed
        grid_state = self._analyze_grid_state(observation)
        
        # Check if we're in reference topology state
        tested_action = self.action_space.get_back_to_ref_state(observation)
        if not tested_action or len(tested_action) == 0:
            # No reset actions available means we're already in reference state
            self.is_reference_topology = True
        else:
            self.is_reference_topology = False
        
        # PRIORITY 1: Auto-reconnect disconnected lines if any exist (always beneficial)
        if grid_state['n_disconnected'] > 0:
            reconnect_action = self._auto_reconnect_lines(observation)
            if reconnect_action is not None:
                # Line reconnections don't affect topology flag
                return reconnect_action, None
        
        max_rho = observation.rho.max()
        
        if not grid_state['needs_intervention']:
            # PRIORITY 2: If grid is very safe, reset topology to reference (only if not already in reference)
            reset_threshold = self.config.get('topology_reset_threshold', 0.75)
            if max_rho <= reset_threshold and not self.is_reference_topology:
                reset_action = self._reset_topology_if_modified(observation)
                if reset_action is not None:
                    # Successful topology reset - we're back to reference
                    self.is_reference_topology = True
                    return reset_action, None
            
            # Grid is safe - do nothing
            if AGENT_DEBUG:
                print(f"🔍 SAFE STATE: No action needed (rho={max_rho:.3f})")
            return self.action_space(), None
        
        # Grid needs intervention - use mode-based action selection
        if AGENT_DEBUG:
            print(f"🔍 INTERVENTION STATE: Using {self.mode} mode (rho={max_rho:.3f})")
        
        if self.mode == 'train':
            # Training: Full MCTS with detailed debug output
            base_action, base_action_idx, training_data = self._get_mcts_action(observation, grid_state)
            # Store training data for collection by trainer
            self.last_training_data = training_data
        elif self.mode == 'test':
            # Evaluation: Fast neural network inference only
            base_action, base_action_idx = self._get_fast_nn_action(observation, grid_state)
            self.last_training_data = None
        else:
            # Fallback: use existing AlphaZero method
            base_action, base_action_idx = self._get_alphazero_action(observation, grid_state)
            self.last_training_data = None
        
        if base_action is None:
            # Fallback if selected method fails
            base_action = self.action_space()
            base_action_idx = None
        else:
            # Any topology action means no longer in reference
            self.is_reference_topology = False
        
        return base_action, base_action_idx

    def _get_action_combined(self, observation: BaseObservation):
        """
        Select an action based on observation using action composition.
        Combines AlphaZero topology decisions with line reconnections/disconnections.
            
        Returns:
        --------
        tuple : (action, action_idx)
            Combined action and AlphaZero catalog index (None if no AlphaZero action)
        """
        max_rho = observation.rho.max()
        if AGENT_DEBUG:
            print(f"🔍 _get_action_combined ENTRY: rho={max_rho:.3f}")
        
        # Analyze grid state to determine action strategy
        grid_state = self._analyze_grid_state(observation)
        
        # Check if we're in reference topology state
        tested_action = self.action_space.get_back_to_ref_state(observation)
        if not tested_action or len(tested_action) == 0:
            # No reset actions available means we're already in reference state
            self.is_reference_topology = True
        else:
            self.is_reference_topology = False
        
        if not grid_state['needs_intervention']:
            # SAFE STATE: do-nothing + reconnections + maybe topology reset
            if AGENT_DEBUG:
                print(f"🔍 SAFE STATE: Building combined safe action (rho={max_rho:.3f})")
            
            base_action = self.action_space()  # do nothing
            base_action_idx = None
            
            # Always get reconnection recommendations when safe
            reconnections = self._get_reconnection_recommendations(observation)
            
            # Get topology reset if very safe (only if not already in reference)
            safety_threshold = self.config.get('actions', {}).get('topology_reset_module', {}).get('safety_threshold', 0.75)
            topology_reset = None
            if max_rho <= safety_threshold and not self.is_reference_topology:
                topology_reset = self._get_topology_reset_recommendations(observation)
            
            # No disconnections in safe state
            disconnections = []
            
        else:
            # INTERVENTION STATE: Mode-based action selection + reconnections + disconnections
            if AGENT_DEBUG:
                print(f"🔍 INTERVENTION STATE: Using {self.mode} mode (rho={max_rho:.3f})")
            
            if self.mode == 'train':
                # Training: Full MCTS with detailed debug output
                base_action, base_action_idx, training_data = self._get_mcts_action(observation, grid_state)
                # Store training data for collection by trainer
                self.last_training_data = training_data
            elif self.mode == 'test':
                # Evaluation: Fast neural network inference only
                base_action, base_action_idx = self._get_fast_nn_action(observation, grid_state)
                self.last_training_data = None
            else:
                # Fallback: use existing AlphaZero method
                base_action, base_action_idx = self._get_alphazero_action(observation, grid_state)
                self.last_training_data = None
            
            if base_action is None:
                # Fallback if selected method fails
                base_action = self.action_space()
                base_action_idx = None
            else:
                # Any topology action means no longer in reference
                self.is_reference_topology = False
            
            # Get reconnections (always beneficial - adds capacity and routing options)
            reconnections = self._get_reconnection_recommendations(observation)
            
            # Get emergency disconnections (module handles its own logic)
            disconnections = self._get_disconnection_recommendations(observation)
            
            # No topology reset during intervention
            topology_reset = None
            
        # Combine all action components
        combined_action = self._combine_actions(
            base_action, reconnections, disconnections, topology_reset
        )
        
        # Debug output
        action_types = []
        if base_action_idx is not None:
            action_types.append(f"AlphaZero-{base_action_idx}")
        elif not grid_state['needs_intervention']:
            action_types.append("DoNothing")
        else:
            action_types.append("Fallback")
            
        if reconnections:
            action_types.append(f"Reconnect({len(reconnections)})")
        if disconnections:
            action_types.append(f"Disconnect({len(disconnections)})")
        if topology_reset:
            action_types.append("TopoReset")
            
        action_desc = "+".join(action_types)
        if AGENT_DEBUG:
            print(f"🔑 COMBINED ACTION: {action_desc} (rho={max_rho:.3f})")
        
        return combined_action, base_action_idx
    
    def _get_alphazero_action(self, observation, grid_state):
        """
        Get AlphaZero's topology action recommendation.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        grid_state : dict
            Grid state analysis
            
        Returns:
        --------
        tuple : (action, action_idx)
            AlphaZero action and catalog index, or (None, None) if failed
        """
        if self.neural_network is None or not self.catalog_ready:
            return None, None
            
        try:
            # Import required functions
            from src.networks.neural_network_factory import get_neural_network_functions
            nn_funcs = get_neural_network_functions(self.config)
            
            num_actions = self.action_catalog.size
            mask = None
            if self.actions_config and self.actions_config.get('masking', True) and compute_action_mask is not None:
                mask = compute_action_mask(observation, self.action_catalog)
            
            if AGENT_DEBUG:
                print(f"🔍 DEBUG: Using neural network: {type(self.neural_network).__name__}")
            
            action_probs, value = nn_funcs.neural_network_forward(
                self.neural_network,
                observation,
                num_actions,
                mask=mask,
                config=self.config,
                env=self.current_env
            )
            
            # Debug: Always show neural network recommendations for top 3 actions
            print(f"\n🧠 AlphaZero Topology Analysis:")
            print(f"   Max line load: {grid_state['max_rho']:.3f}")
            print(f"   Overloaded lines: {grid_state['overloaded_lines']}")
            print(f"   Network value estimate: {value:.3f}")

            # Show top 3 action recommendations
            top_indices = np.argsort(action_probs)[-3:][::-1]
            print(f"   🧠 Top 3 AlphaZero Recommendations:")
            for i, idx in enumerate(top_indices):
                prob = action_probs[idx]
                sub_action = self.action_catalog.actions[idx]
                action_desc = f"Sub {sub_action.sub_id} layout #{idx}"
                print(f"      {i+1}. Action {idx}: {action_desc} (prob: {prob:.3f})")

            # Select action (use the highest probability action)
            selected_action_idx = int(np.argmax(action_probs))
            
            if selected_action_idx >= self.action_catalog.size:
                raise ValueError(f"Selected index {selected_action_idx} outside catalog size {self.action_catalog.size}")
                
            selected_action = self.action_catalog.actions[selected_action_idx].apply(self.action_space)
            print(f"   🎯 AlphaZero selected: Action {selected_action_idx} (Sub {self.action_catalog.actions[selected_action_idx].sub_id})")
            
            return selected_action, selected_action_idx
            
        except Exception as e:
            print(f"❌ AlphaZero topology action failed: {e}")
            return None, None
    
    def _get_mcts_action(self, observation, grid_state):
        """
        Get action using full MCTS search with neural network guidance (training mode).
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        grid_state : dict
            Grid state analysis
            
        Returns:
        --------
        tuple : (action, action_idx, training_data_list)
            Action, catalog index, and training data collected from MCTS
        """
        if not self.catalog_ready or self.mcts_config is None:
            print(f"❌ MCTS action failed: missing requirements (catalog={self.catalog_ready}, config={self.mcts_config is not None})")
            return None, None, []

        value_method = self.mcts_config.get('value_target_method', 'mcts_root')
        if value_method != 'no_guidance' and self.neural_network is None:
            print(f"❌ MCTS action failed: neural network required for mode '{value_method}'")
            return None, None, []
            
        try:
            max_rho = observation.rho.max()
            critical_threshold = self.mcts_config.get('core_agent', {}).get('critical_threshold', 0.98)
            
            if AGENT_DEBUG:
                print(f"\n🔴 MCTS ACTION: rho={max_rho:.3f} (critical_threshold={critical_threshold})")
            
            # COMPARISON: What would do-nothing do?
            do_nothing_failed = False
            try:
                do_nothing_action = self.action_space()
                sim_obs, sim_reward, sim_done, sim_info = observation.simulate(do_nothing_action)
                do_nothing_rho = sim_obs.rho.max()
                rho_delta = do_nothing_rho - max_rho
                if sim_done:
                    if AGENT_DEBUG:
                        print(f"  🚫 Do-nothing would FAIL (rho={do_nothing_rho:.3f})")
                    do_nothing_failed = True
                elif rho_delta > 0:
                    if AGENT_DEBUG:
                        print(f"  ⬆️ Do-nothing would increase rho: {max_rho:.3f} → {do_nothing_rho:.3f} (+{rho_delta:.3f})")
                elif rho_delta < 0:
                    if AGENT_DEBUG:
                        print(f"  ⬇️ Do-nothing would decrease rho: {max_rho:.3f} → {do_nothing_rho:.3f} ({rho_delta:.3f})")
                else:
                    if AGENT_DEBUG:
                        print(f"  ➡️ Do-nothing would maintain rho: {max_rho:.3f}")
            except Exception as e:
                if AGENT_DEBUG:
                    print(f"  ⚠️ Could not simulate do-nothing: {e}")
                do_nothing_failed = True
            
            # Determine value function based on value_target_method
            value_method = self.mcts_config.get('value_target_method', 'mcts_root')
            if value_method == 'heuristic':
                # Use heuristic value function during MCTS search
                from training.alphazero_mcts_v2 import compute_heuristic_value
                from rewards.reward_factory import get_reward_class
                
                # Get reward class from config
                reward_class_name = self.config.get('environment', {}).get('reward_class', 'AlphaZero')
                reward_class = get_reward_class(reward_class_name)
                heuristic_reward_fn = reward_class()
                
                def heuristic_value_fn(obs):
                    actual_reward = heuristic_reward_fn(
                        action=None, env=None, has_error=False,
                        is_done=False, is_illegal=False, is_ambiguous=False, obs=obs
                    )
                    from training.alphazero_mcts_v2 import MCTSNodeV2
                    temp_node = MCTSNodeV2(env=None, observation=obs, edge_reward=actual_reward)
                    return compute_heuristic_value(
                        temp_node,
                        gamma=self.mcts_config.get('gamma', 0.95),
                        horizon=self.mcts_config.get('heuristic_value_horizon', 100)
                    )
                value_fn = heuristic_value_fn
                policy_fn = lambda o: self.nn_funcs.neural_network_forward(self.neural_network, o, self.action_catalog_for_mcts.size, config=self.config, env=self.current_env)[0]
            elif value_method == 'no_guidance':
                # Pure MCTS with no neural network guidance
                policy_fn = None
                value_fn = None
            else:
                # Use neural network for both policy and value
                policy_fn = lambda o: self.nn_funcs.neural_network_forward(self.neural_network, o, self.action_catalog_for_mcts.size, config=self.config, env=self.current_env)[0]
                value_fn = lambda o: self.nn_funcs.neural_network_forward(self.neural_network, o, self.action_catalog_for_mcts.size, config=self.config, env=self.current_env)[1]
            
            # Run MCTS
            if AGENT_DEBUG:
                print(f"  Running MCTS with {self.mcts_config['mcts_simulations']} simulations...")
            
            from training.alphazero_mcts_v2 import run_mcts, select_action, collect_all_tree_nodes
            from networks.neural_network import encode_observation
            
            root, stats = run_mcts(
                env=self.current_env,
                observation=observation,
                action_catalog=self.action_catalog_for_mcts,
                num_simulations=self.mcts_config['mcts_simulations'],
                c_puct=self.mcts_config['c_puct'],
                gamma=self.mcts_config['gamma'],
                policy_fn=policy_fn,
                value_fn=value_fn,
                max_depth=self.mcts_config['max_depth'],
                epsilon=self.mcts_config.get('mcts_epsilon', 0.0),
                verbose=False,
                critical_threshold=critical_threshold,
                t_skipped=self.mcts_config.get('t_skipped', 80),
                t_stopping=self.mcts_config.get('t_stopping', 30),
                auto_reconnect=False,  # Disabled: agent handles reconnections at higher level
                max_reconnections=self.mcts_config.get('max_reconnections_per_action', 1),
                prefilter_rho_increase=0.20,  # Restore original default for proper tree depth
                dirichlet_alpha=self.mcts_config.get('dirichlet_alpha', 0.3),
                dirichlet_epsilon=self.mcts_config.get('dirichlet_epsilon', 0.25),
                penalty_for_failure=self.mcts_config.get('penalty_for_failure', -5.0),
                config=self.config  # Pass config for reward class loading
            )
            
            # Print tree statistics
            recovery_info = f" ({stats.get('recovery_nodes', 0)} recovery nodes)" if stats.get('recovery_nodes', 0) > 0 else ""
            early_stop_info = f" - early stopped at {stats.get('simulations_run', 0)}/{self.mcts_config['mcts_simulations']}" if stats.get('simulations_run', 0) < self.mcts_config['mcts_simulations'] else ""
            if AGENT_DEBUG:
                print(f"  MCTS complete: {len(root.children)} root children, max_depth={stats['max_depth']}{recovery_info}{early_stop_info}")
            
            # Select action using GREEDY selection
            action_idx = select_action(root, temperature=0, epsilon=0.0)
            
            # Get MCTS policy and training data
            num_actions = self.action_catalog_for_mcts.size
            visits = np.array([root.children.get(i, type('MockNode', (), {'visit_count': 0})()).visit_count 
                              for i in range(num_actions)])
            total_visits = visits.sum()
            
            if total_visits > 0:
                mcts_policy = visits / total_visits
            else:
                mcts_policy = np.ones(num_actions) / num_actions
                
            # Calculate policy entropy for debugging
            policy_nonzero = mcts_policy[mcts_policy > 0]
            policy_entropy = -np.sum(policy_nonzero * np.log(policy_nonzero + 1e-10))
            max_entropy = np.log(num_actions)
            normalized_entropy = policy_entropy / max_entropy
            if AGENT_DEBUG:
                print(f"  🎯 Policy entropy: {policy_entropy:.3f} (normalized: {normalized_entropy:.3f})")
            
            # Collect training data based on value_target_method
            training_data = []
            if value_method == 'mcts_root':
                state_vector = encode_observation(observation, self.mcts_config, self.current_env)
                root_value = root.value()
                training_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': root_value,
                    'source': 'mcts_root'
                })
            elif value_method == 'mcts_all_nodes':
                all_nodes_data = collect_all_tree_nodes(root, lambda obs: encode_observation(obs, self.mcts_config, self.current_env))
                training_data.extend(all_nodes_data)
                if AGENT_DEBUG:
                    print(f"  📦 Collected {len(all_nodes_data)} nodes from MCTS tree")
            elif value_method == 'heuristic':
                state_vector = encode_observation(observation, self.mcts_config, self.current_env)
                training_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': None,  # No value target - using heuristic
                    'source': 'heuristic'
                })
            elif value_method == 'no_guidance':
                state_vector = encode_observation(observation, self.mcts_config, self.current_env)
                root_value = root.value()
                training_data.append({
                    'state': state_vector,
                    'policy': mcts_policy,
                    'value': root_value,
                    'source': 'no_guidance'
                })
            
            # Store MCTS tree depth statistics for proper avg_depth calculation
            mcts_tree_depth = stats.get('max_depth', 0)
            if not hasattr(self, 'episode_tree_depths'):
                self.episode_tree_depths = []
            self.episode_tree_depths.append(mcts_tree_depth)
            
            # Store training data with tree depth info for the trainer
            self.last_training_data = training_data
            
            # Get selected action
            selected_action = self.action_catalog_for_mcts.actions[action_idx].apply(self.action_space)
            
            if action_idx in root.children:
                selected_max_steps = root.children[action_idx].max_reachable_steps
                if AGENT_DEBUG:
                    print(f"  ✅ MCTS selected action {action_idx} (max_steps={selected_max_steps}, visits={visits[action_idx]}, tree_depth={mcts_tree_depth})")
            else:
                if AGENT_DEBUG:
                    print(f"  ✅ MCTS selected action {action_idx} (fallback, tree_depth={mcts_tree_depth})")
            
            return selected_action, action_idx, training_data
            
        except Exception as e:
            print(f"❌ MCTS action failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []
    
    def _get_fast_nn_action(self, observation, grid_state):
        """
        Get action using fast neural network inference only (test mode).
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        grid_state : dict
            Grid state analysis
            
        Returns:
        --------
        tuple : (action, action_idx)
            Action and catalog index from direct NN inference
        """
        if self.neural_network is None or not self.catalog_ready:
            return None, None
            
        try:
            print(f"⚡ FAST NN ACTION: rho={grid_state['max_rho']:.3f} (test mode)")
            
            # Direct neural network forward pass
            num_actions = self.action_catalog.size
            mask = None
            if self.actions_config and self.actions_config.get('masking', True) and compute_action_mask is not None:
                mask = compute_action_mask(observation, self.action_catalog)
            
            action_probs, value = self.nn_funcs.neural_network_forward(
                self.neural_network,
                observation,
                num_actions,
                mask=mask,
                config=self.config,
                env=(self.current_env or self._baseline_env)
            )
            
            print(f"  🧠 NN value estimate: {value:.3f}")
            
            # Select highest probability action (greedy)
            selected_action_idx = int(np.argmax(action_probs))
            selected_action = self.action_catalog.actions[selected_action_idx].apply(self.action_space)
            
            print(f"  ⚡ Fast NN selected: Action {selected_action_idx} (prob: {action_probs[selected_action_idx]:.3f})")
            
            return selected_action, selected_action_idx
            
        except Exception as e:
            print(f"❌ Fast NN action failed: {e}")
            return None, None
    
    def _auto_reconnect_lines(self, observation):
        """
        Automatically reconnect disconnected lines when cooldown ends.
        """
        # Find disconnected lines with no cooldown
        disconnected = np.where(observation.line_status == False)[0]
        
        for line_id in disconnected:
            # Check if cooldown has ended
            if observation.time_before_cooldown_line[line_id] == 0:
                # Reconnect this line
                action = self.action_space()
                action.line_set_status = [(line_id, 1)]  # 1 = reconnect
                print(f"   🔌 Auto-reconnecting line {line_id} (cooldown ended)")
                return action
        
        # No lines need reconnection
        return None

    def _reset_topology_if_modified(self, observation):
        """
        Reset topology to reference configuration if grid is safe and topology is modified.
        """
        try:
            from src.actions.topology_reset import get_reference_topology_action
            
            # Check each substation to see if any are modified
            n_sub = observation.n_sub
            modified_subs = []
            
            for sub_id in range(n_sub):
                try:
                    sub_topo = observation.state_of(substation_id=sub_id)
                    topo_vect = sub_topo['topo_vect']
                    
                    # If any element is not on bus 1, substation is modified
                    if np.any(topo_vect != 1):
                        modified_subs.append(sub_id)
                except:
                    continue
            
            # If topology is already reference, no reset needed
            if len(modified_subs) == 0:
                return None
            
            # Create reset action
            env_to_use = self.current_env or self._baseline_env
            if env_to_use is None:
                return None
            reset_action = get_reference_topology_action(observation, env_to_use)
            max_rho = observation.rho.max()
            if AGENT_DEBUG:
                print(f"   🔄 Grid is very safe (rho={max_rho:.3f}) - resetting topology to reference ({len(modified_subs)} modified substations)")
            return reset_action
            
        except Exception as e:
            # If topology reset fails, just return None and do nothing
            print(f"   ⚠️ Topology reset analysis failed: {e}")
            return None
    
    def _get_reconnection_recommendations(self, observation):
        """
        Get line reconnection recommendations using RecoPowerlineModule.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of (line_id, 1) tuples for lines to reconnect
        """
        reconnections = []
        
        # Check if reconnection module is enabled
        reconnection_config = self.config.get('actions', {}).get('reconnection_module', {})
        if not reconnection_config.get('enabled', True):
            return reconnections
        
        try:
            from src.actions.reconnection import RecoPowerlineModule
            
            # Only try to reconnect if there are disconnected lines
            disconnected = np.where(observation.line_status == False)[0]
            if len(disconnected) == 0:
                return reconnections
            
            # Use environment with full action space for line reconnections
            env_to_use = self.current_env or self._baseline_env
            if env_to_use is None:
                return reconnections
                
            # Use RecoPowerlineModule to intelligently reconnect lines
            if RecoPowerlineModule is not None:
                reconnect_agent = RecoPowerlineModule(env_to_use.action_space)
                reconnect_action = reconnect_agent.get_act(observation, base_action=None, reward=0.0)
                
                # Extract reconnection recommendations from action
                if hasattr(reconnect_action, 'line_set_status') and reconnect_action.line_set_status is not None:
                    line_set_status = reconnect_action.line_set_status
                    
                    # Handle different formats that grid2op might return
                    if isinstance(line_set_status, tuple) and len(line_set_status) == 2:
                        line_id, status = line_set_status
                        if status == 1:
                            reconnections.append((line_id, 1))
                            print(f"   🔌 RecoPowerlineModule recommends reconnecting line {line_id}")
                    elif isinstance(line_set_status, list) and len(line_set_status) > 0:
                        try:
                            for line_id, status in line_set_status:
                                if status == 1:
                                    reconnections.append((line_id, 1))
                            if reconnections:
                                print(f"   🔌 RecoPowerlineModule recommends reconnecting {len(reconnections)} line(s): {[lid for lid, _ in reconnections]}")
                        except (ValueError, TypeError):
                            if AGENT_DEBUG:
                                print(f"   ⚠️ Unexpected line_set_status format in list: {line_set_status}")
                    elif isinstance(line_set_status, np.ndarray):
                        for line_id, status in enumerate(line_set_status):
                            if status == 1:
                                reconnections.append((line_id, 1))
                        if reconnections:
                            if AGENT_DEBUG:
                                print(f"   🔌 RecoPowerlineModule recommends reconnecting {len(reconnections)} line(s): {[lid for lid, _ in reconnections]}")
            else:
                # Check if fallback is enabled
                if reconnection_config.get('fallback_to_manual', True):
                    # Fallback to manual reconnection if module not available
                    cooldown_respect = reconnection_config.get('cooldown_respect', True)
                    for line_id in disconnected:
                        if not cooldown_respect or observation.time_before_cooldown_line[line_id] == 0:
                            reconnections.append((line_id, 1))
                            print(f"   🔌 Manual reconnection recommendation: line {line_id} (cooldown ended or ignored)")
                        
        except Exception as e:
            print(f"   ⚠️ Line reconnection analysis failed: {e}")
            
        return reconnections
    
    def _get_disconnection_recommendations(self, observation):
        """
        Get emergency line disconnection recommendations.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of (line_id, -1) tuples for lines to disconnect
        """
        try:
            from src.actions.disconnect import DisconnectOverloadedModule
            
            # Use environment with full action space
            env_to_use = self.current_env or self._baseline_env
            if env_to_use is None:
                return []
                
            disconnect_agent = DisconnectOverloadedModule(env_to_use.action_space, self.config)
            disconnections = disconnect_agent.get_disconnection_recommendations(observation, overflow_history=self.overflow_history)
            
            return disconnections
            
        except Exception as e:
            print(f"   ⚠️ Line disconnection analysis failed: {e}")
            return []
    
    def _get_topology_reset_recommendations(self, observation):
        """
        Get topology reset recommendations when grid is very safe.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
            
        Returns:
        --------
        Action or None
            Grid2Op Action object for resetting to reference topology, or None if not needed
        """
        # Check if topology reset module is enabled
        topology_config = self.config.get('actions', {}).get('topology_reset_module', {})
        if not topology_config.get('enabled', True):
            return None
            
        try:
            from src.actions.topology_reset import get_reference_topology_action
            
            # Check each substation to see if any are modified
            n_sub = observation.n_sub
            modified_subs = []
            
            for sub_id in range(n_sub):
                try:
                    sub_topo = observation.state_of(substation_id=sub_id)
                    topo_vect = sub_topo['topo_vect']
                    
                    # If any element is not on bus 1, substation is modified
                    if np.any(topo_vect != 1):
                        modified_subs.append(sub_id)
                except:
                    continue
            
            # If topology is already reference, no reset needed
            if len(modified_subs) == 0:
                return None
            
            # Create reset action using environment with full action space
            env_to_use = self.current_env or self._baseline_env
            if env_to_use is None:
                return None
                
            reset_action = get_reference_topology_action(observation, env_to_use)
            max_rho = observation.rho.max()
            if AGENT_DEBUG:
                print(f"   🔄 Grid is very safe (rho={max_rho:.3f}) - recommending topology reset ({len(modified_subs)} modified substations)")
            
            # Return the actual Grid2Op Action object (not a dict!)
            return reset_action
            
        except Exception as e:
            print(f"   ⚠️ Topology reset analysis failed: {e}")
            return None
    
    def _combine_actions(self, base_action, reconnections, disconnections, topology_reset):
        """
        Combine multiple action components into a single Grid2Op action using Grid2Op's + operator.
        
        Parameters:
        -----------
        base_action : Action
            Base action (AlphaZero topology or do-nothing)
        reconnections : List[Tuple[int, int]]
            List of (line_id, 1) for reconnections
        disconnections : List[Tuple[int, int]]
            List of (line_id, -1) for disconnections
        topology_reset : Action or None
            Topology reset Grid2Op Action from RecoverInitTopoModule
            
        Returns:
        --------
        Action
            Combined Grid2Op action
        """
        try:
            # Start with base action
            combined = base_action
            
            # Create separate actions for each component and combine them using Grid2Op's + operator
            actions_to_combine = []
            
            # Handle reconnections
            for line_id, status in reconnections:
                if status == 1:  # reconnect
                    reconnect_action = self.action_space({"set_line_status": [(line_id, 1)]})
                    actions_to_combine.append(reconnect_action)
                    
            # Handle disconnections  
            for line_id, status in disconnections:
                if status == -1:  # disconnect
                    disconnect_action = self.action_space({"set_line_status": [(line_id, -1)]})
                    actions_to_combine.append(disconnect_action)
            
            # Add topology reset (only if base action doesn't have topology changes)
            if topology_reset is not None:
                # Check if base action has topology changes
                has_topo_changes = (hasattr(combined, 'sub_set_bus') and 
                                   combined.sub_set_bus is not None and 
                                   any(combined.sub_set_bus))
                
                if not has_topo_changes:
                    actions_to_combine.append(topology_reset)
                    if AGENT_DEBUG:
                        print(f"   ➕ Adding topology reset action")
                else:
                    if AGENT_DEBUG:
                        print(f"   ⚠️ Skipping topology reset - base action already has topology changes")
            
            # Combine all actions using Grid2Op's + operator
            for action in actions_to_combine:
                combined = combined + action
                
            if len(reconnections) > 0 and AGENT_DEBUG:
                print(f"   ➕ Added {len(reconnections)} reconnections")
            if len(disconnections) > 0 and AGENT_DEBUG:
                print(f"   ➕ Added {len(disconnections)} disconnections")
            
            return combined
            
        except Exception as e:
            print(f"   ⚠️ Action combination failed: {e}")
            # Return base action as fallback
            return base_action
        
    def _analyze_grid_state(self, observation: BaseObservation):
        """
        Analyze the current grid state to determine if line switching is needed.
        
        For line switching, we mainly care about:
        - Line overloads (rho > threshold)
        - Disconnected lines that could help
        - Critical situations requiring immediate action
        
        Parameters:
        -----------
        observation : BaseObservation
            Current observation
            
        Returns:
        --------
        state_info : dict
            Dictionary with grid state analysis focused on line switching needs
        """
        max_rho = observation.rho.max()
        overloaded_lines = np.where(observation.rho > 1.0)[0]
        high_load_lines = np.where(observation.rho > self.intervention_threshold)[0] 
        disconnected_lines = np.where(observation.line_status == False)[0]
        
        state_info = {
            'needs_intervention': max_rho > self.intervention_threshold,
            'critical_overload': max_rho > self.critical_threshold,
            'max_rho': float(max_rho),
            'overloaded_lines': overloaded_lines.tolist(),
            'high_load_lines': high_load_lines.tolist(), 
            'disconnected_lines': disconnected_lines.tolist(),
            'n_overloaded': len(overloaded_lines),
            'n_disconnected': len(disconnected_lines)
        }
        
        return state_info
    
    def _fallback_line_switching_action(self, observation):
        """
        Fallback to simple line switching when bus actions are not available
        
        Parameters:
        -----------
        observation : BaseObservation
            Current observation
            
        Returns:
        --------
        action : Action
            Simple line switching action or do nothing
        """
        # Simple rule: disconnect the most overloaded line if it can be switched
        max_rho = observation.rho.max()
        
        if max_rho > self.critical_threshold:
            # Find most overloaded line that can be switched
            overloaded_lines = np.where(observation.rho > 1.0)[0]
            
            for line_id in overloaded_lines:
                if observation.time_before_cooldown_line[line_id] == 0 and observation.line_status[line_id]:
                    # Disconnect this overloaded line
                    action = self.action_space()
                    action.line_set_status = [(line_id, -1)]
                    print(f"   🔧 Fallback: Disconnecting overloaded line {line_id} (rho: {observation.rho[line_id]:.3f})")
                    return action
        
        # No action needed or possible
        return self.action_space()
        

    
    def _build_line_switching_actions(self, observation):
        """
        Build list of valid line switching actions + do nothing
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
            
        Returns:
        --------
        valid_actions : list
            List of Grid2Op actions (line switching + do nothing)
        """
        actions = []
        
        # Add do-nothing action
        actions.append(self.action_space())
        
        # Add line switching actions (disconnect/reconnect each line)
        for line_id in range(observation.n_line):
            # Only add if line status can be changed (no cooldown)
            if observation.time_before_cooldown_line[line_id] == 0:
                # Toggle line status (disconnect if connected, reconnect if disconnected)
                if observation.line_status[line_id]:
                    # Line is connected, add disconnect action
                    action = self.action_space({"set_line_status": [(line_id, -1)]})
                else:
                    # Line is disconnected, add reconnect action  
                    action = self.action_space({"set_line_status": [(line_id, 1)]})
                actions.append(action)
        
        return actions
    
    def save_model(self, filepath):
        """
        Save the trained neural network model
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.neural_network is not None:
            try:
                # Use interface method - works with any implementation (AlphaZero, RBM, etc.)
                self.neural_network.save_model(filepath)
                print(f"✅ Model saved to: {filepath}")
            except Exception as e:
                print(f"❌ Model save failed: {e}")
        else:
            print("⚠️ No neural network to save")
    
    def load_model(self, filepath):
        """
        Load a trained neural network model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        try:
            from src.networks.neural_network import load_neural_network
            
            # Use standalone loading function - no need to create empty network first!
            old_network = self.neural_network
            self.neural_network = load_neural_network(filepath)
            print(f"✅ Model loaded from: {filepath}")
            if AGENT_DEBUG:
                print(f"🔍 DEBUG: Old network: {type(old_network).__name__ if old_network else 'None'}")
                print(f"🔍 DEBUG: New network: {type(self.neural_network).__name__ if self.neural_network else 'None'}")
            print(f"✅ Ready for inference!")
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            if AGENT_DEBUG:
                print(f"🔍 DEBUG: Exception type: {type(e).__name__}")
                import traceback
                print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            self.neural_network = None
    
    def _update_overflow_history(self, observation):
        """
        Update overflow history tracking for sustained overflow detection.
        
        This method tracks how many consecutive timesteps each line has been overloaded,
        which is used by the disconnect module for sustained overflow detection.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        """
        try:
            # Get disconnect module config for threshold
            disconnect_config = self.config.get('actions', {}).get('disconnect_module', {})
            overflow_threshold = disconnect_config.get('emergency_threshold', 1.0)
            
            # Check which lines are currently overloaded
            overloaded_lines = np.where(observation.rho > overflow_threshold)[0]
            
            # Update history for all lines
            for line_id in range(observation.n_line):
                if line_id in overloaded_lines:
                    # Line is overloaded - increment counter
                    self.overflow_history[line_id] = self.overflow_history.get(line_id, 0) + 1
                else:
                    # Line is not overloaded - reset counter
                    if line_id in self.overflow_history:
                        del self.overflow_history[line_id]
            
            # Debug: Show sustained overflows if any exist
            sustained_overflows = {lid: count for lid, count in self.overflow_history.items() if count >= 2}
            if sustained_overflows and os.getenv("AGENT_DEBUG", "false").lower() == "true":
                print(f"   📊 Sustained overflows: {sustained_overflows}")
                
        except Exception as e:
            print(f"   ⚠️ Overflow history update failed: {e}")