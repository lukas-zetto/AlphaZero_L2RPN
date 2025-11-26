import numpy as np
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

        # Action configuration (new catalog or legacy)
        self.actions_config = self.config.get('ACTIONS_CONFIG') or self.config.get('actions_config')

        # Catalog attributes
        self.action_catalog = None
        self.catalog_ready = False
        self._baseline_env = None  # temporary env used for catalog build
        self._initialize_actions()

        # Initialize neural network for training
        self.neural_network = None
        self._initialize_neural_network()

        # Initialize agent parameters
        self.intervention_threshold = self.config.get('intervention_threshold', 0.95)
        self.critical_threshold = self.config.get('critical_threshold', 1.0)

        # Store reference to current environment
        self.current_env = None
        
        # Track last action taken for evaluation statistics
        self.last_action_taken = None
    
    def set_env(self, env):
        """Update environment reference."""
        self.current_env = env
    
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

            subs = self.actions_config.get('substations', [3,4,5,8])
            reduction = self.actions_config.get('reduction', 'N1')
            drop_identity = self.actions_config.get('drop_identity', True)
            
            # Build full catalog first
            full_catalog = build_action_catalog(env, subs, reduction, drop_identity)
            
            # Apply reduced action space if enabled
            try:
                from src.config import USE_REDUCED_ACTION_SPACE, REDUCED_ACTIONS
                if USE_REDUCED_ACTION_SPACE:
                    from dataclasses import replace
                    reduced_actions = [full_catalog.actions[idx] for idx in REDUCED_ACTIONS if idx < len(full_catalog.actions)]
                    self.action_catalog = replace(full_catalog, actions=reduced_actions)
                    print(f"✅ Built REDUCED catalog-based action space with {len(reduced_actions)} actions (indices: {REDUCED_ACTIONS})")
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
            input_size = self.config['input_size']  # Required config value

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
        Select an action based on observation
            
        Returns:
        --------
        tuple : (action, action_idx)
            Selected action and its catalog index (None if not a catalog action)
        """
        # PRIORITY 1: Auto-reconnect disconnected lines if cooldown ended
        reconnect_action = self._auto_reconnect_lines(observation)
        if reconnect_action is not None:
            return reconnect_action, None
        
        # Analyze grid state to determine if action is needed
        grid_state = self._analyze_grid_state(observation)
        
        # DEBUG: Print intervention decision
        max_rho = observation.rho.max()
        
        if not grid_state['needs_intervention']:
            # PRIORITY 2: If grid is very safe, reset topology to reference
            reset_threshold = self.config.get('topology_reset_threshold', 0.75)
            if max_rho <= reset_threshold:
                reset_action = self._reset_topology_if_modified(observation)
                if reset_action is not None:
                    return reset_action, None
            
            # Grid is safe - do nothing
            return self.action_space(), None
        
        # Grid needs intervention - use trained neural network if available
        if self.neural_network is not None and self.catalog_ready:
            try:
                # Import required functions
                from src.networks.neural_network_factory import get_neural_network_functions
                nn_funcs = get_neural_network_functions(self.config)
                
                num_actions = self.action_catalog.size
                mask = None
                if self.actions_config and self.actions_config.get('masking', True) and compute_action_mask is not None:
                    mask = compute_action_mask(observation, self.action_catalog)
                
                action_probs, value = nn_funcs.neural_network_forward(
                    self.neural_network,
                    observation,
                    num_actions,
                    mask=mask
                )
                
                # Debug: Always show neural network recommendations for top 3 actions
                print(f"\n🧠 Neural Network Debug Info:")
                print(f"   Max line load: {grid_state['max_rho']:.3f}")
                print(f"   Overloaded lines: {grid_state['overloaded_lines']}")
                print(f"   Network value estimate: {value:.3f}")

                # Show top 3 action recommendations
                top_indices = np.argsort(action_probs)[-3:][::-1]
                print(f"   🧠 Top 3 Neural Network Recommendations:")
                for i, idx in enumerate(top_indices):
                    prob = action_probs[idx]
                    sub_action = self.action_catalog.actions[idx]
                    action_desc = f"Sub {sub_action.sub_id} layout #{idx}"
                    print(f"      {i+1}. Action {idx}: {action_desc} (prob: {prob:.3f})")

                # Select action (use the highest probability action)
                selected_action_idx = int(np.argmax(action_probs))
                
                if self.action_space is None:
                    raise ValueError("action_space is None")
                if selected_action_idx >= self.action_catalog.size:
                    raise ValueError(f"Selected index {selected_action_idx} outside catalog size {self.action_catalog.size}")
                    
                selected_action = self.action_catalog.actions[selected_action_idx].apply(self.action_space)
                print(f"   🎯 Selected catalog action {selected_action_idx} (Sub {self.action_catalog.actions[selected_action_idx].sub_id})")
                return selected_action, selected_action_idx
                
            except Exception as e:
                print(f"   ❌ Neural network action failed: {e}")
                return self.action_space(), None  # Do nothing on error
                
            except Exception as e:
                print(f"⚠️ Neural network failed: {e}")
                # Fallback to line switching only if bus actions fail
                return self._fallback_line_switching_action(observation), None
        else:
            # No neural network loaded - use fallback behavior
            return self._fallback_line_switching_action(observation), None
    
    def _auto_reconnect_lines(self, observation):
        """
        Automatically reconnect disconnected lines when cooldown ends.
        
        This handles the case where:
        - Grid2Op auto-disconnected a line due to persistent overload
        - OR we manually disconnected a line
        - Cooldown period has ended (time_before_cooldown_line == 0)
        - Line can now be safely reconnected
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
            
        Returns:
        --------
        action : Action or None
            Reconnection action if a line should be reconnected, None otherwise
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
        
        This helps prevent topology drift by returning to the known-good baseline
        configuration when the grid is stable.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
            
        Returns:
        --------
        action : Action or None
            Reset action if topology should be reset, None otherwise
        """
        try:
            from actions.topology_reset import get_reference_topology_action
            
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
            reset_action = get_reference_topology_action(observation, self.action_space)
            max_rho = observation.rho.max()
            print(f"   🔄 Grid is very safe (rho={max_rho:.3f}) - resetting topology to reference ({len(modified_subs)} modified substations)")
            return reset_action
            
        except Exception as e:
            # If topology reset fails, just return None and do nothing
            return None
        
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
                import torch
                # Save model state dict
                torch.save({
                    'model_state_dict': self.neural_network.state_dict(),
                    'config': self.config,
                    'input_size': getattr(self.neural_network, 'input_size', None),
                    'output_size': getattr(self.neural_network, 'output_size', None)
                }, filepath)
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
            import torch
            from src.networks.neural_network_factory import get_neural_network_functions
            
            # Get functions for current implementation
            nn_funcs = get_neural_network_functions(self.config)
            
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Create neural network if not exists
            if self.neural_network is None:
                # Detect the correct number of actions from the saved model
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Get the number of actions from the policy head shape
                if 'policy_head.weight' in state_dict:
                    num_actions = checkpoint['num_actions']
                    print(f"🔍 Detected {num_actions} actions from saved model")
                else:
                    raise ValueError("Checkpoint must contain 'num_actions' field")
                
                input_size = self.config['input_size']
                
                self.neural_network = nn_funcs.create_neural_network(
                    input_size=input_size,
                    num_actions=num_actions,
                    config=self.config
                )
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Standard format with wrapper
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state dict format (from Alpha Zero training)
                state_dict = checkpoint
            
            # Load state dict
            self.neural_network.load_state_dict(state_dict)
            self.neural_network.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded from: {filepath}")
            print(f"✅ Using trained Alpha Zero weights!")
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            self.neural_network = None