"""
D3QN-2020 Composite Reward Implementation
Multi-component reward from the D3QN-2020 challenge
"""

import numpy as np
from grid2op.Reward import BaseReward


class D3QN2020Reward(BaseReward):
    """
    D3QN-2020 composite reward implementation
    
    r(t) = w1·RSandbox(t) + w2·RCloseToOverflow(t) + w3·RDistance(t) + w4·RLineCapa(t)
    
    Where:
    - RSandbox(t): based on costs for redispatch and loss
    - RCloseToOverflow(t): based on overflow of lines; less overflow gives higher reward
    - RDistance(t): based on distance from reference topology; smaller distance gives higher reward
    - RLineCapa(t): based on lines capacity usage; more available capacity gives higher reward
    
    Weights: w1=30, w2=200, w3=20, w4=3.0
    """
    
    # Unique identifier for verification
    REWARD_ID = "D3QN-2020-COMPOSITE"
    REWARD_VERSION = "v1.0"
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        
        # Weights from D3QN-2020 paper
        self.w1 = 30.0   # RSandbox weight
        self.w2 = 200.0  # RCloseToOverflow weight
        self.w3 = 20.0   # RDistance weight
        self.w4 = 3.0    # RLineCapa weight
        
        # Reference topology (will be set in initialize)
        self.reference_topology = None
        
    def initialize(self, env):
        """Initialize with reference topology"""
        try:
            obs = env.get_obs()
            self.reference_topology = obs.topo_vect.copy()
        except (AttributeError, TypeError):
            # Handle case where environment is not fully initialized
            # Will be set later when first reward is computed
            self.reference_topology = None
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous, obs=None):
        """
        Compute the D3QN-2020 composite reward
        
        Parameters:
        -----------
        action : Action
            The action taken by the agent
        env : Environment  
            The grid environment
        has_error : bool
            Whether there was an error in the action
        is_done : bool
            Whether the episode is finished
        is_illegal : bool
            Whether the action was illegal
        is_ambiguous : bool
            Whether the action was ambiguous
            
        Returns:
        --------
        reward : float
            Composite reward value
        """
        
        # Handle error cases
        if has_error or is_illegal or is_ambiguous or is_done:
            return -1000.0  # Large penalty for failure
        
        # Get current observation (use provided obs or get from environment)
        if obs is None:
            obs = env.get_obs(_do_copy=False)
        
        # Initialize reference topology if not set (in case initialize() failed)
        if self.reference_topology is None:
            try:
                self.reference_topology = obs.topo_vect.copy()
            except Exception:
                # Fallback to all ones if topology access fails
                raise ValueError("Unable to access topology from observation for reference initialization.")
        
        # Calculate individual reward components
        r_sandbox = self._calculate_sandbox_reward(obs)
        r_close_to_overflow = self._calculate_close_to_overflow_reward(obs)
        r_distance = self._calculate_distance_reward(obs)
        r_line_capa = self._calculate_line_capacity_reward(obs)
        
        # Weighted combination
        total_reward = (
            self.w1 * r_sandbox +
            self.w2 * r_close_to_overflow +
            self.w3 * r_distance +
            self.w4 * r_line_capa
        )
        
        return total_reward
    
    def _calculate_sandbox_reward(self, obs):
        """Calculate reward based on redispatch costs and losses"""
        # Calculate power losses
        gen_p = obs.gen_p.sum()
        load_p = obs.load_p.sum()
        
        if gen_p <= 0:
            return -1.0
            
        # Efficiency (1 - losses)
        efficiency = load_p / gen_p
        loss_penalty = -(1.0 - efficiency) * 100
        
        # Redispatch penalty (based on deviation from nominal production)
        # Simplified: penalize high generator variations
        gen_std = obs.gen_p.std() if len(obs.gen_p) > 1 else 0.0
        redispatch_penalty = -gen_std * 0.1
        
        return loss_penalty + redispatch_penalty
    
    def _calculate_close_to_overflow_reward(self, obs):
        """Calculate reward based on line overflow risk"""
        # Penalty for lines close to overflow
        max_rho = obs.rho.max()
        
        if max_rho >= 1.0:
            return -10.0  # Heavy penalty for actual overflow
        elif max_rho >= 0.95:
            return -5.0 * (max_rho - 0.95) / 0.05  # Steep penalty near overflow
        elif max_rho >= 0.85:
            return -2.0 * (max_rho - 0.85) / 0.10  # Moderate penalty
        else:
            return 1.0 - max_rho  # Reward for low loading
    
    def _calculate_distance_reward(self, obs):
        """Calculate reward based on distance from reference topology"""
        if self.reference_topology is None:
            return 0.0
            
        # Calculate topology distance (number of different elements)
        current_topo = obs.topo_vect
        distance = np.sum(current_topo != self.reference_topology)
        
        # Normalize by topology size and convert to reward
        # Smaller distance = higher reward
        normalized_distance = distance / len(current_topo)
        return 1.0 - normalized_distance
    
    def _calculate_line_capacity_reward(self, obs):
        """Calculate reward based on available line capacity"""
        # Reward based on average available capacity
        avg_rho = obs.rho.mean()
        available_capacity = 1.0 - avg_rho
        
        # More available capacity = higher reward
        return available_capacity