"""
Power Loss Minimization Reward Implementation
Efficiency-based reward minimizing power system losses
"""

import numpy as np
from grid2op.Reward import BaseReward


class LossReward(BaseReward):
    """
    Power loss minimization reward based on grid efficiency.
    
    RLoss(t) = (Σloads / Σgens) - 0.9
    
    Rewards:
    - High efficiency (low losses) with positive rewards
    - Penalizes illegal actions (-0.5) and game over (-1.0)
    - Typical efficiency ~0.97-0.99, so reward ~0.07-0.09
    """
    
    # Unique identifier for verification
    REWARD_ID = "LOSS-EFFICIENCY"
    REWARD_VERSION = "v1.0"
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def initialize(self, env):
        """Optional initialization - not needed for this reward"""
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous, obs=None):
        """
        Compute the loss-based reward focusing on grid efficiency.
        
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
        obs : Observation, optional
            Grid observation (if provided, avoids calling env.get_obs())
            
        Returns:
        --------
        reward : float
            The efficiency-based reward value
        """
        
        # Handle error cases first
        if has_error or is_illegal or is_ambiguous:
            return float(self.reward_illegal)
        if is_done:
            return float(self.reward_min)
            
        # Get current observation (use provided obs or get from environment)
        if obs is None:
            try:
                obs = env.get_obs(_do_copy=False)
            except Exception:
                return float(self.reward_illegal)
                
        # Calculate efficiency-based reward: (load_sum / gen_sum) - 0.9
        try:
            total_gen = obs.gen_p.sum()
            total_load = obs.load_p.sum()
            
            if total_gen <= 0:
                return float(self.reward_min)  # Avoid division by zero
                
            efficiency = total_load / total_gen
            reward = efficiency - 0.9  # Direct formula implementation
            
            return float(np.clip(reward, self.reward_min, self.reward_max))
            
        except Exception:
            return float(self.reward_illegal)