"""
MaxRho Reward Implementation
L2RPN 2023 LJN Agent MaxRho reward - line loading minimization
"""

import numpy as np
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float


class MaxRhoReward(BaseReward):
    """
    L2RPN 2023 LJN Agent MaxRho reward implementation.
    
    Simple line loading reward: 2.0 - obs.rho.max()
    
    Encourages keeping maximum line loading (ρ) low:
    - Max ρ = 0.5 → reward = 1.5
    - Max ρ = 1.0 (at capacity) → reward = 1.0  
    - Max ρ = 1.2 (overloaded) → reward = 0.8
    
    From the winning L2RPN 2023 solution.
    """
    
    # Unique identifier for verification
    REWARD_ID = "MAXRHO-L2RPN2023"
    REWARD_VERSION = "v1.0"
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous, obs=None):
        """
        Compute the MaxRho reward based on maximum line loading.
        
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
            The MaxRho reward value
        """
        
        # Handle error cases first
        if has_error or is_illegal or is_ambiguous or is_done:
            return float(self.reward_min)
            
        # Get current observation fresh from environment
        try:
            obs = env.get_obs(_do_copy=False)
        except Exception:
            return float(self.reward_min)
                
        # Calculate MaxRho reward: 2.0 - obs.rho.max()
        try:
            max_rho = obs.rho.max()
            reward = 2.0 - max_rho
            
            return float(np.clip(reward, self.reward_min, self.reward_max))
            
        except Exception:
            return float(self.reward_min)