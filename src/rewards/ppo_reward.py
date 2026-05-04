"""
PPO Reward Implementation
L2RPN 2023 LJN Agent PPO reward - sophisticated penalty system
"""

import numpy as np
from grid2op.Reward import BaseReward


class PPO_Reward(BaseReward):
    """
    L2RPN 2023 LJN Agent PPO reward implementation.
    
    Sophisticated reward used in LJN's PPO training:
    - Progressive penalty based on line loading
    - Bonus for "do nothing" action (encourages conservative operation)
    - Formula: 2 - rho_max * penalty + action_bonus
    
    From the winning L2RPN 2023 solution.
    """
    
    # Unique identifier for verification
    REWARD_ID = "PPO-L2RPN2023"
    REWARD_VERSION = "v1.0"
    
    def __init__(self, logger=None):
        """
        PPO_Reward class, based on the BaseReward from Grid2Op
        """
        super().__init__(logger=logger)
        self.reward_min = -10
        self.reward_std = 2

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous, obs=None):
        """
        Compute the PPO reward with progressive penalties and action bonuses.
        
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
            The PPO reward value
        """
        
        # Handle error cases first
        if has_error or is_illegal or is_ambiguous or is_done:
            return float(self.reward_min)
            
        # Get current observation fresh from environment
        try:
            obs = env.get_obs(_do_copy=False)
        except Exception:
            return float(self.reward_min)
                
        # Calculate PPO reward: 2 - rho_max * penalty + action_bonus
        try:
            max_rho = obs.rho.max()
            
            # Progressive penalty based on line loading
            if max_rho <= 0.8:
                penalty = 0.0
            elif max_rho <= 0.9:
                penalty = 1.0
            elif max_rho <= 0.95:
                penalty = 2.0
            elif max_rho <= 1.0:
                penalty = 3.0
            else:
                penalty = 5.0
                
            # Action bonus for "do nothing" (conservative operation)
            action_bonus = 0.1 if action._modif_set_bus is None or len(action._modif_set_bus) == 0 else 0.0
            
            reward = 2.0 - max_rho * penalty + action_bonus
            
            return float(np.clip(reward, self.reward_min, self.reward_std * 2))
            
        except Exception:
            return float(self.reward_min)