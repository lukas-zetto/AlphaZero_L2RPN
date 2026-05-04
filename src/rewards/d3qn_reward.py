"""
D3QN-2022 Reward Implementation
Simple survival-based reward from the D3QN challenge

r(t) = {
    1, if survived time step t,
    0, if game over
}
"""

import numpy as np
from grid2op.Reward import BaseReward


class D3QNSurvivalReward(BaseReward):
    """
    D3QN-2022 survival reward implementation
    
    Very simple reward:
    - +1 for each time step survived
    - 0 if game over (environment handles this automatically)
    
    This encourages the agent to survive as long as possible
    without any complex reward shaping that might bias toward inaction.
    """
    
    # Unique identifier for verification
    REWARD_ID = "D3QN-2022-SURVIVAL"
    REWARD_VERSION = "v1.0"
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous, obs=None):
        """
        Compute the D3QN survival reward
        
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
            1.0 if survived this time step, 0.0 if game over
        """
        
        # If there was an error, illegal action, or episode is done -> game over
        if has_error or is_illegal or is_done:
            return 0.0
        
        # Otherwise, survived this time step -> +1 reward
        return 1.0