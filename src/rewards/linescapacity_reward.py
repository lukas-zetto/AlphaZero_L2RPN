"""
Lines Capacity Reward from Grid2Op
Standard built-in reward that penalizes line capacity overflows
"""

from grid2op.Reward import LinesCapacityReward as Grid2OpLinesCapacityReward


class LinesCapacityReward(Grid2OpLinesCapacityReward):
    """
    Lines Capacity Reward wrapper from Grid2Op.
    
    This is the standard built-in reward that penalizes when transmission lines
    exceed their capacity limits. It is the default reward used in many L2RPN baselines.
    
    The reward is computed based on the maximum relative line loading (rho):
    - Rewards when all lines are within capacity (rho ≤ 1.0)
    - Penalizes when any line exceeds capacity (rho > 1.0)
    """
    
    # Unique identifier for verification
    REWARD_ID = "LINESCAPACITY-GRID2OP"
    REWARD_VERSION = "v1.0"
    
    def __init__(self, logger=None):
        """Initialize the Lines Capacity Reward"""
        super().__init__(logger=logger)
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous, obs=None):
        """
        Wrapper to handle the obs parameter that Grid2Op LinesCapacityReward doesn't expect.
        
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
        obs : BaseObservation, optional
            Current observation (ignored, Grid2Op reward uses env.get_obs())
            
        Returns:
        --------
        float
            The computed reward value
        """
        # Call parent method without the obs parameter
        return super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
