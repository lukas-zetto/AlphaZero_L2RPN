import numpy as np
from grid2op.Reward import BaseReward


class MyCustomReward(BaseReward):
    """
    Custom shaped reward based on cumulative sum of overflowing line loads.
    
    The reward formulation utilizes a coefficient u that summarizes overflowing line loads,
    which the agent aims to minimize. The reward is calculated as:
    
    r = exp(-u - 0.5 * n_offline)
    
    Where:
    - u depends on line loads (ρ):
      * If ρ_max ≤ 1: u = max(ρ_max - 0.5, 0)
      * If ρ_max > 1: u = Σ(ρ_i - 0.5) for all overflowing lines i
    - n_offline is the number of lines currently offline due to overflow or actions
    """
    
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        """
        Compute the shaped reward based on line loads and offline lines.
        
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
            The computed shaped reward value
        """
        
        # Return zero reward for illegal actions or errors (let the environment handle penalties)
        if has_error or is_illegal:
            return 0.0
            
        # Get current observation
        obs = env.get_obs()
        
        # Calculate the coefficient u based on line loads
        u = self._calculate_u_coefficient(obs)
        
        # Count offline lines (excluding maintenance and opponent attacks)
        n_offline = self._count_offline_lines(obs)
        
        # Calculate shaped reward using exponential decay
        reward = np.exp(-u - 0.5 * n_offline)
        
        return float(reward)
        
    def _calculate_u_coefficient(self, obs):
        """
        Calculate the coefficient u based on current line loads.
        
        Parameters:
        -----------
        obs : Observation
            Current grid observation
            
        Returns:
        --------
        u : float
            The coefficient summarizing overflowing line loads
        """
        # Get line loads (ρ values)
        rho = obs.rho
        rho_max = np.max(rho)
        
        if rho_max <= 1.0:
            # No overflow case: u = max(ρ_max - 0.5, 0)
            u = max(rho_max - 0.5, 0.0)
        else:
            # Overflow case: u = Σ(ρ_i - 0.5) for all overflowing lines i
            overflowing_mask = rho > 1.0
            overflowing_loads = rho[overflowing_mask]
            u = np.sum(overflowing_loads - 0.5)
            
        return u
        
    def _count_offline_lines(self, obs):
        """
        Count the number of lines that are offline due to overflow or agent actions.
        
        This excludes lines that are offline due to maintenance or opponent attacks.
        
        Parameters:
        -----------
        obs : Observation
            Current grid observation
            
        Returns:
        --------
        n_offline : int
            Number of lines offline due to overflow or agent actions
        """
        # Lines are considered offline if they are not connected
        # In Grid2Op, line_status indicates if lines are connected (True) or not (False)
        offline_lines = ~obs.line_status
        
        # Filter out maintenance outages if information is available
        # Note: We assume obs.time_before_cooldown_line can help identify maintenance
        # Lines under maintenance typically have a cooldown time
        if hasattr(obs, 'time_before_cooldown_line'):
            # If a line is offline but has no cooldown, it's likely due to overflow/actions
            # If it has cooldown > 0, it might be maintenance or recent reconnection attempt
            maintenance_mask = (obs.time_before_cooldown_line > 0) & offline_lines
            n_offline = np.sum(offline_lines & ~maintenance_mask)
        else:
            # If maintenance information is not available, count all offline lines
            n_offline = np.sum(offline_lines)
            
        return int(n_offline)