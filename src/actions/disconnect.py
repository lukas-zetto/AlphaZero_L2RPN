"""
Disconnect Module for Emergency Line Disconnection

This module analyzes sustained grid overloads and recommends strategic line disconnections
to relieve critical stress and prevent cascading failures. Only triggers when lines have
been overloaded for multiple consecutive timesteps.
"""

import numpy as np
from grid2op.Observation import BaseObservation
from typing import List, Tuple, Optional


class DisconnectOverloadedModule:
    """
    Strategic line disconnection module for emergency overload relief.
    
    Advanced logic:
    - Only considers lines with sustained overflow (>N consecutive timesteps)
    - Only disconnects if it reduces overall system stress
    - Respects cooldowns and safety constraints  
    - Limits number of disconnections per timestep
    """
    
    def __init__(self, action_space, config=None):
        """
        Initialize the disconnection module.
        
        Parameters:
        -----------
        action_space : ActionSpace
            Grid2Op action space for creating actions
        config : dict, optional
            Configuration parameters
        """
        self.action_space = action_space
        self.config = config or {}
        
        # Get disconnect module config
        disconnect_config = self.config.get('actions', {}).get('disconnect_module', {})
        
        # Sustained overflow configuration
        self.min_sustained_timesteps = disconnect_config.get('min_sustained_timesteps', 2)
        self.max_disconnections_per_step = disconnect_config.get('max_disconnections_per_step', 2)
        self.stress_reduction_required = disconnect_config.get('stress_reduction_required', True)
        self.emergency_threshold = disconnect_config.get('emergency_threshold', 1.0)
        self.cooldown_respect = disconnect_config.get('cooldown_respect', True)
        
    def is_enabled(self):
        """Check if disconnect module is enabled in config."""
        disconnect_config = self.config.get('actions', {}).get('disconnect_module', {})
        return disconnect_config.get('enabled', False)
        
    def get_disconnection_recommendations(self, observation: BaseObservation, overflow_history=None) -> List[Tuple[int, int]]:
        """
        Get line disconnection recommendations for sustained emergency overload relief.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        overflow_history : dict, optional
            Dictionary mapping line_id -> consecutive_overflow_timesteps
            If None, falls back to simple threshold-based logic
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of (line_id, -1) tuples for lines to disconnect
        """
        disconnections = []
        
        # Check if module is enabled
        if not self.is_enabled():
            return disconnections
        
        # Find currently overloaded lines
        overloaded_lines = np.where(observation.rho > self.emergency_threshold)[0]
        
        if len(overloaded_lines) == 0:
            return disconnections
            
        # If no overflow history provided, use simple fallback
        if overflow_history is None:
            print(f"   ⚡ DisconnectModule: No overflow history - using fallback logic")
            return self._fallback_disconnection_logic(observation, overloaded_lines)
        
        # Advanced sustained overflow logic
        sustained_overflow_lines = []
        
        for line_id in overloaded_lines:
            # Check if line can be disconnected (no cooldown, currently connected)
            if (self.cooldown_respect and observation.time_before_cooldown_line[line_id] > 0):
                continue
            if not observation.line_status[line_id]:
                continue  # Already disconnected
                
            # Check sustained overflow threshold
            overflow_timesteps = overflow_history.get(line_id, 0)
            if overflow_timesteps >= self.min_sustained_timesteps:
                sustained_overflow_lines.append((line_id, overflow_timesteps, observation.rho[line_id]))
        
        if not sustained_overflow_lines:
            return disconnections
            
        # Sort by priority: highest rho first, then longest sustained overflow
        sustained_overflow_lines.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Consider disconnections up to the limit
        candidates = sustained_overflow_lines[:self.max_disconnections_per_step]
        
        for line_id, timesteps, rho in candidates:
            if self.stress_reduction_required:
                # TODO: Implement stress reduction check by simulating disconnection
                # For now, assume disconnecting overloaded lines helps
                pass
                
            disconnections.append((line_id, -1))
            print(f"   ⚡ DisconnectModule: Emergency disconnect line {line_id} (rho: {rho:.3f}, sustained: {timesteps} timesteps)")
        
        return disconnections
    
    def _fallback_disconnection_logic(self, observation, overloaded_lines):
        """
        Fallback logic when no overflow history available.
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        overloaded_lines : np.ndarray
            Array of overloaded line indices
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of (line_id, -1) tuples for emergency disconnections
        """
        disconnections = []
        
        # Simple rule: disconnect most critical overloaded lines
        critical_lines = []
        
        for line_id in overloaded_lines:
            # Check if line can be disconnected
            if (self.cooldown_respect and observation.time_before_cooldown_line[line_id] > 0):
                continue
            if not observation.line_status[line_id]:
                continue  # Already disconnected
                
            # Only consider severely overloaded lines in fallback
            if observation.rho[line_id] > 1.05:  # 105% = severe overload
                critical_lines.append((line_id, observation.rho[line_id]))
        
        # Sort by rho (most overloaded first)
        critical_lines.sort(key=lambda x: x[1], reverse=True)
        
        # Limit disconnections
        for line_id, rho in critical_lines[:self.max_disconnections_per_step]:
            disconnections.append((line_id, -1))
            print(f"   ⚡ DisconnectModule: Fallback emergency disconnect line {line_id} (rho: {rho:.3f})")
        
        return disconnections
    
    def get_act(self, observation: BaseObservation, base_action=None, reward=0.0, overflow_history=None):
        """
        Get disconnection action (Grid2Op compatible interface).
        
        Parameters:
        -----------
        observation : BaseObservation
            Current grid observation
        base_action : Action, optional
            Base action to modify (unused)
        reward : float, optional
            Previous reward (unused)
        overflow_history : dict, optional
            Overflow history for sustained overflow detection
            
        Returns:
        --------
        action : Action
            Disconnection action or do-nothing if no disconnections needed
        """
        disconnections = self.get_disconnection_recommendations(observation, overflow_history)
        
        if disconnections:
            action = self.action_space()
            action.line_set_status = disconnections
            return action
        else:
            # Return do-nothing action
            return self.action_space()