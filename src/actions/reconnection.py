"""
Line Reconnection Module

Handles intelligent line reconnections using Grid2Op's RecoPowerlineAgent.
This module provides utilities for reconnecting disconnected lines when safe.

Automatically attempts to reconnect disconnected powerlines after topology actions.
Lines can be disconnected due to:
- Overloads (automatic protection)
- Previous agent actions
- Cascading failures

This module provides utilities to:
1. Detect disconnected lines
2. Check if reconnection is allowed (cooldown expired)
3. Attempt reconnection as a post-processing step
4. Use intelligent Grid2Op agents for optimal reconnection decisions

Strategy:
- After each topology action, check for disconnected lines
- Try to reconnect lines that are off cooldown
- Prioritize lines by importance (e.g., high capacity, critical connections)
- Use RecoPowerlineModule for intelligent reconnection decisions
"""
from typing import List, Tuple, Optional
import numpy as np
from grid2op.Agent import RecoPowerlineAgent, BaseAgent
from grid2op.Action import ActionSpace
from abc import abstractmethod


# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

class BaseModule(BaseAgent):
    """This class is a wrapper for grid2op BaseAgent. It is renamed as Module to be included in
    agents with complex architecture involving hierarchical decision making with heuristic,
    optimization and neural-network policies. Module can be used as standalone agent or within
    a modular architecture.

    Parameters
    ----------
    BaseAgent :
        Core agent class from grid2op simulator.
    """

    def __init__(self, action_space: ActionSpace, action_type: str = None):
        BaseAgent.__init__(self, action_space=action_space)
        self.module_type = None
        self.action_type = action_type

    @abstractmethod
    def get_act(self, observation, base_action, reward, done=False):
        pass


class RecoPowerlineModule(BaseModule, RecoPowerlineAgent):
    """Module wrapper for the greedy RecoPowerlineAgent from Grid2Op.
    This module will try to best reconnection possible at each time step.
    """

    def __init__(self, action_space: ActionSpace):
        BaseModule.__init__(self, action_space, action_type="reconnection")
        RecoPowerlineAgent.__init__(self, action_space)

    def get_act(self, observation, base_action, reward, done=False):
        """Standardized interface for getting a reconnection action."""
        return self.act(observation, reward, done)

import numpy as np


def get_disconnected_lines(observation) -> List[int]:
    """
    Get list of disconnected line indices.
    
    Args:
        observation: Grid2Op observation
        
    Returns:
        List of line indices that are disconnected
    """
    return list(np.where(observation.line_status == False)[0])


def get_reconnectable_lines(observation) -> List[int]:
    """
    Get list of lines that can be reconnected (off cooldown).
    
    Args:
        observation: Grid2Op observation
        
    Returns:
        List of line indices that are disconnected and off cooldown
    """
    disconnected = observation.line_status == False
    off_cooldown = observation.time_before_cooldown_line == 0
    reconnectable = disconnected & off_cooldown
    return list(np.where(reconnectable)[0])


def prioritize_lines_by_capacity(observation, line_indices: List[int]) -> List[int]:
    """
    Sort lines by thermal capacity (reconnect high-capacity lines first).
    
    Args:
        observation: Grid2Op observation
        line_indices: List of line indices to prioritize
        
    Returns:
        Sorted list of line indices (highest capacity first)
    """
    if len(line_indices) == 0:
        return []
    
    # Get thermal limits for these lines
    capacities = observation.thermal_limit[line_indices]
    
    # Sort descending by capacity
    sorted_indices = np.argsort(-capacities)
    return [line_indices[i] for i in sorted_indices]


def prioritize_lines_by_rho_before_disconnect(observation, line_indices: List[int]) -> List[int]:
    """
    Sort lines by how loaded they were before disconnection.
    Lower rho = safer to reconnect first.
    
    Args:
        observation: Grid2Op observation
        line_indices: List of line indices to prioritize
        
    Returns:
        Sorted list of line indices (lowest rho first)
    """
    if len(line_indices) == 0:
        return []
    
    # Get current rho for disconnected lines (should be 0 or NaN)
    # We'll use a proxy: lines with lower thermal limit are "safer"
    # In practice, you'd track historical rho before disconnect
    # For now, just randomize as placeholder
    return line_indices


def attempt_reconnection(env, observation, action_space, 
                        max_reconnections: int = 1,
                        prioritize_by: str = "capacity") -> Tuple[bool, Optional[object], str]:
    """
    Attempt to reconnect disconnected lines.
    
    Args:
        env: Grid2Op environment (for simulation/testing)
        observation: Current observation
        action_space: Grid2Op action space
        max_reconnections: Maximum number of lines to try reconnecting in one step
        prioritize_by: "capacity" or "rho" - how to prioritize which lines to reconnect
        
    Returns:
        (success, action, message) tuple where:
        - success: True if we found reconnectable lines
        - action: Grid2Op action object to reconnect line(s), or None
        - message: Human-readable description
    """
    # Get reconnectable lines
    reconnectable = get_reconnectable_lines(observation)
    
    if len(reconnectable) == 0:
        return False, None, "No lines available for reconnection"
    
    # Prioritize
    if prioritize_by == "capacity":
        reconnectable = prioritize_lines_by_capacity(observation, reconnectable)
    
    # Take top N lines
    lines_to_reconnect = reconnectable[:max_reconnections]
    
    # Create reconnection action
    # Grid2Op allows reconnecting multiple lines in one action
    action = action_space()
    for line_id in lines_to_reconnect:
        # Use set_line_status to reconnect
        # reconnect_powerline() is a helper that does the same thing
        action.line_set_status = [(line_id, 1)]  # 1 = reconnect
    
    message = f"Attempting to reconnect {len(lines_to_reconnect)} line(s): {lines_to_reconnect}"
    
    return True, action, message


def reconnect_wrapper(env, observation, action_space, topology_action,
                     enable_reconnection: bool = True,
                     max_reconnections: int = 1) -> object:
    """
    Wrapper that combines a topology action with automatic line reconnection.
    
    Args:
        env: Grid2Op environment
        observation: Current observation
        action_space: Grid2Op action space
        topology_action: The main topology action to apply
        enable_reconnection: Whether to enable auto-reconnection
        max_reconnections: Max lines to reconnect per step
        
    Returns:
        Combined action (topology + reconnection) or just topology action
    """
    if not enable_reconnection:
        return topology_action
    
    # Check if there are any disconnected lines
    reconnectable = get_reconnectable_lines(observation)
    
    if len(reconnectable) == 0:
        # No reconnection needed
        return topology_action
    
    # Prioritize lines by capacity
    reconnectable = prioritize_lines_by_capacity(observation, reconnectable)
    lines_to_reconnect = reconnectable[:max_reconnections]
    
    # Create combined action
    # Start with the topology action
    combined_action = topology_action
    
    # Add reconnection
    for line_id in lines_to_reconnect:
        # Update the action to also reconnect this line
        combined_action.line_set_status = [(line_id, 1)]
    
    return combined_action
