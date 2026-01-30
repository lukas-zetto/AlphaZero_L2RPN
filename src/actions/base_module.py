# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from abc import abstractmethod

import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Agent import BaseAgent, GreedyAgent
from grid2op.dtypes import dt_float


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


class GreedyModule(GreedyAgent, BaseModule):
    """Similar to the Grid2op GreedyAgent but implements a get_act method that performs
    similarly to the act method but combines tested actions with a base action.
    """

    def __init__(self, action_space):
        super().__init__(action_space)
        self.null_action_reward = -100.0

    def get_act(self, observation, base_action, reward, done=False, **kwargs):
        print(f"🔍 GreedyModule.get_act() called with {kwargs.keys()} kwargs")
        rho_threshold = (
            min(observation.rho.max(), kwargs["rho_threshold"])
            if "rho_threshold" in kwargs
            else observation.rho.max()
        )
        print(f"🔍 GreedyModule using rho_threshold: {rho_threshold:.3f}")

        self.tested_action = self._get_tested_action(observation)
        print(f"🔍 GreedyModule got {len(self.tested_action)} actions to test")
        
        if len(self.tested_action) == 0:
            print("🔍 GreedyModule: No actions to test, returning None")
            return None
            
        if len(self.tested_action) > 0:
            print("🔍 GreedyModule: Starting action simulation...")
            self.resulting_rewards = np.full(
                shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float
            )
            for i, action in enumerate(self.tested_action):
                print(f"🔍 GreedyModule: Simulating action {i+1}/{len(self.tested_action)}")
                try:
                    (
                        simul_obs,
                        simul_reward,
                        simul_has_error,
                        simul_info,
                    ) = observation.simulate(action + base_action)
                    
                    print(f"🔍 Simulation {i+1}: rho {simul_obs.rho.max():.3f}, error: {simul_has_error}, exceptions: {len(simul_info['exception'])}")
                    
                    if (
                        (0.0 < simul_obs.rho.max() < rho_threshold)
                        and (len(simul_info["exception"]) == 0)
                        and not simul_has_error
                    ):
                        self.resulting_rewards[i] = simul_reward
                        print(f"🔍 Action {i+1}: ACCEPTED with reward {simul_reward:.3f}")
                    else:
                        self.resulting_rewards[i] = self.null_action_reward
                        print(f"🔍 Action {i+1}: REJECTED (rho check: {0.0 < simul_obs.rho.max() < rho_threshold}, no exceptions: {len(simul_info['exception']) == 0}, no error: {not simul_has_error})")
                except Exception as e:
                    print(f"🔍 Action {i+1}: SIMULATION FAILED - {e}")
                    self.resulting_rewards[i] = self.null_action_reward
                    
            print(f"🔍 GreedyModule rewards: {self.resulting_rewards}")
            print(f"🔍 Max reward: {np.max(self.resulting_rewards):.3f}, null_action_reward: {self.null_action_reward}")
            
            if np.max(self.resulting_rewards) > self.null_action_reward:
                reward_idx = int(np.argmax(self.resulting_rewards))
                best_action = self.tested_action[reward_idx]
                print(f"🔍 GreedyModule: Selected action {reward_idx+1} with reward {self.resulting_rewards[reward_idx]:.3f}")
                return best_action
            else:
                print("🔍 GreedyModule: All actions rejected, returning None")
                return None