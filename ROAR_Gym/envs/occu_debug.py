try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
import math
from collections import OrderedDict


class OccuDebug(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        # action_space = target_speed, long_k, lat_k
        self.view_size = 100

        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)  # long (y) throttle, lat (x) steering,
        # observation_space = curr_speed, curr_throttle, curr_steering
        self.observation_space = gym.spaces.Box(low=-10, high=1,
                                                shape=(self.view_size, self.view_size, 1), dtype=np.float32)
        self._prev_speed = 0
        self._prev_waypoint = None

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, dict]:
        """

        Args:
            action:

        Returns:

        """
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        control = VehicleControl(throttle=action[0], steering=action[1])
        self.agent.kwargs["control"] = control
        agent, reward, is_done, other_info = super(OccuDebug, self).step(action=action)
        obs = self._get_obs()
        return obs, reward, is_done, other_info

    def _get_obs(self) -> Any:
        if self.agent.occupancy_map is not None:
            occu_map = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                        view_size=(self.view_size, self.view_size),
                                                        vehicle_value=-10)
            occu_map = np.expand_dims(occu_map, axis=2)
        else:
            occu_map = np.zeros(shape=(self.view_size, self.view_size, 1))
        return occu_map

    def get_reward(self) -> float:
        reward: float = 0.0

        return reward

    def _get_info(self) -> dict:
        info_dict = OrderedDict()
        info_dict["reward"] = self.get_reward()

        info_dict["speed"] = Vehicle.get_speed(self.agent.vehicle)

        info_dict["throttle"] = self.agent.vehicle.control.throttle
        info_dict["steering"] = self.agent.vehicle.control.steering
        return info_dict

    def reset(self) -> Any:
        super(OccuDebug, self).reset()
        return self._get_obs()

    def _terminal(self) -> bool:
        is_done = super(OccuDebug, self)._terminal()
        if is_done:
            return True
        else:
            if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 1:
                return True
            else:
                return False