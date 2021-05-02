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


class ROARPIDEnv(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        # action_space = target_speed, long_k, lat_k
        self.action_space = gym.spaces.Box(low=np.array([10, 0, 0, 0, 0, 0, 0]),
                                           high=np.array([200, 1, 1, 1, 1, 1, 1]), dtype=np.float64)
        # observation_space = curr_speed, curr_throttle, curr_steering
        self.observation_space = gym.spaces.Box(low=np.array([-200, -1, -1]),
                                                high=np.array([200, 1, 1]),
                                                dtype=np.float64)
        self._prev_speed = 0
        self._prev_waypoint = None

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, dict]:
        """

        Args:
            action: array of [target_speed,
                              long_kp, long_kd, long_ki,
                              lat_kp, lat_kd, lat_ki]

        Returns:

        """
        assert type(action) == list or type(action) == np.ndarray, f"Action is of type {type(action)}"
        assert len(action) == 7, f"Action of shape {np.shape(action)} is not correct"
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        if len(self.agent.local_planner.way_points_queue) > 0:
            self._prev_waypoint = self.agent.local_planner.way_points_queue[0].to_array()
        target_speed = action[0]
        long_k_p, long_k_d, long_k_i = action[1], action[2], action[3]
        lat_k_p, lat_k_d, lat_k_i = action[4], action[5], action[6]
        self.agent.kwargs["long_k_p"] = long_k_p
        self.agent.kwargs["long_k_d"] = long_k_d
        self.agent.kwargs["long_k_i"] = long_k_i
        self.agent.kwargs["target_speed"] = target_speed
        self.agent.kwargs["lat_k_p"] = lat_k_p
        self.agent.kwargs["lat_k_d"] = lat_k_d
        self.agent.kwargs["lat_k_i"] = lat_k_i

        agent, reward, is_done, other_info = super(ROARPIDEnv, self).step(action=action)
        print("is_done: ", str(is_done))
        obs = self._get_obs()
        return obs, reward, is_done, other_info

    def _get_obs(self) -> Any:
        curr_speed = np.array([Vehicle.get_speed(self.agent.vehicle),
                               self.agent.vehicle.control.throttle,
                               self.agent.vehicle.control.steering])
        return curr_speed

    def get_reward(self) -> float:
        reward: float = 0.0
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            reward -= 1000000

        # if the agent is able to complete a lap, reward heavy
        if self.agent.is_done:
            reward += 100

        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 1:
            # so that the agent is encouraged to move
            reward -= 100000
        try:
            if len(self.agent.local_planner.way_points_queue) > 0 and \
                    self._prev_waypoint != self.agent.local_planner.way_points_queue[0].to_array():
                # so that the agent is encouraged to go to the next waypoint
                reward += 100
        except Exception as e:
            pass

        # if the agent tries to turn too much, penalize
        if abs(self.agent.vehicle.control.steering) > 0.3:
            # prevent it from over steering
            reward -= 10
        if current_speed < 10:
            # again, agent is encouraged to move
            reward = 0
        else:
            reward += current_speed

        if current_speed < 80:
            # i know that you can drive around the track at least with this speed
            reward -= 100

        if current_speed > self._prev_speed:
            # agent is incentivized to go faster
            reward += 1
        return reward

    def _get_info(self) -> dict:
        info_dict = OrderedDict()
        info_dict["reward"] = self.get_reward()

        info_dict["speed"] = Vehicle.get_speed(self.agent.vehicle)
        info_dict["target_speed"] = self.agent.kwargs["target_speed"]

        info_dict["throttle"] = self.agent.vehicle.control.throttle
        info_dict["steering"] = self.agent.vehicle.control.steering

        info_dict["long_kp"] = self.agent.kwargs["long_k_p"]
        info_dict["long_kd"] = self.agent.kwargs["long_k_d"]
        info_dict["long_ki"] = self.agent.kwargs["long_k_i"]
        info_dict["lat_kp"] = self.agent.kwargs["lat_k_p"]
        info_dict["lat_kd"] = self.agent.kwargs["lat_k_d"]
        info_dict["lat_ki"] = self.agent.kwargs["lat_k_i"]
        return info_dict

    def reset(self) -> Any:
        super(ROARPIDEnv, self).reset()
        return self._get_obs()

    def _terminal(self) -> bool:
        is_done = super(ROARPIDEnv, self)._terminal()
        if is_done:
            return True
        else:
            if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 1:
                return True
            else:
                return False
