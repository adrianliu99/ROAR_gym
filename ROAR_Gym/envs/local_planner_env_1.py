from typing import Any, Tuple, Dict
from collections import OrderedDict

try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv
import gym
import numpy as np
from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation
from pathlib import Path
import os
from ROAR.utilities_module.vehicle_models import Vehicle
import cv2
import time


class LocalPlannerEnv1(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.obs_size = 200
        self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                           high=np.array([self.obs_size, self.obs_size]),
                                           dtype=np.int64)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size+1, 2), dtype=np.int64)

        self._prev_speed = 0
        self._prev_location = None
        self.correct_next_waypoint_world = None
        self.correct_next_waypoint_occu = None
        self.my_guess_next_waypoint_world = None
        self.my_guess_next_waypoint_occu = None
        self.reward = 0
        self.action = None

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        self._prev_location = self.agent.vehicle.transform.location

        action = np.array(action).astype(np.int64)
        self.action = action
        if len(self.agent.traditional_local_planner.way_points_queue) > 0:
            self.correct_next_waypoint_world = self.agent.traditional_local_planner.way_points_queue[0]
            self.my_guess_next_waypoint_occu = action
            self.my_guess_next_waypoint_world = self.agent.occupancy_map.cropped_occu_to_world(
                cropped_occu_coord=self.my_guess_next_waypoint_occu,
                vehicle_transform=self.agent.vehicle.transform,
                occu_vehicle_center=np.array([self.obs_size // 2, self.obs_size // 2]))

        self.agent.kwargs["next_waypoint"] = self.my_guess_next_waypoint_world
        obs, reward, is_done, other_info = super(LocalPlannerEnv1, self).step(action=action)
        return obs, reward, is_done, other_info

    def get_reward(self) -> float:
        """
        Reward policy:
            Surviving = +1
            Going forward (positive velocity) = +1
            Going toward a waypoint = +10
            Speed < 10 = -10
            Speed > 80 = +50
            Collision = -10000
            abs(steering) > 0.5 = -100
        Returns:
            reward according to the aforementioned policy
        """
        reward: float = 1.0

        if Vehicle.get_speed(self.agent.vehicle) > 10:
            reward += 1
        if len(self.agent.traditional_local_planner.way_points_queue) > 0 and self._prev_location is not None:
            next_waypoint: Transform = self.agent.traditional_local_planner.way_points_queue[0]
            curr_location: Location = self.agent.vehicle.transform.location
            if curr_location.distance(next_waypoint.location) < curr_location.distance(self._prev_location):
                print("Getting closer to next waypoint!!!")
                reward += 1000
        if Vehicle.get_speed(self.agent.vehicle) < 10:
            reward -= 10
        if Vehicle.get_speed(self.agent.vehicle) > 80:
            reward += 50
        if self.carla_runner.get_num_collision() > 0:
            reward -= 10000
        if abs(self.agent.vehicle.control.steering) > 0.5:
            reward -= 100

        self.reward = reward
        return reward

    def _get_info(self) -> dict:
        info = OrderedDict()
        info['speed'] = Vehicle.get_speed(self.agent.vehicle)
        info['reward'] = self.reward
        info['action'] = self.action
        info["obs_size"] = self.obs_size
        info["num_collision"] = self.carla_runner.get_num_collision()
        info["correct_next_waypoint_world"] = self.correct_next_waypoint_world.location.to_array()
        info["my_guess_next_waypoint_world"] = self.my_guess_next_waypoint_world.location.to_array()
        info["my_guess_next_waypoint_occu"] = self.my_guess_next_waypoint_occu

        return info

    def _get_obs(self) -> Any:
        obs = np.zeros(shape=(self.obs_size+1, 2))
        obs[-1] = [self.obs_size // 2, self.obs_size // 2] # where the ego vehicle is gaurenteed be at
        if self.agent.occupancy_map is not None:
            occu_map: np.ndarray = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                                    view_size=(200, 200))
            obstacle_coords = np.array(list(zip(np.where(occu_map == 1)))).squeeze().T  # Nx2
            if len(obstacle_coords) < self.obs_size:
                obs[0:len(obstacle_coords)] = obstacle_coords
            else:
                sampled_indices = np.random.choice(len(obstacle_coords), self.obs_size)
                obs[0: self.obs_size] = obstacle_coords[sampled_indices]
            return obs
        else:
            return obs

    def _terminal(self) -> bool:
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 10:
            return True
        elif self.carla_runner.get_num_collision() > 2:
            return True
        else:
            return False
