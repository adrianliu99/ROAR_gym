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
from ROAR.utilities_module.track_visualizer import read_txt


class LocalPlannerNewRewardEnv(ROAREnv):
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

        self.track_points = np.array(read_txt(Path("../ROAR_Sim/data/easy_map_waypoints.txt")))

    def reset(self):
        rtn = super().reset()
        self._prev_locations = []
        self.best_track_point_idx = 0
        self.steps = 0
        return rtn


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
        obs, reward, is_done, other_info = super().step(action=action)

        self.steps += 1

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
        reward: float = -1.0


        end_idx = np.min([self.best_track_point_idx + 100, len(self.track_points)])
        track_points_to_consider = self.track_points[self.best_track_point_idx: end_idx]
        curr_location = self.agent.vehicle.transform.location.to_array()
        distances = np.sum((track_points_to_consider - curr_location) ** 2, axis=1)
        best = min(range(len(distances)), key=lambda x: distances[x])
        
        reward += best * 10

        if best > 0:
            closest_location = self.track_points[self.best_track_point_idx + best]
            print("closest advanced")
            print(self.steps)
            print(f"location : {self.agent.vehicle.transform.location.to_string()}")
            print(f"new best : {closest_location[0]}, {closest_location[1]}, {closest_location[2]}")
        
        self.best_track_point_idx += best

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
        # elif self.stuck():
        #     return True
        else:
            return False


    # Checks if the car is stuck on a wall, used to determin if we should reset
    def stuck(self):
        curr_location = self.agent.vehicle.transform.location
        self._prev_locations.append(curr_location)
        if self.steps > 100 :
            print("checking stuck")
            print(len(self._prev_locations))
            referce_location = self._prev_locations.pop()
            dist = curr_location.distance(referce_location)
            print(dist)
            if dist < 10:
                return True
        return False


