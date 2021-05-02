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
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from typing import Optional
import cv2
import time
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.rl_occu_map_e2e_training_agent import RLOccuMapE2ETrainingAgent


class OccuMapE2EEnv(ROAREnv):
    def __init__(self):
        # Set gym-carla environment
        agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
        carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

        params = {
            "agent_config": agent_config,
            "carla_config": carla_config,
            "ego_agent_class": RLOccuMapE2ETrainingAgent,
            "max_collision": 5,
        }
        super().__init__(params)
        # action space = next waypoint
        self.view_size = 200
        self.max_steering_angle = 1
        self.action_space = gym.spaces.Box(low=np.array([0.4, -self.max_steering_angle]),
                                           high=np.array([1, self.max_steering_angle]),
                                           dtype=np.float32)  # throttle, steering
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(self.view_size, self.view_size, 1), dtype=np.uint8)
        self.debug_info: OrderedDict = OrderedDict()
        self.prev_location: Optional[Location] = None
        self.prev_next_waypoint: Optional[Location] = None
        self.dist_diff = 0

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        self.prev_location = self.agent.vehicle.transform.location
        control = VehicleControl(throttle=action[0], steering=action[1])
        self.agent.kwargs["control"] = control
        self.debug_info["control"] = control
        before_dist = self.agent.vehicle.transform.location.distance(self.agent.traditional_local_planner.way_points_queue[
            self.agent.traditional_local_planner.get_curr_waypoint_index()].location)
        result = super(OccuMapE2EEnv, self).step(action)
        after_dist = self.agent.vehicle.transform.location.distance(self.agent.traditional_local_planner.way_points_queue[
            self.agent.traditional_local_planner.get_curr_waypoint_index()].location)
        self.dist_diff = before_dist - after_dist
        return result

    def get_reward(self) -> float:
        """
        Reward policy:
            Surviving = +1
            Going toward a waypoint = +10
            Collision = -10000
            abs(steering) > 0.5 = -1000
            abs(steering) < 0.2 = +2000
        Returns:
            reward according to the aforementioned policy
        """
        reward: float = 1.0
        if self.dist_diff <= 0:
            reward += 100
            self.debug_info["isGettingCloserToNextwaypoint"] = True
        else:
            reward -= 10
            self.debug_info["isGettingCloserToNextwaypoint"] = False

        if Vehicle.get_speed(self.agent.vehicle) < 10:
            reward -= 10
        if self.carla_runner.get_num_collision() > 0:
            reward -= 10000
        if abs(self.agent.vehicle.control.steering) > 0.1:
            reward -= 1000

        if abs(self.agent.vehicle.control.steering) < 0.1:
            reward += 200

        self.debug_info['reward'] = reward
        return reward

    def _get_info(self) -> dict:
        self.debug_info["speed"] = Vehicle.get_speed(self.agent.vehicle)
        return self.debug_info

    def _get_obs(self) -> Any:
        obs = np.zeros(shape=(self.view_size, self.view_size, 1))
        if self.agent.occupancy_map is not None:
            arbitrary_points = [self.agent.traditional_local_planner.way_points_queue[0].location]
            occu_map = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                        view_size=(self.view_size, self.view_size),
                                                        vehicle_value=-10,
                                                        arbitrary_locations=arbitrary_points,
                                                        arbitrary_point_value=-5)
            # print(np.where(occu_map == -5))
            obs[:, :, 0] = occu_map
        return obs

    def _terminal(self) -> bool:
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 10:
            return True
        elif self.carla_runner.get_num_collision() > 0:
            return True
        else:
            return False
