try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Dict, Any, Tuple
import gym
import numpy as np
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import OrderedDict
from pathlib import Path
from ROAR.utilities_module.track_visualizer import read_txt
import cv2

class DepthE2ENewRewardEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)  #  long (y) throttle, lat (x) steering,
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(84, 84, 1), dtype=np.uint8)
        self.curr_debug_info: OrderedDict = OrderedDict()

        self.track_points = np.array(read_txt(Path("../ROAR_Sim/data/easy_map_waypoints.txt")))
        self.trajectories = 0
        self._prev_locations = []
        self._prev_location = None

    def reset(self):
        self.best_track_point_idx = 97
        self.steps = 0
        self.trajectories += 1
        self.cum_reward = 0
        self.prev_obs = None
        self.prev_action = None
        return super().reset()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        control = VehicleControl(throttle=action[0], steering=action[1])
        self.agent.kwargs["control"] = control
        self.curr_debug_info["control"] = control
        self.steps += 1
        rtn =  super(DepthE2ENewRewardEnv, self).step(action=action)

        obs = rtn[1]
        diff = np.sum((obs - self.prev_obs) ** 2)
        print(f"difference between obs {diff}")
        if self.prev_action is not None:
            action_diff = np.sum((self.prev_action - action) ** 2)
            print(f"difference between actions {action_diff}")
        self.prev_obs = obs
        self.prev_action = np.array(action)
        self.render()
        return rtn

    def _get_info(self) -> dict:
        return self.curr_debug_info

    
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

        if self.steps < 25:
            return 0

        reward: float = -1.0


        start_idx = max(0, self.best_track_point_idx - 30)
        end_idx = min(self.best_track_point_idx + 30, len(self.track_points))
        track_points_to_consider = self.track_points[start_idx: end_idx]
        curr_location = self.agent.vehicle.transform.location.to_array()
        distances = np.sum((track_points_to_consider - curr_location) ** 2, axis=1)
        best_idx = min(range(len(distances)), key=lambda x: distances[x])
        best_idx += start_idx
        
        reward += best_idx - self.best_track_point_idx if self.best_track_point_idx < best_idx else 0

        print(f"best: {best_idx} abs_best: {self.best_track_point_idx}")

        if self._terminal():
            reward -= 100

        if best_idx > self.best_track_point_idx:
            closest_location = self.track_points[best_idx]
            print("closest advanced")
            print(f"location : {self.agent.vehicle.transform.location.to_string()}")
            print(f"new best : {closest_location[0]}, {closest_location[1]}, {closest_location[2]}")
        
        self.best_track_point_idx = max(self.best_track_point_idx, best_idx)
        self.reward = reward
        self.cum_reward += reward
        print(f"Steps {self.steps} \n Trajectoreis {self.trajectories}\n cum_reward {self.cum_reward}")
        return reward

    def _get_obs(self) -> Any:
        cam_param = (self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x, 1)
        obs = np.ones(shape=cam_param)
        obs = obs * 255
        if self.agent.front_depth_camera is not None and self.agent.front_depth_camera.data is not None:
            obs[:, :, 0] = (self.agent.front_depth_camera.data * 255).astype(np.uint8)
        obs_ = cv2.resize(obs, (84, 84))
        obs = np.ones(shape=(84,84,1))
        obs[:,:,0] = obs_[:,:]
        self.prev_obs = obs
        return obs

    def _terminal(self) -> bool:
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 10:
            return True
        elif self.carla_runner.get_num_collision() > 2:
            return True
        elif self.stuck():
            return True
        else:
            return False

    def stuck(self):
        curr_location = self.agent.vehicle.transform.location
        self._prev_locations.append(curr_location)
        if self.steps > 100 :
            referce_location = self._prev_locations.pop()
            dist = curr_location.distance(referce_location)
            if dist < 0.01:
                return True
        return False