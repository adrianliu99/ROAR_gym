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
import time

class DepthE2EMLPNewRewardEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)  #  long (y) throttle, lat (x) steering,
        self.observation_space = gym.spaces.Box(low=np.zeros([84 * 84 + 2]), 
                                                high=np.ones([84 * 84 + 2]) * 255, 
                                                dtype=np.float32)
        self.curr_debug_info: OrderedDict = OrderedDict()

        self.track_points = np.array(read_txt(Path("../ROAR_Sim/data/easy_map_waypoints.txt")))
        self.trajectories = 0
        self._prev_locations = []
        self._prev_location = None

        self.action_repeat = params.get("action_repeat", 1)

    def reset(self):
        self.best_track_point_idx = 97
        self.steps = 0
        self.trajectories += 1
        self.cum_reward = 0
        self.prev_obs = None
        self.prev_action = None
        self.prev_steering = 0
        return super().reset()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        action[0] = action[0] * 0.4 + 0.6
        control = VehicleControl(throttle=action[0], steering=action[1])
        self.agent.kwargs["control"] = control
        self.curr_debug_info["control"] = control
        self.steps += 1

        self.throttle = action[0]
        self.steering = action[1]

        reward_ = 0
        for _ in range(self.action_repeat):
            self.agent.kwargs["control"] = control
            self.curr_debug_info["control"] = control
            obs, reward, done, info =  super().step(action=action)
            reward_ += reward
            self.cum_reward += reward
            if done:
                break
        self.render()
        if done:
            print(self.best_track_point_idx)
            print(self.cum_reward)
        self.prev_steering = self.steering
        return obs, reward_, done, info

    def _get_info(self) -> dict:
        return self.curr_debug_info

    def render(self, mode='ego'):
        super().render(mode)
        obs = self._get_obs()[2:].reshape((84, 84))
        cv2.imshow("depth_resized", obs)
        cv2.waitKey(1)
    
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

        if self.steps < (25 / self.action_repeat):
            return 0

        reward: float = -0.1

        # if abs(self.steering - self.prev_steering) > 0.2:
        #     reward -= 0.7

        start_idx = max(0, self.best_track_point_idx - 30)
        end_idx = min(self.best_track_point_idx + 50, len(self.track_points))
        track_points_to_consider = self.track_points[start_idx: end_idx]
        curr_location = self.agent.vehicle.transform.location.to_array()
        distances = np.sum((track_points_to_consider - curr_location) ** 2, axis=1)
        best_idx = min(range(len(distances)), key=lambda x: distances[x])
        best_idx += start_idx
        
        reward += best_idx - self.best_track_point_idx if self.best_track_point_idx < best_idx else 0

        if self._terminal():
            reward -= 100

        # if best_idx > self.best_track_point_idx:
        #     closest_location = self.track_points[best_idx]
        #     print("closest advanced")
        #     print(f"location : {self.agent.vehicle.transform.location.to_string()}")
        #     print(f"new best : {closest_location[0]}, {closest_location[1]}, {closest_location[2]}")
        
        self.best_track_point_idx = max(self.best_track_point_idx, best_idx)
        self.reward = reward
        # print(f"Steps {self.steps} \n Trajectoreis {self.trajectories}\n cum_reward {self.cum_reward}")
        return reward

    def _get_obs(self) -> Any:
        cam_param = (self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x, 1)
        obs = np.ones(shape=cam_param)
        obs = obs * 255
        if self.agent.front_depth_camera is not None and self.agent.front_depth_camera.data is not None:
            obs[:, :, 0] = self.agent.front_depth_camera.data[:,:]
        obs_ = cv2.resize(obs, (84, 84))
        obs = np.ones(84*84 + 2)
        obs[2:] = obs_.flatten()[:]
        obs[0] = Vehicle.get_speed(self.agent.vehicle)
        obs[1] = self.agent.vehicle.control.steering
        return obs

    def _terminal(self) -> bool:
        if self.agent.time_counter > 200 and Vehicle.get_speed(self.agent.vehicle) < 5:
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
        if self.steps > 200 :
            referce_location = self._prev_locations.pop()
            dist = curr_location.distance(referce_location)
            if dist < 0.01:
                return True
        return False