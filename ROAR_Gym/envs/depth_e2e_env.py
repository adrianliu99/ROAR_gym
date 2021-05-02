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


class DepthE2EEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)  #  long (y) throttle, lat (x) steering,
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(600, 800, 1), dtype=np.uint8)
        self.curr_debug_info: OrderedDict = OrderedDict()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        control = VehicleControl(throttle=action[0], steering=action[1])
        self.agent.kwargs["control"] = control
        self.curr_debug_info["control"] = control
        return super(DepthE2EEnv, self).step(action=action)

    def _get_info(self) -> dict:
        return self.curr_debug_info

    def get_reward(self) -> float:
        return 1.0

    def _get_obs(self) -> Any:
        cam_param = (self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x, 1)
        obs = np.ones(shape=cam_param)
        obs = obs * 255
        if self.agent.front_depth_camera is not None and self.agent.front_depth_camera.data is not None:
            obs[:, :, 0] = (self.agent.front_depth_camera.data * 255).astype(np.uint8)
        return obs

    def _terminal(self) -> bool:
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 10:
            return True
        elif self.carla_runner.get_num_collision() > 2:
            return True
        else:
            return False
