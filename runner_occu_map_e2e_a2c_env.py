import warnings
import logging
from typing import Optional, Dict

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym

from stable_baselines.a2c import A2C
# from stable_baselines.common.policies import CnnPolicy
# from stable_baselines import DDPG
from stable_baselines.common.policies import CnnPolicy
from datetime import datetime
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList
from utilities import find_latest_model
from ROAR_Gym.envs.roar_env import LoggingCallback
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.cmd_util import make_vec_env
from pprint import pformat
from collections import OrderedDict


def main(output_folder_path: Path):
    env = make_vec_env('roar-occu-map-e2e-v0')
    env.reset()
    model_params: dict = {
        "verbose": 1,
        "env": env,
        "n_steps": 100
        # "render": True,
    }
    model, callbacks = setup(model_params, output_folder_path)
    model = model.learn(total_timesteps=1e6, callback=callbacks, reset_num_timesteps=False)
    # model = model.learn(total_timesteps=1000, callback=callbacks, reset_num_timesteps=False)


def setup(model_params, output_folder_path):
    latest_model_path = find_latest_model(Path(output_folder_path))
    if latest_model_path is None:
        print("Creating model...")
        model = A2C(CnnPolicy, **model_params)
    else:
        print("Loading model...")
        model = A2C.load(latest_model_path, **model_params)
    tensorboard_dir = (output_folder_path / "tensorboard")
    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=200, verbose=2, save_path=ckpt_dir.as_posix())
    # event_callback = EveryNTimesteps(n_steps=100, callback=checkpoint_callback)
    logging_callback = CustomCallback(model=model, verbose=1)
    callbacks = CallbackList([checkpoint_callback, logging_callback])
    return model, callbacks


class CustomCallback(BaseCallback):
    def __init__(self, model, verbose=0):
        super().__init__(verbose)
        self.init_callback(model=model)

    def _on_step(self) -> bool:
        m = OrderedDict()
        m["value_loss"] = self.locals.get("value_loss")
        m["policy_entropy"] = self.locals.get("policy_entropy")
        m["rewards"] = self.locals.get("rewards")
        m["n_seconds"] = self.locals.get("n_seconds")
        m["control"] = self.locals.get("info", dict()).get("control")
        m["speed"] = self.locals.get("info", dict()).get("speed")
        m["isGettingCloserToNextwaypoint"] = self.locals.get("info", dict()).get("isGettingCloserToNextwaypoint")
        m["reward"] = self.locals.get("info", dict()).get("reward")
        m["episode"] = self.locals.get("info", dict()).get("episode")
        m["fps"] = self.locals.get("fps")
        msg = f"{pformat(m)}\n"
        self.logger.log(msg)
        return True


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main(output_folder_path=Path(os.getcwd()) / "output" / "occu_map_e2e_a2c")
