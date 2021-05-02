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
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.rl_occu_map_e2e_training_agent import RLOccuMapE2ETrainingAgent
from stable_baselines.ddpg.policies import CnnPolicy
# from stable_baselines.common.policies import CnnPolicy
from stable_baselines import DDPG
from datetime import datetime
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList
from utilities import find_latest_model
from ROAR_Gym.envs.roar_env import LoggingCallback


def main(output_folder_path: Path):
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLOccuMapE2ETrainingAgent,
        "max_collision": 5,
    }

    env = gym.make('roar-occu-map-e2e-v0', params=params)
    env.reset()
    model_params: dict = {
        "verbose": 1,
        "env": env,
        "render": True,
        "tensorboard_log": (output_folder_path / "tensorboard").as_posix(),
        "buffer_size": 10000,
        "nb_rollout_steps": 100,
        # "batch_size": 16,
        "nb_eval_steps": 50
    }
    model, callbacks = setup(model_params, output_folder_path)
    model = model.learn(total_timesteps=int(1e6), callback=callbacks, reset_num_timesteps=False)
    # model.save(f"occu_map_e2e_ddpg_{datetime.now()}")


def setup(model_params, output_folder_path):
    latest_model_path = find_latest_model(Path(output_folder_path))
    if latest_model_path is None:
        print("Creating model...")
        model = DDPG(CnnPolicy, **model_params)
    else:
        print("Loading model...")
        model = DDPG.load(latest_model_path, **model_params)
    tensorboard_dir = (output_folder_path / "tensorboard")
    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=200, verbose=2, save_path=ckpt_dir.as_posix())
    # event_callback = EveryNTimesteps(n_steps=100, callback=checkpoint_callback)
    logging_callback = LoggingCallback(model=model, verbose=1)
    callbacks = CallbackList([checkpoint_callback, logging_callback])
    return model, callbacks


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main(output_folder_path=Path(os.getcwd()) / "output" / "occu_map_e2e")
