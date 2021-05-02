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
from ROAR.agent_module.rl_local_planner_agent import RLLocalPlannerAgent
from stable_baselines.ddpg.policies import CnnPolicy
# from stable_baselines.common.policies import CnnPolicy
from stable_baselines import DDPG, PPO2
from datetime import datetime
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList
from utilities import find_latest_model

try:
    from ROAR_Gym.envs.roar_env import LoggingCallback
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import LoggingCallback


def main(output_folder_path: Path):
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLLocalPlannerAgent,
        "max_collision": 5,
    }

    env = gym.make('roar-local-planner-v0', params=params)
    env.reset()

    model_params: dict = {
        "verbose": 1,
        "render": True,
        "env": env,
        "n_cpu_tf_sess": None,
        "buffer_size": 1000,
        "nb_train_steps": 50,
        "nb_rollout_steps": 100,
        # "nb_eval_steps": 50,
        "batch_size": 32,
    }
    latest_model_path = find_latest_model(Path(output_folder_path))
    if latest_model_path is None:
        model = DDPG(CnnPolicy, **model_params)
    else:
        model = DDPG.load(latest_model_path, **model_params)
    tensorboard_dir = (output_folder_path / "tensorboard")
    ckpt_dir = (output_folder_path / "checkpoints")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.tensorboard_log = tensorboard_dir.as_posix()
    model.render = True
    logging_callback = LoggingCallback(model=model)
    checkpoint_callback = CheckpointCallback(save_freq=1000, verbose=2, save_path=ckpt_dir.as_posix())
    event_callback = EveryNTimesteps(n_steps=100, callback=checkpoint_callback)
    callbacks = CallbackList([checkpoint_callback, event_callback, logging_callback])
    model = model.learn(total_timesteps=int(1e10), callback=callbacks, reset_num_timesteps=False)
    model.save(f"local_planner_ddpg_{datetime.now()}")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main(output_folder_path=Path(os.getcwd()) / "output" / "local_planner")
