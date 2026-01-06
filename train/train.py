
import gymnasium as gym
from stable_baselines3 import PPO , DQN ,A2C, SAC
from sb3_contrib import RecurrentPPO  ,QRDQN, TRPO,MaskablePPO

from inertia_warpper import  ActionSubsetWrapper,PongDelayStochasticInertiaWrapper
from inertia_warpper import make_wrapper_chain

from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.env_util import  make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import ALE_ACTION_MAP
import os

import ale_py
gym.register_envs(ale_py)


POLICY = PPO

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(500e6)
BATCH_SIZE = 64
DELAY = 18
REW = 0.0
MODEL_PATH = f"{POLICY.__name__}_delay{DELAY}_reward={REW}"
N_ENVS = 8
SEED = 5

if __name__ == "__main__":

  lat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"latencies.pkl")

  wrap = make_wrapper_chain([
    (PongDelayStochasticInertiaWrapper,{"delay_steps": DELAY,"target_fps":30,"lat_pickle_filename":lat_file}),
    (AtariWrapper, {}),
    (ActionSubsetWrapper, {"allowed_actions": 
                          [k for k, v in ALE_ACTION_MAP.items() if v in {"NO-OP", "RIGHT", "LEFT"}]}),
  ])

  env = make_vec_env(ENV_NAME,N_ENVS,wrapper_class=wrap,vec_env_cls=SubprocVecEnv, env_kwargs= {"seed" : SEED})
  
  obs = env.reset()

  model = POLICY("CnnPolicy", 
                env, 
                verbose=1 , 
                device="cuda", # cuda:0 , cuda:1
                tensorboard_log="./PONG_tensorboard/", 
                batch_size=BATCH_SIZE,
                )


  checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./EXT_DATA/Pong-checkpoints/",
    name_prefix=f"{POLICY.__name__}_stochastic_{SEED}",
    save_replay_buffer=False,
    save_vecnormalize=True,
  )

  model.learn(
              TOTAL_TIMESTEPS, 
              tb_log_name= f"{POLICY.__name__}_stochastic_{SEED}" ,
              callback= checkpoint_callback,
              )

  model.save(MODEL_PATH)

  env.close()