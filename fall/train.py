
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import gymnasium as gym
from stable_baselines3 import PPO , DQN
from sb3_contrib import RecurrentPPO

from inertia_warpper import  PongDelayInertiaWrapper , make_wrapper_chain , ActionSubsetWrapper
# from make_env import make_pong_env
from functools import partial

from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_atari_env , make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from utils import ALE_ACTION_MAP

import ale_py
gym.register_envs(ale_py)


POLICY = RecurrentPPO

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(100e6)
BATCH_SIZE = 64
DELAY = 0
REW = 0.0
MODEL_PATH = f"{POLICY.__name__}_delay{DELAY}_reward={REW}"
N_ENVS = 8

def make_pong_env():

  env = gym.make(ENV_NAME)
  env = AtariWrapper(env)
  env = PongDelayInertiaWrapper(env,delay_steps=DELAY)

  return env

if __name__ == "__main__":

  wrap = make_wrapper_chain([
    (AtariWrapper, {}),
    (PongDelayInertiaWrapper,{"delay_steps": DELAY}),
    (ActionSubsetWrapper, {"allowed_actions": 
                           [k for k, v in ALE_ACTION_MAP.items() if v in {"NO-OP", "RIGHT", "LEFT"}]}),
  ])

  env = make_vec_env(ENV_NAME,N_ENVS,wrapper_class=wrap)
  
  obs = env.reset()

  model = POLICY("CnnLstmPolicy", 
                 env, 
                 verbose=1 , 
                 device="cuda",
                 tensorboard_log="./PONG_tensorboard/", 
                 batch_size=BATCH_SIZE,
                 )


  # checkpoint_callback = CheckpointCallback(
  #   save_freq=10_000,
  #   save_path="./EXT_DATA/Pong-checkpoints/",
  #   name_prefix=f"{POLICY.__name__}_Delay={DELAY}_REW={REW}",
  #   save_replay_buffer=True,
  #   save_vecnormalize=True,
  # )

  model.learn(
              TOTAL_TIMESTEPS, 
              tb_log_name= f"{POLICY.__name__}_Delay={DELAY}_REW={REW}" , 
              )

  model.save(MODEL_PATH)

  env.close()