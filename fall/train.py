
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import gymnasium as gym
from stable_baselines3 import PPO , DQN ,A2C
from sb3_contrib import RecurrentPPO  ,QRDQN, TRPO,MaskablePPO

from inertia_warpper import  PongDelayInertiaWrapper , ActionSubsetWrapper,ActionPenalty
from inertia_warpper import make_wrapper_chain

from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.env_util import make_atari_env , make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from utils import ALE_ACTION_MAP ,REV_ACTION_MAP

import ale_py
gym.register_envs(ale_py)


POLICY = DQN

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(100e6)
BATCH_SIZE = 64
DELAY = 18
REW = 0.0
MODEL_PATH = f"{POLICY.__name__}_delay{DELAY}_reward={REW}"
N_ENVS = 8

if __name__ == "__main__":

  wrap = make_wrapper_chain([
    (PongDelayInertiaWrapper,{"delay_steps": DELAY}),
    (AtariWrapper, {}),
    (ActionSubsetWrapper, {"allowed_actions": 
                           [k for k, v in ALE_ACTION_MAP.items() if v in {"NO-OP", "RIGHT", "LEFT"}]}),
    # (ActionPenalty,{"penalized_actions": [REV_ACTION_MAP["NO-OP"]],"penalty":REW}),
  ])

  env = make_vec_env(ENV_NAME,N_ENVS,wrapper_class=wrap)
  
  obs = env.reset()

  model = POLICY("CnnPolicy", 
                 env, 
                 verbose=1 , 
                 device="cuda",
                 tensorboard_log="./PONG_tensorboard/", 
                 batch_size=BATCH_SIZE,
                 )


  checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./EXT_DATA/Pong-checkpoints/",
    name_prefix=f"{POLICY.__name__}_Delay={DELAY}_REW={REW}",
    save_replay_buffer=False,
    save_vecnormalize=True,
  )

  model.learn(
              TOTAL_TIMESTEPS, 
              tb_log_name= f"{POLICY.__name__}_Delay={DELAY}_REW={REW}" ,
              callback= checkpoint_callback,
              )

  model.save(MODEL_PATH)

  env.close()