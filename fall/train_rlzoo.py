import gymnasium as gym
from stable_baselines3 import PPO , DQN ,A2C, SAC
from sb3_contrib import RecurrentPPO  ,QRDQN, TRPO,MaskablePPO

from inertia_warpper import  PongDelayInertiaWrapper , ActionSubsetWrapper,ActionPenalty,PongDelayStochasticInertiaWrapper
from inertia_warpper import make_wrapper_chain

from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.utils import LinearSchedule

from stable_baselines3.common.env_util import make_atari_env , make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import yaml
from utils import ALE_ACTION_MAP ,REV_ACTION_MAP
import os

import ale_py
gym.register_envs(ale_py)


POLICY = PPO


yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"rlzoo_config",f"{POLICY.__name__.lower()}.yml")

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)["atari"]

for key, value in config.items():
    if isinstance(value, str) and value.startswith("lin_"):
        # extract numeric part after "lin_"
        start_value = float(value.split("_")[1])
        config[key] = LinearSchedule(start_value,end=start_value*0.1,end_fraction=0.2)
    elif isinstance(value, str) and value.strip().startswith("dict("):
      config[key] = eval(value)

  
pid = os.getpid()
print("Current PID:", pid)
print(f"RL ZOO, {POLICY.__name__}")

try:
  config.pop("env_wrapper")
  config.pop("frame_stack")
  config.pop("n_timesteps")
except: pass


ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(500e6)
BATCH_SIZE = 64
DELAY = 18
REW = 0.0
MODEL_PATH = f"{POLICY.__name__}_delay{DELAY}_reward={REW}"
N_ENVS = config.pop("n_envs", 8)

if __name__ == "__main__":

  lat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"latencies.pkl")

  wrap = make_wrapper_chain([
    (PongDelayStochasticInertiaWrapper,{"delay_steps": DELAY,"target_fps":30,"lat_pickle_filename":lat_file}),
    (AtariWrapper, {}),
    (ActionSubsetWrapper, {"allowed_actions": 
                          [k for k, v in ALE_ACTION_MAP.items() if v in {"NO-OP", "RIGHT", "LEFT"}]}),
  ])

  env = make_vec_env(ENV_NAME,N_ENVS,wrapper_class=wrap,vec_env_cls=SubprocVecEnv)
  
  obs = env.reset()

  model = POLICY(
                env=env, 
                verbose=1 , 
                device="cuda", # cuda:0 , cuda:1
                tensorboard_log="./PONG_tensorboard/", 
                **config,
                )


  checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./EXT_DATA/Pong-checkpoints/",
    name_prefix=f"{POLICY.__name__}_stochastic",
    save_replay_buffer=False,
    save_vecnormalize=True,
  )

  model.learn(
              TOTAL_TIMESTEPS, 
              tb_log_name= f"{POLICY.__name__}_stochastic" ,
              callback= checkpoint_callback,
              )

  model.save(MODEL_PATH)

  env.close()