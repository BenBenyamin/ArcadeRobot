import gymnasium as gym
from stable_baselines3 import PPO , DQN ,A2C, SAC
from sb3_contrib import RecurrentPPO  ,QRDQN, TRPO,MaskablePPO

from inertia_warpper import  PongDelayInertiaWrapper , ActionSubsetWrapper,ActionPenalty,PongDelayStochasticInertiaWrapper ,RewardMultiplier
from inertia_warpper import make_wrapper_chain

from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.utils import LinearSchedule

from stable_baselines3.common.env_util import make_atari_env , make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import yaml
from utils import ALE_ACTION_MAP ,REV_ACTION_MAP
import os
from gymnasium.wrappers import FrameStackObservation

import ale_py
gym.register_envs(ale_py)


POLICY =  PPO


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



try:
  config.pop("env_wrapper")
  config.pop("n_timesteps")
except: pass


ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(500e6)
N_ENVS = config.pop("n_envs", 16)
FRAME_STACK =   config.pop("frame_stack",4)
PENALTY = 0.02
REW_MUL = 0.0
CHECKPOINT_NAME = os.path.expanduser(f"~/EXT_DATA/Pong-checkpoints/QRDQN_stochastic_466000000_steps")

MODEL_NAME = f"{POLICY.__name__}_S{FRAME_STACK}_P+{PENALTY}_M{REW_MUL}"


if __name__ == "__main__":

  pid = os.getpid()
  print("Current PID:", pid)
  print(f"RL ZOO, {MODEL_NAME}")

  lat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"durations.pkl")

  wrap = make_wrapper_chain([
    (PongDelayStochasticInertiaWrapper,{"target_fps":30,"lat_pickle_filename":lat_file}),
    (AtariWrapper, {}),
    (ActionSubsetWrapper, {"allowed_actions": 
                          [k for k, v in ALE_ACTION_MAP.items() if v in {"NO-OP", "RIGHT", "LEFT"}]}),
    (ActionPenalty , {"penalized_actions" : [0], "penalty" :  -PENALTY}),
  ])

  env = make_vec_env(ENV_NAME,N_ENVS,wrapper_class=wrap,vec_env_cls=SubprocVecEnv)
  
  print(env.reset()[0].shape)

  # model = POLICY(
  #               env=env, 
  #               verbose=1 , 
  #               device="cuda:0", # cuda:0 , cuda:1
  #               tensorboard_log="./PONG_tensorboard/", 
  #               **config,
  #               )
  model = POLICY.load(
     CHECKPOINT_NAME,
     device= "cpu",
     env = env,
  )




  checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./EXT_DATA/Pong-checkpoints/",
    name_prefix=MODEL_NAME,
    save_replay_buffer=False,
    save_vecnormalize=True,
  )

  model.learn(
              TOTAL_TIMESTEPS, 
              tb_log_name= "QRDQN_stochastic",
              callback= checkpoint_callback,
              reset_num_timesteps=False,
              )


  env.close()