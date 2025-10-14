import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3 import PPO , DQN
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack , VecTransposeImage

from wrapper import  VectorizedPongDelayInertiaWrapper , VectorizedActionLoggerWrapper
from callback import VideoRecorderCallback , OneTimeAttachLoggerCallback

import ale_py
gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(30e6)
VANILLA_MODEL_PATH = f"ppo_pong_vanilla"
BATCH_SIZE = 64
DELAY = 20
MODEL_PATH = f"RPPO_delay{DELAY}"
STACK_SIZE = 0
N_ENVS = 8

env = make_atari_env(ENV_NAME,n_envs=N_ENVS)
env = VectorizedPongDelayInertiaWrapper(env,delay_steps=DELAY)
env = VectorizedActionLoggerWrapper(env)
# env = VecFrameStack(env, n_stack=STACK_SIZE)
env = VecTransposeImage(env)

video_callback = VideoRecorderCallback(
  record_freq=1e5//N_ENVS,
  video_path="./pong_videos",
  verbose=0,
  delay=DELAY,
  stack_size=STACK_SIZE,
)

checkpoint_callback = CheckpointCallback(
  save_freq=10_000,
  save_path="./Pong-checkpoints/",
  name_prefix=f"RPPO_Delay={DELAY}_Stack={STACK_SIZE}",
  save_replay_buffer=True,
  save_vecnormalize=True,
)


callback = CallbackList([checkpoint_callback, OneTimeAttachLoggerCallback(verbose=1)])

model = RecurrentPPO("CnnLstmPolicy", env, verbose=1 , device="cuda",tensorboard_log="./PONG_tensorboard/", batch_size=BATCH_SIZE)


model.learn(
            TOTAL_TIMESTEPS, 
            callback= callback , 
            tb_log_name= f"RPPO_Delay={DELAY}_Stack={STACK_SIZE}" , 
            )

model.save(MODEL_PATH)

env.close()