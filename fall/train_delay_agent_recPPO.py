import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3 import PPO , DQN
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecMonitor , VecTransposeImage

from wrapper import  VectorizedPongDelayInertiaWrapper , VectorizedActionLoggerWrapper , ActionPenaltyVecWrapper
from callback import VideoRecorderCallback , OneTimeAttachLoggerCallback

import ale_py
gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = int(100e6)
VANILLA_MODEL_PATH = f"ppo_pong_vanilla"
BATCH_SIZE = 64
DELAY = 36
MODEL_PATH = f"RPPO_delay{DELAY}"
STACK_SIZE = 0
N_ENVS = 32

REW = 1.0

env = make_atari_env(ENV_NAME,n_envs=N_ENVS,wrapper_kwargs={"frame_skip": 0},)
env = VectorizedPongDelayInertiaWrapper(env,delay_steps=DELAY)
env = ActionPenaltyVecWrapper(env,[0,1],-REW)
# env = VecMonitor(env, info_keywords=('original_reward',))
env = VectorizedActionLoggerWrapper(env)
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
  save_path="./EXT_DATA/Pong-checkpoints/",
  name_prefix=f"RPPO_Delay={DELAY}_REW={REW}",
  save_replay_buffer=True,
  save_vecnormalize=True,
)


callback = CallbackList([video_callback,checkpoint_callback, OneTimeAttachLoggerCallback(verbose=1)])

# model = RecurrentPPO("CnnLstmPolicy", env, verbose=1 , device="cuda",tensorboard_log="./PONG_tensorboard/", batch_size=BATCH_SIZE)



model = RecurrentPPO.load("./Pong-checkpoints/RPPO_Delay=30_BEST.zip", device="cuda", env=env)


model.learn(
            TOTAL_TIMESTEPS, 
            callback= callback , 
            tb_log_name= f"RPPO_Delay={DELAY}_REW={REW}" , 
            )

model.save(MODEL_PATH)

env.close()