import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack , DummyVecEnv
from stable_baselines3 import PPO , DQN
from callback import VideoRecorderCallback
from wrapper import IconOverlayVideoWrapper
from utils import make_pong_env

import ale_py
gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 80_000
MODEL_PATH = f"ppo_pong_vanilla"


env = make_atari_env(ENV_NAME,n_envs=8)
env = VecFrameStack(env,n_stack=4)

env = make_pong_env(
    video_wrapper=IconOverlayVideoWrapper,
    delay_wrapper=None,
    delay_steps=60,
    train=True,
    show_video=False,
    save_video=False,
    video_path="",
    video_fps=60,
    icon_xy=(130, 0),
    scale=6,
    waitkey_ms=16,
)()


video_callback = VideoRecorderCallback(record_freq=10*8*256,video_path="./pong_videos",verbose=0)

model = PPO("CnnPolicy", env, verbose=1 , device="cuda",tensorboard_log="./tensorboard/", batch_size=256)

model.learn(TOTAL_TIMESTEPS , callback= video_callback)

model.save(MODEL_PATH)

env.close()