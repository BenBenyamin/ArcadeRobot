import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py
import cv2
import os
gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 500_000
MODEL_PATH = "models/ppo_pong"

def train():
    env = make_atari_env(ENV_NAME, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    env.close()

def play():
    model = PPO.load(MODEL_PATH)
    env = make_atari_env(ENV_NAME, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    # vis_env = make_atari_env(ENV_NAME, n_envs=1, seed=0)
    obs = env.reset()
    env.render("human")
    done = False

    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        time.sleep(0.01)
            
    env.close()

def generate_dataset():
    model = PPO.load(MODEL_PATH)
    env = make_atari_env(ENV_NAME, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    # vis_env = make_atari_env(ENV_NAME, n_envs=1, seed=0)
    obs = env.reset()
    # env.render("human")
    done = False
    os.makedirs("pong_unbiased_dataset", exist_ok=True)

    for i in range(10_000//4):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        [cv2.imwrite(f"pong_unbiased_dataset/image_{i*obs.shape[-1]+j}.png", obs.squeeze()[:, :, j]) for j in range(obs.shape[-1])]
        # env.render("human")
        time.sleep(0.01)
            
    env.close()

generate_dataset()