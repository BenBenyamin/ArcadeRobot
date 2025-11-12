import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from collections import deque
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat
import matplotlib.pyplot as plt
import io
from PIL import Image
from utils import ALE_ACTION_MAP

from utils import alpha_blit_rgb , load_icon_and_resize , convert_obs_to_grayscale
from utils import UP, DOWN, NO_OP
from utils import ALE_EFFECTIVE_ACTION_MAP
from gymnasium import spaces
import pickle


class PongDelayStochasticInertiaWrapper(gym.Wrapper):
    """
    Adds inertia behavior to discrete actions (NO_OP, UP, DOWN) by inserting
    internal steps with previous/NO_OP actions before executing the requested
    action.
    """

    def __init__(self, env, delay_steps: int = 10,lat_pickle_filename:str = "",target_fps:int = 30):
        super().__init__(env)
        self._delay_steps = int(delay_steps)
        self.prev_action = NO_OP

        with open(lat_pickle_filename, "rb") as f:
            lat = pickle.load(f)
        
        self.target_fps = target_fps
        
        self.latencies = (target_fps*np.array(lat)/1000).astype(int)

        self.sigma = np.std(self.latencies, ddof=1)


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Gymnasium API
        self.prev_action = NO_OP
        return obs, info

    @property
    def delay_steps(self):

        lat = np.random.choice(self.latencies) + np.random.normal(0, self.sigma)
        lat = np.maximum(lat, np.min(self.latencies))
        return round(lat)


    def _run_steps(self, action: int, n_steps: int, total_reward: float):
        """
        Advance n_steps internally, accumulating rewards, and break if episode ends.
        Returns: obs, total_reward, terminated, truncated, info, done_flag
        """
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info, True
        return obs, total_reward, terminated, truncated, info, False

    def step(self, action: int):
        total_reward = 0.0
        if self.prev_action != action:
            if action == NO_OP:
                # Coasting inertia: half delay on previous action
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(self.prev_action, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info
            else:
                # First half: previous action
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(self.prev_action, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info
                # Second half: NO_OP
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(NO_OP, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info

        # Finally, execute desired action
        self.prev_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        total = float(reward) + total_reward
        return obs, total, terminated, truncated, info

class PongDelayInertiaWrapper(gym.Wrapper):
    """
    Adds inertia behavior to discrete actions (NO_OP, UP, DOWN) by inserting
    internal steps with previous/NO_OP actions before executing the requested
    action.
    """

    def __init__(self, env, delay_steps: int = 10):
        super().__init__(env)
        self.delay_steps = int(delay_steps)
        self.prev_action = NO_OP

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Gymnasium API
        self.prev_action = NO_OP
        return obs, info

    def _run_steps(self, action: int, n_steps: int, total_reward: float):
        """
        Advance n_steps internally, accumulating rewards, and break if episode ends.
        Returns: obs, total_reward, terminated, truncated, info, done_flag
        """
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info, True
        return obs, total_reward, terminated, truncated, info, False

    def step(self, action: int):
        total_reward = 0.0
        if self.prev_action != action:
            if action == NO_OP:
                # Coasting inertia: half delay on previous action
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(self.prev_action, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info
            else:
                # First half: previous action
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(self.prev_action, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info
                # Second half: NO_OP
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(NO_OP, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info

        # Finally, execute desired action
        self.prev_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        total = float(reward) + total_reward
        return obs, total, terminated, truncated, info


class ActionSubsetWrapper(gym.ActionWrapper):
    """
    Restrict the environment's discrete actions to a chosen subset.
    Example:
        env = ActionSubsetWrapper(env, allowed_actions=[0, 2, 3])
    """
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        # Replace original action space with smaller one
        self.action_space = spaces.Discrete(len(allowed_actions))

    def action(self, act):
        # Map reduced index to actual environment action
        return self.allowed_actions[act]

class ActionPenalty(gym.Wrapper):

    def __init__(self, env , penalized_actions:list, penalty:int = 0.2):
        
        super().__init__(env)
        self.penalty = penalty
        self.penalized_actions = penalized_actions
    
    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        if action in self.penalized_actions:

            reward -= self.penalty
        
        return obs, reward, terminated, truncated, info


def make_wrapper_chain(wrappers):
    class WrapperChain(gym.Wrapper):
        def __init__(self, env):
            for wrapper, kwargs in wrappers:
                env = wrapper(env, **kwargs)
            super().__init__(env)
    return WrapperChain