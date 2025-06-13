import gymnasium as gym
from collections import deque

NO_OP = 0

class PongWrapper(gym.Wrapper):

    def __init__(self, env, buffer_stack_size=10):
        super().__init__(env)

        self.buffer_stack_size = buffer_stack_size

        self.action_buffer = deque(
            [NO_OP] * buffer_stack_size, 
            maxlen=buffer_stack_size
        )

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        self.action_buffer.clear()
        self.action_buffer.extend([NO_OP] * self.buffer_stack_size)

        return obs
    
    # def step(self, action):
    #     self.action_buffer.append(action)
    #     delayed_action = self.action_buffer.popleft()
    #     obs, reward, terminated, truncated, info = self.env.step(delayed_action)
    #     return obs, reward, terminated, truncated, info

    def step(self, action):
        total_reward = 0.0
        info = {}
        terminated = False
        truncated = False

        # Send (stack_size - 1) NOOP actions first
        for _ in range(self.buffer_stack_size - 1):
            obs, reward, terminated, truncated, info = self.env.step(NO_OP)
            total_reward += reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

        # Then send the real delayed action
        self.action_buffer.append(action)
        delayed_action = self.action_buffer.popleft()
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        total_reward += reward

        return obs, total_reward, terminated, truncated, info

