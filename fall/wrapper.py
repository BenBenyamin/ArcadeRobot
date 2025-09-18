import gymnasium as gym
from collections import deque

NO_OP = 0

class PongDelayWrapper(gym.Wrapper):

    def __init__(self, env, buffer_stack_size=1):
        super().__init__(env)

        self.buffer_stack_size = buffer_stack_size

        self.action_queue = deque(
            [NO_OP] * buffer_stack_size, 
            maxlen=buffer_stack_size
        )

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        self.action_queue.clear()
        self.action_queue.extend([NO_OP] * self.buffer_stack_size)

        return obs


    def step(self,action):

        delayed_action = self.action_queue.popleft()
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        self.action_queue.append(action)

        return obs, reward, terminated, truncated, info