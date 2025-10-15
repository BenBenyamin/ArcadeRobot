import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from pathlib import Path
import time
import numpy as np
import gymnasium as gym
import pygame
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
import ale_py
gym.register_envs(ale_py)


ENV_NAME = "PongNoFrameskip-v4"
MODEL_FILENAME = "RPPO_Delay=30_BEST.zip"
SCALE = 3


class PongEnvNode(Node):

    def __init__(self):
        super().__init__('pong_env_node')
        
        self.env = VecTransposeImage(
            make_atari_env(
                env_id=ENV_NAME, 
                seed=5, 
                env_kwargs={"render_mode": "rgb_array"}
                )
            )
        
        pygame.init()
        
        self.env.reset()

        frame = self.env.render()  # get initial frame
        height, width, _ = frame.shape

        self.h , self.w = height , width

        self.screen = pygame.display.set_mode((width * SCALE, height * SCALE))
        pygame.display.set_caption("Pong (Pygame Render)")

        fps = 30 
        self.timer = self.create_timer(1.0/fps, self.timer_callback)
        self.i = 0
        self.publisher_ = self.create_publisher(String, 'topic', 10)
    
    def render(self):

        frame = self.env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (self.w * SCALE, self.h * SCALE))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def timer_callback(self):
        self.env.step(np.array([0]))
        self.render()


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = PongEnvNode()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
