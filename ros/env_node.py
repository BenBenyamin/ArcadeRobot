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
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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
        
        ## INIT JOYSTICK
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick detected: {self.joystick.get_name()}")

        self.env.reset()

        frame = self.env.render()  # get initial frame
        height, width, _ = frame.shape

        self.h , self.w = height , width

        self.screen = pygame.display.set_mode((width * SCALE, height * SCALE))
        pygame.display.set_caption("Pong (Pygame Render)")

        fps = 30 
        self.timer = self.create_timer(1.0/fps, self.timer_callback)
        self.image_publisher = self.create_publisher(Image, 'pong_frame',qos_profile=10)
        self.bridge = CvBridge()
    
    def read_action(self):
        """
        Read Up/Down from keyboard and joystick.
        Returns: numpy array with Pong action.
        """
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        action = 0  # default NOOP

        # Keyboard
        if keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_DOWN]:
            action = 3

        # Joystick (buttons override keyboard)
        if self.joystick:
            hat_x, hat_y = self.joystick.get_hat(0)  # read first hat
            if hat_y == 1:   # UP
                action = 2
            elif hat_y == -1:  # DOWN
                action = 3

        return np.array([action])

    def render(self):

        frame = self.env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (self.w * SCALE, self.h * SCALE))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def publish_frame(self,obs):
        # Remove batch and channel dimensions: (84,84)
        img = obs[0, 0, :, :]
        
        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
        
        # Publish
        self.image_publisher.publish(msg)

    def timer_callback(self):
        action = self.read_action()
        
        obs, rewards, dones, infos = self.env.step(action)

        self.publish_frame(obs)
        
        self.env.render()
        if dones.any():
            self.get_logger().info("Episode done, resetting...")
            self.env.reset()

        self.render()


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = PongEnvNode()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()