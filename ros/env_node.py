import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import numpy as np
import gymnasium as gym
import pygame
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
import ale_py
gym.register_envs(ale_py)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2


ENV_NAME = "PongNoFrameskip-v4"
SCALE = 4

FPS = 60

UP = 0.038
DOWN = 0.022
NO_OP = 0.03

SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

class PongEnvNode(Node):

    def __init__(self):
        super().__init__('pong_env_node')
        
        self.env = VecTransposeImage(
            make_atari_env(
                env_id=ENV_NAME, 
                seed=5, 
                env_kwargs={"render_mode": "rgb_array"},
                wrapper_kwargs={"frame_skip": 0},
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

        fps = FPS 
        self.timer = self.create_timer(1.0/fps, self.timer_callback)
        self.obs = None
        self.image_publisher = self.create_publisher(Image, 'pong_frame', qos_profile=SENSOR_QOS)
        # changed to SENSOR_QOS to drop stale joystick states too
        self.joystick_state_pub = self.create_publisher(Float32, "joystick_state", qos_profile=SENSOR_QOS)
        self.bridge = CvBridge()

        self.prev_action = 0
        self.action_cnt = 0
        self.action_limt = 1000

        icon_size = (30*SCALE, 24*SCALE)
        self.icons_map = {
            0: pygame.transform.scale(
                pygame.image.load("./ICONS/NO_OP.png").convert_alpha(), icon_size
            ),
            2: pygame.transform.scale(
                pygame.image.load("./ICONS/UP.png").convert_alpha(), icon_size
            ),
            3: pygame.transform.scale(
                pygame.image.load("./ICONS/DOWN.png").convert_alpha(), icon_size
            ),
        }

        # removed extra sleep + second publish timer (caused drift)

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
            self.get_logger().info(f"UP was pressed")
        elif keys[pygame.K_DOWN]:
            action = 3
            self.get_logger().info(f"DOWN was pressed")
        
        # Joystick (buttons override keyboard)
        if self.joystick:
            hat_x, hat_y = self.joystick.get_hat(0)  # read first hat
            if hat_y == 1:   # UP
                action = 2
                self._publish_joystick_state(UP)
            elif hat_y == -1:  # DOWN
                action = 3
                self._publish_joystick_state(DOWN)
            else:
                self._publish_joystick_state(NO_OP)

        if (self.prev_action == action):
            self.action_cnt += 1
            if (self.action_cnt >= self.action_limt):
                action = 0
        else:
            self.action_cnt = 0
            self.prev_action = action

        return np.array([action])

    def _publish_joystick_state(self, state:String):
        msg = Float32()
        msg.data = state
        self.joystick_state_pub.publish(msg)

    def render(self):
        frame = self.env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (self.w * SCALE, self.h * SCALE))
        self._embedd_icon(self.action, surf)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def publish_frame(self):
        if self.obs is None: return

        # Remove batch and channel dimensions: (84,84)
        img = self.obs[0, 0, :, :]
        
        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Publish
        self.image_publisher.publish(msg)

    def timer_callback(self):
        self.action = self.read_action()
        
        self.obs, rewards, dones, infos = self.env.step(self.action)

        # publish the frame immediately after the step
        self.publish_frame()
        
        self.env.render()
        if dones.any():
            self.get_logger().info("Episode done, resetting...")
            self.env.reset()

        self.render()
    
    def _embedd_icon(self,action,surface):

        icon = self.icons_map[int(action)]
        pos = (130*SCALE,0)
        surface.blit(icon, pos)




def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = PongEnvNode()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
