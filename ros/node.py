import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float32

from hello_misc import HelloNode
from sensor_msgs.msg import Image
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
import ale_py
from enum import Enum

MODEL_FILENAME = "RPPO_Delay=50_BEST.zip"
ENV_NAME = "PongNoFrameskip-v4"

class STATE(Enum):

    NO_OP = 0,
    UP = 1,
    DOWN = 2,

UP = 0.038
DOWN = 0.022
NO_OP = 0.03

class StretchControlNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('strech_node')
            
        self.image_subscriber = self.create_subscription(Image,"pong_frame",self.image_callback,10)

        self.command_publisher = self.create_publisher(Float32,"move_arm",10)

        env = make_atari_env(env_id=ENV_NAME, seed=5)
        env = VecTransposeImage(env)
        self.model = RecurrentPPO.load(MODEL_FILENAME, env=env, device="cuda")
        self.lstm_states = None
        self.obs = None

        self.state = STATE.NO_OP
        self.move_arm(NO_OP)

        self.timer = self.create_timer(0.2, self.timer_callback)


    def image_callback(self,msg:Image):
        
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.obs = img[np.newaxis, np.newaxis, :, :]
        
    def get_action(self):

        if self.obs is None:
            return

        action, lstm_states = self.model.predict(self.obs, state=self.lstm_states, deterministic=True)
        self.lstm_states = lstm_states

        return action[0]

    def timer_callback(self):

        action = self.get_action()

        if action == 4 or action == 2 and self.state != STATE.UP:
            self.joystick_up()
        
        if action == 3 or action == 5 and self.state != STATE.DOWN:
            self.joysick_down()
    
    def joystick_up(self):

        self.state = STATE.UP
        self.move_arm(UP)
        # self.get_logger().info(f"UP")
    
    def joysick_down(self):
        
        self.state = STATE.DOWN
        self.move_arm(DOWN)
        # self.get_logger().info(f"DOWN")

    def move_arm(self, x_m):

        msg = Float32()
        msg.data = x_m

        self.command_publisher.publish(msg)
        

def main(args=None):

    rclpy.init(args=args)
    minimal_publisher = StretchControlNode()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
