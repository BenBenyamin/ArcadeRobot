import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
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

class StretchControlNode(HelloNode):
    def __init__(self):
        super().__init__()
        HelloNode.main(self, 'arm_extend_retract', 'arm_extend_retract', wait_for_first_pointcloud=False)

        while not self.joint_state.position:
            self.get_logger().info("Waiting for joint states...", throttle_duration_sec = 1.0)
            
        self.image_subscriber = self.create_subscription(Image,"pong_frame",self.image_callback,10)

        env = make_atari_env(env_id=ENV_NAME, seed=5)
        env = VecTransposeImage(env)
        self.model = RecurrentPPO.load(MODEL_FILENAME, env=env, device="cuda")
        self.lstm_states = None
        self.obs = None

        self.waiting = False
        self.state = STATE.NO_OP
        self.move_arm(extension=0.03,duration_sec=0)

        self.timer = self.create_timer(1.0/50, self.timer_callback)



    def _action_response_callback(self,future):

        self.waiting = False

    def image_callback(self,msg:Image):
        
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        self.obs = img[np.newaxis, np.newaxis, :, :]

        # self.get_logger().info(f"Sum = {np.sum(self.obs)}")
        
    def get_action(self):

        if self.obs is None:
            return

        action, lstm_states = self.model.predict(self.obs, state=self.lstm_states, deterministic=True)
        self.lstm_states = lstm_states

        # self.get_logger().info(f"Desired action = {action}", throttle_duration_sec=0.5)

        return action[0]

    def timer_callback(self):

        if self.waiting:
            self.get_logger().info(f"Waiting...", throttle_duration_sec=0.5)
            return

        action = self.get_action()

        if action == 4 or action == 2 and self.state != STATE.UP:
            self.joystick_up()
        
        if action == 3 or action == 5 and self.state != STATE.DOWN:
            self.joysick_down()
    
    def joystick_up(self):

        self.state = STATE.UP
        self.move_arm(extension=0.04,duration_sec=0)
        self.get_logger().info(f"UP")
    
    def joysick_down(self):
        
        self.state = STATE.DOWN
        self.move_arm(extension=0.02,duration_sec=0)
        self.get_logger().info(f"DOWN")

    def move_arm(self, extension,duration_sec=4.0):
        """Send a single FollowJointTrajectory goal and wait for completion."""
        
        self.waiting = True
        joint_state = self.joint_state

        arm_idx = joint_state.name.index('wrist_extension')
        yaw_idx = joint_state.name.index('joint_wrist_yaw')

        start_point = JointTrajectoryPoint()
        end_point = JointTrajectoryPoint()

        start_point.time_from_start = Duration(seconds=0.0).to_msg()
        end_point.time_from_start = Duration(seconds=duration_sec).to_msg()

        # Start from current position
        start_point.positions = [
            joint_state.position[arm_idx],
        ]
        # Move to target
        end_point.positions = [extension]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['wrist_extension']
        # goal.trajectory.points = [start_point, end_point]
        goal.trajectory.points = [end_point]

        # self.get_logger().info(f"Moving to lift={lift:.2f}, extension={extension:.2f}, yaw={yaw:.2f}")

        # Send the goal
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        # Wait for goal acceptance
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().warn(f"Trajectory goal was rejected by the server , extention = {extension}.\n error code = {send_goal_future.result()}")
            self.waiting = False
            return

        # self.get_logger().info("Goal accepted. Executing trajectory...")

        # Wait for the motion to finish
        # get_result_future = goal_handle.get_result_async()
        # rclpy.spin_until_future_complete(self, get_result_future)
        # result = get_result_future.result()
        
        send_goal_future.add_done_callback(self._action_response_callback)
        # self.get_logger().info(f"Trajectory finished with status: {result.status}")


def main(args=None):

    minimal_publisher = StretchControlNode()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
