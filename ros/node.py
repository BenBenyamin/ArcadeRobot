import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from hello_misc import HelloNode
from sensor_msgs.msg import Image
import numpy as np

class ArmExtendRetract(HelloNode):
    def __init__(self):
        super().__init__()
        HelloNode.main(self, 'arm_extend_retract', 'arm_extend_retract', wait_for_first_pointcloud=False)

        
        self.image_subscriber = self.create_subscription(Image,"pong_frame",self.image_callback,10)


    def image_callback(self,msg:Image):
        
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        # self.get_logger().info(f"GOT IMAGE! shape={img.shape}", throttle_duration_sec=0.5)

    def move_arm(self, lift, extension, yaw, duration_sec=4.0):
        """Send a single FollowJointTrajectory goal and wait for completion."""
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
            joint_state.position[yaw_idx],
        ]
        # Move to target
        end_point.positions = [extension, yaw]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['wrist_extension', 'joint_wrist_yaw']
        goal.trajectory.points = [start_point, end_point]

        self.get_logger().info(f"Moving to lift={lift:.2f}, extension={extension:.2f}, yaw={yaw:.2f}")

        # Send the goal
        send_goal_future = self.trajectory_client.send_goal_async(goal)

        # Wait for goal acceptance
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal was rejected by the server.")
            return

        self.get_logger().info("Goal accepted. Executing trajectory...")

        # Wait for the motion to finish
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()

        self.get_logger().info(f"Trajectory finished with status: {result.status}")

    def extend_and_retract(self):
        """Extend the arm and then retract."""
        while not self.joint_state.position:
            self.get_logger().info("Waiting for joint states...", throttle_duration_sec = 1.0)

        # while True:
        #     # Extend
        #     self.move_arm(lift=0.2, extension=0.2, yaw=0.0, duration_sec=1.0)

        #     # Retract
        #     self.move_arm(lift=0.2, extension=0.0, yaw=0.0, duration_sec=1.0)


def main(args=None):

    minimal_publisher = ArmExtendRetract()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
