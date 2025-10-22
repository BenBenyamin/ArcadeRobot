import stretch_body.robot
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class StrechControlNode(Node):

    def __init__(self):
        super().__init__('strech_node')

        self.robot = stretch_body.robot.Robot()
        self.robot.startup()

        self.target_pos = 0.03

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.create_subscription(msg_type=Float32,
                                 topic="move_arm",
                                 callback=self.move_arm_callback,
                                 qos_profile=qos,
                                 )
        self.move_arm_echo  = self.create_publisher(Float32, "move_arm_echo",10)

        ## Move in max velocity

        self.vel = self.robot.arm.params['motion']['max']['vel_m']
        self.acc = self.robot.arm.params['motion']['max']['accel_m']
        
        self.get_logger().info(f"\nArm velocity: {self.vel} m/s\nArm acceleration: {self.acc} m/s²")

        self.move_arm(0.03)
    
    def move_arm(self,x_m):

        self.target_pos = x_m

        self.robot.arm.move_to(x_m,v_m=self.vel, a_m=self.acc)
        self.robot.push_command()


    def move_arm_callback(self,msg):
        
        self.get_logger().info(f"\nTarget pos = {self.target_pos}\nCurr pos = {self.robot.arm.status['pos']}")

        self.move_arm_echo.publish(msg)

        # if abs(self.robot.arm.status['pos'] - self.target_pos) > 0.001: return

        self.move_arm(msg.data)

def main(args=None):

    rclpy.init(args=args)

    node = StrechControlNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
