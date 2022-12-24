import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point


from .pid import pid_controller
import argparse
from geometry_msgs.msg import Twist, Vector3

class Pid(Node):
    def __init__(self,kp,ki,kd):
        super().__init__('pid')

        # Create listener for transforms:
        #self.tf_buffer = Buffer()
       # self.bot_pub = TransformListener(self.tf_buffer, self)        

        # Create publisher for ground spot:
        self.bot_pub = self.create_publisher(Twist, 'cmd_vel', 1)    
        self.target_screen_x = 960
        self.pid = pid_controller(kp,ki,kd)

        self.speed = 0.5
        self.grd_pt_sub =  self.create_subscription(PointStamped, 'line_point', self.callback, 1)
        self.grd_pt_sub

       
    def callback(self, msg):
        x_pt = msg.point.x
        time = msg.header.stamp.sec + msg.header.stamp.nanosec*(1e-9)
        yaw_rate, yaw_err, int_err, diff_err = self.pid.update_control(self.target_screen_x, x_pt, time)

        bot_move = Twist( linear = Vector3(x=self.speed),angular= Vector3(z=yaw_rate/1696))
        self.bot_pub.publish(bot_move)
        
        
def main(args=None):
    parser = argparse.ArgumentParser(description="command yaw")
    parser.add_argument('kp',type=float,default=1,help='kp')
    parser.add_argument('ki',type=float,default=1,help='ki')
    parser.add_argument('kd',type=float,default=1,help='kd')
    
    args_temp = parser.parse_args()
    rclpy.init(args=args)
    node = Pid(args_temp.kp,args_temp.ki,args_temp.kd)
    rclpy.spin(node) 