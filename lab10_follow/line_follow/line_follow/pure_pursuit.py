import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import cv2 as cv
import numpy as np
from threading import Lock
from scipy.spatial.transform import Rotation as R

import argparse
from geometry_msgs.msg import Twist, Vector3

class PurePursuit(Node):
    def __init__(self,speed):
        super().__init__('pure_pursuit')

        # Create listener for transforms:
        #self.tf_buffer = Buffer()
       # self.bot_pub = TransformListener(self.tf_buffer, self)        

        # Create publisher for ground spot:
        self.bot_pub = self.create_publisher(Twist, 'cmd_vel', 1)    
        self.speed = speed
        self.grd_pt_sub =  self.create_subscription(PointStamped, 'ground_point', self.ground_point_callback, 1)
        self.grd_pt_sub

       
    def ground_point_callback(self, msg):
        x_pt = msg.point.x
        y_pt = msg.point.y
        dist = (x_pt**2)+(y_pt**2)
        if(dist == 0 ):
            bot_move = Twist( linear = Vector3(x=0.0),angular= Vector3(z=0.5))
        else: 
            k = (2*y_pt)/dist
            bot_move = Twist( linear = Vector3(x=self.speed),angular= Vector3(z=k*self.speed))
        self.bot_pub.publish(bot_move)
       
        
def main(args=None):
    parser = argparse.ArgumentParser(description="command yaw")
    parser.add_argument('--speed',type=float,default=0.15,help='speed')
    args_temp = parser.parse_args()
    rclpy.init(args=args)
    node = PurePursuit(args_temp.speed)
    rclpy.spin(node) 
