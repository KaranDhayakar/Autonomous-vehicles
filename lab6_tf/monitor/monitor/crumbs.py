#!/usr/bin/env python
''' Exercise 2

    Copyright: Daniel Morris, 2022
'''
import numpy as np
import sys
import math
import time

import rclpy
from rclpy.node import Node

from turtlesim.msg import Pose
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class Crumbs(Node):
    def __init__(self):
        super().__init__('crumbs')
        
        super().__init__('crumbs')
       
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listner = TransformListener(self.tf_buffer,self)
        self.timer = self.create_timer(1.0, self.call_back)
        self.list_frame = -1

    def call_back(self):
 
        if self.list_frame == -1:
                robo_frame = "odom"
        else:
                robo_frame = ''.join(["crumb_", str(self.list_frame)])

        from_frame = "base_footprint"

        try:
                t = self.tf_buffer.lookup_transform(robo_frame,from_frame,rclpy.time.Time())    
        except:
                self.get_logger().info(f'error in transformin {from_frame} frame: {ex}')
                return

        
        displace = math.sqrt((t.transform.translation.x)**2 + (t.transform.translation.y)**2)
        if (displace > 0.5) or (self.list_frame == -1):
                self.list_frame += 1
                t.child_frame_id = ''.join(["crumb_", str(self.list_frame)])
                self.tf_static_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = Crumbs()
    rclpy.spin(node)
    rclpy.shutdown()

    
