#!/usr/bin/env python
''' Exercise 2

    Copyright: Daniel Morris, 2022
'''
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

class Obstacles(Node):
    def __init__(self, ground_height, sub_topic='/points_raw', pub_topic='/points_obs'):
        super().__init__('obstacles')
        # Initialize the node
        self.ground_height = ground_height
        self.get_logger().info(f"Node subscribes to: {sub_topic} and publishes to: {pub_topic}")
        # Create a publisher for topic: pub_topic
        self.publisher_ = self.create_publisher(PointCloud2, pub_topic, 1)       
        # Define a subscriber for sub_topic that is type PointCloud2 and calls self.lidar_callback
        self.subscription = self.create_subscription(PointCloud2, sub_topic, self.lidar_callback, 1)
        self.subscription  

    def lidar_callback(self, msg):

        pts=point_cloud2.read_points(msg, field_names=['x','y','z','intensity'])
        # Convert to 2D numpy array with single data type:
        pts = structured_to_unstructured(pts)

        # Filter point cloud:
        # <...> Here is where you need to process the input points
        obs_range = np.logical_and((pts[:,2] > 0.3+self.ground_height),(pts[:,2] < 2+self.ground_height), ((pts[:,0]**2 + pts[:,1]**2)**0.5<20))
        # <...> You will set pts_new to the final points (as a numpy array) which is published below
        
        new_pts = pts[obs_range,:]
  
        # <...> Don't forget to use self.ground_height
        
        # Define fields for output PointCloud2
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)]

        new_msg = point_cloud2.create_cloud(msg.header, fields, new_pts)  # Create message from pts_new

        self.publisher_.publish( new_msg )

        
def main(args=None):
    rclpy.init(args=args)
    obs = Obstacles(ground_height=-1.6)  # Also not required to pass in ground height
    rclpy.spin(obs)
    
