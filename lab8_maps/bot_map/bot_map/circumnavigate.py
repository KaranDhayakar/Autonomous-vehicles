import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, Pose, TransformStamped
from visualization_msgs.msg import MarkerArray 
from tf2_ros.buffer import Buffer
import tf2_ros as tf2_ros
from .tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from .navigate_waypoint import BasicNavigator
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np


class Circumnavigate(Node):

    def __init__(self, radius=1):
        super().__init__('node1')
        self.radius = radius
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)        
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.navigate_obj = BasicNavigator()
        self.subscription = self.create_subscription(MarkerArray, 'obstacles', self.listener_callback, 1)
        self.subscription  # prevent unused variable warning


    def pose_transform(self, pose: PoseStamped, target_frame: str) -> PoseStamped:
        ''' pose: will be transformed to target_frame '''
        duration=rclpy.duration.Duration(seconds=0.5)
        source_frame = pose.header.frame_id   # This is the frame pose is in
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, pose.header.stamp, duration )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rclpy.logging.get_logger("pose_transform").info(f'Cannot find transformation from {source_frame} to {target_frame}')
            raise Exception(f'Cannot find transformation from {source_frame} to {target_frame}') from e
        
        pose_transformed = PoseStamped(header=Header(stamp=pose.header.stamp,frame_id=target_frame), 
                                       pose=do_transform_pose(pose.pose, transform))
        return pose_transformed
    
    def listener_callback(self, msg):
        
        # msg.markers define the centroids of the obstacles
        # Find the closest obstacle to the robot
        # Then define target location behind the closest marker
        # as a position in base_footprint coordinates
        new_msg = []
        new_msg = msg.markers
        dist = []
        arr = msg.markers
        x = []
        y = []
        for centroid in arr:
            dist.append(centroid.pose.position.x**2+centroid.pose.position.y**2)
            x.append(centroid.pose.position.x)
            y.append(centroid.pose.position.y)

        min_dist = min(dist)
        index = dist.index(min_dist)

        x_hat = x[index]/min_dist
        y_hat = y[index]/min_dist

        target_x =  (min_dist+2)*x_hat# replace 0 with the actually position in 
        target_y =  (min_dist+2)*y_hat#lidar_centroids.ycen # replace 0 with actual position
       

        # set bot initial pose to be base_footprint
        stamp = rclpy.time.Time().to_msg()   # Current time for stamp
        initial_pose_base = PoseStamped(header=Header(stamp=stamp,frame_id='base_footprint'))
        # set target pose to be target location relative to base_footprint        
        goal_pose_base = PoseStamped(header=Header(stamp=stamp,frame_id='base_footprint'))
        goal_pose_base.pose = Pose(position=Point(x=target_x,y=target_y))
         
        # Now transform initial pose and target pose to map:
        initial_pose = self.pose_transform(initial_pose_base, target_frame='map')
        goal_pose = self.pose_transform(goal_pose_base, target_frame='map')
        
        # Create a frame at target location:
        target_frame = TransformStamped(header=Header(stamp=stamp,frame_id='map'))
        target_frame.child_frame_id = "target"
        target_frame.transform.translation.x = goal_pose.pose.position.x        
        target_frame.transform.translation.y = goal_pose.pose.position.y        
        target_frame.transform.translation.z = goal_pose.pose.position.z        
        target_frame.transform.rotation = goal_pose.pose.orientation
        
        self.tf_static_broadcaster.sendTransform(target_frame)

        # Ready to navigate:
        self.navigate_obj.setInitialPose(initial_pose.pose)

        # Wait for navigation to fully activate
        self.navigate_obj.waitUntilNav2Active()

        self.navigate_obj.goToPose(goal_pose)
        
        i = 0
        print("Initial pose: ",initial_pose)
        print("Goal pose: ",goal_pose)
        
        while not self.navigate_obj.isNavComplete():
            i = i + 1
            feedback = self.navigate_obj.getFeedback()
            if feedback and i % 10 == 0:
                print("Feedback ",i,"  ",feedback)
        print("--------Done Navigation----------")


def main(args=None):
    rclpy.init(args=args)
    print("In main")
    node = Circumnavigate()

    rclpy.spin(node)
