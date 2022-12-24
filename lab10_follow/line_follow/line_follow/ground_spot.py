import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped, Point
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import cv2 as cv
import numpy as np
from threading import Lock
from scipy.spatial.transform import Rotation as R

class GroundSpot(Node):
    def __init__(self):
        super().__init__('ground_spot')

        # Create listener for transforms:
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)        

        # Create publisher for ground spot:
        self.publisher = self.create_publisher(PointStamped, 'ground_point', 1)    

        # Store camera parameters from camera_info topic:
        self.cam_info_lock = Lock()
        self.cam_info = None
        self.cam_info_is_init = False
        # Future ROS releases may have a wait_for_message which we can use instead of 
        # a subscription that only needs to be used once:
        self.sub_cam_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.copy_cam_info, 1)
        self.sub_cam_info        

        # Convert camera parameters into form usable by OpenCV:
        self.line_point_is_init = False
        self.D = None
        self.K = None

        # Store camera pose
        self.cam_pose_init = False
        self.cam_rot = None
        self.cam_tran = None

        self.sub_line_point = self.create_subscription(PointStamped, '/line_point', self.calc_ground_spot, 1)
        self.sub_line_point       
                 

    def copy_cam_info(self, msg):
        ''' Callback to read in camera intrinsics from camera_info 
        '''
        with self.cam_info_lock:
            # Only need to copy cam_info once, since fixed-focal length camera
            if not self.cam_info_is_init:
                self.cam_info = msg
                self.cam_info_is_init = True
                self.get_logger().info('Initialized camera intrinsics')   

    def calc_ground_spot(self, msg):
        ''' Callback to convert image line_point into ground_point
        '''
        if not self.line_point_is_init:
            # We need to initialize camera intrinsics from cam_info just once:
            with self.cam_info_lock:
                if self.cam_info_is_init:
                    self.D = np.array(self.cam_info.d)
                    self.K = np.array(self.cam_info.k).reshape( (3,3) )
                    self.line_point_is_init = True
        if not self.line_point_is_init:
            self.get_logger().info('Waiting for camera_info')    
            return

        # Initialize camera extrinsics:
        if not self.cam_pose_init:
            try:
                t = self.tf_buffer.lookup_transform(
                        "base_footprint",
                        "camera_rgb_optical_frame",
                        msg.header.stamp)
            except TransformException as ex:
                self.get_logger().info(
                        f'Could not transform camera_rgb_optical_frame to base_footprint: {ex}')
                return
            self.cam_rot = R.from_quat([t.transform.rotation.x, t.transform.rotation.y, 
                                        t.transform.rotation.z, t.transform.rotation.w])
            self.cam_tran = t.transform.translation
            self.cam_pose_init = True
            self.get_logger().info('Initialized camera extrinsics')   

        # We will publish a PointStamped in base_footprint coordinates:
        out_header = msg.header
        out_header.frame_id = "base_footprint"

        # test if input point is zeros -- if so then no visible green line
        # and so publish an all-zero point in base_footprint
        if msg.point.x==0 and msg.point.y==0:
            self.publisher.publish( PointStamped(header=out_header) )
            return

        # The following is an implementation from Week 6 Computer Vision and Calibration
        # See slide 53: What can we do with Extrinsics + Intrinsics
        # It finds the ground location corresponding to the line_point in the image

        # Get pixel image coordinates:
        pix = np.array([msg.point.x, msg.point.y]).reshape( (-1,2,1) )  #[N x 2 x 1]
        # Undistort these to unit focal plane:
        unit_focal = cv.undistortPoints(pix, self.K, self.D, R=None, P=None)
        unit_focal = unit_focal.reshape( (2,) )
        # Rotate vector to base_footprint coordinates:
        nvec = self.cam_rot.apply( np.array([unit_focal[0], unit_focal[1], 1.]) )
        # Find intersection of this vector that starts at camera origin with the ground plane
        scale = -self.cam_tran.z / nvec[2]   # How far to scale z component to hit the ground
        ground_point = Point(x=nvec[0] * scale + self.cam_tran.x,
                             y=nvec[1] * scale + self.cam_tran.y,
                             z=nvec[2] * scale + self.cam_tran.z ) 

        # Publish it as a PointStamped in base_footprint:
        ground_spot = PointStamped(header=out_header,point=ground_point)
        self.publisher.publish( ground_spot)


def main(args=None):

    rclpy.init(args=args)
    node = GroundSpot()
    rclpy.spin(node) 

