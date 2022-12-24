import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry

import cv2
import time
import os
import argparse
import numpy as np

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from .logist_reg import LogisticReg

class Detect(Node):
	def __init__(self):     
		super().__init__('detect')        

		self.declare_parameter('image','')

		self.folder = os.path.join(os.getenv('HOME'),'av/ros_ws/src/dhayakar_av/lab9_line/line_detect/images')

		self.sub = self.create_subscription(CompressedImage,'/camera/image_raw/compressed', self.callback, 10) 
		self.sub
		self.log_r = LogisticReg()
		self.log_r.set_model(np.array([[ 0.0597354, 0.12373394, -0.15512554]]),np.array([-10.03989255]))
		self.pub = self.create_publisher(PointStamped, 'line_point', 1)
		#self.get_logger().info(f'Subscribed to: /camera/image_raw/compressed')
		self.dot_publisher = PointStamped()
		


	def callback(self, img_data):
		bridge = CvBridge()
		try:
			robo_img = bridge.compressed_imgmsg_to_cv2(img_data)
		except CvBridgeError as e:
			print(e)
		
		img =  robo_img.copy() 
		robo_img = robo_img[1073:,:,:]
		prob = self.log_r.apply(robo_img) 
		prob_val = self.log_r.prob_target( prob )
		#print("log r done")
		bin_mask = (prob_val>0.98).astype(np.uint8)

		N, label_img, bbn, centroids = cv2.connectedComponentsWithStats( bin_mask )
		Npix = bbn[:,4]

		cen_clust = np.argsort( bbn[:,4] )
		cen_clust = cen_clust[::-1]

		all_cen = []

		for c in cen_clust:
			if ((bin_mask[label_img == c].astype(float).mean() > 0.98) and (Npix[c] > 20)):
				all_cen.append( centroids[c,:] )

		i = 0
		for c in all_cen:
			c[1] += 1073
			c[0] = c[0]
			c[1] = c[1]
			all_cen[i] = c
			i += 1
		#print("cent done")
		for c in all_cen:
			cv2.circle( img, tuple([int(c[0]),int(c[1])]), radius=15, color=(0,0,255), thickness=-1 )

		cv2.namedWindow('Image detection from camera', cv2.WINDOW_KEEPRATIO)
		cv2.imshow('Image detection from camera',img)
		cv2.resizeWindow('Image detection from camera', 1000, 600)
		#print("img done")
		if not all_cen:
			self.dot_publisher.point.x = 0.0
			self.dot_publisher.point.y = 0.0
		else:
			self.dot_publisher.point.x , self.dot_publisher.point.y = all_cen[0]

		self.dot_publisher.point.x = float(self.dot_publisher.point.x)
		self.dot_publisher.point.y = float(self.dot_publisher.point.y)
		
		val = self.get_parameter('image').get_parameter_value().string_value
		if val == 'save_line':
			img_path = os.path.join(self.folder,"line.png")
			cv2.imwrite(img_path, robo_img)
			
		self.pub.publish( self.dot_publisher )

		cv2.waitKey(1)

		return




def main(args=None):
	rclpy.init(args=args)

	node = Detect()
	try:
		rclpy.spin(node)
	except SystemExit:
		node.destroy_node()
		rclpy.shutdown()
	except KeyboardInterrupt:
		pass

